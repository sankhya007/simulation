# agent.py

import random
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Protocol

from environment import EnvironmentGraph
from config import (
    AGENT_REPLAN_PROB,
    PERCEPTION_RADIUS,
    AGENT_TYPE_SPEEDS,
    DENSITY_THRESHOLD,
)


Node = Tuple[int, int]
EdgeKey = Tuple[Node, Node]


@dataclass
class DecisionState:
    """
    Snapshot of crowd-level information used by the agent to take a decision.
    This is what an external Agent AI / RL policy would see.
    """
    time_step: int
    node_occupancy: Dict[Node, int]
    group_targets: Dict[int, Node]          # group_id -> leader position
    edge_over_capacity: Dict[EdgeKey, bool] # (u, v) -> True if overcrowded

    # NEW: richer info for Agent AI integration
    agent_positions: Dict[int, Node]        # agent_id -> node
    agent_types: Dict[int, str]             # agent_id -> "leader"/"follower"/...
    global_density: float                   # avg agents per occupied node


class AgentPolicy(Protocol):
    """
    Pluggable policy interface. You can later create your own RL/AI policy:

        class MyRLPolicy:
            def choose_action(self, agent, state, env) -> Node:
                ...

        agent.set_policy(MyRLPolicy())
    """

    def choose_action(
        self,
        agent: "Agent",
        state: DecisionState,
        env: EnvironmentGraph,
    ) -> Node:
        ...


class Agent:
    """
    Single agent moving on the EnvironmentGraph.

    Attributes:
        - agent_type: "leader" | "follower" | "normal" | "panic"
        - group_id: int or None
        - speed: probability of moving in a given tick (0..1)
        - perception_radius: how far it "sees" group / congestion
        - policy: optional external decision policy (AgentPolicy)
    """

    _id_counter = 0

    def __init__(
        self,
        env: EnvironmentGraph,
        agent_type: str = "normal",
        group_id: Optional[int] = None,
        speed: Optional[float] = None,
        perception_radius: float = PERCEPTION_RADIUS,
        policy: Optional[AgentPolicy] = None,
    ):
        self.env = env
        self.id = Agent._id_counter
        Agent._id_counter += 1

        self.agent_type = agent_type
        self.group_id = group_id
        self.is_leader = (agent_type == "leader")

        # Speed from config, overridable
        if speed is None:
            self.speed = AGENT_TYPE_SPEEDS.get(agent_type, 0.8)
        else:
            self.speed = speed

        self.perception_radius = perception_radius
        self.policy: Optional[AgentPolicy] = policy

        # --- position / path ---
        self.current_node: Node = env.get_random_node()
        self.goal_node: Node = env.get_random_node()
        while self.goal_node == self.current_node:
            self.goal_node = env.get_random_node()

        self.path = self._compute_path(self.current_node, self.goal_node)
        self.path_index = 0

        self.finished = False
        self.steps_taken = 0
        self.collisions = 0

    # ---------- policy management ----------

    def set_policy(self, policy: AgentPolicy):
        """
        Attach an external decision policy (e.g., RL agent).
        """
        self.policy = policy

    # ---------- internal helpers ----------

    def _weight_attr_for_routing(self) -> str:
        """
        Decide which edge attribute to use when planning:
        - panic agents use pure geometric distance (ignore congestion)
        - others use congestion-aware 'weight'
        """
        return "distance" if self.agent_type == "panic" else "weight"

    def _compute_path(self, start: Node, goal: Node):
        weight_attr = self._weight_attr_for_routing()
        return self.env.shortest_path_weighted(start, goal, weight_attr=weight_attr)

    # ---------- goal / path management ----------

    def choose_new_goal(self):
        """
        Pick a new random goal and recompute path.
        """
        self.goal_node = self.env.get_random_node()
        while self.goal_node == self.current_node:
            self.goal_node = self.env.get_random_node()
        self.path = self._compute_path(self.current_node, self.goal_node)
        self.path_index = 0
        self.finished = False

    # ---------- main decision function ----------

    def desired_next_node(self, state: DecisionState) -> Node:
        """
        Decide which node this agent wants to move to this tick.

        If a custom policy is attached, defer to that.
        Otherwise, use built-in rule-based logic (_rule_based_desired_next_node).
        """
        if self.policy is not None:
            return self.policy.choose_action(self, state, self.env)
        return self._rule_based_desired_next_node(state)

    def _rule_based_desired_next_node(self, state: DecisionState) -> Node:
        """
        Original rule-based behaviour, now using DecisionState:
          - group following
          - speed-based waiting
          - local congestion avoidance
          - global congestion-aware routing (via edge weights)
          - avoids edges that are over capacity when possible
          - mild 'lane preference' towards the geometric goal direction
        """
        node_occupancy = state.node_occupancy
        group_targets = state.group_targets
        edge_over_capacity = state.edge_over_capacity

        # Reached end of path?
        if self.path_index >= len(self.path) - 1:
            self.finished = True

        # Occasionally pick a fresh random goal when finished
        if self.finished and random.random() < AGENT_REPLAN_PROB:
            self.choose_new_goal()

        if self.finished:
            return self.current_node

        # --- group-following behaviour for followers ---

        target_node = self.goal_node

        if self.agent_type == "follower" and self.group_id is not None:
            leader_node = group_targets.get(self.group_id)
            if leader_node is not None:
                x, y = self.env.get_pos(self.current_node)
                gx, gy = self.env.get_pos(leader_node)
                dist_to_leader = math.dist((x, y), (gx, gy))

                # If leader is within perception radius * 2, try to follow them
                if dist_to_leader <= self.perception_radius * 2.0:
                    target_node = leader_node
                    # occasionally re-plan directly toward leader
                    if random.random() < 0.2:
                        self.path = self._compute_path(self.current_node, target_node)
                        self.path_index = 0

        # --- speed: slow agents sometimes wait ---

        if random.random() > self.speed:
            return self.current_node

        # --- ensure path is current (weights may have changed due to congestion) ---

        if self.path_index >= len(self.path) - 1:
            self.path = self._compute_path(self.current_node, target_node)
            self.path_index = 0

        if len(self.path) <= 1:
            return self.current_node

        candidate_next = self.path[self.path_index + 1]

        # --- avoid edges that are over capacity (crowd-level interaction) ---

        edge_key = (self.current_node, candidate_next)
        if not edge_over_capacity.get(edge_key, False):
            # ok, edge not over capacity; we'll still check node congestion below
            pass
        else:
            # this edge is overcrowded -> try to replan, or sidestep
            # first, recompute path to target using updated weights
            self.path = self._compute_path(self.current_node, target_node)
            self.path_index = 0
            if len(self.path) > 1:
                candidate_next = self.path[1]

        # --- simple congestion avoidance at node level ---

        candidate_density = node_occupancy.get(candidate_next, 0)

        threshold = DENSITY_THRESHOLD
        if self.agent_type == "panic":
            threshold += 2  # panic agents tolerate more crowding

        if candidate_density <= threshold:
            return candidate_next

        # Try to sidestep to neighbours with lower density
        neighbors = self.env.get_neighbors(self.current_node, accessible_only=True)
        if not neighbors:
            return self.current_node

        remaining_path = set(self.path[self.path_index + 1 :])

        # for mild lane formation: prefer neighbours roughly toward our goal
        gx, gy = self.env.get_pos(target_node)
        cx, cy = self.env.get_pos(self.current_node)
        goal_dir = (gx - cx, gy - cy)

        def direction_alignment(n: Node) -> float:
            nx_, ny_ = self.env.get_pos(n)
            step_vec = (nx_ - cx, ny_ - cy)
            norm_goal = math.hypot(*goal_dir) or 1.0
            norm_step = math.hypot(*step_vec) or 1.0
            # cosine similarity: [-1, 1]
            return (goal_dir[0] * step_vec[0] + goal_dir[1] * step_vec[1]) / (
                norm_goal * norm_step
            )

        best_node = self.current_node
        # score = (density, path_penalty, -alignment) so we prefer lower density,
        # stay on our planned path, and align with goal direction
        best_score = (candidate_density, 1, 0.0)

        for n in neighbors:
            density_n = node_occupancy.get(n, 0)
            on_path = n in remaining_path
            path_penalty = 0 if on_path else 1
            align = direction_alignment(n)
            score = (density_n, path_penalty, -align)
            if score < best_score:
                best_score = score
                best_node = n

        if best_node == self.current_node and candidate_density > threshold:
            # everything around is crowded, we just wait
            return self.current_node

        return best_node

    # ---------- movement & position ----------

    def move_to(self, node: Node):
        if node != self.current_node:
            self.current_node = node
            self.steps_taken += 1

            # if node is the expected next node, just advance the path index
            if self.path_index + 1 < len(self.path) and self.path[self.path_index + 1] == node:
                self.path_index += 1
            else:
                # we deviated (due to avoidance); recompute path to current goal
                self.path = self._compute_path(self.current_node, self.goal_node)
                self.path_index = 0

    def get_position(self):
        return self.env.get_pos(self.current_node)
