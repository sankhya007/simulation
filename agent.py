# agent.py

import random
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Protocol, Any

from environment import EnvironmentGraph
from config import (
    AGENT_REPLAN_PROB,
    PERCEPTION_RADIUS,
    AGENT_TYPE_SPEEDS,
    DENSITY_THRESHOLD,
    GLOBAL_DENSITY_REPLAN_THRESHOLD,
    EVACUATION_MODE,
)


Node = Tuple[int, int]
EdgeKey = Tuple[Node, Node]


@dataclass
class DecisionState:
    """
    Snapshot of crowd-level information used by the agent to take a decision.
    """
    time_step: int
    node_occupancy: Dict[Node, int]
    group_targets: Dict[int, Node]  # group_id -> leader position
    edge_over_capacity: Dict[EdgeKey, bool]  # (u, v) -> True if overcrowded

    # richer info for Agent AI integration
    agent_positions: Dict[int, Node]  # agent_id -> node
    agent_types: Dict[int, str]  # agent_id -> "leader"/"follower"/...
    global_density: float  # avg agents per occupied node


class AgentPolicy(Protocol):
    def choose_action(
        self,
        agent: "Agent",
        state: DecisionState,
        env: EnvironmentGraph,
    ) -> Node: ...


class Agent:
    """
    Single agent moving on the EnvironmentGraph.
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
        navigation_strategy: str = "congestion",
    ):
        # ---- Ensure env reference exists immediately ----
        self.env = env

        # identity
        self.id = Agent._id_counter
        Agent._id_counter += 1

        # basic attributes
        self.agent_type = agent_type
        self.group_id = group_id
        self.is_leader = agent_type == "leader"

        # navigation strategy (AI behaviour)
        self.strategy = navigation_strategy  # "shortest", "congestion", "safe"

        # speed
        if speed is None:
            self.speed = AGENT_TYPE_SPEEDS.get(agent_type, 0.8)
        else:
            self.speed = speed

        self.perception_radius = perception_radius
        self.policy: Optional[AgentPolicy] = policy

        # --- metrics ---
        self.steps_taken: int = 0
        self.wait_steps: int = 0
        self.replans: int = 0
        self.goals_reached: int = 0
        self.exit_reached: bool = False
        self.exit_time_step: Optional[int] = None

        self.initial_start: Optional[Node] = None
        self.initial_goal: Optional[Node] = None
        self.initial_shortest_path_len: int = 0

        # ---------- position / path (choose start early) ----------
        self.current_node: Node = self.env.get_random_node()

        # initial goal & path
        if EVACUATION_MODE:
            exit_path = self.env.shortest_path_to_nearest_exit(self.current_node)
            if exit_path:
                self.goal_node = exit_path[-1]
                self.path = exit_path
            else:
                self.goal_node = self.current_node
                self.path = [self.current_node]
        else:
            self.goal_node: Node = self.env.get_random_node()
            while self.goal_node == self.current_node:
                self.goal_node = self.env.get_random_node()
            self.path = self._compute_path(self.current_node, self.goal_node)

        self.path_index = 0

        # store initial start/goal for path optimality analysis
        self.initial_start = self.current_node
        self.initial_goal = self.goal_node

        # ideal shortest path length ignoring congestion
        ideal_path = self.env.shortest_path_weighted(
            self.initial_start,
            self.initial_goal,
            weight_attr="distance",
        )
        self.initial_shortest_path_len = max(0, len(ideal_path) - 1)

        # initial planning is not counted as "replan"
        self.replans = 0

        self.finished = False
        self.collisions = 0

        # ---------- continuous-motion fields (usable by SF / RVO) ----------
        try:
            # env.get_pos should return world coords
            self.pos = tuple(self.env.get_pos(self.current_node))
        except Exception:
            # fallback to grid indices as floats
            self.pos = (float(self.current_node[0]), float(self.current_node[1]))
        self.vel = (0.0, 0.0)
        self.radius = 0.3

        # default motion model adapter (GraphMotionModel) if available
        try:
            from motion_models import GraphMotionModel

            self.motion_model = GraphMotionModel()
        except Exception:
            self.motion_model = None

    # ---------- policy management ----------
    def set_policy(self, policy: AgentPolicy):
        self.policy = policy

    # ---------- internal helpers ----------
    def _weight_attr_for_routing(self) -> str:
        if self.strategy == "shortest":
            return "distance"
        else:
            return "weight"

    def _compute_path(self, start: Node, goal: Node):
        self.replans += 1
        weight_attr = self._weight_attr_for_routing()
        return self.env.shortest_path_weighted(start, goal, weight_attr=weight_attr)

    def _ensure_path_valid(self):
        if self.path_index >= len(self.path) - 1:
            return
        remaining = self.path[self.path_index + 1 :]
        for n in remaining:
            if not self.env.is_accessible(n):
                self.path = self._compute_path(self.current_node, self.goal_node)
                self.path_index = 0
                return

    # ---------- goal / path management ----------
    def choose_new_goal(self):
        if EVACUATION_MODE:
            self.finished = True
            return

        self.goals_reached += 1

        self.goal_node = self.env.get_random_node()
        while self.goal_node == self.current_node:
            self.goal_node = self.env.get_random_node()
        self.path = self._compute_path(self.current_node, self.goal_node)
        self.path_index = 0
        self.finished = False

    # ---------- main decision function ----------
    def desired_next_node(self, state: DecisionState) -> Node:
        if self.policy is not None:
            return self.policy.choose_action(self, state, self.env)
        return self._rule_based_desired_next_node(state)

    def _rule_based_desired_next_node(self, state: DecisionState) -> Node:
        node_occupancy = state.node_occupancy
        group_targets = state.group_targets
        edge_over_capacity = state.edge_over_capacity
        global_density = state.global_density

        self._ensure_path_valid()

        if self.path_index >= len(self.path) - 1:
            self.finished = True

        if (not EVACUATION_MODE) and self.finished and random.random() < AGENT_REPLAN_PROB:
            self.choose_new_goal()

        if self.finished:
            return self.current_node

        target_node = self.goal_node

        if self.agent_type == "follower" and self.group_id is not None:
            leader_node = group_targets.get(self.group_id)
            if leader_node is not None:
                x, y = self.env.get_pos(self.current_node)
                gx, gy = self.env.get_pos(leader_node)
                dist_to_leader = math.dist((x, y), (gx, gy))

                if dist_to_leader <= self.perception_radius * 2.0:
                    target_node = leader_node
                    if random.random() < 0.2:
                        self.path = self._compute_path(self.current_node, target_node)
                        self.path_index = 0

        if global_density > GLOBAL_DENSITY_REPLAN_THRESHOLD and random.random() < 0.2:
            self.path = self._compute_path(self.current_node, target_node)
            self.path_index = 0

        if self.path_index >= len(self.path) - 1:
            self.path = self._compute_path(self.current_node, target_node)
            self.path_index = 0

        if len(self.path) <= 1:
            return self.current_node

        candidate_next = self.path[self.path_index + 1]

        edge_key = (self.current_node, candidate_next)
        if edge_over_capacity.get(edge_key, False):
            self.path = self._compute_path(self.current_node, target_node)
            self.path_index = 0
            if len(self.path) > 1:
                candidate_next = self.path[1]

        if candidate_next == self.current_node:
            effective_speed = self.speed
        else:
            slowdown = self.env.get_edge_slowdown(self.current_node, candidate_next)
            effective_speed = self.speed / slowdown

        effective_speed = max(0.05, min(effective_speed, 1.0))

        if random.random() > effective_speed:
            self.wait_steps += 1
            return self.current_node

        candidate_density = node_occupancy.get(candidate_next, 0)

        threshold = DENSITY_THRESHOLD
        if self.agent_type == "panic":
            threshold += 2

        if candidate_density <= threshold:
            return candidate_next

        neighbors = self.env.get_neighbors(self.current_node, accessible_only=True)
        if not neighbors:
            self.wait_steps += 1
            return self.current_node

        remaining_path = set(self.path[self.path_index + 1 :])

        gx, gy = self.env.get_pos(target_node)
        cx, cy = self.env.get_pos(self.current_node)
        goal_dir = (gx - cx, gy - cy)

        def direction_alignment(n: Node) -> float:
            nx_, ny_ = self.env.get_pos(n)
            step_vec = (nx_ - cx, ny_ - cy)
            norm_goal = math.hypot(*goal_dir) or 1.0
            norm_step = math.hypot(*step_vec) or 1.0
            return (goal_dir[0] * step_vec[0] + goal_dir[1] * step_vec[1]) / (norm_goal * norm_step)

        best_node = self.current_node
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
            self.wait_steps += 1
            return self.current_node

        return best_node

    # ---------- movement & position ----------
    def move_to(self, node: Node):
        """
        Move agent to a graph node. Update both the discrete node (current_node)
        and the continuous position (pos) so visualizers that read get_position()
        show the movement.
        """
        if node == self.current_node:
            return

        # update discrete bookkeeping
        self.current_node = node
        self.steps_taken += 1

        # update continuous world position so visualizations show agent moved
        try:
            # env.get_pos(node) should return world (x,y)
            self.pos = tuple(self.env.get_pos(node))
        except Exception:
            # fallback: set to node indices as floats
            self.pos = (float(node[0]), float(node[1]))

        # update path index if we were following a path, else replan
        if self.path_index + 1 < len(self.path) and self.path[self.path_index + 1] == node:
            self.path_index += 1
        else:
            # path diverged — recompute a path from new location
            self.path = self._compute_path(self.current_node, self.goal_node)
            self.path_index = 0


    def get_position(self):
        try:
            if hasattr(self, "pos") and self.pos is not None:
                return float(self.pos[0]), float(self.pos[1])
        except Exception:
            pass
        return self.env.get_pos(self.current_node)

    def integrate_continuous(self, vx: float, vy: float, dt: float = 1.0):
        px, py = self.get_position()
        nx, ny = px + vx * dt, py + vy * dt
        self.pos = (nx, ny)
        self.vel = (vx, vy)
        # mapping to node handled by simulation
