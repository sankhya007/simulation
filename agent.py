# agent.py

import random
import math
from typing import Tuple, Dict, Optional

from environment import EnvironmentGraph
from config import (
    AGENT_REPLAN_PROB,
    PERCEPTION_RADIUS,
    AGENT_TYPE_SPEEDS,
    DENSITY_THRESHOLD,
)


Node = Tuple[int, int]


class Agent:
    """
    Single agent moving on the EnvironmentGraph.

    Attributes:
        - agent_type: "leader" | "follower" | "normal" | "panic"
        - group_id: int or None
        - speed: probability of moving in a given tick (0..1)
        - perception_radius: how far it "sees" group / congestion
    """

    _id_counter = 0

    def __init__(
        self,
        env: EnvironmentGraph,
        agent_type: str = "normal",
        group_id: Optional[int] = None,
        speed: Optional[float] = None,
        perception_radius: float = PERCEPTION_RADIUS,
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

    def desired_next_node(
        self,
        node_occupancy: Dict[Node, int],
        group_targets: Dict[int, Node],
    ) -> Node:
        """
        Decide which node this agent wants to move to this tick.

        Uses:
            - group-following for followers
            - speed (slower agents may wait)
            - local congestion avoidance using node_occupancy
        """
        # Reached end of path?
        if self.path_index >= len(self.path) - 1:
            self.finished = True

        # Occasionally pick a brand new goal when finished
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

        # --- ensure path is current (can be affected by new congestion weights) ---

        if self.path_index >= len(self.path) - 1:
            self.path = self._compute_path(self.current_node, target_node)
            self.path_index = 0

        if len(self.path) <= 1:
            return self.current_node

        candidate_next = self.path[self.path_index + 1]

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

        best_node = self.current_node
        best_score = (candidate_density, 1)  # (density, path_penalty)

        for n in neighbors:
            density_n = node_occupancy.get(n, 0)
            on_path = n in remaining_path
            path_penalty = 0 if on_path else 1
            score = (density_n, path_penalty)
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
