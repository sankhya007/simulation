# simulation.py

import math
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np

from environment import EnvironmentGraph
from agent import Agent, DecisionState
from config import (
    COLLISION_DISTANCE,
    GROUP_SIZE,
    EDGE_BASE_CAPACITY,
    DYNAMIC_BLOCKS_ENABLED,
    BLOCK_NODE_EVERY_N_STEPS,
    BLOCK_NODE_PROB,
    DYNAMIC_EXITS_ENABLED,
    EXIT_TOGGLE_EVERY_N_STEPS,
)


Node = Tuple[int, int]
EdgeKey = Tuple[Node, Node]


class CrowdSimulation:
    """Manages the whole crowd, interactions, and metrics."""

    def __init__(self, env: EnvironmentGraph, num_agents: int):
        self.env = env
        self.agents: List[Agent] = []

        # Assign agents into groups (leaders + followers + normals)
        self._init_agents_with_groups(num_agents)

        self.time_step = 0

        # Metrics
        self.node_visit_counts = defaultdict(int)
        self.total_collisions = 0
        self.last_node_occupancy: Dict[Node, int] = defaultdict(int)

        for agent in self.agents:
            self.node_visit_counts[agent.current_node] += 1

    # ---------- initialization helpers ----------

    def _init_agents_with_groups(self, num_agents: int):
        full_groups = num_agents // GROUP_SIZE
        agent_index = 0
        group_id = 1

        # Full groups
        for _ in range(full_groups):
            # leader
            self.agents.append(
                Agent(
                    env=self.env,
                    agent_type="leader",
                    group_id=group_id,
                )
            )
            agent_index += 1

            # followers
            for _ in range(GROUP_SIZE - 1):
                self.agents.append(
                    Agent(
                        env=self.env,
                        agent_type="follower",
                        group_id=group_id,
                    )
                )
                agent_index += 1

            group_id += 1

        # Leftover agents as normals
        while agent_index < num_agents:
            self.agents.append(
                Agent(
                    env=self.env,
                    agent_type="normal",
                    group_id=None,
                )
            )
            agent_index += 1

    # ---------- dynamic events (Step 4) ----------

    def _apply_dynamic_events(self, node_occupancy: Dict[Node, int]):
        """
        Introduce dynamic obstacles and exit changes during the simulation.
        - Randomly blocks some unoccupied nodes.
        - Randomly opens/closes exits.
        """
        import random

        # Dynamic blocking / unblocking of nodes
        if DYNAMIC_BLOCKS_ENABLED and self.time_step % BLOCK_NODE_EVERY_N_STEPS == 0:
            candidates = [
                n
                for n, data in self.env.graph.nodes(data=True)
                if self.env.is_accessible(n)
                and node_occupancy.get(n, 0) == 0
            ]
            if candidates and random.random() < BLOCK_NODE_PROB:
                node = random.choice(candidates)
                self.env.block_node(node)

        # Dynamic exit opening/closing
        if DYNAMIC_EXITS_ENABLED and self.time_step % EXIT_TOGGLE_EVERY_N_STEPS == 0:
            exits = [n for n in self.env.graph.nodes() if self.env.is_exit(n)]
            border_candidates = [
                n
                for n, data in self.env.graph.nodes(data=True)
                if self.env.is_accessible(n)
                and not self.env.is_exit(n)
                and (
                    n[0] == 0
                    or n[1] == 0
                    or n[0] == (self.env.width - 1)
                    or n[1] == (self.env.height - 1)
                )
            ]

            import random

            if exits and random.random() < 0.5:
                node = random.choice(exits)
                self.env.unmark_exit(node)
            elif border_candidates:
                node = random.choice(border_candidates)
                self.env.mark_exit(node)

    # ---------- core step ----------

    def step(self):
        """
        Single sim tick:
        - compute node occupancy
        - apply dynamic events (blocked nodes / exit changes)
        - approximate edge congestion and update edge weights
        - compute group leader positions
        - build DecisionState and query each agent for desired move
        - resolve conflicts & move
        - measure collisions
        """
        self.time_step += 1

        # 1) node occupancy at current time
        node_occupancy: Dict[Node, int] = defaultdict(int)
        for agent in self.agents:
            node_occupancy[agent.current_node] += 1
        self.last_node_occupancy = node_occupancy

        # 2) apply dynamic environment events (blocked paths, exits)
        self._apply_dynamic_events(node_occupancy)

        # 3) edge occupancy approximation from node occupancy, then update weights
        edge_occupancy: Dict[EdgeKey, int] = {}
        edge_over_capacity: Dict[EdgeKey, bool] = {}

        for u, v in self.env.graph.edges():
            occ = node_occupancy.get(u, 0) + node_occupancy.get(v, 0)
            edge_occupancy[(u, v)] = occ

            capacity = self.env.graph[u][v].get("max_capacity", EDGE_BASE_CAPACITY)
            edge_over_capacity[(u, v)] = occ > capacity

        self.env.update_all_edge_weights_from_occupancy(edge_occupancy)

        # 4) group leader positions
        group_targets: Dict[int, Node] = {}
        for agent in self.agents:
            if agent.is_leader and agent.group_id is not None:
                group_targets[agent.group_id] = agent.current_node

        # 5) agent positions / types + global density
        agent_positions: Dict[int, Node] = {a.id: a.current_node for a in self.agents}
        agent_types: Dict[int, str] = {a.id: a.agent_type for a in self.agents}

        total_agents = len(self.agents)
        occupied_nodes = len(node_occupancy)
        global_density = total_agents / max(1, occupied_nodes)

        # 6) build state snapshot for this tick
        state = DecisionState(
            time_step=self.time_step,
            node_occupancy=node_occupancy,
            group_targets=group_targets,
            edge_over_capacity=edge_over_capacity,
            agent_positions=agent_positions,
            agent_types=agent_types,
            global_density=global_density,
        )

        # 7) each agent chooses desired next node
        desires: Dict[int, Node] = {}
        for agent in self.agents:
            desired = agent.desired_next_node(state)
            desires[agent.id] = desired

        # 8) conflict resolution
        node_to_agents = defaultdict(list)
        for agent_id, node in desires.items():
            node_to_agents[node].append(agent_id)

        allowed_moves: Dict[int, Node | None] = {}
        for node, agent_ids in node_to_agents.items():
            if len(agent_ids) == 1:
                allowed_moves[agent_ids[0]] = node
            else:
                import random

                winner = random.choice(agent_ids)
                allowed_moves[winner] = node
                for loser in agent_ids:
                    if loser != winner:
                        allowed_moves[loser] = None

        # 9) apply movements
        for agent in self.agents:
            target = allowed_moves.get(agent.id, None)
            if target is None:
                continue
            agent.move_to(target)
            self.node_visit_counts[agent.current_node] += 1

        # 10) collisions
        self._update_collisions()

    # ---------- metrics & helpers ----------

    def _update_collisions(self):
        positions = [agent.get_position() for agent in self.agents]
        n = len(positions)
        step_collisions = 0
        for i in range(n):
            x1, y1 = positions[i]
            for j in range(i + 1, n):
                x2, y2 = positions[j]
                dist = math.dist((x1, y1), (x2, y2))
                if dist < COLLISION_DISTANCE:
                    step_collisions += 1
                    self.agents[i].collisions += 1
                    self.agents[j].collisions += 1
        self.total_collisions += step_collisions

    def get_agent_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for agent in self.agents:
            x, y = agent.get_position()
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def get_density_matrix(self) -> np.ndarray:
        mat = np.zeros((self.env.height, self.env.width))
        for (x, y), count in self.node_visit_counts.items():
            mat[y, x] = count
        return mat

    def summary(self):
        print("\n=== Simulation Summary ===")
        print(f"Time steps: {self.time_step}")
        print(f"Total collisions: {self.total_collisions}")

        step_counts = [a.steps_taken for a in self.agents]
        print(f"Average steps/agent: {np.mean(step_counts):.2f}")
        print(f"Max steps: {np.max(step_counts)}")
        print(f"Min steps: {np.min(step_counts)}")

        coll_counts = [a.collisions for a in self.agents]
        print(f"Average collisions/agent: {np.mean(coll_counts):.2f}")
