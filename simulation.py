# simulation.py

import math
import random
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
    NAV_STRATEGY_MODE,
    NAV_STRATEGY_MIX,
)


Node = Tuple[int, int]
EdgeKey = Tuple[Node, Node]


class CrowdSimulation:
    """Manages the whole crowd, interactions, and metrics."""
    
    def get_metrics_summary(self) -> Dict[str, dict]:
        """
        Compute global, per-agent-type, and per-strategy metrics.

        Returns a dict like:
        {
            "global": {...},
            "by_type": {...},
            "by_strategy": {...},
        }
        """
        num_agents = len(self.agents)
        step_counts = np.array([a.steps_taken for a in self.agents])
        wait_counts = np.array([a.wait_steps for a in self.agents])
        replan_counts = np.array([a.replans for a in self.agents])
        coll_counts = np.array([a.collisions for a in self.agents])

        exit_flags = np.array([1 if a.exit_reached else 0 for a in self.agents])
        exit_times = np.array(
            [a.exit_time_step for a in self.agents if a.exit_time_step is not None]
        )

        # path optimality ratios (ignore agents with 0 ideal length)
        ratios = []
        for a in self.agents:
            if a.initial_shortest_path_len > 0:
                ratios.append(a.steps_taken / a.initial_shortest_path_len)
        ratios = np.array(ratios) if ratios else np.array([])

        global_metrics = {
            "num_agents": num_agents,
            "time_steps": self.time_step,
            "total_collisions": int(self.total_collisions),
            "avg_steps": float(step_counts.mean()) if num_agents > 0 else 0.0,
            "avg_waits": float(wait_counts.mean()) if num_agents > 0 else 0.0,
            "avg_replans": float(replan_counts.mean()) if num_agents > 0 else 0.0,
            "avg_collisions_per_agent": float(coll_counts.mean()) if num_agents > 0 else 0.0,
            "exit_rate": float(exit_flags.mean()) if num_agents > 0 else 0.0,
            "avg_exit_time": float(exit_times.mean()) if len(exit_times) > 0 else None,
            "avg_steps_over_optimal": float(ratios.mean()) if len(ratios) > 0 else None,
            "max_density_over_time": self.max_density_per_step,
        }

        # group by agent_type
        from collections import defaultdict
        agents_by_type = defaultdict(list)
        for a in self.agents:
            agents_by_type[a.agent_type].append(a)

        by_type: Dict[str, dict] = {}
        for t, group in agents_by_type.items():
            n = len(group)
            steps_t = np.array([a.steps_taken for a in group])
            waits_t = np.array([a.wait_steps for a in group])
            replans_t = np.array([a.replans for a in group])
            colls_t = np.array([a.collisions for a in group])
            exits_t = np.array([1 if a.exit_reached else 0 for a in group])

            ratios_t = []
            for a in group:
                if a.initial_shortest_path_len > 0:
                    ratios_t.append(a.steps_taken / a.initial_shortest_path_len)
            ratios_t = np.array(ratios_t) if ratios_t else np.array([])

            by_type[t] = {
                "count": n,
                "avg_steps": float(steps_t.mean()) if n > 0 else 0.0,
                "avg_waits": float(waits_t.mean()) if n > 0 else 0.0,
                "avg_replans": float(replans_t.mean()) if n > 0 else 0.0,
                "avg_collisions": float(colls_t.mean()) if n > 0 else 0.0,
                "exit_rate": float(exits_t.mean()) if n > 0 else 0.0,
                "avg_steps_over_optimal": float(ratios_t.mean()) if len(ratios_t) > 0 else None,
            }

        # group by navigation strategy
        agents_by_strategy = defaultdict(list)
        for a in self.agents:
            agents_by_strategy[a.strategy].append(a)

        by_strategy: Dict[str, dict] = {}
        for s, group in agents_by_strategy.items():
            n = len(group)
            steps_s = np.array([a.steps_taken for a in group])
            waits_s = np.array([a.wait_steps for a in group])
            replans_s = np.array([a.replans for a in group])
            colls_s = np.array([a.collisions for a in group])
            exits_s = np.array([1 if a.exit_reached else 0 for a in group])

            ratios_s = []
            for a in group:
                if a.initial_shortest_path_len > 0:
                    ratios_s.append(a.steps_taken / a.initial_shortest_path_len)
            ratios_s = np.array(ratios_s) if ratios_s else np.array([])

            by_strategy[s] = {
                "count": n,
                "avg_steps": float(steps_s.mean()) if n > 0 else 0.0,
                "avg_waits": float(waits_s.mean()) if n > 0 else 0.0,
                "avg_replans": float(replans_s.mean()) if n > 0 else 0.0,
                "avg_collisions": float(colls_s.mean()) if n > 0 else 0.0,
                "exit_rate": float(exits_s.mean()) if n > 0 else 0.0,
                "avg_steps_over_optimal": float(ratios_s.mean()) if len(ratios_s) > 0 else None,
            }

        return {
            "global": global_metrics,
            "by_type": by_type,
            "by_strategy": by_strategy,
        }





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
        self.max_density_per_step: List[int] = []

        for agent in self.agents:
            self.node_visit_counts[agent.current_node] += 1

    # ---------- initialization helpers ----------

    def _init_agents_with_groups(self, num_agents: int):
        # --- decide navigation strategies for each agent ---
        if NAV_STRATEGY_MODE == "mixed":
            frac_short = NAV_STRATEGY_MIX.get("shortest", 0.0)
            frac_cong = NAV_STRATEGY_MIX.get("congestion", 0.0)
            frac_safe = NAV_STRATEGY_MIX.get("safe", 0.0)

            n_short = int(num_agents * frac_short)
            n_cong = int(num_agents * frac_cong)
            n_safe = num_agents - n_short - n_cong

            strategies = (
                ["shortest"] * n_short
                + ["congestion"] * n_cong
                + ["safe"] * n_safe
            )
            random.shuffle(strategies)
        else:
            # uniform population of a single strategy
            strategies = [NAV_STRATEGY_MODE] * num_agents

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
                    navigation_strategy=strategies[agent_index],
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
                        navigation_strategy=strategies[agent_index],
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
                    navigation_strategy=strategies[agent_index],
                )
            )
            agent_index += 1


    # ---------- dynamic events ----------

    def _apply_dynamic_events(self, node_occupancy: Dict[Node, int]):
        """
        Introduce dynamic obstacles and exit changes during the simulation.
        - Randomly blocks some unoccupied nodes.
        - Randomly opens/closes exits.
        """
        import random

        # Dynamic blocking of nodes
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
        - track exit arrivals and density metrics
        - measure collisions
        """
        self.time_step += 1

        # 1) node occupancy at current time
        node_occupancy: Dict[Node, int] = defaultdict(int)
        for agent in self.agents:
            node_occupancy[agent.current_node] += 1
        self.last_node_occupancy = node_occupancy

        # track max density this step
        max_density = max(node_occupancy.values()) if node_occupancy else 0
        self.max_density_per_step.append(max_density)

        # 2) apply dynamic environment events
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

        # 10) mark exit arrivals (for evacuation metrics)
        for agent in self.agents:
            if agent.exit_time_step is None and self.env.is_exit(agent.current_node):
                agent.exit_reached = True
                agent.exit_time_step = self.time_step

        # 11) collisions
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

        wait_counts = [a.wait_steps for a in self.agents]
        print(f"Average waits/agent: {np.mean(wait_counts):.2f}")

        replan_counts = [a.replans for a in self.agents]
        print(f"Average replans/agent: {np.mean(replan_counts):.2f}")

        # Path optimality (based on initial goal)
        ratios = []
        for a in self.agents:
            if a.initial_shortest_path_len > 0:
                ratios.append(a.steps_taken / a.initial_shortest_path_len)
        if ratios:
            print(f"Avg steps / optimal(initial) ratio: {np.mean(ratios):.2f}")

        # Exit metrics (for evacuation scenarios)
        exit_agents = [a for a in self.agents if a.exit_reached]
        if exit_agents:
            times = [a.exit_time_step for a in exit_agents if a.exit_time_step is not None]
            print(f"Agents that reached an exit: {len(exit_agents)}/{len(self.agents)}")
            print(f"Average exit time step: {np.mean(times):.2f}")
        else:
            print("No agents reached an exit (in this scenario).")
