# simulation.py

import math
import random
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import config

from environment import EnvironmentGraph
from agent import Agent, DecisionState
from config import (
    COLLISION_DISTANCE,
    ENABLE_CONGESTION_WEIGHTS,
    ENABLE_DYNAMIC_EVENTS,
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
# motion models imported if present
try:
    from motion_models import MotionState, SocialForceModel, RVOMotionModel, GraphMotionModel
except Exception:
    MotionState = None
    SocialForceModel = None
    RVOMotionModel = None
    GraphMotionModel = None


Node = Tuple[int, int]
EdgeKey = Tuple[Node, Node]


class CrowdSimulation:
    """Manages the whole crowd, interactions, and metrics."""

    def get_metrics_summary(self) -> Dict[str, dict]:
        num_agents = len(self.agents)
        step_counts = np.array([a.steps_taken for a in self.agents])
        wait_counts = np.array([a.wait_steps for a in self.agents])
        replan_counts = np.array([a.replans for a in self.agents])
        coll_counts = np.array([a.collisions for a in self.agents])

        exit_flags = np.array([1 if a.exit_reached else 0 for a in self.agents])
        exit_times = np.array(
            [a.exit_time_step for a in self.agents if a.exit_time_step is not None]
        )

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

        # Motion model selection
        mmode = getattr(config, "MOTION_MODEL", "graph")
        if mmode == "social_force" and SocialForceModel is not None:
            self.global_motion_model = SocialForceModel(
                relaxation_time=getattr(config, "SF_RELAX_T", 0.5),
                repulsion_A=getattr(config, "SF_A", 2.0),
                repulsion_B=getattr(config, "SF_B", 0.5),
                agent_radius=getattr(config, "AGENT_RADIUS", 0.3),
                max_speed=getattr(config, "SF_MAX_SPEED", 1.5),
            )
        elif mmode == "rvo" and RVOMotionModel is not None:
            self.global_motion_model = RVOMotionModel(
                neighbor_dist=getattr(config, "RVO_NEIGHBOR_DIST", 3.0),
                max_speed=getattr(config, "RVO_MAX_SPEED", 1.5),
                samples=getattr(config, "RVO_SAMPLES", 16),
            )
        else:
            # fallback to Graph adapter if available
            if GraphMotionModel is not None:
                self.global_motion_model = GraphMotionModel()
            else:
                self.global_motion_model = None

        for a in self.agents:
            if self.global_motion_model is not None:
                a.motion_model = self.global_motion_model

    # ---------- initialization helpers ----------
    def _init_agents_with_groups(self, num_agents: int):
        if NAV_STRATEGY_MODE == "mixed":
            frac_short = NAV_STRATEGY_MIX.get("shortest", 0.0)
            frac_cong = NAV_STRATEGY_MIX.get("congestion", 0.0)
            frac_safe = NAV_STRATEGY_MIX.get("safe", 0.0)

            n_short = int(num_agents * frac_short)
            n_cong = int(num_agents * frac_cong)
            n_safe = num_agents - n_short - n_cong

            strategies = ["shortest"] * n_short + ["congestion"] * n_cong + ["safe"] * n_safe
            random.shuffle(strategies)
        else:
            strategies = [NAV_STRATEGY_MODE] * num_agents

        full_groups = num_agents // GROUP_SIZE
        agent_index = 0
        group_id = 1

        for _ in range(full_groups):
            self.agents.append(
                Agent(
                    env=self.env,
                    agent_type="leader",
                    group_id=group_id,
                    navigation_strategy=strategies[agent_index],
                )
            )
            agent_index += 1

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
        import random

        if DYNAMIC_BLOCKS_ENABLED and self.time_step % BLOCK_NODE_EVERY_N_STEPS == 0:
            candidates = [
                n
                for n, data in self.env.graph.nodes(data=True)
                if self.env.is_accessible(n) and node_occupancy.get(n, 0) == 0
            ]
            if candidates and random.random() < BLOCK_NODE_PROB:
                node = random.choice(candidates)
                self.env.block_node(node)

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
        Single sim tick.
        """
        # advance time
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
        if ENABLE_DYNAMIC_EVENTS:
            self._apply_dynamic_events(node_occupancy)

        # 3) edge occupancy approximation from node occupancy, then update weights
        edge_occupancy: Dict[EdgeKey, int] = {}
        edge_over_capacity: Dict[EdgeKey, bool] = {}

        if ENABLE_CONGESTION_WEIGHTS:
            for u, v in self.env.graph.edges():
                occ = node_occupancy.get(u, 0) + node_occupancy.get(v, 0)
                edge_occupancy[(u, v)] = occ

                cap = self.env.graph[u][v].get("max_capacity", EDGE_BASE_CAPACITY)
                over = occ > cap
                edge_over_capacity[(u, v)] = over

                self.env.set_edge_dynamic_weight(u, v, occ)
        else:
            self.env.reset_edge_weights_to_distance()

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

        # Continuous motion models branch
        if getattr(config, "MOTION_MODEL", "graph") in ("social_force", "rvo") and MotionState is not None:
            # prepare MotionState inputs
            pos_map = {a.id: a.get_position() for a in self.agents}
            vel_map = {a.id: getattr(a, "vel", (0.0, 0.0)) for a in self.agents}

            motion_state = MotionState(
                time_step=self.time_step,
                positions=pos_map,
                velocities=vel_map,
                node_occupancy=node_occupancy,
                group_targets=group_targets,
                edge_over_capacity=edge_over_capacity,
                global_density=global_density,
            )

            # compute candidate velocities
            new_vels = {}
            for agent in self.agents:
                if getattr(agent, "motion_model", None) is not None:
                    vx, vy = agent.motion_model.compute_velocity(agent, motion_state, self.env)
                else:
                    # fallback: zero velocity
                    vx, vy = (0.0, 0.0)
                new_vels[agent.id] = (vx, vy)

            # integrate and map to nodes
            for agent in self.agents:
                vx, vy = new_vels[agent.id]
                # simple dt=1 step; you can reduce dt via substeps if unstable
                agent.integrate_continuous(vx, vy, dt=1.0)

                # robust mapping: prefer MapMeta.cad_to_grid but clamp to valid range,
                # and if result is not a node in the graph fall back to nearest-node search.
                mapped_node = None
                try:
                    mm = getattr(self.env, "_mapmeta", None)
                    if mm is not None and callable(getattr(mm, "cad_to_grid", None)):
                        gx, gy = mm.cad_to_grid(agent.pos[0], agent.pos[1])
                        # clamp to valid integer grid indices
                        gw, gh = self.env.width, self.env.height
                        igx = max(0, min(int(gx), gw - 1))
                        igy = max(0, min(int(gy), gh - 1))
                        candidate = (igx, igy)
                        # ensure candidate exists in graph (some maps may have inaccessible border)
                        if candidate in self.env.graph:
                            mapped_node = candidate
                        else:
                            mapped_node = None  # force fallback
                    # else fallthrough to nearest-node fallback
                except Exception:
                    mapped_node = None

                if mapped_node is None:
                    # fallback: find nearest existing node in graph by Euclidean distance
                    try:
                        best = min(
                            self.env.graph.nodes(),
                            key=lambda n: math.hypot(agent.pos[0] - self.env.get_pos(n)[0], agent.pos[1] - self.env.get_pos(n)[1]),
                        )
                        mapped_node = best
                    except Exception:
                        # last resort: keep the previous node to avoid creating invalid key
                        mapped_node = getattr(agent, "current_node", None)

                # assign mapped node if valid
                if mapped_node is not None:
                    agent.current_node = mapped_node

                # update node visit (guard against None)
                if agent.current_node is not None:
                    self.node_visit_counts[agent.current_node] += 1


            # collisions & exits
            self._update_collisions()

            for agent in self.agents:
                if agent.exit_time_step is None and self.env.is_exit(agent.current_node):
                    agent.exit_reached = True
                    agent.exit_time_step = self.time_step

            # update occupancy snapshot
            node_occupancy = {}
            for a in self.agents:
                node_occupancy[a.current_node] = node_occupancy.get(a.current_node, 0) + 1
            self.last_node_occupancy = node_occupancy
            self.max_density_per_step.append(max(node_occupancy.values()) if node_occupancy else 0)

            return

        # Legacy discrete graph-based behaviour (unchanged)
        state = DecisionState(
            time_step=self.time_step,
            node_occupancy=node_occupancy,
            group_targets=group_targets,
            edge_over_capacity=edge_over_capacity,
            agent_positions=agent_positions,
            agent_types=agent_types,
            global_density=global_density,
        )

        desires: Dict[int, Node] = {}
        for agent in self.agents:
            desired = agent.desired_next_node(state)
            desires[agent.id] = desired

        node_to_agents = defaultdict(list)
        for agent_id, node in desires.items():
            node_to_agents[node].append(agent_id)

        allowed_moves: Dict[int, Node | None] = {}
        for node, agent_ids in node_to_agents.items():
            if len(agent_ids) == 1:
                allowed_moves[agent_ids[0]] = node
            else:
                winner = random.choice(agent_ids)
                allowed_moves[winner] = node
                for loser in agent_ids:
                    if loser != winner:
                        allowed_moves[loser] = None

        for agent in self.agents:
            target = allowed_moves.get(agent.id, None)
            if target is None:
                continue
            agent.move_to(target)
            self.node_visit_counts[agent.current_node] += 1

        for agent in self.agents:
            if agent.exit_time_step is None and self.env.is_exit(agent.current_node):
                agent.exit_reached = True
                agent.exit_time_step = self.time_step

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

    def get_agent_positions(self):
        xs, ys = [], []
        for agent in self.agents:
            x, y = agent.get_position()
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def get_density_matrix(self):
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

        ratios = []
        for a in self.agents:
            if a.initial_shortest_path_len > 0:
                ratios.append(a.steps_taken / a.initial_shortest_path_len)
        if ratios:
            print(f"Avg steps / optimal(initial) ratio: {np.mean(ratios):.2f}")

        exit_agents = [a for a in self.agents if a.exit_reached]
        if exit_agents:
            times = [a.exit_time_step for a in exit_agents if a.exit_time_step is not None]
            print(f"Agents that reached an exit: {len(exit_agents)}/{len(self.agents)}")
            print(f"Average exit time step: {np.mean(times):.2f}")
        else:
            print("No agents reached an exit (in this scenario).")
