# simulation.py

import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from environment import EnvironmentGraph
from agent import Agent
from config import COLLISION_DISTANCE


class CrowdSimulation:
    """Manages the whole crowd, interactions, and metrics."""

    def __init__(self, env: EnvironmentGraph, num_agents: int):
        self.env = env
        self.agents: List[Agent] = [Agent(env) for _ in range(num_agents)]
        self.time_step = 0

        self.node_visit_counts = defaultdict(int)
        self.total_collisions = 0

        for agent in self.agents:
            self.node_visit_counts[agent.current_node] += 1

    def step(self):
        """Single sim tick: desire → conflict resolution → move → collisions."""
        self.time_step += 1

        # 1) each agent chooses desired next node
        desires = {agent.id: agent.desired_next_node() for agent in self.agents}

        # 2) conflict resolution
        node_to_agents = defaultdict(list)
        for aid, node in desires.items():
            node_to_agents[node].append(aid)

        allowed_moves = {}
        for node, aids in node_to_agents.items():
            if len(aids) == 1:
                allowed_moves[aids[0]] = node
            else:
                import random

                winner = random.choice(aids)
                allowed_moves[winner] = node
                for loser in aids:
                    if loser != winner:
                        allowed_moves[loser] = None

        # 3) apply movements
        for agent in self.agents:
            target = allowed_moves.get(agent.id, None)
            if target is None:
                continue
            agent.move_to(target)
            self.node_visit_counts[agent.current_node] += 1

        # 4) collisions
        self._update_collisions()

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
