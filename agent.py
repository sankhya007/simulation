# agent.py

import random
from typing import Tuple

from environment import EnvironmentGraph
from config import AGENT_REPLAN_PROB


class Agent:
    """Single agent moving along a path on the EnvironmentGraph."""

    _id_counter = 0

    def __init__(self, env: EnvironmentGraph):
        self.env = env
        self.id = Agent._id_counter
        Agent._id_counter += 1

        self.current_node: Tuple[int, int] = env.get_random_node()
        self.goal_node: Tuple[int, int] = env.get_random_node()
        while self.goal_node == self.current_node:
            self.goal_node = env.get_random_node()

        self.path = self.env.shortest_path(self.current_node, self.goal_node)
        self.path_index = 0

        self.finished = False
        self.steps_taken = 0
        self.collisions = 0

    # ---------- goal / path ----------

    def choose_new_goal(self):
        self.goal_node = self.env.get_random_node()
        while self.goal_node == self.current_node:
            self.goal_node = self.env.get_random_node()
        self.path = self.env.shortest_path(self.current_node, self.goal_node)
        self.path_index = 0
        self.finished = False

    def desired_next_node(self):
        """Node this agent wants to move to at this step."""
        if self.path_index >= len(self.path) - 1:
            self.finished = True

        if self.finished and random.random() < AGENT_REPLAN_PROB:
            self.choose_new_goal()

        if self.finished:
            return self.current_node

        return self.path[self.path_index + 1]

    # ---------- movement / position ----------

    def move_to(self, node):
        if node != self.current_node:
            self.current_node = node
            self.path_index += 1
            self.steps_taken += 1

    def get_position(self):
        return self.env.get_pos(self.current_node)
