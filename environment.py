# environment.py

import random
import networkx as nx


class EnvironmentGraph:
    """
    Grid environment stored as a graph.
    Nodes: (x, y) grid positions
    Edges: 4-neighbour connectivity
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.graph = nx.Graph()
        self._build_grid()

    def _build_grid(self):
        # Add nodes
        for y in range(self.height):
            for x in range(self.width):
                self.graph.add_node((x, y), pos=(x, y))

        # Add edges (right & up neighbours)
        for y in range(self.height):
            for x in range(self.width):
                if x + 1 < self.width:
                    self.graph.add_edge((x, y), (x + 1, y), weight=1.0)
                if y + 1 < self.height:
                    self.graph.add_edge((x, y), (x, y + 1), weight=1.0)

    def get_random_node(self):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        return (x, y)

    def get_pos(self, node):
        return self.graph.nodes[node]["pos"]

    def shortest_path(self, start, goal):
        """A* (same as Dijkstra here, heuristic=0)."""
        try:
            path = nx.astar_path(
                self.graph,
                start,
                goal,
                heuristic=lambda a, b: 0,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            path = [start]
        return path
