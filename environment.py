# environment.py

import random
from typing import List, Sequence, Optional, Dict, Tuple

import networkx as nx

from config import EDGE_BASE_CAPACITY


Node = Tuple[int, int]
LayoutMatrix = Sequence[Sequence[str]]


class EnvironmentGraph:
    """
    Environment represented as a graph.

    Nodes: (x, y) integer positions
        Attributes:
            - pos: (x, y) float coordinates for drawing
            - accessibility: "open" | "blocked" | "exit"
            - type: "corridor" | "wall" | "exit" | ...

    Edges: undirected connections between neighbouring nodes
        Attributes:
            - distance: Euclidean distance between nodes
            - max_capacity: how many agents can be comfortably on this edge
            - weight: cost used by pathfinding (initially == distance)
            - dynamic_weight: congestion-adjusted cost
    """

    def __init__(
        self,
        width: int,
        height: int,
        layout_matrix: Optional[LayoutMatrix] = None,
    ):
        """
        If layout_matrix is None: build a fully open grid of size (width x height).
        If layout_matrix is provided: infer width/height from it and build from symbols.
        """
        self.graph = nx.Graph()

        if layout_matrix is None:
            self.width = width
            self.height = height
            self._build_open_grid()
        else:
            self._build_from_layout(layout_matrix)

    # ------------------------------------------------------------------
    # PUBLIC UTILITIES (NODES)
    # ------------------------------------------------------------------

    def get_random_node(self) -> Node:
        accessible_nodes = [
            n
            for n, data in self.graph.nodes(data=True)
            if data.get("accessibility") in ("open", "exit")
        ]
        if not accessible_nodes:
            raise RuntimeError("No accessible nodes available in the environment.")
        return random.choice(accessible_nodes)

    def get_random_exit_node(self) -> Optional[Node]:
        exits = [
            n
            for n, data in self.graph.nodes(data=True)
            if data.get("accessibility") == "exit"
        ]
        if not exits:
            return None
        return random.choice(exits)

    def get_pos(self, node: Node) -> Tuple[float, float]:
        return self.graph.nodes[node]["pos"]

    def is_accessible(self, node: Node) -> bool:
        data = self.graph.nodes[node]
        return data.get("accessibility") in ("open", "exit")

    def is_exit(self, node: Node) -> bool:
        data = self.graph.nodes[node]
        return data.get("accessibility") == "exit"

    def get_neighbors(self, node: Node, accessible_only: bool = True) -> List[Node]:
        neighbors = list(self.graph.neighbors(node))
        if not accessible_only:
            return neighbors
        return [n for n in neighbors if self.is_accessible(n)]

    def block_node(self, node: Node):
        """
        Mark a node as blocked and remove its edges.
        Useful for dynamic obstacles / blocked path scenario.
        """
        if node not in self.graph:
            return

        self.graph.nodes[node]["accessibility"] = "blocked"
        self.graph.nodes[node]["type"] = "wall"

        for nbr in list(self.graph.neighbors(node)):
            self.graph.remove_edge(node, nbr)

    def unblock_node(self, node: Node, node_type: str = "corridor"):
        """
        Mark a node as open and reconnect it to accessible neighbours.
        """
        if node not in self.graph:
            return

        self.graph.nodes[node]["accessibility"] = "open"
        self.graph.nodes[node]["type"] = node_type

        x, y = node
        candidate_neighbors = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ]
        for nbr in candidate_neighbors:
            if nbr in self.graph and self.is_accessible(nbr):
                self._add_edge_with_defaults(node, nbr)

    def mark_exit(self, node: Node):
        """
        Convert a node into an exit node. Agents can still stand on it.
        """
        if node not in self.graph:
            return
        self.graph.nodes[node]["accessibility"] = "exit"
        self.graph.nodes[node]["type"] = "exit"

    def unmark_exit(self, node: Node):
        """
        Convert an exit back into a normal corridor.
        """
        if node not in self.graph:
            return
        if self.is_exit(node):
            self.graph.nodes[node]["accessibility"] = "open"
            self.graph.nodes[node]["type"] = "corridor"

    # ------------------------------------------------------------------
    # PATHFINDING
    # ------------------------------------------------------------------

    def shortest_path(self, start: Node, goal: Node) -> List[Node]:
        """
        Default shortest path using the 'weight' attribute (congestion-aware).
        """
        return self.shortest_path_weighted(start, goal, weight_attr="weight")

    def shortest_path_weighted(
        self,
        start: Node,
        goal: Node,
        weight_attr: str = "weight",
    ) -> List[Node]:
        """
        Generic weighted shortest path helper.
        - weight_attr="weight"    -> uses congestion-aware weights
        - weight_attr="distance"  -> ignores congestion, uses geometric distance
        """
        try:
            path = nx.astar_path(
                self.graph,
                start,
                goal,
                heuristic=lambda a, b: 0,
                weight=weight_attr,
            )
        except nx.NetworkXNoPath:
            path = [start]
        return path

    def shortest_path_to_nearest_exit(self, start: Node) -> List[Node]:
        """
        Finds the nearest exit (by path cost) and returns the path to it.
        If no exit exists or no path, returns [start].
        """
        exit_nodes = [
            n
            for n, data in self.graph.nodes(data=True)
            if data.get("accessibility") == "exit"
        ]
        if not exit_nodes:
            return [start]

        best_path: Optional[List[Node]] = None
        best_cost: float = float("inf")

        for exit_node in exit_nodes:
            try:
                path = nx.astar_path(
                    self.graph,
                    start,
                    exit_node,
                    heuristic=lambda a, b: 0,
                    weight="weight",
                )
                cost = 0.0
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    cost += self.graph[u][v].get("weight", 1.0)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
            except nx.NetworkXNoPath:
                continue

        if best_path is None:
            return [start]
        return best_path

    # ------------------------------------------------------------------
    # BUILDERS
    # ------------------------------------------------------------------

    def _build_open_grid(self):
        for y in range(self.height):
            for x in range(self.width):
                self.graph.add_node(
                    (x, y),
                    pos=(float(x), float(y)),
                    accessibility="open",
                    type="corridor",
                )

        for y in range(self.height):
            for x in range(self.width):
                if x + 1 < self.width:
                    self._add_edge_with_defaults((x, y), (x + 1, y))
                if y + 1 < self.height:
                    self._add_edge_with_defaults((x, y), (x, y + 1))

    def _build_from_layout(self, layout_matrix: LayoutMatrix):
        if not layout_matrix:
            raise ValueError("layout_matrix is empty")

        self.height = len(layout_matrix)
        self.width = len(layout_matrix[0])

        for row in layout_matrix:
            if len(row) != self.width:
                raise ValueError("All rows in layout_matrix must have same length")

        for y, row in enumerate(layout_matrix):
            for x, cell in enumerate(row):
                cell = str(cell)

                if cell in (".", "0"):
                    accessibility = "open"
                    node_type = "corridor"
                elif cell in ("#", "1"):
                    accessibility = "blocked"
                    node_type = "wall"
                elif cell.upper() == "E":
                    accessibility = "exit"
                    node_type = "exit"
                else:
                    accessibility = "open"
                    node_type = "corridor"

                self.graph.add_node(
                    (x, y),
                    pos=(float(x), float(y)),
                    accessibility=accessibility,
                    type=node_type,
                )

        for y in range(self.height):
            for x in range(self.width):
                node = (x, y)
                if not self.is_accessible(node):
                    continue

                if x + 1 < self.width:
                    nbr = (x + 1, y)
                    if self.is_accessible(nbr):
                        self._add_edge_with_defaults(node, nbr)

                if y + 1 < self.height:
                    nbr = (x, y + 1)
                    if self.is_accessible(nbr):
                        self._add_edge_with_defaults(node, nbr)

    # ------------------------------------------------------------------
    # EDGE HELPERS
    # ------------------------------------------------------------------

    def _add_edge_with_defaults(self, u: Node, v: Node):
        x1, y1 = self.graph.nodes[u]["pos"]
        x2, y2 = self.graph.nodes[v]["pos"]

        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        self.graph.add_edge(
            u,
            v,
            distance=distance,
            max_capacity=EDGE_BASE_CAPACITY,
            weight=distance,           # used by A*
            dynamic_weight=distance,   # updated when congestion-aware
        )

    def reset_edge_weights_to_distance(self):
        for _, _, data in self.graph.edges(data=True):
            base = data.get("distance", 1.0)
            data["weight"] = base
            data["dynamic_weight"] = base

    def set_edge_dynamic_weight(self, u: Node, v: Node, current_occupancy: int):
        edge_data = self.graph[u][v]
        base_distance = edge_data.get("distance", 1.0)
        capacity = edge_data.get("max_capacity", EDGE_BASE_CAPACITY)

        over = max(0, current_occupancy - capacity)
        congestion_ratio = over / max(1, capacity)

        dynamic_weight = base_distance * (1.0 + congestion_ratio)

        edge_data["dynamic_weight"] = dynamic_weight
        edge_data["weight"] = dynamic_weight

    def update_all_edge_weights_from_occupancy(
        self,
        occupancy_map: Dict[Tuple[Node, Node], int],
    ):
        for u, v, _ in self.graph.edges(data=True):
            key1 = (u, v)
            key2 = (v, u)
            occ = occupancy_map.get(key1, occupancy_map.get(key2, 0))
            self.set_edge_dynamic_weight(u, v, occ)
