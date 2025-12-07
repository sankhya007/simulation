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
            - dynamic_weight: congestion-adjusted cost (optional, Step 3)
    """

    def __init__(
        self,
        width: int,
        height: int,
        layout_matrix: Optional[LayoutMatrix] = None,
    ):
        """
        If layout_matrix is None:
            build a fully open grid of size (width x height).

        If layout_matrix is provided:
            - width and height are ignored, inferred from layout_matrix
            - each cell defines node accessibility/type:
                '.' or '0' -> open corridor
                '#' or '1' -> blocked wall
                'E'        -> exit
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
        """
        Returns a random *accessible* node: accessibility in {"open", "exit"}.
        """
        accessible_nodes = [
            n
            for n, data in self.graph.nodes(data=True)
            if data.get("accessibility") in ("open", "exit")
        ]
        if not accessible_nodes:
            raise RuntimeError("No accessible nodes available in the environment.")
        return random.choice(accessible_nodes)

    def get_random_exit_node(self) -> Optional[Node]:
        """
        Returns a random exit node, or None if there is no exit.
        """
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
        """
        True if node is open or exit (i.e., agents can stand on it).
        """
        data = self.graph.nodes[node]
        return data.get("accessibility") in ("open", "exit")

    def is_exit(self, node: Node) -> bool:
        """
        True if node is marked as an exit.
        """
        data = self.graph.nodes[node]
        return data.get("accessibility") == "exit"

    def get_neighbors(self, node: Node, accessible_only: bool = True) -> List[Node]:
        """
        Return neighbours of given node.
        If accessible_only is True, filters out blocked nodes.
        """
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

        # remove all edges from this node
        for nbr in list(self.graph.neighbors(node)):
            self.graph.remove_edge(node, nbr)

    def unblock_node(self, node: Node, node_type: str = "corridor"):
        """
        Mark a node as open corridor and reconnect it to accessible neighbours.
        """
        if node not in self.graph:
            return

        self.graph.nodes[node]["accessibility"] = "open"
        self.graph.nodes[node]["type"] = node_type

        x, y = node
        # Reconnect to 4-neighbours that are accessible
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

    # ------------------------------------------------------------------
    # PATHFINDING
    # ------------------------------------------------------------------

    def shortest_path(self, start: Node, goal: Node) -> List[Node]:
        """
        Shortest path using A* with the 'weight' edge attribute.
        """
        try:
            path = nx.astar_path(
                self.graph,
                start,
                goal,
                heuristic=lambda a, b: 0,  # can be replaced with geometric heuristic
                weight="weight",
            )
        except nx.NetworkXNoPath:
            path = [start]
        return path

    def shortest_path_to_nearest_exit(self, start: Node) -> List[Node]:
        """
        Finds the nearest exit (by path cost) and returns the path to it.
        If no exit exists or no path, returns [start].
        """
        # collect all exits
        exit_nodes = [
            n
            for n, data in self.graph.nodes(data=True)
            if data.get("accessibility") == "exit"
        ]
        if not exit_nodes:
            # no exits defined
            return [start]

        # find best exit by running shortest path to each (could be optimized)
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
                # path cost = sum of edge weights
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
        """
        Default: fully open rectangular grid with corridor nodes.
        """
        # nodes
        for y in range(self.height):
            for x in range(self.width):
                self.graph.add_node(
                    (x, y),
                    pos=(float(x), float(y)),
                    accessibility="open",
                    type="corridor",
                )

        # edges (4-neighbour)
        for y in range(self.height):
            for x in range(self.width):
                if x + 1 < self.width:
                    self._add_edge_with_defaults((x, y), (x + 1, y))
                if y + 1 < self.height:
                    self._add_edge_with_defaults((x, y), (x, y + 1))

    def _build_from_layout(self, layout_matrix: LayoutMatrix):
        """
        Build graph from a given layout matrix (list of rows).

        Example layout (string rows):
            "..E.."
            ".###."
            "..0.."

        Legend (you can customize as you like):
            '.' or '0' -> open corridor
            '#' or '1' -> blocked wall
            'E'        -> exit
        """
        if not layout_matrix:
            raise ValueError("layout_matrix is empty")

        self.height = len(layout_matrix)
        self.width = len(layout_matrix[0])

        # sanity check: rectangular
        for row in layout_matrix:
            if len(row) != self.width:
                raise ValueError("All rows in layout_matrix must have same length")

        # --- create nodes with attributes ---
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

        # --- create edges only between accessible nodes ---
        for y in range(self.height):
            for x in range(self.width):
                node = (x, y)
                if not self.is_accessible(node):
                    continue

                # right neighbour
                if x + 1 < self.width:
                    nbr = (x + 1, y)
                    if self.is_accessible(nbr):
                        self._add_edge_with_defaults(node, nbr)

                # up neighbour
                if y + 1 < self.height:
                    nbr = (x, y + 1)
                    if self.is_accessible(nbr):
                        self._add_edge_with_defaults(node, nbr)

    # ------------------------------------------------------------------
    # EDGE HELPERS
    # ------------------------------------------------------------------

    def _add_edge_with_defaults(self, u: Node, v: Node):
        """
        Add an edge between u and v with default distance, capacity, and weight.
        """
        x1, y1 = self.graph.nodes[u]["pos"]
        x2, y2 = self.graph.nodes[v]["pos"]

        # Euclidean distance; could be 1 for grid, but this is more general
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        self.graph.add_edge(
            u,
            v,
            distance=distance,
            max_capacity=EDGE_BASE_CAPACITY,
            weight=distance,           # used by A*
            dynamic_weight=distance,   # will be updated when congestion-aware
        )

    def reset_edge_weights_to_distance(self):
        """
        Reset all edges so that 'weight' == base 'distance' (no congestion).
        """
        for _, _, data in self.graph.edges(data=True):
            base = data.get("distance", 1.0)
            data["weight"] = base
            data["dynamic_weight"] = base

    def set_edge_dynamic_weight(self, u: Node, v: Node, current_occupancy: int):
        """
        Update a single edge's dynamic weight based on how many agents
        are currently using it (current_occupancy).

        Formula (simple, can be changed later):
            congestion_ratio = max(0, occupancy - capacity) / capacity
            dynamic_weight = distance * (1 + congestion_ratio)

        And 'weight' is also updated so A* uses the new cost.
        """
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
        """
        occupancy_map: dict[(u, v)] -> occupancy count
        where (u, v) are edge endpoint tuples (order not important).

        This lets the simulation pass in live congestion info and get
        congestion-aware edge weights.
        """
        for u, v, _ in self.graph.edges(data=True):
            key1 = (u, v)
            key2 = (v, u)
            occ = occupancy_map.get(key1, occupancy_map.get(key2, 0))
            self.set_edge_dynamic_weight(u, v, occ)
