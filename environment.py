# environment.py

import random
from typing import List, Sequence, Optional, Dict, Tuple

import networkx as nx

from config import EDGE_BASE_CAPACITY
from maps.map_meta import MapMeta
from typing import Optional
from typing import Any
import numpy as np

Node = Tuple[int, int]
LayoutMatrix = Sequence[Sequence[str]]

# Removed stray undefined code: node/world coordinates should be set when nodes are created
# (see _build_open_grid and _build_from_layout which set node "pos" attributes).


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

    def get_edge_slowdown(self, u: Node, v: Node) -> float:
        """
        Returns a slowdown factor >= 1.0 based on how 'heavy' this edge is.

        slowdown = weight / distance

        - If there is no congestion: weight ~= distance -> slowdown ≈ 1.
        - If congestion increases weight, slowdown > 1 -> agent moves less often.
        """
        if not self.graph.has_edge(u, v):
            return 1.0

        data = self.graph[u][v]
        dist = data.get("distance", 1.0)
        weight = data.get("weight", dist)

        if dist <= 0:
            return 1.0

        slowdown = weight / dist
        return max(1.0, slowdown)

    def __init__(
        self,
        width: int,
        height: int,
        layout_matrix: Optional[LayoutMatrix] = None,
        mapmeta: Optional[MapMeta] = None,
        *,
        graph_type: str = "grid",  # "grid" | "centerline" | "hybrid"
        cell_size: Optional[float] = None,  # world units per cell (smaller => more nodes)
    ):
        """
        Build environment. New options:
          - graph_type: "grid" (default), "centerline", "hybrid"
          - cell_size: optional world units per grid cell. If provided with mapmeta,
                       will determine grid resolution. If None, uses mapmeta.grid_shape.
        Backwards compatible: if layout_matrix provided, old layout-based builder used
        (but can still pass mapmeta and graph_type to control node positions).
        """
        self.graph = nx.Graph()
        self._mapmeta = mapmeta  # may be None
        self.graph_type = graph_type
        self.cell_size = cell_size

        # If layout provided, keep compatibility but allow graph_type handling
        if layout_matrix is not None:
            # prefer mapmeta-driven building if mapmeta given and graph_type != "grid"
            if mapmeta is not None and graph_type in ("centerline", "hybrid"):
                # build from mapmeta rasterization
                if graph_type == "centerline":
                    self._build_centerline_graph(mapmeta)
                else:
                    self._build_hybrid_graph(mapmeta, cell_size=cell_size)
            else:
                # fallback to legacy layout builder (grid aligned to layout_matrix)
                self._build_from_layout(layout_matrix, mapmeta=mapmeta)
        else:
            # No layout matrix: build an open grid with requested resolution
            if mapmeta is not None and graph_type in ("centerline", "hybrid"):
                if graph_type == "centerline":
                    self._build_centerline_graph(mapmeta)
                else:
                    self._build_hybrid_graph(mapmeta, cell_size=cell_size)
            else:
                # grid builder uses width/height (legacy behavior)
                self.width = int(width)
                self.height = int(height)
                self._build_open_grid(mapmeta=mapmeta)

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
            n for n, data in self.graph.nodes(data=True) if data.get("accessibility") == "exit"
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
            n for n, data in self.graph.nodes(data=True) if data.get("accessibility") == "exit"
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

    def _build_open_grid(self, mapmeta: Optional[MapMeta] = None):
        """
        Build a width x height open grid. If mapmeta is provided, use its transform
        to compute node positions in world coordinates; otherwise use integer coords.
        """
        for y in range(self.height):
            for x in range(self.width):
                if mapmeta is not None:
                    # mapmeta.transform expects grid indices (gx,gy) -> (real_x, real_y)
                    wx, wy = mapmeta.transform(x, y)
                    pos = (float(wx), float(wy))
                else:
                    pos = (float(x), float(y))

                self.graph.add_node(
                    (x, y),
                    pos=pos,
                    accessibility="open",
                    type="corridor",
                )

        for y in range(self.height):
            for x in range(self.width):
                if x + 1 < self.width:
                    self._add_edge_with_defaults((x, y), (x + 1, y))
                if y + 1 < self.height:
                    self._add_edge_with_defaults((x, y), (x, y + 1))

    def _build_from_layout(self, layout_matrix: LayoutMatrix, mapmeta: Optional[MapMeta] = None):
        """
        Build graph from layout_matrix. If mapmeta provided, use it to compute world
        pos for each node; otherwise fallback to grid coords.
        """
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

                if mapmeta is not None:
                    wx, wy = mapmeta.transform(x, y)
                    pos = (float(wx), float(wy))
                else:
                    pos = (float(x), float(y))

                self.graph.add_node(
                    (x, y),
                    pos=pos,
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
            weight=distance,  # used by A*
            dynamic_weight=distance,  # updated when congestion-aware
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

    # ============================================================
    #  NEW GRAPH BUILDERS (CENTERLINE + HYBRID)
    # ============================================================

    def _build_centerline_graph(self, mapmeta):
        """
        Build a centerline / skeleton graph from the raster layout.
        Requires scikit-image.
        """
        try:
            from skimage.morphology import skeletonize
            from skimage import img_as_bool
        except Exception:
            raise ImportError(
                "scikit-image is required for graph_type='centerline'. "
                "Install with: python -m pip install scikit-image scipy"
            )

        layout = mapmeta.layout
        H = len(layout)
        W = len(layout[0])

        import numpy as np

        # Build walkable mask
        mask = np.zeros((H, W), dtype=bool)
        for y in range(H):
            for x in range(W):
                c = layout[y][x]
                if c in (".", "0", "E"):  # walkable or exit
                    mask[y, x] = True

        # Skeleton
        skel = skeletonize(mask)

        self.width = W
        self.height = H

        # Add nodes at skeleton pixels
        for y in range(H):
            for x in range(W):
                if not skel[y, x]:
                    continue

                wx, wy = mapmeta.transform(x, y)
                self.graph.add_node(
                    (x, y),
                    pos=(wx, wy),
                    accessibility="open",
                    type="corridor",
                )

        # 8-connected graph edges
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for x, y in list(self.graph.nodes()):
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.graph:
                    self._add_edge_with_defaults((x, y), (nx, ny))

        print(f"[EnvironmentGraph] centerline graph built: {self.graph.number_of_nodes()} nodes")

    def _build_hybrid_graph(self, mapmeta, cell_size=0.5):
        """
        Hybrid graph:
          - Start with centerline skeleton
          - Add sparse nodes in open areas
        """
        # Start by building the centerline
        try:
            self._build_centerline_graph(mapmeta)
        except ImportError:
            print("[EnvironmentGraph] scikit-image missing: hybrid → fallback to grid.")
            self._build_from_layout(mapmeta.extras["layout"], mapmeta=mapmeta)
            return

        layout = mapmeta.layout
        H = len(layout)
        W = len(layout[0])
        import numpy as np

        # How densely to sample extra nodes
        step = max(1, int(1 / cell_size))

        # Add sampled open-space nodes
        for y in range(0, H, step):
            for x in range(0, W, step):
                c = layout[y][x]
                if c not in (".", "0", "E"):
                    continue
                if (x, y) in self.graph:
                    continue

                wx, wy = mapmeta.transform(x, y)
                self.graph.add_node(
                    (x, y),
                    pos=(wx, wy),
                    accessibility="open",
                    type="corridor",
                )

        # Connect neighbors
        for x, y in list(self.graph.nodes()):
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.graph:
                    self._add_edge_with_defaults((x, y), (nx, ny))

        print(f"[EnvironmentGraph] hybrid graph built: {self.graph.number_of_nodes()} nodes")

    # ----------------------------
    # New graph builders (resolution / centerline / hybrid)
    # ----------------------------

    def _compute_raster_from_layout(self, mapmeta: MapMeta, gw: int, gh: int):
        """
        Produce a binary occupancy (walls=1, free=0) image of shape (gh, gw)
        from the layout (list of rows). This is useful for skeletonization.
        """
        # layout aligned grid from mapmeta (we assume mm.grid_shape = (gw, gh))
        layout = None
        try:
            layout = (
                mapmeta.extras.get("layout")
                if mapmeta.extras and "layout" in mapmeta.extras
                else None
            )
        except Exception:
            layout = None

        # fallback to building from env if layout not present in extras
        if layout is None:
            # try to create layout from current env graph if available
            # build a grid occupancy from nodes
            occupancy = np.ones((gh, gw), dtype=np.uint8)  # 1=wall by default
            for y in range(gh):
                for x in range(gw):
                    node = (x, y)
                    try:
                        if self.graph.nodes[node].get("accessibility") in ("open", "exit"):
                            occupancy[y, x] = 0
                    except Exception:
                        # if graph not yet filled, try mapmeta layout
                        pass
            return occupancy

        arr = np.zeros((gh, gw), dtype=np.uint8)
        for y, row in enumerate(layout):
            for x, ch in enumerate(row):
                if str(ch) in ("#", "1"):
                    arr[y, x] = 1
                else:
                    arr[y, x] = 0
        return arr

    def _build_centerline_graph(self, mapmeta: MapMeta):
        """
        Build a graph using the medial axis (skeleton) of the free space mask.
        Requires scikit-image (skimage). If not available, raises ImportError with hint.
        """
        try:
            from skimage.morphology import skeletonize
            from skimage.filters import threshold_otsu
            from skimage import img_as_bool
        except Exception as e:
            raise ImportError(
                "Centerline graph requires scikit-image. Install via: pip install scikit-image"
            ) from e

        # Derive grid shape to rasterize (prefer mapmeta.grid_shape)
        gw, gh = mapmeta.grid_shape
        # get occupancy mask 1=wall, 0=free
        occ = self._compute_raster_from_layout(mapmeta, gw, gh)
        # skeletonize expects boolean image where True = foreground; we want skeleton of free space
        free_mask = occ == 0
        # Convert to boolean
        free_mask = img_as_bool(free_mask)
        skeleton = skeletonize(free_mask)

        # Map skeleton pixels to nodes: each True pixel -> node
        node_idxs = {}
        for y in range(skeleton.shape[0]):
            for x in range(skeleton.shape[1]):
                if skeleton[y, x]:
                    # world pos from mapmeta.transform
                    wx, wy = mapmeta.transform(x, y)
                    node = (int(x), int(y))
                    self.graph.add_node(
                        node, pos=(float(wx), float(wy)), accessibility="open", type="corridor"
                    )
                    node_idxs[(x, y)] = node

        # Connect adjacent skeleton pixels (8-neighbour)
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for (x, y), _ in node_idxs.items():
            u = (x, y)
            for dx, dy in nbrs:
                v = (x + dx, y + dy)
                if v in node_idxs:
                    self._add_edge_with_defaults(u, v)

        # Mark exits if they align with skeleton (optional: detect near exit layers in mapmeta.extras)
        # leave walls/blocked nodes out — centerline nodes are inherently walkable

        # store metadata
        self.width = gw
        self.height = gh

    def _build_hybrid_graph(self, mapmeta: MapMeta, cell_size: Optional[float] = None):
        """
        Build a hybrid graph: denser grid in narrow corridors, sparser along centerline in open areas.
        Simple heuristic:
          1. Build full grid at resolution gw x gh from mapmeta.grid_shape
          2. Compute skeleton of free space
          3. Keep all grid nodes within narrow regions (distance transform threshold)
          4. In open regions, prune grid nodes and keep only nodes close to skeleton (sparser)
        """
        # attempt to import necessary skimage functions; if missing, fallback to grid
        try:
            from skimage.morphology import skeletonize
            from scipy.ndimage import distance_transform_edt
            from skimage import img_as_bool
        except Exception:
            # if libraries missing, fallback to grid graph
            self._build_from_layout(mapmeta.extras.get("layout", [[]]), mapmeta=mapmeta)
            return

        gw, gh = mapmeta.grid_shape
        occ = self._compute_raster_from_layout(mapmeta, gw, gh)
        free_mask = img_as_bool(occ == 0)
        skeleton = skeletonize(free_mask)
        dist = distance_transform_edt(free_mask)  # distance to nearest wall in pixels

        # thresholds (in pixels): narrow corridor threshold => keep full grid
        narrow_threshold = 3  # pixels (tuneable)
        for y in range(gh):
            for x in range(gw):
                if occ[y, x] == 1:
                    continue  # wall
                # keep node if near skeleton or if narrow corridor
                if dist[y, x] <= narrow_threshold or skeleton[y, x]:
                    wx, wy = mapmeta.transform(x, y)
                    node = (x, y)
                    self.graph.add_node(
                        node, pos=(float(wx), float(wy)), accessibility="open", type="corridor"
                    )

        # connect neighbours for nodes that exist
        for x, y in list(self.graph.nodes()):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nbr = (x + dx, y + dy)
                if nbr in self.graph:
                    self._add_edge_with_defaults((x, y), nbr)

        self.width = gw
        self.height = gh

    def _build_grid_graph_from_mapmeta(
        self, mapmeta: MapMeta, target_cell_size: Optional[float] = None
    ):
        """
        Build a grid graph derived from mapmeta but allowing specifying a desired cell_size
        in world units. If target_cell_size is None, fall back to mapmeta.grid_shape (existing behaviour).
        """
        gw, gh = mapmeta.grid_shape
        if target_cell_size is None:
            # legacy mode: use gridshape as-is
            self._build_from_layout(mapmeta.extras.get("layout", None), mapmeta=mapmeta)
            return

        # compute desired number of columns/rows to roughly match world bbox and cell size
        minx, maxx, miny, maxy = mapmeta.bbox
        world_w = maxx - minx if maxx != minx else 1.0
        world_h = maxy - miny if maxy != miny else 1.0
        cols = max(3, int(round(world_w / target_cell_size)))
        rows = max(3, int(round(world_h / target_cell_size)))

        # create a synthetic layout where each cell is computed from downsampled layout
        # if mapmeta.extras contains layout, we will sample it; otherwise fallback to open grid
        layout_src = mapmeta.extras.get("layout") if mapmeta.extras else None
        new_layout = [["." for _ in range(cols)] for _ in range(rows)]
        if layout_src:
            src_h = len(layout_src)
            src_w = len(layout_src[0]) if src_h else 0
            for ry in range(rows):
                for rx in range(cols):
                    # map back to source cell (nearest neighbor sampling)
                    sx = int(rx * src_w / cols)
                    sy = int(ry * src_h / rows)
                    sx = min(src_w - 1, max(0, sx))
                    sy = min(src_h - 1, max(0, sy))
                    new_layout[ry][rx] = layout_src[sy][sx]
        else:
            # leave as all open
            pass

        # finally build from new layout
        self._build_from_layout(new_layout, mapmeta=None)
        # set width/height to cols/rows and positions scale to cell centers in world units
        self.width = cols
        self.height = rows


def attach_mapmeta_to_environment(env: Any, mapmeta: MapMeta) -> None:
    """
    Attach world coordinates (mapmeta.transform) to nodes in an existing environment object `env`.
    This function is intentionally defensive — it tries a few common internal shapes:
      - env.nodes: iterable of node objects
      - env.graph.nodes(data=True) for networkx-like graphs
      - env._nodes or env.nodes_list
    For each node it tries to find grid coordinates in these common attribute/key names:
      ('gx','gy'), ('grid_x','grid_y'), ('x_idx','y_idx'), ('i','j')
    and then sets world coordinates on the node as attributes:
      node.world_x, node.world_y
    or on the node data dict if node is a dict: node['world_x'], node['world_y'].
    """
    if mapmeta is None:
        return

    def _set_world_on_obj(obj, gx, gy):
        wx, wy = mapmeta.transform(int(gx), int(gy))
        # try several attribute names
        try:
            setattr(obj, "world_x", wx)
            setattr(obj, "world_y", wy)
            return True
        except Exception:
            pass
        # fallback attribute variants
        for nx, ny in (("x_world", "y_world"), ("wx", "wy")):
            try:
                setattr(obj, nx, wx)
                setattr(obj, ny, wy)
                return True
            except Exception:
                pass
        # if it's a dict-like object
        try:
            obj["world_x"] = wx
            obj["world_y"] = wy
            return True
        except Exception:
            pass
        return False

    def _get_grid_coords_from_obj(obj):
        # Try attribute names for grid indices
        for ax, ay in (
            ("gx", "gy"),
            ("grid_x", "grid_y"),
            ("x_idx", "y_idx"),
            ("i", "j"),
            ("col", "row"),
        ):
            gx = getattr(obj, ax, None)
            gy = getattr(obj, ay, None)
            if gx is not None and gy is not None:
                return gx, gy
        # Try dict keys
        for kx, ky in (
            ("gx", "gy"),
            ("grid_x", "grid_y"),
            ("x_idx", "y_idx"),
            ("i", "j"),
            ("col", "row"),
        ):
            try:
                if kx in obj and ky in obj:
                    return obj[kx], obj[ky]
            except Exception:
                pass
        return None

    # 1) networkx-style env.graph.nodes(data=True)
    try:
        graph = getattr(env, "graph", None)
        if graph is not None:
            # networkx graph
            try:
                nodes_iter = graph.nodes(data=True)
                for nid, data in nodes_iter:
                    # data may contain grid coords
                    coords = (
                        _get_grid_coords_from_obj(data)
                        or _get_grid_coords_from_obj(data.get("attrs", {}))
                        if isinstance(data, dict)
                        else None
                    )
                    if coords:
                        gx, gy = coords
                        # set on data dict
                        data["world_x"], data["world_y"] = mapmeta.transform(int(gx), int(gy))
                    else:
                        # try to inspect node object if nodes store objects as values
                        # some graphs store objects as data['obj']
                        obj = data.get("obj") if isinstance(data, dict) and "obj" in data else None
                        if obj:
                            coords = _get_grid_coords_from_obj(obj)
                            if coords:
                                gx, gy = coords
                                _set_world_on_obj(obj, gx, gy)
                return
            except Exception:
                # not a networkx graph or unexpected structure — fall through
                pass
    except Exception:
        pass

    # 2) env.nodes as an iterable of node objects or dicts
    nodes_iterable = None
    for candidate in ("nodes", "_nodes", "nodes_list", "node_list"):
        nodes_iterable = getattr(env, candidate, None)
        if nodes_iterable is not None:
            break

    if nodes_iterable is None:
        # 3) maybe env has a method to iterate nodes
        if hasattr(env, "iter_nodes"):
            nodes_iterable = env.iter_nodes()

    if nodes_iterable is None:
        # nothing we can do safely
        return

    # Iterate and assign world coords
    try:
        for node in nodes_iterable:
            coords = _get_grid_coords_from_obj(node)
            if coords:
                gx, gy = coords
                _set_world_on_obj(node, gx, gy)
            else:
                # sometimes node is a (gx,gy,obj) tuple
                try:
                    if isinstance(node, tuple) and len(node) >= 3:
                        maybe_obj = node[2]
                        coords = _get_grid_coords_from_obj(maybe_obj)
                        if coords:
                            gx, gy = coords
                            _set_world_on_obj(maybe_obj, gx, gy)
                except Exception:
                    pass
    except Exception:
        # be tolerant: don't raise on unexpected structures
        return
