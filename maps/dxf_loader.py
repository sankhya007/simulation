# maps/dxf_loader.py

from typing import List, Tuple, Sequence

import ezdxf
import math

from config import (
    DXF_GRID_WIDTH,
    DXF_GRID_HEIGHT,
    DXF_WALL_LAYERS,
    DXF_EXIT_LAYERS,
    DXF_WALL_DISTANCE_THRESHOLD,
    DXF_EXIT_DISTANCE_THRESHOLD,
)

LayoutMatrix = List[List[str]]  # rows of characters: ".", "#", "E"
Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def _collect_segments(doc, layer_names: Sequence[str]) -> List[Segment]:
    """
    Collect line segments from a given list of DXF layers.
    Supports LINE and (LW)POLYLINE.
    """
    msp = doc.modelspace()
    wanted = set(layer_names)
    segments: List[Segment] = []

    for e in msp:
        # Only keep entities on the layers we care about
        if e.dxf.layer not in wanted:
            continue

        if e.dxftype() == "LINE":
            p1 = (float(e.dxf.start.x), float(e.dxf.start.y))
            p2 = (float(e.dxf.end.x), float(e.dxf.end.y))
            segments.append((p1, p2))

        elif e.dxftype() in ("LWPOLYLINE", "POLYLINE"):
            try:
                pts = [(float(p[0]), float(p[1])) for p in e.get_points()]
            except TypeError:
                # Some POLYLINE types (e.g. 3D) may not support get_points()
                continue

            for i in range(len(pts) - 1):
                segments.append((pts[i], pts[i + 1]))

    return segments


def _compute_extents(segments: List[Segment]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for (x1, y1), (x2, y2) in segments:
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    if not xs or not ys:
        # fallback extents
        return 0.0, 1.0, 0.0, 1.0

    return min(xs), max(xs), min(ys), max(ys)


def _point_to_segment_distance(px: float, py: float, seg: Segment) -> float:
    """Euclidean distance from point (px, py) to a line segment."""
    (x1, y1), (x2, y2) = seg
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def load_dxf_floorplan_to_layout(path: str) -> LayoutMatrix:
    """
    Load a DXF floorplan and convert it to a layout matrix.

    Conventions:
      - Walls      : segments on layers in DXF_WALL_LAYERS
      - Exit doors : segments on layers in DXF_EXIT_LAYERS

    We compute the drawing extents, map them to a fixed grid, and
    mark cells near wall segments as "#" and near exit segments as "E".
    Everything else is walkable ".".
    """
    doc = ezdxf.readfile(path)

    wall_segments = _collect_segments(doc, DXF_WALL_LAYERS)
    exit_segments = _collect_segments(doc, DXF_EXIT_LAYERS)

    # If no segments at all, return empty open layout
    all_segments = wall_segments + exit_segments
    if not all_segments:
        return [["." for _ in range(DXF_GRID_WIDTH)] for _ in range(DXF_GRID_HEIGHT)]

    min_x, max_x, min_y, max_y = _compute_extents(all_segments)
    width = max_x - min_x
    height = max_y - min_y

    if width == 0:
        width = 1.0
    if height == 0:
        height = 1.0

    # --- derive thresholds relative to cell size ---
    cell_w = width / DXF_GRID_WIDTH
    cell_h = height / DXF_GRID_HEIGHT
    cell_size = min(cell_w, cell_h)

    # DXF_*_DISTANCE_THRESHOLD are factors, e.g. 0.4 * cell_size
    wall_thresh = DXF_WALL_DISTANCE_THRESHOLD * cell_size
    exit_thresh = DXF_EXIT_DISTANCE_THRESHOLD * cell_size

    # --- build grid ---
    layout: LayoutMatrix = []
    for gy in range(DXF_GRID_HEIGHT):
        row: List[str] = []
        for gx in range(DXF_GRID_WIDTH):
            # grid cell center in CAD coordinates
            cx = min_x + (gx + 0.5) * (width / DXF_GRID_WIDTH)
            cy = min_y + (gy + 0.5) * (height / DXF_GRID_HEIGHT)

            # Check proximity to wall segments
            is_wall = False
            for seg in wall_segments:
                if _point_to_segment_distance(cx, cy, seg) <= wall_thresh:
                    is_wall = True
                    break

            if is_wall:
                row.append("#")
                continue

            # Check proximity to exit segments
            is_exit = False
            for seg in exit_segments:
                if _point_to_segment_distance(cx, cy, seg) <= exit_thresh:
                    is_exit = True
                    break

            if is_exit:
                row.append("E")
            else:
                row.append(".")

        layout.append(row)

    return layout
