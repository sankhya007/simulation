# maps/dxf_loader.py

from typing import List, Tuple

import ezdxf
import math

from config import (
    DXF_GRID_WIDTH,
    DXF_GRID_HEIGHT,
    DXF_WALL_LAYER,
    DXF_EXIT_LAYER,
    DXF_WALL_DISTANCE_THRESHOLD,
    DXF_EXIT_DISTANCE_THRESHOLD,
)


LayoutMatrix = List[List[str]]  # rows of characters: ".", "#", "E"
Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def _collect_segments(doc, layer_name: str) -> List[Segment]:
    """Collect line segments from a given DXF layer."""
    msp = doc.modelspace()
    segments: List[Segment] = []

    for e in msp:
        if e.dxf.layer != layer_name:
            continue

        if e.dxftype() == "LINE":
            p1 = (float(e.dxf.start.x), float(e.dxf.start.y))
            p2 = (float(e.dxf.end.x), float(e.dxf.end.y))
            segments.append((p1, p2))

        elif e.dxftype() in ("LWPOLYLINE", "POLYLINE"):
            points = [(float(p[0]), float(p[1])) for p in e.get_points()]
            for i in range(len(points) - 1):
                segments.append((points[i], points[i + 1]))

    return segments


def _compute_extents(segments: List[Segment]) -> Tuple[float, float, float, float]:
    xs = []
    ys = []
    for (x1, y1), (x2, y2) in segments:
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    if not xs or not ys:
        return 0.0, 1.0, 0.0, 1.0

    return min(xs), max(xs), min(ys), max(ys)


def _point_to_segment_distance(px: float, py: float, seg: Segment) -> float:
    (x1, y1), (x2, y2) = seg
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:
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
      - Walls      : segments on DXF_WALL_LAYER
      - Exit doors : segments on DXF_EXIT_LAYER

    We compute the drawing extents, map them to a fixed grid, and
    mark cells near wall segments as "#" and near exit segments as "E".
    """
    doc = ezdxf.readfile(path)

    wall_segments = _collect_segments(doc, DXF_WALL_LAYER)
    exit_segments = _collect_segments(doc, DXF_EXIT_LAYER)

    # If no walls, we still need something meaningful
    all_segments = wall_segments + exit_segments
    if not all_segments:
        # fallback: simple empty layout
        return [["." for _ in range(DXF_GRID_WIDTH)] for _ in range(DXF_GRID_HEIGHT)]

    min_x, max_x, min_y, max_y = _compute_extents(all_segments)
    width = max_x - min_x
    height = max_y - min_y

    if width == 0:
        width = 1.0
    if height == 0:
        height = 1.0

    # Build grid
    layout: LayoutMatrix = []
    for gy in range(DXF_GRID_HEIGHT):
        row = []
        for gx in range(DXF_GRID_WIDTH):
            # grid cell center in normalized CAD coordinates
            cx = min_x + (gx + 0.5) * (width / DXF_GRID_WIDTH)
            cy = min_y + (gy + 0.5) * (height / DXF_GRID_HEIGHT)

            # check distance to wall segments
            is_wall = False
            for seg in wall_segments:
                if _point_to_segment_distance(cx, cy, seg) <= DXF_WALL_DISTANCE_THRESHOLD:
                    is_wall = True
                    break

            if is_wall:
                row.append("#")
                continue

            # check distance to exit segments
            is_exit = False
            for seg in exit_segments:
                if _point_to_segment_distance(cx, cy, seg) <= DXF_EXIT_DISTANCE_THRESHOLD:
                    is_exit = True
                    break

            if is_exit:
                row.append("E")
            else:
                row.append(".")

        layout.append(row)

    return layout
