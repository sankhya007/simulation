# maps/dxf_loader.py
"""
Enhanced DXF -> layout loader + metadata.

Features:
 - Accepts configurable layer lists from config (WALL / DOOR / EXIT candidates)
 - Collects LINE, LWPOLYLINE, POLYLINE and simple INSERT (block ref) geometries
 - Performs lightweight endpoint snapping + segment merging to reduce duplicate/overlapping segments
 - Wall buffering heuristic (wall width in CAD units) applied at rasterization time
 - Door detection: explicit DOOR layers OR short gaps between wall segments under a threshold
 - Returns (layout_matrix, metadata) via load_dxf_floorplan_with_meta
 - Backwards-compatible helper load_dxf_floorplan_to_layout(layout_only) available

Metadata dictionary fields:
 - bbox: (min_x, max_x, min_y, max_y) in CAD coords
 - grid_width, grid_height: integers used for rasterization
 - cell_size: CAD units per grid cell (min of width/grid_width, height/grid_height)
 - cad_to_grid: tuple (min_x, min_y, scale_x, scale_y) to map CAD -> grid
 - layers_found: list of layer names present in the DXF
 - walls_layer_candidates / doors_layer_candidates / exits_layer_candidates: lists from config or mapping file
 - wall_buffer: wall buffer used in CAD units (float)
"""

from typing import List, Tuple, Sequence, Dict, Any, Optional
from maps.map_meta import MapMeta
import math
import json
from pathlib import Path

import ezdxf
import numpy as np

from config import (
    DXF_GRID_WIDTH,
    DXF_GRID_HEIGHT,
    DXF_WALL_LAYERS,
    DXF_EXIT_LAYERS,
    DXF_DOOR_LAYERS,
    DXF_WALL_DISTANCE_THRESHOLD,
    DXF_EXIT_DISTANCE_THRESHOLD,
    DXF_WALL_BUFFER,  # new optional config (CAD units)
    DXF_ENDPOINT_SNAP_DISTANCE,  # new optional config for merging endpoints
    DXF_DOOR_GAP_THRESHOLD,  # new optional config (CAD units)
)


LayoutMatrix = List[List[str]]  # rows of chars: ".", "#", "E"
Point = Tuple[float, float]
Segment = Tuple[Point, Point]


# ------------------------
# Utilities (collection / geometry)
# ------------------------
def _collect_segments(doc, layer_names: Sequence[str]) -> List[Segment]:
    """
    Collect line segments from a given list of DXF layers.
    Supports LINE and (LW)POLYLINE and POLYLINE and simple INSERT(block) geometries.
    """
    msp = doc.modelspace()
    wanted = set(layer_names) if layer_names else set()
    segments: List[Segment] = []

    for e in msp:
        layer = getattr(e.dxf, "layer", None)
        if layer is None:
            continue
        # If layer list is empty, collect nothing (caller can pass None to collect all)
        if wanted and layer not in wanted:
            continue

        t = e.dxftype()
        try:
            if t == "LINE":
                p1 = (float(e.dxf.start.x), float(e.dxf.start.y))
                p2 = (float(e.dxf.end.x), float(e.dxf.end.y))
                segments.append((p1, p2))

            elif t in ("LWPOLYLINE", "POLYLINE"):
                try:
                    pts = [(float(pt[0]), float(pt[1])) for pt in e.get_points()]
                except Exception:
                    # fallback for some POLYLINE types
                    continue
                for i in range(len(pts) - 1):
                    segments.append((pts[i], pts[i + 1]))

            elif t == "INSERT":
                # block insert: try to extract geometry of the block's entities where possible
                try:
                    blk = doc.blocks[e.dxf.name]
                    ins_x = float(e.dxf.insert.x)
                    ins_y = float(e.dxf.insert.y)
                    sx = float(getattr(e.dxf, "xscale", 1.0))
                    sy = float(getattr(e.dxf, "yscale", 1.0))
                    ang = float(getattr(e.dxf, "rotation", 0.0))
                    # naive: iterate block entities and translate their LINEs / POLYLINEs
                    for be in blk:
                        bt = be.dxftype()
                        if bt == "LINE":
                            p1 = (ins_x + be.dxf.start.x * sx, ins_y + be.dxf.start.y * sy)
                            p2 = (ins_x + be.dxf.end.x * sx, ins_y + be.dxf.end.y * sy)
                            segments.append((p1, p2))
                        elif bt in ("LWPOLYLINE", "POLYLINE"):
                            try:
                                pts = [
                                    (ins_x + float(p[0]) * sx, ins_y + float(p[1]) * sy)
                                    for p in be.get_points()
                                ]
                                for i in range(len(pts) - 1):
                                    segments.append((pts[i], pts[i + 1]))
                            except Exception:
                                continue
                except Exception:
                    # ignore complex/inaccessible blocks
                    continue

            # arcs and circles are not directly added as segments here (could be approximated)
        except Exception:
            # be robust to odd/evolving DXF types
            continue

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
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


# ------------------------
# Lightweight merging: snap endpoints that are very close, join connected collinear segments
# ------------------------
def _snap_and_merge_segments(segments: List[Segment], snap_tol: float) -> List[Segment]:
    """
    Snap endpoints within snap_tol and merge contiguous segments.
    This is a pragmatic, dependency-free merging (not a full topology simplifier).
    """
    if not segments or snap_tol <= 0:
        return segments[:]

    # Build list of endpoints and cluster them by proximity
    pts = []
    for a, b in segments:
        pts.append(a)
        pts.append(b)
    pts = np.array(pts, dtype=float)

    # cluster by simple greedy union-find on near neighbors (O(n^2) but okay for moderate DXFs)
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            if (abs(xi - xj) <= snap_tol) and (abs(yi - yj) <= snap_tol):
                if math.hypot(xi - xj, yi - yj) <= snap_tol:
                    union(i, j)

    # representative point for each cluster: average of cluster
    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    rep_map = {}
    for r, inds in clusters.items():
        xs = pts[inds, 0].mean()
        ys = pts[inds, 1].mean()
        for i in inds:
            rep_map[tuple(pts[i])] = (float(xs), float(ys))

    # rebuild segments with snapped endpoints
    snapped = []
    it = iter(pts.reshape(-1, 2))
    for a, b in segments:
        aa = rep_map.get(tuple((float(a[0]), float(a[1]))), a)
        bb = rep_map.get(tuple((float(b[0]), float(b[1]))), b)
        if aa != bb:
            snapped.append((aa, bb))

    # merge contiguous collinear segments: group by sorted endpoints and attempt to connect
    # Build adjacency dict
    adj = {}
    for a, b in snapped:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    visited = set()
    merged = []

    def is_almost_collinear(a, b, c, tol=1e-6):
        # checks if vectors AB and BC are collinear
        (ax, ay), (bx, by), (cx, cy) = a, b, c
        ux, uy = bx - ax, by - ay
        vx, vy = cx - bx, cy - by
        cross = abs(ux * vy - uy * vx)
        return cross <= tol

    for node in adj.keys():
        if node in visited:
            continue
        # perform path walks from node while degree==2 and collinear
        if len(adj[node]) != 2:
            # try to follow each outgoing edge as a separate chain
            for nb in adj.get(node, []):
                if (node, nb) in visited or (nb, node) in visited:
                    continue
                chain = [node, nb]
                visited.add((node, nb))
                visited.add((nb, node))
                cur = nb
                prev = node
                while True:
                    nbrs = [x for x in adj.get(cur, []) if x != prev]
                    if len(nbrs) != 1:
                        break
                    nxt = nbrs[0]
                    if not is_almost_collinear(prev, cur, nxt):
                        break
                    chain.append(nxt)
                    prev, cur = cur, nxt
                # add merged segment from chain[0] to chain[-1]
                if chain[0] != chain[-1]:
                    merged.append((chain[0], chain[-1]))
        else:
            # interior node with degree 2; skip here, will be visited from endpoints
            continue

    # If no merged segments produced (e.g., closed loops), fall back to snapped list
    if not merged:
        return snapped
    return merged


# ------------------------
# Door detection (heuristic)
# ------------------------
def _detect_doors_from_wall_gaps(segments: List[Segment], gap_threshold: float) -> List[Point]:
    """
    Simple heuristic: find near-collinear segment endpoints pairs whose gap is < gap_threshold.
    Return list of candidate door center points in CAD coordinates.
    """
    doors = []
    # Build endpoint list
    endpoints = []
    for a, b in segments:
        endpoints.append(a)
        endpoints.append(b)
    # naive pairwise check: if two endpoints are close and approximately collinear with neighbor segments, treat as gap
    n = len(endpoints)
    for i in range(n):
        x1, y1 = endpoints[i]
        for j in range(i + 1, n):
            x2, y2 = endpoints[j]
            d = math.hypot(x1 - x2, y1 - y2)
            if 0 < d <= gap_threshold:
                # middle point
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                doors.append((mx, my))
    return doors


# ------------------------
# Main loader: returns layout + metadata
# ------------------------
def load_dxf_floorplan_with_meta(path: str) -> Tuple[LayoutMatrix, Dict[str, Any]]:
    """
    Load DXF and return (layout_matrix, metadata).

    metadata: dict described at top of file.
    """
    doc = ezdxf.readfile(path)

    # Gather a conservative list of layer candidates from config (fall back to config constants)
    wall_layers = DXF_WALL_LAYERS or []
    exit_layers = DXF_EXIT_LAYERS or []
    door_layers = DXF_DOOR_LAYERS or []

    # If user provided mapping file exists near DXF (same base name + .layers.json), load it
    p = Path(path)
    mapping_file = p.with_suffix(p.suffix + ".layers.json")
    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf8") as fh:
                mapping = json.load(fh)
            # mapping expected keys: "WALL", "DOOR", "EXIT" -> list of layer names
            wall_layers = mapping.get("WALL", wall_layers)
            door_layers = mapping.get("DOOR", door_layers)
            exit_layers = mapping.get("EXIT", exit_layers)
        except Exception:
            pass

    # collect segments
    wall_segments = _collect_segments(doc, wall_layers)
    exit_segments = _collect_segments(doc, exit_layers)
    door_segments = _collect_segments(doc, door_layers)

    # If wall_segments empty, fallback to collecting ALL segments so user still gets something
    if not wall_segments:
        # collect everything and treat by layer name heuristics later
        all_segments = _collect_segments(
            doc, []
        )  # empty list means collect nothing, so we need to collect all layers manually
        # alternative: iterate modelspace and collect by default behavior (we reuse previous collector but with wanted==empty we skip)
        # Simpler: manual collection of all lines/polylines
        msp = doc.modelspace()
        all_seg = []
        for e in msp:
            try:
                t = e.dxftype()
                if t == "LINE":
                    all_seg.append(
                        (
                            (float(e.dxf.start.x), float(e.dxf.start.y)),
                            (float(e.dxf.end.x), float(e.dxf.end.y)),
                        )
                    )
                elif t in ("LWPOLYLINE", "POLYLINE"):
                    try:
                        pts = [(float(p[0]), float(p[1])) for p in e.get_points()]
                    except Exception:
                        continue
                    for i in range(len(pts) - 1):
                        all_seg.append((pts[i], pts[i + 1]))
            except Exception:
                continue
        wall_segments = all_seg

    # Merge / snap segments to remove tiny duplicates
    snap_tol = float(
        getattr(__import__("config"), "DXF_ENDPOINT_SNAP_DISTANCE", DXF_ENDPOINT_SNAP_DISTANCE)
    )
    merged_wall_segments = _snap_and_merge_segments(wall_segments, snap_tol)

    # Door candidates: from explicit door layers OR heuristic gaps
    door_centers = []
    if door_segments:
        # mark midpoints of door segments
        for a, b in door_segments:
            door_centers.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
    else:
        gap_thresh = float(
            getattr(__import__("config"), "DXF_DOOR_GAP_THRESHOLD", DXF_DOOR_GAP_THRESHOLD)
        )
        door_centers = _detect_doors_from_wall_gaps(merged_wall_segments, gap_thresh)

    # Compute extents & transform to grid
    all_segments_for_extents = merged_wall_segments + exit_segments + door_segments
    if not all_segments_for_extents:
        min_x, max_x, min_y, max_y = 0.0, 1.0, 0.0, 1.0
    else:
        min_x, max_x, min_y, max_y = _compute_extents(all_segments_for_extents)

    width = max_x - min_x
    height = max_y - min_y
    if width == 0:
        width = 1.0
    if height == 0:
        height = 1.0

    grid_w = int(DXF_GRID_WIDTH)
    grid_h = int(DXF_GRID_HEIGHT)

    cell_w = width / grid_w
    cell_h = height / grid_h
    cell_size = min(cell_w, cell_h)

    # thresholds in CAD units
    wall_thresh = (
        float(
            getattr(
                __import__("config"), "DXF_WALL_DISTANCE_THRESHOLD", DXF_WALL_DISTANCE_THRESHOLD
            )
        )
        * cell_size
    )
    exit_thresh = (
        float(
            getattr(
                __import__("config"), "DXF_EXIT_DISTANCE_THRESHOLD", DXF_EXIT_DISTANCE_THRESHOLD
            )
        )
        * cell_size
    )
    wall_buffer = float(getattr(__import__("config"), "DXF_WALL_BUFFER", DXF_WALL_BUFFER))

    # Build layout grid (row major: y from top->bottom as earlier convention: row 0 = top)
    layout: LayoutMatrix = []
    for gy in range(grid_h):
        row: List[str] = []
        for gx in range(grid_w):
            # compute center in CAD coords
            cx = min_x + (gx + 0.5) * (width / grid_w)
            cy = min_y + (gy + 0.5) * (height / grid_h)

            # Check proximity to wall segments (with optional wall_buffer)
            is_wall = False
            for seg in merged_wall_segments:
                dist = _point_to_segment_distance(cx, cy, seg)
                # consider wall buffer: if provided, add it to threshold
                effective_thresh = wall_thresh + (wall_buffer if wall_buffer else 0.0)
                if dist <= effective_thresh:
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

    # Metadata
    metadata: Dict[str, Any] = {
        "bbox": (min_x, max_x, min_y, max_y),
        "grid_width": grid_w,
        "grid_height": grid_h,
        "cell_size": cell_size,
        "cad_to_grid": {
            "min_x": min_x,
            "min_y": min_y,
            "scale_x": grid_w / width,
            "scale_y": grid_h / height,
        },
        "layers_found": list(set([getattr(e.dxf, "layer", "") for e in doc.modelspace()])),
        "walls_layer_candidates": wall_layers,
        "doors_layer_candidates": door_layers,
        "exits_layer_candidates": exit_layers,
        "wall_buffer": wall_buffer,
        "door_centers": door_centers,
    }

    return layout, metadata


# Backwards-compatible helper (returns layout only)
def load_dxf_floorplan_to_layout(path: str) -> LayoutMatrix:
    layout, _meta = load_dxf_floorplan_with_meta(path)
    return layout


def load_dxf_floorplan_mapmeta(path: str) -> MapMeta:
    """
    New API: return MapMeta object.
    Adapts existing (layout, meta) DXF loader output into a MapMeta with full
    CAD↔grid mapping (grid_to_cad + cad_to_grid + transform).
    """
    layout, meta = load_dxf_floorplan_with_meta(path)

    # --- Extract bbox & grid dims ---
    bbox = meta.get("bbox", (0.0, 1.0, 0.0, 1.0))
    min_x, max_x, min_y, max_y = bbox
    grid_w = int(meta.get("grid_width", meta.get("grid_w", 110)))
    grid_h = int(meta.get("grid_height", meta.get("grid_h", 70)))

    # world span
    width = (max_x - min_x) or 1.0
    height = (max_y - min_y) or 1.0

    # --- Cell size in CAD space ---
    cell_w = width / grid_w
    cell_h = height / grid_h

    # ---------------------------------------------
    # 1) Build grid→CAD mapping (official transform)
    # ---------------------------------------------
    def grid_to_cad(gx: int, gy: int):
        return (
            float(min_x + (gx + 0.5) * cell_w),
            float(min_y + (gy + 0.5) * cell_h),
        )

    # ---------------------------------------------
    # 2) Build CAD→grid mapping (reverse transform)
    # ---------------------------------------------
    def cad_to_grid(x: float, y: float):
        gx = int((x - min_x) / cell_w)
        gy = int((y - min_y) / cell_h)
        return gx, gy

    # Primary MapMeta.transform = grid→CAD
    transform = grid_to_cad

    # --- extras: keep all metadata + stable mapping names ---
    extras = dict(meta)
    extras["grid_to_cad"] = grid_to_cad
    extras["cad_to_grid"] = cad_to_grid

    # --- Build MapMeta object ---
    return MapMeta(
        layout=layout,
        bbox=bbox,
        grid_shape=(grid_w, grid_h),
        transform=transform,
        extras=extras,
        grid_to_cad=grid_to_cad,
        cad_to_grid=cad_to_grid,
    )



# Backwards-compatible helper: old API still works
def load_dxf_floorplan_to_layout(path: str):
    """Legacy API kept for compatibility: returns layout only."""
    mm = load_dxf_floorplan_mapmeta(path)
    return mm.layout
