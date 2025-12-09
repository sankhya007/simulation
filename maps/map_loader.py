# maps/map_loader.py
"""
Central map loader that returns a standardized MapMeta object for the rest
of the pipeline.

Usage:
    from maps.map_loader import load_mapmeta_from_config
    mm = load_mapmeta_from_config()
    layout = mm.layout
    env = EnvironmentGraph(width=mm.grid_shape[0], height=mm.grid_shape[1],
                           layout_matrix=layout, mapmeta=mm)
"""

from typing import Callable, Dict, List, Optional, Tuple, Any
from pathlib import Path
import pprint

import config

# Try to import the project's MapMeta dataclass / class
try:
    from maps.map_meta import MapMeta
except Exception:
    # Minimal fallback MapMeta (if your map_meta.py is missing temporarily)
    from dataclasses import dataclass

    @dataclass
    class MapMeta:
        layout: List[List[str]]
        bbox: Tuple[float, float, float, float]
        grid_shape: Tuple[int, int]
        transform: Callable[[int, int], Tuple[float, float]]
        extras: Dict[str, Any]


# Try to import optional loaders (raster & dxf)
try:
    from maps.floorplan_image_loader import load_floorplan_image_to_layout
except Exception:
    load_floorplan_image_to_layout = None

try:
    # prefer loader that returns layout + meta if available
    from maps.dxf_loader import load_dxf_floorplan_with_meta  # returns (layout, meta)
except Exception:
    load_dxf_floorplan_with_meta = None

# fallback dxf loader that returns only layout
try:
    from maps.dxf_loader import load_dxf_floorplan_to_layout
except Exception:
    load_dxf_floorplan_to_layout = None


def _identity_transform(gx: int, gy: int) -> Tuple[float, float]:
    """Simple identity mapping (grid coords -> same world coords)."""
    return float(gx), float(gy)


def _grid_cell_center_transform_factory(
    cell_origin_x: float, cell_origin_y: float, cell_w: float, cell_h: float
):
    """
    Returns a transform(gx,gy) that maps grid indices to world coordinates using
    the provided origin and cell sizes. By default it maps to the cell center:
        world_x = cell_origin_x + (gx + 0.5) * cell_w
        world_y = cell_origin_y + (gy + 0.5) * cell_h
    """

    def transform(gx: int, gy: int) -> Tuple[float, float]:
        return (
            float(cell_origin_x + (gx + 0.5) * cell_w),
            float(cell_origin_y + (gy + 0.5) * cell_h),
        )

    return transform


def _make_mapmeta_from_grid(layout: List[List[str]], source_path: Optional[str] = None) -> MapMeta:
    """
    Construct MapMeta for a simple grid layout (layout is list-of-rows).
    World bbox will be in "grid units": minx=0, maxx=width, miny=0, maxy=height.
    Transform maps grid cell centers to world coords (gx->x = gx+0.5).
    """
    if not layout:
        raise ValueError("Empty layout provided to grid MapMeta builder.")

    gh = len(layout)
    gw = len(layout[0])
    for r in layout:
        if len(r) != gw:
            raise ValueError("All rows in layout must have same length")

    minx, maxx = 0.0, float(gw)
    miny, maxy = 0.0, float(gh)
    cell_w, cell_h = 1.0, 1.0
    transform = _grid_cell_center_transform_factory(minx, miny, cell_w, cell_h)

    extras: Dict[str, Any] = {}
    extras["layout"] = layout
    extras["source_path"] = source_path

    mm = MapMeta(
        layout=layout,
        bbox=(minx, maxx, miny, maxy),
        grid_shape=(gw, gh),
        transform=transform,
        extras=extras,
    )
    return mm


def _make_mapmeta_from_raster(
    layout: List[List[str]], source_path: Optional[str] = None
) -> MapMeta:
    """
    For raster-based loader we assume each grid cell corresponds to a pixel cell.
    The bbox (world coords) will use pixel coordinates. Transform maps cell indices
    to pixel-centers (x = gx + 0.5).
    """
    if not layout:
        raise ValueError("Empty layout provided to raster MapMeta builder.")

    gh = len(layout)
    gw = len(layout[0])
    for r in layout:
        if len(r) != gw:
            raise ValueError("All rows in layout must have same length")

    # world bbox: x in [0, gw], y in [0, gh]
    minx, maxx = 0.0, float(gw)
    miny, maxy = 0.0, float(gh)
    cell_w, cell_h = 1.0, 1.0
    transform = _grid_cell_center_transform_factory(minx, miny, cell_w, cell_h)

    extras: Dict[str, Any] = {}
    extras["layout"] = layout
    extras["source_path"] = source_path

    mm = MapMeta(
        layout=layout,
        bbox=(minx, maxx, miny, maxy),
        grid_shape=(gw, gh),
        transform=transform,
        extras=extras,
    )
    return mm


def _make_mapmeta_from_dxf(
    meta: Dict[str, Any],
    layout: Optional[List[List[str]]] = None,
    source_path: Optional[str] = None,
) -> MapMeta:
    """
    Build MapMeta from DXF loader metadata. The loader (load_dxf_floorplan_with_meta)
    is expected to return a dict-like meta containing:
      - 'bbox' : (minx, maxx, miny, maxy)
      - 'grid_width', 'grid_height' or 'grid_shape'
      - 'cell_size' (optional)
      - 'transform' or 'cad_to_grid' (optional) — if transform callable present, use it
    If layout is missing, we will create a minimal empty layout based on reported grid dims.
    """
    extras = dict(meta) if isinstance(meta, dict) else {}
    # prefer explicit keys
    bbox = extras.get("bbox")
    gw = extras.get("grid_width") or (
        extras.get("grid_shape")[0] if extras.get("grid_shape") else None
    )
    gh = extras.get("grid_height") or (
        extras.get("grid_shape")[1] if extras.get("grid_shape") else None
    )

    # fallback: if meta uses 'grid_shape'
    if gw is None or gh is None:
        gs = extras.get("grid_shape")
        if gs and isinstance(gs, (tuple, list)) and len(gs) == 2:
            gw, gh = int(gs[0]), int(gs[1])

    if gw is None or gh is None:
        # as ultimate fallback, use config DXF_GRID_* if present
        try:
            gw = int(config.DXF_GRID_WIDTH)
            gh = int(config.DXF_GRID_HEIGHT)
        except Exception:
            raise ValueError("DXF meta lacks grid dimensions and no config fallback available")

    # Build layout if not provided
    if layout is None:
        layout = [["." for _ in range(gw)] for _ in range(gh)]

    # bbox fallback to grid extents if not present
    if bbox is None:
        minx, maxx = 0.0, float(gw)
        miny, maxy = 0.0, float(gh)
    else:
        minx, maxx, miny, maxy = bbox

    # If the loader provided a callable transform use it, otherwise construct one mapping grid to bbox
    transform_callable = extras.get("transform") or extras.get("cad_to_grid")
    if callable(transform_callable):
        transform = transform_callable
    else:
        # compute cell sizes from bbox
        world_w = maxx - minx if maxx != minx else float(gw)
        world_h = maxy - miny if maxy != miny else float(gh)
        cell_w = world_w / gw
        cell_h = world_h / gh
        transform = _grid_cell_center_transform_factory(minx, miny, cell_w, cell_h)

    # Ensure extras contain helpful values and source path
    extras["layout"] = layout
    extras["source_path"] = source_path

    mm = MapMeta(
        layout=layout,
        bbox=(minx, maxx, miny, maxy),
        grid_shape=(int(gw), int(gh)),
        transform=transform,
        extras=extras,
    )
    return mm


# -------------------------
# Public loader entrypoints
# -------------------------
def load_mapmeta(map_mode: str, map_file: Optional[str] = None) -> MapMeta:
    """
    Load a map and return MapMeta object.

    - map_mode: "grid" | "raster" | "dxf"
    - map_file: path to raster image or dxf file (optional for grid)
    """
    mode = (map_mode or config.MAP_MODE or "raster").lower()
    source_path = map_file or getattr(config, "MAP_FILE", None)
    if source_path:
        source_path = str(source_path)

    print(f"[map_loader] Loading map (mode={mode}) source={source_path}")

    if mode == "grid":
        # Create a simple grid layout from config sizes
        gw = getattr(config, "GRID_WIDTH", None) or 15
        gh = getattr(config, "GRID_HEIGHT", None) or 10
        layout = [["." for _ in range(gw)] for _ in range(gh)]
        mm = _make_mapmeta_from_grid(layout, source_path)
        print(f"[map_loader] built grid MapMeta: grid_shape={mm.grid_shape} bbox={mm.bbox}")
        return mm

    if mode == "raster":
        if load_floorplan_image_to_layout is None:
            raise RuntimeError("Raster map requested but floorplan_image_loader is not available.")
        if not source_path:
            raise ValueError(
                "RASTER map requires MAP_FILE (image path) in config or passed map_file"
            )
        # loader returns layout (list of rows) — we assume it uses config for thresholds
        print(f"[map_loader] calling raster loader for {source_path}")
        layout = load_floorplan_image_to_layout(source_path)
        mm = _make_mapmeta_from_raster(layout, source_path)
        print(f"[map_loader] raster MapMeta built: grid_shape={mm.grid_shape} bbox={mm.bbox}")
        return mm

    if mode == "dxf":
        if load_dxf_floorplan_with_meta is not None:
            if not source_path:
                raise ValueError("DXF map requires MAP_FILE path or map_file argument")
            print(f"[map_loader] calling dxf loader (with meta) for {source_path}")
            layout, meta = load_dxf_floorplan_with_meta(source_path)
            # meta expected to be dict-like
            mm = _make_mapmeta_from_dxf(meta, layout=layout, source_path=source_path)
            print(f"[map_loader] dxf MapMeta built: grid_shape={mm.grid_shape} bbox={mm.bbox}")
            return mm
        elif load_dxf_floorplan_to_layout is not None:
            if not source_path:
                raise ValueError("DXF map requires MAP_FILE path or map_file argument")
            print(f"[map_loader] calling dxf loader (layout-only) for {source_path}")
            layout = load_dxf_floorplan_to_layout(source_path)
            # best-effort construct MapMeta using config DXF grid params
            try:
                gw = int(config.DXF_GRID_WIDTH)
                gh = int(config.DXF_GRID_HEIGHT)
            except Exception:
                gw = len(layout[0]) if layout else 110
                gh = len(layout) if layout else 70
            mm = _make_mapmeta_from_dxf(
                {"grid_width": gw, "grid_height": gh}, layout=layout, source_path=source_path
            )
            print(
                f"[map_loader] dxf(layout-only) MapMeta built: grid_shape={mm.grid_shape} bbox={mm.bbox}"
            )
            return mm
        else:
            raise RuntimeError(
                "DXF map requested but no DXF loader is available (install ezdxf and ensure maps.dxf_loader exists)"
            )

    raise ValueError(f"Unknown map mode: {mode}")


def load_mapmeta_from_config() -> MapMeta:
    """
    Convenience helper: uses values from config.py to load the configured map.
    """
    return load_mapmeta(
        map_mode=getattr(config, "MAP_MODE", "raster"), map_file=getattr(config, "MAP_FILE", None)
    )


# -------------------------
# Backwards-compatible helpers
# -------------------------
def load_layout_matrix_from_config() -> list:
    """
    Backwards-compat shim: returns the layout matrix only (old API).
    New code should prefer load_mapmeta_from_config() which returns MapMeta.
    """
    mm = load_mapmeta_from_config()
    return mm.layout
