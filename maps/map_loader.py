# maps/map_loader.py

from typing import Sequence

from config import MAP_MODE, MAP_FILE, GRID_WIDTH, GRID_HEIGHT
from .floorplan_image_loader import load_floorplan_image_to_layout

LayoutMatrix = Sequence[Sequence[str]]


def load_layout_matrix_from_config() -> LayoutMatrix | None:
    """
    Return a layout_matrix based on MAP_MODE and MAP_FILE.

    - If MAP_MODE == "grid":    returns None (use simple grid).
    - If MAP_MODE == "raster":  returns layout from PNG/JPG.
    - If MAP_MODE == "dxf":     returns layout from DXF.
    """
    mode = MAP_MODE.lower()

    if mode == "grid":
        # Use simple rectangular grid
        return None

    if mode == "raster":
        if not MAP_FILE:
            raise ValueError("MAP_MODE='raster' but MAP_FILE is not set.")
        return load_floorplan_image_to_layout(MAP_FILE)

    if mode == "dxf":
        if not MAP_FILE:
            raise ValueError("MAP_MODE='dxf' but MAP_FILE is not set.")
        from .dxf_loader import load_dxf_floorplan_to_layout  # lazy import
        return load_dxf_floorplan_to_layout(MAP_FILE)

    raise ValueError(f"Unknown MAP_MODE '{MAP_MODE}'. Expected 'grid', 'raster', or 'dxf'.")
