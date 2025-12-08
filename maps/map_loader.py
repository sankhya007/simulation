# maps/map_loader.py

from typing import Sequence

from config import MAP_MODE, MAP_FILE, GRID_WIDTH, GRID_HEIGHT
from .raster_loader import load_raster_floorplan_to_layout

LayoutMatrix = Sequence[Sequence[str]]


def load_layout_matrix_from_config() -> LayoutMatrix | None:
    """
    Return a layout_matrix based on MAP_MODE and MAP_FILE.

    - If MAP_MODE == "synthetic": returns None (use simple grid).
    - If MAP_MODE == "raster":   returns layout from PNG/JPG.
    - If MAP_MODE == "dxf":      returns layout from DXF.

    EnvironmentGraph can then be constructed as:

        layout = load_layout_matrix_from_config()
        if layout is None:
            env = EnvironmentGraph(GRID_WIDTH, GRID_HEIGHT)
        else:
            env = EnvironmentGraph(width=0, height=0, layout_matrix=layout)
    """
    mode = MAP_MODE.lower()

    if mode == "synthetic":
        # Use simple rectangular grid
        return None

    if mode == "raster":
        if not MAP_FILE:
            raise ValueError("MAP_MODE='raster' but MAP_FILE is not set.")
        return load_raster_floorplan_to_layout(MAP_FILE)

    if mode == "dxf":
        if not MAP_FILE:
            raise ValueError("MAP_MODE='dxf' but MAP_FILE is not set.")
        # Lazy import so DXF loader (and DXF_* config constants) are only required
        # when you actually use DXF mode.
        from .dxf_loader import load_dxf_floorplan_to_layout
        return load_dxf_floorplan_to_layout(MAP_FILE)

    raise ValueError(f"Unknown MAP_MODE '{MAP_MODE}'. Expected 'synthetic', 'raster', or 'dxf'.")
