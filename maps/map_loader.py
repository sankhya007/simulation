# maps/map_loader.py

from typing import Sequence

from maps.map_meta import MapMeta
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

def load_mapmeta_from_config() -> MapMeta:
    """
    Centralized loader that returns MapMeta regardless of MAP_MODE.
    """
    import config
    mode = getattr(config, "MAP_MODE", "raster")
    path = getattr(config, "MAP_FILE", None)

    if mode == "grid":
        # build simple empty grid
        w = getattr(config, "GRID_WIDTH", 15)
        h = getattr(config, "GRID_HEIGHT", 10)
        layout = [["." for _ in range(w)] for _ in range(h)]
        bbox = (0.0, float(w), 0.0, float(h))
        def transform(gx, gy):
            return bbox[0] + (gx + 0.5) * ((bbox[1] - bbox[0]) / w), bbox[2] + (gy + 0.5) * ((bbox[3] - bbox[2]) / h)
        return MapMeta(layout=layout, bbox=bbox, grid_shape=(w, h), transform=transform, extras={"mode":"grid"})

    elif mode == "raster":
        from maps.floorplan_image_loader import load_floorplan_image_to_mapmeta
        return load_floorplan_image_to_mapmeta(path)

    elif mode == "dxf":
        from maps.dxf_loader import load_dxf_floorplan_mapmeta
        return load_dxf_floorplan_mapmeta(path)

    else:
        raise ValueError(f"Unknown MAP_MODE: {mode}")