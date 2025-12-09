# maps/map_meta.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Any, Dict

LayoutMatrix = List[List[str]]
BBox = Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y)
GridShape = Tuple[int, int]               # (width, height)


@dataclass
class MapMeta:
    """
    Stable map metadata object.

    Fields:
    --------
    layout : 2D list of chars
        Grid of cells (rows x cols). Values typically '.', '#', 'E'.

    bbox : (min_x, max_x, min_y, max_y)
        Real-world (CAD) bounding box covered by the raster/grid.

    grid_shape : (grid_width, grid_height)
        Number of grid cells horizontally and vertically.

    transform : callable (gx:int, gy:int) -> (real_x:float, real_y:float)
        Forward mapping from grid coordinates to CAD/world coordinates.
        Always maps to the *center* of a grid cell.

    extras : dict
        Arbitrary loader metadata (doors, layers, thresholds, etc).

    cad_to_grid : optional callable (x:float, y:float) -> (gx:int, gy:int)
        Inverse transform: convert CAD coords → nearest grid cell.

    grid_to_cad : optional callable (gx:int, gy:int) -> (x:float, y:float)
        Alternate forward transform (preferred name for CAD coordinates).
        If not provided, defaults to `transform`.
    """

    layout: List[List[str]]
    bbox: Tuple[float, float, float, float]
    grid_shape: Tuple[int, int]
    transform: Callable[[int, int], Tuple[float, float]]
    extras: Dict[str, Any]

    # Optional CAD mapping fields
    cad_to_grid: Callable[[float, float], Tuple[int, int]] = None
    grid_to_cad: Callable[[int, int], Tuple[float, float]] = None

    def __post_init__(self):
        """
        Ensure grid_to_cad exists (fallback to transform).
        If no cad_to_grid is provided, a linear inverse is generated.
        """
        minx, maxx, miny, maxy = self.bbox
        gw, gh = self.grid_shape

        # Default grid_to_cad = transform
        if self.grid_to_cad is None:
            self.grid_to_cad = self.transform

        # Default cad_to_grid: linear inverse mapping
        if self.cad_to_grid is None:
            cell_w = (maxx - minx) / gw if gw > 0 else 1.0
            cell_h = (maxy - miny) / gh if gh > 0 else 1.0

            def cad_to_grid_default(x: float, y: float) -> Tuple[int, int]:
                gx = int((x - minx) / cell_w)
                gy = int((y - miny) / cell_h)
                return max(0, min(gx, gw - 1)), max(0, min(gy, gh - 1))

            self.cad_to_grid = cad_to_grid_default

    def to_dict(self) -> Dict[str, Any]:
        """Non-callable metadata for debug/logging."""
        return {
            "layout_shape": (len(self.layout[0]) if self.layout else 0, len(self.layout)),
            "bbox": self.bbox,
            "grid_shape": self.grid_shape,
            "extras": self.extras,
            "has_grid_to_cad": self.grid_to_cad is not None,
            "has_cad_to_grid": self.cad_to_grid is not None,
        }

    def inverse_transform(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert CAD/world coordinates → nearest grid coordinate.

        If a custom cad_to_grid exists (DXF loader), it is used.
        Otherwise falls back on linear interpolation from bbox.
        """
        if callable(self.cad_to_grid):
            try:
                return self.cad_to_grid(x, y)
            except Exception:
                pass  # fallback to linear method below

        # fallback linear version
        minx, maxx, miny, maxy = self.bbox
        gw, gh = self.grid_shape

        cell_w = (maxx - minx) / gw
        cell_h = (maxy - miny) / gh

        gx = int((x - minx) / cell_w)
        gy = int((y - miny) / cell_h)

        return max(0, min(gx, gw - 1)), max(0, min(gy, gh - 1))
