# maps/map_meta.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Any, Dict


LayoutMatrix = List[List[str]]
BBox = Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y)
GridShape = Tuple[int, int]  # (width, height)


@dataclass
class MapMeta:
    """
    Stable map metadata object.

    - layout: grid of chars (rows x cols) where each char is '.', '#', 'E'
    - bbox: (min_x, max_x, min_y, max_y) in CAD/world coordinates
    - grid_shape: (grid_width, grid_height) integers
    - transform: callable (gx:int, gy:int) -> (real_x:float, real_y:float)
    - extras: dict for arbitrary loader metadata (door_centers, layers, etc.)
    """
    layout: LayoutMatrix
    bbox: BBox
    grid_shape: GridShape
    transform: Callable[[int, int], Tuple[float, float]]
    extras: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Non-callable-friendly dict view (transform omitted)."""
        return {
            "layout_shape": (len(self.layout[0]) if self.layout else 0, len(self.layout)),
            "bbox": self.bbox,
            "grid_shape": self.grid_shape,
            "extras": self.extras,
        }
