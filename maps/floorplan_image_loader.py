# maps/floorplan_image_loader.py
"""
Stable pure-Pillow + NumPy raster floorplan -> layout converter.

Simple, robust implementation using:
 - Otsu global thresholding (pure NumPy)
 - Small-component removal via flood-fill labeling (pure Python + NumPy)
 - RGB-based exit detection (simple heuristic)

Output: LayoutMatrix = List[List[str]]
  '.' = walkable
  '#' = wall
  'E' = exit
"""

from typing import List
from pathlib import Path
from maps.map_meta import MapMeta
from PIL import Image
import numpy as np

import config

LayoutMatrix = List[List[str]]


def _rgb_to_luma(img_arr: np.ndarray) -> np.ndarray:
    r = img_arr[..., 0].astype(np.float32)
    g = img_arr[..., 1].astype(np.float32)
    b = img_arr[..., 2].astype(np.float32)
    luma = (0.299 * r + 0.587 * g + 0.114 * b).round().astype(np.uint8)
    return luma


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    numerator = (mu_t * omega - mu) ** 2
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1.0
    sigma_b_squared = numerator / denom
    thresh = int(np.nanargmax(sigma_b_squared))
    return thresh


def _remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    h, w = binary.shape
    labels = -np.ones((h, w), dtype=np.int32)
    label_id = 0
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    stack = []
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or labels[y, x] != -1:
                continue
            labels[y, x] = label_id
            stack.append((y, x))
            comp_pixels = [(y, x)]
            while stack:
                yy, xx = stack.pop()
                for ny, nx in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == -1:
                        labels[ny, nx] = label_id
                        stack.append((ny, nx))
                        comp_pixels.append((ny, nx))
            area = len(comp_pixels)
            if area >= min_area:
                for (py, px) in comp_pixels:
                    cleaned[py, px] = 1
            label_id += 1
    return cleaned


def _detect_exit_by_rgb(img_arr: np.ndarray) -> np.ndarray:
    r = img_arr[..., 0].astype(np.int32)
    g = img_arr[..., 1].astype(np.int32)
    b = img_arr[..., 2].astype(np.int32)

    green_min = getattr(config, "RASTER_EXIT_GREEN_MIN", 200)
    factor = 1.2
    mask = (g >= green_min) & (g >= (r * factor)) & (g >= (b * factor))
    return mask.astype(np.uint8)


def load_floorplan_image_to_layout(path: str) -> LayoutMatrix:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    pil = Image.open(p).convert("RGB")
    ds = max(1, int(getattr(config, "RASTER_DOWNSCALE_FACTOR", 1)))
    if ds > 1:
        w0, h0 = pil.size
        pil = pil.resize((max(1, w0 // ds), max(1, h0 // ds)), Image.NEAREST)
    img_arr = np.asarray(pil)
    h, w = img_arr.shape[:2]

    gray = _rgb_to_luma(img_arr)

    # --- Otsu threshold (dark pixels -> walls) ---
    otsu_t = _otsu_threshold(gray)
    wall_mask = (gray <= otsu_t).astype(np.uint8)

    # Remove small noise
    min_area = max(1, int(getattr(config, "RASTER_MIN_WALL_AREA", 3)))
    if min_area > 1:
        wall_mask = _remove_small_components(wall_mask, min_area)

    # Exit detection by RGB
    exit_mask = _detect_exit_by_rgb(img_arr) if getattr(config, "RASTER_USE_COLOR_SEGMENTATION", True) else np.zeros((h, w), dtype=np.uint8)

    # Build layout matrix
    layout: LayoutMatrix = []
    for y in range(h):
        row = []
        for x in range(w):
            if exit_mask[y, x]:
                row.append("E")
            elif wall_mask[y, x]:
                row.append("#")
            else:
                row.append(".")
        layout.append(row)

    return layout

def load_floorplan_image_to_mapmeta(path: str) -> MapMeta:
    """
    New API: return MapMeta object for raster floorplans.
    Keeps legacy function load_floorplan_image_to_layout for backward compatibility.
    """
    layout = load_floorplan_image_to_layout(path)  # existing function
    # derive bbox from image: simple unit bbox where one grid cell = 1.0 units
    grid_h = len(layout)
    grid_w = len(layout[0]) if grid_h else 0
    # Map world bbox to 0..grid_w and 0..grid_h (units arbitrary; for CAD mapping this is coarse)
    bbox = (0.0, float(grid_w), 0.0, float(grid_h))

    def transform(gx: int, gy: int):
        # center of cell in "image-space" units
        real_x = bbox[0] + (gx + 0.5) * ((bbox[1] - bbox[0]) / grid_w) if grid_w else 0.0
        real_y = bbox[2] + (gy + 0.5) * ((bbox[3] - bbox[2]) / grid_h) if grid_h else 0.0
        return real_x, real_y

    extras = {
        "source_path": path,
        "downscale": getattr(__import__("config"), "RASTER_DOWNSCALE_FACTOR", None),
    }

    return MapMeta(layout=layout, bbox=bbox, grid_shape=(grid_w, grid_h), transform=transform, extras=extras)


# legacy API preserved
def load_floorplan_image_to_layout_legacy(path: str):
    return load_floorplan_image_to_layout(path)