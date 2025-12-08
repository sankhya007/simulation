# maps/raster_loader.py

from typing import List

from PIL import Image

from config import (
    RASTER_DOWNSCALE_FACTOR,
    RASTER_WALL_THRESHOLD,
    RASTER_EXIT_GREEN_MIN,
)

LayoutMatrix = List[List[str]]  # rows of characters: ".", "#", "E"


def load_raster_floorplan_to_layout(path: str) -> LayoutMatrix:
    """
    Load a PNG/JPG floorplan and convert it to a layout matrix.

    Conventions:
      - Walls: dark pixels (low brightness/luma) -> "#"
      - Exits: bright green (G large, R/B small) -> "E"
      - Otherwise: walkable "." (corridor)
    """
    img = Image.open(path).convert("RGB")

    # Downscale large images so the grid isn't huge
    if RASTER_DOWNSCALE_FACTOR > 1:
        w, h = img.size
        img = img.resize(
            (max(1, w // RASTER_DOWNSCALE_FACTOR), max(1, h // RASTER_DOWNSCALE_FACTOR)),
            Image.NEAREST,
        )

    w, h = img.size
    pixels = img.load()

    layout: LayoutMatrix = []

    for y in range(h):
        row: List[str] = []
        for x in range(w):
            r, g, b = pixels[x, y]

            # --- 1) Brightness (luma) based wall detection ---
            # Use standard luma formula: Y = 0.299 R + 0.587 G + 0.114 B
            luma = 0.299 * r + 0.587 * g + 0.114 * b

            # Dark pixel (any dark colour: black, dark grey, dark blue, etc.) -> wall
            if luma <= RASTER_WALL_THRESHOLD:
                row.append("#")
                continue

            # --- 2) Exit detection (bright green-ish) ---
            # Only check exits for non-dark pixels (so exits are not swallowed by the wall test).
            if (
                g >= RASTER_EXIT_GREEN_MIN
                and r < RASTER_EXIT_GREEN_MIN / 2
                and b < RASTER_EXIT_GREEN_MIN / 2
            ):
                row.append("E")
                continue

            # --- 3) Otherwise: walkable corridor ---
            row.append(".")

        layout.append(row)

    return layout
