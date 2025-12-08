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
      - Walls: near-black pixels (R,G,B all < RASTER_WALL_THRESHOLD) -> "#"
      - Exits: bright green (G >= RASTER_EXIT_GREEN_MIN, R,B low) -> "E"
      - Otherwise: walkable "." (corridor)
    """
    img = Image.open(path).convert("RGB")

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

            # Check wall (black / very dark)
            if r < RASTER_WALL_THRESHOLD and g < RASTER_WALL_THRESHOLD and b < RASTER_WALL_THRESHOLD:
                row.append("#")
                continue

            # Check exit (bright green-ish)
            if (
                g >= RASTER_EXIT_GREEN_MIN
                and r < RASTER_EXIT_GREEN_MIN // 2
                and b < RASTER_EXIT_GREEN_MIN // 2
            ):
                row.append("E")
                continue

            # Otherwise, walkable corridor
            row.append(".")
        layout.append(row)

    return layout
