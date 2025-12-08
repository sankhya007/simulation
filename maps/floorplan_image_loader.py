# maps/floorplan_image_loader.py

from typing import List

from PIL import Image

from config import (
    RASTER_DOWNSCALE_FACTOR,
    RASTER_WALL_THRESHOLD,
    RASTER_EXIT_GREEN_MIN,
)

LayoutMatrix = List[List[str]]  # rows of characters: ".", "#", "E"


def _downscale_image(img: Image.Image, factor: int) -> Image.Image:
    """Optionally downscale the image so the grid is manageable."""
    if factor <= 1:
        return img
    w, h = img.size
    new_size = (max(1, w // factor), max(1, h // factor))
    return img.resize(new_size, Image.NEAREST)


def load_floorplan_image_to_layout(path: str) -> LayoutMatrix:
    """
    Load a PNG/JPG floorplan and convert it into a layout matrix.

    Conventions (must match your drawing):
      - Walls/obstacles  : dark/black pixels
      - Walkable area    : light/white pixels
      - Exits/doors      : bright green (or strong green channel)

    Output is a list of list of chars:
      '.' = walkable
      '#' = wall
      'E' = exit
    """
    img = Image.open(path).convert("RGB")
    img = _downscale_image(img, RASTER_DOWNSCALE_FACTOR)

    width, height = img.size
    pixels = img.load()

    layout: LayoutMatrix = []

    # Note: row 0 = top of image. If you want origin at bottom-left,
    # you can flip the rows at the end with layout[::-1].
    for y in range(height):
        row: List[str] = []
        for x in range(width):
            r, g, b = pixels[x, y]

            # brightness for wall detection
            max_rgb = max(r, g, b)

            # --- classify pixel ---
            # 1) wall: very dark
            if max_rgb <= RASTER_WALL_THRESHOLD:
                row.append("#")
                continue

            # 2) exit: strong green channel (and relatively bright)
            if g >= RASTER_EXIT_GREEN_MIN and g >= r and g >= b:
                row.append("E")
                continue

            # 3) everything else: walkable
            row.append(".")

        layout.append(row)

    return layout
