# tools/preview_raster.py
"""
Preview raster floorplan -> layout as an upscaled PNG.
Pure-Pillow version (no cv2). Ensures project root is on sys.path so imports like
`from maps.floorplan_image_loader import ...` work when running this script directly.
"""

import sys
import os
from pathlib import Path

# --- ensure repo root is on sys.path so local packages import correctly ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
from maps.floorplan_image_loader import load_floorplan_image_to_layout

# Color map for visualization
COLOR_W = (0, 0, 0)  # walls -> black
COLOR_E = (0, 255, 0)  # exits -> green
COLOR_D = (255, 255, 255)  # walkable -> white


def layout_to_image(layout, scale=8):
    """
    Convert LayoutMatrix (list of lists of '.', '#', 'E')
    into a Pillow RGB image upscaled by 'scale'.
    """
    h = len(layout)
    w = len(layout[0]) if h else 0

    # Create base image (small scale 1:1)
    img = Image.new("RGB", (w, h))

    px = img.load()
    for y in range(h):
        for x in range(w):
            if layout[y][x] == "#":
                px[x, y] = COLOR_W
            elif layout[y][x] == "E":
                px[x, y] = COLOR_E
            else:
                px[x, y] = COLOR_D

    # Upscale for readability
    img_big = img.resize((w * scale, h * scale), Image.NEAREST)
    return img_big


def preview(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    print(f"[INFO] Loading floorplan: {p}")
    layout = load_floorplan_image_to_layout(str(p))

    print("[INFO] Converting layout → preview image")
    img_big = layout_to_image(layout, scale=8)

    # Save next to original file
    out_path = p.with_suffix(".layout_preview.png")
    img_big.save(out_path)
    print(f"[OK] Saved preview: {out_path}")

    # Optionally show it with default OS viewer
    print("[INFO] Opening preview...")
    img_big.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python tools/preview_raster.py path/to/floorplan.png\n")
        sys.exit(1)
    preview(sys.argv[1])
