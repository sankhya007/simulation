# tools/dxf_overlay_preview.py
"""
Create a PNG overlay that shows:
 - CAD geometry (lines) from the DXF (drawn in light gray)
 - Grid cell centers marked as wall '#' (red) or exit 'E' (green)
 - Saves <dxfname>.overlay.png next to the DXF
Usage:
    python tools/dxf_overlay_preview.py maps/examples/call_center_pt2.dxf
"""

import sys
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Ensure repo root is on sys.path so local imports like `maps.dxf_loader` work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ezdxf

# local loader (now importable)
from maps.dxf_loader import load_dxf_floorplan_with_meta

# colors
CAD_COLOR = (180, 180, 180)  # light gray for CAD lines
WALL_COLOR = (255, 0, 0)  # red for walls
EXIT_COLOR = (0, 192, 0)  # green for exits
BG_COLOR = (255, 255, 255)


def cad_to_image_coords(x, y, bbox, img_w, img_h):
    """
    Map CAD coords (x,y) to image pixel coords (px,py).
    bbox = (min_x, max_x, min_y, max_y)
    image origin at top-left; CAD y assumed same orientation as DXF (y increasing up).
    We'll flip y so CAD y increases upwards visually maps correctly.
    """
    min_x, max_x, min_y, max_y = bbox
    cad_w = max_x - min_x
    cad_h = max_y - min_y
    if cad_w == 0:
        cad_w = 1.0
    if cad_h == 0:
        cad_h = 1.0
    px = int(round((x - min_x) / cad_w * (img_w - 1)))
    py = int(round((max_y - y) / cad_h * (img_h - 1)))
    return px, py


def draw_dxf_geometry(draw, doc, bbox, img_w, img_h):
    msp = doc.modelspace()
    for e in msp:
        t = e.dxftype()
        try:
            if t == "LINE":
                x1, y1 = float(e.dxf.start.x), float(e.dxf.start.y)
                x2, y2 = float(e.dxf.end.x), float(e.dxf.end.y)
                p1 = cad_to_image_coords(x1, y1, bbox, img_w, img_h)
                p2 = cad_to_image_coords(x2, y2, bbox, img_w, img_h)
                draw.line([p1, p2], fill=CAD_COLOR, width=1)
            elif t in ("LWPOLYLINE", "POLYLINE"):
                try:
                    pts = [(float(p[0]), float(p[1])) for p in e.get_points()]
                except Exception:
                    continue
                if len(pts) >= 2:
                    pix = [cad_to_image_coords(x, y, bbox, img_w, img_h) for (x, y) in pts]
                    draw.line(pix, fill=CAD_COLOR, width=1)
        except Exception:
            continue


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/dxf_overlay_preview.py path/to/file.dxf")
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print("File not found:", p)
        sys.exit(1)

    # load layout + metadata from your loader
    layout, meta = load_dxf_floorplan_with_meta(str(p))
    bbox = meta["bbox"]  # min_x, max_x, min_y, max_y
    grid_h = meta["grid_height"]
    grid_w = meta["grid_width"]

    # choose pixels_per_cell so output is reasonably large
    ppc = 6
    img_w = grid_w * ppc
    img_h = grid_h * ppc

    im = Image.new("RGB", (img_w, img_h), BG_COLOR)
    draw = ImageDraw.Draw(im)

    # draw CAD geometry from DXF (light gray)
    try:
        doc = ezdxf.readfile(str(p))
        draw_dxf_geometry(draw, doc, bbox, img_w, img_h)
    except Exception as e:
        print("Warning: failed to draw DXF geometry:", e)

    # overlay grid cell centers
    for gy in range(grid_h):
        for gx in range(grid_w):
            val = layout[gy][gx]
            # center CAD coords of this cell
            min_x, max_x, min_y, max_y = bbox
            cad_x = min_x + (gx + 0.5) * ((max_x - min_x) / grid_w)
            cad_y = min_y + (gy + 0.5) * ((max_y - min_y) / grid_h)
            px, py = cad_to_image_coords(cad_x, cad_y, bbox, img_w, img_h)
            r = max(1, ppc // 2)
            if val == "#":
                draw.rectangle([px - r, py - r, px + r, py + r], fill=WALL_COLOR)
            elif val == "E":
                draw.rectangle([px - r, py - r, px + r, py + r], fill=EXIT_COLOR)

    out = p.with_suffix(".overlay.png")
    im.save(out)
    print("Saved overlay:", out)


if __name__ == "__main__":
    main()
