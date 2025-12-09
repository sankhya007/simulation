# tools/dxf_layer_mapper.py
"""
Interactive DXF layer mapper.
Usage:
  python tools/dxf_layer_mapper.py path/to/floorplan.dxf

This script lists layers in the DXF and allows the user to assign them to WALL / DOOR / EXIT.
It writes a JSON mapping file next to the DXF named <dxfname>.layers.json.
"""

import sys
import json
from pathlib import Path
import ezdxf


def prompt_multiple(options, prompt_text):
    print(prompt_text)
    print("Enter comma-separated indices (e.g. 0,2,3) or leave blank to skip.")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    raw = input("Your selection: ").strip()
    if not raw:
        return []
    inds = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < len(options):
                inds.append(options[idx])
    return inds


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/dxf_layer_mapper.py path/to/file.dxf")
        return
    p = Path(sys.argv[1])
    doc = ezdxf.readfile(str(p))
    layers = sorted(list({getattr(e.dxf, "layer", "") for e in doc.modelspace()}))
    print(f"Found {len(layers)} layers in {p.name}")
    walls = prompt_multiple(layers, "Select layers to treat as WALL (common: WALL,WALLS):")
    doors = prompt_multiple(layers, "Select layers to treat as DOOR (optional):")
    exits = prompt_multiple(layers, "Select layers to treat as EXIT (optional):")

    mapping = {"WALL": walls, "DOOR": doors, "EXIT": exits}
    out = p.with_suffix(p.suffix + ".layers.json")
    with open(out, "w", encoding="utf8") as fh:
        json.dump(mapping, fh, indent=2)
    print(f"Wrote mapping to {out}")


if __name__ == "__main__":
    main()
