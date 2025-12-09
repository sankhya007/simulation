# tests/test_dxf_loader.py
import ezdxf
from pathlib import Path

import config
from maps.dxf_loader import load_dxf_floorplan_to_layout


def make_test_dxf(path: Path):
    """
    Create a DXF with:
      - a rectangular boundary polyline on the wall layer (to set extents)
      - many short horizontal 'wall' segments inside the rectangle on the wall layer
      - a short exit segment on the exit layer
    This dense set of short segments ensures at least some grid cell centers
    are within the distance threshold used by the loader.
    """
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    wall_layer = config.DXF_WALL_LAYERS[0] if config.DXF_WALL_LAYERS else "WALL"
    exit_layer = config.DXF_EXIT_LAYERS[0] if config.DXF_EXIT_LAYERS else wall_layer

    # Rectangular boundary to set extents (non-degenerate)
    rect = [(0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0), (0.0, 0.0)]
    msp.add_lwpolyline(rect, dxfattribs={"layer": wall_layer})

    # Add many short "wall" segments inside the rectangle (dense sampling).
    # Choose a grid spacing that will map to multiple grid cells.
    for y in range(5, 56, 5):
        for x in range(5, 96, 5):
            # short horizontal segment of length 3 units
            msp.add_line(
                (float(x), float(y)), (float(x + 3), float(y)), dxfattribs={"layer": wall_layer}
            )

    # Add a short exit segment near the top edge on exit layer
    msp.add_line((50.0, 60.0), (60.0, 60.0), dxfattribs={"layer": exit_layer})

    doc.saveas(str(path))


def test_dxf_loader_generates_layout(tmp_path):
    p = tmp_path / "test.dxf"
    make_test_dxf(p)
    layout = load_dxf_floorplan_to_layout(str(p))

    # should be rectangular grid (DXF_GRID_WIDTH x DXF_GRID_HEIGHT)
    assert len(layout) == config.DXF_GRID_HEIGHT
    assert all(len(row) == config.DXF_GRID_WIDTH for row in layout)

    # at least one wall cell '#' must exist (dense interior segments should create walls)
    found_wall = any("#" in row for row in layout)
    assert found_wall
