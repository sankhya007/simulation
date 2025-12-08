from PIL import Image
from pathlib import Path
import tempfile

import config
from maps.floorplan_image_loader import load_floorplan_image_to_layout


def make_test_image(path: Path, cell_count: int = 5):
    # Create an image sized so that after downscale -> cell_count x cell_count
    factor = max(1, getattr(config, "RASTER_DOWNSCALE_FACTOR", 1))
    width = cell_count * factor
    height = cell_count * factor

    img = Image.new("RGB", (width, height), (255, 255, 255))
    pixels = img.load()

    # draw a black border (walls) at full resolution so they survive downsampling
    for x in range(width):
        for dx in range(factor):  # thicken border so downsampling keeps it
            if 0 + dx < height:
                pixels[x, dx] = (0, 0, 0)
            if height - 1 - dx >= 0:
                pixels[x, height - 1 - dx] = (0, 0, 0)
    for y in range(height):
        for dy in range(factor):
            if 0 + dy < width:
                pixels[dy, y] = (0, 0, 0)
            if width - 1 - dy >= 0:
                pixels[width - 1 - dy, y] = (0, 0, 0)

    # mark a green exit pixel somewhere near center (in high-res coords)
    exit_x = factor * 1 + factor // 2
    exit_y = factor * 2 + factor // 2
    pixels[exit_x, exit_y] = (0, 255, 0)

    img.save(path)


def test_raster_loader_produces_expected_layout(tmp_path):
    p = tmp_path / "test_floor.png"
    make_test_image(p, cell_count=5)
    layout = load_floorplan_image_to_layout(str(p))

    # Expected dims after downscale
    expected_h = 5
    expected_w = 5

    assert len(layout) == expected_h
    assert all(len(row) == expected_w for row in layout)

    # border cells should be walls '#'
    assert layout[0][0] == "#"
    assert layout[-1][-1] == "#"

    # exit should be detected somewhere (we placed at cell (1,2) in high-res)
    # after downscale, we expect that the cell (1,2) is 'E'
    assert layout[2][1] == "E"
