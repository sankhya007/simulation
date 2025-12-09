import os
from maps.map_meta import MapMeta
from maps.map_loader import _make_mapmeta_from_dxf
from environment import EnvironmentGraph
from simulation import CrowdSimulation

def test_grid_to_cad_basic():
    meta = {
        "bbox": (0.0, 100.0, 0.0, 200.0),
        "grid_width": 10,
        "grid_height": 20,
    }

    mm = _make_mapmeta_from_dxf(meta)
    gx, gy = 5, 10
    cad_x, cad_y = mm.grid_to_cad(gx, gy)

    assert 0 < cad_x < 100
    assert 0 < cad_y < 200
    assert isinstance(cad_x, float)
    assert isinstance(cad_y, float)

def test_environment_cad_positions():
    meta = {
        "bbox": (0.0, 10.0, 0.0, 10.0),
        "grid_width": 10,
        "grid_height": 10,
    }
    mm = _make_mapmeta_from_dxf(meta)
    layout = [["." for _ in range(10)] for _ in range(10)]

    env = EnvironmentGraph(width=10, height=10, layout_matrix=layout, mapmeta=mm)
    cad = env.get_cad_pos((3, 3))
    assert cad is not None
    cx, cy = cad
    assert isinstance(cx, float)
    assert isinstance(cy, float)

def test_bottleneck_csv_output(tmp_path):
    from main import _save_overlay_and_csv

    # Minimal fake simulation + environment
    class FakeAgent:
        has_exited = False

    class FakeSim:
        def __init__(self):
            self.agents = [FakeAgent()]
            self.time_step = 5

        def to_serializable(self):
            return {"ok": True}

    # build env
    meta = {
        "bbox": (0, 10, 0, 10),
        "grid_width": 10,
        "grid_height": 10,
    }
    mm = _make_mapmeta_from_dxf(meta)
    layout = [["." for _ in range(10)] for _ in range(10)]
    env = EnvironmentGraph(width=10, height=10, layout_matrix=layout, mapmeta=mm)

    out = tmp_path / "test"
    bottlenecks = [(2, 3)]

    _save_overlay_and_csv(FakeSim(), env, bottlenecks, out, "unit")

    csv_file = out / "unit_bottlenecks.csv"
    assert csv_file.exists()

    text = csv_file.read_text()
    assert "2" in text
    assert "3" in text
