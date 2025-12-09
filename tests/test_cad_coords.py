# tests/test_cad_coords.py
import math
from typing import Tuple

import pytest

from maps.map_loader import load_mapmeta_from_config
from environment import EnvironmentGraph


def _is_callable(obj) -> bool:
    return callable(obj)


def _close(a: Tuple[float, float], b: Tuple[float, float], rel_tol=1e-6, abs_tol=1e-6) -> bool:
    return math.isclose(a[0], b[0], rel_tol=rel_tol, abs_tol=abs_tol) and math.isclose(a[1], b[1], rel_tol=rel_tol, abs_tol=abs_tol)


def _find_accessible_node(env):
    for n, data in env.graph.nodes(data=True):
        if data.get("accessibility") in ("open", "exit"):
            return n, data
    return None, None


def test_nodes_have_cad_pos_or_grid_to_cad_transform():
    """
    Ensure that either:
      - nodes have 'cad_pos' stored in node data, OR
      - MapMeta.extras provides a callable 'grid_to_cad_transform' and node cad_pos
        (if present) matches that transform.
    The test is skipped when no CAD/grid mapping is available (e.g. purely raster projects
    with no DXF mapping).
    """
    mm = load_mapmeta_from_config()
    # prefer grid graph for deterministic node indices
    gw, gh = mm.grid_shape

    env = EnvironmentGraph(width=gw, height=gh, layout_matrix=mm.layout, mapmeta=mm, graph_type="grid")

    # Check whether MapMeta provided a grid->CAD transform
    grid_to_cad = None
    try:
        extras = getattr(mm, "extras", {}) or {}
        grid_to_cad = extras.get("grid_to_cad_transform") or extras.get("grid_to_cad") or extras.get("grid_to_real")
    except Exception:
        grid_to_cad = None

    # Find an accessible node to sample
    node, node_data = _find_accessible_node(env)
    assert node is not None, "No accessible node found in the environment (test environment may be empty)."

    # If neither grid->cad transform is available nor nodes include cad_pos, skip the test.
    node_has_cad = "cad_pos" in node_data or "cad" in node_data or "cad_coords" in node_data

    if not _is_callable(grid_to_cad) and not node_has_cad:
        pytest.skip("No grid->CAD transform and no per-node cad_pos present — skipping CAD coord fidelity test.")

    # If MapMeta provides callable, verify it returns reasonable values and match node cad_pos if present
    if _is_callable(grid_to_cad):
        gx, gy = node
        try:
            expected = grid_to_cad(int(gx), int(gy))
        except Exception as e:
            pytest.fail(f"grid_to_cad_transform callable raised an exception for node {(gx,gy)}: {e}")

        assert isinstance(expected, (tuple, list)) and len(expected) == 2, (
            "grid_to_cad_transform must return a 2-tuple (x,y) for CAD coordinates."
        )

        # If node_data contains cad_pos, ensure it matches the callable result
        if node_has_cad:
            # try possible key names
            cad_val = node_data.get("cad_pos") or node_data.get("cad") or node_data.get("cad_coords")
            assert cad_val is not None, "Node reports cad field key but value is None"
            assert isinstance(cad_val, (tuple, list)) and len(cad_val) == 2, "node cad value must be a 2-tuple (x,y)"
            assert _close((float(cad_val[0]), float(cad_val[1])), (float(expected[0]), float(expected[1])), rel_tol=1e-5, abs_tol=1e-5), (
                f"Node cad_pos {cad_val} does not match grid_to_cad_transform result {expected} (node {node})."
            )
        else:
            # Node did not carry cad_pos but transform exists — we at least ensure transform returns numbers
            assert isinstance(expected[0], (int, float)) and isinstance(expected[1], (int, float)), (
                "grid_to_cad_transform must return numeric coordinates."
            )

    else:
        # No callable, but node data must include cad coordinates
        cad_val = node_data.get("cad_pos") or node_data.get("cad") or node_data.get("cad_coords")
        assert cad_val is not None, "No cad mapping and node lacking cad_pos."
        assert isinstance(cad_val, (tuple, list)) and len(cad_val) == 2, "node cad value must be a 2-tuple (x,y)"
        assert isinstance(cad_val[0], (int, float)) and isinstance(cad_val[1], (int, float)), "cad coordinates must be numeric."
