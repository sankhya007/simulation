# tests/test_graph_builders.py
from maps.map_loader import load_mapmeta_from_config
from environment import EnvironmentGraph


def test_grid_graph_basic():
    mm = load_mapmeta_from_config()
    env = EnvironmentGraph(
        width=mm.grid_shape[0],
        height=mm.grid_shape[1],
        layout_matrix=mm.layout,
        mapmeta=mm,
        graph_type="grid",
    )
    assert env.graph.number_of_nodes() > 0
    assert env.graph.number_of_edges() > 0


def test_centerline_or_fallback():
    mm = load_mapmeta_from_config()
    try:
        env = EnvironmentGraph(
            width=mm.grid_shape[0],
            height=mm.grid_shape[1],
            layout_matrix=mm.layout,
            mapmeta=mm,
            graph_type="centerline",
        )
    except ImportError:
        # acceptable if scikit-image not installed
        assert True
    else:
        assert env.graph.number_of_nodes() > 0
