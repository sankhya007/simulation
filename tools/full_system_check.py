import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from maps.map_loader import load_mapmeta_from_config
from environment import EnvironmentGraph
from simulation import CrowdSimulation


def run_short_sim(env, steps=200, agents=80):
    sim = CrowdSimulation(env, agents)
    for _ in range(steps):
        sim.step()
    exited = sum(1 for a in sim.agents if getattr(a, "has_exited", False))
    return exited, getattr(sim, "time_step", steps)


def test_graph_type(mm, graph_type, cell_size=None):
    print(f"\n=== Testing graph_type = {graph_type}, cell_size={cell_size} ===")
    env = EnvironmentGraph(
        width=mm.grid_shape[0],
        height=mm.grid_shape[1],
        layout_matrix=mm.layout,
        mapmeta=mm,
        graph_type=graph_type,
        cell_size=cell_size,
    )

    print(f"Nodes: {env.graph.number_of_nodes()}  Edges: {env.graph.number_of_edges()}")

    start = time.time()
    exited, steps = run_short_sim(env, steps=200, agents=80)
    dt = time.time() - start

    print(f"Exited: {exited}/80   Steps: {steps}   Runtime: {dt:.2f}s")


def test_mapmeta(mm):
    print("\n=== MapMeta ===")
    print("Grid shape:", mm.grid_shape)
    print("BBox:", mm.bbox)
    print("Layout size:", len(mm.layout), "x", len(mm.layout[0]))
    print("Extras:", list(mm.extras.keys()))
    print("Transform(0,0) ->", mm.transform(0, 0))


def main():
    print("==============================================")
    print("        FULL SYSTEM CHECK — BEGIN             ")
    print("==============================================")

    mm = load_mapmeta_from_config()
    test_mapmeta(mm)

    # Test GRID
    test_graph_type(mm, "grid")

    # Test CENTERLINE
    test_graph_type(mm, "centerline")

    # Test HYBRID (coarse)
    test_graph_type(mm, "hybrid", cell_size=1.0)

    # Test HYBRID (medium)
    test_graph_type(mm, "hybrid", cell_size=0.6)

    # Test HYBRID (fine)
    test_graph_type(mm, "hybrid", cell_size=0.3)

    print("\n==============================================")
    print("        FULL SYSTEM CHECK — COMPLETE          ")
    print("==============================================")


if __name__ == "__main__":
    main()
