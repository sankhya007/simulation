#!/usr/bin/env python3
"""
tools/preview_graph.py
Preview the constructed graph (grid / centerline / hybrid) over the floorplan.
Usage:
    python tools/preview_graph.py --type centerline --cell-size 0.5
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from maps.map_loader import load_mapmeta_from_config
from environment import EnvironmentGraph
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--type", choices=["grid", "centerline", "hybrid"], default="grid")
    p.add_argument("--cell-size", type=float, default=None)
    p.add_argument("--scenario", type=str, default=None)
    args = p.parse_args()

    mm = load_mapmeta_from_config()
    layout = mm.layout
    gw, gh = mm.grid_shape

    env = EnvironmentGraph(
        width=gw,
        height=gh,
        layout_matrix=layout,
        mapmeta=mm,
        graph_type=args.type,
        cell_size=args.cell_size,
    )

    pos = {n: env.get_pos(n) for n in env.graph.nodes()}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    fig, ax = plt.subplots(figsize=(10, 8))

    # optional background image if source path available
    try:
        import matplotlib.image as mpimg

        fp = mm.extras.get("source_path")
        if fp:
            img = mpimg.imread(fp)
            ax.imshow(
                img, origin="upper", extent=(min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1)
            )
    except Exception:
        pass

    lines = [(pos[u], pos[v]) for u, v in env.graph.edges()]
    if lines:
        lc = LineCollection(lines, colors=(0.6, 0.6, 0.6), linewidths=0.6)
        ax.add_collection(lc)

    ax.scatter(xs, ys, s=8, c="red")
    ax.set_title(f"Graph preview: {args.type} (nodes={len(xs)})")
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
