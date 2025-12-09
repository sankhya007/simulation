# demos/compare_motion_models.py
"""
Run three short simulations (graph, social_force, rvo) and save PNG snapshots.
This script prepends the project root to sys.path so project imports like
`import config` work when invoked as `python demos/compare_motion_models.py`.
"""

from pathlib import Path
import sys
import os
import random

# --- ensure repo root is on sys.path so imports like `import config` work ---
# file is at <repo>/demos/compare_motion_models.py -> repo root is parent of parent? parent is demos, parent of that is repo root
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent  # one level up from demos/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# now safe to import project modules
import config
from maps.map_loader import load_mapmeta_from_config
from environment import EnvironmentGraph
from simulation import CrowdSimulation

import matplotlib.pyplot as plt

OUT_DIR = REPO_ROOT / "demos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_and_snapshot(model_name: str, agents=60, steps=80, seed=123):
    print(f"[demo] running {model_name} agents={agents} steps={steps}")
    # override config for this run
    config.MOTION_MODEL = model_name
    random.seed(seed)

    mm = load_mapmeta_from_config()
    # use grid graph to make snapshots deterministic
    env = EnvironmentGraph(width=mm.grid_shape[0], height=mm.grid_shape[1], layout_matrix=mm.layout, mapmeta=mm, graph_type="grid")
    sim = CrowdSimulation(env, num_agents=agents)

    for t in range(steps):
        sim.step()

    xs = []
    ys = []
    for a in sim.agents:
        x, y = a.get_position()
        xs.append(x)
        ys.append(y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, s=8)
    ax.set_title(f"Model: {model_name}  (after {steps} steps)")
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    out = OUT_DIR / f"compare_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] saved {out}")


if __name__ == "__main__":
    run_and_snapshot("graph", agents=60, steps=120, seed=42)
    run_and_snapshot("social_force", agents=60, steps=120, seed=42)
    run_and_snapshot("rvo", agents=60, steps=120, seed=42)
    print("[demo] complete — check the demos/ folder for compare_*.png")
