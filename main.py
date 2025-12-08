# main.py
"""
Main runner for the crowd simulation project.
Features:
 - CLI to run single visual simulation (existing behaviour)
 - Batch runner to run multiple trials (parallel) and aggregate heatmaps / bottlenecks
 - Overlay aggregated heatmap onto the raster image when available
"""

import argparse
import os
import multiprocessing as mp
from typing import Dict, Any, Tuple
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from maps import load_layout_matrix_from_config
from environment import EnvironmentGraph
from simulation import CrowdSimulation
import config


# -----------------------
# Small helpers
# -----------------------
def make_env_from_layout(layout):
    """Return an EnvironmentGraph built from layout or from grid-sized config."""
    if layout is None:
        return EnvironmentGraph(config.GRID_WIDTH, config.GRID_HEIGHT)
    else:
        # keep width=0,height=0 so EnvironmentGraph uses layout_matrix constructor
        return EnvironmentGraph(width=0, height=0, layout_matrix=layout)


def compute_evacuated_fraction(sim: CrowdSimulation) -> float:
    """
    Try to compute fraction evacuated. We attempt a few common attributes.
    Returns fraction in [0,1].
    """
    agents = getattr(sim, "agents", None)
    if agents is None:
        return 0.0

    evacuated = 0
    for a in agents:
        # common attributes: exit_time_step, has_exited, evacuated, reached_exit
        if getattr(a, "exit_time_step", None) is not None:
            evacuated += 1
        elif getattr(a, "has_exited", False):
            evacuated += 1
        elif getattr(a, "evacuated", False):
            evacuated += 1
        elif getattr(a, "reached_exit", False):
            evacuated += 1
    return evacuated / max(1, len(agents))


def get_density_matrix_safe(sim: CrowdSimulation) -> np.ndarray:
    """
    Return sim.get_density_matrix() as numpy array.
    If the simulation class stores density differently, try to adapt.
    """
    if hasattr(sim, "get_density_matrix"):
        m = sim.get_density_matrix()
        return np.array(m)
    # fallback: attempt attribute 'density_matrix'
    dm = getattr(sim, "density_matrix", None)
    if dm is not None:
        return np.array(dm)
    # else return zeros sized from env
    return np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH), dtype=float)


def find_top_k_cells(density: np.ndarray, k: int = 10) -> list:
    """Return list of (row, col, value) sorted descending by value."""
    flat = density.ravel()
    if flat.size == 0:
        return []
    idx = np.argsort(flat)[-k:][::-1]
    rows, cols = np.unravel_index(idx, density.shape)
    return [(int(r), int(c), float(density[r, c])) for r, c in zip(rows, cols)]


# -----------------------
# Single-trial runner (must be picklable top-level for multiprocessing)
# -----------------------
def _run_single_trial(args: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    args is (trial_id, params)
    params keys: num_agents, max_steps, target_percent, map_mode_override (optional)
    Returns a dict containing:
      - 'trial_id'
      - 'density' : 2D numpy array (cumulative node visits)
      - 'steps' : steps executed
      - 'evacuated_fraction'
      - 'top_nodes' : list of top-k nodes (r,c,value)
    """
    trial_id, params = args
    try:
        # Load layout each trial to avoid shared state issues
        layout = load_layout_matrix_from_config()
        env = make_env_from_layout(layout)

        num_agents = params.get("num_agents", config.NUM_AGENTS)
        max_steps = params.get("max_steps", config.MAX_STEPS)
        target_percent = params.get("target_percent", 1.0)  # fraction (0..1)
        top_k = params.get("top_k", 10)

        sim = CrowdSimulation(env, num_agents)

        # step until either max_steps or target fraction evacuated
        for step in range(max_steps):
            sim.step()
            frac = compute_evacuated_fraction(sim)
            if frac >= target_percent:
                break

        density = get_density_matrix_safe(sim)
        top_nodes = find_top_k_cells(density, k=top_k)

        return {
            "trial_id": trial_id,
            "density": density,
            "steps": getattr(sim, "time_step", step + 1),
            "evacuated_fraction": frac,
            "top_nodes": top_nodes,
        }
    except Exception as e:
        # Return the exception info in result so parent can handle gracefully
        return {"trial_id": trial_id, "error": str(e)}


# -----------------------
# Batch runner
# -----------------------
def run_batch(
    n_trials: int = 5,
    num_agents: int = None,
    max_steps: int = None,
    target_percent: float = 1.0,
    parallel_workers: int = None,
    top_k: int = 10,
):
    """
    Run many trials (parallel) and aggregate results.
    Returns an aggregate dict.
    """

    if num_agents is None:
        num_agents = config.NUM_AGENTS
    if max_steps is None:
        max_steps = config.MAX_STEPS
    if parallel_workers is None:
        parallel_workers = max(1, mp.cpu_count() - 1)

    params = {
        "num_agents": num_agents,
        "max_steps": max_steps,
        "target_percent": target_percent,
        "top_k": top_k,
    }

    tasks = [(i + 1, params) for i in range(n_trials)]

    print(f"Running {n_trials} trials with {num_agents} agents, max_steps={max_steps}, "
          f"target_percent={target_percent*100:.0f}%, workers={parallel_workers}")

    t0 = time.time()
    with mp.Pool(parallel_workers) as pool:
        results = pool.map(_run_single_trial, tasks)
    t1 = time.time()
    print(f"Completed in {t1 - t0:.1f}s")

    # collect successful results
    succ = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]

    if not succ:
        print("All trials failed. First errors:")
        for e in errs[:5]:
            print(e)
        return {"success": False, "errors": errs}

    # Determine density grid size from first success
    base_density = succ[0]["density"]
    agg_density = np.zeros_like(base_density, dtype=float)

    for r in succ:
        d = r["density"]
        # if shape mismatch, attempt resize (simple fallback)
        if d.shape != agg_density.shape:
            # try to resize using numpy (simple pad or crop)
            new = np.zeros_like(agg_density)
            rr = min(new.shape[0], d.shape[0])
            cc = min(new.shape[1], d.shape[1])
            new[:rr, :cc] = d[:rr, :cc]
            d = new
        agg_density += d

    # average over runs
    mean_density = agg_density / len(succ)

    # Aggregate top node votes
    node_votes = {}
    for r in succ:
        for (row, col, val) in r["top_nodes"]:
            node_votes[(row, col)] = node_votes.get((row, col), 0) + 1

    # Sort nodes by votes desc and by mean_density
    voted_nodes = sorted(
        [(pos, votes, mean_density[pos]) for pos, votes in node_votes.items()],
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )

    # Basic statistics
    evacuated_fracs = [r["evacuated_fraction"] for r in succ]
    steps = [r["steps"] for r in succ]

    report = {
        "success": True,
        "n_trials": n_trials,
        "completed_trials": len(succ),
        "errors": errs,
        "mean_density": mean_density,
        "voted_nodes": voted_nodes,
        "evacuated_fracs": evacuated_fracs,
        "steps": steps,
    }

    # Save outputs
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "mean_density.npy"), mean_density)
    print(f"Saved mean density to {os.path.join(out_dir, 'mean_density.npy')}")

    # Create overlay if raster map present
    try:
        layout = load_layout_matrix_from_config()
        env = make_env_from_layout(layout)
        if config.MAP_MODE.lower() == "raster" and os.path.exists(config.MAP_FILE):
            overlay_path = os.path.join(out_dir, "aggregated_overlay.png")
            overlay_heatmap_on_image(
                config.MAP_FILE, mean_density, env, save_path=overlay_path, top_nodes=voted_nodes[:10]
            )
            report["overlay_path"] = overlay_path
            print(f"Saved overlay to {overlay_path}")
        else:
            # also create a plain heatmap image (no background)
            plain_path = os.path.join(out_dir, "aggregated_heatmap.png")
            save_plain_heatmap(mean_density, plain_path)
            report["plain_heatmap"] = plain_path
            print(f"Saved plain heatmap to {plain_path}")
    except Exception as e:
        print("Failed to make overlay:", e)

    # Print summary
    print("Batch run summary:")
    print(f"  Trials requested: {n_trials}")
    print(f"  Trials completed: {len(succ)}")
    print(f"  Evacuated fraction (per trial): {evacuated_fracs}")
    print(f"  Steps (per trial): {steps}")
    print("  Top voted bottleneck nodes (row, col), votes, mean_density:")
    for pos, votes, density_val in voted_nodes[:10]:
        print(f"    {pos}  votes={votes}  mean_density={density_val:.2f}")

    return report


# -----------------------
# Visualization helpers for overlays
# -----------------------
def overlay_heatmap_on_image(image_path: str, density: np.ndarray, env: EnvironmentGraph, save_path: str = None, top_nodes=None):
    """
    Overlay a density heatmap on top of the raster image and optionally mark top_nodes.
    `density` shape should be (height, width) matching env.height, env.width or grid mapping.
    """
    img = Image.open(image_path).convert("RGBA")
    img_w, img_h = img.size

    # Map density array extent to grid coordinates: use same extent as visualization,
    # origin lower, extent (-0.5, env.width - 0.5, -0.5, env.height - 0.5)
    extent = (-0.5, env.width - 0.5, -0.5, env.height - 0.5)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, extent=extent, origin="upper")  # origin=upper for PIL coordinates; extent aligns with grid
    # Show density (rescale to image resolution by imshow with extent)
    im = ax.imshow(density, origin="lower", extent=extent, cmap="Reds", alpha=0.5, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean visit count")

    # mark exits if env supports it
    try:
        exit_nodes = [n for n in env.graph.nodes() if env.is_exit(n)]
        pos = {n: env.get_pos(n) for n in env.graph.nodes()}
        if exit_nodes:
            exit_xy = np.array([pos[n] for n in exit_nodes])
            ax.scatter(exit_xy[:, 0], exit_xy[:, 1], s=80, marker="s", facecolors="limegreen", edgecolors="black", label="Exit")
    except Exception:
        pass

    # mark top nodes
    if top_nodes:
        # top_nodes is list of ((row,col), votes, density_val) - convert to (x,y)
        for (row, col), votes, dval in top_nodes:
            # density array is indexed [row, col] -> map to x=col, y=row
            x = col
            y = row
            ax.scatter([x + 0.0], [y + 0.0], s=120, marker="o", facecolors="none", edgecolors="yellow", linewidths=2)
            ax.text(x + 0.2, y + 0.2, f"v{votes}", color="yellow", fontsize=9, weight="bold")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title("Aggregated Heatmap Overlay")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_plain_heatmap(density: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(density, origin="lower", interpolation="nearest", cmap="Reds")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean visit count")
    ax.set_title("Aggregated Mean Density")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# -----------------------
# CLI entrypoint
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Crowd simulation runner")
    p.add_argument("scenario", nargs="?", default="normal", help="scenario name (normal / evacuation / floorplan_image / batch)")
    p.add_argument("--trials", type=int, default=5, help="number of trials for batch mode")
    p.add_argument("--workers", type=int, default=None, help="parallel workers (default = cpu_count()-1)")
    p.add_argument("--agents", type=int, default=None, help="override number of agents")
    p.add_argument("--steps", type=int, default=None, help="override max steps per trial")
    p.add_argument("--target-percent", type=float, default=1.0, help="stop when this fraction evacuated (0..1)")
    p.add_argument("--top-k", type=int, default=10, help="top-k nodes per trial to record")
    return p.parse_args()


def main():
    args = parse_args()

    # Quick path: run the interactive visual simulation (existing behaviour)
    if args.scenario in ("normal", "visual", "single"):
        # Build env
        layout = load_layout_matrix_from_config()
        env = make_env_from_layout(layout)

        # call the existing visualization runner (keeps GUI)
        try:
            from visualization import run_visual_simulation
            run_visual_simulation(env)
        except Exception as e:
            print("Visualization failed:", e)

        return

    # Batch / floorplan modes: run many trials and aggregate
    if args.scenario in ("batch", "floorplan_batch", "floorplan_image"):
        report = run_batch(
            n_trials=args.trials,
            num_agents=args.agents,
            max_steps=args.steps,
            target_percent=args.target_percent,
            parallel_workers=args.workers,
            top_k=args.top_k,
        )
        print("Batch finished. Report keys:", list(report.keys()))
        return

    print("Unknown scenario:", args.scenario)
    print("Try 'normal' (visual) or 'batch' (aggregate multiple runs).")


if __name__ == "__main__":
    main()
