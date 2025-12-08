# main.py
"""
Main CLI for running the crowd sim. Supports:
 - list
 - visual <scenario>
 - run <scenario>
 - batch <scenario>

Examples:
  python main.py list
  python main.py visual normal
  python main.py run evacuation --agents 300 --steps 1200 --target-percent 0.95 --overlay --out-dir experiment1
  python main.py batch evacuation --trials 7 --workers 6 --target-percent 0.95 --agents 300 --steps 1200 --out-dir batch1
"""

import argparse
import os
import csv
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# correct import
from maps.map_loader import load_mapmeta_from_config

# Project imports
from scenarios import SCENARIO_PRESETS, load_and_apply_scenario, configure_environment_for_active_scenario
from visualization import run_visual_simulation, show_density_heatmap
from environment import EnvironmentGraph
from simulation import CrowdSimulation
import config

# analysis helpers: try to import compute_bottlenecks from analysis; fall back if missing
# analysis helpers
try:
    from analysis import compute_bottlenecks
except Exception:
    compute_bottlenecks = None

mm = load_mapmeta_from_config()
layout = mm.layout


def list_scenarios():
    print("Available scenarios:")
    for k in sorted(SCENARIO_PRESETS.keys()):
        print(" -", k)


def _save_overlay_and_csv(sim: CrowdSimulation, env: EnvironmentGraph, bottlenecks: List[Tuple[int, int]], out_dir: Path, tag: str):
    """
    Save overlay image and CSV with bottleneck coordinates.
    - bottlenecks: list of node ids or (x,y) cell indices depending on analysis output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save serializable sim state if available
    try:
        data = sim.to_serializable()
    except Exception:
        # fallback: minimal serializable
        data = {
            "time_step": getattr(sim, "time_step", None),
            "total_collisions": getattr(sim, "total_collisions", None),
            "num_agents": len(getattr(sim, "agents", [])),
        }
    (out_dir / f"{tag}_sim.json").write_text(str(data))

    # 2) Write bottlenecks to CSV (best-effort mapping)
    csv_path = out_dir / f"{tag}_bottlenecks.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "node_or_cell", "x", "y", "notes"])
        for i, b in enumerate(bottlenecks, start=1):
            # b may be node id, (x,y), or (node,score)
            if isinstance(b, tuple) and len(b) >= 2 and all(isinstance(x, (int, float)) for x in b[:2]):
                x, y = b[0], b[1]
                writer.writerow([i, f"cell", x, y, ""])
            else:
                writer.writerow([i, str(b), "", "", ""])
    print("Saved bottlenecks CSV to", csv_path)

    # 3) Create overlay image (visualization.overlay_results_on_image expected) - fallback: just show density
    try:
        # visualization may have overlay function that accepts sim, env, bottlenecks, out_path
        from visualization import overlay_results_on_image

        out_png = out_dir / f"{tag}_overlay.png"
        overlay_results_on_image(sim, env, bottlenecks, out_png)
        print("Saved overlay image to", out_png)
    except Exception:
        print("overlay_results_on_image not available; skipping image overlay.")


def _run_single_trial(scenario_name: str, agents: Optional[int], steps: Optional[int], target_percent: Optional[float], trial_index: int, out_dir: Path = Path("."), overlay: bool = False) -> Dict[str, Any]:
    """
    Run a single trial (no visualization). Returns a result dict with summary & bottlenecks.
    """
    print(f"[trial {trial_index}] building env for scenario '{scenario_name}'")
    env, meta = load_and_apply_scenario(scenario_name)
    # If the scenario didn't set num agents, override from args
    num_agents = agents if agents is not None else getattr(env, "num_agents", config.NUM_AGENTS)

    sim = CrowdSimulation(env, num_agents)
    max_steps = steps if steps is not None else config.MAX_STEPS

    target_count = None
    if target_percent:
        target_count = int(round(target_percent * num_agents))

    # run sim until either max_steps or target_count reached
    for t in range(max_steps):
        sim.step()
        if target_count:
            exited = sum(1 for a in sim.agents if getattr(a, "has_exited", False))
            if exited >= target_count:
                break

    # compute bottlenecks
    bottlenecks = None
    if compute_bottlenecks:
        try:
            bottlenecks = compute_bottlenecks(sim)
        except Exception:
            bottlenecks = None

    if bottlenecks is None:
        # fallback: choose top nodes by visit count (assuming sim.node_visit_counts exists)
        try:
            counts = getattr(sim, "node_visit_counts", None)
            if counts is None:
                # try sim.env or sim.environment
                counts = getattr(sim, "environment", None)
            # counts expected to be {node: visits}
            if isinstance(counts, dict):
                items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                bottlenecks = [k for k, v in items[:10]]
            else:
                # try sim.get_density_matrix -> find top cells
                dm = sim.get_density_matrix()
                flat_idx = dm.flatten().argsort()[::-1][:10]
                # convert flat indices back to (x,y)
                h, w = dm.shape
                b = []
                for idx in flat_idx:
                    r = idx // w
                    c = idx % w
                    b.append((c, r))
                bottlenecks = b
        except Exception:
            bottlenecks = []

    # save outputs
    tag = f"{scenario_name}_trial{trial_index}"
    _save_overlay_and_csv(sim, env, bottlenecks, out_dir, tag)

    # return summary
    summary = {
        "scenario": scenario_name,
        "trial": trial_index,
        "steps": getattr(sim, "time_step", None),
        "num_agents": len(sim.agents),
        "bottlenecks": bottlenecks,
    }
    return summary

grid_w, grid_h = mm.grid_shape
env = EnvironmentGraph(
    width=grid_w,
    height=grid_h,
    layout_matrix=layout,
    mapmeta=mm
)

def run_batch(scenario_name: str, trials: int = 5, workers: int = 2, agents: int = None, steps: int = None, target_percent: float = None, out_dir: str = "out", overlay: bool = False):
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    pool = mp.Pool(processes=workers)
    runner = partial(_run_single_trial, scenario_name, agents, steps, target_percent, out_dir=out_dir_p, overlay=overlay)

    try:
        results = pool.map(runner, range(1, trials + 1))
    finally:
        pool.close()
        pool.join()

    # Aggregate bottleneck frequencies and write summary CSV
    agg_csv = out_dir_p / f"{scenario_name}_batch_summary.csv"
    with open(agg_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial", "steps", "num_agents", "bottlenecks"])
        for r in results:
            writer.writerow([r["trial"], r["steps"], r["num_agents"], ";".join(map(str, r["bottlenecks"]))])
    print("Batch finished. Summary written to", agg_csv)


def run_single_visual(scenario_name: str):
    env, meta = load_and_apply_scenario(scenario_name)
    run_visual_simulation(env)


def run_single_nonvisual(scenario_name: str, agents: int = None, steps: int = None, target_percent: float = None, overlay: bool = False, out_dir: str = "out"):
    res = _run_single_trial(scenario_name, agents, steps, target_percent, trial_index=1, out_dir=Path(out_dir), overlay=overlay)
    print("Run complete:", res)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub_list = sub.add_parser("list", help="List available scenarios")
    sub_visual = sub.add_parser("visual", help="Visual run")
    sub_visual.add_argument("scenario", type=str)

    sub_run = sub.add_parser("run", help="Single run (non-visual)")
    sub_run.add_argument("scenario", type=str)
    sub_run.add_argument("--agents", type=int, default=None)
    sub_run.add_argument("--steps", type=int, default=None)
    sub_run.add_argument("--target-percent", type=float, default=None)
    sub_run.add_argument("--overlay", action="store_true")
    sub_run.add_argument("--out-dir", type=str, default="out_run")

    sub_batch = sub.add_parser("batch", help="Batch run (multiprocessing)")
    sub_batch.add_argument("scenario", type=str)
    sub_batch.add_argument("--trials", type=int, default=5)
    sub_batch.add_argument("--workers", type=int, default=2)
    sub_batch.add_argument("--agents", type=int, default=None)
    sub_batch.add_argument("--steps", type=int, default=None)
    sub_batch.add_argument("--target-percent", type=float, default=None)
    sub_batch.add_argument("--overlay", action="store_true")
    sub_batch.add_argument("--out-dir", type=str, default="out_batch")

    args = p.parse_args()

    if args.cmd == "list":
        list_scenarios()
        return

    if args.cmd == "visual":
        run_single_visual(args.scenario)
        return

    if args.cmd == "run":
        run_single_nonvisual(args.scenario, agents=args.agents, steps=args.steps, target_percent=args.target_percent, overlay=args.overlay, out_dir=args.out_dir)
        return

    if args.cmd == "batch":
        run_batch(args.scenario, trials=args.trials, workers=args.workers, agents=args.agents, steps=args.steps, target_percent=args.target_percent, overlay=args.overlay, out_dir=args.out_dir)
        return


if __name__ == "__main__":
    main()
