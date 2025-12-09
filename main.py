# main.py
"""
Main CLI for running the crowd sim. Supports:
 - list
 - visual <scenario>
 - run <scenario>
 - batch <scenario>
"""

import argparse
import os
import csv
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from maps.map_loader import load_mapmeta_from_config
from scenarios import (
    SCENARIO_PRESETS,
    load_and_apply_scenario,
    configure_environment_for_active_scenario,
)
from visualization import run_visual_simulation, show_density_heatmap
from environment import EnvironmentGraph
from simulation import CrowdSimulation
import config

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


def _save_overlay_and_csv(
    sim: CrowdSimulation,
    env: EnvironmentGraph,
    bottlenecks: List[Tuple[int, int]],
    out_dir: Path,
    tag: str,
):
    """
    Save overlay + CSV with grid + CAD coordinates.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # save sim json
    try:
        data = sim.to_serializable()
    except Exception:
        data = {
            "time_step": getattr(sim, "time_step", None),
            "total_collisions": getattr(sim, "total_collisions", None),
            "num_agents": len(getattr(sim, "agents", [])),
        }
    (out_dir / f"{tag}_sim.json").write_text(str(data))

    # -------------------------
    # helper: grid → CAD
    # -------------------------
    def _grid_to_cad(gx: int, gy: int):
        try:
            cad = env.get_cad_pos((gx, gy))
            if cad and cad != (None, None):
                return cad
        except Exception:
            pass

        try:
            fn = getattr(mm, "grid_to_cad", None)
            if callable(fn):
                return fn(gx, gy)
        except Exception:
            pass

        try:
            ext = mm.extras
            for key in ("grid_to_cad", "grid_to_cad_transform", "grid_to_cad_callable"):
                fn = ext.get(key)
                if callable(fn):
                    return fn(gx, gy)
        except Exception:
            pass

        try:
            minx, maxx, miny, maxy = mm.bbox
            gw, gh = mm.grid_shape
            cw = (maxx - minx) / gw
            ch = (maxy - miny) / gh
            return (
                minx + (gx + 0.5) * cw,
                miny + (gy + 0.5) * ch,
            )
        except Exception:
            return (None, None)

    # -------------------------
    # Write CSV
    # -------------------------
    csv_path = out_dir / f"{tag}_bottlenecks.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "node_or_cell", "x", "y", "cad_x", "cad_y", "notes"])

        for i, b in enumerate(bottlenecks, start=1):
            gx = gy = None

            if isinstance(b, tuple) and len(b) >= 2 and all(isinstance(x, (int, float)) for x in b[:2]):
                gx, gy = int(b[0]), int(b[1])
                label = "cell"
            elif isinstance(b, tuple) and len(b) >= 1 and isinstance(b[0], tuple):
                gx, gy = int(b[0][0]), int(b[0][1])
                label = "cell"
            else:
                label = str(b)

            if gx is not None and gy is not None:
                cad_x, cad_y = _grid_to_cad(gx, gy)
                writer.writerow([i, label, gx, gy, cad_x or "", cad_y or "", ""])
            else:
                writer.writerow([i, label, "", "", "", "", ""])

    print("Saved bottlenecks CSV to", csv_path)

    # overlay if available
    try:
        from visualization import overlay_results_on_image
        out_png = out_dir / f"{tag}_overlay.png"
        overlay_results_on_image(sim, env, bottlenecks, out_png)
        print("Saved overlay image to", out_png)
    except Exception:
        print("overlay_results_on_image not available; skipping overlay.")


def _run_single_trial(
    scenario_name: str,
    agents: Optional[int],
    steps: Optional[int],
    target_percent: Optional[float],
    trial_index: int,
    out_dir: Path = Path("."),
    overlay: bool = False,
):
    print(f"[trial {trial_index}] building env for scenario '{scenario_name}'")

    env, meta = load_and_apply_scenario(scenario_name)
    num_agents = agents if agents is not None else getattr(env, "num_agents", config.NUM_AGENTS)

    sim = CrowdSimulation(env, num_agents)
    max_steps = steps if steps is not None else config.MAX_STEPS

    target_count = int(target_percent * num_agents) if target_percent else None

    for t in range(max_steps):
        sim.step()
        if target_count:
            exited = sum(1 for a in sim.agents if getattr(a, "has_exited", False))
            if exited >= target_count:
                break

    # bottlenecks
    if compute_bottlenecks:
        try:
            bottlenecks = compute_bottlenecks(sim)
        except Exception:
            bottlenecks = None
    else:
        bottlenecks = None

    if bottlenecks is None:
        # fallback simple method
        try:
            dm = sim.get_density_matrix()
            flat_idx = dm.flatten().argsort()[::-1][:10]
            h, w = dm.shape
            bottlenecks = [(i % w, i // w) for i in flat_idx]
        except Exception:
            bottlenecks = []

    tag = f"{scenario_name}_trial{trial_index}"
    _save_overlay_and_csv(sim, env, bottlenecks, out_dir, tag)

    return {
        "scenario": scenario_name,
        "trial": trial_index,
        "steps": getattr(sim, "time_step", None),
        "num_agents": len(sim.agents),
        "bottlenecks": bottlenecks,
    }


grid_w, grid_h = mm.grid_shape
env = EnvironmentGraph(width=grid_w, height=grid_h, layout_matrix=layout, mapmeta=mm)


def run_batch(
    scenario_name: str,
    trials: int = 5,
    workers: int = 2,
    agents: int = None,
    steps: int = None,
    target_percent: float = None,
    out_dir: str = "out",
    overlay: bool = False,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool = mp.Pool(processes=workers)
    runner = partial(
        _run_single_trial,
        scenario_name,
        agents,
        steps,
        target_percent,
        out_dir=out,
        overlay=overlay,
    )

    try:
        results = pool.map(runner, range(1, trials + 1))
    finally:
        pool.close()
        pool.join()

    summary_csv = out / f"{scenario_name}_batch_summary.csv"
    with open(summary_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial", "steps", "num_agents", "bottlenecks"])
        for r in results:
            writer.writerow([r["trial"], r["steps"], r["num_agents"], str(r["bottlenecks"])])

    print("Batch finished. Summary written to", summary_csv)


def run_single_visual(scenario_name: str):
    env, meta = load_and_apply_scenario(scenario_name)
    run_visual_simulation(env)


def run_single_nonvisual(
    scenario_name: str,
    agents: int = None,
    steps: int = None,
    target_percent: float = None,
    overlay: bool = False,
    out_dir: str = "out",
):
    res = _run_single_trial(
        scenario_name, agents, steps, target_percent,
        trial_index=1, out_dir=Path(out_dir), overlay=overlay
    )
    print("Run complete:", res)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    v = sub.add_parser("visual")
    v.add_argument("scenario")

    r = sub.add_parser("run")
    r.add_argument("scenario")
    r.add_argument("--agents", type=int)
    r.add_argument("--steps", type=int)
    r.add_argument("--target-percent", type=float)
    r.add_argument("--overlay", action="store_true")
    r.add_argument("--out-dir", type=str, default="out_run")

    b = sub.add_parser("batch")
    b.add_argument("scenario")
    b.add_argument("--trials", type=int, default=5)
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--agents", type=int)
    b.add_argument("--steps", type=int)
    b.add_argument("--target-percent", type=float)
    b.add_argument("--overlay", action="store_true")
    b.add_argument("--out-dir", type=str, default="out_batch")

    args = p.parse_args()

    if args.cmd == "list":
        list_scenarios()
    elif args.cmd == "visual":
        run_single_visual(args.scenario)
    elif args.cmd == "run":
        run_single_nonvisual(
            args.scenario,
            agents=args.agents,
            steps=args.steps,
            target_percent=args.target_percent,
            overlay=args.overlay,
            out_dir=args.out_dir,
        )
    elif args.cmd == "batch":
        run_batch(
            args.scenario,
            trials=args.trials,
            workers=args.workers,
            agents=args.agents,
            steps=args.steps,
            target_percent=args.target_percent,
            overlay=args.overlay,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
