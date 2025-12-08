# experiments_runner.py
"""
Run repeated simulation trials on the same EnvironmentGraph and
aggregate bottleneck / evacuation metrics.

Usage: call `run_multiple_trials(...)` from main.py or a REPL after you
create `env` (EnvironmentGraph).
"""

import csv
import os
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from simulation import CrowdSimulation
from analysis import compute_evacuation_metrics  # optional, if present


def run_single_trial(env, num_agents: int, evac_target_fraction: float = 0.9,
                     max_steps: Optional[int] = None, verbose: bool = False):
    """
    Run one trial of the simulation.

    Returns a dict:
        {
            "density": 2D numpy array (cumulative visits per cell),
            "exit_times": list of exit step per agent (None if not exited),
            "time_steps": number of steps run,
            "total_collisions": int,
            "agents_exited_fraction": float
        }

    Stopping conditions:
      - If evac_target_fraction is provided: stop when fraction of agents that have exited >= target.
      - Otherwise stop at max_steps if provided, else run until sim.time_step >= some safe upper bound.
    """
    sim = CrowdSimulation(env, num_agents)

    if max_steps is None:
        # reasonable ceiling to avoid infinite runs
        max_steps = 2000

    target_count = int(np.ceil(evac_target_fraction * num_agents)) if evac_target_fraction else None

    if verbose:
        print(f"[trial] starting: num_agents={num_agents}, evac_target={evac_target_fraction}, max_steps={max_steps}")

    # run loop
    start = time.time()
    while True:
        sim.step()
        # compute how many agents have evacuated (we assume Agent.exit_time or agent.exited property)
        exited = sum(1 for a in sim.agents if getattr(a, "exited", False) or getattr(a, "exit_time_step", None) is not None)
        if verbose and sim.time_step % 50 == 0:
            print(f" step {sim.time_step}  exited={exited}/{num_agents}")

        # stop if reached target fraction
        if target_count is not None and exited >= target_count:
            if verbose:
                print(f"[trial] reached evac target {exited}/{num_agents} at step {sim.time_step}")
            break

        # stop if reached max steps
        if sim.time_step >= max_steps:
            if verbose:
                print(f"[trial] reached max_steps={max_steps}; exited={exited}/{num_agents}")
            break

    elapsed = time.time() - start

    # get density matrix from simulation (cumulative node visits)
    density = np.array(sim.get_density_matrix())  # shape: (H, W) or similar

    # try to extract exit times per agent
    exit_times = []
    for a in sim.agents:
        et = getattr(a, "exit_time_step", None)
        # some implementations use a boolean `exited` and store exit time elsewhere
        exit_times.append(et)

    # collisions and sim length
    total_collisions = getattr(sim, "total_collisions", None)
    time_steps = sim.time_step

    return {
        "density": density,
        "exit_times": exit_times,
        "time_steps": time_steps,
        "total_collisions": total_collisions,
        "agents_exited_fraction": exited / num_agents,
        "elapsed_sec": elapsed,
    }


def aggregate_trials(trial_results: List[dict]) -> dict:
    """
    Aggregate multiple trial results.

    Returns summary dict:
      - avg_density: average density matrix across trials
      - std_density: std dev density matrix across trials
      - aggregated_exit_times: flattened list of exit times (not-None)
      - per_trial_summary: list of useful per-trial stats
    """
    densities = [tr["density"] for tr in trial_results]
    # verify consistent shapes
    shapes = {d.shape for d in densities}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent density shapes across trials: {shapes}")
    stack = np.stack(densities, axis=0)  # (T, H, W)
    avg_density = np.mean(stack, axis=0)
    std_density = np.std(stack, axis=0)

    aggregated_exit_times = []
    for tr in trial_results:
        for et in tr["exit_times"]:
            if et is not None:
                aggregated_exit_times.append(et)

    per_trial_summary = []
    for i, tr in enumerate(trial_results):
        per_trial_summary.append({
            "trial": i,
            "time_steps": tr["time_steps"],
            "exited_fraction": tr["agents_exited_fraction"],
            "total_collisions": tr["total_collisions"],
            "elapsed_sec": tr["elapsed_sec"],
            "sum_visits": int(np.sum(tr["density"])),
        })

    return {
        "avg_density": avg_density,
        "std_density": std_density,
        "aggregated_exit_times": aggregated_exit_times,
        "per_trial_summary": per_trial_summary,
    }


def find_topk_bottleneck_cells(avg_density: np.ndarray, k: int = 10) -> List[Tuple[int, int, float]]:
    """
    Given avg_density (H x W), return top-k cells as list of tuples:
      (row_index, col_index, avg_visit_value), sorted by value desc.
    """
    flat = avg_density.flatten()
    idx = np.argsort(flat)[::-1][:k]
    h, w = avg_density.shape
    res = []
    for i in idx:
        r = i // w
        c = i % w
        res.append((r, c, float(avg_density[r, c])))
    return res


def save_aggregated_results(out_dir: str, aggregated: dict, trial_results: List[dict]):
    os.makedirs(out_dir, exist_ok=True)
    # save avg / std matrices
    np.save(os.path.join(out_dir, "avg_density.npy"), aggregated["avg_density"])
    np.save(os.path.join(out_dir, "std_density.npy"), aggregated["std_density"])

    # save per-trial summaries
    with open(os.path.join(out_dir, "trial_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregated["per_trial_summary"][0].keys()))
        writer.writeheader()
        for r in aggregated["per_trial_summary"]:
            writer.writerow(r)

    # save raw densities as .npy per-trial
    for i, tr in enumerate(trial_results):
        np.save(os.path.join(out_dir, f"trial_{i}_density.npy"), tr["density"])


def plot_avg_heatmap_on_axes(ax, avg_density: np.ndarray, env, alpha=0.55, cmap="Reds"):
    """
    Draw average density heatmap over the environment coordinate extents.
    Uses same extent as the visualization code: (-0.5, width-0.5, -0.5, height-0.5)
    """
    extent = (-0.5, env.width - 0.5, -0.5, env.height - 0.5)
    im = ax.imshow(avg_density, origin="lower", extent=extent, interpolation="nearest", cmap=cmap, alpha=alpha)
    return im


def run_multiple_trials(env,
                        trials: int = 5,
                        num_agents: int = 60,
                        evac_target_fraction: float = 0.9,
                        max_steps: Optional[int] = None,
                        out_dir: str = "experiments/output",
                        top_k_bottlenecks: int = 10,
                        verbose: bool = True):
    """
    Run `trials` independent runs on the same `env`, aggregate and produce results.

    Returns a dictionary with aggregated results and saves outputs to out_dir.
    """
    all_results = []
    for t in range(trials):
        if verbose:
            print(f"=== Running trial {t+1}/{trials} ===")
        tr = run_single_trial(env, num_agents=num_agents,
                              evac_target_fraction=evac_target_fraction,
                              max_steps=max_steps, verbose=verbose)
        all_results.append(tr)

    aggregated = aggregate_trials(all_results)
    top_cells = find_topk_bottleneck_cells(aggregated["avg_density"], k=top_k_bottlenecks)

    # Output & save
    os.makedirs(out_dir, exist_ok=True)
    save_aggregated_results(out_dir, aggregated, all_results)

    # small text report
    report_lines = []
    report_lines.append(f"Trials: {trials}")
    report_lines.append(f"Agents per trial: {num_agents}")
    report_lines.append(f"Evac target fraction: {evac_target_fraction}")
    report_lines.append(f"Top {top_k_bottlenecks} bottleneck cells (row, col, avg_visits):")
    for r, c, v in top_cells:
        report_lines.append(f"  ({r}, {c}) -> {v:.2f}")

    report_txt = "\n".join(report_lines)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report_txt)

    if verbose:
        print(report_txt)

    # produce an image heatmap with the top cells annotated
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title("Average Visit Heatmap (aggregated across trials)")
    im = plot_avg_heatmap_on_axes(ax, aggregated["avg_density"], env, alpha=0.6, cmap="Reds")

    # annotate top cells
    for idx, (r, c, v) in enumerate(top_cells):
        # map grid indices to node centers (x = c, y = r)
        ax.scatter([c], [r], s=80, marker="o", facecolors="none", edgecolors="cyan", linewidths=1.6, zorder=5)
        ax.text(c + 0.2, r + 0.2, f"B{idx+1}", color="white", fontsize=9, zorder=6)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Avg visits")
    out_img = os.path.join(out_dir, "avg_heatmap_bottlenecks.png")
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    if verbose:
        print(f"[saved] {out_img}")

    return {
        "trial_results": all_results,
        "aggregated": aggregated,
        "top_cells": top_cells,
        "out_dir": out_dir,
    }
