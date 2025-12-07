# experiment.py

import argparse
import random
from typing import Dict, Any, List, Optional

import numpy as np

import config
from scenarios import load_and_apply_scenario, configure_environment_for_active_scenario
from environment import EnvironmentGraph
from simulation import CrowdSimulation
from agent import Agent
from analysis import (
    plot_travel_time_histogram,
    plot_max_density_over_time,
    plot_metrics_by_agent_type,
)


def compute_evacuation_percentiles(sim: CrowdSimulation) -> Dict[str, Optional[float]]:
    """
    For evacuation scenarios:
    - time_to_50: time when 50% of agents reached an exit
    - time_to_80
    - time_to_90
    Returns None for each if no agent reached an exit, or scenario isn't evacuation.
    """
    times = [a.exit_time_step for a in sim.agents if a.exit_time_step is not None]
    if not times:
        return {"t50": None, "t80": None, "t90": None}

    times_sorted = np.sort(times)
    n = len(times_sorted)

    def percentile(p: float) -> float:
        idx = int(np.ceil(p * n)) - 1
        idx = max(0, min(idx, n - 1))
        return float(times_sorted[idx])

    return {
        "t50": percentile(0.5),
        "t80": percentile(0.8),
        "t90": percentile(0.9),
    }


def run_single_simulation(scenario_name: str, seed_offset: int = 0) -> Dict[str, Any]:
    """
    Run one simulation for a given scenario in HEADLESS mode (no animation),
    and return a rich metrics dictionary.
    """
    # Reset agent IDs so they don't grow forever across runs
    Agent._id_counter = 0

    # Apply scenario to config
    scenario = load_and_apply_scenario(scenario_name)

    # Seed RNGs for reproducibility
    base_seed = config.SEED + seed_offset
    random.seed(base_seed)
    np.random.seed(base_seed)

    # Build environment
    env = EnvironmentGraph(config.GRID_WIDTH, config.GRID_HEIGHT)
    configure_environment_for_active_scenario(env)

    # Run simulation
    sim = CrowdSimulation(env, config.NUM_AGENTS)
    for _ in range(config.MAX_STEPS):
        sim.step()

    # Metrics summary (global + per-type)
    summary = sim.get_metrics_summary()
    global_metrics = summary["global"]

    # Evacuation percentiles
    evac_stats = compute_evacuation_percentiles(sim)

    # Add peak density
    max_density_peak = max(global_metrics["max_density_over_time"]) if global_metrics["max_density_over_time"] else 0
    global_metrics["max_density_peak"] = max_density_peak

    result = {
        "scenario": scenario.name,
        "config": {
            "num_agents": config.NUM_AGENTS,
            "max_steps": config.MAX_STEPS,
        },
        "global": global_metrics,
        "by_type": summary["by_type"],
        "evac": evac_stats,
        "sim": sim,  # keep full sim object so we can plot last run if needed
    }
    return result


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute mean ± std over multiple runs for key metrics.
    """
    def collect(key: str) -> np.ndarray:
        vals = []
        for r in results:
            val = r["global"].get(key)
            if val is not None:
                vals.append(val)
        return np.array(vals, dtype=float) if vals else np.array([])

    def collect_evac(key: str) -> np.ndarray:
        vals = []
        for r in results:
            val = r["evac"].get(key)
            if val is not None:
                vals.append(val)
        return np.array(vals, dtype=float) if vals else np.array([])

    def mean_std(arr: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        if arr.size == 0:
            return None, None
        return float(arr.mean()), float(arr.std(ddof=0))

    metrics_keys = [
        "avg_steps",
        "avg_waits",
        "avg_replans",
        "avg_collisions_per_agent",
        "exit_rate",
        "avg_exit_time",
        "avg_steps_over_optimal",
        "max_density_peak",
        "total_collisions",
    ]

    agg = {}
    for k in metrics_keys:
        arr = collect(k)
        m, s = mean_std(arr)
        agg[k] = {"mean": m, "std": s}

    # Evacuation percentiles
    for k in ["t50", "t80", "t90"]:
        arr = collect_evac(k)
        m, s = mean_std(arr)
        agg[k] = {"mean": m, "std": s}

    return agg


def fmt_ms(x: Optional[float]) -> str:
    if x is None:
        return "N/A ± N/A"
    return f"{x:.2f}"


def print_experiment_summary(
    scenario_name: str,
    runs: int,
    first_run_config: Dict[str, Any],
    agg: Dict[str, Any],
):
    print()
    print("================= Experiment Summary =================")
    print(f"Scenario   : {scenario_name}")
    print(f"Runs       : {runs}")
    print()
    print("--- Global config (from first run) ---")
    print(f"Agents     : {first_run_config['num_agents']}")
    print(f"Time steps : {first_run_config['max_steps']}")
    print()
    print("--- Aggregated metrics (mean ± std) ---")

    def mstd(key: str) -> str:
        d = agg.get(key, {})
        m = d.get("mean")
        s = d.get("std")
        if m is None or s is None:
            return "N/A ± N/A"
        return f"{m:.2f} ± {s:.2f}"

    print(f"avg_steps               : {mstd('avg_steps')}")
    print(f"avg_waits               : {mstd('avg_waits')}")
    print(f"avg_replans             : {mstd('avg_replans')}")
    print(f"avg_collisions_per_agent: {mstd('avg_collisions_per_agent')}")
    print(f"exit_rate               : {mstd('exit_rate')}")
    print(f"avg_exit_time           : {mstd('avg_exit_time')}")
    print(f"avg_steps_over_optimal  : {mstd('avg_steps_over_optimal')}")
    print(f"max_density_peak        : {mstd('max_density_peak')}")
    print(f"total_collisions        : {mstd('total_collisions')}")
    print()
    print("--- Evacuation times (if applicable) ---")
    print(f"Time to evacuate 50%    : {mstd('t50')}")
    print(f"Time to evacuate 80%    : {mstd('t80')}")
    print(f"Time to evacuate 90%    : {mstd('t90')}")
    print("=====================================================")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run batch experiments on crowd scenarios.")
    parser.add_argument(
        "scenario",
        type=str,
        help="Scenario name (e.g., normal, high_density, blocked, evacuation)",
    )
    parser.add_argument(
        "runs",
        type=int,
        nargs="?",
        default=5,
        help="Number of runs to execute (default: 5)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots for the LAST run to visually demonstrate how metrics are obtained.",
    )
    args = parser.parse_args()

    scenario_name = args.scenario
    num_runs = args.runs

    results: List[Dict[str, Any]] = []

    for i in range(num_runs):
        print(f"[INFO] Running {scenario_name} (run {i+1}/{num_runs})...")
        run_result = run_single_simulation(scenario_name, seed_offset=i)
        results.append(run_result)

    # Aggregate
    first_config = results[0]["config"]
    agg = aggregate_results(results)
    print_experiment_summary(scenario_name, num_runs, first_config, agg)

    # Optionally show plots for the last run (how we get the data)
    if args.show:
        print("[INFO] Showing plots for the LAST run to illustrate metric computation...")
        last_sim: CrowdSimulation = results[-1]["sim"]
        # Reuse the same analysis functions used in interactive visualization
        plot_travel_time_histogram(last_sim)
        plot_max_density_over_time(last_sim)
        plot_metrics_by_agent_type(last_sim)


if __name__ == "__main__":
    main()
