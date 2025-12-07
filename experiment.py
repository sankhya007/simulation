# experiment.py

import sys
import math
import random
from typing import Dict, List, Tuple

import numpy as np

import config
from scenarios import load_and_apply_scenario, configure_environment_for_active_scenario
from environment import EnvironmentGraph
from simulation import CrowdSimulation


def run_single_simulation(scenario_name: str) -> Tuple[Dict, Dict]:
    """
    Run one simulation for a given scenario and return:
      - global_metrics (from get_metrics_summary())
      - evac_stats (t50, t80, t90 if applicable)
    """
    # Apply scenario to config
    scenario = load_and_apply_scenario(scenario_name)

    # Set seeds for reproducibility but with variation across runs
    base_seed = config.SEED
    random.seed(base_seed)
    np.random.seed(base_seed)

    # Build environment + apply scenario-specific config (exits etc.)
    env = EnvironmentGraph(config.GRID_WIDTH, config.GRID_HEIGHT)
    configure_environment_for_active_scenario(env)

    # Create simulation
    sim = CrowdSimulation(env, config.NUM_AGENTS)

    # Run for MAX_STEPS
    for _ in range(config.MAX_STEPS):
        sim.step()

    # Metrics from sim
    metrics = sim.get_metrics_summary()
    global_metrics = metrics["global"]

    # Evacuation times (if any exits reached)
    exit_times = sorted(
        t
        for a in sim.agents
        for t in ([a.exit_time_step] if a.exit_time_step is not None else [])
    )

    evac_stats = {"t50": None, "t80": None, "t90": None}

    if exit_times:
        n = len(exit_times)

        def percentile_time(p: float) -> int:
            k = max(1, int(math.ceil(p * n))) - 1
            return exit_times[k]

        evac_stats["t50"] = percentile_time(0.5)
        evac_stats["t80"] = percentile_time(0.8)
        evac_stats["t90"] = percentile_time(0.9)

    return global_metrics, evac_stats


def aggregate_runs(
    global_list: List[Dict],
    evac_list: List[Dict],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """
    Aggregate metrics across runs.
    Returns:
      - agg_global: metric_name -> (mean, std) or (nan, nan) if no data
      - agg_evac: evac_metric_name -> (mean, std) or (nan, nan)
    """
    # Metrics we'll report
    metric_keys = [
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

    agg_global: Dict[str, Tuple[float, float]] = {}

    for key in metric_keys:
        values = []
        for gm in global_list:
            v = gm.get(key, None)
            # Only include if not None
            if v is not None:
                values.append(float(v))
        if values:
            arr = np.array(values, dtype=float)
            agg_global[key] = (float(arr.mean()), float(arr.std(ddof=0)))
        else:
            agg_global[key] = (float("nan"), float("nan"))

    # Evacuation times
    evac_keys = ["t50", "t80", "t90"]
    agg_evac: Dict[str, Tuple[float, float]] = {}

    for key in evac_keys:
        values = []
        for ev in evac_list:
            v = ev.get(key, None)
            if v is not None:
                values.append(float(v))
        if values:
            arr = np.array(values, dtype=float)
            agg_evac[key] = (float(arr.mean()), float(arr.std(ddof=0)))
        else:
            agg_evac[key] = (float("nan"), float("nan"))

    return agg_global, agg_evac


def format_mean_std(mean: float, std: float) -> str:
    """
    Format mean ± std, handling NaN as 'N/A'.
    """
    if math.isnan(mean) or math.isnan(std):
        return "N/A ± N/A"
    return f"{mean:.2f} ± {std:.2f}"


def main():
    # Usage: python experiment.py [scenario_name] [num_runs]
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
    else:
        scenario_name = config.DEFAULT_SCENARIO_NAME

    if len(sys.argv) > 2:
        try:
            num_runs = int(sys.argv[2])
        except ValueError:
            print("[WARN] Invalid num_runs, defaulting to 5")
            num_runs = 5
    else:
        num_runs = 5

    global_results: List[Dict] = []
    evac_results: List[Dict] = []

    for i in range(num_runs):
        print(f"[INFO] Running {scenario_name} (run {i+1}/{num_runs})...")
        gm, ev = run_single_simulation(scenario_name)
        global_results.append(gm)
        evac_results.append(ev)

    # Aggregate
    agg_global, agg_evac = aggregate_runs(global_results, evac_results)

    # Use first run's global config metadata for header
    first = global_results[0]
    agents = int(first.get("num_agents", config.NUM_AGENTS))
    steps = int(first.get("time_steps", config.MAX_STEPS))

    print("\n================= Experiment Summary =================")
    print(f"Scenario   : {scenario_name}")
    print(f"Runs       : {num_runs}\n")
    print("--- Global config (from first run) ---")
    print(f"Agents     : {agents}")
    print(f"Time steps : {steps}\n")

    print("--- Aggregated metrics (mean ± std) ---")
    print(f"{'avg_steps':24}: {format_mean_std(*agg_global['avg_steps'])}")
    print(f"{'avg_waits':24}: {format_mean_std(*agg_global['avg_waits'])}")
    print(f"{'avg_replans':24}: {format_mean_std(*agg_global['avg_replans'])}")
    print(
        f"{'avg_collisions_per_agent':24}: "
        f"{format_mean_std(*agg_global['avg_collisions_per_agent'])}"
    )
    print(f"{'exit_rate':24}: {format_mean_std(*agg_global['exit_rate'])}")
    print(f"{'avg_exit_time':24}: {format_mean_std(*agg_global['avg_exit_time'])}")
    print(
        f"{'avg_steps_over_optimal':24}: "
        f"{format_mean_std(*agg_global['avg_steps_over_optimal'])}"
    )
    print(
        f"{'max_density_peak':24}: "
        f"{format_mean_std(*agg_global['max_density_peak'])}"
    )
    print(
        f"{'total_collisions':24}: "
        f"{format_mean_std(*agg_global['total_collisions'])}"
    )

    print("\n--- Evacuation times (if applicable) ---")
    print(
        f"{'Time to evacuate 50%':24}: "
        f"{format_mean_std(*agg_evac['t50'])}"
    )
    print(
        f"{'Time to evacuate 80%':24}: "
        f"{format_mean_std(*agg_evac['t80'])}"
    )
    print(
        f"{'Time to evacuate 90%':24}: "
        f"{format_mean_std(*agg_evac['t90'])}"
    )
    print("=====================================================")


if __name__ == "__main__":
    main()
