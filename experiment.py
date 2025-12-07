# experiment.py

import sys
from collections import defaultdict

import numpy as np

import config
from environment import EnvironmentGraph
from simulation import CrowdSimulation
from scenarios import load_and_apply_scenario, configure_environment_for_active_scenario
from analysis import compute_evacuation_metrics


def run_single_simulation(scenario_name: str):
    """
    Run one headless simulation for the given scenario and
    return (global_metrics, evac_metrics).
    """
    # Apply scenario config to global config
    scenario = load_and_apply_scenario(scenario_name)

    # Build environment & apply scenario-specific env setup (exits etc.)
    env = EnvironmentGraph(config.GRID_WIDTH, config.GRID_HEIGHT)
    configure_environment_for_active_scenario(env)

    # Create simulation
    sim = CrowdSimulation(env, config.NUM_AGENTS)

    # Run for MAX_STEPS (no visualization)
    for _ in range(config.MAX_STEPS):
        sim.step()

    metrics = sim.get_metrics_summary()        # global + per-type
    global_m = metrics["global"]
    evac_m = compute_evacuation_metrics(sim)   # evacuation KPIs

    return scenario, global_m, evac_m


def aggregate_experiments(scenario_name: str, runs: int):
    """
    Run the scenario multiple times and aggregate metrics (mean ± std).
    """
    # lists for metrics per run
    per_run_globals = defaultdict(list)
    per_run_evac = defaultdict(list)

    scenario_ref = None

    for i in range(1, runs + 1):
        print(f"[INFO] Running {scenario_name} (run {i}/{runs})...")
        scenario, g, e = run_single_simulation(scenario_name)
        scenario_ref = scenario

        # collect global metrics of interest
        for key in [
            "avg_steps",
            "avg_waits",
            "avg_replans",
            "avg_collisions_per_agent",
            "exit_rate",
            "avg_exit_time",
            "avg_steps_over_optimal",
            "total_collisions",
        ]:
            per_run_globals[key].append(g.get(key, None))

        # max density peak per run
        max_density_peak = max(g["max_density_over_time"]) if g.get("max_density_over_time") else 0
        per_run_globals["max_density_peak"].append(max_density_peak)

        # evac metrics
        for key in ["t_50", "t_80", "t_90"]:
            per_run_evac[key].append(e.get(key, None))
            per_run_evac["bottlenecks"] = e.get("bottlenecks", [])


    return scenario_ref, per_run_globals, per_run_evac


def _mean_std(values):
    """
    Handle None values by converting to NaN (so we can detect N/A).
    """
    arr = np.array(
        [np.nan if v is None else v for v in values],
        dtype=float,
    )
    mean = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else None
    std = float(np.nanstd(arr)) if not np.all(np.isnan(arr)) else None
    return mean, std


def print_experiment_summary(scenario, per_run_globals, per_run_evac, runs: int):
    print("\n================= Experiment Summary =================")
    print(f"Scenario   : {scenario.name}")
    print(f"Runs       : {runs}\n")

    print("--- Global config (from first run) ---")
    print(f"Agents     : {scenario.num_agents}")
    print(f"Time steps : {config.MAX_STEPS}\n")

    print("--- Aggregated metrics (mean ± std) ---")
    for key in [
        "avg_steps",
        "avg_waits",
        "avg_replans",
        "avg_collisions_per_agent",
        "exit_rate",
        "avg_exit_time",
        "avg_steps_over_optimal",
        "max_density_peak",
        "total_collisions",
    ]:
        mean, std = _mean_std(per_run_globals[key])
        label = f"{key:24s}: "
        if mean is None:
            print(f"{label}N/A ± N/A")
        else:
            print(f"{label}{mean:.2f} ± {std:.2f}")

    # Print bottleneck nodes (from the LAST run)
    print("\n--- Bottleneck nodes (top hotspots) ---")
    last_bottlenecks = per_run_evac.get("bottlenecks", [])
    if not last_bottlenecks:
        print("N/A")
    else:
        for i, (node, count) in enumerate(last_bottlenecks, start=1):
            print(f"{i}. Node {node} → {count} visits")

    print("\n--- Evacuation metrics (mean ± std) ---")
    for key in ["t_50", "t_80", "t_90"]:
        mean, std = _mean_std(per_run_evac[key])
        label_fmt = f"{key:24s}: "
        if mean is None:
            print(f"{label_fmt}N/A ± N/A")
        else:
            print(f"{label_fmt}{mean:.2f} ± {std:.2f}")

    print("=====================================================")


def main():
    if len(sys.argv) < 3:
        print("Usage: python experiment.py <scenario_name> <num_runs>")
        print("Example: python experiment.py normal 5")
        print("         python experiment.py evacuation 5")
        sys.exit(1)

    scenario_name = sys.argv[1]
    try:
        num_runs = int(sys.argv[2])
    except ValueError:
        print("num_runs must be an integer.")
        sys.exit(1)

    scenario, per_run_globals, per_run_evac = aggregate_experiments(
        scenario_name, num_runs
    )
    print_experiment_summary(scenario, per_run_globals, per_run_evac, num_runs)


if __name__ == "__main__":
    main()
