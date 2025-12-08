# experiment.py

import sys
import statistics
from typing import List, Dict

import numpy as np

import config
from scenarios import load_and_apply_scenario, configure_environment_for_active_scenario
from environment import EnvironmentGraph
from simulation import CrowdSimulation
from analysis import compute_evacuation_metrics


def run_single_simulation(scenario_name: str) -> dict:
    """
    Run a single simulation for the given scenario, return metrics summary dict.
    """
    scenario = load_and_apply_scenario(scenario_name)

    env = EnvironmentGraph(config.GRID_WIDTH, config.GRID_HEIGHT)
    configure_environment_for_active_scenario(env)

    sim = CrowdSimulation(env, config.NUM_AGENTS)

    for _ in range(config.MAX_STEPS):
        sim.step()

    metrics = sim.get_metrics_summary()
    evac = compute_evacuation_metrics(sim) if config.EVACUATION_MODE else None
    metrics["evacuation"] = evac
    return metrics


def aggregate_runs(metrics_list: List[dict]) -> Dict[str, dict]:
    """
    Given a list of metrics dicts (from multiple runs), compute mean & std
    for selected scalar metrics and evacuation KPIs.
    """
    def collect(key):
        return [m["global"][key] for m in metrics_list]

    global_samples = {}
    for k in [
        "avg_steps",
        "avg_waits",
        "avg_replans",
        "avg_collisions_per_agent",
        "exit_rate",
        "avg_exit_time",
        "avg_steps_over_optimal",
        "total_collisions",
    ]:
        vals = collect(k)
        # filter None
        vals_clean = [v for v in vals if v is not None]
        if vals_clean:
            mean = float(np.mean(vals_clean))
            std = float(np.std(vals_clean))
        else:
            mean = None
            std = None
        global_samples[k] = {"mean": mean, "std": std}

    # max density peak
    max_density_peaks = [
        max(m["global"]["max_density_over_time"])
        if m["global"]["max_density_over_time"]
        else 0
        for m in metrics_list
    ]
    global_samples["max_density_peak"] = {
        "mean": float(np.mean(max_density_peaks)),
        "std": float(np.std(max_density_peaks)),
    }

    # evacuation KPIs if present
    evac_metrics = [m.get("evacuation") for m in metrics_list if m.get("evacuation")]
    evac_summary = None
    if evac_metrics:
        def collect_evac(key):
            vals = [em[key] for em in evac_metrics if em[key] is not None]
            if not vals:
                return {"mean": None, "std": None}
            return {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

        evac_summary = {
            "t_50": collect_evac("t_50"),
            "t_80": collect_evac("t_80"),
            "t_90": collect_evac("t_90"),
        }

    return {
        "global": global_samples,
        "evacuation": evac_summary,
        "raw": metrics_list,
    }


def print_experiment_summary(scenario_name: str, runs: int, agg: Dict[str, dict]):
    print("\n================= Experiment Summary =================")
    print(f"Scenario   : {scenario_name}")
    print(f"Runs       : {runs}\n")

    first = agg["raw"][0]
    g0 = first["global"]
    print("--- Global config (from first run) ---")
    print(f"Agents     : {g0['num_agents']}")
    print(f"Time steps : {g0['time_steps']}\n")

    def fmt(mean, std):
        if mean is None:
            return "N/A ± N/A"
        return f"{mean:.2f} ± {std:.2f}"

    print("--- Aggregated metrics (mean ± std) ---")
    g = agg["global"]
    print(f"avg_steps               : {fmt(g['avg_steps']['mean'], g['avg_steps']['std'])}")
    print(f"avg_waits               : {fmt(g['avg_waits']['mean'], g['avg_waits']['std'])}")
    print(f"avg_replans             : {fmt(g['avg_replans']['mean'], g['avg_replans']['std'])}")
    print("avg_collisions_per_agent: "
          f"{fmt(g['avg_collisions_per_agent']['mean'], g['avg_collisions_per_agent']['std'])}")
    print(f"exit_rate               : {fmt(g['exit_rate']['mean'], g['exit_rate']['std'])}")
    print(f"avg_exit_time           : {fmt(g['avg_exit_time']['mean'], g['avg_exit_time']['std'])}")
    print("avg_steps_over_optimal  : "
          f"{fmt(g['avg_steps_over_optimal']['mean'], g['avg_steps_over_optimal']['std'])}")
    print(f"max_density_peak        : {fmt(g['max_density_peak']['mean'], g['max_density_peak']['std'])}")
    print(f"total_collisions        : {fmt(g['total_collisions']['mean'], g['total_collisions']['std'])}")

    print("\n--- Evacuation times (if applicable) ---")
    evac = agg["evacuation"]
    if evac is None:
        print("Time to evacuate 50%    : N/A ± N/A")
        print("Time to evacuate 80%    : N/A ± N/A")
        print("Time to evacuate 90%    : N/A ± N/A")
    else:
        print("Time to evacuate 50%    : "
              f"{fmt(evac['t_50']['mean'], evac['t_50']['std'])}")
        print("Time to evacuate 80%    : "
              f"{fmt(evac['t_80']['mean'], evac['t_80']['std'])}")
        print("Time to evacuate 90%    : "
              f"{fmt(evac['t_90']['mean'], evac['t_90']['std'])}")
    print("=====================================================\n")


def run_batch_experiment(scenario_name: str, runs: int):
    metrics_list = []
    for i in range(1, runs + 1):
        print(f"[INFO] Running {scenario_name} (run {i}/{runs})...")
        m = run_single_simulation(scenario_name)
        metrics_list.append(m)
    agg = aggregate_runs(metrics_list)
    print_experiment_summary(scenario_name, runs, agg)


# --------- Strategy comparison ---------

def run_strategy_comparison(scenario_name: str, runs: int):
    """
    Compare navigation strategies on the same scenario.
    Uses NAV_STRATEGY_MODE from config.
    """
    strategies = ["shortest", "congestion", "safe", "mixed"]
    results = {}

    print(f"[INFO] Strategy comparison on scenario '{scenario_name}' with {runs} runs each.\n")

    for mode in strategies:
        print(f"=== Strategy: {mode} ===")
        config.NAV_STRATEGY_MODE = mode
        metrics_list = []
        for i in range(1, runs + 1):
            print(f"[INFO] Running {mode} (run {i}/{runs})...")
            m = run_single_simulation(scenario_name)
            metrics_list.append(m)
        agg = aggregate_runs(metrics_list)
        results[mode] = agg
        print_experiment_summary(f"{scenario_name} [{mode}]", runs, agg)

    # Simple comparison table for a few key metrics
    print("\n========= Strategy Comparison (mean values) =========")
    print(f"{'Strategy':<12} {'avg_steps':>10} {'collisions':>12} {'exit_rate':>10}")
    for mode in strategies:
        g = results[mode]["global"]
        avg_steps = g["avg_steps"]["mean"]
        coll = g["total_collisions"]["mean"]
        exit_rate = g["exit_rate"]["mean"]
        print(f"{mode:<12} {avg_steps:>10.2f} {coll:>12.2f} {exit_rate:>10.2f}")
    print("=====================================================\n")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python experiment.py <scenario> <runs>")
        print("  python experiment.py compare_strategies <scenario> <runs>")
        print("Examples:")
        print("  python experiment.py normal 5")
        print("  python experiment.py compare_strategies normal 3")
        sys.exit(1)

    if sys.argv[1] == "compare_strategies":
        if len(sys.argv) < 4:
            print("Usage: python experiment.py compare_strategies <scenario> <runs>")
            sys.exit(1)
        scenario_name = sys.argv[2]
        runs = int(sys.argv[3])
        run_strategy_comparison(scenario_name, runs)
    else:
        scenario_name = sys.argv[1]
        runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        run_batch_experiment(scenario_name, runs)


if __name__ == "__main__":
    main()
