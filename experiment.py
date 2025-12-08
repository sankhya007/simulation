# experiment.py

import sys
import statistics
from typing import Dict, List, Tuple, Optional

import numpy as np

import config
from environment import EnvironmentGraph
from simulation import CrowdSimulation
from scenarios import load_and_apply_scenario, configure_environment_for_active_scenario
from analysis import compute_evacuation_metrics


def run_single_simulation(
    scenario_name: str,
    seed_offset: int = 0,
) -> Tuple[CrowdSimulation, Dict, Dict]:
    """
    Run a single simulation for the given scenario.
    Returns (sim, metrics_summary, evacuation_metrics).
    """
    # Apply scenario (sets NUM_AGENTS, dynamic flags, EVACUATION_MODE, etc.)
    scenario = load_and_apply_scenario(scenario_name)

    # Optional: you can vary the seed across runs using seed_offset
    # (assuming your code uses config.SEED or random.seed somewhere)
    # For now, we just keep config.SEED as global.

    env = EnvironmentGraph(config.GRID_WIDTH, config.GRID_HEIGHT)
    configure_environment_for_active_scenario(env)

    sim = CrowdSimulation(env, config.NUM_AGENTS)

    for _ in range(config.MAX_STEPS):
        sim.step()

    # metrics from simulation (global + per-type)
    summary = sim.get_metrics_summary()
    # evacuation KPIs (may be mostly N/A for non-evac scenarios)
    evac = compute_evacuation_metrics(sim, print_report=False)

    return sim, summary, evac


def _extract_run_metrics(summary: Dict, evac: Dict) -> Dict:
    """
    Flatten the parts we care about into a single dict per run.
    """
    g = summary["global"]

    max_density_peak = 0
    if g.get("max_density_over_time"):
        max_density_peak = max(g["max_density_over_time"])

    return {
        "avg_steps": g.get("avg_steps", 0.0),
        "avg_waits": g.get("avg_waits", 0.0),
        "avg_replans": g.get("avg_replans", 0.0),
        "avg_collisions_per_agent": g.get("avg_collisions_per_agent", 0.0),
        "exit_rate": g.get("exit_rate", 0.0),
        "avg_exit_time": g.get("avg_exit_time", None),
        "avg_steps_over_optimal": g.get("avg_steps_over_optimal", None),
        "max_density_peak": max_density_peak,
        "total_collisions": g.get("total_collisions", 0),
        # evacuation metrics
        "t50": evac.get("t50", None),
        "t80": evac.get("t80", None),
        "t90": evac.get("t90", None),
    }


def _aggregate_runs(run_metrics: List[Dict]) -> Dict:
    """
    Compute mean ± std over multiple runs.
    """
    def mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
        vals = [v for v in values if v is not None]
        if not vals:
            return None, None
        if len(vals) == 1:
            return float(vals[0]), 0.0
        return float(statistics.mean(vals)), float(statistics.stdev(vals))

    keys = run_metrics[0].keys()
    agg: Dict[str, Dict[str, Optional[float]]] = {}

    for k in keys:
        vals = [rm[k] for rm in run_metrics]
        m, s = mean_std(vals)
        agg[k] = {"mean": m, "std": s}

    return agg


def _format_mean_std(m: Optional[float], s: Optional[float]) -> str:
    if m is None:
        return "N/A ± N/A"
    if s is None:
        return f"{m:.2f} ± N/A"
    return f"{m:.2f} ± {s:.2f}"


def print_experiment_summary(
    scenario_name: str,
    runs: int,
    first_summary: Dict,
    agg: Dict,
):
    g = first_summary["global"]

    print("\n================= Experiment Summary =================")
    print(f"Scenario   : {scenario_name}")
    print(f"Runs       : {runs}\n")

    print("--- Global config (from first run) ---")
    print(f"Agents     : {g.get('num_agents')}")
    print(f"Time steps : {g.get('time_steps')}\n")

    print("--- Aggregated metrics (mean ± std) ---")
    print(f"avg_steps               : {_format_mean_std(agg['avg_steps']['mean'], agg['avg_steps']['std'])}")
    print(f"avg_waits               : {_format_mean_std(agg['avg_waits']['mean'], agg['avg_waits']['std'])}")
    print(f"avg_replans             : {_format_mean_std(agg['avg_replans']['mean'], agg['avg_replans']['std'])}")
    print(f"avg_collisions_per_agent: {_format_mean_std(agg['avg_collisions_per_agent']['mean'], agg['avg_collisions_per_agent']['std'])}")
    print(f"exit_rate               : {_format_mean_std(agg['exit_rate']['mean'], agg['exit_rate']['std'])}")
    print(f"avg_exit_time           : {_format_mean_std(agg['avg_exit_time']['mean'], agg['avg_exit_time']['std'])}")
    print(f"avg_steps_over_optimal  : {_format_mean_std(agg['avg_steps_over_optimal']['mean'], agg['avg_steps_over_optimal']['std'])}")
    print(f"max_density_peak        : {_format_mean_std(agg['max_density_peak']['mean'], agg['max_density_peak']['std'])}")
    print(f"total_collisions        : {_format_mean_std(agg['total_collisions']['mean'], agg['total_collisions']['std'])}\n")

    print("--- Evacuation times (if applicable) ---")
    print(f"Time to evacuate 50%    : {_format_mean_std(agg['t50']['mean'], agg['t50']['std'])}")
    print(f"Time to evacuate 80%    : {_format_mean_std(agg['t80']['mean'], agg['t80']['std'])}")
    print(f"Time to evacuate 90%    : {_format_mean_std(agg['t90']['mean'], agg['t90']['std'])}")
    print("=====================================================")


def run_experiment_for_scenario(scenario_name: str, runs: int):
    run_metrics: List[Dict] = []
    first_summary: Optional[Dict] = None

    for i in range(runs):
        print(f"[INFO] Running {scenario_name} (run {i+1}/{runs})...")
        sim, summary, evac = run_single_simulation(scenario_name, seed_offset=i)
        if first_summary is None:
            first_summary = summary
        run_metrics.append(_extract_run_metrics(summary, evac))

    agg = _aggregate_runs(run_metrics)
    print_experiment_summary(scenario_name, runs, first_summary, agg)


# ---------- Strategy comparison ----------

def run_strategy_comparison(scenario_name: str, runs: int):
    """
    Compare navigation strategies under the same scenario.
    Assumes your code uses config.NAV_STRATEGY_MODE inside agent logic.
    """
    strategies = ["shortest", "congestion", "safe"]

    all_results = {}

    for strat in strategies:
        print(f"\n========== Strategy: {strat} ==========")
        config.NAV_STRATEGY_MODE = strat  # this should be used in your agent logic
        run_metrics: List[Dict] = []
        first_summary: Optional[Dict] = None

        for i in range(runs):
            print(f"[INFO] Running {scenario_name} [{strat}] (run {i+1}/{runs})...")
            sim, summary, evac = run_single_simulation(scenario_name, seed_offset=i)
            if first_summary is None:
                first_summary = summary
            run_metrics.append(_extract_run_metrics(summary, evac))

        agg = _aggregate_runs(run_metrics)
        all_results[strat] = (first_summary, agg)
        print_experiment_summary(f"{scenario_name} [{strat}]", runs, first_summary, agg)

    # Optional: brief comparison table for quick view
    print("\n===== Strategy Comparison (avg_steps / collisions / exit_rate) =====")
    for strat in strategies:
        _, agg = all_results[strat]
        s = agg["avg_steps"]["mean"]
        c = agg["avg_collisions_per_agent"]["mean"]
        e = agg["exit_rate"]["mean"]
        print(
            f"{strat:12s}  steps={s:.2f if s is not None else 'N/A'}  "
            f"coll/agent={c:.2f if c is not None else 'N/A'}  "
            f"exit_rate={e:.2f if e is not None else 'N/A'}"
        )


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python experiment.py <scenario> <runs>")
        print("  python experiment.py compare <scenario> <runs>")
        print("Examples:")
        print("  python experiment.py normal 5")
        print("  python experiment.py compare evacuation 5")
        return

    if sys.argv[1] == "compare":
        scenario_name = sys.argv[2]
        runs = int(sys.argv[3])
        run_strategy_comparison(scenario_name, runs)
    else:
        scenario_name = sys.argv[1]
        runs = int(sys.argv[2])
        run_experiment_for_scenario(scenario_name, runs)


if __name__ == "__main__":
    main()
