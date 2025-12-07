# analysis.py

import math
import matplotlib.pyplot as plt
import numpy as np

from simulation import CrowdSimulation


def plot_travel_time_histogram(sim: CrowdSimulation):
    """
    Plot distribution of per-agent travel effort (steps_taken).
    In evacuation scenarios you can modify this to use exit_time_step instead.
    """
    steps = [a.steps_taken for a in sim.agents]

    plt.figure(figsize=(6, 4))
    plt.hist(steps, bins=10, edgecolor="black")
    plt.xlabel("Steps taken per agent")
    plt.ylabel("Number of agents")
    plt.title("Distribution of Agent Travel Effort")
    plt.tight_layout()

    # Optional: save for report
    # plt.savefig("metrics_travel_time_histogram.png", dpi=200)

    plt.show()


def plot_max_density_over_time(sim: CrowdSimulation):
    """
    Plot max node density per timestep (crowding over time).
    """
    if not sim.max_density_per_step:
        return

    steps = np.arange(1, len(sim.max_density_per_step) + 1)
    max_dens = np.array(sim.max_density_per_step)

    plt.figure(figsize=(6, 4))
    plt.plot(steps, max_dens, marker="o", linewidth=1)
    plt.xlabel("Simulation step")
    plt.ylabel("Max node density")
    plt.title("Maximum Crowd Density Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Optional: save for report
    # plt.savefig("metrics_max_density_over_time.png", dpi=200)

    plt.show()


def plot_metrics_by_agent_type(sim: CrowdSimulation):
    """
    Compare average steps, waits, and replans per agent type.

    Great for showing how leaders/followers/normals/panic behave differently.
    """
    summary = sim.get_metrics_summary()
    by_type = summary["by_type"]

    if not by_type:
        return

    types = sorted(by_type.keys())
    avg_steps = [by_type[t]["avg_steps"] for t in types]
    avg_waits = [by_type[t]["avg_waits"] for t in types]
    avg_replans = [by_type[t]["avg_replans"] for t in types]

    x = np.arange(len(types))
    width = 0.25

    plt.figure(figsize=(7, 4))
    plt.bar(x - width, avg_steps, width=width, label="Steps")
    plt.bar(x, avg_waits, width=width, label="Waits")
    plt.bar(x + width, avg_replans, width=width, label="Replans")

    plt.xticks(x, types)
    plt.xlabel("Agent type")
    plt.ylabel("Average count")
    plt.title("Per-agent-type Behaviour Comparison")
    plt.legend()
    plt.tight_layout()

    # Optional: save for report
    # plt.savefig("metrics_by_agent_type.png", dpi=200)

    plt.show()


# ==========================
# Evacuation KPIs (Step 2)
# ==========================

def compute_evacuation_metrics(
    sim: CrowdSimulation,
    percentiles=(0.5, 0.8, 0.9),
    top_k_bottlenecks: int = 5,
):
    """
    Compute evacuation KPIs for scenarios where agents head to exits.

    Returns a dict:
    {
        "num_agents": int,
        "num_evacuated": int,
        "percentile_times": {0.5: t50 or None, 0.8: t80 or None, ...},
        "bottlenecks": [((x, y), count), ...],
        "max_density_peak": int,
    }
    """
    exit_times = sorted(
        t for a in sim.agents for t in [a.exit_time_step] if t is not None
    )
    num_agents = len(sim.agents)
    num_evacuated = len(exit_times)

    percentile_times = {}
    for p in percentiles:
        if num_agents == 0 or num_evacuated == 0:
            percentile_times[p] = None
            continue

        target = p * num_agents
        if num_evacuated < target:
            # Not enough evacuated agents to reach this percentile
            percentile_times[p] = None
        else:
            idx = max(0, math.ceil(target) - 1)
            idx = min(idx, num_evacuated - 1)
            percentile_times[p] = exit_times[idx]

    # Bottlenecks: top-K nodes by cumulative visit count
    items = list(sim.node_visit_counts.items())
    items.sort(key=lambda kv: kv[1], reverse=True)
    bottlenecks = items[:top_k_bottlenecks]

    max_density_peak = max(sim.max_density_per_step) if sim.max_density_per_step else 0

    return {
        "num_agents": num_agents,
        "num_evacuated": num_evacuated,
        "percentile_times": percentile_times,
        "bottlenecks": bottlenecks,
        "max_density_peak": max_density_peak,
    }


def show_evacuation_report(evac_metrics: dict):
    """
    Nicely formatted text report for evacuation KPIs.
    Call this after running an evacuation scenario.
    """
    num_agents = evac_metrics["num_agents"]
    num_evacuated = evac_metrics["num_evacuated"]
    pct_times = evac_metrics["percentile_times"]
    bottlenecks = evac_metrics["bottlenecks"]

    print("\n=== Evacuation Metrics ===")
    if num_agents == 0:
        print("No agents in simulation.")
        return

    evac_pct = 100.0 * num_evacuated / num_agents
    print(f"Agents evacuated: {num_evacuated}/{num_agents} ({evac_pct:.1f}%)")

    for p in sorted(pct_times.keys()):
        t = pct_times[p]
        label = int(p * 100)
        if t is None:
            print(f"Time to evacuate {label}%: N/A (not enough agents evacuated)")
        else:
            print(f"Time to evacuate {label}%: step {t}")

    print("\nTop bottleneck nodes (node -> visit count):")
    if not bottlenecks:
        print("  (no data)")
    else:
        for (node, count) in bottlenecks:
            print(f"  {node}: {count} visits")
