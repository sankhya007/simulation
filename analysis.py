# analysis.py

import numpy as np
import matplotlib.pyplot as plt

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


def compute_evacuation_kpis(sim: CrowdSimulation):
    """
    Compute evacuation KPIs for scenarios where EVACUATION_MODE is True.
    Returns a dict with times to evacuate 50%, 80%, 90% of agents (if possible).
    """
    exit_times = [
        a.exit_time_step for a in sim.agents if a.exit_time_step is not None
    ]
    n = len(sim.agents)

    if not exit_times or n == 0:
        return {
            "t_50": None,
            "t_80": None,
            "t_90": None,
        }

    exit_times_sorted = sorted(exit_times)
    def percentile_time(p: float):
        k = int(np.ceil(p * n) - 1)
        if k < 0 or k >= len(exit_times_sorted):
            return None
        return exit_times_sorted[k]

    return {
        "t_50": percentile_time(0.5),
        "t_80": percentile_time(0.8),
        "t_90": percentile_time(0.9),
    }


def print_evacuation_report(sim: CrowdSimulation):
    """
    Print a small text report for evacuation scenarios using the KPIs above.
    """
    kpis = compute_evacuation_kpis(sim)

    print("\n=== Evacuation KPIs ===")
    if all(v is None for v in kpis.values()):
        print("No agents reached an exit or evacuation mode not active.")
        return

    def fmt(t):
        return "N/A" if t is None else f"{t}"

    print(f"Time to evacuate 50% of agents: {fmt(kpis['t_50'])}")
    print(f"Time to evacuate 80% of agents: {fmt(kpis['t_80'])}")
    print(f"Time to evacuate 90% of agents: {fmt(kpis['t_90'])}")
