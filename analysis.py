# analysis.py

import math
import numpy as np
import matplotlib.pyplot as plt

from simulation import CrowdSimulation


# =========================================================
# 1. Basic metrics plots
# =========================================================

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
    plt.bar(x,         avg_waits, width=width, label="Waits")
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


def compute_evacuation_metrics(sim: CrowdSimulation, top_k_bottlenecks: int = 10) -> dict:
    """
    Compute evacuation KPIs for scenarios where EVACUATION_MODE is enabled.

    Returns a dict with:
        - t_50, t_80, t_90      : time to evacuate 50/80/90% of agents (or None)
        - exit_rate             : fraction of agents that reached an exit
        - bottleneck_nodes      : list of ((x, y), visit_count) sorted by density
        - peak_density          : max node density over the entire run
    """
    agents = sim.agents
    n_agents = len(agents)

    # all exit times for agents that reached an exit
    exit_times = sorted(
        a.exit_time_step for a in agents if a.exit_time_step is not None
    )

    def time_to_evacuate(p: float):
        if not exit_times or n_agents == 0:
            return None
        needed = math.ceil(p * n_agents)
        if len(exit_times) < needed:
            return None
        return exit_times[needed - 1]

    t_50 = time_to_evacuate(0.5)
    t_80 = time_to_evacuate(0.8)
    t_90 = time_to_evacuate(0.9)

    exit_rate = (
        float(sum(1 for a in agents if a.exit_reached)) / n_agents
        if n_agents > 0
        else 0.0
    )

    # bottlenecks from node visit counts
    visit_items = list(sim.node_visit_counts.items())  # ((x, y), count)
    visit_items.sort(key=lambda x: x[1], reverse=True)
    bottlenecks = visit_items[:top_k_bottlenecks]

    # max density peak over time
    peak_density = max(sim.max_density_per_step) if sim.max_density_per_step else 0

    return {
        "t_50": t_50,
        "t_80": t_80,
        "t_90": t_90,
        "exit_rate": exit_rate,
        "bottleneck_nodes": bottlenecks,
        "peak_density": peak_density,
    }


def print_evacuation_report(sim: CrowdSimulation):
    """
    Pretty-print evacuation KPIs to the console.
    Safe to call even if the scenario is not an evacuation one.
    """
    metrics = compute_evacuation_metrics(sim)
    t_50 = metrics["t_50"]
    t_80 = metrics["t_80"]
    t_90 = metrics["t_90"]

    def fmt(v):
        return "N/A" if v is None else f"{v}"

    print("\n=== Evacuation KPIs ===")
    print(f"Exit rate              : {metrics['exit_rate'] * 100:.2f}%")
    print(f"Time to evacuate 50%   : {fmt(t_50)}")
    print(f"Time to evacuate 80%   : {fmt(t_80)}")
    print(f"Time to evacuate 90%   : {fmt(t_90)}")
    print(f"Peak node density      : {metrics['peak_density']}")
    print("Top bottleneck nodes   :")
    for (node, count) in metrics["bottleneck_nodes"]:
        print(f"  Node {node} -> visits = {count}")
    print("========================\n")


# =========================================================
# 2. Evacuation KPIs (core function)
# =========================================================

def compute_evacuation_metrics(
    sim: CrowdSimulation,
    percentiles=(0.5, 0.8, 0.9),
    top_k_bottlenecks: int = 5,
):
    """
    Compute evacuation KPIs for a completed simulation.

    Returns a dict with:
        - num_agents
        - num_exited
        - exit_rate
        - evac_times: {percentile -> time_step or None}
        - max_density_peak
        - bottlenecks: list[(node, visit_count)]
    """
    agents = sim.agents
    num_agents = len(agents)

    # Collect exit times (only for agents that actually reached an exit)
    exit_times = [a.exit_time_step for a in agents if a.exit_time_step is not None]
    num_exited = len(exit_times)

    exit_rate = (num_exited / num_agents) if num_agents > 0 else 0.0

    evac_times = {p: None for p in percentiles}
    if exit_times:
        exit_times_sorted = sorted(exit_times)
        exit_times_arr = np.array(exit_times_sorted, dtype=float)

        for p in percentiles:
            # we define "time to evacuate p%" relative to total agents,
            # not just those who actually escaped
            target_count = p * num_agents
            if num_exited < target_count:
                # not enough agents escaped to reach this percentile
                evac_times[p] = None
            else:
                # index in sorted exits: ceil(target_count) - 1
                idx = int(max(0, np.ceil(target_count) - 1))
                idx = min(idx, len(exit_times_arr) - 1)
                evac_times[p] = float(exit_times_arr[idx])

    # Max density peak across the simulation
    max_density_peak = max(sim.max_density_per_step) if sim.max_density_per_step else 0

    # Bottleneck nodes: top-K nodes by cumulative visit count
    node_counts = sim.node_visit_counts
    # node_counts: dict[(x, y) -> int]
    bottlenecks = sorted(
        node_counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top_k_bottlenecks]

    return {
        "num_agents": num_agents,
        "num_exited": num_exited,
        "exit_rate": exit_rate,
        "evac_times": evac_times,
        "max_density_peak": max_density_peak,
        "bottlenecks": bottlenecks,
    }


def print_evacuation_report(sim: CrowdSimulation):
    """
    Print a nicely formatted evacuation report to the console.

    You can call this after a single run (main.py) or from within experiment.py
    for evacuation scenarios.
    """
    metrics = compute_evacuation_metrics(sim)

    print("\n================= Evacuation Report =================")
    print(f"Agents total           : {metrics['num_agents']}")
    print(f"Agents reached exit    : {metrics['num_exited']}")
    print(f"Exit rate              : {metrics['exit_rate'] * 100:.1f}%")
    print(f"Max density peak       : {metrics['max_density_peak']}")
    print()

    # Percentiles
    print("--- Evacuation times (in simulation steps) ---")
    for p, t in metrics["evac_times"].items():
        label = f"{int(p * 100)}%"
        if t is None:
            print(f"Time to evacuate {label}: N/A (not enough agents escaped)")
        else:
            print(f"Time to evacuate {label}: {t:.1f}")

    # Bottlenecks
    print("\n--- Bottleneck nodes (top visit counts) ---")
    if not metrics["bottlenecks"]:
        print("No bottlenecks detected (no node visits recorded).")
    else:
        for i, (node, visits) in enumerate(metrics["bottlenecks"], start=1):
            print(f"{i}. Node {node} -> {visits} visits")

    print("=====================================================\n")


# =========================================================
# 3. Optional: bottleneck visualization
# =========================================================

def plot_bottleneck_nodes(sim: CrowdSimulation, top_k: int = 5):
    """
    Visualize bottleneck nodes on top of the density heatmap.

    - Uses sim.get_density_matrix() as background.
    - Overlays top-K nodes from sim.node_visit_counts with markers.
    """
    density = sim.get_density_matrix()
    env = sim.env

    # Get top-K bottlenecks
    metrics = compute_evacuation_metrics(sim, top_k_bottlenecks=top_k)
    bottlenecks = metrics["bottlenecks"]  # list[(node, visits)]

    plt.figure(figsize=(6, 4))
    plt.imshow(density, origin="lower", interpolation="nearest", cmap="Reds")
    plt.colorbar(label="Visit count")
    plt.title("Bottleneck Nodes on Density Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Overlay bottleneck nodes
    if bottlenecks:
        xs = []
        ys = []
        for (x, y), visits in bottlenecks:
            xs.append(x)
            ys.append(y)

        plt.scatter(xs, ys, s=80, marker="X", edgecolors="black", facecolors="cyan", label="Bottlenecks")
        plt.legend(loc="upper right")

    plt.tight_layout()

    # Optional: save for report
    # plt.savefig("evacuation_bottlenecks.png", dpi=200)

    plt.show()
