# analysis.py

import math
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from simulation import CrowdSimulation


Node = Tuple[int, int]


# ---------- Basic plots ----------

def plot_travel_time_histogram(sim: CrowdSimulation):
    """
    Plot distribution of per-agent travel effort (steps_taken).

    In evacuation scenarios, you can adapt this to use exit_time_step instead.
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


# ---------- Evacuation KPIs & bottlenecks ----------

def compute_evacuation_metrics(
    sim: CrowdSimulation,
    top_k_bottlenecks: int = 5,
    print_report: bool = True,
) -> Dict:
    """
    Compute evacuation KPIs:
      - time to evacuate 50%, 80%, 90% of agents
      - how many agents reached an exit
      - top-K bottleneck nodes (highest cumulative density)

    Returns a dict with all metrics. If print_report is True, prints nicely.
    """
    agents = sim.agents
    n_agents = len(agents)

    # --- exit times array ---
    exit_times: List[int] = [
        a.exit_time_step for a in agents if a.exit_time_step is not None
    ]
    exit_times_sorted = sorted(exit_times)

    def percentile_time(p: float) -> Optional[int]:
        """
        p in (0,1]. Returns time when p fraction of agents have exited,
        or None if not enough agents exited.
        """
        if not exit_times_sorted:
            return None
        k = math.ceil(p * n_agents)
        if k == 0:
            return None
        # if fewer agents exited than k, return None
        if len(exit_times_sorted) < k:
            return None
        return exit_times_sorted[k - 1]

    t50 = percentile_time(0.5)
    t80 = percentile_time(0.8)
    t90 = percentile_time(0.9)

    # --- bottleneck detection (top-K nodes by density) ---
    density = sim.get_density_matrix()  # shape (H, W)
    flat = density.flatten()
    if top_k_bottlenecks > 0:
        top_indices = np.argsort(flat)[::-1]  # descending
        bottlenecks: List[Dict] = []
        h, w = density.shape
        count = 0
        for idx in top_indices:
            val = flat[idx]
            if val <= 0:
                break
            y = idx // w
            x = idx % w
            bottlenecks.append({"node": (x, y), "visits": int(val)})
            count += 1
            if count >= top_k_bottlenecks:
                break
    else:
        bottlenecks = []

    evac_metrics = {
        "n_agents": n_agents,
        "n_exited": len(exit_times),
        "exit_fraction": len(exit_times) / n_agents if n_agents > 0 else 0.0,
        "t50": t50,
        "t80": t80,
        "t90": t90,
        "bottlenecks": bottlenecks,
    }

    if print_report:
        print("\n=== Evacuation KPIs ===")
        print(f"Agents evacuated      : {evac_metrics['n_exited']}/{evac_metrics['n_agents']}")
        print(f"Evacuation fraction   : {evac_metrics['exit_fraction']:.2f}")
        print(
            f"Time to 50% evacuated : {t50 if t50 is not None else 'N/A'}"
        )
        print(
            f"Time to 80% evacuated : {t80 if t80 is not None else 'N/A'}"
        )
        print(
            f"Time to 90% evacuated : {t90 if t90 is not None else 'N/A'}"
        )

        if bottlenecks:
            print("\nTop bottleneck nodes (by cumulative visits):")
            for i, b in enumerate(bottlenecks, start=1):
                print(f"  {i}. node={b['node']}  visits={b['visits']}")
        else:
            print("\nNo bottlenecks detected (no high-density nodes).")

    return evac_metrics


def show_bottlenecks_on_heatmap(sim: CrowdSimulation, top_k: int = 5):
    """
    Optional visualization helper:
    - Shows cumulative density heatmap
    - Highlights top-K bottleneck nodes
    """
    density = sim.get_density_matrix()
    h, w = density.shape

    # reuse bottleneck logic
    flat = density.flatten()
    top_indices = np.argsort(flat)[::-1]
    xs, ys = [], []
    vals = []
    count = 0
    for idx in top_indices:
        val = flat[idx]
        if val <= 0:
            break
        y = idx // w
        x = idx % w
        xs.append(x)
        ys.append(y)
        vals.append(val)
        count += 1
        if count >= top_k:
            break

    plt.figure(figsize=(6, 4))
    plt.imshow(density, origin="lower", interpolation="nearest", cmap="Reds")
    plt.colorbar(label="Visit count")
    plt.title("Evacuation Bottlenecks (Top-K Nodes)")
    plt.xlabel("X")
    plt.ylabel("Y")

    if xs:
        plt.scatter(xs, ys, s=80, facecolors="none", edgecolors="cyan", linewidths=2)
        for (x, y, v) in zip(xs, ys, vals):
            plt.text(x + 0.1, y + 0.1, str(v), color="cyan", fontsize=8)

    plt.tight_layout()

    # Optional: save for report
    # plt.savefig("evacuation_bottlenecks.png", dpi=200)

    plt.show()
