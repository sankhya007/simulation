# analysis.py

import numpy as np
import matplotlib.pyplot as plt
import csv

from PIL import Image
from typing import List, Tuple, Dict
from typing import Tuple

from simulation import CrowdSimulation
from config import MAP_MODE, MAP_FILE

Node = Tuple[int, int]


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


def find_bottleneck_cells(sim, top_k: int = 5, min_visits: int = 1):
    """
    Find top-k bottleneck cells from the density matrix.

    Returns a list of (x, y, visits) in grid coordinates.
    """
    density = sim.get_density_matrix()
    if density is None:
        return []

    density = np.asarray(density)
    h, w = density.shape

    flat = density.ravel()
    # sort indices by descending visit count
    indices = np.argsort(flat)[::-1]

    bottlenecks = []
    for idx in indices:
        if len(bottlenecks) >= top_k:
            break
        visits = flat[idx]
        if visits < min_visits:
            break
        y, x = divmod(idx, w)
        bottlenecks.append((x, y, int(visits)))
    return bottlenecks


def overlay_results_on_floorplan(sim, env, top_k: int = 5):
    """
    Overlay the density heatmap + bottleneck markers on top of the
    original raster floorplan.

    Works only when MAP_MODE == 'raster' and MAP_FILE is a PNG/JPG.
    """
    if MAP_MODE.lower() != "raster":
        print("[overlay_results_on_floorplan] Skipping: MAP_MODE is not 'raster'.")
        return
    if not MAP_FILE:
        print("[overlay_results_on_floorplan] Skipping: MAP_FILE is not set.")
        return

    try:
        img = Image.open(MAP_FILE).convert("RGB")
    except Exception as e:
        print(f"[overlay_results_on_floorplan] Could not open floorplan image: {e}")
        return

    density = sim.get_density_matrix()
    density = np.asarray(density)
    h, w = density.shape

    # Resize the image to match the grid resolution for clean alignment
    img_resized = img.resize((w, h))  # (width, height)
    img_arr = np.asarray(img_resized)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Density & Bottlenecks over Floorplan", fontsize=14, pad=12)

    # Floorplan as background
    ax.imshow(
        img_arr,
        origin="lower",
        extent=(-0.5, w - 0.5, -0.5, h - 0.5),
        alpha=0.9,
    )

    # Density heatmap on top
    dens_img = ax.imshow(
        density,
        origin="lower",
        extent=(-0.5, w - 0.5, -0.5, h - 0.5),
        cmap="Reds",
        alpha=0.5,
    )
    cbar = fig.colorbar(dens_img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cumulative node visits (congestion)")

    # Bottleneck markers
    bottlenecks = find_bottleneck_cells(sim, top_k=top_k, min_visits=1)
    if bottlenecks:
        xs = [x for x, y, v in bottlenecks]
        ys = [y for x, y, v in bottlenecks]
        ax.scatter(
            xs,
            ys,
            s=80,
            edgecolors="black",
            facecolors="cyan",
            marker="o",
            label="Bottleneck",
            zorder=5,
        )
        # Label them B1, B2, ...
        for i, (x, y, v) in enumerate(bottlenecks, start=1):
            ax.text(
                x + 0.2,
                y + 0.2,
                f"B{i}",
                color="yellow",
                fontsize=8,
                zorder=6,
            )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.show()

def compute_bottlenecks(sim, top_k: int = 10) -> List[Tuple[Node, int]]:
    """
    Return the top_k bottleneck nodes as (node_tuple, visit_count) sorted descending.
    Uses sim.node_visit_counts (a dict mapping node -> visit_count).
    """
    counts = sim.node_visit_counts  # expected: dict {(x,y): int}
    if not counts:
        return []

    # Convert to list and sort
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top = items[:top_k]
    return top  # list of ((x,y), count)

def export_bottlenecks_to_csv(bottlenecks: List[Tuple[Node, int]], out_path: str):
    """
    Save top bottlenecks to CSV with columns: x,y,visits
    """
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "visits"])
        for (x, y), visits in bottlenecks:
            writer.writerow([x, y, visits])
            
            
def map_cell_to_dxf_coords(cell: Tuple[int, int], env, dxf_bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Map a grid cell (x,y) to DXF coordinates using the bbox returned by the DXF loader:
        dxf_bbox = (min_x, max_x, min_y, max_y)
    The EnvironmentGraph built from layout has env.width and env.height (grid dims).
    We'll map the center of the cell to the corresponding DXF coordinate.

    Returns (real_x, real_y).
    """
    (x, y) = cell
    min_x, max_x, min_y, max_y = dxf_bbox

    grid_w = env.width
    grid_h = env.height

    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("Environment grid has invalid width/height")

    # cell center in normalized CAD coords
    real_x = min_x + (x + 0.5) * ((max_x - min_x) / grid_w)
    real_y = min_y + (y + 0.5) * ((max_y - min_y) / grid_h)
    return (real_x, real_y)