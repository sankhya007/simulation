# visualization.py

import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from environment import EnvironmentGraph
from simulation import CrowdSimulation
from config import GRID_WIDTH, GRID_HEIGHT, NUM_AGENTS, MAX_STEPS, SEED
from scenarios import configure_environment_for_active_scenario, get_active_scenario

# Analysis helpers (all optional; if you comment them out, core sim still works)
from analysis import (
    plot_travel_time_histogram,
    plot_max_density_over_time,
    plot_metrics_by_agent_type,
)

# If you have evacuation KPIs helper, you can import it like this:
try:
    from analysis import compute_evacuation_metrics
    HAS_EVAC_METRICS = True
except ImportError:
    HAS_EVAC_METRICS = False


def _get_agent_xy_and_colors(sim: CrowdSimulation, type_to_color: dict) -> Tuple[np.ndarray, np.ndarray, list]:
    xs, ys, cs = [], [], []
    for agent in sim.agents:
        x, y = agent.get_position()
        xs.append(x)
        ys.append(y)
        cs.append(type_to_color.get(agent.agent_type, "black"))
    return np.array(xs), np.array(ys), cs


def run_visual_simulation():
    """Run one interactive simulation with live visualization and post-run plots."""
    random.seed(SEED)

    # --- 1. Create environment & apply scenario-specific configuration ---
    env = EnvironmentGraph(GRID_WIDTH, GRID_HEIGHT)
    configure_environment_for_active_scenario(env)  # sets exits based on chosen scenario

    # --- 2. Build simulation ---
    sim = CrowdSimulation(env, NUM_AGENTS)
    pos = {n: env.get_pos(n) for n in env.graph.nodes()}

    scenario = get_active_scenario()
    scenario_name = scenario.name if scenario is not None else "custom"
    scenario_desc = scenario.description if scenario is not None else "No scenario description available."

    # --- 3. Matplotlib figure setup ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"Crowd Simulation – Scenario: {scenario_name}")
    ax.set_xlim(-1, env.width)
    ax.set_ylim(-1, env.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (graph coordinate)")
    ax.set_ylabel("Y (graph coordinate)")

    # Small text box with scenario description
    ax.text(
        0.01,
        -0.12,
        scenario_desc,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        wrap=True,
    )

    # --- 4. Draw static edges as light background graph ---
    for (u, v) in env.graph.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], linewidth=0.4, alpha=0.25, zorder=0)

    # --- 5. Helper functions for node classification ---

    def get_exit_nodes():
        return [n for n in env.graph.nodes() if env.is_exit(n)]

    def get_blocked_nodes():
        return [n for n, d in env.graph.nodes(data=True) if not env.is_accessible(n)]

    # Initial sets
    exit_nodes = get_exit_nodes()
    blocked_nodes = get_blocked_nodes()

    exit_xy = np.array([pos[n] for n in exit_nodes]) if exit_nodes else np.empty((0, 2))
    blocked_xy = np.array([pos[n] for n in blocked_nodes]) if blocked_nodes else np.empty((0, 2))

    # --- 6. Scatter overlays for exits & blocked nodes ---
    exit_scat = ax.scatter(
        exit_xy[:, 0] if len(exit_xy) > 0 else [],
        exit_xy[:, 1] if len(exit_xy) > 0 else [],
        s=60,
        marker="s",
        edgecolors="black",
        facecolors="limegreen",
        label="Exit node",
        zorder=3,
    )

    blocked_scat = ax.scatter(
        blocked_xy[:, 0] if len(blocked_xy) > 0 else [],
        blocked_xy[:, 1] if len(blocked_xy) > 0 else [],
        s=50,
        marker="X",
        edgecolors="darkred",
        facecolors="red",
        label="Blocked node",
        zorder=3,
    )

    # --- 7. Density heatmap (cumulative congestion) ---
    density_mat = sim.get_density_matrix()
    density_img = ax.imshow(
        density_mat,
        origin="lower",
        extent=(-0.5, env.width - 0.5, -0.5, env.height - 0.5),
        cmap="Reds",
        alpha=0.4,
        interpolation="nearest",
        zorder=1,
    )

    cbar = fig.colorbar(density_img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cumulative node visits (congestion)")

    # --- 8. Agents: initial positions + colors by type ---
    type_to_color = {
        "leader": "tab:blue",
        "follower": "tab:orange",
        "normal": "gray",
        "panic": "tab:red",
    }

    xs, ys, cs = _get_agent_xy_and_colors(sim, type_to_color)
    agent_scat = ax.scatter(
        xs,
        ys,
        s=40,
        c=cs,
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
    )

    # --- 9. Info box on top-left ---
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        zorder=5,
    )

    # --- 10. Legend for node types & agent types ---
    legend_elements = [
        Patch(facecolor="limegreen", edgecolor="black", label="Exit node"),
        Patch(facecolor="red", edgecolor="darkred", label="Blocked node"),
        Line2D(
            [0], [0],
            marker="o", color="w",
            label="Leader",
            markerfacecolor=type_to_color["leader"],
            markeredgecolor="black",
            markersize=8,
        ),
        Line2D(
            [0], [0],
            marker="o", color="w",
            label="Follower",
            markerfacecolor=type_to_color["follower"],
            markeredgecolor="black",
            markersize=8,
        ),
        Line2D(
            [0], [0],
            marker="o", color="w",
            label="Normal",
            markerfacecolor=type_to_color["normal"],
            markeredgecolor="black",
            markersize=8,
        ),
        Line2D(
            [0], [0],
            marker="o", color="w",
            label="Panic",
            markerfacecolor=type_to_color["panic"],
            markeredgecolor="black",
            markersize=8,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.9,
        fontsize=9,
    )

    # --- 11. Update function for animation ---

    def update(frame_idx: int):
        # Advance simulation one tick
        sim.step()

        # Agents
        xs, ys, cs = _get_agent_xy_and_colors(sim, type_to_color)
        agent_scat.set_offsets(np.column_stack([xs, ys]))
        agent_scat.set_facecolors(cs)

        # Density heatmap (cumulative visits)
        density_mat = sim.get_density_matrix()
        density_img.set_data(density_mat)

        # Exits & blocked nodes (dynamic events from simulation)
        exit_nodes = get_exit_nodes()
        blocked_nodes = get_blocked_nodes()

        if exit_nodes:
            exit_xy = np.array([pos[n] for n in exit_nodes])
        else:
            exit_xy = np.empty((0, 2))

        if blocked_nodes:
            blocked_xy = np.array([pos[n] for n in blocked_nodes])
        else:
            blocked_xy = np.empty((0, 2))

        exit_scat.set_offsets(exit_xy)
        blocked_scat.set_offsets(blocked_xy)

        # Info box
        info_text.set_text(
            f"Step: {sim.time_step}\n"
            f"Agents: {len(sim.agents)}\n"
            f"Total collisions: {sim.total_collisions}\n"
            f"Occupied nodes: {len(sim.last_node_occupancy)}"
        )

        return agent_scat, density_img, exit_scat, blocked_scat, info_text

    # --- 12. Run animation ---
    anim = FuncAnimation(
        fig,
        update,
        frames=MAX_STEPS,
        interval=60,  # ms between frames
        blit=False,
    )

    # If you want automatic MP4 export for your demo/report, uncomment:
    # anim.save("crowd_simulation_dynamic.mp4", fps=20, dpi=150)

    plt.tight_layout()
    plt.show()

    # --- 13. After simulation: summary + analysis plots ---
    sim.summary()

    # Evacuation KPIs if available and meaningful
    if HAS_EVAC_METRICS:
        try:
            compute_evacuation_metrics(sim)
        except Exception as e:
            print(f"[WARN] compute_evacuation_metrics failed: {e}")

    # Spatial congestion map
    show_density_heatmap(sim)

    # Time/congestion metrics
    plot_travel_time_histogram(sim)
    plot_max_density_over_time(sim)
    plot_metrics_by_agent_type(sim)


def show_density_heatmap(sim: CrowdSimulation):
    """Standalone density heatmap (cumulative node visits)."""
    density = sim.get_density_matrix()

    plt.figure(figsize=(6, 4))
    plt.imshow(density, origin="lower", interpolation="nearest", cmap="Reds")
    plt.colorbar(label="Visit count")
    plt.title("Crowd Density Heatmap (Cumulative Visits)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()

    # Uncomment to save PNG for your report:
    # plt.savefig("crowd_density_heatmap.png", dpi=200)

    plt.show()
