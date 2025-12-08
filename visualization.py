# visualization.py

import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from simulation import CrowdSimulation
from config import NUM_AGENTS, MAX_STEPS, SEED, EVACUATION_MODE
from analysis import (
    plot_travel_time_histogram,
    plot_max_density_over_time,
    plot_metrics_by_agent_type,
    print_evacuation_report,
)
from scenarios import configure_environment_for_active_scenario


def run_visual_simulation(env):
    """
    Run the visual crowd simulation on a pre-constructed EnvironmentGraph.

    `env` is created in main.py, possibly from:
      - a synthetic grid, or
      - a raster floorplan (PNG/JPG), or
      - later: a DXF floorplan.
    """
    random.seed(SEED)

    # --- configure environment according to the active scenario (exits etc.) ---
    configure_environment_for_active_scenario(env)

    # --- create simulation object ---
    sim = CrowdSimulation(env, NUM_AGENTS)

    # Precompute node positions for drawing
    pos = {n: env.get_pos(n) for n in env.graph.nodes()}

    # --- figure & axes setup ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Graph-based Crowd Simulation")
    ax.set_xlim(-1, env.width)
    ax.set_ylim(-1, env.height)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # --- draw static edges (background) ---
    for (u, v) in env.graph.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], linewidth=0.4, alpha=0.25)

    # --- helpers for node types ---

    def get_exit_nodes():
        return [n for n in env.graph.nodes() if env.is_exit(n)]

    def get_blocked_nodes():
        return [n for n, d in env.graph.nodes(data=True) if not env.is_accessible(n)]

    # Initial sets
    exit_nodes = get_exit_nodes()
    blocked_nodes = get_blocked_nodes()

    exit_xy = np.array([pos[n] for n in exit_nodes]) if exit_nodes else np.empty((0, 2))
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

    blocked_xy = (
        np.array([pos[n] for n in blocked_nodes]) if blocked_nodes else np.empty((0, 2))
    )
    blocked_scat = ax.scatter(
        blocked_xy[:, 0] if len(blocked_xy) > 0 else [],
        blocked_xy[:, 1] if len(blocked_xy) > 0 else [],
        s=40,
        marker="X",
        edgecolors="darkred",
        facecolors="red",
        label="Blocked node",
        zorder=3,
    )

    # --- live density heatmap ---
    density_mat = sim.get_density_matrix()
    density_img = ax.imshow(
        density_mat,
        origin="lower",
        extent=(-0.5, env.width - 0.5, -0.5, env.height - 0.5),
        cmap="Reds",
        alpha=0.4,
        interpolation="nearest",
        zorder=0,
    )

    cbar = fig.colorbar(density_img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cumulative node visits (congestion)")

    # --- agents: initial positions + color by type ---
    type_to_color = {
        "leader": "tab:blue",
        "follower": "tab:orange",
        "normal": "gray",
        "panic": "tab:red",
    }

    def get_agent_xy_and_colors():
        xs, ys, cs = [], [], []
        for agent in sim.agents:
            x, y = agent.get_position()
            xs.append(x)
            ys.append(y)
            cs.append(type_to_color.get(agent.agent_type, "black"))
        return np.array(xs), np.array(ys), cs

    xs, ys, cs = get_agent_xy_and_colors()
    agent_scat = ax.scatter(
        xs,
        ys,
        s=40,
        c=cs,
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
    )

    # --- info text ---
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

    # --- legend ---
    legend_elements = [
        Patch(
            facecolor="limegreen",
            edgecolor="black",
            label="Exit node",
        ),
        Patch(
            facecolor="red",
            edgecolor="darkred",
            label="Blocked node",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Leader",
            markerfacecolor=type_to_color["leader"],
            markeredgecolor="black",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Follower",
            markerfacecolor=type_to_color["follower"],
            markeredgecolor="black",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Normal",
            markerfacecolor=type_to_color["normal"],
            markeredgecolor="black",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
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

    # --- update function for animation ---

    def update(frame):
        # Advance simulation one step
        sim.step()

        # Update agent positions & colors
        xs, ys, cs = get_agent_xy_and_colors()
        if len(xs) > 0:
            agent_scat.set_offsets(np.column_stack([xs, ys]))
        else:
            agent_scat.set_offsets(np.empty((0, 2)))
        agent_scat.set_facecolors(cs)

        # Update density heatmap
        density_mat = sim.get_density_matrix()
        density_img.set_data(density_mat)

        # Update exits & blocked nodes (in case scenario changes during sim)
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

        # Info panel
        info_text.set_text(
            f"Step: {sim.time_step}\n"
            f"Agents: {len(sim.agents)}\n"
            f"Total collisions: {sim.total_collisions}\n"
            f"Occupied nodes: {len(sim.last_node_occupancy)}"
        )

        return agent_scat, density_img, exit_scat, blocked_scat, info_text

    anim = FuncAnimation(
        fig,
        update,
        frames=MAX_STEPS,
        interval=60,
        blit=False,
    )

    # Optional: save MP4
    # anim.save("crowd_simulation_dynamic.mp4", fps=20, dpi=150)

    plt.tight_layout()
    plt.show()

    # After simulation, print summary & show metrics
    sim.summary()
    if EVACUATION_MODE:
        print_evacuation_report(sim)
    show_density_heatmap(sim)
    plot_travel_time_histogram(sim)
    plot_max_density_over_time(sim)
    plot_metrics_by_agent_type(sim)


def show_density_heatmap(sim: CrowdSimulation):
    density = sim.get_density_matrix()

    plt.figure(figsize=(6, 4))
    plt.imshow(density, origin="lower", interpolation="nearest", cmap="Reds")
    plt.colorbar(label="Visit count")
    plt.title("Crowd Density Heatmap (Cumulative Visits)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()

    # Optional: save figure for report
    # plt.savefig("crowd_density_heatmap.png", dpi=200)

    plt.show()
