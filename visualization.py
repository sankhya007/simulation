# visualization.py

import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from environment import EnvironmentGraph, Node
from simulation import CrowdSimulation
from config import NUM_AGENTS, MAX_STEPS, SEED, EVACUATION_MODE
from analysis import (
    plot_travel_time_histogram,
    plot_max_density_over_time,
    plot_metrics_by_agent_type,
    print_evacuation_report,
)
from scenarios import configure_environment_for_active_scenario


# ----------------------------------------------------------------------
# Helpers to visualize the map / layout
# ----------------------------------------------------------------------

def _build_background_matrix(env: EnvironmentGraph) -> np.ndarray:
    """
    Create a 2D matrix where:
      - walls/blocked cells are bright
      - exits are medium
      - open corridors are dark

    This is what we draw underneath the agents as the "map".
    """
    h, w = env.height, env.width
    bg = np.zeros((h, w), dtype=float)

    for y in range(h):
        for x in range(w):
            node: Node = (x, y)
            if node not in env.graph:
                continue

            if not env.is_accessible(node):
                # wall / blocked
                bg[y, x] = 1.0
            elif env.is_exit(node):
                # exit
                bg[y, x] = 0.5
            else:
                # open corridor
                bg[y, x] = 0.1

    return bg


def _make_edge_collection(env: EnvironmentGraph, pos: dict) -> LineCollection:
    """
    Prepare a LineCollection for all graph edges (faster than many ax.plot calls).
    """
    segments = []
    for u, v in env.graph.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        segments.append([(x1, y1), (x2, y2)])

    lc = LineCollection(
        segments,
        linewidths=0.4,
        alpha=0.25,
        zorder=-1,
    )
    return lc


# ----------------------------------------------------------------------
# Main visual simulation
# ----------------------------------------------------------------------

def run_visual_simulation(env: EnvironmentGraph):
    random.seed(SEED)

    # Apply scenario-specific environment configuration (exits, etc.)
    configure_environment_for_active_scenario(env)

    # Create simulation on this environment
    sim = CrowdSimulation(env, NUM_AGENTS)

    # Positions of nodes
    pos = {n: env.get_pos(n) for n in env.graph.nodes()}

    # --- figure & axes setup ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Crowd Simulation on Floorplan Graph")
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # --- draw static floorplan background (walls / exits) ---
    bg_mat = _build_background_matrix(env)
    bg_img = ax.imshow(
        bg_mat,
        origin="lower",
        extent=(-0.5, env.width - 0.5, -0.5, env.height - 0.5),
        cmap="Greys",
        alpha=0.7,
        interpolation="nearest",
        zorder=-2,
    )

    # --- draw static graph edges on top of map ---
    edge_collection = _make_edge_collection(env, pos)
    ax.add_collection(edge_collection)

    # --- helpers for node types ---
    def get_exit_nodes():
        return [n for n in env.graph.nodes() if env.is_exit(n)]

    def get_blocked_nodes():
        return [n for n, _ in env.graph.nodes(data=True) if not env.is_accessible(n)]

    # initial sets
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
        s=10,
        marker="s",
        edgecolors="none",
        facecolors="0.3",  #dark gray
        label="Blocked node",
        zorder=1.5,
    )

    # --- live density heatmap (congestion) ---
    density_mat = sim.get_density_matrix()
    density_img = ax.imshow(
        density_mat,
        origin="lower",
        extent=(-0.5, env.width - 0.5, -0.5, env.height - 0.5),
        cmap="Reds",
        alpha=0.4,
        interpolation="nearest",
        zorder=-0.5,
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
        Patch(facecolor="limegreen", edgecolor="black", label="Exit node"),
        Patch(facecolor="red", edgecolor="darkred", label="Blocked node"),
        Line2D(
            [0], [0],
            marker="o", color="w", label="Leader",
            markerfacecolor=type_to_color["leader"],
            markeredgecolor="black", markersize=8,
        ),
        Line2D(
            [0], [0],
            marker="o", color="w", label="Follower",
            markerfacecolor=type_to_color["follower"],
            markeredgecolor="black", markersize=8,
        ),
        Line2D(
            [0], [0],
            marker="o", color="w", label="Normal",
            markerfacecolor=type_to_color["normal"],
            markeredgecolor="black", markersize=8,
        ),
        Line2D(
            [0], [0],
            marker="o", color="w", label="Panic",
            markerfacecolor=type_to_color["panic"],
            markeredgecolor="black", markersize=8,
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9)

    # --- update function for animation ---

    def update(frame: int):
        sim.step()

        xs, ys, cs = get_agent_xy_and_colors()
        agent_scat.set_offsets(np.column_stack([xs, ys]))
        agent_scat.set_facecolors(cs)

        density_mat = sim.get_density_matrix()
        density_img.set_data(density_mat)

        # update exits & blocked nodes (in case of dynamic events)
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
    plt.show()
