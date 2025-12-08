# visualization.py

import random
from PIL import Image
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from config import RASTER_DOWNSCALE_FACTOR

import config
from simulation import CrowdSimulation
from analysis import (
    plot_travel_time_histogram,
    plot_max_density_over_time,
    plot_metrics_by_agent_type,
    print_evacuation_report,
    overlay_results_on_floorplan,

)
from scenarios import configure_environment_for_active_scenario

# ---- visualization knobs ----
SHOW_DENSITY_HEATMAP = True
EDGE_ALPHA = 0.25
AGENT_SIZE = 30


def run_visual_simulation(env):
    """
    Run the visual crowd simulation on a pre-constructed EnvironmentGraph.

    `env` is created in main.py (grid / raster / dxf).
    """
    random.seed(config.SEED)

    # Apply scenario-specific environment configuration (exits, initial blocks, etc.)
    configure_environment_for_active_scenario(env)

    # Use NUM_AGENTS from *current* config (scenario may have changed it)
    sim = CrowdSimulation(env, config.NUM_AGENTS)

    # Node positions
    pos = {n: env.get_pos(n) for n in env.graph.nodes()}

    # Precompute edges as segments for fast drawing
    edge_segments = [(pos[u], pos[v]) for u, v in env.graph.edges()]

    # Helpers
    def get_exit_nodes():
        return [n for n in env.graph.nodes() if env.is_exit(n)]

    def get_blocked_nodes():
        return [n for n, d in env.graph.nodes(data=True) if not env.is_accessible(n)]

    exit_nodes = get_exit_nodes()
    blocked_nodes = get_blocked_nodes()

    exit_xy = np.array([pos[n] for n in exit_nodes]) if exit_nodes else np.empty((0, 2))
    blocked_xy = (
        np.array([pos[n] for n in blocked_nodes]) if blocked_nodes else np.empty((0, 2))
    )

    # Style
    plt.style.use("dark_background")

    # Figure
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Crowd Simulation on Floorplan Graph", fontsize=14, pad=12)
    ax.set_xlim(-1, env.width)
    ax.set_ylim(-1, env.height)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Static edges
    if edge_segments:
        lc = LineCollection(
            edge_segments,
            linewidths=0.4,
            alpha=EDGE_ALPHA,
            zorder=1,
        )
        ax.add_collection(lc)

    # Exits & blocked nodes
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
        s=40,
        marker="X",
        edgecolors="darkred",
        facecolors="red",
        label="Blocked node",
        zorder=3,
    )

    # Density heatmap
    if SHOW_DENSITY_HEATMAP:
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
    else:
        density_img = None

    # Agent colors
    type_to_color = {
        "leader": "tab:blue",
        "follower": "tab:orange",
        "normal": "0.8",
        "panic": "tab:red",
    }

    def get_agent_xy_and_colors():
        xs, ys, cs = [], [], []
        for agent in sim.agents:
            x, y = agent.get_position()
            xs.append(x)
            ys.append(y)
            cs.append(type_to_color.get(agent.agent_type, "white"))
        if not xs:
            return np.empty((0,)), np.empty((0,)), []
        return np.array(xs), np.array(ys), cs

    xs, ys, cs = get_agent_xy_and_colors()
    agent_scat = ax.scatter(
        xs,
        ys,
        s=AGENT_SIZE,
        c=cs,
        edgecolors="black",
        linewidths=0.4,
        zorder=4,
    )

    # Info panel
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
        zorder=5,
    )

    # Legend
    legend_elements = [
        Patch(facecolor="limegreen", edgecolor="black", label="Exit node"),
        Patch(facecolor="red", edgecolor="darkred", label="Blocked node"),
        Line2D([0], [0], marker="o", color="w",
               label="Leader",
               markerfacecolor=type_to_color["leader"],
               markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w",
               label="Follower",
               markerfacecolor=type_to_color["follower"],
               markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w",
               label="Normal",
               markerfacecolor=type_to_color["normal"],
               markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w",
               label="Panic",
               markerfacecolor=type_to_color["panic"],
               markeredgecolor="black", markersize=8),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9)

    # --- animation update ---

    def update(frame):
        sim.step()

        xs, ys, cs = get_agent_xy_and_colors()
        if len(xs) > 0:
            agent_scat.set_offsets(np.column_stack([xs, ys]))
        else:
            agent_scat.set_offsets(np.empty((0, 2)))
        agent_scat.set_facecolors(cs)

        if SHOW_DENSITY_HEATMAP and density_img is not None:
            density_mat = sim.get_density_matrix()
            density_img.set_data(density_mat)

        # exits & blocked nodes may change in dynamic scenarios
        exit_nodes = get_exit_nodes()
        blocked_nodes = get_blocked_nodes()

        exit_xy = np.array([pos[n] for n in exit_nodes]) if exit_nodes else np.empty((0, 2))
        blocked_xy = np.array([pos[n] for n in blocked_nodes]) if blocked_nodes else np.empty((0, 2))

        exit_scat.set_offsets(exit_xy)
        blocked_scat.set_offsets(blocked_xy)

        info_text.set_text(
            f"Step: {sim.time_step}\n"
            f"Agents: {len(sim.agents)}\n"
            f"Total collisions: {sim.total_collisions}\n"
            f"Occupied nodes: {len(sim.last_node_occupancy)}"
        )

        if SHOW_DENSITY_HEATMAP and density_img is not None:
            return agent_scat, density_img, exit_scat, blocked_scat, info_text
        else:
            return agent_scat, exit_scat, blocked_scat, info_text

    anim = FuncAnimation(
        fig,
        update,
        frames=config.MAX_STEPS,
        interval=60,
        blit=False,
    )

    plt.tight_layout()
    plt.show()

    # After simulation, print summary & metrics
    sim.summary()
    if config.EVACUATION_MODE:
        print_evacuation_report(sim)
    show_density_heatmap(sim)
    plot_travel_time_histogram(sim)
    plot_max_density_over_time(sim)
    plot_metrics_by_agent_type(sim)

    # NEW: overlay results on the original floorplan (for raster mode)
    overlay_results_on_floorplan(sim, env)


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


def overlay_results_on_image(sim, ax, map_path: str):
    """
    Draw the original floorplan image under the simulation axes `ax` so
    grid cells align with pixel blocks used to produce the layout.

    Uses the same downscale factor as raster_loader.load_raster_floorplan_to_layout
    so the resulting image aligns exactly with the grid (one image cell -> one grid cell).
    (raster_loader resizes image using Image.NEAREST with factor RASTER_DOWNSCALE_FACTOR).
    See raster_loader.py for the original logic. :contentReference[oaicite:1]{index=1}
    """

    if not map_path or not os.path.exists(map_path):
        return  # nothing to overlay

    # Open original image and downscale same as raster_loader
    img = Image.open(map_path).convert("RGB")
    if RASTER_DOWNSCALE_FACTOR > 1:
        w0, h0 = img.size
        w = max(1, w0 // RASTER_DOWNSCALE_FACTOR)
        h = max(1, h0 // RASTER_DOWNSCALE_FACTOR)
        img_ds = img.resize((w, h), Image.NEAREST)
    else:
        img_ds = img
        w, h = img.size

    # env grid dimensions (should match layout->env mapping)
    env = sim.env  # CrowdSimulation should reference env; adjust if your API differs
    # The grid cell <-> image pixel mapping: each grid cell corresponds to one downsampled pixel.
    # We use the same extent as the density heatmap: (-0.5, env.width-0.5, -0.5, env.height-0.5)
    # so image covers the same coordinate frame as the grid.
    extent = (-0.5, env.width - 0.5, -0.5, env.height - 0.5)

    # Draw the image under everything
    ax.imshow(img_ds, origin="lower", extent=extent, zorder=-1, interpolation="nearest", alpha=0.9)