# visualization.py

import random
from PIL import Image
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import agent
from config import RASTER_DOWNSCALE_FACTOR
from typing import Any, Tuple, Optional

import config
from simulation import CrowdSimulation
from environment import EnvironmentGraph
from maps.map_loader import load_mapmeta_from_config
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

# Step 10: collision visualization
SHOW_COLLISIONS = True          # highlight agents involved in collisions
COLLISION_COLOR = "yellow"      # color used to mark colliding agents
COLLISION_SIZE_BOOST = 1.6      # size multiplier for colliding agents


def _get_node_world_pos(env: EnvironmentGraph, node: Tuple[int, int]) -> Tuple[float, float]:
    """
    Return world pos for a node (x,y).
    Preference order:
      - CAD coordinates exposed on node (cad_pos attribute)
      - env.get_cad_pos(node) if available
      - node data 'pos' attribute (world coords)
      - fallback to grid indices as floats
    """
    try:
        data = env.graph.nodes[node]
    except Exception:
        # node not present or env not networkx graph-like
        try:
            # try env.get_cad_pos
            cad = getattr(env, "get_cad_pos", None)
            if callable(cad):
                cx, cy = env.get_cad_pos(node)
                if cx is not None and cy is not None:
                    return float(cx), float(cy)
        except Exception:
            pass
        return float(node[0]), float(node[1])

    # 1) node-level cad_pos attribute (set during building)
    cad = data.get("cad_pos")
    if cad and cad != (None, None):
        try:
            return float(cad[0]), float(cad[1])
        except Exception:
            pass

    # 2) environment-level mapping
    try:
        if hasattr(env, "get_cad_pos"):
            cx, cy = env.get_cad_pos(node)
            if cx is not None and cy is not None:
                return float(cx), float(cy)
    except Exception:
        pass

    # 3) world pos attribute
    pos = data.get("pos")
    if pos is not None:
        try:
            return float(pos[0]), float(pos[1])
        except Exception:
            pass

    # fallback to grid coords
    return float(node[0]), float(node[1])


def _get_agent_world_pos(agent: Any, env: EnvironmentGraph) -> Tuple[float, float]:
    """
    Defensive lookup of agent world position.
    Tries (in order):
      - agent.x, agent.y  (world coords)
      - agent.pos (tuple)
      - agent.node (grid node tuple) -> env.node pos (prefers CAD if available)
      - agent.gx, agent.gy or agent.grid_x, agent.grid_y -> resolve via env
      - fallback: (0,0)
    """
    # world coords
    if hasattr(agent, "x") and hasattr(agent, "y"):
        try:
            return float(agent.x), float(agent.y)
        except Exception:
            pass
    if hasattr(agent, "pos"):
        try:
            p = agent.pos
            return float(p[0]), float(p[1])
        except Exception:
            pass
    # node tuple
    if hasattr(agent, "node"):
        try:
            n = agent.node
            return _get_node_world_pos(env, n)
        except Exception:
            pass
    # gx/gy or grid_x/grid_y
    for a_gx, a_gy in (("gx", "gy"), ("grid_x", "grid_y")):
        if hasattr(agent, a_gx) and hasattr(agent, a_gy):
            try:
                gx = int(getattr(agent, a_gx))
                gy = int(getattr(agent, a_gy))
                return _get_node_world_pos(env, (gx, gy))
            except Exception:
                pass
    # fallback
    return 0.0, 0.0


def run_visual_simulation(env):
    """
    Visual simulation runner (matplotlib animation).

    :param env_or_tuple: either EnvironmentGraph or (env, mapmeta) as returned by load_and_apply_scenario
    :param agents: number of agents to spawn (overrides config.NUM_AGENTS if provided)
    :param steps: maximum simulation steps (overrides config.MAX_STEPS if provided)
    :param interval_ms: frame interval in milliseconds
    :param show_trails: if True, show short trails for agents
    """    
    max_steps = config.MAX_STEPS
    interval_ms = 60
    show_trails = False

    if not isinstance(env, EnvironmentGraph):
        raise ValueError("run_visual_simulation expects an EnvironmentGraph or (env, meta) tuple.")

    # Create simulation
    sim = CrowdSimulation(env, config.NUM_AGENTS)

    # Prepare figure and axes
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Crowd Simulation — visual mode")

    # Collect node world coords and types
    xs = []
    ys = []
    node_colors = []
    for node, data in env.graph.nodes(data=True):
        wx, wy = _get_node_world_pos(env, node)
        xs.append(wx)
        ys.append(wy)
        typ = data.get("type", "corridor")
        if data.get("accessibility") == "exit" or typ == "exit":
            node_colors.append((0.0, 0.6, 0.0))  # green
        elif typ == "wall" or data.get("accessibility") == "blocked":
            node_colors.append((0.0, 0.0, 0.0))  # black
        else:
            node_colors.append((0.9, 0.9, 0.9))  # light gray for walkable

    xs = np.array(xs)
    ys = np.array(ys)

    # Draw edges (optional, lightweight)
    # We draw as a LineCollection for performance if many edges
    try:
        from matplotlib.collections import LineCollection

        edge_lines = []
        for u, v in env.graph.edges():
            x1, y1 = _get_node_world_pos(env, u)
            x2, y2 = _get_node_world_pos(env, v)
            edge_lines.append(((x1, y1), (x2, y2)))
        if edge_lines:
            lc = LineCollection(edge_lines, colors=(0.8, 0.8, 0.8), linewidths=0.5, zorder=0)
            ax.add_collection(lc)
    except Exception:
        pass

    # Scatter for nodes (background)
    node_scatter = ax.scatter(xs, ys, s=12, c=node_colors, zorder=1, edgecolors="none")

    # Agent scatter
    agent_scatter = ax.scatter([], [], s=30, c="red", zorder=3)

    # Trails (if enabled)
    if show_trails:
        trails_len = 8
        trails = [ax.plot([], [], lw=1, alpha=0.8, zorder=2)[0] for _ in sim.agents]
        trails_data = [[] for _ in sim.agents]
    else:
        trails = []
        trails_data = []

    # Auto-scale view to world extents with margin
    pad = 0.05
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    dx = max_x - min_x if max_x - min_x != 0 else 1.0
    dy = max_y - min_y if max_y - min_y != 0 else 1.0
    ax.set_xlim(min_x - dx * pad, max_x + dx * pad)
    ax.set_ylim(min_y - dy * pad, max_y + dy * pad)

    # Text for status
    status_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.6),
    )

    step_counter = {"t": 0}

    def _update(frame):
        # Run simulation step (stop if reached max_steps)
        if step_counter["t"] >= max_steps:
            ani.event_source.stop()
            return (agent_scatter,)

        sim.step()

        # Gather agent positions defensively
        agent_positions_x = []
        agent_positions_y = []
        for i, agent in enumerate(sim.agents):
            wx, wy = _get_agent_world_pos(agent, env)
            agent_positions_x.append(wx)
            agent_positions_y.append(wy)

            if show_trails:
                trails_data[i].append((wx, wy))
                if len(trails_data[i]) > trails_len:
                    trails_data[i].pop(0)
                xs_tr, ys_tr = zip(*trails_data[i]) if trails_data[i] else ([], [])
                trails[i].set_data(xs_tr, ys_tr)

        if agent_positions_x:
            agent_scatter.set_offsets(np.column_stack([agent_positions_x, agent_positions_y]))
        else:
            agent_scatter.set_offsets(np.empty((0, 2)))

        step_counter["t"] += 1
        status_text.set_text(f"step: {step_counter['t']}  agents: {len(sim.agents)}")
        return (agent_scatter, status_text) + tuple(trails)

    ani = animation.FuncAnimation(fig, _update, interval=interval_ms, blit=False)

    plt.show()


def build_world_to_screen(env, width=900, height=700, margin=20):
    """
    Computes a transform that maps world coords (from MapMeta) to screen coords.
    Returns: (transform_fn, screen_w, screen_h)
    """

    xs, ys = [], []
    for _, data in env.graph.nodes(data=True):
        x, y = data["pos"]
        xs.append(x)
        ys.append(y)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    world_w = max_x - min_x
    world_h = max_y - min_y

    scale_x = (width - 2 * margin) / world_w if world_w else 1.0
    scale_y = (height - 2 * margin) / world_h if world_h else 1.0
    scale = min(scale_x, scale_y)  # uniform scaling

    def world_to_screen(wx, wy):
        sx = margin + (wx - min_x) * scale
        sy = margin + (max_y - wy) * scale  # invert Y (screen down, world up)
        return sx, sy

    return world_to_screen, width, height


def run_visual_simulation(env):
    """
    Run the visual crowd simulation on a pre-constructed EnvironmentGraph.

    `env` is created in main.py (grid / raster / dxf).
    """
    random.seed(config.SEED)

    # NOTE: environment is already configured by load_and_apply_scenario().
    # Do NOT re-run configure_environment_for_active_scenario here.

    # Use NUM_AGENTS from *current* config (scenario may have changed it)
    sim = CrowdSimulation(env, config.NUM_AGENTS)

    # Node positions (prefer CAD if available)
    pos = {n: _get_node_world_pos(env, n) for n in env.graph.nodes()}

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
    blocked_xy = np.array([pos[n] for n in blocked_nodes]) if blocked_nodes else np.empty((0, 2))

    # Style
    plt.style.use("dark_background")

    # Figure
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Crowd Simulation on Floorplan Graph", fontsize=14, pad=12)

    # Determine world bounds from node positions (use these for axis limits)
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        wx = max_x - min_x if max_x - min_x != 0 else 1.0
        wy = max_y - min_y if max_y - min_y != 0 else 1.0
        pad = 0.05
        ax.set_xlim(min_x - wx * pad, max_x + wx * pad)
        ax.set_ylim(min_y - wy * pad, max_y + wy * pad)
    else:
        # fallback to grid extents (legacy)
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
    density_img = None
    if SHOW_DENSITY_HEATMAP:
        density_mat = sim.get_density_matrix()

        # Compute extent in world coordinates if mapmeta present
        mapmeta = getattr(env, "_mapmeta", None)
        if mapmeta is not None:
            gw, gh = mapmeta.grid_shape
            # cell size in world units
            minx, maxx, miny, maxy = mapmeta.bbox
            world_w = maxx - minx if maxx - minx != 0 else float(gw)
            world_h = maxy - miny if maxy - miny != 0 else float(gh)
            cell_w = world_w / gw
            cell_h = world_h / gh
            extent = (minx - cell_w / 2, maxx + cell_w / 2, miny - cell_h / 2, maxy + cell_h / 2)
        else:
            # fallback: grid-space extent
            extent = (-0.5, env.width - 0.5, -0.5, env.height - 0.5)

        density_img = ax.imshow(
            density_mat,
            origin="lower",
            extent=extent,
            cmap="Reds",
            alpha=0.4,
            interpolation="nearest",
            zorder=0,
        )
        cbar = fig.colorbar(density_img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cumulative node visits (congestion)")

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
            # Defensive: prefer world coords if agent exposes them; otherwise fall back to node pos
            try:
                # Many agents expose get_position() returning world coords or grid coords
                x, y = agent.get_position()
            except Exception:
                x = getattr(agent, "x", None)
                y = getattr(agent, "y", None)

            # If x,y are None or appear to be grid indices, try to resolve via env.graph
            if x is None or y is None:
                gx = getattr(agent, "gx", None) or getattr(agent, "grid_x", None)
                gy = getattr(agent, "gy", None) or getattr(agent, "grid_y", None)
                if gx is not None and gy is not None:
                    try:
                        wx, wy = _get_node_world_pos(env, (int(gx), int(gy)))
                        x, y = wx, wy
                    except Exception:
                        x, y = 0.0, 0.0
                else:
                    x, y = 0.0, 0.0

            xs.append(x)
            ys.append(y)
            # base color by agent type
            base_color = type_to_color.get(getattr(agent, "agent_type", None), "white")

            # Step 10: collision highlighting
            if SHOW_COLLISIONS and getattr(agent, "_collided_this_step", False):
                cs.append(COLLISION_COLOR)
            else:
                cs.append(base_color)
        if not xs:
            return np.empty((0,)), np.empty((0,)), []
        return np.array(xs), np.array(ys), cs

    xs, ys, cs = get_agent_xy_and_colors()
    
    sizes = []
    for agent in sim.agents:
        if SHOW_COLLISIONS and getattr(agent, "collisions", 0) > 0:
            sizes.append(AGENT_SIZE * COLLISION_SIZE_BOOST)
        else:
            sizes.append(AGENT_SIZE)
    
    agent_scat = ax.scatter(
        xs,
        ys,
        s=sizes,
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
        
        # Step 10: update sizes for collision visualization
        if SHOW_COLLISIONS:
            sizes = []
            for agent in sim.agents:
                if getattr(agent, "collisions", 0) > 0:
                    sizes.append(AGENT_SIZE * COLLISION_SIZE_BOOST)
                else:
                    sizes.append(AGENT_SIZE)
            agent_scat.set_sizes(sizes)


        if SHOW_DENSITY_HEATMAP and density_img is not None:
            density_mat = sim.get_density_matrix()
            density_img.set_data(density_mat)

        # exits & blocked nodes may change in dynamic scenarios
        exit_nodes = get_exit_nodes()
        blocked_nodes = get_blocked_nodes()

        exit_xy = np.array([pos[n] for n in exit_nodes]) if exit_nodes else np.empty((0, 2))
        blocked_xy = (
            np.array([pos[n] for n in blocked_nodes]) if blocked_nodes else np.empty((0, 2))
        )

        exit_scat.set_offsets(exit_xy)
        blocked_scat.set_offsets(blocked_xy)

        info_text.set_text(
            f"Step: {getattr(sim, 'time_step', None)}\n"
            f"Agents: {len(sim.agents)}\n"
            f"Total collisions: {getattr(sim, 'total_collisions', 0)}\n"
            f"Agents w/ collisions: "
            f"{sum(1 for a in sim.agents if getattr(a, 'collisions', 0) > 0)}\n"
            f"Occupied nodes: {len(getattr(sim, 'last_node_occupancy', []))}"
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

    # After simulation, print summary & metrics if available
    try:
        sim.summary()
    except Exception:
        pass

    if config.EVACUATION_MODE:
        try:
            print_evacuation_report(sim)
        except Exception:
            pass

    # Optional post plots (best-effort)
    try:
        show_density_heatmap(sim)
    except Exception:
        pass

    try:
        plot_travel_time_histogram(sim)
    except Exception:
        pass

    try:
        plot_max_density_over_time(sim)
    except Exception:
        pass

    try:
        plot_metrics_by_agent_type(sim)
    except Exception:
        pass

    # NEW: overlay results on the original floorplan (for raster mode) or skip for DXF
    try:
        overlay_results_on_floorplan(sim, env)
    except Exception:
        pass


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
    """
    if not map_path or not os.path.exists(map_path):
        return  # nothing to overlay

    # If simulation environment is DXF-based (has grid_to_cad mapping), skip image overlay.
    env = getattr(sim, "env", None) or getattr(sim, "environment", None) or getattr(sim, "environment_graph", None)
    mapmeta = getattr(env, "_mapmeta", None) if env is not None else None
    if mapmeta is not None:
        # If mapmeta has explicit grid_to_cad/cad_to_grid, this is likely DXF-derived.
        extras = getattr(mapmeta, "extras", {}) or {}
        if callable(getattr(mapmeta, "grid_to_cad", None)) or "grid_to_cad" in extras:
            # DXF overlays should be handled by a different function that plots CAD coords.
            # Skip raster image overlay here to avoid misalignment.
            return

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
    # The grid cell <-> image pixel mapping: each grid cell corresponds to one downsampled pixel.
    # We use the same extent as the density heatmap: (-0.5, env.width-0.5, -0.5, env.height-0.5)
    try:
        env = getattr(sim, "env", None) or getattr(sim, "environment", None) or getattr(sim, "environment_graph", None)
        extent = (-0.5, env.width - 0.5, -0.5, env.height - 0.5)
    except Exception:
        extent = (-0.5, 0.5, -0.5, 0.5)

    # Draw the image under everything
    ax.imshow(img_ds, origin="lower", extent=extent, zorder=-1, interpolation="nearest", alpha=0.9)
