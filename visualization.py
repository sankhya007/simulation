# visualization.py

import random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from environment import EnvironmentGraph
from simulation import CrowdSimulation
from config import GRID_WIDTH, GRID_HEIGHT, NUM_AGENTS, MAX_STEPS, SEED


def run_visual_simulation():
    random.seed(SEED)

    env = EnvironmentGraph(GRID_WIDTH, GRID_HEIGHT)
    sim = CrowdSimulation(env, NUM_AGENTS)

    pos = {n: env.get_pos(n) for n in env.graph.nodes()}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Graph-based Crowd Simulation")
    ax.set_xlim(-1, env.width)
    ax.set_ylim(-1, env.height)
    ax.set_aspect("equal")

    # background grid
    for (u, v) in env.graph.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], linewidth=0.3, alpha=0.4)

    xs, ys = sim.get_agent_positions()
    agent_scat = ax.scatter(xs, ys, s=40)

    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="w", alpha=0.7),
    )

    def update(frame):
        sim.step()
        xs, ys = sim.get_agent_positions()
        agent_scat.set_offsets(list(zip(xs, ys)))

        info_text.set_text(
            f"Step: {sim.time_step}\n"
            f"Agents: {len(sim.agents)}\n"
            f"Total collisions: {sim.total_collisions}"
        )
        return agent_scat, info_text

    anim = FuncAnimation(fig, update, frames=MAX_STEPS, interval=50, blit=False)

    # Uncomment to save MP4 for presentation
    # anim.save("crowd_simulation.mp4", fps=20, dpi=150)

    plt.tight_layout()
    plt.show()

    sim.summary()
    show_density_heatmap(sim)


def show_density_heatmap(sim: CrowdSimulation):
    density = sim.get_density_matrix()

    plt.figure(figsize=(6, 4))
    plt.imshow(density, origin="lower", interpolation="nearest")
    plt.colorbar(label="Visit count")
    plt.title("Crowd Density Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()
