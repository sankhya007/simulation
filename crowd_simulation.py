"""
Graph-based Crowd Simulation Prototype
--------------------------------------
- Environment: grid world represented as a NetworkX graph
- Agents: move from random start to random goal using shortest paths
- Interaction: simple conflict resolution if multiple agents want same node
- Visualization: animated 2D plot with matplotlib
- Metrics: travel time, collisions, node visit counts (for density)

Requirements:
    pip install networkx matplotlib numpy
"""

import random
import math
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =========================
# CONFIGURATION
# =========================

GRID_WIDTH = 15          # number of columns
GRID_HEIGHT = 10         # number of rows
NUM_AGENTS = 35          # how many agents in the crowd
MAX_STEPS = 400          # how many simulation steps
SEED = 42                # random seed for reproducibility

AGENT_REPLAN_PROB = 0.01  # small chance to pick a new random goal
COLLISION_DISTANCE = 0.25  # distance below which we count a collision


# =========================
# ENVIRONMENT
# =========================

class EnvironmentGraph:
    """
    Simple grid environment stored as a graph.
    Nodes: (x, y) integer grid positions
    Edges: 4-neighbour connectivity with weight = 1
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.graph = nx.Graph()
        self._build_grid()

    def _build_grid(self):
        # Add nodes with coordinates
        for y in range(self.height):
            for x in range(self.width):
                self.graph.add_node((x, y), pos=(x, y))

        # Add edges (4-neighbour connectivity)
        for y in range(self.height):
            for x in range(self.width):
                if x + 1 < self.width:
                    self.graph.add_edge((x, y), (x + 1, y), weight=1.0)
                if y + 1 < self.height:
                    self.graph.add_edge((x, y), (x, y + 1), weight=1.0)

    def get_random_node(self):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        return (x, y)

    def get_pos(self, node):
        return self.graph.nodes[node]["pos"]

    def shortest_path(self, start, goal):
        # Using A* for pathfinding (same as Dijkstra here because heuristic is 0)
        try:
            path = nx.astar_path(
                self.graph, start, goal,
                heuristic=lambda a, b: 0,
                weight="weight"
            )
        except nx.NetworkXNoPath:
            path = [start]
        return path


# =========================
# AGENT
# =========================

class Agent:
    """
    Agent that moves along a precomputed path of graph nodes.
    """
    _id_counter = 0

    def __init__(self, env: EnvironmentGraph):
        self.env = env
        self.id = Agent._id_counter
        Agent._id_counter += 1

        self.current_node = env.get_random_node()
        self.goal_node = env.get_random_node()
        while self.goal_node == self.current_node:
            self.goal_node = env.get_random_node()

        self.path = self.env.shortest_path(self.current_node, self.goal_node)
        self.path_index = 0  # index into self.path

        self.finished = False
        self.steps_taken = 0
        self.collisions = 0

    def choose_new_goal(self):
        self.goal_node = self.env.get_random_node()
        while self.goal_node == self.current_node:
            self.goal_node = self.env.get_random_node()
        self.path = self.env.shortest_path(self.current_node, self.goal_node)
        self.path_index = 0
        self.finished = False

    def desired_next_node(self):
        """
        Returns the node this agent wants to move to this step.
        If already at goal, may choose a new goal.
        """
        # If at end of path, mark finished
        if self.path_index >= len(self.path) - 1:
            self.finished = True

        # Occasionally pick a new random goal to keep things dynamic
        if self.finished and random.random() < AGENT_REPLAN_PROB:
            self.choose_new_goal()

        if self.finished:
            return self.current_node  # stay in place

        # Next node along route
        return self.path[self.path_index + 1]

    def move_to(self, node):
        if node != self.current_node:
            self.current_node = node
            self.path_index += 1
            self.steps_taken += 1

    def get_position(self):
        return self.env.get_pos(self.current_node)


# =========================
# SIMULATION
# =========================

class CrowdSimulation:
    """
    Manages agents, environment, and metrics.
    """
    def __init__(self, env: EnvironmentGraph, num_agents: int):
        self.env = env
        self.agents = [Agent(env) for _ in range(num_agents)]
        self.time_step = 0

        # Metrics
        self.node_visit_counts = defaultdict(int)
        self.total_collisions = 0

        # Record initial visits
        for agent in self.agents:
            self.node_visit_counts[agent.current_node] += 1

    def step(self):
        """
        One simulation tick:
        - each agent picks desired next node
        - conflicts resolved (two agents wanting same tile)
        - agents moved
        - collisions measured
        """
        self.time_step += 1

        # 1) Each agent chooses desired next node
        desires = {}
        for agent in self.agents:
            desired = agent.desired_next_node()
            desires[agent.id] = desired

        # 2) Conflict resolution: if multiple agents want same node,
        #    randomly allow only one, others stay.
        node_to_agents = defaultdict(list)
        for agent_id, node in desires.items():
            node_to_agents[node].append(agent_id)

        allowed_moves = {}
        for node, agent_ids in node_to_agents.items():
            if len(agent_ids) == 1:
                allowed_moves[agent_ids[0]] = node
            else:
                # One winner, others stay put
                winner = random.choice(agent_ids)
                allowed_moves[winner] = node
                for loser in agent_ids:
                    if loser != winner:
                        allowed_moves[loser] = None  # stay

        # 3) Apply movements
        for agent in self.agents:
            target = allowed_moves.get(agent.id, None)
            if target is None:
                # staying
                continue
            agent.move_to(target)
            self.node_visit_counts[agent.current_node] += 1

        # 4) Count collisions (agents too close in continuous space)
        self._update_collisions()

    def _update_collisions(self):
        positions = [agent.get_position() for agent in self.agents]
        n = len(positions)
        step_collisions = 0
        for i in range(n):
            x1, y1 = positions[i]
            for j in range(i + 1, n):
                x2, y2 = positions[j]
                dist = math.dist((x1, y1), (x2, y2))
                if dist < COLLISION_DISTANCE:
                    step_collisions += 1
                    self.agents[i].collisions += 1
                    self.agents[j].collisions += 1
        self.total_collisions += step_collisions

    # Helper methods for visualization / metrics
    def get_agent_positions(self):
        xs, ys = [], []
        for agent in self.agents:
            x, y = agent.get_position()
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def get_density_matrix(self):
        """
        Convert node visit counts into a 2D array for heatmap.
        """
        mat = np.zeros((self.env.height, self.env.width))
        for (x, y), count in self.node_visit_counts.items():
            mat[y, x] = count
        return mat

    def summary(self):
        """
        Print simple metrics at the end of the simulation.
        """
        print("\n=== Simulation Summary ===")
        print(f"Time steps: {self.time_step}")
        print(f"Total collisions: {self.total_collisions}")

        step_counts = [a.steps_taken for a in self.agents]
        print(f"Average steps taken per agent: {np.mean(step_counts):.2f}")
        print(f"Max steps taken by an agent:   {np.max(step_counts)}")
        print(f"Min steps taken by an agent:   {np.min(step_counts)}")

        coll_counts = [a.collisions for a in self.agents]
        print(f"Average collisions per agent:  {np.mean(coll_counts):.2f}")


# =========================
# VISUALIZATION
# =========================

def run_visual_simulation():
    random.seed(SEED)
    np.random.seed(SEED)

    env = EnvironmentGraph(GRID_WIDTH, GRID_HEIGHT)
    sim = CrowdSimulation(env, NUM_AGENTS)

    # Get graph positions for drawing edges
    pos = nx.get_node_attributes(env.graph, "pos")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Graph-based Crowd Simulation")
    ax.set_xlim(-1, env.width)
    ax.set_ylim(-1, env.height)
    ax.set_aspect("equal")

    # Draw grid edges lightly as background
    for (u, v) in env.graph.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], linewidth=0.3, alpha=0.4)

    # Agent scatter plot (updated each frame)
    xs, ys = sim.get_agent_positions()
    agent_scat = ax.scatter(xs, ys, s=40)

    # Text annotation to show time step and collisions
    info_text = ax.text(
        0.02, 0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="w", alpha=0.7)
    )

    def update(frame):
        # Advance simulation one step
        sim.step()
        xs, ys = sim.get_agent_positions()
        agent_scat.set_offsets(np.c_[xs, ys])

        info_text.set_text(
            f"Step: {sim.time_step}\n"
            f"Agents: {len(sim.agents)}\n"
            f"Total collisions: {sim.total_collisions}"
        )
        return agent_scat, info_text

    anim = FuncAnimation(fig, update, frames=MAX_STEPS, interval=50, blit=False)

    # Optional: to save as MP4 for presentation, uncomment:
    # anim.save('crowd_simulation.mp4', fps=20, dpi=150)

    plt.tight_layout()
    plt.show()

    # After visualization, we can print metrics and a density heatmap
    sim.summary()
    show_density_heatmap(sim)


def show_density_heatmap(sim: CrowdSimulation):
    """
    Simple density heatmap of how often agents visited each grid cell.
    Great for including as a figure in the report.
    """
    density = sim.get_density_matrix()

    plt.figure(figsize=(6, 4))
    plt.imshow(
        density,
        origin="lower",
        interpolation="nearest"
    )
    plt.colorbar(label="Visit count")
    plt.title("Crowd Density Heatmap (Node Visit Counts)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    run_visual_simulation()
