# motion_models.py
"""
Motion models for crowd simulation:
 - GraphMotionModel: adapter that converts a discrete next-node into a short velocity (keeps legacy behaviour)
 - SocialForceModel: simplified Helbing social force continuous model
 - RVOMotionModel: sampling-based ORCA-like avoidance heuristic
"""

from __future__ import annotations
import math
import random
from typing import Protocol, Tuple, Dict, Any, List

Node = Tuple[int, int]
EdgeKey = Tuple[Node, Node]


class MotionState:
    def __init__(
        self,
        time_step: int,
        positions: Dict[int, Tuple[float, float]],
        velocities: Dict[int, Tuple[float, float]],
        node_occupancy: Dict[Node, int],
        group_targets: Dict[int, Node],
        edge_over_capacity: Dict[EdgeKey, bool],
        global_density: float,
    ):
        self.time_step = time_step
        self.positions = positions
        self.velocities = velocities
        self.node_occupancy = node_occupancy
        self.group_targets = group_targets
        self.edge_over_capacity = edge_over_capacity
        self.global_density = global_density


class MotionModel(Protocol):
    def compute_velocity(self, agent: Any, state: MotionState, env: Any) -> Tuple[float, float]:
        ...


# ---------------------------
# Graph adapter (legacy)
# ---------------------------
class GraphMotionModel:
    """
    Convert discrete next-node decisions into a short velocity vector
    so visualization and continuous integration behave the same.
    """

    def __init__(self):
        pass

    def compute_velocity(self, agent, state: MotionState, env):
        # Use agent.desired_next_node to know where agent wants to go
        try:
            desired_node = agent.desired_next_node.__call__(agent._last_decision_state) if hasattr(agent, "_last_decision_state") else None
        except Exception:
            desired_node = None

        if desired_node is None:
            # fallback: follow agent.path if available
            if hasattr(agent, "path") and agent.path and agent.path_index + 1 < len(agent.path):
                desired_node = agent.path[agent.path_index + 1]
            else:
                return (0.0, 0.0)

        # world positions
        try:
            tx, ty = env.get_pos(desired_node)
        except Exception:
            return (0.0, 0.0)

        cx, cy = agent.get_position()
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return (0.0, 0.0)
        pref_speed = getattr(agent, "speed", 0.8)
        vx = dx / dist * pref_speed
        vy = dy / dist * pref_speed
        return (vx, vy)


# ---------------------------
# Social Force Model (Helbing-style)
# ---------------------------
class SocialForceModel:
    def __init__(
        self,
        relaxation_time: float = 0.5,
        repulsion_A: float = 2.0,
        repulsion_B: float = 0.5,
        agent_radius: float = 0.3,
        max_speed: float = 1.5,
    ):
        self.tau = float(relaxation_time)
        self.A = float(repulsion_A)
        self.B = float(repulsion_B)
        self.agent_radius = float(agent_radius)
        self.max_speed = float(max_speed)

    def desired_velocity(self, agent, env):
        # prefer heading toward goal_node's world coords
        try:
            gx, gy = env.get_pos(agent.goal_node)
        except Exception:
            # fallback: remain still
            return (0.0, 0.0)
        cx, cy = agent.get_position()
        dx, dy = gx - cx, gy - cy
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return (0.0, 0.0)
        pref_speed = getattr(agent, "speed", 0.8)
        sp = min(pref_speed, self.max_speed)
        return (dx / d * sp, dy / d * sp)

    def compute_velocity(self, agent, state: MotionState, env):
        # current velocity
        vx, vy = state.velocities.get(agent.id, (0.0, 0.0))
        # driving force to desired velocity
        vx_des, vy_des = self.desired_velocity(agent, env)
        fx = (vx_des - vx) / self.tau
        fy = (vy_des - vy) / self.tau

        # social repulsive forces from other agents
        px, py = agent.get_position()
        eff_r = self.agent_radius * 2.0
        for other_id, (ox, oy) in state.positions.items():
            if other_id == agent.id:
                continue
            dx, dy = px - ox, py - oy
            r = math.hypot(dx, dy)
            if r <= 1e-6:
                continue
            # Helbing repulsion
            fmag = self.A * math.exp((eff_r - r) / self.B)
            fx += fmag * (dx / r)
            fy += fmag * (dy / r)

        # optional wall force if env provides helper (not required)
        try:
            if hasattr(env, "get_nearest_wall_force"):
                wx, wy = env.get_nearest_wall_force((px, py))
                fx += wx
                fy += wy
        except Exception:
            pass

        # integrate: treat mass=1, dt=1 -> velocity += force
        nvx = vx + fx
        nvy = vy + fy

        # clamp speed
        s = math.hypot(nvx, nvy)
        if s > self.max_speed and s > 0:
            nvx *= self.max_speed / s
            nvy *= self.max_speed / s
        return (nvx, nvy)


# ---------------------------
# Simple RVO-like sampler
# ---------------------------
class RVOMotionModel:
    def __init__(self, neighbor_dist: float = 3.0, max_speed: float = 1.5, samples: int = 16):
        self.neighbor_dist = float(neighbor_dist)
        self.max_speed = float(max_speed)
        self.samples = int(samples)

    def preferred_velocity(self, agent, env):
        try:
            gx, gy = env.get_pos(agent.goal_node)
        except Exception:
            return (0.0, 0.0)
        cx, cy = agent.get_position()
        dx, dy = gx - cx, gy - cy
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return (0.0, 0.0)
        pref_speed = getattr(agent, "speed", 0.8)
        sp = min(pref_speed, self.max_speed)
        return (dx / d * sp, dy / d * sp)

    def compute_velocity(self, agent, state: MotionState, env):
        px, py = agent.get_position()
        neighbors = []
        for other_id, (ox, oy) in state.positions.items():
            if other_id == agent.id:
                continue
            dist = math.hypot(px - ox, py - oy)
            if dist <= self.neighbor_dist:
                neighbors.append((other_id, ox, oy, dist))

        pref_v = self.preferred_velocity(agent, env)

        # candidate set: include zero, preferred, and sampled circle velocities
        candidates: List[Tuple[float, float]] = [(0.0, 0.0), pref_v]
        for i in range(self.samples):
            ang = 2.0 * math.pi * (i / max(1, self.samples))
            vx = math.cos(ang) * self.max_speed
            vy = math.sin(ang) * self.max_speed
            # bias toward preferred velocity
            vx = 0.6 * vx + 0.4 * pref_v[0]
            vy = 0.6 * vy + 0.4 * pref_v[1]
            candidates.append((vx, vy))

        best = pref_v
        best_score = float("inf")
        dt = 1.0
        agent_radius = getattr(agent, "radius", 0.3)
        min_sep = agent_radius * 2.0

        for (vx, vy) in candidates:
            # score: deviation from pref + predicted collision penalty
            score = math.hypot(vx - pref_v[0], vy - pref_v[1])
            # predicted separation after dt
            for other_id, ox, oy, _d in neighbors:
                ovx, ovy = state.velocities.get(other_id, (0.0, 0.0))
                rx = (px + vx * dt) - (ox + ovx * dt)
                ry = (py + vy * dt) - (oy + ovy * dt)
                dpred = math.hypot(rx, ry)
                if dpred < min_sep:
                    score += 1e3 * (min_sep - dpred)
                else:
                    score += max(0.0, (min_sep + 0.5) - dpred) * 10.0
            if score < best_score:
                best_score = score
                best = (vx, vy)

        # clamp speed
        sp = math.hypot(best[0], best[1])
        if sp > self.max_speed and sp > 0:
            return (best[0] * self.max_speed / sp, best[1] * self.max_speed / sp)
        return best
