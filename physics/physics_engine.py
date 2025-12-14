# physics/physics_engine.py

from typing import List, Optional
from .collision_models import (
    BaseCollisionModel,
    ElasticCollisionModel,
)


class PhysicsEngine:
    """
    Central physics orchestrator.
    This is called once per simulation timestep.
    """

    def __init__(
        self,
        collision_model: Optional[BaseCollisionModel] = None,
    ):
        # Default to elastic collisions if none specified
        self.collision_model = collision_model or ElasticCollisionModel()

        # Metrics
        self.total_collisions = 0
        self.collisions_per_step = []

    def step(self, agents: List, dt: float = 1.0):
        """
        Apply physics for one timestep.

        Args:
            agents: list of Agent objects
            dt: timestep duration
        """
        if not agents:
            return 0

        
        # Ensure required fields exist (non-breaking)
        for a in agents:
            if not hasattr(a, "collisions"):
                a.collisions = 0
            if not hasattr(a, "vel"):
                a.vel = (0.0, 0.0)
            if not hasattr(a, "pos"):
                raise RuntimeError("Agent missing 'pos' for physics processing")
            
        # Reset per-step collision flags (visual + logic)
        for a in agents:
            a._collided_this_step = False


        # Resolve collisions
        step_collisions = self.collision_model.resolve(agents, dt)

        # Mark agents involved in collisions this step
        if step_collisions > 0:
            for a in agents:
                if getattr(a, "collisions", 0) > 0:
                    a._collided_this_step = True

        # Dampen velocities to avoid runaway physics
        for a in agents:
            vx, vy = getattr(a, "vel", (0.0, 0.0))
            a.vel = (vx * 0.5, vy * 0.5)
            
        # Record metrics
        self.total_collisions += step_collisions
        self.collisions_per_step.append(step_collisions)

        return step_collisions
