# physics/collision_models.py

import math
from typing import List, Tuple


class BaseCollisionModel:
    """
    Abstract collision model.
    Subclasses must implement resolve().
    """

    def resolve(self, agents: List, dt: float) -> int:
        """
        Resolve collisions among agents.

        Args:
            agents: list of Agent objects (must have pos, radius, vel)
            dt: timestep

        Returns:
            Number of collisions resolved in this step
        """
        raise NotImplementedError


class ElasticCollisionModel(BaseCollisionModel):
    """
    Soft collision model:
    - Agents repel each other elastically when overlapping
    - Causes slowdown and natural jams
    """

    def resolve(self, agents: List, dt: float) -> int:
        collisions = 0

        for i in range(len(agents)):
            a = agents[i]
            ax, ay = a.pos

            for j in range(i + 1, len(agents)):
                b = agents[j]
                bx, by = b.pos

                dx = bx - ax
                dy = by - ay
                dist = math.hypot(dx, dy)
                min_dist = a.radius + b.radius

                if dist < min_dist and dist > 1e-6:
                    collisions += 1

                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist

                    # Push agents apart equally
                    shift = overlap * 0.5
                    a.pos = (ax - nx * shift, ay - ny * shift)
                    b.pos = (bx + nx * shift, by + ny * shift)

                    # Slow down both agents
                    avx, avy = a.vel
                    bvx, bvy = b.vel
                    a.vel = (avx * 0.5, avy * 0.5)
                    b.vel = (bvx * 0.5, bvy * 0.5)

                    a.collisions += 1
                    b.collisions += 1

        return collisions


class HardBlockingCollisionModel(BaseCollisionModel):
    """
    Hard collision model:
    - Agents are not allowed to overlap
    - One agent yields, the other stops
    - Produces strong jams
    """

    def resolve(self, agents: List, dt: float) -> int:
        collisions = 0

        for i in range(len(agents)):
            a = agents[i]
            ax, ay = a.pos

            for j in range(i + 1, len(agents)):
                b = agents[j]
                bx, by = b.pos

                dx = bx - ax
                dy = by - ay
                dist = math.hypot(dx, dy)
                min_dist = a.radius + b.radius

                if dist < min_dist:
                    collisions += 1

                    # Stop both agents
                    a.vel = (0.0, 0.0)
                    b.vel = (0.0, 0.0)

                    a.collisions += 1
                    b.collisions += 1

        return collisions
