import random
import numpy as np
import config

from simulation import CrowdSimulation
from environment import EnvironmentGraph
from maps.map_loader import load_mapmeta_from_config


def _reset_determinism():
    random.seed(0)
    np.random.seed(0)
    config.SEED = 0
    config.AGENT_REPLAN_PROB = 0.0
    config.ENABLE_DYNAMIC_EVENTS = False
    config.ENABLE_CONGESTION_WEIGHTS = False


# --------------------------------------------------
# FORCE-BASED COLLISION TEST
# --------------------------------------------------
def test_force_collision_simple():
    """
    Two agents overlapping in continuous space should
    generate at least one collision event.
    """
    _reset_determinism()

    config.MOTION_MODEL = "social_force"
    config.MOTION_COLLISION_MODE = "force"

    mm = load_mapmeta_from_config()
    env = EnvironmentGraph(
        width=mm.grid_shape[0],
        height=mm.grid_shape[1],
        layout_matrix=mm.layout,
        mapmeta=mm,
        graph_type="grid",
    )

    sim = CrowdSimulation(env, num_agents=2)
    a0, a1 = sim.agents

    # force overlapping positions
    a0.pos = (5.0, 5.0)
    a1.pos = (5.2, 5.0)  # < physical collision distance

    # zero velocities to prevent instant separation
    a0.vel = (0.0, 0.0)
    a1.vel = (0.0, 0.0)

    sim.step()

    assert (
        len(a0.collision_history) > 0
        or len(a1.collision_history) > 0
    ), "Expected force-based collision to be recorded"


# --------------------------------------------------
# PRIORITY-BASED COLLISION TEST
# --------------------------------------------------
def test_priority_push():
    """
    Two agents attempting to move into the same node
    should be resolved by priority logic.
    """
    _reset_determinism()

    config.MOTION_MODEL = "graph"
    config.MOTION_COLLISION_MODE = "priority"

    mm = load_mapmeta_from_config()
    env = EnvironmentGraph(
        width=mm.grid_shape[0],
        height=mm.grid_shape[1],
        layout_matrix=mm.layout,
        mapmeta=mm,
        graph_type="grid",
    )

    sim = CrowdSimulation(env, num_agents=2)
    a0, a1 = sim.agents

    # place agents on adjacent nodes
    a0.current_node = (5, 5)
    a1.current_node = (6, 5)

    # force both to desire the SAME target
    target = (5, 6)

    def force_target(_state):
        return target

    a0.desired_next_node = force_target
    a1.desired_next_node = force_target

    sim.step()

    moved = [a.current_node == target for a in sim.agents]
    waited = [a.wait_steps > 0 for a in sim.agents]

    assert any(moved), "One agent should have moved into target"
    assert any(waited), "One agent should have been blocked or waited"
