from environment import EnvironmentGraph
from simulation import CrowdSimulation


def test_simulation_step_basic():
    # small grid
    env = EnvironmentGraph(width=6, height=4)
    sim = CrowdSimulation(env, num_agents=8)

    initial_time = sim.time_step
    sim.step()
    assert sim.time_step == initial_time + 1

    # After a step, node visit counts should exist and be non-zero for some nodes
    counts = sim.node_visit_counts
    assert isinstance(counts, dict)
    assert len(counts) > 0

    # run a few more steps to ensure no exceptions and time advances
    for _ in range(5):
        sim.step()
    assert sim.time_step >= initial_time + 6
