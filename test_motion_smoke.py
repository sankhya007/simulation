# test_motion_smoke.py
import config
import random
from maps.map_loader import load_mapmeta_from_config
from environment import EnvironmentGraph
from simulation import CrowdSimulation

def smoke_test(motion_model="graph", agents=20, steps=10, seed=123):
    print(f"[smoke] motion_model={motion_model} agents={agents} steps={steps}")
    config.MOTION_MODEL = motion_model
    random.seed(seed)

    mm = load_mapmeta_from_config()
    env = EnvironmentGraph(width=mm.grid_shape[0], height=mm.grid_shape[1], layout_matrix=mm.layout, mapmeta=mm, graph_type="grid")
    sim = CrowdSimulation(env, num_agents=agents)

    # print first 6 agent ids + pos
    print("Initial positions (id: x,y):")
    for a in sim.agents[:6]:
        print(f"  {a.id}: {a.get_position()}  node={a.current_node}")

    for t in range(steps):
        sim.step()

    print("\nAfter steps:")
    for a in sim.agents[:6]:
        print(f"  {a.id}: {a.get_position()}  node={a.current_node}  collided={a.collisions}")

    print("\nSummary:")
    sim.summary()

if __name__ == "__main__":
    # run three quick smoke tests
    smoke_test("graph", agents=30, steps=12)
