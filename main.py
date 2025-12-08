# main.py

import sys

import config
from scenarios import load_and_apply_scenario
from visualization import run_visual_simulation
from maps import load_layout_matrix_from_config
from environment import EnvironmentGraph


def main():
    # 1. Scenario selection from CLI
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
    else:
        scenario_name = config.DEFAULT_SCENARIO_NAME

    try:
        scenario = load_and_apply_scenario(scenario_name)
    except ValueError as e:
        print(e)
        print("Falling back to 'normal' scenario.")
        scenario = load_and_apply_scenario("normal")

    print(f"Running scenario : {scenario.name}")
    print(f"Description      : {scenario.description}")
    print(f"Map mode         : {config.MAP_MODE}")
    print(f"Map file         : {config.MAP_FILE if config.MAP_MODE != 'synthetic' else 'N/A'}")
    print(f"Num agents       : {config.NUM_AGENTS}")
    print(f"Evacuation mode  : {config.EVACUATION_MODE}")
    print()

    # 2. Build environment either from synthetic grid or from layout matrix
    layout = load_layout_matrix_from_config()
    if layout is None:
        # original behaviour: open grid
        env = EnvironmentGraph(config.GRID_WIDTH, config.GRID_HEIGHT)
    else:
        env = EnvironmentGraph(width=0, height=0, layout_matrix=layout)

    # 3. Run visual simulation (visualization.py will construct CrowdSimulation)
    run_visual_simulation(env)


if __name__ == "__main__":
    main()
