# main.py

import sys

import config
from scenarios import load_and_apply_scenario, list_scenarios
from visualization import run_visual_simulation


def main():
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

    print(f"Running scenario: {scenario.name}")
    print(f"Description    : {scenario.description}")
    print(f"Num agents     : {scenario.num_agents}")
    print(f"Evacuation mode: {scenario.evacuation_mode}")
    print(f"Dynamic blocks : {scenario.dynamic_blocks}")
    print(f"Dynamic exits  : {scenario.dynamic_exits}")
    print()

    run_visual_simulation()


if __name__ == "__main__":
    main()
