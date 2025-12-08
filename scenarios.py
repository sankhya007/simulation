# scenarios.py

from dataclasses import dataclass
from typing import Optional, Dict

import config
from environment import EnvironmentGraph


@dataclass
class Scenario:
    name: str
    description: str
    num_agents: int
    dynamic_blocks: bool
    dynamic_exits: bool
    block_every_n: int
    block_prob: float
    exit_toggle_every_n: int
    evacuation_mode: bool
    # How to set initial exits on the grid:
    #   "none"    -> don't change exits (use map / layout defaults)
    #   "borders" -> mark all border cells as exits
    #   "corners" -> only corners as exits
    initial_exit_strategy: str


# --- Scenario presets -------------------------------------------------------

SCENARIO_PRESETS: Dict[str, Scenario] = {
    # 1. Normal navigation
    "normal": Scenario(
        name="normal",
        description="Medium crowd, no dynamic obstacles or exits.",
        num_agents=60,
        dynamic_blocks=False,
        dynamic_exits=False,
        block_every_n=40,
        block_prob=0.4,
        exit_toggle_every_n=80,
        evacuation_mode=False,
        initial_exit_strategy="none",
    ),

    # 2. High-density crowd
    "high_density": Scenario(
        name="high_density",
        description="High-density crowd to stress-test congestion handling.",
        num_agents=300,
        dynamic_blocks=False,
        dynamic_exits=False,
        block_every_n=40,
        block_prob=0.4,
        exit_toggle_every_n=80,
        evacuation_mode=False,
        initial_exit_strategy="none",
    ),

    # 3. Blocked paths / dynamic obstacles
    "blocked": Scenario(
        name="blocked",
        description="Dynamic blocked nodes forcing frequent rerouting.",
        num_agents=120,
        dynamic_blocks=True,
        dynamic_exits=False,
        block_every_n=25,
        block_prob=0.6,
        exit_toggle_every_n=80,
        evacuation_mode=False,
        initial_exit_strategy="none",
    ),

    # 4. Emergency evacuation
    "evacuation": Scenario(
        name="evacuation",
        description="Evacuation to border exits; agents head to nearest exit.",
        num_agents=150,
        dynamic_blocks=True,
        dynamic_exits=False,   # keep exits stable during evacuation
        block_every_n=35,
        block_prob=0.3,
        exit_toggle_every_n=80,
        evacuation_mode=True,
        initial_exit_strategy="borders",   # mark border as exits
    ),
}

_ACTIVE_SCENARIO: Optional[Scenario] = None


# --- Scenario management ----------------------------------------------------

def load_and_apply_scenario(name: str) -> Scenario:
    """
    Select a scenario preset by name and apply it to config globals.
    This is called from main.py and experiment.py.
    """
    global _ACTIVE_SCENARIO
    key = name.lower()

    if key not in SCENARIO_PRESETS:
        raise ValueError(
            f"Unknown scenario '{name}'. "
            f"Available: {', '.join(sorted(SCENARIO_PRESETS.keys()))}"
        )

    scenario = SCENARIO_PRESETS[key]
    _ACTIVE_SCENARIO = scenario

    # Apply to config: this is why high_density will really use 300 agents, etc.
    config.NUM_AGENTS = scenario.num_agents
    config.DYNAMIC_BLOCKS_ENABLED = scenario.dynamic_blocks
    config.DYNAMIC_EXITS_ENABLED = scenario.dynamic_exits
    config.BLOCK_NODE_EVERY_N_STEPS = scenario.block_every_n
    config.BLOCK_NODE_PROB = scenario.block_prob
    config.EXIT_TOGGLE_EVERY_N_STEPS = scenario.exit_toggle_every_n
    config.EVACUATION_MODE = scenario.evacuation_mode

    return scenario


def get_active_scenario() -> Optional[Scenario]:
    return _ACTIVE_SCENARIO


def configure_environment_for_active_scenario(env: EnvironmentGraph):
    """
    Apply environment-level settings for the active scenario
    (initial exits etc.). Called from visualization.py.
    """
    scenario = _ACTIVE_SCENARIO
    if scenario is None:
        return

    # If we don't want to touch exits, just return.
    if scenario.initial_exit_strategy == "none":
        return

    if scenario.initial_exit_strategy == "borders":
        # Mark ALL border nodes as exits (works both for grid and map-based envs)
        for x in range(env.width):
            env.mark_exit((x, 0))
            env.mark_exit((x, env.height - 1))
        for y in range(env.height):
            env.mark_exit((0, y))
            env.mark_exit((env.width - 1, y))

    elif scenario.initial_exit_strategy == "corners":
        corners = [
            (0, 0),
            (0, env.height - 1),
            (env.width - 1, 0),
            (env.width - 1, env.height - 1),
        ]
        for node in corners:
            env.mark_exit(node)


def list_scenarios():
    """Small helper if you want to print available scenarios somewhere."""
    return sorted(SCENARIO_PRESETS.keys())
