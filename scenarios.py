# scenarios.py
"""
Scenario presets and helpers for configuring EnvironmentGraph instances.

Provides:
 - SCENARIO_PRESETS: mapping of names -> simple config dict
 - configure_environment_for_scenario(name, env): apply changes to an EnvironmentGraph
 - configure_environment_for_active_scenario(env): convenience wrapper used by visualization.py
 - load_and_apply_scenario(name): builds environment (grid or from map) and returns it
"""

from typing import Dict, Any, Optional, Tuple

import config
from maps import load_layout_matrix_from_config
from environment import EnvironmentGraph
from maps.map_loader import load_mapmeta_from_config

# Simple scenario presets. Each entry contains optional hooks / params that will be
# applied to the EnvironmentGraph instance by configure_environment_for_scenario.
SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "normal": {"num_agents": config.NUM_AGENTS, "evacuation": False},
    "high_density": {"num_agents": max(config.NUM_AGENTS, 300), "evacuation": False},
    "blocked": {"num_agents": config.NUM_AGENTS, "evacuation": False, "dynamic_blocks": True},
    "evacuation": {"num_agents": config.NUM_AGENTS, "evacuation": True},
    # Floorplan-oriented scenarios (aliases)
    "floorplan_image_normal": {"num_agents": config.NUM_AGENTS, "evacuation": False},
    "floorplan_image_evacuation": {"num_agents": config.NUM_AGENTS, "evacuation": True},
    "floorplan_cad_evac": {"num_agents": config.NUM_AGENTS, "evacuation": True},
}


def configure_environment_for_scenario(name: str, env: EnvironmentGraph, **overrides) -> None:
    """
    Apply scenario-specific configuration to an EnvironmentGraph instance.

    - name: scenario name in SCENARIO_PRESETS
    - env: EnvironmentGraph to modify in-place
    - overrides: optional params (e.g. num_agents, evacuation, dynamic_blocks)
    """
    preset = SCENARIO_PRESETS.get(name)
    if preset is None:
        raise ValueError(f"Unknown scenario '{name}'")

    # merge
    cfg = dict(preset)
    cfg.update(overrides)

    # Example effects:
    # - Evacuation mode: mark env.evacuate_mode or set agent defaults; we set config.EVACUATION_MODE if present
    if cfg.get("evacuation"):
        try:
            import config as _conf

            _conf.EVACUATION_MODE = True
        except Exception:
            pass

    # - dynamic blocks toggle (the EnvironmentGraph should check this flag)
    if cfg.get("dynamic_blocks"):
        try:
            import config as _conf

            _conf.DYNAMIC_BLOCKS_ENABLED = True
        except Exception:
            pass

    # - You may want to set other per-env flags here (e.g., create initial blocked nodes)
    # If the preset carries explicit metadata to apply on env (not used by default), do it:
    env.scenario_name = name  # store name for downstream code / logging


def configure_environment_for_active_scenario(env, preset):
    """
    Apply scenario settings to the environment.
    """
    # Example (adjust based on your preset structure)
    if "evacuation_mode" in preset:
        config.EVACUATION_MODE = preset["evacuation_mode"]

    if "agents" in preset:
        config.NUM_AGENTS = preset["agents"]

    if "blocked_nodes" in preset:
        for n in preset["blocked_nodes"]:
            env.block_node(n)

    if "exits" in preset:
        for n in preset["exits"]:
            env.mark_exit(n)


def load_and_apply_scenario(name: str):
    """
    Loads MapMeta → builds EnvironmentGraph → applies scenario overrides.
    Returns: (env, mapmeta)
    """
    if name not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario '{name}'")

    preset = SCENARIO_PRESETS[name]

    # Always load mapmeta FIRST
    mm = load_mapmeta_from_config()
    layout = mm.layout
    grid_w, grid_h = mm.grid_shape

    # Build env using world coords
    env = EnvironmentGraph(width=grid_w, height=grid_h, layout_matrix=layout, mapmeta=mm)

    # Apply scenario modifiers (speed, exits, density, evacuation, blocked nodes, etc.)
    configure_environment_for_active_scenario(env, preset)

    return env, mm
