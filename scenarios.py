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


def configure_environment_for_active_scenario(env: EnvironmentGraph) -> None:
    """
    Convenience wrapper used in other modules: apply the scenario currently stored on env,
    or fallback to DEFAULT_SCENARIO_NAME.
    """
    name = getattr(env, "scenario_name", None) or config.DEFAULT_SCENARIO_NAME
    configure_environment_for_scenario(name, env)


def load_and_apply_scenario(name: str, **overrides) -> Tuple[EnvironmentGraph, Dict[str, Any]]:
    """
    Build an EnvironmentGraph according to MAP_MODE (grid / raster / dxf) and apply scenario.
    Returns (env, meta) where meta can contain map loader metadata (e.g., raster extents).
    """
    layout_or_tuple = load_layout_matrix_from_config()

    # Support two possible return types:
    #  - layout (List[List[str]])
    #  - (layout, meta) tuple if loader provides metadata
    meta: Dict[str, Any] = {}
    if isinstance(layout_or_tuple, tuple) and len(layout_or_tuple) == 2:
        layout, meta = layout_or_tuple
    else:
        layout = layout_or_tuple

    if layout is None:
        # synthetic grid
        env = EnvironmentGraph(config.GRID_WIDTH, config.GRID_HEIGHT)
    else:
        # layout provided -> use the matrix to build the env
        # Some EnvironmentGraph constructors expect width=0,height=0 when layout_matrix supplied
        env = EnvironmentGraph(width=0, height=0, layout_matrix=layout)

    # persist meta and scenario name to env for downstream use
    env.map_meta = meta
    env.scenario_name = name

    # Apply scenario changes
    configure_environment_for_scenario(name, env, **overrides)

    return env, meta
