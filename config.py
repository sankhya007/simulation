# config.py

# === Grid / simulation size ===
GRID_WIDTH = 15
GRID_HEIGHT = 10

# Will be overridden by scenarios
NUM_AGENTS = 35
MAX_STEPS = 400

SEED = 42

# === Agent behaviour ===
AGENT_REPLAN_PROB = 0.01
COLLISION_DISTANCE = 0.25

PERCEPTION_RADIUS = 3.0     # how far agents "see" group / congestion
GROUP_SIZE = 4              # leader + followers

AGENT_TYPE_SPEEDS = {
    "leader": 1.0,
    "follower": 0.9,
    "normal": 0.8,
    "panic": 1.0,
}

DENSITY_THRESHOLD = 2
GLOBAL_DENSITY_REPLAN_THRESHOLD = 2.0   # avg agents per occupied node

# === Environment / edges ===
EDGE_BASE_CAPACITY = 5   # comfortable edge capacity for congestion cost

# === Dynamic scenario controls (blocked paths / exits) ===
DYNAMIC_BLOCKS_ENABLED = False
BLOCK_NODE_EVERY_N_STEPS = 40
BLOCK_NODE_PROB = 0.4

DYNAMIC_EXITS_ENABLED = False
EXIT_TOGGLE_EVERY_N_STEPS = 80

# === Evacuation mode ===
# When True:
# - Agents' goals become exits.
# - They do not pick new random goals after reaching exits.
EVACUATION_MODE = False

# Default scenario name (used if no CLI arg)
DEFAULT_SCENARIO_NAME = "normal"
# === Navigation strategy comparison (AI) ===
# Strategies:
#   - "shortest": ignore congestion (pure geometric distance)
#   - "congestion": use dynamic weights (current behaviour)
#   - "safe": avoid dense regions aggressively
#
# NAV_STRATEGY_MODE controls how agents are assigned:
#   - "shortest"    -> all agents shortest-path
#   - "congestion"  -> all agents congestion-aware
#   - "safe"        -> all agents safe
#   - "mixed"       -> use NAV_STRATEGY_MIX fractions
NAV_STRATEGY_MODE = "mixed"  # "shortest", "congestion", "safe", "mixed"

# Fractions used only when NAV_STRATEGY_MODE == "mixed"
NAV_STRATEGY_MIX = {
    "shortest": 0.34,
    "congestion": 0.33,
    "safe": 0.33,
}
