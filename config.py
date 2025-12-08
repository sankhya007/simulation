# config.py
"""
Global configuration for the crowd simulation project.

All tunable knobs should live here so experiments are reproducible.
"""

# =========================
# Core grid parameters
# =========================
GRID_WIDTH = 15
GRID_HEIGHT = 10

# These may be overridden by scenarios.py
NUM_AGENTS = 60
MAX_STEPS = 400

SEED = 42

# =========================
# Agent behaviour
# =========================
AGENT_REPLAN_PROB = 0.01        # prob. of random new goal in NON-evacuation mode
COLLISION_DISTANCE = 0.25       # when two agents are "too close"

PERCEPTION_RADIUS = 3.0         # how far agents "see" others / leader / congestion
GROUP_SIZE = 4                  # one leader + followers in a group

AGENT_TYPE_SPEEDS = {
    "leader": 1.0,
    "follower": 0.9,
    "normal": 0.8,
    "panic": 1.0,
}

DENSITY_THRESHOLD = 2           # node-level crowding threshold
GLOBAL_DENSITY_REPLAN_THRESHOLD = 2.0   # avg agents per occupied node

# =========================
# Edge / congestion model
# =========================
EDGE_BASE_CAPACITY = 5          # comfortable edge capacity for congestion

# =========================
# Dynamic environment (blocked paths, exits)
# =========================
DYNAMIC_BLOCKS_ENABLED = False
BLOCK_NODE_EVERY_N_STEPS = 40
BLOCK_NODE_PROB = 0.4

DYNAMIC_EXITS_ENABLED = False
EXIT_TOGGLE_EVERY_N_STEPS = 80

# =========================
# Evacuation mode
# =========================
# When True:
#  - Agents start with goal = nearest exit
#  - They do NOT pick new random goals after reaching exits
EVACUATION_MODE = False

# Default scenario name for main.py if no arg given
DEFAULT_SCENARIO_NAME = "normal"

# =========================
# AI navigation strategies
# =========================
# How agents choose to route:
#   "shortest"   -> pure geometric distance (ignores congestion)
#   "congestion" -> congestion-aware dynamic weights
#   "safe"       -> avoids dense areas more aggressively
#   "mixed"      -> mixture defined by NAV_STRATEGY_MIX
NAV_STRATEGY_MODE = "mixed"   # "shortest" | "congestion" | "safe" | "mixed"

NAV_STRATEGY_MIX = {
    "shortest": 0.34,
    "congestion": 0.33,
    "safe": 0.33,
}

# =========================
# Map / floorplan selection
# =========================
# How we build the environment graph:
#   "grid"   -> simple grid of size GRID_WIDTH x GRID_HEIGHT (no external file)
#   "raster" -> from a PNG/JPG floorplan
#   "dxf"    -> from a CAD DXF file (requires ezdxf & loader implementation)
MAP_MODE = "raster"              # "grid" | "raster" | "dxf"
MAP_FILE = "maps/examples/example_floorplan.png"   # used when MAP_MODE != "grid"

# =========================
# Raster floorplan parameters
# =========================
# How we interpret pixels when MAP_MODE = "raster"
# Convention (can be changed, but must match raster_loader.py):
#   - Walls/obstacles  : dark/black
#   - Walkable area    : light/white
#   - Exits/doors      : bright green (#00FF00) or similar
RASTER_DOWNSCALE_FACTOR = 4    # shrink big images so grid is manageable

# Grayscale thresholds (0–255) – your raster_loader may or may not use them;
# they are here so imports don't fail and code can be extended safely.
RASTER_WALL_MAX_LUMA = 50      # <= 50 => wall
RASTER_WALKABLE_MIN_LUMA = 200 # >= 200 => walkable

# Alias values used by raster_loader.py
RASTER_WALL_THRESHOLD = RASTER_WALL_MAX_LUMA      # RGB < 50 → wall
RASTER_EXIT_GREEN_MIN = RASTER_WALKABLE_MIN_LUMA  # G ≥ 200 → exit

# If your raster_loader expects explicit RGB colors, you can use these:
RASTER_EXIT_COLOR_RGB = (0, 255, 0)      # green exits
RASTER_WALL_COLOR_RGB = (0, 0, 0)        # black walls
RASTER_WALKABLE_COLOR_RGB = (255, 255, 255)  # white walkable

# =========================
# DXF / CAD parameters (future)
# =========================
# These are placeholders so that dxf_loader can read config safely.
DXF_WALL_LAYERS = ["WALL", "WALLS"]
DXF_DOOR_LAYERS = ["DOOR", "DOORS"]
DXF_EXIT_LAYERS = ["EXIT", "EXIT_DOOR", "EXITS"]

# =========================
# Misc / plotting
# =========================
# You can add flags like:
# SHOW_TRAILS = False
# SAVE_ANIMATION = False
