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
MAP_MODE = "raster"                              # "grid" | "raster" | "dxf"
MAP_FILE = "maps/examples/example_floorplan.png"    # <--- adjust filename if needed

# =========================
# Raster floorplan parameters
# =========================
# Convention:
#   - Walls/obstacles  : dark/black
#   - Walkable area    : light/white
#   - Exits/doors      : bright green (#00FF00) or similar
RASTER_DOWNSCALE_FACTOR = 5    # shrink big images so grid is manageable

# Grayscale thresholds (0–255)
RASTER_WALL_MAX_LUMA = 170
RASTER_WALKABLE_MIN_LUMA = 200

# Alias values used by raster_loader.py
RASTER_WALL_THRESHOLD = RASTER_WALL_MAX_LUMA
RASTER_EXIT_GREEN_MIN = RASTER_WALKABLE_MIN_LUMA

# Optional explicit colors
RASTER_EXIT_COLOR_RGB = (0, 255, 0)          # green exits
RASTER_WALL_COLOR_RGB = (0, 0, 0)           # black walls
RASTER_WALKABLE_COLOR_RGB = (255, 255, 255) # white walkable

# =========================
# DXF / CAD parameters
# =========================
# Your current DWG/DXF has:
#   - walls on layer "WALL"
#   - generic stuff on layer "0"
# We tell the loader to treat only "WALL" as blocking.
DXF_GRID_WIDTH = 110
DXF_GRID_HEIGHT = 70

# Multiple possible wall / exit layers (for future flexibility)
DXF_WALL_LAYERS = ["0"]          # only your white walls
DXF_EXIT_LAYERS = ["WALL"]          # create this layer & draw short lines at exits

# These are *scale factors* relative to one grid cell size (NOT drawing units).
# 0.4 means "about 40% of a cell from a wall line".
DXF_WALL_DISTANCE_THRESHOLD = 0.4
DXF_EXIT_DISTANCE_THRESHOLD = 0.4

# =========================
# Misc / plotting
# =========================
# You can add flags like:
# SHOW_TRAILS = False
# SAVE_ANIMATION = False


# =========================
# Performance / model detail switches
# =========================
# If False, skip expensive edge congestion weighting and use base distances.
ENABLE_CONGESTION_WEIGHTS = False

# If True, skip dynamic obstacles & exits (faster, simpler behaviour)
ENABLE_DYNAMIC_EVENTS = False   # we'll use this in simulation.py
