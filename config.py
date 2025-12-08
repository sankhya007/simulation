# config.py

# === Grid / simulation size (fallback for synthetic maps) ===
GRID_WIDTH = 15
GRID_HEIGHT = 10

# Will be overridden by scenarios
NUM_AGENTS = 35
MAX_STEPS = 400

SEED = 42

# === Map source (NEW) =======================================

# MAP_MODE controls how the environment is built:
# - "synthetic": generate a simple open grid (original behaviour)
# - "raster":    load from PNG/JPG floorplan
# - "dxf":       load from DXF CAD floorplan
MAP_MODE = "synthetic"  # "synthetic" | "raster" | "dxf"

# Path to the floorplan file when MAP_MODE != "synthetic"
MAP_FILE = "maps/examples/mall_floorplan.png"  # or mall_floorplan.dxf

# Raster floorplan settings (for PNG/JPG)
# Conventions:
# - walls  : near black (0, 0, 0)
# - walk   : white / light
# - exits  : near bright green (0, 255, 0)
RASTER_DOWNSCALE_FACTOR = 4   # >1 to shrink big images
RASTER_WALL_THRESHOLD = 40    # intensity threshold for black
RASTER_EXIT_GREEN_MIN = 180   # minimum G value to consider as exit

# DXF settings (for CAD floorplans)
# Conventions:
# - WALL layer: walls as LINE or LWPOLYLINE
# - EXIT_DOOR layer: doors that act as exits
DXF_GRID_WIDTH = 60
DXF_GRID_HEIGHT = 40
DXF_WALL_LAYER = "WALL"
DXF_EXIT_LAYER = "EXIT_DOOR"
DXF_WALL_DISTANCE_THRESHOLD = 0.6  # how close to wall segment → wall cell
DXF_EXIT_DISTANCE_THRESHOLD = 0.8  # how close to exit segment → exit cell

# === Agent behaviour ========================================

AGENT_REPLAN_PROB = 0.01
COLLISION_DISTANCE = 0.25

PERCEPTION_RADIUS = 3.0
GROUP_SIZE = 4

AGENT_TYPE_SPEEDS = {
    "leader": 1.0,
    "follower": 0.9,
    "normal": 0.8,
    "panic": 1.0,
}

DENSITY_THRESHOLD = 2
GLOBAL_DENSITY_REPLAN_THRESHOLD = 2.0

# === Environment / edges ====================================

EDGE_BASE_CAPACITY = 5

# === Dynamic scenario controls ==============================

DYNAMIC_BLOCKS_ENABLED = False
BLOCK_NODE_EVERY_N_STEPS = 40
BLOCK_NODE_PROB = 0.4

DYNAMIC_EXITS_ENABLED = False
EXIT_TOGGLE_EVERY_N_STEPS = 80

# === Evacuation mode ========================================

EVACUATION_MODE = False

# Default scenario name (used if no CLI arg)
DEFAULT_SCENARIO_NAME = "normal"
