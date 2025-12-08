"""
Global configuration for the crowd simulation project.

All tunable knobs live here so experiments are reproducible.
"""

# ============================================================
# CORE GRID PARAMETERS (used when MAP_MODE = "grid")
# ============================================================
GRID_WIDTH = 15
GRID_HEIGHT = 10

NUM_AGENTS = 60
MAX_STEPS = 400
SEED = 42

# ============================================================
# AGENT BEHAVIOUR
# ============================================================
AGENT_REPLAN_PROB = 0.01        # probability an agent picks a new random goal
COLLISION_DISTANCE = 0.25       # "too close" for collision checks

PERCEPTION_RADIUS = 3.0
GROUP_SIZE = 4                  # leader + followers

AGENT_TYPE_SPEEDS = {
    "leader": 1.0,
    "follower": 0.9,
    "normal": 0.8,
    "panic": 1.0,
}

DENSITY_THRESHOLD = 2
GLOBAL_DENSITY_REPLAN_THRESHOLD = 2.0

# ============================================================
# EDGE / CONGESTION MODEL
# ============================================================
EDGE_BASE_CAPACITY = 5          # comfortable edge capacity

# ============================================================
# DYNAMIC ENVIRONMENT
# ============================================================
DYNAMIC_BLOCKS_ENABLED = False
BLOCK_NODE_EVERY_N_STEPS = 40
BLOCK_NODE_PROB = 0.4

DYNAMIC_EXITS_ENABLED = False
EXIT_TOGGLE_EVERY_N_STEPS = 80

# ============================================================
# EVACUATION MODE
# ============================================================
EVACUATION_MODE = False
DEFAULT_SCENARIO_NAME = "normal"

# ============================================================
# NAVIGATION STRATEGY
# ============================================================
NAV_STRATEGY_MODE = "mixed"  # "shortest", "congestion", "safe", "mixed"

NAV_STRATEGY_MIX = {
    "shortest": 0.34,
    "congestion": 0.33,
    "safe": 0.33,
}

# ============================================================
# MAP / FLOORPLAN SELECTION
# ============================================================
# "grid"   → generate simple grid
# "raster" → load PNG/JPG via floorplan_image_loader.py
# "dxf"    → load CAD DXF via dxf_loader.py
MAP_MODE = "dxf"
MAP_FILE = "maps/examples/call_center_pt2.dxf"  # path to PNG/JPG/DXF file

# ============================================================
# ============================================================
# RASTER FLOORPLAN PARAMETERS  (Pillow + NumPy ONLY)
# ============================================================
RASTER_DOWNSCALE_FACTOR = 5       # shrink large floorplans to speed processing

# Thresholds used by pure-PIL loader:
RASTER_WALL_THRESHOLD = 170       # luma <= this → wall candidate (Otsu overrides this)
RASTER_EXIT_GREEN_MIN = 200       # green >= this → exit candidate (simple RGB rule)

# Exit color heuristics
RASTER_USE_COLOR_SEGMENTATION = True  # simple RGB segmentation
RASTER_EXIT_COLOR_RGB = (0, 255, 0)

# Small component removal
RASTER_MIN_WALL_AREA = 3          # remove wall specks smaller than this (post-otsu)

# Colors for potential debug rendering
RASTER_WALL_COLOR_RGB = (0, 0, 0)
RASTER_WALKABLE_COLOR_RGB = (255, 255, 255)

# ============================================================
# ============================================================
# DXF / CAD PARAMETERS  (Enhanced DXF Loader)
# ============================================================

# Grid resolution into which DXF is rasterized
DXF_GRID_WIDTH = 110
DXF_GRID_HEIGHT = 70

# Your DXF layer naming:
DXF_WALL_LAYERS = ["0"]          # your actual wall layer
DXF_EXIT_LAYERS = ["WALL"]       # short lines drawn at exit locations
DXF_DOOR_LAYERS = []             # fill this if your DXF has an explicit DOOR layer

# Rasterization thresholds (in *grid cell units*, scaled to CAD units inside loader)
DXF_WALL_DISTANCE_THRESHOLD = 0.4     # how close grid-center must be to a wall segment
DXF_EXIT_DISTANCE_THRESHOLD = 0.4

# NEW: wall thickness handling (CAD units)
DXF_WALL_BUFFER = 0.0                 # radius added around wall segments

# NEW: endpoint merging
DXF_ENDPOINT_SNAP_DISTANCE = 0.5       # endpoints closer than this snap together

# NEW: door detection via gap detection (CAD units)
DXF_DOOR_GAP_THRESHOLD = 1.0           # small gap between wall segment endpoints → possible door

# ============================================================
# PERFORMANCE SWITCHES
# ============================================================
ENABLE_CONGESTION_WEIGHTS = False
ENABLE_DYNAMIC_EVENTS = False

# ============================================================
# OPTIONAL VISUALIZATION SETTINGS
# ============================================================
# SHOW_TRAILS = False
# SAVE_ANIMATION = False
