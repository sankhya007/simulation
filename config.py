# config.py
# =========
# Global configuration / experiment knobs for the crowd simulation project.
#
# Each block below is grouped by purpose. Every parameter has a comment
# explaining what it controls, reasonable default values, and notes about
# how it interacts with other parameters.
#
# Keep this file under version control so experiments are reproducible.
# Changing config values will affect simulation behavior globally.

# -------------------------
# CORE GRID / RUN PARAMETERS
# -------------------------
# Parameters that affect the simulation grid and basic experiment length.
# These are fundamental: they change graph sizes and runtime complexity.
GRID_WIDTH = 15
GRID_HEIGHT = 10
# - GRID_WIDTH / GRID_HEIGHT:
#   Used when MAP_MODE == "grid". They define number of columns and rows.
#   Larger grids increase routing options but also CPU cost (graphs get bigger).

NUM_AGENTS = 60
# - NUM_AGENTS:
#   Default number of agents for scenarios when not overridden.
#   Increasing this increases interaction density and computational load.

MAX_STEPS = 400
# - MAX_STEPS:
#   Maximum number of timesteps each simulation run will take (headless).
#   Shorten for fast debugging, increase for long experiments.

SEED = 42
# - SEED:
#   Global RNG seed for repeatability. Set to None for nondeterministic runs.

# -------------------------
# AGENT BEHAVIOUR / SOCIAL
# -------------------------
# Parameters controlling agent decision-making, perception and basic timing.

AGENT_REPLAN_PROB = 0.01
# - AGENT_REPLAN_PROB:
#   Probability an agent picks a brand-new random goal when finished.
#   Use small values for mostly persistent tasks; higher values for exploratory agents.

COLLISION_DISTANCE = 0.25
# - COLLISION_DISTANCE:
#   Threshold used for counting collisions (pairwise distance).
#   Smaller values mean "closer" collisions; used only for metrics (not physics).

PERCEPTION_RADIUS = 3.0
# - PERCEPTION_RADIUS:
#   How far an agent "sees" to react to leaders, congestion or events.
#   Affects group-following and context-aware replanning.

GROUP_SIZE = 4
# - GROUP_SIZE:
#   Number of agents in a single group (1 leader + (GROUP_SIZE-1) followers).

# nominal per-type preferred movement speed (used by Graph adapter and as nominal for continuous models)
AGENT_TYPE_SPEEDS = {
    "leader": 1.0,
    "follower": 0.9,
    "normal": 0.8,
    "panic": 1.0,
}
# - AGENT_TYPE_SPEEDS:
#   Preferred speed used by decision logic and graph->velocity adapters.
#   For continuous models (Social Force / RVO) this is the target preferred speed,
#   but the physics model may override / clamp actual speed.

DENSITY_THRESHOLD = 2
# - DENSITY_THRESHOLD:
#   Node-level occupancy threshold agents use to decide sidesteps vs waiting.

GLOBAL_DENSITY_REPLAN_THRESHOLD = 2.0
# - GLOBAL_DENSITY_REPLAN_THRESHOLD:
#   If average agents per occupied-node > this threshold, some agents replan routes.
#   Helps simulate crowd-aware rerouting in congested scenarios.

# -------------------------
# EDGE / CONGESTION WEIGHTS
# -------------------------
# These control how edge weights are dynamically modified based on occupancy.

EDGE_BASE_CAPACITY = 5
# - EDGE_BASE_CAPACITY:
#   Comfortable capacity for an edge (two adjacent nodes). If real occupancy > capacity,
#   the edge can be considered "over capacity" and weight increased.

# Enable dynamic congestion weighting (computational cost: small but nonzero)
ENABLE_CONGESTION_WEIGHTS = False
# - ENABLE_CONGESTION_WEIGHTS:
#   If True, `simulation` will compute an occupancy-based dynamic weight for each edge.
#   Set True to model slowdowns on congested links, False for pure shortest-path dynamics.

# -------------------------
# DYNAMIC ENVIRONMENT (events)
# -------------------------
# Runtime events: nodes blocking / exits toggling. Useful for stress-testing replanning.

ENABLE_DYNAMIC_EVENTS = False
# - ENABLE_DYNAMIC_EVENTS:
#   Master switch for dynamic blocking/opening behaviour used in experiments.

DYNAMIC_BLOCKS_ENABLED = False
# - DYNAMIC_BLOCKS_ENABLED:
#   When True, unoccupied nodes may be randomly blocked at intervals.

BLOCK_NODE_EVERY_N_STEPS = 40
BLOCK_NODE_PROB = 0.4
# - BLOCK_NODE_EVERY_N_STEPS & BLOCK_NODE_PROB:
#   Frequency and probability of randomly blocking a candidate node.
#   Useful to simulate sudden obstacles or failures.

DYNAMIC_EXITS_ENABLED = False
EXIT_TOGGLE_EVERY_N_STEPS = 80
# - DYNAMIC_EXITS_ENABLED & EXIT_TOGGLE_EVERY_N_STEPS:
#   Periodically open/close exits to simulate evolving floorplan access.

# -------------------------
# EVACUATION MODE
# -------------------------
# In evacuation mode agents will head for exits and will not pick new goals.
EVACUATION_MODE = False
# - EVACUATION_MODE:
#   If True, reaching an exit is terminal; agents do not continue to roam.
DEFAULT_SCENARIO_NAME = "normal"

# -------------------------
# NAVIGATION STRATEGY (population mix)
# -------------------------
# Controls whether all agents use the same nav strategy or a mixture.

NAV_STRATEGY_MODE = "mixed"
# - NAV_STRATEGY_MODE: "shortest" | "congestion" | "safe" | "mixed"
#   "shortest" uses distance-only routing, "congestion" uses dynamic weights,
#   "safe" could be used for risk-averse planners (currently an alias to congestion),
#   "mixed" distributes strategies among agents based on NAV_STRATEGY_MIX.

NAV_STRATEGY_MIX = {
    "shortest": 0.34,
    "congestion": 0.33,
    "safe": 0.33,
}
# - NAV_STRATEGY_MIX:
#   Only used when NAV_STRATEGY_MODE == "mixed". These fractions determine how many
#   agents of each strategy are created. They should sum to ≈1.0.

# -------------------------
# MAP / FLOORPLAN SELECTION
# -------------------------
# Choose how the environment graph is created.
# Options:
#  - "grid"   -> programmatic grid (GRID_WIDTH x GRID_HEIGHT)
#  - "raster" -> raster floorplan loader (PNG/JPG) -> floorplan_image_loader.py
#  - "dxf"    -> DXF/CAD loader -> dxf_loader.py

MAP_MODE = "dxf"  # "grid" | "raster" | "dxf"
# - MAP_MODE:
#   "raster" will load MAP_FILE and run the raster->layout converter; good for floorplans.
#   "dxf" uses CAD input (if you have a DXF file and loader tuned).
#   "grid" is fastest and simplest for algorithmic debugging.

MAP_FILE = "maps/examples/call_center_pt2.dxf"
# - MAP_FILE:
#   Path to map file used by raster/dxf loaders. When MAP_MODE == "grid" this is ignored.

# -------------------------
# RASTER FLOORPLAN PARAMETERS (pure Pillow + NumPy loader)
# -------------------------
# These control how a raster image is converted to a grid layout.

RASTER_DOWNSCALE_FACTOR = 5
# - RASTER_DOWNSCALE_FACTOR:
#   Downscales very large images to speed processing. Larger factor -> smaller grid.
#   Use 1 to keep full resolution; increase for speed.

RASTER_WALL_THRESHOLD = 170
# - RASTER_WALL_THRESHOLD:
#   Fallback luma threshold: pixels with luma <= threshold considered wall candidates.
#   The loader also runs Otsu thresholding so this is rarely required to tweak.

RASTER_EXIT_GREEN_MIN = 200
RASTER_USE_COLOR_SEGMENTATION = True
RASTER_EXIT_COLOR_RGB = (0, 255, 0)
# - RASTER_EXIT_*:
#   Heuristics to detect exits by green pixels (common convention). Turn off color segmentation
#   if your map does not use colored exit markers.

RASTER_MIN_WALL_AREA = 3
# - RASTER_MIN_WALL_AREA:
#   Remove small connected wall components smaller than this (speck removal).

RASTER_WALL_COLOR_RGB = (0, 0, 0)
RASTER_WALKABLE_COLOR_RGB = (255, 255, 255)
# - Color constants used for visualization/debug output only.

# -------------------------
# DXF / CAD PARAMETERS (if using DXF loader)
# -------------------------
# These are used by dxf_loader.py. They relate rasterization & snapping thresholds.

DXF_GRID_WIDTH = 110
DXF_GRID_HEIGHT = 70
# - DXF_GRID_*:
#   Resolution used for rasterizing CAD geometry into a grid. Higher -> finer detail.

DXF_WALL_LAYERS = ["0"]
DXF_EXIT_LAYERS = ["WALL"]
DXF_DOOR_LAYERS = []
# - Layer lists:
#   Names of DXF layers to treat as walls/exits/doors. Adjust for your CAD file conventions.

DXF_WALL_DISTANCE_THRESHOLD = 0.4
DXF_EXIT_DISTANCE_THRESHOLD = 0.4
DXF_WALL_BUFFER = 0.0
DXF_ENDPOINT_SNAP_DISTANCE = 0.5
DXF_DOOR_GAP_THRESHOLD = 1.0
# - DXF thresholds:
#   Tuning knobs for snapping endpoints, merging close endpoints, and door-gap detection.

# -------------------------
# PERFORMANCE / FEATURE SWITCHES
# -------------------------
ENABLE_CONGESTION_WEIGHTS = False
# - Toggle dynamic weights on edges. Repeated: set True for congestion-aware routing.

ENABLE_DYNAMIC_EVENTS = False
# - Master switch for dynamic blocking / exit toggles. Set True for stress-tests.

# -------------------------
# MOTION MODEL (Agent movement physics)
# -------------------------
# This selects which low-level motion model agents use. It impacts both how they
# choose their next move and how they are integrated in continuous space.
#
# Options:
#  - "graph"        : legacy discrete grid-based moves (agents jump node-to-node)
#  - "social_force" : continuous Helbing-style forces (smooth physics + repulsion)
#  - "rvo"          : reciprocal-velocity-obstacle style sampled avoidance (local collision-free behavior)
#
# Differences / when to use:
#  - Use "graph" for exact reproducibility with legacy experiments and when you only
#    care about routing/strategies (fastest, simplest).
#  - Use "social_force" for realistic continuous pedestrian flows, lane formation, and physics-like interactions.
#    Social Force tends to need parameter tuning (relaxation time, repulsion A/B) and sometimes smaller dt/sub-stepping.
#  - Use "rvo" when you want robust local collision avoidance with fewer stability issues than naive force models.
MOTION_MODEL = "graph"  # "graph" | "social_force" | "rvo"

# -------------------------
# Social Force model hyperparameters
# -------------------------
# If you choose MOTION_MODEL = "social_force", tune these values.
# Typical trade-offs:
#  - Higher SF_A -> stronger repulsive forces -> fewer overlaps but risk of oscillations.
#  - Larger SF_B -> longer-range but gentler repulsion.
#  - Lower SF_RELAX_T -> agents accelerate to desired velocity faster (can cause instability).

SF_RELAX_T = 0.5
# - Relaxation time (seconds) for velocity adaptation. Smaller -> quicker changes.

SF_A = 2.0
# - Repulsion strength (A). Exponential factor multiplies into force magnitude.

SF_B = 0.5
# - Repulsion range (B). Larger -> broader but weaker repulsion spread.

SF_MAX_SPEED = 1.5
# - Max speed clamp used by the Social Force integrator.

AGENT_RADIUS = 0.3
# - Personal space radius used for repulsion and collision approximations.

# -------------------------
# RVO (ORCA-like) hyperparameters
# -------------------------
# The RVO implementation here is sampling-based and approximate; tune these for
# denser crowds or sparser interactions.

RVO_NEIGHBOR_DIST = 3.0
# - How far (in world units) an agent considers neighbors for collision avoidance.

RVO_MAX_SPEED = 1.5
# - Maximum candidate speed used by the RVO sampler.

RVO_SAMPLES = 16
# - Number of candidate velocities sampled per agent per step. More samples -> better avoidance but slower.

# -------------------------
# VISUALIZATION / OUTPUT
# -------------------------
# Non-critical knobs for saving, debug traces and visuals.

# SHOW_TRAILS = False
# SAVE_ANIMATION = False
# (not used by default — kept commented so users can enable easily)

# -------------------------
# NOTES & TUNING GUIDANCE
# -------------------------
# - For quick debugging: use MAP_MODE="grid", NUM_AGENTS small (e.g., 10-30), MOTION_MODEL="graph".
# - To compare models: run three short simulations with identical RNG seed and switch MOTION_MODEL
#   between "graph", "social_force", and "rvo".
# - If Social Force causes jitter: reduce SF_A, increase SF_B, reduce SF_MAX_SPEED, and/or perform more integration
#   sub-steps per sim tick (dt < 1.0).
# - For performance: reduce RVO_SAMPLES or RVO_NEIGHBOR_DIST; reduce NUM_AGENTS; enable spatial partitioning
#   in the environment (if you add it).
#
# Keep this file tidy — it's the easiest place to create reproducible experiments.
