# config.py

GRID_WIDTH = 15
GRID_HEIGHT = 10

NUM_AGENTS = 35
MAX_STEPS = 400

SEED = 42

AGENT_REPLAN_PROB = 0.01
COLLISION_DISTANCE = 0.25

# === Environment / Edge properties ===
EDGE_BASE_CAPACITY = 5   # how many agents an edge can comfortably support

# === Agent behaviour parameters ===
PERCEPTION_RADIUS = 3.0     # how far agents 'see' congestion / group

GROUP_SIZE = 4              # size of a leader + followers group

# Preferred movement speed (probability of moving each tick)
AGENT_TYPE_SPEEDS = {
    "leader": 1.0,          # moves almost every tick
    "follower": 0.9,
    "normal": 0.8,
    "panic": 1.0,           # later we can give them different logic
}

# Above this many agents in a node, others will try to avoid it
DENSITY_THRESHOLD = 2
