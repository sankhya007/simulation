# config.py

GRID_WIDTH = 15
GRID_HEIGHT = 10

# You can change this to stress-test scalability (10–500 agents)
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
    "panic": 1.0,           # more aggressive, ignores congestion in routing
}

# Above this many agents in a node, others will try to avoid it
DENSITY_THRESHOLD = 2

# If global density is high, agents may replan their routes more often
GLOBAL_DENSITY_REPLAN_THRESHOLD = 2.0   # avg agents per occupied node


# === Dynamic scenario controls (Step 4) ===

# Toggle dynamic blocking/unblocking of nodes (blocked path scenario)
DYNAMIC_BLOCKS_ENABLED = True
BLOCK_NODE_EVERY_N_STEPS = 40    # how often to consider blocking nodes
BLOCK_NODE_PROB = 0.4            # probability to actually block a candidate node

# Toggle dynamic exit opening/closing (evacuation / environment change)
DYNAMIC_EXITS_ENABLED = True
EXIT_TOGGLE_EVERY_N_STEPS = 80   # how often to toggle exits
