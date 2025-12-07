# Graph-Based Crowd Simulation with Agent AI

## Overview

This project implements a **graph-based crowd simulation framework** designed for academic and research use, particularly as a **final-year Computer Science & Engineering project**.  
It models crowds as intelligent agents navigating a graph-based environment with **congestion-aware routing**, **group behavior**, **multiple AI strategies**, **dynamic obstacles**, and **emergency evacuation logic**.

The system is not just a visual demo — it supports **quantitative experimentation**, **batch runs**, and **comparative analysis of agent strategies**, making it suitable for evaluation, reporting, and viva demonstrations.

---

## Key Concepts

- **Graph-based environment**: Nodes represent locations, edges represent walkable paths.
- **Agents**: Autonomous entities with goals, speed, perception, group membership, and navigation strategy.
- **Congestion-aware routing**: Edge weights and traversal time increase with crowd density.
- **Scenarios**: Predefined settings such as normal navigation, high-density crowds, blocked paths, and evacuation.
- **Metrics & analysis**: Travel time, congestion, collisions, evacuation KPIs.
- **Visualization**: 2D top-down view with live density, exits, and blocked regions.

---

## Project Structure

```
crowd/
│
├── main.py                # Interactive visual simulation runner
├── experiment.py          # Batch experiment runner for quantitative analysis
│
├── config.py              # Global configuration & tunable parameters
├── scenarios.py           # Scenario presets (normal, evacuation, blocked, etc.)
│
├── environment.py         # Graph environment & congestion modeling
├── agent.py               # Agent logic, strategies, and decision-making
├── simulation.py          # Core simulation engine & metric tracking
│
├── visualization.py       # 2D visualization & overlays
├── analysis.py            # Metrics plots & evacuation KPIs
│
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── CODE_OF_CONDUCT.md
```

---

## Installation

### Requirements
- Python 3.9+
- numpy
- matplotlib
- networkx

Install dependencies:
```bash
pip install numpy matplotlib networkx
```

---

## Running the Simulation (main.py)

`main.py` is used for **interactive runs with visualization**.

### Basic run
```bash
python main.py
```

This will:
1. Load default configuration
2. Initialize the environment and agents
3. Run the simulation step-by-step
4. Display live 2D visualization
5. Show plots and metrics at the end

### Choosing a Scenario

Edit `main.py` or `scenarios.py`:
```python
SCENARIO_NAME = "normal"
# options: normal, high_density, blocked, evacuation
```

This changes:
- Number of agents
- Dynamic obstacles
- Exit configuration
- Evaluation behavior

---

## Running Experiments (experiment.py)

`experiment.py` runs **multiple simulations automatically** to produce stable, averaged results.

### Example
```bash
python experiment.py normal 5
```

This means:
- Scenario: `normal`
- Runs: `5`
- Output: mean ± standard deviation of all metrics

### Output Includes
- Average travel time
- Average waits and replans
- Collision counts
- Exit success rate
- Congestion peaks
- Evacuation times (if applicable)

This is the **recommended mode for evaluation and report results**.

---

## Agent Navigation Strategies (AI Comparison)

Configured in `config.py`:

```python
NAV_STRATEGY_MODE = "mixed"
# options: shortest, congestion, safe, mixed
```

### Strategies

- **shortest**  
  Ignores congestion. Always uses geometric shortest path.

- **congestion**  
  Uses dynamic edge weights based on crowd density.

- **safe**  
  Aggressively avoids dense areas, even if path is longer.

- **mixed**  
  Mix of all strategies using proportions:
```python
NAV_STRATEGY_MIX = {
    "shortest": 0.34,
    "congestion": 0.33,
    "safe": 0.33,
}
```

---

## Scenarios Explained

### Normal
- Medium number of agents
- No dynamic obstacles
- Used as baseline

### High Density
- Large number of agents (stress test)
- Same environment
- Highlights congestion behavior

### Blocked Paths
- Nodes/edges get blocked during runtime
- Forces rerouting and replanning

### Emergency Evacuation
- Exits enabled
- All agents aim for nearest exit
- Measures evacuation time and bottlenecks

---

## Dynamic Time Model

Movement time is **not constant**.

- Each edge has:
  - `distance`
  - `max_capacity`
  - `dynamic weight`
- Agents move slower on congested or long edges
- Traversal time increases naturally without physics simulation

This creates realistic delays in bottleneck areas.

---

## Metrics Collected

### Per-Agent Metrics
- Steps taken
- Wait steps
- Replans
- Collisions
- Exit reached (yes/no)
- Exit time
- Path optimality vs ideal shortest path

### Global Metrics
- Average travel time
- Exit rate
- Total collisions
- Max density over time
- Evacuation KPIs (50%, 80%, 90%)

### Strategy-wise Metrics
- Average steps per strategy
- Average waits and replans
- Collision comparisons

Plots are generated automatically via `analysis.py`.

---

## Visualization Output

- Nodes and edges (graph)
- Agents as colored circles
- Exit nodes (green)
- Blocked nodes (red)
- Density heatmap overlay
- Strategy and type-based coloring

Outputs suitable for:
- Live demo
- Screenshots for report
- Result interpretation

---

## Configuration: What You Can Change

All tunable parameters live in `config.py`, including:
- Number of agents
- Max simulation steps
- Agent speed
- Perception radius
- Congestion thresholds
- Dynamic obstacle frequency
- Exit toggling
- Strategy mix
- Scenario flags

You are encouraged to modify these to study different behaviors.

---

## Academic Use & Evaluation

This project is designed for:
- Final-year project submission
- Research experimentation
- AI strategy evaluation
- Crowd and evacuation studies

It supports reproducible experiments and quantitative comparison — not just visualization.

---

## Documentation & Contribution

- See `CONTRIBUTING.md` for contribution rules
- See `CODE_OF_CONDUCT.md` for community standards
- Licensed under the MIT License (see `LICENSE`)

---

## Acknowledgment

This project builds upon standard concepts in:
- Graph algorithms (Dijkstra / A*)
- Multi-agent systems
- Crowd simulation research

All implementation and experimentation logic is original unless otherwise stated.
