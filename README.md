# Crowd Simulation Engine
A complete modular crowd simulation framework supporting:
- DXF & raster floorplans
- Graph-based navigation
- Social Force & RVO continuous motion models
- Agent heterogeneity (speed, size, visibility, reaction time)
- Panic propagation & group behavior
- Real-time visualization, bottleneck detection, metrics export

---

# 🏁 Quick Start

## 1️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

---

# 🚀 RUNNING THE PROJECT (A → Z)

Below is **EVERY command available** in the project.

---

# 🅰 MAIN SIMULATION COMMANDS

## ▶ Run a standard simulation (non-visual)
```bash
python main.py normal
```

## ▶ Run a visual simulation (live animation)
```bash
python main.py visual normal
```

## ▶ Run with a specific navigation strategy
```bash
python main.py visual shortest
python main.py visual congestion
python main.py visual safe
```

## ▶ Switch motion model inside config.py
```python
MOTION_MODEL = "graph"
MOTION_MODEL = "social_force"
MOTION_MODEL = "rvo"
```

---

# 🅱 DEMO COMMANDS

## 🎮 Motion Model Comparison
Produces images comparing:
- Graph model
- Social Force model
- RVO model

```bash
python demos/compare_motion_models.py
```

---

# 🅲 MAP PROCESSING COMMANDS

## 🗺️ Preview DXF → Grid Conversion
```bash
python tools/dxf_overlay_preview.py maps/my_map.dxf
```

## 🖼️ Preview Raster (PNG/JPG) Map Conversion
```bash
python tools/preview_raster.py maps/floorplan.png
```

## 🔍 Preview Navigation Graph Structure
```bash
python tools/preview_graph.py --type grid
python tools/preview_graph.py --type centerline
python tools/preview_graph.py --type hybrid
```

---

# 🅳 TESTING COMMANDS

## 🧪 Test DXF Loader
```bash
pytest tests/test_dxf_loader.py -q
```

## 🧪 Test Raster Loader
```bash
pytest tests/test_raster_loader.py -q
```

## 🧪 Test Graph Builder
```bash
pytest tests/test_graph_builder.py -q
```

## 🧪 Test Simulation Step
```bash
pytest tests/test_simulation_step.py -q
```

## 🧪 Test Motion Models
```bash
pytest tests/test_motion_smoke.py -q
```

## 🧪 Run ALL tests
```bash
pytest -q
```

---

# 🅴 UTILITY & DEBUG COMMANDS

## 📏 Coordinate Mapping Repair Test
```bash
python tools/test_coordinate_mapping.py
```

## 🧱 Full System Check
```bash
python tools/full_system_check.py
```

## 🧭 Layer Mapping Helper for DXF Files
```bash
python tools/dxf_layer_mapper.py maps/my_map.dxf
```

## 🖼️ Preview graph with overlay
```bash
python tools/preview_graph.py --overlay maps/my_map.png
```

---

# 🅵 OUTPUTS & ANALYSIS

## 📊 Bottleneck CSV Export  
After running:
```bash
python main.py normal
```

Check:
```
out_runX/*_bottlenecks.csv
```

## 📈 View agent visit density map
```python
from simulation import CrowdSimulation
sim.get_density_matrix()
```

---

# 🅶 CONFIGURATION OPTIONS

Edit these in **config.py**:

| Feature | Variable |
|--------|----------|
| Map file path | MAP_FILE |
| Motion model | MOTION_MODEL |
| Agent heterogeneity | AGENT_SPEED_MEAN, AGENT_RADIUS_MEAN, etc. |
| Panic mechanics | PANIC_SPREAD_PROB, PANIC_SPREAD_RADIUS |
| Group behaviors | GROUP_SIZE |
| Dynamic events | DYNAMIC_BLOCKS_ENABLED, DYNAMIC_EXITS_ENABLED |
| Visualization | VISUAL_FRAME_DELAY |

---

# 🅷 PROJECT STRUCTURE

```
crowd-simulation/
│
├── agent.py
├── simulation.py
├── environment.py
├── motion_models.py
├── config.py
├── main.py
│
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── DEMO.md
├── LICENSE
├── .gitignore
│
├── maps/
│   ├── __init__.py
│   ├── map_loader.py
│   ├── map_meta.py
│   ├── dxf_loader.py
│   ├── raster_loader.py
│   ├── floorplan_image_loader.py
│   │
│   ├── examples/
│   │   ├── example1.png
│   │   ├── example2.png
│   │   ├── example_map.dxf
│   │   └── README.md
│   │
│   └── custom_maps/
│       └── README.md
│
├── tools/
│   ├── preview_graph.py
│   ├── dxf_overlay_preview.py
│   ├── preview_raster.py
│   ├── preview_graph.py
│   ├── full_system_check.py
│   ├── dxf_layer_mapper.py
│   └── test_coordinate_mapping.py
│
├── visualization.py
│
├── demos/
│   ├── compare_motion_models.py
│   ├── example_run.py
│   └── generated_images/
│       ├── compare_graph.png
│       ├── compare_sf.png
│       └── compare_rvo.png
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_simulation_step.py
│   ├── test_coordinate_mapping.py
│   ├── test_graph_builder.py
│   ├── test_raster_loader.py
│   ├── test_dxf_loader.py
│   ├── test_cad_coords.py
│   ├── test_motion_smoke.py
│   ├── test_motion_models.py
│   └── test_agent_behavior.py   (optional future test)
│
├── outputs/
│   ├── logs/
│   ├── density_maps/
│   ├── bottleneck_reports/
│   └── simulation_runs/
│       ├── run_001/
│       │   ├── frames/
│       │   ├── density.csv
│       │   ├── bottlenecks.csv
│       │   └── summary.json
│       └── run_002/
│
├── data/
│   ├── training_data/      (if ML is added later)
│   └── exported_maps/
│
└── docs/
    ├── architecture.md
    ├── design_principles.md
    ├── roadmap.md
    ├── api_reference/
    │   ├── simulation.md
    │   ├── agent.md
    │   ├── environment.md
    │   └── motion_models.md
    └── images/
```

---

# 🅸 CONTRIBUTING

See:
```
CONTRIBUTING.md
```

---

# 🅹 LICENSE
MIT License (or add your own)

---

# 🎉 You're Ready!

Run any simulation:
```bash
python main.py visual normal
```

Generate demos:
```bash
python demos/compare_motion_models.py
```

Debug maps:
```bash
python tools/preview_graph.py --type grid
```

To report issues or request features, open a GitHub issue.

Happy simulating ❤️
