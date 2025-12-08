# Crowd Simulation System – Floorplan-Aware Multi-Agent Evacuation Simulator

This project is a **floorplan-aware crowd simulation engine** that supports:

- Raster floorplans (PNG/JPG)
- CAD floorplans (DXF)
- Dynamic scenario loading
- Evacuation modelling
- Bottleneck detection
- Batch simulation with multiprocessing
- Heatmap + overlay visualisation on real maps
- Exporting metrics, CSV reports, and annotated images

The simulation builds a grid/graph from maps and runs multi-agent navigation using:
- Shortest-path routing
- Congestion-aware routing
- Safe routing
- Mixed strategies

---

## 🚀 Features

### ✔ 1. Raster Floorplan Loader (PNG/JPG)
- Converts floorplans into grid layout
- Detects:
  - Walls (black / dark grey)
  - Walkable regions
  - Doors
  - Exits (green or user-defined)
- Supports downsampling for performance
- Produces a `layout_matrix` used by the `EnvironmentGraph`

### ✔ 2. CAD Floorplan Loader (DXF)
- Reads DXF via `ezdxf`
- Recognizes:
  - Wall layers
  - Door layers
  - Exit layers
- Rasterizes DXF into grid
- Returns:
  - `layout_matrix`
  - Metadata: bounding box, resolution, transformation mapping
- Enables mapping bottleneck grid cells back to real DXF coordinates

---

## ✔ 3. Simulation Engine
Built around:
- `EnvironmentGraph`
- `CrowdSimulation`
- `Agent` logic

Supports:
- Collision detection
- Density tracking
- Dynamic blocked nodes
- Evacuation mode (nearest exit routing)
- Multi-agent behaviours

Each simulation step logs:
- Agent movement
- Node occupancy
- Collisions
- Evacuation times

---

## ✔ 4. Scenarios System
Selectable at runtime:

| Scenario | Description |
|---------|-------------|
| `normal` | Normal dispersed navigation |
| `high_density` | Many agents, stress-test the map |
| `blocked` | Dynamic blocked paths |
| `evacuation` | Emergency evacuation to nearest exits |
| `floorplan_image_*` | Raster floorplan simulations |
| `floorplan_dxf_*` | CAD floorplan simulations |

You can extend scenarios easily through `scenarios.py`.

---

## ✔ 5. Visualization
Two layers:

### **Grid-based visual simulation**
Shows:
- Agents (colored by type)
- Exits
- Blocked cells
- Density heatmap
- Congestion level

### **Overlay visualization (NEW)**
- Shows bottleneck heatmaps *directly on the original floorplan image*
- Supports PNG and DXF (via rasterization)
- Marks top-K bottlenecks using:
  - Red circles
  - Labels `B1`, `B2`, ...

---

## ✔ 6. Bottleneck Detection
Identifies high-density graph nodes using:

- Total visit counts
- Time-window average density
- Edge congestion
- Repeated batch trials aggregation

Outputs:
- A ranked list of bottleneck grid cells
- (DXF mode) CSV mapping grid cells → real world CAD coordinates

---

## ✔ 7. Batch Simulation (Multiprocessing)
Run parallel experiments:

```bash
python main.py batch --trials 7 --workers 6 --target-percent 0.95 --agents 300 --steps 1200
```

Batch runner:
- Runs N trials in parallel
- Gathers bottlenecks across runs
- Saves:
  - Metrics JSON
  - CSV bottleneck coordinates
  - Overlayed PNG heatmaps
  - Serialized simulation dumps (if enabled)

---

## ✔ 8. CLI Commands

### List scenarios
```bash
python main.py list
```

### Visual simulation
```bash
python main.py visual normal
```

### Run a batch of trials
```bash
python main.py batch --trials 5 --workers 4
```

### Run evacuation with overlays
```bash
python main.py run evacuation --agents 300 --steps 1500 --overlay --out-dir results
```

---

## 📁 Folder Structure

```
crowd/
│
├── main.py
├── config.py
├── scenarios.py
├── simulation.py
├── environment.py
├── agent.py
├── analysis.py
│
├── maps/
│   ├── __init__.py
│   ├── map_loader.py
│   ├── map_meta.py
│   ├── raster_loader.py
│   ├── dxf_loader.py
│   ├── floorplan_image_loader.py
│   ├── examples/
│   │   ├── example_floorplan.png
│   │   └── call_center_pt2.dxf
│   └── misc/
│
├── visualization.py       # patched run_visual_simulation
│
├── tools/
│   ├── preview_raster.py
│   ├── dxf_overlay_preview.py
│   ├── export_graph.py (optional)
│   ├── map_debug.py (optional)
│   └── ...
│
├── tests/
│   ├── test_dxf_loader.py
│   ├── test_raster_loader.py
│   ├── test_simulation_step.py
│   ├── fixtures/
│   │   └── ...
│   └── ...
│
├── requirements.txt
├── README.md
├── DEMO.md
├── .gitignore
│
└── (NO __pycache__/ directories should exist)

```

---

## 🛠 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

DXF support:
```bash
pip install ezdxf
```

Image processing:
```bash
pip install pillow
```

---

## 🧪 Testing Example

### PNG Floorplan
```bash
MAP_MODE="raster"
MAP_FILE="maps/examples/floor.png"

python main.py visual floorplan_image_normal
```

### DXF Floorplan
```bash
MAP_MODE="dxf"
MAP_FILE="maps/examples/mall.dxf"

python main.py visual floorplan_dxf_evac
```

---

## 📊 Outputs

- Density heatmaps
- Bottleneck overlays
- Evacuation KPIs (50%, 80%, 90%)
- CSV bottleneck coordinate mapping
- JSON serialized trial data

---

## 📘 Future Work

- 3D multi-level support
- Fire propagation / hazard modelling
- Calibration with real-world evacuation videos
- Multi-floor connectivity (stairs / elevators)

---

## 🤝 Contributions
Feel free to modify scenarios, add new loaders, or upgrade the visual overlays.

---

## 📝 License
MIT License unless changed by your institution.
