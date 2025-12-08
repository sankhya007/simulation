# DEMO GUIDE — Crowd Simulation on Real Floorplans

This quick guide helps you **run the full system**, including:
- Loading a PNG or DXF map  
- Running visual simulation  
- Running evacuation trials  
- Detecting bottlenecks  
- Generating heatmap overlays  
- Saving results for reports  

This document is beginner-friendly and focuses on **how to run the program**, not the internal code.

---

# 🎯 1. Prerequisites

Install Python packages:

```bash
pip install -r requirements.txt
pip install ezdxf pillow matplotlib numpy
```

Make sure you have:

- A **PNG/JPG floorplan** *or*
- A **DXF floorplan** (AutoCAD → Save As → DXF R2010 recommended)

Place them inside:

```
crowd/maps/examples/
```

Example:
```
crowd/maps/examples/mall.png
crowd/maps/examples/mall.dxf
```

---

# 🗺 2. Choose Map Mode (PNG or DXF)

In `config.py`, set:

### For PNG / raster image:
```python
MAP_MODE = "raster"
MAP_FILE = "maps/examples/mall.png"
```

### For DXF:
```python
MAP_MODE = "dxf"
MAP_FILE = "maps/examples/mall.dxf"
```

You can switch anytime.

---

# ▶️ 3. Run a Simple Visual Simulation

Run:

```bash
python main.py visual normal
```

You should see:

- The map grid  
- Agents moving  
- Density heatmap  
- Collisions updating  
- Exits detected  

---

# 🚨 4. Run an Evacuation Simulation

```bash
python main.py visual evacuation
```

What you’ll see:

- Agents rushing to exits  
- High congestion at narrow corridors  
- Paths dynamically adjusting  

---

# 🔥 5. High-Density Stress Test

```bash
python main.py visual high_density
```

This loads many more agents to purposely produce congestion.

---

# 🚧 6. Blocked Path Scenario

```bash
python main.py visual blocked
```

This introduces new blocked nodes every few steps.  
Useful for “disaster-mode” simulation.

---

# 🧪 7. Batch Experiments (Multiprocessing)

Run N trials in parallel to detect REAL bottlenecks:

```bash
python main.py batch --trials 10 --workers 6 --agents 300 --steps 1500 --target-percent 0.9 --out-dir results_mall
```

This produces:

- `results_mall/bottlenecks.csv`
- `results_mall/overlay.png`
- `results_mall/heatmap.png`
- Trial logs (`trial_01.json` … `trial_10.json`)

---

# 📊 8. Understanding Results

## Bottleneck CSV
Contains columns:

- `grid_x`, `grid_y`
- `visit_count`
- `dxf_x`, `dxf_y` (if DXF provided)
- `rank` (“B1”, “B2”, etc.)

## Overlay PNG
Shows bottlenecks directly on the original floorplan image.  
Red = severe congestion  
Yellow = moderate  
Green = exit positions  

This is suitable for report screenshots.

---

# 🧩 9. Troubleshooting

### ❗ Map looks wrong / missing corridors
Likely reasons:
- Wall threshold too strict  
- Wrong DXF layer names  
- Too much downscaling  

Try adjusting in `config.py`:

```python
RASTER_WALL_MAX_LUMA = 90
```

Or edit DXF so walls are in a layer named:

- `WALL`  
- `WALLS`

### ❗ Exits not detected  
Add a thin line in a layer named:

- `EXIT`
- `EXIT_DOOR`

### ❗ Simulation is too slow
Increase downscale:

```python
RASTER_DOWNSCALE_FACTOR = 6 or 8
```

Or reduce agent count.

---

# 🧪 10. Recommended Demo Flow for Presentation

1. Show the **original PNG/DXF map**
2. Show the **grid conversion**
3. Run **normal simulation**
4. Run **evacuation**
5. Show **heatmap + overlay**
6. Run a **batch** and show aggregated bottlenecks
7. Present CSV → “top risk areas” in the building

This gives a perfect FYP demonstration.

---

# 🏁 End of DEMO  
You’re now ready to demonstrate your system in class, presentations, or research!