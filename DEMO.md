# Crowd Simulation Demo Guide

This guide explains how to run demos, generate visual outputs, and explore the behavior of the Crowd Simulation Engine.

---

## 🎮 1. Quick Visual Simulation

Run a real‑time visualization:

```bash
python main.py visual normal
```

This displays:
- Agents moving across the map  
- Leaders, followers, panic agents (color‑coded)  
- Continuous or graph-based motion (based on config)  
- Walls, exits, and obstacles  

---

## 🧩 2. Motion Model Comparison Demo

Run a side-by-side benchmark of the three main motion models:

```bash
python demos/compare_motion_models.py
```

This will generate PNG files inside `demos/generated_images/`:

- `compare_graph.png`
- `compare_social_force.png`
- `compare_rvo.png`

These images show clear differences in:
- Path smoothness  
- Collision avoidance  
- Crowd pressure  
- Agent spreading behavior  

---

## 🗺️ 3. DXF Floorplan Overlay Preview

To visualize how DXF geometry is interpreted:

```bash
python tools/dxf_overlay_preview.py maps/my_map.dxf
```

Outputs a PNG overlay showing:
- Wall detection  
- Exit recognition  
- Grid rasterization  
- CAD → grid alignment  

Useful for debugging layout or verifying map preprocessing.

---

## 🖼️ 4. Raster (PNG/JPG) Floorplan Preview

Preview how a raster floorplan converts into walls & exits:

```bash
python tools/preview_raster.py maps/floorplan.png
```

Shows:
- Otsu-thresholded walls  
- Exit color detection  
- Resulting grid layout  

---

## 🧭 5. Graph Structure Preview

To visualize the navigation graph:

```bash
python tools/preview_graph.py --type grid
python tools/preview_graph.py --type centerline
python tools/preview_graph.py --type hybrid
```

You will see:
- Nodes and edges  
- Weight differences  
- Optional map overlay for context  

---

## 🔥 6. Panic Propagation Demo

Force‑enable panic spread for demonstration:

Inside `config.py`:

```python
PANIC_SPREAD_PROB = 1.0
PANIC_SPREAD_RADIUS = 999
```

Then run:

```bash
python test_motion_smoke.py
```

You should observe:
- Panic seeded in one agent
- Rapid propagation across the population  

---

## 📊 7. Bottleneck Detection Demo

Running a simulation through `main.py` automatically generates CSV reports:

```bash
python main.py normal
```

Find outputs in:

```
out_runX/*_bottlenecks.csv
```

Each file contains:
- Node congestion counts  
- Time‑step density peaks  
- Bottleneck severity scores  

Great for evacuation research & architectural analysis.

---

## 🎥 8. Recording Videos (Optional)

Install video tools:

```bash
pip install matplotlib==3.7.0 ffmpeg-python
```

Modify `visualization.py` to save frames, then combine them using ffmpeg:

```bash
ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 demo.mp4
```

---

## 🧪 9. Smoke Test / Quick Health Check

Run a fast simulation to verify the entire system:

```bash
python test_motion_smoke.py
```

This prints:
- Agent attribute samples  
- Motion model sanity check  
- Panic propagation behavior  
- Summary metrics  

---

## End of DEMO Guide
For additional help or examples, check the README or open an Issue on GitHub.
