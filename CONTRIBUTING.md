# Contributing to the Crowd Simulation Project  
A clear, beginner‑friendly guide for contributing to the system.

Thank you for your interest in contributing!  
Whether you’re fixing bugs, improving the simulation engine, adding new map loaders, or helping with documentation, **all contributions are welcome**.

This guide explains **how to contribute**, **coding standards**, and **the workflow** used in this project.

---

# 🌱 Ways You Can Contribute

### 🐛 1. Report Bugs
If something doesn’t work as expected:
- Open a GitHub issue
- Include logs, screenshots, and steps to reproduce

### 🧠 2. Suggest Enhancements
Ideas for:
- Better visualization
- Faster simulation
- Improved agents
- CAD/image processing upgrades
- New scenarios

### 🧪 3. Add Features
Good first features:
- Better bottle‑neck detection algorithms
- More evacuation metrics
- Additional raster or DXF formats
- Multi‑floor support (stairs, elevators)
- Exporting result videos

### ✍️ 4. Improve Documentation
You can help by improving:
- README.md
- DEMO.md
- tutorials
- inline code comments

### 🎨 5. Provide Better Demo Maps
Provide:
- Clean PNG mall layouts  
- DXF files with proper WALL/EXIT layers  
- Classroom, auditorium, campus maps  

---

# 🔧 Project Structure (Important!)

```
crowd/
│── main.py                    # CLI, batch runner, visual runner
│── simulation.py              # Agent engine + timestep loop
│── environment.py             # Grid/graph building
│── agent.py                   # Agent behavior logic
│── maps/
│     ├── raster_loader.py     # PNG/JPG → grid
│     ├── dxf_loader.py        # DXF → grid
│     ├── map_loader.py        # Routing for map modes
│── analysis.py                # Metrics, KPIs, bottlenecks, overlays
│── scenarios.py               # Scenario presets (normal, high_density, etc.)
│── visualization.py           # Live animation + overlays
│── README.md / DEMO.md        # Docs for users
```

---

# 🧵 Workflow for Contributing

## 1️⃣ Fork the Repository
```
Click "Fork" on GitHub
```

## 2️⃣ Clone Your Fork
```bash
git clone https://github.com/yourname/crowd-simulation.git
cd crowd-simulation
```

## 3️⃣ Create a Branch
```bash
git checkout -b feature/my-new-feature
```

Examples:
```
feature/add-dxf-overlay
fix/raster-threshold
docs/improve-readme
```

## 4️⃣ Make Your Changes
Follow coding standards below.

## 5️⃣ Run Tests
- Basic simulation (`normal`)
- Raster loader test
- DXF loader test
- Evacuation scenario
- Batch runs (if modified)

## 6️⃣ Commit
```bash
git add .
git commit -m "Add new DXF overlay rendering"
```

## 7️⃣ Push & Create Pull Request
```bash
git push origin feature/my-new-feature
```

Submit a Pull Request describing:
- What you added
- Why it is useful
- How to test

---

# 🧩 Coding Guidelines

### ✔ Python Style
- Follow **PEP‑8**
- Use **type hints** everywhere
- Keep functions short and readable
- Comment tricky algorithms (DXF geometry, heatmaps, etc.)

### ✔ Simulation Components
- Do not hard‑code dimensions; always use config values
- Keep loaders pure (NO side effects)
- Avoid heavy work in visualization loop
- Ensure new scenarios integrate through `scenarios.py`

### ✔ DXF / Raster Processing
- Keep thresholds in `config.py`
- Test on several maps
- Use metadata for accurate overlays

---

# 📦 Pull Request Requirements

Your PR should include:
- Description of changes
- Before/after behavior if visual
- Performance impact (if any)
- Updated docs if feature changes UX

---

# 🤖 Good First Issues

New contributors can work on:
- Improving wall/door detection for PNG maps
- Adding new map overlays
- Optimizing crowded simulations
- Adding evacuation KPIs
- Creating map preview mode
- Improving scenario presets

---

# 🙏 Contributor Expectations

- Be kind and respectful  
- Follow the Code of Conduct  
- Help reviewers understand your changes  
- Be patient with review cycles  

---

# 💬 Need Help?

Open a GitHub issue or start a discussion—maintainers and contributors will help.

---

# ❤️ Thank You

Your contributions help build a powerful **research‑grade** crowd simulation engine that supports PNG/DXF maps, evacuation analytics, and bottleneck detection.

We’re glad to have you here!

