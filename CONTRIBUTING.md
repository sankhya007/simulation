# Contributing to the Crowd Simulation Project

Thank you for your interest in contributing!
This project aims to build a powerful, modular, and realistic crowd simulation engine supporting:

- Pedestrian dynamics
- DXF & raster-based floorplans
- Graph & continuous motion models (Social Force, RVO)
- Agent heterogeneity
- Panic propagation
- Visualization & analytics tools

We welcome contributions from developers, researchers, students, and enthusiasts of all backgrounds.

## Table of Contents
1. How to Ask Questions
2. Code of Conduct
3. Development Environment Setup
4. Project Structure
5. How to Contribute
6. Coding Standards
7. Pull Request Process
8. Writing & Running Tests
9. Reporting Bugs
10. Suggesting Features
11. Community & Recognition

## How to Ask Questions
If you're unsure about anything:
- Open a GitHub Discussion
- Or open an Issue labeled "question"
- Or contact maintainers directly (if available)

No question is too small — we want this project to be accessible for newcomers.

## Code of Conduct
By participating in this project, you agree to uphold our
- CODE_OF_CONDUCT.md

Please read it carefully before contributing.

## Development Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/<yourname>/crowd-simulation.git
cd crowd-simulation
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Activate:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the test suite
```bash
pytest -q
```

## Project Structure
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

## How to Contribute
### Code contributions
- Fix bugs
- Add new features
- Improve motion models
- Add support for new map formats
- Optimize simulation speed
- Improve agent behavior/AI

### Documentation
- Improve README / tutorials
- Add diagrams or explanations
- Extend DEMO.md with screenshots

### Testing
- Add tests for new or existing modules

### Research contributions
- Implement theoretical models (e.g., Social Force variants, panic models)
- Add benchmarking tools or metrics

### Community support
- Answer questions
- Help maintain discussions
- Provide feedback on PRs

## Coding Standards
- Use PEP 8 compliant code
- Use type hints everywhere possible
- Keep functions short and readable
- Avoid hardcoded values (use config.py)
- Document complex logic clearly

## Pull Request Process
### 1. Create a feature branch
```bash
git checkout -b feature/<short-description>
```

### 2. Write clear, documented code

### 3. Ensure tests pass
```bash
pytest -q
```

### 4. Open a PR
Your pull request should include:
- A descriptive title
- Summary of changes
- Screenshots or animations if visual
- Any performance considerations
- List of new test cases

## Writing & Running Tests
Run the full suite:
```bash
pytest
```

## Reporting Bugs
Open a GitHub Issue with:
- Clear title
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if visual)
- Relevant config values
- OS + Python version
- Logs or traceback

## Suggesting Features
Include:
- Use case
- Why this feature matters
- Potential approach
- References (papers, models, links)

## Community & Recognition
We appreciate every contributor — from beginners to experts.
All contributors will be listed in the README under **Contributors**.

## Contact
Maintainer Email: add-your-email@example.com
