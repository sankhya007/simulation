# Contributing Guidelines

Thank you for your interest in contributing to the **Graph-Based Crowd Simulation with Agent AI** project.

This repository is primarily developed for **academic, research, and educational purposes**. Contributions are welcome as long as they maintain code quality, reproducibility, and academic integrity.

---

## 1. Who Can Contribute

Contributions are welcome from:
- Students and researchers
- Contributors interested in crowd simulation and multi-agent systems
- Developers improving performance, robustness, or documentation

Both code and non-code contributions (documentation, experiments, diagrams) are equally valuable.

---

## 2. Types of Contributions

You may contribute in the following ways:

### Code Contributions
- Bug fixes
- Performance optimizations
- New agent behaviors or navigation strategies
- New scenarios or experimental setups
- Improvements to metrics, analysis, or visualization

### Research & Experimentation
- Additional evaluation metrics
- Scenario comparisons
- Sensitivity or ablation studies
- Documentation of experimental results

### Documentation
- Improving README clarity
- Adding diagrams or explanations
- Writing usage examples or tutorials

---

## 3. Contribution Workflow

### Step 1: Fork the Repository
Create your own fork of the repository on GitHub.

### Step 2: Create a Feature Branch
```bash
git checkout -b feature/short-description
```

Use descriptive branch names such as:
- `feature/safe-navigation-strategy`
- `fix/congestion-weight-bug`
- `docs/architecture-diagram`

### Step 3: Make Your Changes
- Keep commits focused and minimal
- Do not mix unrelated changes in a single commit
- Write clean, readable, well-documented code

---

## 4. Commit with Clear Messages

Use clear and descriptive commit messages.

Examples:
```text
feat(agent): add safe navigation strategy
fix(simulation): prevent deadlock at blocked exits
docs(readme): add evaluation methodology section
```

---

## 5. Open a Pull Request

When opening a Pull Request (PR), include:
- A brief summary of changes
- Motivation and expected impact
- Any new configuration options
- Screenshots or plots (if applicable)

---

## 6. Coding Standards

### General Guidelines
- Follow **PEP 8** style conventions
- Use meaningful variable and function names
- Favor readability over micro-optimizations

### Configuration Management
- Do not hardcode constants
- All tunable parameters must live in `config.py`
- Scenario logic belongs in `scenarios.py`

### Module Responsibilities
- `agent.py` → agent logic & behavior
- `environment.py` → graph & congestion modeling
- `simulation.py` → time-step loop & metrics
- `analysis.py` → metrics computation & plotting
- `visualization.py` → rendering & overlays

Avoid cross-cutting logic across unrelated modules.

---

## 7. Adding New Features

### Adding a New Navigation Strategy
- Extend behavior in `agent.py`
- Register the strategy name in `config.py`
- Ensure it appears in per-strategy metrics
- Add at least one comparison plot

### Adding a New Scenario
- Define the scenario in `scenarios.py`
- Clearly describe expected behavior
- Test with `experiment.py`
- Document metrics affected by the scenario

---

## 8. Testing & Verification

Before submitting contributions, run:

```bash
python main.py
python experiment.py normal 3
```

Ensure that:
- The simulation runs without errors
- Plots and metrics are sensible
- No import or runtime issues exist

---

## 9. Documentation Requirements

If your contribution changes behavior or results:
- Update the relevant documentation
- Explain why the change was made
- Mention any assumptions or limitations

Code without documentation may be requested to be revised.

---

## 10. Academic Integrity & Ethics

This project follows strict academic standards.

Contributors must:
- Avoid plagiarized code
- Properly acknowledge external ideas or algorithms
- Avoid falsifying or selectively reporting results
- Make experiments reproducible wherever possible

Violations of academic integrity may result in removal of contributions.

---

## 11. Communication & Conduct

All contributors must follow the project’s **Code of Conduct**.

Be respectful, constructive, and professional in all discussions.

---

## 12. Final Notes

By contributing to this project, you agree that:
- Your contributions may be reviewed and modified
- Your code may be used for academic and educational purposes
- You are contributing in good faith to improve research quality

We appreciate all thoughtful contributions that help improve this project.
