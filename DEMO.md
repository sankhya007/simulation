# 1. Normal scenario with visualization
python main.py normal

# 2. High-density stress test (many agents)
python main.py high_density

# 3. Blocked paths / dynamic obstacles scenario
python main.py blocked

# 4. Emergency evacuation scenario (shows exits + bottlenecks + KPIs)
python main.py evacuation

# 5. Batch experiment on normal scenario (5 runs, averaged stats)
python experiment.py normal 5

# 6. Batch experiment on evacuation scenario (5 runs, evacuation KPIs)
python experiment.py evacuation 5

# 7. AI strategy comparison on normal scenario (3 runs per strategy)
python experiment.py compare_strategies normal 3

# 8. AI strategy comparison on evacuation scenario
python experiment.py compare_strategies evacuation 3
