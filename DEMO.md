# 1. Normal navigation, visual demo
python main.py normal

# 2. High-density stress test (lots of agents, same map)
python main.py high_density

# 3. Blocked paths scenario (dynamic obstacles, rerouting)
python main.py blocked

# 4. Emergency evacuation scenario (border exits, EVACUATION_MODE)
python main.py evacuation


### experiments, no gui, just metrics

# 5. Baseline experiment: normal scenario, 5 runs (aggregated metrics)
python experiment.py normal 5

# 6. Evacuation experiment: evacuation scenario, 5 runs
python experiment.py evacuation 5

# 7. AI strategy comparison on evacuation scenario
python experiment.py compare evacuation 5
