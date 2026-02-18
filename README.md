Red Light MARL
A Multi-Agent Reinforcement Learning (MARL) project simulating the classic "Red Light, Green Light" game. This repository explores how multiple agents learn to navigate a shared environment with global constraints and individual goals.

🚀 Overview
This project implements a reinforcement learning environment where multiple agents must reach a finish line while adhering to a "Red Light" signal. If an agent moves during a red light, it faces penalties or is reset. The agents must learn to balance speed with safety using MARL algorithms.

📂 Repository Structure
red_light_marl.py: The core environment and training logic. This script defines the agent behaviors, reward structures, and the MARL training loop.

red_light_marl_vis.py: A visualization script to render the environment and observe agent performance in real-time.

.gitignore: Standard Python gitignore to keep the repository clean of bytecode and local environment files.

🛠️ Installation
To run this simulation locally, clone the repository and install the necessary dependencies (typically Python 3.x with numpy, matplotlib, and a MARL framework like PettingZoo or Ray Rllib depending on your specific implementation).

Bash

git clone https://github.com/sohampattanayek/red-light-marl.git
cd red-light-marl
🎮 Usage
Training the Agents
To begin training the agents in the environment:

Bash

python red_light_marl.py
Visualizing the Results
To see the trained agents in action:

Bash

python red_light_marl_vis.py
🧠 Key Features
Multi-Agent Coordination: Agents must account for the global state (the light color) while pursuing individual progress.

Custom Reward Shaping: Implements penalties for movement during red lights and rewards for reaching the goal efficiently.

Dynamic Simulation: The environment provides visual feedback on agent decision-making processes.

Custom Non-Commercial License for red-light-marl

Copyright (c) 2026 Soham Pattanayek. All rights reserved.

Permission is granted to view and study this source code for personal and educational purposes only.

You may not:
- Copy, reuse, or redistribute this project or any portion of it.
- Modify or adapt the design, code, or assets for your own website or publication.
- Use this project for commercial purposes, including selling, hosting, or promoting derivative works.

Any form of reproduction or redistribution of this project’s content without written permission from the author is strictly prohibited.

This software and its contents are provided “as is,” without warranty of any kind, express or implied.

For permission requests or inquiries, contact: sohampattanayek1234@gmail.com
