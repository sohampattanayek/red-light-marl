# 🚦 Red Light MARL

A **Multi-Agent Reinforcement Learning (MARL)** project simulating the dynamics of the "Red Light, Green Light" game. This repository explores how multiple autonomous agents learn to navigate a shared environment with global constraints and individual goals using a **Centralized Training, Decentralized Execution (CTDE)** framework.

---

## 🚀 Project Overview
This project implements a reinforcement learning environment where five agents compete to reach a finish line. The simulation is built on a single **Proximal Policy Optimization (PPO)** model that controls the agents, forcing them to adapt to a shared global state.

* **The Challenge:** Agents must learn to maximize speed during "Green Light" phases while coming to a complete halt during "Red Light" phases.
* **The Penalty:** Moving during a red light results in heavy penalties or a position reset, forcing agents to develop "patience" through reinforcement.

## 📂 Repository Structure
* **`red_light_marl.py`**: The core environment logic. Contains the `CentralizedRedLightEnv` class, defining the state space, action space, and reward functions.
* **`red_light_marl_vis.py`**: The visualization engine. Used to render the environment and observe the agents' learned behaviors in real-time.
* **`.gitignore`**: Standard configuration to prevent tracking of unnecessary local files.

## 🛠️ Installation
To run this simulation locally, clone the repository and ensure you have your Python environment ready.

```bash
# Clone the repository
git clone [https://github.com/sohampattanayek/red-light-marl.git](https://github.com/sohampattanayek/red-light-marl.git)

# Navigate into the project directory
cd red-light-marl
