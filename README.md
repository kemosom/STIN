# Lightweight Q-Exchange Multi-Agent Reinforcement Learning for QoE-Driven Delay-Tolerant Routing in Satellite-Terrestrial Integrated Networks

This repository contains the simulation framework for the Hybrid Q-Exchange Multi-Agent Reinforcement Learning (MARL) routing protocol in Space-Terrestrial Integrated Networks (STIN).

## Overview
The simulation framework models a highly dynamic LEO satellite-terrestrial network. It provides:
- A custom Gym-compliant environment (`STINEnv`) that handles orbital mechanics (via SGP4/TLE data), link budgeting, and dynamic queue dynamics.
- Implementation of the proposed **Hybrid Q-Exchange** MARL algorithm.
- Implementations of several routing baselines: Centralized MAPPO, Distributed IQL, OSPF, and Contact Graph Routing (CGR).
- Automated experimental sweeps across scalability, mobility (speed), and traffic load.

## Setup and Installation

This framework is built using Python 3 and relies on standard data science and RL libraries. It is fully cross-platform and natively supported on Ubuntu, Windows, and macOS.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd STIN
   ```

2. **Install the dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Running the Simulation

The primary entry point is `run_experiments.py`, which provides a command-line interface (CLI) to configure the training steps, evaluation steps, and the type of experiments to run.

### Basic Usage
To run the full suite of experiments (Scalability, Mobility, and Traffic-Load) with the default settings (1000 training episodes, 50 evaluation episodes):
```bash
python run_experiments.py
```

### CLI Options
You can customize the simulation parameters using the following flags:
- `--train_steps`: Number of training episodes per configuration (default: 1000). Set to 0 if you want to skip training.
- `--eval_steps`: Number of evaluation episodes for testing (default: 50).
- `--sweep`: Which experimental sweep to run. Choices are `scalability`, `mobility`, `load`, or `all` (default: `all`).
- `--alg`: Comma-separated list of algorithms to evaluate. Choices include `Proposed,Centralized,Distributed,CGR,OSPF`. (default: `all`).

**Example: Running only the Traffic-Load sweep for the Proposed and Distributed algorithms with 500 training steps:**
```bash
python run_experiments.py --sweep load --alg Proposed,Distributed --train_steps 500 --eval_steps 20
```

### Generating Plots
After running the experiments, CSV files will be saved in the `results/` directory. To generate the final figures shown in the paper:
```bash
python scripts/plot_results.py
```
Figures will be saved as high-resolution PNGs in the `figures/` folder.

## Repository Structure
- `envs/`: Contains `stin_env.py`, the core custom OpenAI Gym environment.
- `agents/`: Contains the algorithm implementations (`hybrid_q_agent.py` and `baselines.py`).
- `scripts/`: Contains `plot_results.py` for visualization.
- `run_experiments.py`: The main CLI script to execute the simulation.
- `gp.php` / `stations.txt`: Real-world TLE (Two-Line Element) orbital data for the satellite constellation.

## Citation
If you use this code in your research, please consider citing our IEEE OJ-COMS paper.
