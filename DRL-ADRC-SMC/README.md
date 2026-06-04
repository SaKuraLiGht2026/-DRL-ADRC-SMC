# DRL-ADRC-SMC: Deep Reinforcement Learning-Based Adaptive ADRC with Sliding Mode Compensation

A framework for online adaptive tuning of Active Disturbance Rejection Control (ADRC) parameters using Soft Actor-Critic (SAC) reinforcement learning, with nonsingular terminal sliding mode control (NTSMC) compensation for robotic manipulator trajectory tracking.

## Architecture

```
DRL Layer (SAC Agent)
    ↓ ω_o, ω_c (residual output)
Control Layer (ADRC + SMC)
    ↓ torque u
Physical Layer (Franka Panda Robot Arm)
    ↑ state feedback (q, q̇)
```

## Project Structure

```
drl_adrc_robot/
├── controllers/
│   ├── adrc.py              # ADRC controller with ESO
│   ├── pid.py               # PID controller (baseline)
│   ├── sliding_mode.py      # Nonsingular terminal SMC
│   ├── composite.py         # ADRC-SMC composite controller
│   └── neural_eso.py        # Neural ESO (experimental)
├── envs/
│   └── robot_arm_env.py     # Custom Gym environment (PyBullet Panda)
├── train/
│   └── train_sac.py         # SAC training script
├── eval/
│   ├── evaluate.py          # Method comparison evaluation
│   ├── disturbance_test.py  # Disturbance robustness test
│   └── full_evaluation.py   # Complete evaluation (all tables)
├── configs/
│   └── default.py           # Hyperparameters and configurations
├── utils/
│   ├── trajectory.py        # Trajectory generators
│   └── plotting.py          # Plotting utilities
├── results/                 # Saved models and results
├── run_all.py               # Run all experiments
├── run_eval_only.py         # Run evaluation only
├── submit_job.sh            # Slurm job submission script
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch >= 2.0
- stable-baselines3 >= 2.0
- gymnasium
- pybullet
- matplotlib
- numpy, scipy

## Installation

```bash
conda create -n drl_adrc python=3.10
conda activate drl_adrc
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install 'stable-baselines3[extra]' gymnasium pybullet matplotlib
```

## Quick Start

### Training

Train SAC agent with different configurations:

```bash
# DRL-ADRC with residual parameter output (recommended)
python train/train_sac.py --mode adrc --residual --timesteps 200000

# DRL-ADRC with absolute parameter output (ablation)
python train/train_sac.py --mode adrc --no_residual --timesteps 200000

# DRL-ADRC-SMC composite controller
python train/train_sac.py --mode composite --residual --timesteps 200000
```

### Evaluation

```bash
# Full evaluation (generates all 5 tables from the paper)
python eval/full_evaluation.py

# Run everything (training + evaluation)
python run_all.py
```

### Slurm Submission (GPU Cluster)

```bash
sbatch submit_job.sh
```

## Method Overview

### ADRC Controller
- Extended State Observer (ESO) estimates position, velocity, and total disturbance
- Bandwidth parameterization: observer bandwidth ω_o, controller bandwidth ω_c
- PD state error feedback with disturbance compensation

### SAC Agent
- Observation: [tracking error, error derivative, joint position, joint velocity, reference position, reference velocity, reference acceleration, disturbance estimate]
- Action: [δω_o, δω_c] ∈ [-1, 1]² (residual adjustments to default ADRC parameters)
- Reward: tracking accuracy + energy penalty + chattering penalty + precision bonus

### Sliding Mode Compensation
- Nonsingular terminal sliding mode surface
- ESO disturbance estimate as feedforward compensation
- Weighted combination: u = (1-α)·u_ADRC + α·u_SMC

## Results

### Tracking Performance (Nominal Conditions)

| Method | RMSE (rad) | Chattering |
|--------|-----------|------------|
| PID | 0.1546 | 1.65 |
| Fixed-ADRC | 0.1409 | 5.46 |
| **DRL-ADRC (ours)** | **0.0885** | 5.02 |
| DRL-ADRC-SMC (ours) | 0.1464 | **1.48** |

### Disturbance Robustness (RMSE)

| Scenario | PID | Fixed-ADRC | DRL-ADRC | DRL-ADRC-SMC |
|----------|-----|-----------|----------|-------------|
| No disturbance | 0.1546 | 0.1409 | **0.1144** | 0.1464 |
| External force | 0.1604 | 0.1412 | **0.1334** | 0.1469 |
| Load change | 0.1765 | 0.1461 | **0.1221** | 0.1640 |
| Friction change | 0.1910 | 0.1499 | **0.1280** | 0.1714 |

## Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026drl_adrc_smc,
  title={Deep Reinforcement Learning-Based Adaptive Parameter Tuning for Active Disturbance Rejection Control with Sliding Mode Compensation in Robotic Manipulators},
  author={Chen, Guangzhao},
  journal={},
  year={2026}
}
```

## License

MIT License
