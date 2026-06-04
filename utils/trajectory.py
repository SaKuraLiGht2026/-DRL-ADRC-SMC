"""Trajectory generation utilities."""
import numpy as np


def sinusoidal_trajectory(t, freq=0.5, amp=0.8):
    pos = amp * np.sin(2 * np.pi * freq * t)
    vel = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
    acc = -amp * (2 * np.pi * freq) ** 2 * np.sin(2 * np.pi * freq * t)
    return pos, vel, acc


def step_trajectory(t, step_time=1.0, step_value=1.0):
    if t < step_time:
        return 0.0, 0.0, 0.0
    else:
        return step_value, 0.0, 0.0


def multi_freq_trajectory(t):
    """Multi-frequency trajectory for generalization test."""
    pos = 0.5 * np.sin(2 * np.pi * 0.3 * t) + 0.3 * np.sin(2 * np.pi * 0.7 * t)
    vel = 0.5 * 2 * np.pi * 0.3 * np.cos(2 * np.pi * 0.3 * t) + \
          0.3 * 2 * np.pi * 0.7 * np.cos(2 * np.pi * 0.7 * t)
    acc = -0.5 * (2 * np.pi * 0.3) ** 2 * np.sin(2 * np.pi * 0.3 * t) - \
          0.3 * (2 * np.pi * 0.7) ** 2 * np.sin(2 * np.pi * 0.7 * t)
    return pos, vel, acc
