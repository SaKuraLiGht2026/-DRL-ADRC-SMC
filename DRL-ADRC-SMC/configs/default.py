"""Default configuration for experiments."""

TRAIN_CONFIG = {
    "algorithm": "SAC",
    "total_timesteps": 200_000,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": "auto",
    "n_envs": 1,
    "seed": 42,
}

ENV_CONFIG = {
    "sim_dt": 1.0 / 240.0,
    "ctrl_dt": 0.005,
    "episode_length": 5.0,
    "control_mode": "adrc",  # "adrc" or "composite"
}

EVAL_CONFIG = {
    "n_eval_episodes": 10,
    "seeds": [42, 123, 456],
}

# Disturbance scenarios for robustness testing
DISTURBANCE_CONFIGS = {
    "none": {},
    "external_force": {
        "external_force": True,
        "force_magnitude": 5.0,
    },
    "load_change": {
        "load_change": True,
        "extra_mass": 2.0,
    },
    "friction_change": {
        "friction_change": True,
        "extra_friction": 3.0,
    },
}

# Fixed ADRC parameters for baseline comparison
FIXED_ADRC_PARAMS = {
    "conservative": {"omega_o": 10.0, "omega_c": 5.0},
    "moderate": {"omega_o": 20.0, "omega_c": 10.0},
}
