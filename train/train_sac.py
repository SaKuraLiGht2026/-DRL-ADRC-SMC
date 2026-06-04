"""
Train SAC agent to adaptively tune ADRC parameters.
Supports: residual RL, neural ESO, composite control.

Usage:
    python train/train_sac.py --mode adrc --residual --timesteps 200000
    python train/train_sac.py --mode adrc --neural_eso --residual --timesteps 200000
    python train/train_sac.py --mode composite --residual --timesteps 200000
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.robot_arm_env import RobotArmTrackingEnv
from configs.default import TRAIN_CONFIG, ENV_CONFIG


def make_env(control_mode="adrc", use_neural_eso=False, use_residual=True,
             disturbance_config=None):
    def _init():
        env = RobotArmTrackingEnv(
            control_mode=control_mode,
            use_neural_eso=use_neural_eso,
            use_residual=use_residual,
            sim_dt=ENV_CONFIG["sim_dt"],
            ctrl_dt=ENV_CONFIG["ctrl_dt"],
            episode_length=ENV_CONFIG["episode_length"],
            disturbance_config=disturbance_config,
        )
        env = Monitor(env)
        return env
    return _init


def train(args):
    tag = f"{args.mode}"
    if args.residual:
        tag += "_residual"
    if args.neural_eso:
        tag += "_neso"

    print(f"Training SAC | mode={args.mode} | residual={args.residual} "
          f"| neural_eso={args.neural_eso} | timesteps={args.timesteps}")

    env = make_env(
        control_mode=args.mode,
        use_neural_eso=args.neural_eso,
        use_residual=args.residual,
    )()

    eval_env = make_env(
        control_mode=args.mode,
        use_neural_eso=args.neural_eso,
        use_residual=args.residual,
    )()

    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", f"sac_{tag}"
    )
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=save_dir,
        name_prefix="sac_checkpoint",
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        batch_size=TRAIN_CONFIG["batch_size"],
        buffer_size=TRAIN_CONFIG["buffer_size"],
        learning_starts=TRAIN_CONFIG["learning_starts"],
        gamma=TRAIN_CONFIG["gamma"],
        tau=TRAIN_CONFIG["tau"],
        ent_coef=TRAIN_CONFIG["ent_coef"],
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
        device="auto",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, "sac_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="adrc",
                        choices=["adrc", "composite"])
    parser.add_argument("--residual", action="store_true", default=True,
                        help="Use residual RL (default: True)")
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--neural_eso", action="store_true", default=False)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.no_residual:
        args.residual = False
    train(args)
