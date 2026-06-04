"""
Full evaluation: 8 methods comparison.
1. PID
2. Pure SMC
3. Fixed-ADRC (conservative)
4. Fixed-ADRC (moderate)
5. DRL-ADRC (absolute params) - ablation
6. DRL-ADRC (residual)
7. DRL-ADRC + Neural ESO - core method
8. DRL-ADRC-SMC (full)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pybullet as p
import pybullet_data

from stable_baselines3 import SAC
from controllers.adrc import ADRC
from controllers.pid import PIDController
from controllers.sliding_mode import NonsingularTerminalSMC
from controllers.composite import CompositeController
from controllers.neural_eso import NeuralADRC
from envs.robot_arm_env import RobotArmTrackingEnv
from configs.default import FIXED_ADRC_PARAMS, ENV_CONFIG
from utils.plotting import (
    plot_tracking_comparison, plot_torque_comparison, plot_adrc_params
)


def run_classic_controller(controller, disturbance_config=None,
                           episode_length=5.0, ctrl_dt=0.005, sim_dt=1.0/240.0):
    """Run a classic controller (PID, SMC, or fixed ADRC) without RL."""
    ctrl_steps = int(ctrl_dt / sim_dt)
    max_steps = int(episode_length / ctrl_dt)

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(sim_dt)

    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf",
                          basePosition=[0, 0, 0], useFixedBase=True)
    controlled_joint = 0
    p.resetJointState(robot_id, controlled_joint, 0.0, 0.0)
    for i in range(p.getNumJoints(robot_id)):
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)

    controller.reset()
    dist_config = disturbance_config or {}

    records = {"time": [], "ref": [], "actual": [], "error": [],
               "torque": [], "omega_o": [], "omega_c": []}

    freq, amp = 0.5, 0.8
    prev_torque = 0.0

    for step in range(max_steps):
        t = step * ctrl_dt
        ref_pos = amp * np.sin(2 * np.pi * freq * t)
        ref_vel = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
        ref_acc = -amp * (2 * np.pi * freq) ** 2 * np.sin(2 * np.pi * freq * t)

        js = p.getJointState(robot_id, controlled_joint)
        joint_pos, joint_vel = js[0], js[1]

        if isinstance(controller, NonsingularTerminalSMC):
            e = ref_pos - joint_pos
            ed = ref_vel - joint_vel
            torque, _ = controller.compute(e, ed, ref_acc)
        elif isinstance(controller, CompositeController):
            torque, _ = controller.compute(joint_pos, ref_pos, ref_vel, ref_acc, joint_vel)
        else:
            torque = controller.compute(joint_pos, ref_pos, ref_vel, ref_acc)

        # Apply disturbance
        ext_dist = 0.0
        if dist_config.get("external_force", False) and 2.0 < t < 3.5:
            ext_dist += dist_config.get("force_magnitude", 5.0)
        if dist_config.get("load_change", False) and t > 2.5:
            ext_dist += -dist_config.get("extra_mass", 2.0) * joint_vel * 0.5
        if dist_config.get("friction_change", False) and t > 1.5:
            ext_dist += -dist_config.get("extra_friction", 3.0) * np.sign(joint_vel)

        for _ in range(ctrl_steps):
            p.setJointMotorControl2(robot_id, controlled_joint,
                                    p.TORQUE_CONTROL, force=torque + ext_dist)
            p.stepSimulation()

        records["time"].append(t)
        records["ref"].append(ref_pos)
        records["actual"].append(joint_pos)
        records["error"].append(ref_pos - joint_pos)
        records["torque"].append(torque)
        records["omega_o"].append(0)
        records["omega_c"].append(0)

    p.disconnect(client)
    return records


def run_drl_model(model_path, control_mode="adrc", use_neural_eso=False,
                  use_residual=True, disturbance_config=None,
                  episode_length=5.0, ctrl_dt=0.005):
    """Run trained DRL model."""
    env = RobotArmTrackingEnv(
        control_mode=control_mode,
        use_neural_eso=use_neural_eso,
        use_residual=use_residual,
        episode_length=episode_length,
        ctrl_dt=ctrl_dt,
        disturbance_config=disturbance_config or {},
    )
    model = SAC.load(model_path)
    obs, _ = env.reset()

    records = {"time": [], "ref": [], "actual": [], "error": [],
               "torque": [], "omega_o": [], "omega_c": []}

    done = False
    step = 0
    while not done:
        t = step * ctrl_dt
        ref_pos, _, _ = env._generate_reference(t)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        js = p.getJointState(env.robot_id, env.controlled_joint)
        records["time"].append(t)
        records["ref"].append(ref_pos)
        records["actual"].append(js[0])
        records["error"].append(ref_pos - js[0])
        records["torque"].append(info["torque"])
        records["omega_o"].append(info["omega_o"])
        records["omega_c"].append(info["omega_c"])
        step += 1

    env.close()
    return records


def compute_metrics(records):
    errors = np.array(records["error"])
    torques = np.array(records["torque"])
    dt = records["time"][1] - records["time"][0] if len(records["time"]) > 1 else 0.005

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    iae = np.sum(np.abs(errors)) * dt
    torque_rms = np.sqrt(np.mean(torques ** 2))
    chattering = np.sqrt(np.mean(np.diff(torques) ** 2))

    return {
        "RMSE": rmse, "MAE": mae, "Max Error": max_error,
        "IAE": iae, "Torque RMS": torque_rms, "Chattering": chattering,
    }


def main(args):
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "evaluation"
    )
    os.makedirs(result_dir, exist_ok=True)

    results = {}

    # 1. PID
    print("Running PID...")
    results["PID"] = run_classic_controller(PIDController(kp=50, ki=5, kd=10, dt=0.005))

    # 2. Pure SMC
    print("Running Pure SMC...")
    results["Pure SMC"] = run_classic_controller(
        NonsingularTerminalSMC(eta=5.0, phi=0.05, dt=0.005))

    # 3-4. Fixed ADRC baselines
    for name, params in FIXED_ADRC_PARAMS.items():
        print(f"Running Fixed ADRC ({name})...")
        ctrl = ADRC(omega_o=params["omega_o"], omega_c=params["omega_c"], dt=0.005)
        results[f"Fixed-ADRC ({name})"] = run_classic_controller(ctrl)

    # 5. DRL-ADRC (absolute params, no residual) - ablation
    abs_model = "results/sac_adrc/sac_final"
    abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), abs_model)
    if os.path.exists(abs_path + ".zip"):
        print("Running DRL-ADRC (absolute)...")
        results["DRL-ADRC (absolute)"] = run_drl_model(
            abs_path, control_mode="adrc", use_residual=False)

    # 6. DRL-ADRC (residual)
    res_model = "results/sac_adrc_residual/sac_final"
    res_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), res_model)
    if os.path.exists(res_path + ".zip"):
        print("Running DRL-ADRC (residual)...")
        results["DRL-ADRC (residual)"] = run_drl_model(
            res_path, control_mode="adrc", use_residual=True)

    # 7. Neural ESO skipped (requires separate pretraining)

    # 8. DRL-ADRC-SMC (full)
    comp_model = "results/sac_composite_residual/sac_final"
    comp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), comp_model)
    if os.path.exists(comp_path + ".zip"):
        print("Running DRL-ADRC-SMC...")
        results["DRL-ADRC-SMC (Ours)"] = run_drl_model(
            comp_path, control_mode="composite", use_residual=True)

    # Plot
    print("\nGenerating plots...")
    plot_tracking_comparison(results,
        os.path.join(result_dir, "tracking_comparison.png"),
        title="Trajectory Tracking Comparison")
    plot_torque_comparison(results,
        os.path.join(result_dir, "torque_comparison.png"))
    plot_adrc_params(results,
        os.path.join(result_dir, "adrc_params.png"))

    # Metrics table
    print("\n" + "=" * 90)
    print(f"{'Method':<30} {'RMSE':>8} {'MAE':>8} {'MaxErr':>8} {'IAE':>8} {'Chatter':>8}")
    print("-" * 90)
    for name, records in results.items():
        m = compute_metrics(records)
        print(f"{name:<30} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
              f"{m['Max Error']:>8.4f} {m['IAE']:>8.4f} {m['Chattering']:>8.4f}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adrc_model", type=str, default="")
    parser.add_argument("--composite_model", type=str, default="")
    args = parser.parse_args()
    main(args)
