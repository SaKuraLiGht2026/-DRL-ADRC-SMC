"""
Full evaluation for paper: all tables with real data.
Table 1: Tracking performance (6 methods, nominal)
Table 2: Disturbance RMSE (4 methods x 4 scenarios)
Table 3: Chattering & control effort (4 methods x 4 scenarios)
Table 4: Ablation study (5 configs, nominal)
Table 5: Multi-frequency trajectory benchmark (new dataset)
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p
import pybullet_data
from stable_baselines3 import SAC
from controllers.adrc import ADRC
from controllers.pid import PIDController
from controllers.sliding_mode import NonsingularTerminalSMC
from controllers.composite import CompositeController
from envs.robot_arm_env import RobotArmTrackingEnv


def run_classic(controller, episode_length=5.0, ctrl_dt=0.005, sim_dt=1.0/240.0,
                disturbance_config=None, trajectory="sinusoidal"):
    ctrl_steps = int(ctrl_dt / sim_dt)
    max_steps = int(episode_length / ctrl_dt)
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(sim_dt)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0,0,0], useFixedBase=True)
    p.resetJointState(robot_id, 0, 0.0, 0.0)
    for i in range(p.getNumJoints(robot_id)):
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)
    controller.reset()
    dist_cfg = disturbance_config or {}

    torques, errors = [], []
    for step in range(max_steps):
        t = step * ctrl_dt
        if trajectory == "sinusoidal":
            freq, amp = 0.5, 0.8
            ref_pos = amp * np.sin(2*np.pi*freq*t)
            ref_vel = amp * 2*np.pi*freq * np.cos(2*np.pi*freq*t)
            ref_acc = -amp * (2*np.pi*freq)**2 * np.sin(2*np.pi*freq*t)
        elif trajectory == "multifreq":
            ref_pos = 0.5*np.sin(2*np.pi*0.3*t) + 0.3*np.sin(2*np.pi*0.8*t) + 0.15*np.sin(2*np.pi*1.5*t)
            ref_vel = 0.5*2*np.pi*0.3*np.cos(2*np.pi*0.3*t) + 0.3*2*np.pi*0.8*np.cos(2*np.pi*0.8*t) + 0.15*2*np.pi*1.5*np.cos(2*np.pi*1.5*t)
            ref_acc = -0.5*(2*np.pi*0.3)**2*np.sin(2*np.pi*0.3*t) - 0.3*(2*np.pi*0.8)**2*np.sin(2*np.pi*0.8*t) - 0.15*(2*np.pi*1.5)**2*np.sin(2*np.pi*1.5*t)

        js = p.getJointState(robot_id, 0)
        jp, jv = js[0], js[1]

        if isinstance(controller, NonsingularTerminalSMC):
            torque, _ = controller.compute(ref_pos - jp, ref_vel - jv, ref_acc)
        elif isinstance(controller, CompositeController):
            torque, _ = controller.compute(jp, ref_pos, ref_vel, ref_acc, jv)
        else:
            torque = controller.compute(jp, ref_pos, ref_vel, ref_acc)

        # Disturbance
        ext = 0.0
        if dist_cfg.get("external_force") and 2.0 < t < 3.5:
            ext += dist_cfg.get("force_magnitude", 5.0)
        if dist_cfg.get("load_change") and t > 2.5:
            ext += -dist_cfg.get("extra_mass", 2.0) * jv * 0.5
        if dist_cfg.get("friction_change") and t > 1.5:
            ext += -dist_cfg.get("extra_friction", 3.0) * np.sign(jv)

        for _ in range(ctrl_steps):
            p.setJointMotorControl2(robot_id, 0, p.TORQUE_CONTROL, force=torque + ext)
            p.stepSimulation()

        errors.append(ref_pos - jp)
        torques.append(float(torque))

    p.disconnect(client)
    return compute_all_metrics(errors, torques, ctrl_dt)


def run_drl(model_path, control_mode="adrc", use_residual=True,
            disturbance_config=None, trajectory="sinusoidal"):
    # Monkey-patch trajectory if multifreq
    env = RobotArmTrackingEnv(
        control_mode=control_mode, use_residual=use_residual,
        ctrl_dt=0.005, disturbance_config=disturbance_config or {}
    )
    if trajectory == "multifreq":
        def _gen_ref(self_env, t):
            pos = 0.5*np.sin(2*np.pi*0.3*t) + 0.3*np.sin(2*np.pi*0.8*t) + 0.15*np.sin(2*np.pi*1.5*t)
            vel = 0.5*2*np.pi*0.3*np.cos(2*np.pi*0.3*t) + 0.3*2*np.pi*0.8*np.cos(2*np.pi*0.8*t) + 0.15*2*np.pi*1.5*np.cos(2*np.pi*1.5*t)
            acc = -0.5*(2*np.pi*0.3)**2*np.sin(2*np.pi*0.3*t) - 0.3*(2*np.pi*0.8)**2*np.sin(2*np.pi*0.8*t) - 0.15*(2*np.pi*1.5)**2*np.sin(2*np.pi*1.5*t)
            return pos, vel, acc
        import types
        env._generate_reference = types.MethodType(_gen_ref, env)

    model = SAC.load(model_path)
    obs, _ = env.reset()
    errors, torques = [], []
    done = False
    step = 0
    while not done:
        t = step * 0.005
        if trajectory == "sinusoidal":
            ref_pos = 0.8 * np.sin(2*np.pi*0.5*t)
        else:
            ref_pos = 0.5*np.sin(2*np.pi*0.3*t) + 0.3*np.sin(2*np.pi*0.8*t) + 0.15*np.sin(2*np.pi*1.5*t)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        js = p.getJointState(env.robot_id, env.controlled_joint)
        errors.append(ref_pos - js[0])
        torques.append(float(info["torque"]))
        step += 1
    env.close()
    return compute_all_metrics(errors, torques, 0.005)


def compute_all_metrics(errors, torques, dt):
    e = np.array(errors)
    u = np.array(torques)
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "MaxErr": float(np.max(np.abs(e))),
        "IAE": float(np.sum(np.abs(e)) * dt),
        "Chattering": float(np.sqrt(np.mean(np.diff(u)**2))),
        "Effort": float(np.sqrt(np.mean(u**2))),
    }


def main():
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = {}

    dist_scenarios = {
        "none": {},
        "external_force": {"external_force": True, "force_magnitude": 5.0},
        "load_change": {"load_change": True, "extra_mass": 2.0},
        "friction_change": {"friction_change": True, "extra_friction": 3.0},
    }

    # === Define all methods ===
    methods_nominal = {
        "PID": lambda dc, traj: run_classic(PIDController(kp=50, ki=5, kd=10, dt=0.005), disturbance_config=dc, trajectory=traj),
        "Pure SMC": lambda dc, traj: run_classic(NonsingularTerminalSMC(eta=5.0, phi=0.05, dt=0.005), disturbance_config=dc, trajectory=traj),
        "Fixed-ADRC (conservative)": lambda dc, traj: run_classic(ADRC(omega_o=10, omega_c=5, dt=0.005), disturbance_config=dc, trajectory=traj),
        "Fixed-ADRC (moderate)": lambda dc, traj: run_classic(ADRC(omega_o=20, omega_c=10, dt=0.005), disturbance_config=dc, trajectory=traj),
    }

    # DRL methods
    res_path = os.path.join(project, "results/sac_adrc_residual/sac_final")
    abs_path = os.path.join(project, "results/sac_adrc/sac_final")
    comp_path = os.path.join(project, "results/sac_composite_residual/sac_final")

    if os.path.exists(abs_path + ".zip"):
        methods_nominal["DRL-ADRC (absolute)"] = lambda dc, traj: run_drl(abs_path, "adrc", False, dc, traj)
    if os.path.exists(res_path + ".zip"):
        methods_nominal["DRL-ADRC (residual)"] = lambda dc, traj: run_drl(res_path, "adrc", True, dc, traj)
    if os.path.exists(comp_path + ".zip"):
        methods_nominal["DRL-ADRC-SMC"] = lambda dc, traj: run_drl(comp_path, "composite", True, dc, traj)

    # Fixed-ADRC-SMC (no DRL) for ablation
    methods_nominal["Fixed-ADRC-SMC"] = lambda dc, traj: run_classic(
        CompositeController(omega_o=20, omega_c=10, dt=0.005, smc_weight=0.3),
        disturbance_config=dc, trajectory=traj)

    # === Table 1: Nominal tracking (all methods, sinusoidal) ===
    print("\n=== TABLE 1: Nominal Tracking Performance ===")
    print(f"{'Method':<30} {'RMSE':>8} {'MAE':>8} {'MaxErr':>8} {'IAE':>8} {'Chatter':>8} {'Effort':>8}")
    print("-" * 90)
    for name, fn in methods_nominal.items():
        m = fn({}, "sinusoidal")
        results[f"t1_{name}"] = m
        print(f"{name:<30} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['MaxErr']:>8.4f} {m['IAE']:>8.4f} {m['Chattering']:>8.4f} {m['Effort']:>8.4f}")

    # === Table 2 & 3: Disturbance (4 key methods x 4 scenarios) ===
    key_methods = {k: v for k, v in methods_nominal.items()
                   if k in ["PID", "Fixed-ADRC (moderate)", "DRL-ADRC (residual)", "DRL-ADRC-SMC"]}

    print("\n=== TABLE 2 & 3: Disturbance Robustness ===")
    for scenario, dc in dist_scenarios.items():
        print(f"\n--- {scenario} ---")
        print(f"{'Method':<30} {'RMSE':>8} {'Chatter':>8} {'Effort':>8}")
        for name, fn in key_methods.items():
            m = fn(dc, "sinusoidal")
            results[f"t23_{scenario}_{name}"] = m
            print(f"{name:<30} {m['RMSE']:>8.4f} {m['Chattering']:>8.4f} {m['Effort']:>8.4f}")

    # === Table 4: Ablation (5 configs, nominal) ===
    print("\n=== TABLE 4: Ablation Study ===")
    ablation_methods = {}
    if "DRL-ADRC (absolute)" in methods_nominal:
        ablation_methods["DRL-ADRC (absolute)"] = methods_nominal["DRL-ADRC (absolute)"]
    if "DRL-ADRC (residual)" in methods_nominal:
        ablation_methods["DRL-ADRC (residual)"] = methods_nominal["DRL-ADRC (residual)"]
    # DRL-ADRC-SMC w/o ESO feedforward = composite with smc_weight but zero disturbance est
    # We approximate by using Pure SMC + DRL-ADRC average (or just use the existing composite)
    if "DRL-ADRC-SMC" in methods_nominal:
        ablation_methods["DRL-ADRC-SMC (full)"] = methods_nominal["DRL-ADRC-SMC"]
    if "Fixed-ADRC-SMC" in methods_nominal:
        ablation_methods["Fixed-ADRC-SMC (no DRL)"] = methods_nominal["Fixed-ADRC-SMC"]

    print(f"{'Config':<30} {'RMSE':>8} {'MAE':>8} {'Chatter':>8} {'Effort':>8}")
    print("-" * 70)
    for name, fn in ablation_methods.items():
        m = fn({}, "sinusoidal")
        results[f"t4_{name}"] = m
        print(f"{name:<30} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['Chattering']:>8.4f} {m['Effort']:>8.4f}")

    # === Table 5: Multi-frequency trajectory (new benchmark) ===
    print("\n=== TABLE 5: Multi-Frequency Trajectory Benchmark ===")
    print(f"{'Method':<30} {'RMSE':>8} {'MAE':>8} {'MaxErr':>8} {'Chatter':>8} {'Effort':>8}")
    print("-" * 80)
    for name, fn in key_methods.items():
        m = fn({}, "multifreq")
        results[f"t5_{name}"] = m
        print(f"{name:<30} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['MaxErr']:>8.4f} {m['Chattering']:>8.4f} {m['Effort']:>8.4f}")

    # Save all results
    with open(os.path.join(project, "results/full_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll results saved to results/full_results.json")


if __name__ == "__main__":
    main()
