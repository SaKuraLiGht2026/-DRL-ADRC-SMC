"""
Custom Gymnasium environment: Single-joint robot arm trajectory tracking.
v3: Added Residual RL + Neural ESO support.

Key improvements:
1. Residual RL: action = default_params + delta (not absolute values)
2. Neural ESO: GRU-based observer replaces linear ESO (optional)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

from controllers.adrc import ADRC
from controllers.neural_eso import NeuralADRC
from controllers.composite import CompositeController


class RobotArmTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, control_mode="adrc",
                 use_neural_eso=False, use_residual=True,
                 sim_dt=1.0/240.0, ctrl_dt=0.005, episode_length=5.0,
                 disturbance_config=None):
        super().__init__()

        self.render_mode = render_mode
        self.control_mode = control_mode
        self.use_neural_eso = use_neural_eso
        self.use_residual = use_residual
        self.sim_dt = sim_dt
        self.ctrl_dt = ctrl_dt
        self.ctrl_steps = int(ctrl_dt / sim_dt)
        self.episode_length = episode_length
        self.max_steps = int(episode_length / ctrl_dt)
        self.disturbance_config = disturbance_config or {}

        self.controlled_joint = 0

        # Default (base) ADRC parameters for residual RL
        self.default_omega_o = 15.0
        self.default_omega_c = 8.0

        # Residual range: how much RL can adjust from default
        self.delta_omega_o_max = 10.0  # default +/- 10
        self.delta_omega_c_max = 5.0   # default +/- 5

        # Absolute range (for non-residual mode)
        self.omega_o_range = [8.0, 30.0]
        self.omega_c_range = [3.0, 15.0]

        # Action space: normalized [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation: [error, error_dot, joint_pos, joint_vel,
        #               ref_pos, ref_vel, ref_acc, dist_est]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        self._init_controller()

        self.physics_client = None
        self.robot_id = None
        self.step_count = 0
        self.prev_error = 0.0
        self.prev_torque = 0.0
        self.cumulative_error = 0.0
        self.torque_history = []

        # Normalization constants
        self.pos_scale = 1.0
        self.vel_scale = 5.0
        self.acc_scale = 25.0
        self.torque_scale = 50.0
        self.dist_scale = 10.0

    def _init_controller(self):
        if self.use_neural_eso:
            self.controller = NeuralADRC(
                omega_c=self.default_omega_c, dt=self.ctrl_dt, b0=1.0
            )
        elif self.control_mode == "composite":
            self.controller = CompositeController(
                omega_o=self.default_omega_o,
                omega_c=self.default_omega_c,
                dt=self.ctrl_dt, b0=1.0, smc_weight=0.3
            )
        else:
            self.controller = ADRC(
                omega_o=self.default_omega_o,
                omega_c=self.default_omega_c,
                dt=self.ctrl_dt, b0=1.0
            )

    def _generate_reference(self, t):
        freq = 0.5
        amp = 0.8
        pos = amp * np.sin(2 * np.pi * freq * t)
        vel = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
        acc = -amp * (2 * np.pi * freq) ** 2 * np.sin(2 * np.pi * freq * t)
        return pos, vel, acc

    def _apply_disturbance(self, t):
        dist = 0.0
        cfg = self.disturbance_config

        if cfg.get("external_force", False):
            if 2.0 < t < 3.5:
                dist += cfg.get("force_magnitude", 5.0)

        if cfg.get("load_change", False):
            if t > 2.5:
                joint_state = p.getJointState(self.robot_id, self.controlled_joint)
                vel = joint_state[1]
                extra_inertia = cfg.get("extra_mass", 2.0)
                dist += -extra_inertia * vel * 0.5

        if cfg.get("friction_change", False):
            if t > 1.5:
                joint_state = p.getJointState(self.robot_id, self.controlled_joint)
                vel = joint_state[1]
                extra_friction = cfg.get("extra_friction", 3.0)
                dist += -extra_friction * np.sign(vel)

        return dist

    def _decode_action(self, action):
        """
        Decode normalized action to ADRC parameters.
        Residual mode: params = default + delta * action
        Absolute mode: params = range_min + (action+1)/2 * range
        """
        if self.use_residual:
            omega_o = self.default_omega_o + action[0] * self.delta_omega_o_max
            omega_c = self.default_omega_c + action[1] * self.delta_omega_c_max
            # Clamp to safe range
            omega_o = np.clip(omega_o, 5.0, 35.0)
            omega_c = np.clip(omega_c, 3.0, 18.0)
        else:
            omega_o = self.omega_o_range[0] + (action[0] + 1) / 2 * \
                      (self.omega_o_range[1] - self.omega_o_range[0])
            omega_c = self.omega_c_range[0] + (action[1] + 1) / 2 * \
                      (self.omega_c_range[1] - self.omega_c_range[0])
        return omega_o, omega_c

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.sim_dt)

        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )

        init_pos = 0.0
        p.resetJointState(self.robot_id, self.controlled_joint, init_pos, 0.0)

        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            p.setJointMotorControl2(
                self.robot_id, i, p.VELOCITY_CONTROL, force=0
            )

        self.controller.reset()
        self.step_count = 0
        self.prev_error = 0.0
        self.prev_torque = 0.0
        self.cumulative_error = 0.0
        self.torque_history = []

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        t = self.step_count * self.ctrl_dt
        ref_pos, ref_vel, ref_acc = self._generate_reference(t)

        joint_state = p.getJointState(self.robot_id, self.controlled_joint)
        joint_pos = joint_state[0]
        joint_vel = joint_state[1]

        error = ref_pos - joint_pos
        error_dot = ref_vel - joint_vel

        # Get disturbance estimate from whichever ESO type
        if hasattr(self.controller, 'neural_eso'):
            dist_est = self.controller.neural_eso.z[2]
        elif hasattr(self.controller, 'eso'):
            dist_est = self.controller.eso.z[2]
        elif hasattr(self.controller, 'adrc'):
            dist_est = self.controller.adrc.eso.z[2]
        else:
            dist_est = 0.0

        obs = np.array([
            error / self.pos_scale,
            error_dot / self.vel_scale,
            joint_pos / self.pos_scale,
            joint_vel / self.vel_scale,
            ref_pos / self.pos_scale,
            ref_vel / self.vel_scale,
            ref_acc / self.acc_scale,
            dist_est / self.dist_scale,
        ], dtype=np.float32)
        return obs

    def step(self, action):
        omega_o, omega_c = self._decode_action(action)
        self.controller.set_params(omega_o, omega_c)

        t = self.step_count * self.ctrl_dt
        ref_pos, ref_vel, ref_acc = self._generate_reference(t)

        joint_state = p.getJointState(self.robot_id, self.controlled_joint)
        joint_pos = joint_state[0]
        joint_vel = joint_state[1]

        if isinstance(self.controller, CompositeController):
            torque, _ = self.controller.compute(
                joint_pos, ref_pos, ref_vel, ref_acc, joint_vel
            )
        else:
            torque = self.controller.compute(
                joint_pos, ref_pos, ref_vel, ref_acc
            )

        ext_dist = self._apply_disturbance(t)

        for _ in range(self.ctrl_steps):
            p.setJointMotorControl2(
                self.robot_id, self.controlled_joint,
                p.TORQUE_CONTROL, force=torque + ext_dist
            )
            p.stepSimulation()

        self.step_count += 1

        error = ref_pos - joint_pos
        self.cumulative_error += error ** 2

        # === Normalized reward ===
        r_tracking = -np.clip(error ** 2, 0, 4.0) / 0.5
        norm_torque = torque / self.torque_scale
        r_energy = -0.05 * norm_torque ** 2
        torque_diff = (torque - self.prev_torque) / self.torque_scale
        r_chatter = -0.1 * np.clip(torque_diff ** 2, 0, 4.0)

        if abs(error) < 0.05:
            r_bonus = 0.5
        elif abs(error) < 0.1:
            r_bonus = 0.2
        else:
            r_bonus = 0.0

        reward = r_tracking + r_energy + r_chatter + r_bonus

        self.prev_error = error
        self.prev_torque = torque
        self.torque_history.append(torque)

        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "tracking_error": abs(error),
            "torque": torque,
            "omega_o": omega_o,
            "omega_c": omega_c,
            "rmse": np.sqrt(self.cumulative_error / self.step_count),
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
