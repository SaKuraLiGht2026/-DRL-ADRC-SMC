"""
Active Disturbance Rejection Controller (ADRC)
with Extended State Observer (ESO)
"""
import numpy as np


class ESO:
    """Extended State Observer for 2nd-order system."""

    def __init__(self, omega_o, dt, b0=1.0):
        self.omega_o = omega_o
        self.dt = dt
        self.b0 = b0
        # ESO gains (bandwidth parameterization)
        self.beta1 = 3.0 * omega_o
        self.beta2 = 3.0 * omega_o ** 2
        self.beta3 = omega_o ** 3
        # States: [z1 (position), z2 (velocity), z3 (total disturbance)]
        self.z = np.zeros(3)

    def update(self, y, u):
        """Update observer with measurement y and control input u."""
        e = y - self.z[0]
        z1_dot = self.z[1] + self.beta1 * e
        z2_dot = self.z[2] + self.beta2 * e + self.b0 * u
        z3_dot = self.beta3 * e
        self.z[0] += z1_dot * self.dt
        self.z[1] += z2_dot * self.dt
        self.z[2] += z3_dot * self.dt
        return self.z.copy()

    def reset(self):
        self.z = np.zeros(3)

    def set_omega(self, omega_o):
        self.omega_o = omega_o
        self.beta1 = 3.0 * omega_o
        self.beta2 = 3.0 * omega_o ** 2
        self.beta3 = omega_o ** 3


class ADRC:
    """
    ADRC for 2nd-order system.
    Tunable parameters: omega_o (observer bandwidth), omega_c (controller bandwidth)
    """

    def __init__(self, omega_o=10.0, omega_c=5.0, dt=0.001, b0=1.0,
                 u_min=-50.0, u_max=50.0):
        self.omega_o = omega_o
        self.omega_c = omega_c
        self.dt = dt
        self.b0 = b0
        self.u_min = u_min
        self.u_max = u_max
        # Controller gains (bandwidth parameterization)
        self.kp = omega_c ** 2
        self.kd = 2.0 * omega_c
        self.eso = ESO(omega_o, dt, b0)
        self.last_u = 0.0

    def compute(self, y, y_ref, yd_ref=0.0, ydd_ref=0.0):
        """
        Compute control signal.
        y: current output (position)
        y_ref: reference position
        yd_ref: reference velocity
        ydd_ref: reference acceleration
        """
        z = self.eso.update(y, self.last_u)
        z1, z2, z3 = z[0], z[1], z[2]
        # PD control law with disturbance compensation
        u0 = ydd_ref + self.kp * (y_ref - z1) + self.kd * (yd_ref - z2)
        # Compensate total disturbance
        u = (u0 - z3) / self.b0
        u = np.clip(u, self.u_min, self.u_max)
        self.last_u = u
        return u

    def reset(self):
        self.eso.reset()
        self.last_u = 0.0

    def set_params(self, omega_o, omega_c):
        """Set ADRC parameters (called by DRL agent)."""
        self.omega_o = omega_o
        self.omega_c = omega_c
        self.kp = omega_c ** 2
        self.kd = 2.0 * omega_c
        self.eso.set_omega(omega_o)
