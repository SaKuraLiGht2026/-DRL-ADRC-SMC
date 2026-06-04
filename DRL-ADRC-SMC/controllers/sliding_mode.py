"""
Nonsingular Terminal Sliding Mode Controller (NTSMC)
"""
import numpy as np


class NonsingularTerminalSMC:
    """
    Nonsingular terminal sliding mode controller for 2nd-order system.
    Provides finite-time convergence and robustness.
    """

    def __init__(self, alpha=1.5, beta_smc=1.0, p=9, q=7,
                 eta=5.0, phi=0.05, dt=0.001,
                 u_min=-50.0, u_max=50.0):
        """
        alpha: sliding surface parameter
        beta_smc: sliding surface parameter
        p, q: power ratio (p/q > 1, both odd)
        eta: reaching law gain (switching gain)
        phi: boundary layer thickness (to reduce chattering)
        """
        self.alpha = alpha
        self.beta_smc = beta_smc
        self.p = p
        self.q = q
        self.pq_ratio = p / q
        self.qp_ratio = q / p
        self.eta = eta
        self.phi = phi
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max

    def _sat(self, s):
        """Saturation function to replace sign function (reduces chattering)."""
        if abs(s) > self.phi:
            return np.sign(s)
        else:
            return s / self.phi

    def compute(self, e, ed, edd_ref=0.0, disturbance_est=0.0):
        """
        Compute SMC control signal.
        e: tracking error (y_ref - y)
        ed: error derivative
        edd_ref: reference acceleration
        disturbance_est: estimated disturbance from ESO (optional)
        """
        # Nonsingular terminal sliding surface
        # s = e + beta * |ed|^(p/q) * sign(ed)
        s = e + self.beta_smc * (np.abs(ed) ** self.pq_ratio) * np.sign(ed)

        # Control law
        # Equivalent control + switching control
        if abs(ed) > 1e-6:
            u_eq = edd_ref + (self.q / (self.p * self.beta_smc)) * \
                   (np.abs(ed) ** (2.0 - self.pq_ratio))
        else:
            u_eq = edd_ref

        u_sw = self.eta * self._sat(s) + self.alpha * s
        u = u_eq + u_sw - disturbance_est
        u = np.clip(u, self.u_min, self.u_max)
        return u, s

    def reset(self):
        pass
