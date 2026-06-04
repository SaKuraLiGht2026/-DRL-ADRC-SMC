"""
Composite Controller: ADRC + Sliding Mode Control
ADRC provides disturbance estimation, SMC provides robust tracking.
"""
import numpy as np
from .adrc import ADRC
from .sliding_mode import NonsingularTerminalSMC


class CompositeController:
    """
    DRL + ADRC + SMC composite controller.
    - ADRC: estimates total disturbance via ESO
    - SMC: uses disturbance estimate as feedforward, provides robust control
    - DRL: adjusts ADRC bandwidth parameters online
    """

    def __init__(self, omega_o=10.0, omega_c=5.0, dt=0.001, b0=1.0,
                 smc_weight=0.3, u_min=-50.0, u_max=50.0):
        self.adrc = ADRC(omega_o, omega_c, dt, b0, u_min, u_max)
        self.smc = NonsingularTerminalSMC(dt=dt, u_min=u_min, u_max=u_max)
        self.smc_weight = smc_weight
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max

    def compute(self, y, y_ref, yd_ref=0.0, ydd_ref=0.0, velocity=0.0):
        """
        Compute composite control signal.
        """
        # ADRC computes its control and updates ESO
        u_adrc = self.adrc.compute(y, y_ref, yd_ref, ydd_ref)
        # Get disturbance estimate from ESO
        disturbance_est = self.adrc.eso.z[2]
        # Tracking error
        e = y_ref - y
        ed = yd_ref - velocity
        # SMC with disturbance feedforward from ESO
        u_smc, s = self.smc.compute(e, ed, ydd_ref, disturbance_est)
        # Weighted combination
        u = (1.0 - self.smc_weight) * u_adrc + self.smc_weight * u_smc
        u = np.clip(u, self.u_min, self.u_max)
        self.adrc.last_u = u  # Update for ESO
        return u, s

    def set_params(self, omega_o, omega_c):
        self.adrc.set_params(omega_o, omega_c)

    def reset(self):
        self.adrc.reset()
        self.smc.reset()
