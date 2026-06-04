"""PID Controller baseline."""
import numpy as np


class PIDController:
    def __init__(self, kp=50.0, ki=5.0, kd=10.0, dt=0.005,
                 u_min=-50.0, u_max=50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, y, y_ref, yd_ref=0.0, ydd_ref=0.0):
        error = y_ref - y
        self.integral += error * self.dt
        # Anti-windup
        self.integral = np.clip(self.integral, -5.0, 5.0)
        derivative = (error - self.prev_error) / self.dt
        u = self.kp * error + self.ki * self.integral + self.kd * derivative
        u = np.clip(u, self.u_min, self.u_max)
        self.prev_error = error
        return u

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def set_params(self, *args):
        pass
