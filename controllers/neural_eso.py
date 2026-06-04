"""
Neural Extended State Observer (Neural ESO)
Replaces the linear ESO with a GRU-based neural network.
Reference: NN-ESO (Energy, 2023), Neural Luenberger Observer (arXiv, 2025)
"""
import torch
import torch.nn as nn
import numpy as np


class NeuralESO(nn.Module):
    """
    GRU-based Neural ESO for 2nd-order system.
    Input: [y (measurement), u (control input)]
    Output: [z1 (position est), z2 (velocity est), z3 (disturbance est)]
    """

    def __init__(self, hidden_size=32, dt=0.005):
        super().__init__()
        self.dt = dt
        self.hidden_size = hidden_size

        # Input: [y, u] -> 2 dims
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.Tanh(),
            nn.Linear(16, 3),  # [z1, z2, z3]
        )

        self.hidden = None
        self.z = np.zeros(3)

        # Initialize weights small for stability
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, y_u, hidden=None):
        """
        y_u: (batch, seq_len, 2) or (1, 1, 2)
        """
        out, hidden = self.gru(y_u, hidden)
        z = self.fc(out)
        return z, hidden

    def update(self, y, u):
        """
        Single-step update (for online use during control).
        y: scalar measurement
        u: scalar control input
        Returns: z as numpy array [z1, z2, z3]
        """
        with torch.no_grad():
            inp = torch.FloatTensor([[[y, u]]])
            z_tensor, self.hidden = self.forward(inp, self.hidden)
            self.z = z_tensor.squeeze().numpy()
        return self.z.copy()

    def reset(self):
        self.hidden = None
        self.z = np.zeros(3)

    def get_disturbance(self):
        return self.z[2]


class NeuralADRC:
    """
    ADRC with Neural ESO.
    The Neural ESO is pretrained alongside the RL agent (end-to-end).
    For inference, we freeze the Neural ESO and use it as a drop-in
    replacement for the linear ESO.
    """

    def __init__(self, omega_c=8.0, dt=0.005, b0=1.0,
                 hidden_size=32, u_min=-50.0, u_max=50.0):
        self.omega_c = omega_c
        self.dt = dt
        self.b0 = b0
        self.u_min = u_min
        self.u_max = u_max

        self.kp = omega_c ** 2
        self.kd = 2.0 * omega_c

        self.neural_eso = NeuralESO(hidden_size=hidden_size, dt=dt)
        self.last_u = 0.0

    def compute(self, y, y_ref, yd_ref=0.0, ydd_ref=0.0):
        z = self.neural_eso.update(y, self.last_u)
        z1, z2, z3 = z[0], z[1], z[2]

        u0 = ydd_ref + self.kp * (y_ref - z1) + self.kd * (yd_ref - z2)
        u = (u0 - z3) / self.b0
        u = np.clip(u, self.u_min, self.u_max)

        self.last_u = u
        return u

    def reset(self):
        self.neural_eso.reset()
        self.last_u = 0.0

    def set_params(self, omega_o_unused, omega_c):
        """omega_o is not used in Neural ESO (learned internally)."""
        self.omega_c = omega_c
        self.kp = omega_c ** 2
        self.kd = 2.0 * omega_c
