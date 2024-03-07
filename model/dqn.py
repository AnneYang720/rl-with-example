import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64), nn.Tanh(), nn.Linear(64, num_actions)
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)

    def act(self, obs) -> int:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        q_values = self(obs.unsqueeze(0))
        action = torch.argmax(q_values, dim=1).item()
        return action

class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()

        self.netA = nn.Sequential(
            nn.Linear(in_channels, 64), nn.Tanh(), nn.Linear(64, num_actions)
        )
        self.netV = nn.Sequential(
            nn.Linear(in_channels, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, x):
        advantage = self.netA(x)
        value = self.netV(x)
        q_values = value + advantage - advantage.mean()
        return q_values