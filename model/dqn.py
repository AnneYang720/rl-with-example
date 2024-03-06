import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env


class Network(nn.Module):
    def __init__(self, env: Env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.Tanh(), nn.Linear(64, env.action_space.n)
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)

    def act(self, obs) -> int:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        q_values = self(obs.unsqueeze(0))
        action = torch.argmax(q_values, dim=1).item()
        return action
