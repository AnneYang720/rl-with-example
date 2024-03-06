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

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)  # ?
        q_values = self(obs_tensor.unsqueeze(0))  # ?
        max_q_index = torch.argmax(q_values, dim=1)[0]  # ?
        action = max_q_index.detach().item()  # ?
        return action
