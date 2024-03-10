import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, num_actions)
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
    def __init__(self, input_dim, num_actions):
        super(Dueling_DQN, self).__init__()

        self.netA = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, num_actions)
        )
        self.netV = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, x):
        advantage = self.netA(x)
        value = self.netV(x)
        q_values = value + advantage - advantage.mean()
        return q_values
    
    def act(self, obs) -> int:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        q_values = self(obs.unsqueeze(0))
        action = torch.argmax(q_values, dim=1).item()
        return action


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, layer_1_size=40, layer_2_size=35, layer_3_size=30):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, layer_1_size),
            nn.ReLU(),
            nn.Linear(layer_1_size, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, layer_3_size),
            nn.ReLU(),
            nn.Linear(layer_3_size, num_actions),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.fc(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        return self.fc(x)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_size=64):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Sigmoid()
        )
        self.action_bound = action_bound
        
    def forward(self, x):
        x = self.fc(x)
        action = torch.multiply(x, self.action_bound)
        return action

import torch
import torch.nn as nn
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        return self.fc(x)