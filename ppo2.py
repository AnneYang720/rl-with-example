import torch
import torch.nn as nn
import torch.optim as optim

# https://github.com/bnelo12/PPO-Implemnetation/tree/master

class ValueNetwork(nn.Module):
    def __init__(self, num_features, hidden_size, learning_rate=0.01):
        super(ValueNetwork, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        self.model = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, observations):
        return self.model(observations).view(-1)

    def get(self, states):
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32)
            value = self.forward(states)
        return value.numpy()

    def update(self, states, discounted_rewards):
        states = torch.tensor(states, dtype=torch.float32)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        values = self.forward(states)
        loss = self.criterion(values, discounted_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    


class PPOPolicyNetwork(nn.Module):
    def __init__(self, num_features, layer_1_size, layer_2_size, layer_3_size, num_actions, epsilon=.2,
                 learning_rate=9e-4):
        super(PPOPolicyNetwork, self).__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        self.epsilon = epsilon

        self.model = nn.Sequential(
            nn.Linear(num_features, layer_1_size),
            nn.ReLU(),
            nn.Linear(layer_1_size, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, layer_3_size),
            nn.ReLU(),
            nn.Linear(layer_3_size, num_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, observations):
        return self.model(observations)

    def get_dist(self, states):
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32)
            dist = self.forward(states)
        return dist.numpy()

    def update(self, states, chosen_actions, ep_advantages):
        states = torch.tensor(states, dtype=torch.float32)
        chosen_actions = torch.tensor(chosen_actions, dtype=torch.float32)
        ep_advantages = torch.tensor(ep_advantages, dtype=torch.float32)

        old_probabilities = self.forward(states).detach()
        new_probabilities = self.forward(states)
        new_responsible_outputs = torch.sum(chosen_actions * new_probabilities, dim=1)
        old_responsible_outputs = torch.sum(chosen_actions * old_probabilities, dim=1)

        ratio = new_responsible_outputs / old_responsible_outputs
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(ratio * ep_advantages, clipped_ratio * ep_advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

import torch
import numpy as np

class PPO():
    def __init__(self, env, num_features=1, num_actions=1, gamma=.98, lam=1, epsilon=.2,
                 value_network_lr=0.1, policy_network_lr=9e-4, value_network_hidden_size=100,
                 policy_network_hidden_size_1=10, policy_network_hidden_size_2=10, policy_network_hidden_size_3=10):
        
        self.env = env
        self.num_features = num_features
        self.num_actions = num_actions
        self.gamma = gamma
        self.lam = lam
        self.Pi = PPOPolicyNetwork(num_features=num_features, num_actions=num_actions, 
                                   layer_1_size=policy_network_hidden_size_1,
                                   layer_2_size=policy_network_hidden_size_2,
                                   layer_3_size=policy_network_hidden_size_3,
                                   epsilon=epsilon,
                                   learning_rate=policy_network_lr)
        self.V = ValueNetwork(num_features, value_network_hidden_size, learning_rate=value_network_lr)

    def discount_rewards(self, rewards):
        running_total = 0
        discounted = np.zeros_like(rewards)
        for r in reversed(range(len(rewards))):
            running_total = running_total * self.gamma + rewards[r]
            discounted[r] = running_total
        return discounted

    def calculate_advantages(self, rewards, values):
        advantages = np.zeros_like(rewards)
        for t in range(len(rewards)):
            ad = 0
            for l in range(0, len(rewards) - t - 1):
                delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
                ad += ((self.gamma*self.lam)**l)*(delta)
            ad += ((self.gamma*self.lam)**l)*(rewards[t+l] - values[t+l])
            advantages[t] = ad
        return (advantages - np.mean(advantages))/np.std(advantages)

    def run_model(self):
        episode = 1
        running_reward = []
        step = 0
        render = False
        while(True):
            s0 = self.env.reset()
            is_terminal = False
            ep_rewards = []
            ep_actions = []
            ep_states = []
            score = 0
            while not is_terminal:
                if render:
                    self.env.render()
                action = np.random.choice(range(self.num_actions), p=self.Pi.get_dist(np.array(s0)[np.newaxis, :])[0])
                a_binarized = np.zeros(self.num_actions)
                a_binarized[action] = 1
                s1, r, is_terminal, _ = self.env.step(action)
                score += r
                ep_actions.append(a_binarized)
                ep_rewards.append(r)
                ep_states.append(s0)
                s0 = s1
                if is_terminal:
                    ep_actions = np.vstack(ep_actions)
                    ep_rewards = np.array(ep_rewards, dtype=np.float_)
                    ep_states = np.vstack(ep_states)
                    targets = self.discount_rewards(ep_rewards)
                    for i in range(len(ep_states)):
                        self.V.update([ep_states[i]], [targets[i]])
                    ep_advantages = self.calculate_advantages(ep_rewards, self.V.get(ep_states))
                    vs = self.V.get(ep_states)
                    self.Pi.update(ep_states, ep_actions, ep_advantages)
                    ep_rewards = []
                    ep_actions = []
                    ep_states = []
                    running_reward.append(score)
                    if episode % 25 == 0:
                        avg_score = np.mean(running_reward[-25:])
                        print("Episode: " + str(episode) + " Score: " + str(avg_score))
                        if avg_score >= 500:
                            print("Solved!")
                            render = True
                    episode += 1