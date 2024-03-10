import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import gymnasium as gym
from model import PolicyNetwork, ValueNetwork

class PPOAgent(object):
    def __init__(self, env_name:str, gamma:float, lam:float, epsilon:float):
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        
        self.in_features = int(np.prod(self.env.observation_space.shape))
        self.num_actions = self.env.action_space.n
        self.PolicyNet = PolicyNetwork(self.in_features, self.num_actions)
        self.ValueNet = ValueNetwork(self.in_features)
        self.policy_optimizer = optim.Adam(self.PolicyNet.parameters(), lr=0.01)
        self.value_optimizer = optim.Adam(self.ValueNet.parameters(), lr=0.001)

    def train(self, iterations:int, num_traj:int):
        iter_rewards = []
        for i in range(iterations):
            batch_records = []
            iter_reward = 0
            for _ in range(num_traj):
                ep_logits, ep_values, ep_rewards, ep_actions = [], [], [], []
                ep_states = []
                state, _ = self.env.reset()
                terminated = False
                while not terminated:
                    state = torch.tensor(state, dtype=torch.float32)
                    logits = self.PolicyNet(state)

                    value = self.ValueNet(state)
                    action = torch.multinomial(logits, 1, replacement=False)
                    ep_states.append(state)
                    
                    state, reward, terminated, _, _ = self.env.step(action.item())
                    ep_logits.append(logits)
                    ep_values.append(value)
                    ep_rewards.append(reward)
                    ep_actions.append(action)
                    iter_reward += reward
                
                batch_records.append((ep_logits, ep_values, ep_rewards, ep_actions, ep_states))

            if i % 10 == 0:
                print("ITERATION:", i+1, "AVG REWARD:", iter_reward/num_traj)
            iter_rewards.append(iter_reward/num_traj)
            self.update_network(batch_records)
        return iter_rewards
            
    def discount_rewards(self, rewards:list[float]):
        discount_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            discount_rewards.insert(0, R)
        return discount_rewards
    
    def calculate_advantages(self, rewards, values):
        advantages = np.zeros_like(rewards)
        for t in range(len(rewards)):
            ad = 0
            for l in range(0, len(rewards) - t - 1):
                delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
                ad += ((self.gamma*self.lam)**l)*(delta)
            ad += ((self.gamma*self.lam)**l)*(rewards[t+l] - values[t+l])
            advantages[t] = ad
        return torch.tensor((advantages - np.mean(advantages))/np.std(advantages), dtype=torch.float32)
    
    def update_network(self, batch_records):
        total_policy_loss = torch.tensor([0]).float().cuda()
        total_value_loss = torch.tensor([0]).float().cuda()

        for ep_logits,ep_values,ep_rewards,ep_actions,ep_states in batch_records:
            ep_logits = torch.stack(ep_logits)
            ep_actions = torch.stack(ep_actions)
            ep_states = torch.stack(ep_states)
            ep_advantages = self.calculate_advantages(ep_rewards, ep_values)

            targets = self.discount_rewards(ep_rewards)
            value_loss = F.mse_loss(torch.cat(ep_values), torch.tensor(targets))
            total_value_loss += value_loss

            old_probabilities = ep_logits.detach()
            new_probabilities = self.PolicyNet(ep_states)
            new_responsible_outputs = new_probabilities.gather(1, ep_actions).squeeze()
            old_responsible_outputs = old_probabilities.gather(1, ep_actions).squeeze()            
            
            ratio = new_responsible_outputs / old_responsible_outputs
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -torch.min(ratio * ep_advantages, clipped_ratio * ep_advantages).mean()
            total_policy_loss += policy_loss
            
        total_policy_loss /= len(batch_records)
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        total_value_loss /= len(batch_records)
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()