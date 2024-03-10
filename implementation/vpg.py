import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import gymnasium as gym
from .model import PolicyNetwork, ValueNetwork

class VPGAgent(object):
    def __init__(self, env_name:str, gamma:float):
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.in_features = int(np.prod(self.env.observation_space.shape))
        self.num_actions = self.env.action_space.n
        self.PolicyNet = PolicyNetwork(self.in_features, self.num_actions)
        self.ValueNet = ValueNetwork(self.in_features)
        self.policy_optimizer = optim.Adam(self.PolicyNet.parameters(), lr=0.0001)
        self.value_optimizer = optim.Adam(self.ValueNet.parameters(), lr=0.0001)

    def train(self, iterations:int, num_traj:int):
        iter_rewards = []
        for i in range(iterations):
            batch_records = []
            iter_reward = 0
            for _ in range(num_traj):
                ep_logits, ep_values, ep_rewards, ep_actions = [], [], [], []
                state, _ = self.env.reset()
                terminated = False
                while not terminated:
                    state = torch.tensor(state, dtype=torch.float32)
                    logits = self.PolicyNet(state)
                    value = self.ValueNet(state)
                    action = torch.multinomial(logits[0], 1, replacement=False)
                    state, reward, terminated, _, _ = self.env.step(action.item())

                    ep_logits.append(logits)
                    ep_values.append(value)
                    ep_rewards.append(reward)
                    ep_actions.append(action)
                    iter_reward += reward
                
                batch_records.append((ep_logits, ep_values, ep_rewards, ep_actions))

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
    
    def update_network(self, batch_records):
        total_policy_loss = torch.tensor([0]).float().cuda()
        total_value_loss = torch.tensor([0]).float().cuda()

        for ep_logits,ep_values,ep_rewards,ep_actions in batch_records:
            ep_logits = torch.stack(ep_logits).squeeze()
            ep_actions = torch.stack(ep_actions)
            targets = self.discount_rewards(ep_rewards)
            advantage = torch.tensor(targets) - torch.cat(ep_values)
            log_prob = F.log_softmax(ep_logits, dim=1)
            
            policy_loss = (log_prob.gather(1, ep_actions).squeeze().dot(advantage.detach()))
            value_loss = F.mse_loss(torch.cat(ep_values), torch.tensor(targets))
            total_policy_loss += policy_loss
            total_value_loss += value_loss
        
        total_policy_loss /= len(batch_records)
        total_value_loss /= len(batch_records)

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()