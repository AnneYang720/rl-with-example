import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import gymnasium as gym
from model import ActorNetwork, CriticNetwork
from util import *

# TODO
# Current result is not good, will refer to https://github.com/ccjy88/cartpole_ddpg/tree/master for improve

class DDPGAgent(object):
    def __init__(self, env, state_dim, action_dim, action_bound, gamma=0.9, tau=0.02):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.memory = ReplayMemory(memory_size=5000, batch_size=1024)
        self.var = 0 # the variance of the noise
        self.gamma = gamma
        self.tau = tau
        # actor net
        self.ActorEvalNet = ActorNetwork(self.state_dim, self.action_dim, self.action_bound)
        self.ActorTargetNet = ActorNetwork(self.state_dim, self.action_dim, self.action_bound)
        self.actor_eval_optimizer = optim.Adam(self.ActorEvalNet.parameters(), lr=0.001)
        # eval net
        self.CriticEvalNet = CriticNetwork(self.state_dim, self.action_dim)
        self.CriticTargetNet = CriticNetwork(self.state_dim, self.action_dim)
        self.critic_eval_optimizer = optim.Adam(self.CriticEvalNet.parameters(), lr=0.001)
    
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.ActorEvalNet(state) + np.random.normal(0, 0.5) * self.var
        if action >= 0.5:
            action = 1
        else:
            action = 0
        return action
    
    def train(self, episodes:int, max_ep_steps:int=100):
        ep_rewards = []
        for i in range(episodes):
            state, _ = self.env.reset()
            ep_reward = 0
            terminated = False
            step = 0
            learned = False
            while not terminated:
                step += 1
                action = self.get_action(state)
                next_state, reward, terminated, _, _ = self.env.step(action)
                self.memory.add((state, action, reward, terminated, next_state))
                ep_reward += reward

                if not self.memory.is_full():
                    continue
                self.var *= 0.9995
                self.learn()
                learned = True
                state = next_state
                
                # if terminated:
                #     if step <= max_ep_steps and len(self.memory) >= self.memory.batch_size:
                #         self.var *= 0.997
                #         self.learn()
                #         learned = True
                        
                # elif step == max_ep_steps:
                #     self.var *= 0.9995
                #     self.learn()
                #     terminated = True
                #     learned = True
                
            ep_rewards.append(ep_reward)
            if i % 10 == 0:
                print("Episode:", i, "Reward:", ep_reward, "Learned:", learned, "Var:", self.var)
    
    def learn(self):
        state_batch, action_batch, reward_batch, _, next_state_batch = self.memory.sample()
        # Critic Learn
        self.critic_learn(state_batch, action_batch, reward_batch, next_state_batch)
        # Actor Learn
        self.actor_learn(state_batch)
        self.soft_replace()

    def actor_learn(self, state_batch):
        actions = self.ActorEvalNet(state_batch)
        ce_s = torch.cat([state_batch, actions], 1)
        q = self.CriticEvalNet(ce_s)
        a_loss = torch.mean(-q)
        self.actor_eval_optimizer.zero_grad()
        a_loss.backward()
        self.actor_eval_optimizer.step()
    
    def critic_learn(self, state_batch, action_batch, reward_batch, next_state_batch):
        q_pre = self.CriticEvalNet(torch.cat([state_batch, action_batch], 1))
        a_ = self.ActorTargetNet(next_state_batch)
        q_ = self.CriticTargetNet(torch.cat([next_state_batch, a_], 1))
        q_target = reward_batch + self.gamma * q_
        critic_loss = F.mse_loss(q_target, q_pre)

        self.critic_eval_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval_optimizer.step()
    
    def soft_replace(self):
        for eval_param, target_param in zip(self.ActorEvalNet.parameters(), self.ActorTargetNet.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + eval_param.data * self.tau)
        for eval_param, target_param in zip(self.CriticEvalNet.parameters(), self.CriticTargetNet.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + eval_param.data * self.tau)


def main():   
    env = gym.make('CartPole-v1')
    agent = DDPGAgent(env, state_dim=4, action_dim=1, action_bound=1)
    rewards = agent.train(episodes=15000)

main()