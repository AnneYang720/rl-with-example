import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import torch.nn as nn
import torch
import torch.optim as optimizer
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MLP(nn.Module):
    """
    Returns a fully connected neural network of 
    given dimensions. The input is of dimensions
    of the shape of the observations and output
    is of the shape of number of actions.
    """

    def __init__(self, in_channels, num_actions):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, 32), 
            nn.Linear(32, 20), 
            nn.Linear(20, num_actions)
        )

    
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

def discount_rewards(rewards, gamma=0.99):
    # Cumulative discounted sum
    r = np.array([gamma**i * rewards[i] 
                  for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    # Subtracting the baseline reward 
    # Intuitively this means if the network predicts what it
    # expects it should not do too much about it
    # Stabalizes and speeds up the training 
    return r - r.mean()

def train(env_name='CartPole-v1', num_episodes=2000, batch_size=10, lr=1e-2, gamma=0.99):

    env = gym.make(env_name)
    model = MLP(env.observation_space.shape[0], env.action_space.n)

    # Lists for maintaing logs
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []

    batch_counter = 1
    opt = optimizer.Adam(model.parameters(), lr)
    action_space = np.arange(env.action_space.n)

    for ep in range(num_episodes):
        # Reset
        observation, info = env.reset()
        states = []
        rewards = []
        actions = []
        terminated = False
        while terminated == False:
            action_probs = model(observation).detach()
            action = np.random.choice(action_space, p=action_probs)
            next_observation, reward, terminated, _, _ = env.step(action)
            states.append(observation)
            rewards.append(reward)
            actions.append(action)
            observation = next_observation
            if terminated:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                if batch_counter == batch_size:
                    # Prepare the batches for training
                    # Add states, reward and actions to tensor
                    opt.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    action_tensor = torch.LongTensor(batch_actions)

                    # Convert the probs by the model to log probabilities
                    log_probs = torch.log(model(state_tensor)) # [batch_size, num_actions]
                    # Mask the probs of the selected actions
                    selected_log_probs = reward_tensor * log_probs[np.arange(len(action_tensor)), action_tensor]
                    # Loss is negative of expected policy function J = R * log_prob
                    loss = -selected_log_probs.mean()

                    # Do the update gradient descent(with negative reward hence is gradient ascent) 
                    loss.backward()
                    opt.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                print("\rEp: {} Average of last 10: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-10:])), end="")
    
    return total_rewards

if __name__ == '__main__':

    rewards = train()
    window = 10
    smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window 
                        else np.mean(rewards[:i+1]) for i in range(len(rewards))]

    plt.figure(figsize=(12,8))
    plt.plot(rewards)
    plt.plot(smoothed_rewards)
    plt.ylabel('Total Rewards')
    plt.xlabel('Episodes')
    plt.show()