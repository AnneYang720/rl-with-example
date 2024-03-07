import itertools
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from model.dqn import DQN

GAMMA = 0.99
MINI_BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQUENCY = 1000

env = gym.make("CartPole-v1")

replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = DQN(env)
target_net = DQN(env)

state_dict = torch.load("dqn_model_1000.pth")
online_net.load_state_dict(state_dict)
target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize Replay Buffer
observation, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    observation_, reward, terminated, truncated, info = env.step(action)
    memory = (observation, action, reward, terminated, observation_)
    replay_buffer.append(memory)
    observation = observation_

    if terminated:
        break


# Main Training Loop
observation, info = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    random_sample = random.random()
    if random_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(observation)

    observation_, reward, terminated, truncated, info = env.step(action)
    memory = (observation, action, reward, terminated, observation_)
    replay_buffer.append(memory)
    observation = observation_

    episode_reward += reward

    if terminated:
        observation, info = env.reset()
        reward_buffer.append(episode_reward)
        episode_reward = 0.0

    # After solved, watch it play
    if len(reward_buffer) >= 100:
        if np.mean(reward_buffer) >= 30000:
            while True:
                action = online_net.act(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                env.render()

                if terminated:
                    env.reset()

    # Start Gradient Step
    memories = random.sample(replay_buffer, 3)

    all_obs = np.asarray([memo_i[0] for memo_i in memories])
    all_a = np.asarray([memo_i[1] for memo_i in memories])
    all_r = np.asarray([memo_i[2] for memo_i in memories])
    all_done = np.asarray([memo_i[3] for memo_i in memories])
    all_obs_ = np.asarray([memo_i[4] for memo_i in memories])

    all_obs_tensor = torch.as_tensor(all_obs, dtype=torch.float32)
    all_a_tensor = torch.as_tensor(all_a, dtype=torch.int64).unsqueeze(-1)
    all_r_tensor = torch.as_tensor(all_r, dtype=torch.float32).unsqueeze(
        -1
    )  # [batch, 1]
    all_done_tensor = torch.as_tensor(all_done, dtype=torch.float32).unsqueeze(-1)
    all_obs__tensor = torch.as_tensor(all_obs_, dtype=torch.float32)  # [batch, 4]

    # Compute Targets
    target_q_values = target_net(all_obs__tensor)  # [batch, 2]
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # [batch, 1]
    targets = all_r_tensor + GAMMA * (1 - all_done_tensor) * max_target_q_values

    # Compute Loss
    q_values = online_net(all_obs_tensor)

    a_q_values = torch.gather(input=q_values, dim=1, index=all_a_tensor)

    loss = nn.functional.smooth_l1_loss(a_q_values, targets)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network
    if step % TARGET_UPDATE_FREQUENCY == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Print the training progress
    if step % 1000 == 0:
        print()
        print("Step: {}".format(step))
        print("Avg reward: {}".format(np.mean(reward_buffer)))
        torch.save(online_net.state_dict(), "dqn_model_{}.pth".format(step))
