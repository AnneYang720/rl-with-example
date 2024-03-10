import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque
import torch

MEMORY_SIZE = 10000
MINI_BATCH_SIZE = 32
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQUENCY = 1000

class ReplayMemory:
    def __init__(self, memory_size=MEMORY_SIZE, batch_size=MINI_BATCH_SIZE):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replay_memory = deque(maxlen=memory_size)

    def add(self, memory):
        self.replay_memory.append(memory)

    def sample(self):
        memories = random.sample(self.replay_memory, self.batch_size)
        state_batch = torch.as_tensor(np.asarray([memo_i[0] for memo_i in memories]), dtype=torch.float32)  # [batch, 4]
        action_batch = torch.as_tensor(np.asarray([memo_i[1] for memo_i in memories]), dtype=torch.int64).unsqueeze(-1)  # [batch, 1]
        reward_batch = torch.as_tensor(np.asarray([memo_i[2] for memo_i in memories]), dtype=torch.float32).unsqueeze(-1)  # [batch, 1]
        terminal_batch = torch.as_tensor(np.asarray([memo_i[3] for memo_i in memories]), dtype=torch.float32).unsqueeze(-1)  # [batch, 1]
        next_state_batch = torch.as_tensor(np.asarray([memo_i[4] for memo_i in memories]), dtype=torch.float32)  # [batch, 4]
        return state_batch, action_batch, reward_batch, terminal_batch, next_state_batch
    
    def __len__(self):
        return len(self.replay_memory)
    
    def is_full(self):
        return len(self.replay_memory)==self.memory_size

def plot_graph(reward_history):
    plt.plot(range(len(reward_history)), reward_history, marker='', linewidth=1, alpha=0.9, label='y')
    plt.xlabel("episode", fontsize=12)
    plt.ylabel("score", fontsize=12)
    plt.show()
    # plt.savefig('score.png')