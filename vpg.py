import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

# 创建一个简单的神经网络作为Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

# 创建一个简单的神经网络作为Value Network
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义参数
input_dim = 4  # CartPole-v0中的观测空间维度
output_dim = 2  # CartPole-v0中的动作空间维度
learning_rate = 0.1
gamma = 0.99

# 创建环境
env = gym.make('CartPole-v0')
policy_net = PolicyNetwork(input_dim, output_dim)
value_net = ValueNetwork(input_dim)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=learning_rate)
optimizer_value = optim.Adam(value_net.parameters(), lr=learning_rate)

# 定义损失函数
def compute_returns(rewards):
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# 训练模型
for i in range(100):  # 可以根据需要更改训练次数
    logits_list, values_list, rewards_list, actions_list = [], [], [], []
    state, _ = env.reset()

    while True:
        state = torch.tensor(state, dtype=torch.float32)
        logits = policy_net(state)
        value = value_net(state)

        action = torch.multinomial(logits, 1)
        state, reward, done, _, _ = env.step(action.item())

        logits_list.append(logits)
        values_list.append(value)
        rewards_list.append(reward)
        actions_list.append(action)

        if done:
            returns = compute_returns(rewards_list) # 
            advantage = torch.tensor(returns) - torch.cat(values_list)
            log_prob = F.log_softmax(torch.stack(logits_list), dim=1)
            # print(log_prob.gather(1, torch.stack(actions_list)).squeeze().shape)
            # print(advantage.shape)
            policy_loss = (log_prob.gather(1, torch.stack(actions_list)).squeeze().dot(advantage.detach()))
            # print(policy_loss)
            value_loss = F.smooth_l1_loss(torch.cat(values_list), torch.tensor(returns))
            # value_loss = torch.nn.MSELoss()(torch.cat(values_list), torch.tensor(returns))

            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            optimizer_policy.step()
            optimizer_value.step()

            print(len(rewards_list))
            break