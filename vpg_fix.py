import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

# 创建一个简单的神经网络作为Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax()
        )
        
    def forward(self, x):
        x = x.view(1, -1)
        return self.fc(x)

# 创建一个简单的神经网络作为Value Network
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.fc(x)

# 定义参数
input_dim = 4  # CartPole-v0中的观测空间维度
output_dim = 2  # CartPole-v0中的动作空间维度
learning_rate = 0.001
gamma = 0.99

# 创建环境
env = gym.make('CartPole-v0')
policy_net = PolicyNetwork(input_dim, output_dim)
value_net = ValueNetwork(input_dim)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=learning_rate)
optimizer_value = optim.Adam(value_net.parameters(), lr=0.0001)

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

    batch_logits, batch_values, batch_rewards, batch_actions = [], [], [], []
    ITER_REW = 0
    
    for _ in range(32):
        logits_list, values_list, rewards_list, actions_list = [], [], [], []
        state, _ = env.reset()
        while True:
            state = torch.tensor(state, dtype=torch.float32)
            logits = policy_net(state)
            value = value_net(state)
            action = torch.multinomial(logits[0], 1, replacement=False)
            # a2 = logits.detach().cpu().numpy()
            # action = np.random.choice([0, 1], 1, replace=True, p=a2[0])
            state, reward, done, _, _ = env.step(action.item())
            ITER_REW += reward

            logits_list.append(logits)
            values_list.append(value)
            rewards_list.append(reward)
            actions_list.append(action[0])

            if done:
                batch_logits.append(logits_list)
                batch_values.append(values_list)
                batch_rewards.append(rewards_list)
                batch_actions.append(actions_list)
                break
    
    print("ITERATION:", i+1, "AVG REWARD:", ITER_REW/32)


    # ### update way 1 ###
    # total_policy_loss = 0
    # # total_value_loss = 0
    # for rewards_list,values_list,logits_list,actions_list in zip(batch_rewards,batch_values,batch_logits,batch_actions):
    #     # print(len(rewards_list))
    #     # print(len(values_list))
    #     # print(len(logits_list))
    #     # print(len(actions_list))

    #     logits_list = torch.stack(logits_list).squeeze()
    #     actions_list = torch.stack(actions_list)
    #     returns = compute_returns(rewards_list)
    #     advantage = torch.tensor(returns) - torch.cat(values_list)
    #     log_prob = F.log_softmax(logits_list, dim=1)
    #     policy_loss = (log_prob.gather(1, actions_list).squeeze().dot(advantage.detach()))
    #     value_loss = F.smooth_l1_loss(torch.cat(values_list), torch.tensor(returns))
    #     # # value_loss = torch.nn.MSELoss()(torch.cat(values_list), torch.tensor(returns))

    #     total_policy_loss += policy_loss
    #     total_value_loss += value_loss

    ### update way 2 ###
    total_policy_loss = torch.tensor([0]).float().cuda()
    total_value_loss = torch.tensor([0]).float().cuda()
    for rewards_list,values_list,logits_list,actions_list in zip(batch_rewards,batch_values,batch_logits,batch_actions):
        logits_list = torch.stack(logits_list).squeeze().cuda()
        returns = []
        for t in range(len(rewards_list)):
            r_t = 0
            log_prob = logits_list[t][actions_list[t]]
            temp = rewards_list[t:]
            for i, reward in enumerate(temp):
                r_t += gamma**i * reward
            
            returns.append(r_t)
            # advantage = torch.FloatTensor([r_t- values_list[t]]).cuda() 
            advantage = torch.FloatTensor([r_t]).cuda()
            policy_loss = -log_prob * advantage
            total_policy_loss += policy_loss
        
        value_loss = F.mse_loss(torch.cat(values_list), torch.tensor(returns))
        total_value_loss -= value_loss
            
    total_policy_loss /= len(batch_rewards)
    total_value_loss /= len(batch_rewards)

    # print("total_policy_loss:", total_policy_loss, "total_value_loss:", total_value_loss)

    
    
    total_policy_loss.backward()
    optimizer_policy.step()
    optimizer_policy.zero_grad()

    total_value_loss.backward()
    optimizer_value.step()
    optimizer_value.zero_grad()

        # print(len(rewards_list))