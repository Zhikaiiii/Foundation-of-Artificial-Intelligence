import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# 超参数
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # 最优选择动作百分比
GAMMA = 0.9  # 奖励递减参数
TARGET_REPLACE_ITER = 100  # Q 现实网络的更新频率
MEMORY_CAPACITY = 2000  # 记忆库大小
N_ACTIONS = 5  # 动作数量
N_STATES = 3


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc1.bias.data.zero_()
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        self.out.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self,actions_num, learning_rate=0.1, discount_rate=0.9, epsilon =0.5):
        # 建立 target net 和 eval net
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 +2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # torch 的优化器
        self.loss_func = nn.MSELoss()  # 误差公式
        self.actions = actions_num
        self.lr = learning_rate
        self.gamma = discount_rate
        self.epsilon = epsilon

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if self.epsilon < 0.999:
            self.epsilon = self.epsilon + 0.0004
        # x = Variable(x)
        # 这里只输入一个 sample
        if np.random.uniform() < self.epsilon:  # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]  # return the argmax
        else:  # 选随机动作
            action = np.random.randint(0, self.actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1: N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        # max_action = torch.max(q_next,1)[0]
        # max_action = max_action.unsqueeze(-1)
        q_target = b_r  + self.gamma * q_next.max(1)[0].view(BATCH_SIZE, 1) # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 找到最终对应的路径
    def final_move(self, size):
        state = [0, 0, 0]
        path = [[0, 0]]
        action_list = ['down', 'up', 'left', 'right','wait']
        # self.target_net.eval()
        while True:
            # action_value = self.q_table.loc[str(state)]
            # max_value = np.max(action_value)
            # idx = action_value[action_value.values == max_value].index # 找到qtable中价值最大的行动

            # action = np.random.choice(idx)
            state_tensor = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
            actions_value = self.target_net(state_tensor)
            action_num = torch.max(actions_value, 1)[1].numpy()[0]  # return the argmax

            action = action_list[action_num]
            if action == 'down':
                state[0] = state[0] + 1
            elif action == 'up':
                state[0] = state[0] - 1
            elif action == 'left':
                state[1] = state[1] - 1
            elif action == 'right':
                state[1] = state[1] + 1
            state[2] = state[2] + 1
            print(action, "->", state)
            path.append(state.copy())
            if state == [size - 1, size - 1,]:
                break
        return path