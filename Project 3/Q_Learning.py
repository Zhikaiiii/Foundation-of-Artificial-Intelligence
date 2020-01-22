import numpy as np
import pandas as pd


# Q_learning算法主体
class Q_Learning:
    # actions 行动列表
    # learning_rate 学习率
    # discount_rate 折现率
    # epsilon epsilon贪心策略的最优策略概率
    def __init__(self, actions, learning_rate=0.1, discount_rate=0.9, epsilon =0.75):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_rate
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # 根据epsilon贪心策略做行为策略
    def make_decision(self, state, episode, flag):
        self.add_state(state)
        if (episode+1) % 400 == 0 and flag:
            self.epsilon = (1-self.epsilon)*0.5 + self.epsilon
        prob = np.random.uniform()
        if prob >= self.epsilon:
            # 随机选取策略
            action = np.random.choice(self.actions)
        else:
            # 选取最优策略 由于可能有多个最优策略 所以需要做最优选择
            action_value = self.q_table.loc[state]
            max_value = np.max(action_value)
            idx = action_value[action_value.values == max_value].index
            action = np.random.choice(idx)
        return action

    # 更新行为价值
    def update_q_table(self, state, action, next_state, reward, type):
        self.add_state(next_state)
        old_value = self.q_table.loc[state, action]
        #行动价值更新
        if type == 1: # E-Sarsa
            new_value = reward + self.gamma*(self.epsilon*np.max(self.q_table.loc[next_state]) +
                                         (1 - self.epsilon) * np.sum(self.q_table.loc[next_state]) * 0.2)
        else: # q-learning
            new_value = reward + self.gamma*(np.max(self.q_table.loc[next_state]))
        self.q_table.loc[state, action] = self.q_table.loc[state, action] + self.lr*(new_value - old_value)
        self.q_table.loc[state, action] = np.around(self.q_table.loc[state, action], decimals=6)

    # 在q_table中添加状态
    def add_state(self, state):
        if state not in self.q_table.index:
            # 添加状态
            new_state = pd.Series(data=[0] * len(self.actions), index=self.actions, name=state,)
            self.q_table = self.q_table.append(new_state)

    # 找到最终对应的路径
    def final_move(self, size):
        state = [0, 0, 0]
        path = [[0, 0]]
        while True:
            action_value = self.q_table.loc[str(state)]
            max_value = np.max(action_value)
            idx = action_value[action_value.values == max_value].index # 找到qtable中价值最大的行动
            action = np.random.choice(idx)
            if action == 'down':
                state[0] = state[0] + 1
            elif action == 'up':
                state[0] = state[0] - 1
            elif action == 'left':
                state[1] = state[1] - 1
            elif action == 'right':
                state[1] = state[1] + 1
            state[2] = state[2] + 1
            if state[2] > 3:
                state[2] = 0
            print(action, "->", state)
            path.append(state.copy())
            if state[0] == size - 1 and state[1] == size - 1:
                break
        return path