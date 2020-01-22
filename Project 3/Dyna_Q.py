import numpy as np
import pandas as pd


class DynaQ:
    def __init__(self, actions, learning_rate=0.01, discount_rate=0.9, epsilon =0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_rate
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # 根据epsilon贪心策略做行为策略
    def make_decision(self, state):
        self.add_state(state)
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

    # 在q_table中添加状态
    def add_state(self, state):
        if state not in self.q_table.index:
            # 添加状态
            new_state = pd.Series(data=[0] * len(self.actions), index=self.actions, name=state,)
            self.q_table = self.q_table.append(new_state)

    def learn(self, state, action, reward, next_state):
        self.add_state(next_state)
        old_value = self.q_table.loc[state, action]
        new_value = reward + self.gamma*(np.max(self.q_table.loc[next_state]))
        self.q_table.loc[state, action] = self.q_table.loc[state, action] + self.lr*(new_value - old_value)

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
            if state[2] > 6:
                state[2] = 0
            print(action, "->", state)
            path.append(state.copy())
            if state[0] == size - 1 and state[1] == size - 1:
                break
        return path

class EnvModel:
    """Similar to the memory buffer in DQN, you can store past experiences in here.
    Alternatively, the model can generate next state and reward signal accurately."""
    def __init__(self, actions):
        # the simplest case is to think about the model is a memory which has all past transition information
        self.actions = actions
        self.database = pd.DataFrame(columns=actions, dtype=np.object)

    def store_transition(self, state, action, reward, next_state):
        if state not in self.database.index:
            new_record = pd.Series([None] * len(self.actions), index=self.actions, name=state)
            self.database = self.database.append(new_record)
        self.database.set_value(state, action, (reward, next_state))

    # 随机抽取状态
    def sample_s_a(self):
        state = np.random.choice(self.database.index)
        action= np.random.choice(self.database.ix[state].dropna().index)    # filter out the None value
        return state, action

    # 得到其回报
    def get_r_s_(self, state, action):
        reward, next_state = self.database.ix[state, action]
        return reward, next_state