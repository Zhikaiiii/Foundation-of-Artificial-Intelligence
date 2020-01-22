import numpy as np


# 定义迷宫
class Maze:
    def __init__(self, maze_size=6):
        # 0表示道路，1表示墙壁, -1表示老鼠夹子
        self.maze = np.array(
            [[0, 0, 0, 0, 0, 0, ],
             [2, 0, 1, 1, 1, 1, ],
             [1, 0, 2, 0, 0, 0, ],
             [1, 0, 0, 0, 1, 0, ],
             [0, 1, 1, 2, 0, 0, ],
             [1, 1, 0, 1, 0, 0, ]]
        )
        self.maze_size = maze_size
        self.flame_list = {7:[3,1],17:[2,0],32:[4,0]}
        self.origin = [0, 0]  # 起点坐标
        self.terminal = [self.maze_size-1, self.maze_size-1]  # 终点坐标
        self.action = ['down', 'up', 'left', 'right','wait']  # 行动列表
        self.location = [0, 0]  # 老鼠位置
        self.now_time = 0 # 表示时间



    def move(self, action):
        old_location = self.location.copy()
        new_location = old_location.copy()
        self.flame_shine(self.now_time)
        # 根据行动改变位置
        if action == 'down':
            if old_location[0] < self.maze_size - 1 and self.maze[old_location[0] + 1, old_location[1]] != 1:
                new_location[0] = new_location[0] + 1
        elif action == 'up':
            if old_location[0] > 0 and self.maze[old_location[0] - 1, old_location[1]] != 1:
                new_location[0] = new_location[0] - 1
        elif action == 'left':
            if old_location[1] > 0 and self.maze[old_location[0], old_location[1] - 1] != 1:
                new_location[1] = new_location[1] - 1
        elif action == 'right':
            if old_location[1] < self.maze_size - 1 and self.maze[old_location[0], old_location[1] + 1] != 1:
                new_location[1] = new_location[1] + 1
        flag = False  # 是否到达终点
        if new_location == self.terminal:  # 到达终点
            reward = 10
            flag = True
        elif new_location == old_location and action != 'wait':  # 碰墙
            reward = -1
        elif self.maze[new_location[0], new_location[1]] == 2 or self.maze[new_location[0], new_location[1]] == -1:  # 老鼠夹子或火焰
            reward = -10
            flag = True
        else:
            reward = -0.5
        self.now_time = self.now_time + 1
        if self.now_time > 3:
            self.now_time = 0
        self.location = new_location.copy()
        new_location.append(self.now_time)
        return new_location, reward, flag

    # 重新初始化迷宫
    def reset(self):
        self.location = self.origin.copy()
        self.now_time = 0
        initial_state = self.location.copy()
        initial_state.append(self.now_time)
        return initial_state

    def generate_maze(self, size, difficulty):
        self.maze_size = size
        self.terminal = [size-1, size-1]
        square_num = size * size
        self.maze = np.ones([size,size], dtype=int)
        self.flame_list = {}
        while not self.has_solve():
            self.flame_list = {}
            rand_array = np.random.uniform(size=size*size)
            self.maze = np.zeros(size*size, dtype=int)
            if difficulty == 0:
                # 产生墙壁
                self.maze[rand_array < 0.25] = 1
                # 产生老鼠夹子
                self.maze[rand_array > 0.97] = 2
                # 产生火焰
                self.maze[np.argmin(rand_array)] = -1
                # 随机产生火焰对应的周期
                self.flame_list[np.argmin(rand_array)] = [np.random.randint(2, 5, 1), np.random.randint(0, 2, 1)]
                for i in range(2):
                    rand_array[np.argmin(rand_array)] = 1
                    self.maze[np.argmin(rand_array)] = -1
                    self.flame_list[np.argmin(rand_array)] = [np.random.randint(2,5,1),np.random.randint(0,2,1)]
            elif difficulty == 1:
                self.maze[rand_array < 0.35] = 1
                self.maze[rand_array > 0.94] = 2
                self.maze[np.argmin(rand_array)] = -1
                self.flame_list[np.argmin(rand_array)] = [np.random.randint(2, 5, 1), np.random.randint(0, 2, 1)]
                for i in range(3):
                    rand_array[np.argmin(rand_array)] = 1
                    self.maze[np.argmin(rand_array)] = -1
                    self.flame_list[np.argmin(rand_array)] = [np.random.randint(2,5,1),np.random.randint(0,2,1)]
            elif difficulty == 2:
                self.maze[rand_array < 0.45] = 1
                self.maze[rand_array > 0.9] = 2
                self.maze[np.argmin(rand_array)] = -1
                self.flame_list[np.argmin(rand_array)] = [np.random.randint(2, 5, 1), np.random.randint(0, 2, 1)]
                for i in range(5):
                    rand_array[np.argmin(rand_array)] = 1
                    self.maze[np.argmin(rand_array)] = -1
                    self.flame_list[np.argmin(rand_array)] = [np.random.randint(2,5,1),np.random.randint(0,2,1)]
            self.maze = self.maze.reshape([size,size])
            if 0 in self.flame_list:
                self.flame_list.pop(0)
            if size*size-1 in self.flame_list:
                self.flame_list.pop(size*size-1)
            self.maze[0, 0] = 0
            self.maze[size-1, size-1] = 0

    # 火焰周期性变化
    def flame_shine(self,time):
        for flame_loc in self.flame_list.keys():
            row = flame_loc // self.maze_size
            column = flame_loc % self.maze_size
            if (time + self.flame_list[flame_loc][1])%self.flame_list[flame_loc][0] == 0:
                self.maze[row,column] = -1  # 产生火焰 不能通过
            else:
                self.maze[row,column] = 0   # 道路 可以通过

    # A*算法检验迷宫是否有解
    def has_solve(self):
        size = self.maze_size
        n = Node(2*(self.maze_size-1), 0, [0,0])
        open_list = [n]
        close_list = []
        flag = False
        while flag == False and len(open_list) > 0:
            new_node = max(open_list, key=get_f)  # 找出open表中f最大的点
            open_list.remove(new_node)
            close_list.append(new_node)
            for successor in self.get_successor(new_node.loc):
                mark = False
                if successor == self.terminal:
                    flag = True
                    break
                g_value = new_node.g + 1
                for node in open_list:
                    if successor == node.loc:
                        if g_value < node.g:  # 如果后继节点在open表中
                            node.f = node.f - node.g + g_value
                        mark = True   # 表示无需将后继插入open表中
                        continue
                for node in close_list:
                    if successor == node.loc:
                        if g_value < node.g:   # 如果后继节点在close表中
                            close_list.remove(node) # 从close表中移除
                            h_value = abs(successor[0] - size + 1) + abs(successor[1] - size + 1)
                            new_node = Node(g_value + h_value, g_value, successor)
                            open_list.append(node)
                        mark = True
                        continue
                if not mark:
                    h_value = abs(successor[0] - size + 1) + abs(successor[1] - size + 1)
                    node = Node(g_value + h_value, g_value, successor)
                    open_list.append(node)
        return flag

    # 得到后继节点
    def get_successor(self, loc):
        successor = []
        if loc[0]>0 and self.maze[loc[0]-1, loc[1]] <= 0 :
            successor.append([loc[0]-1, loc[1]])
        if loc[0]<self.maze_size-1 and self.maze[loc[0]+1, loc[1]] <= 0:
            successor.append([loc[0]+1, loc[1]])
        if loc[1] > 0 and self.maze[loc[0], loc[1] - 1] <= 0:
            successor.append([loc[0], loc[1] - 1])
        if loc[1] < self.maze_size - 1 and self.maze[loc[0], loc[1] + 1] <= 0:
            successor.append([loc[0], loc[1] + 1])
        return successor

# 根据f值排序
def get_f(n):
    return n.f

# 节点类 包含位置、f值和g值
class Node:
    def __init__(self, f, g, loc):
        self.f = f
        self.g = g
        self.loc = loc

