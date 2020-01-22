import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Maze import Maze
import time
from Q_Learning import Q_Learning

MEMORY_CAPACITY = 2000  # 记忆库大小
class MainWindow(tk.Tk, object):
    def __init__(self, size=6):
        super(MainWindow, self).__init__()
        self.title('maze')
        self.geometry("800x600")

        # 界面控件
        self.button_confirm = tk.Button(text="生成地图", command=self.show_maze, width=8, font=("幼圆",14))
        self.button_confirm.pack()
        self.button_begin = tk.Button(text="寻找路径", command=self.find_path, width=8, font=("幼圆",14))
        self.button_begin.pack()
        self.button_showpath = tk.Button(text="显示路径", command=self.show_path, width=8, font=("幼圆",14))
        self.button_begin.pack()
        self.v = tk.StringVar()
        self.label_progress = tk.Label(textvariable=self.v, font=("幼圆",16))
        self.label_progress.pack()
        info = """
        Tips：
                根据下拉框选择类型
                  并自动生成地图。
                   黑色代表墙壁
                   蓝色代表夹子
                   橙色代表火焰
                   红色代表老鼠
                   黄色代表终点
                    Have fun :)
        """
        self.label_info = tk.Label(text=info,font=("幼圆",14),anchor="w")
        self.label_info.pack()
        self.menu_size = ttk.Combobox(width=8, height=3, font="幼圆")
        self.menu_size['value'] = ("6x6", "8x8", "10x10", "12x12")
        self.size_dict = {"6x6":6, "8x8":8, "10x10":10, "12x12":12}
        self.difficulty_dict = {"简单":0, "中等":1, "困难":2}
        self.type_dict = {"q-learning":0, "E-Sarsa":1}
        self.menu_size.pack()
        self.menu_size.current(0)
        self.menu_difficulty = ttk.Combobox(width=8, height=3, font="幼圆")
        self.menu_difficulty['value'] = ("简单", "中等", "困难")
        self.menu_difficulty.pack()
        self.menu_difficulty.current(0)
        self.menu_type = ttk.Combobox(width=12, height=3, font="幼圆")
        self.menu_type['value'] = ("q-learning", "E-Sarsa")
        self.menu_type.pack()
        self.menu_type.current(0)
        self.menu_type.place(relx=0.77,rely=0.08)
        self.menu_size.place(relx=0.8, rely=0.16)
        self.menu_difficulty.place(relx=0.8, rely=0.24)
        self.button_confirm.place(relx=0.74, rely=0.3)
        self.button_begin.place(relx=0.86, rely=0.3)
        self.button_showpath.place(relx=0.8, rely=0.4)
        self.label_progress.place(relx=0.7, rely=0.5)
        self.label_info.place(relx=0.5, rely=0.65)
        self.canvas = tk.Canvas(self, bg='white', height=480, width=480)
        self.canvas.place(x=40, y=40)
        #初始化其他参数
        self.size = size
        self.difficulty = 0
        self.maze = Maze()  # 迷宫类
        action_list = self.maze.action
        self.model = Q_Learning(action_list)  # Q_Learning算法
        self.show_maze()

    # 生成迷宫
    def show_maze(self):
        self.difficulty = self.difficulty_dict[self.menu_difficulty.get()]
        self.size = self.size_dict[self.menu_size.get()]
        self.maze.generate_maze(self.size, self.difficulty)
        self.canvas.delete('all')
        self.canvas.create_rectangle(2,2,480,480)
        unit = int(480/self.size)
        for c in range(0, 480, unit):
            x0, y0, x1, y1 = c, 0, c, 480
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, 480, unit):
            x0, y0, x1, y1 = 0, r, 480, r
            self.canvas.create_line(x0, y0, x1, y1)
        # 老鼠位置
        self.mouse = self.canvas.create_rectangle(2, 2, unit, unit, fill='green')
        self.terminal = self.canvas.create_rectangle(480-unit, 480-unit, 480, 480, fill="yellow")
        self.maze_update([0,0])

    # 找到路径
    def find_path(self):
        self.model.q_table = pd.DataFrame(columns=self.model.actions, dtype=np.float64)
        self.learn_path()
        # self.path = self.dqn.final_move(self.size)
        info = "训练已完成 : )" + "\n" + "老鼠不会饿死了!!！\n 点击可再次查看路径"
        self.v.set(info)
        self.path = self.model.final_move(self.size)
        now_time = 0
        for loc in self.path:
            # self.flame_shine(now_time)
            now_time = now_time + 1
            self.maze.flame_shine(now_time)
            self.maze_update(loc)
            time.sleep(0.5)

        # 找到路径
    def show_path(self):
        now_time = 0
        for loc in self.path:
            # self.flame_shine(now_time)
            now_time = now_time + 1
            self.maze.flame_shine(now_time)
            self.maze_update(loc)
            time.sleep(0.5)

    def maze_update(self, loc):
        unit = int(480 / self.size)
        # 地图位置
        for i in range(self.size):
            for j in range(self.size):
                if self.maze.maze[i][j] == 1:
                    self.canvas.create_rectangle(j * unit, i * unit, (j + 1) * unit, (i + 1) * unit, fill="black")
                elif self.maze.maze[i][j] == 2:
                    self.canvas.create_rectangle(j * unit, i * unit, (j + 1) * unit, (i + 1) * unit, fill="blue")
                elif self.maze.maze[i][j] == -1:
                    self.canvas.create_rectangle(j * unit, i * unit, (j + 1) * unit, (i + 1) * unit, fill="orange")
                else:
                    self.canvas.create_rectangle(j * unit, i * unit, (j + 1) * unit, (i + 1) * unit, fill="white")
        self.canvas.delete(self.mouse)
        # 移动老鼠位置
        self.terminal = self.canvas.create_rectangle(480-unit, 480-unit, 480, 480, fill="yellow")
        self.mouse = self.canvas.create_rectangle(loc[1] * unit, loc[0] * unit, (loc[1] + 1) * unit,
                                                  (loc[0] + 1) * unit, fill="red")
        self.update()

    # 学习过程
    def learn_path(self):
        type = self.type_dict[self.menu_type.get()]
        action_list = ['down','up','left','right','wait']
        reward_list = []
        k_list = []
        for episode in range(2000):
            info = "正在学习中（:   " + str(round(episode/20)) + "%"
            self.v.set(info)
            self.update()
            # 初始化
            observation = self.maze.reset()
            reward_acc = 0
            k = 0
            flag = True
            while True:
                # 根据Q_Learning选择行动
                action = self.model.make_decision(str(observation),episode,flag)
                flag = False
                # 得到回报和新状态
                new_observation, reward, flag = self.maze.move(action)
                # 更新Q_Table
                self.model.update_q_table(str(observation), action, str(new_observation), reward, type)
                # 得到新状态
                observation = new_observation
                reward_acc += reward
                k +=1
                # 达到终止态时结束
                if flag:
                    break
            # if episode%100 == 0:
            #     reward_list.append(reward_acc)
            #     k_list.append(k)
        print(np.average(k_list))
        # plt.figure()
        # plt.plot(reward_list)
        # plt.title("acc reward")
        # plt.show()
        # plt.figure()
        # plt.plot(k_list)
        # plt.title("running times")
        # plt.show()


