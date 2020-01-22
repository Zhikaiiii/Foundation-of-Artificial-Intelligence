import re
from Equation import  Equation
import random
#求解结果的核心算法

class Result_Search():
    def __init__(self):
        #加一根火柴的变化
        self.add_list = [
            [8], [7], [], [9], [], [6, 9], [8], [], [], [8]
        ]
        #减一根火柴
        self.minus_list = [
            [], [], [], [], [], [], [5], [1], [0, 6, 9], [3,5]
        ]
        #z自身变化
        self.conversion_list =  [
            [6, 9], [], [3], [2, 5], [], [3], [0, 9], [], [], [0, 6]
        ]
        #加两根火柴
        self.double_add_list = [
            [], [4], [8], [8], [9], [8], [], [3], [], []
        ]
        #减两根火柴
        self.double_minus_list = [
            [], [], [], [7], [1], [], [], [], [2, 3, 5], [4]
        ]
        #自身变化
        self.double_conversion_list = [
            [], [], [5], [], [], [2], [], [], [], []
        ]

    #移动一根火柴
    def move_one_match(self, equation = Equation):
        answer_list = []
        oper = equation.operator_1
        num_list = equation.num_list.copy()
        #特殊情况1 等号和减号互换
        if oper == '-':
            temp_list = []
            for i in range(-4,2):
                temp_list.append(num_list[i])
            if self.judge_equation(temp_list, oper):
                num1 = num_list[0] * 10 + num_list[1]
                num2 = num_list[2] * 10 + num_list[3]
                num3 = num_list[4] * 10 + num_list[5]
                answer_list.append(str(num1) + '=' + str(num2) + '-' + str(num3))
        #先寻找每个数自身变换是否成立
        for i in range(6):
            num = num_list[i]
            #如果一个数的十位为0，则跳过
            if i%2 == 0 and num == 0:
                continue
            #在该位数字对应的可以自身变换的数字中
            for conversion_number in self.conversion_list[num]:
                temp_list = num_list.copy()
                temp_list[i] = conversion_number
                if self.judge_equation(temp_list, oper):#判断更换数字后等式是否成立
                    num1 = temp_list[0] * 10 + temp_list[1]
                    num2 = temp_list[2] * 10 + temp_list[3]
                    num3 = temp_list[4] * 10 + temp_list[5]
                    answer_list.append(str(num1)+ oper + str(num2) + '=' + str(num3))
        #寻找数字之间的变换
        for j in range(6):
            num = num_list[j]
            #如果一个数的十位为0，则跳过
            if j%2 == 0 and num == 0:
                continue
            #先寻找一个数减少一根火柴
            for minus_number in self.minus_list[num]:
                temp_list = num_list.copy()
                temp_list[j] = minus_number
                #特殊情况2 减号变加号
                if oper == '-' and self.judge_equation(temp_list, '+'):
                    num1 = temp_list[0] * 10 + temp_list[1]
                    num2 = temp_list[2] * 10 + temp_list[3]
                    num3 = temp_list[4] * 10 + temp_list[5]
                    answer_list.append(str(num1) + '+' + str(num2) + '=' + str(num3))
                #再寻找一个数增加一根火柴
                for k in range(6):
                    temp_num = temp_list[k]
                    #十位为0则跳过
                    if k%2 == 0 and temp_num == 0:
                        continue
                    #变成相应数字
                    for add_number in self.add_list[temp_list[k]]:
                        temp_list[k] = add_number
                        if self.judge_equation(temp_list, oper):#判断等式是否成立
                            num1 = temp_list[0] * 10 + temp_list[1]
                            num2 = temp_list[2] * 10 + temp_list[3]
                            num3 = temp_list[4] * 10 + temp_list[5]
                            answer_list.append(str(num1) + oper + str(num2) + '=' + str(num3))
                        temp_list[k] = temp_num #还原temp_list,进行下一步搜索
        #特殊情况3 加号变减号
        for i in range(6):
            for add_num in self.add_list[num_list[i]]:
                temp_list = num_list.copy()
                temp_list[i] = add_num
                if self.judge_equation(temp_list, '-') and  oper == '+':
                    num1 = temp_list[0] * 10 + temp_list[1]
                    num2 = temp_list[2] * 10 + temp_list[3]
                    num3 = temp_list[4] * 10 + temp_list[5]
                    answer_list.append(str(num1) + '-' + str(num2) + '=' + str(num3))
        final_answer_list = []
        #最终结果去掉了原来就成立的等式
        for answer in answer_list:
            if answer not in final_answer_list and answer != equation.equation_str:
                final_answer_list.append(answer)
        return final_answer_list

    #移动两根火柴
    def move_two_match(self, equation = Equation):
        answer_list = []
        oper = equation.operator_1
        num_list = equation.num_list.copy()
        #特殊情况1 等号和加号互换
        if oper == '+':
            temp_list = []
            for i in range(-4,2):
                temp_list.append(num_list[i])
            if self.judge_equation(temp_list, oper):
                num1 = num_list[0] * 10 + num_list[1]
                num2 = num_list[2] * 10 + num_list[3]
                num3 = num_list[4] * 10 + num_list[5]
                answer_list.append(str(num1) + '=' + str(num2) + '+ ' + str(num3))
        #特殊情况2乘号和加号互相变换
        if oper == '+' and self.judge_equation(num_list, '*'):
            num1 = num_list[0] * 10 + num_list[1]
            num2 = num_list[2] * 10 + num_list[3]
            num3 = num_list[4] * 10 + num_list[5]
            answer_list.append(str(num1) + '*' + str(num2) + '=' + str(num3))
        if oper == '*' and self.judge_equation(num_list, '+'):
            num1 = num_list[0] * 10 + num_list[1]
            num2 = num_list[2] * 10 + num_list[3]
            num3 = num_list[4] * 10 + num_list[5]
            answer_list.append(str(num1) + '+' + str(num2) + '=' + str(num3))
        #寻找自身变换是否成立
        for i in range(6):
            num = num_list[i]
            # 如果一个数的十位为0，则跳过
            if i % 2 == 0 and num == 0:
                continue
            # 在该位数字对应的可以自身变换的数字中
            for conversion_number in self.double_conversion_list[num]:
                temp_list = num_list.copy()
                temp_list[i] = conversion_number
                if self.judge_equation(temp_list, oper):
                    num1 = temp_list[0] * 10 + temp_list[1]
                    num2 = temp_list[2] * 10 + temp_list[3]
                    num3 = temp_list[4] * 10 + temp_list[5]
                    answer_list.append(str(num1) + oper + str(num2) + '=' + str(num3))
        #寻找两个数字之间的变换
        for j in range(6):
            num = num_list[j]
            #如果一个数的十位为0，则跳过
            if j%2 == 0 and num == 0:
                continue
            for minus_number in self.double_minus_list[num]:
                temp_list = num_list.copy()
                temp_list[j] = minus_number
                for k in range(6):
                    for add_number in self.double_add_list[temp_list[k]]:
                        temp_num = temp_list[k]
                        temp_list[k] = add_number
                        if self.judge_equation(temp_list, oper):
                            num1 = temp_list[0] * 10 + temp_list[1]
                            num2 = temp_list[2] * 10 + temp_list[3]
                            num3 = temp_list[4] * 10 + temp_list[5]
                            answer_list.append(str(num1) + oper + str(num2) + '=' + str(num3))
                        temp_list[k] = temp_num
        # 移动一根后再执行move_one_match
        for j in range(6):
            num = num_list[j]
            if j%2 == 0 and num == 0:
                continue
            for minus_number in self.minus_list[num]:
                temp_list = num_list.copy()
                temp_list[j] = minus_number #将选中数字替换为少一根火柴棍的数字
                for k in range(6):
                    temp_num = temp_list[k]
                    if k%2 == 0 and temp_num == 0:
                        continue
                    for add_number in self.add_list[temp_list[k]]:#选择一个数字换为多一根火柴棍的数字
                        temp_list[k] = add_number
                        equation_temp = Equation()
                        equation_temp.set_para(temp_list, oper, '=')
                        answer_list = answer_list + self.move_one_match(equation_temp) #将移动一根火柴的结果加入表中
                        temp_list[k] = temp_num
                #特殊情况 等号在前，减号变加号
                if oper == '-':
                    temp_list_2 = temp_list[2:6] + temp_list[0:2]
                    if self.judge_equation(temp_list_2, '+'):
                        num1 = temp_list[0] * 10 + temp_list[1]
                        num2 = temp_list[2] * 10 + temp_list[3]
                        num3 = temp_list[4] * 10 + temp_list[5]
                        answer_list.append(str(num1) + '=' + str(num2) + '+' + str(num3) )
        final_answer_list = []
        for answer in answer_list:
            if answer not in final_answer_list and answer not in self.move_one_match(equation) and answer != equation.equation_str:
                final_answer_list.append(answer)
        return final_answer_list

    #判断等式是否成立
    def judge_equation(self, num_list, oper):
        num1 = num_list[0]*10 + num_list[1]
        num2 = num_list[2]*10 + num_list[3]
        num3 = num_list[4]*10 + num_list[5]
        if oper == '+':
            return num1 + num2 == num3
        elif oper == '-':
            return num1 - num2 == num3
        elif oper == '*':
            return num1 * num2 == num3
        else:
            return False

    #随机生成等式
    #type =0 表示移动一根火柴  type = 1 表示移动两根火柴
    #mode = 1 表示生成有解的等式  mode = 0 表示生成有解的不等式
    def equation_generate(type=0, mode= 0):
        solution = Result_Search()
        equation = Equation()
        num_1 = 44
        num_2 = 44
        num_3 = 44
        oper = random.choice(equation.operator_list)
        equation.set_text(str(num_1) + oper + str(num_2) + '=' + str(num_3))
        if type == 0:
            while len(solution.move_one_match(equation)) == 0:#x循环至有解为止
                num_1 = random.randint(0, 99)
                num_2 = random.randint(0, 99)
                oper = random.choice(equation.operator_list)
                if mode == 1:#随机生成num1，num2和oper，计算num3
                    if oper == '+':
                        num_3 = num_1 + num_2
                    elif oper == '-':
                        num_3 = num_1 - num_2
                    elif oper == '*':
                        num_3 = num_1 * num_2
                    else:
                        num_3 = -1
                    if num_3 < 0 or num_3 >99: #范围限定
                        continue
                else:
                    num_3 = random.randint(0, 99)
                equation.set_text(str(num_1) + oper + str(num_2) + '=' + str(num_3))
        elif type == 1:
            while len(solution.move_two_match(equation)) == 0:
                num_1 = random.randint(0, 99)
                num_2 = random.randint(0, 99)
                oper = random.choice(equation.operator_list)
                if mode == 1:
                    if oper == '+':
                        num_3 = num_1 + num_2
                    elif oper == '-':
                        num_3 = num_1 - num_2
                    elif oper == '*':
                        num_3 = num_1 * num_2
                    if num_3 < 0 or num_3 >99:
                        continue
                else:
                    num_3 = random.randint(0, 99)
                equation.set_text(str(num_1) + oper + str(num_2) + '=' + str(num_3))
        return equation

    #计算难度
    #type =0 表示移动一根火柴  type = 1 表示移动两根火柴
    #最终结果为所有变换数/解的数目/2
    def cal_difficulty(equation = Equation, type =0):
        solution = Result_Search()
        #如果等式原本就成立，难度为0
        if solution.judge_equation(equation.num_list, equation.operator_1):
            return 0
        if type == 0:
            answer_num = len(solution.move_one_match(equation))
            if answer_num == 0:#无解难度为1000
                return 1000
            conversion_num = 0
            for num in equation.num_list:
                conversion_num = conversion_num + len(solution.add_list[num]) + len(solution.minus_list[num]) \
                                 + len(solution.conversion_list[num])
            conversion_num = conversion_num //4
            return conversion_num // answer_num

        elif type == 1:
            answer_num = len(solution.move_two_match(equation))
            if answer_num == 0:
                return 1000
            conversion_num = 0
            for num in equation.num_list:
                conversion_num = conversion_num + len(solution.add_list[num]) + len(solution.minus_list[num]) + \
                                 len(solution.conversion_list[num]) + len(solution.double_add_list[num]) \
                                 + len(solution.double_minus_list[num]) + len(solution.double_conversion_list[num])
            conversion_num = conversion_num // 4
            return conversion_num // answer_num
