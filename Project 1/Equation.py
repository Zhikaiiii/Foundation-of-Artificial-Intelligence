import re

#等式类
#包含等式的数字、运算符、字符串表示等信息
class Equation():
    num_list = [] #等式的六位数字列表
    operator_1 = '' #第一个运算符
    operator_2 = '' #第二个运算符
    equation_str = ''  #等式的字符串表示
    operator_list = ['+', '-', '*']

    #通过输入的数字列表和运算符来初始化等式
    def set_para(self, num_list, oper1, oper2):
        self.num_list = num_list.copy()
        self.operator_1 = oper1
        self.operator_2 = oper2
        num1 = num_list[0] * 10 + num_list[1]
        num2 = num_list[2] * 10 + num_list[3]
        num3 = num_list[4] * 10 + num_list[5]
        self.equation_str = str(num1) + oper1 + str(num2) + oper2 + str(num3)

    #通过输入的字符串来初始化等式
    def set_text(self, equation_str):
        if re.match('(\d\d{0,1})([+*-=])(\d\d{0,1})([+*-=])(\d\d{0,1})', equation_str):#正则表达式匹配
            result = re.match('(\d\d{0,1})([+*-=])(\d\d{0,1})([+*-=])(\d\d{0,1})', equation_str)
            self.equation_str = equation_str
            self.num_list = [int(result.group(1))//10, int(result.group(1))%10, int(result.group(3))//10, \
                             int(result.group(3))%10, int(result.group(5))//10, int(result.group(5))%10]
            self.operator_1 = result.group(2)
            self.operator_2 = result.group(4)
