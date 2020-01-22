from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from MainWindow import Ui_MainWindow as Ui_Main
from Interface_Equation import Interface_Equation as Ui_Equation
import Operator_rc
import math
from Result_Search import Result_Search
from Equation import Equation
#主界面
#继承自Ui_MainWindow
class Interface(Ui_Main):
    def __init__(self, MyWindow):
        super(Interface, self).__init__()
        self.setupUi(MyWindow)
        self.select_win = QMainWindow()
        self.ui_select = Ui_Equation(self.select_win)#等式库界面
        self.solution = Result_Search()
        #界面ui和qss设置
        self.Lcd_List = [self.lcdNumber_1, self.lcdNumber_2, self.lcdNumber_3, self.lcdNumber_4, self.lcdNumber_5, self.lcdNumber_6]
        self.Lcd_Result_List = [self.lcdResult_1, self.lcdResult_2, self.lcdResult_3, self.lcdResult_4, self.lcdResult_5, self.lcdResult_6]
        self.Matrix = [self.pushButton_0, self.pushButton_1, self.pushButton_2, self.pushButton_3,
                       self.pushButton_4, self.pushButton_5, self.pushButton_6, self.pushButton_7,
                       self.pushButton_8, self.pushButton_9, self.pushButton_Add, self.pushButton_Minus,
                       self.pushButton_Mul, self.pushButton_Equal, self.pushButton_Back, self.pushButton_AC]
        self.Button = [self.Cal, self.Select, self.User_defined, self.RandomEqu ,self.Addtodata]
        for lcd in self.Lcd_List:
            lcd.setStyleSheet(" color: #87322a")
            lcd.setLineWidth(0)
        for lcd_result in self.Lcd_Result_List:
            lcd_result.setStyleSheet("color: #87322a")
            lcd_result.setLineWidth(0)
        self.Result_List.setStyleSheet("QListWidget{background-color:transparent}" "QListWidget::item::selected{background-color:transparent}");
        self.Result_List.setFocusPolicy(Qt.NoFocus)
        self.Result_List.setFrameShape(QFrame.NoFrame)
        self.Result_List_Back.setStyleSheet("QLabel{border-image: url(./resource/ListWidget.png)}")
        for button in self.Matrix:
            button.setStyleSheet("QPushButton{border-image: url(./resource/button1.png)}" "QPushButton:hover{font: 16pt Cambria Math Bold}")
        for button in self.Button:
            button.setStyleSheet("QPushButton{border-image: url(./resource/button2.png)}" "QPushButton:hover{font:  Italic}")
        self.select_win.setStyleSheet("QMainWindow{border-image:url(./resource/background.jpg)} ")
        self.MatchNum.setStyleSheet("QComboBox{border-image:url(./resource/Combox.png)}"
                                    "QComboBox::drop-down{border:none}"
                                    "QComboBox::down-arrow{border-image:url(./resource/ComboxDown.png);  width: 14px ;height: 14px;}")
        self.Equal_select.setStyleSheet("QComboBox{border-image:url(./resource/Combox.png)}"
                                    "QComboBox::drop-down{border:none}"
                                    "QComboBox::down-arrow{border-image:url(./resource/ComboxDown.png);  width: 14px ;height: 14px;}")
        self.oper_1.setScaledContents(True)
        self.oper_2.setScaledContents(True)
        self.oper_3.setScaledContents(True)
        self.oper_4.setScaledContents(True)
        #标识符设置
        self.Count = 0 #表示第几个Lcd显示数字
        self.Count2 = 0 #表示第几个运算框显示运算符
        self.Count3 = 0 #表示当前输入运算符or数字
        self.mode = 0 #表示模式 1为题库选择 2为自定义输入 3 为随机生成
        self.operator = '' #第一个运算符
        self.operator2 = ''#第二个运算符
        self.myfile= open('equation_data.txt','a+')#等式库文件
        #矩阵键盘初始化
        self.unable_oprt()
        self.unable_num()
        self.pushButton_Back.setEnabled(False)
        self.Addtodata.setEnabled(False)
        #信号与槽函数设置
        self.ui_select.pushButton_confirm.clicked.connect(self.get_equation)#等式库界面选中等式槽函数
        self.Select.clicked.connect(lambda :self.equation_select(self.select_win))#进入等式库槽函数
        self.User_defined.clicked.connect(self.user_defined)#自定义输入槽函数
        #矩阵键盘槽函数
        self.pushButton_0.clicked.connect(lambda: self.num_input(0))
        self.pushButton_1.clicked.connect(lambda: self.num_input(1))
        self.pushButton_2.clicked.connect(lambda: self.num_input(2))
        self.pushButton_3.clicked.connect(lambda :self.num_input(3))
        self.pushButton_4.clicked.connect(lambda :self.num_input(4))
        self.pushButton_5.clicked.connect(lambda :self.num_input(5))
        self.pushButton_6.clicked.connect(lambda :self.num_input(6))
        self.pushButton_7.clicked.connect(lambda :self.num_input(7))
        self.pushButton_8.clicked.connect(lambda :self.num_input(8))
        self.pushButton_9.clicked.connect(lambda :self.num_input(9))
        self.pushButton_Add.clicked.connect(self.oper_add)
        self.pushButton_Minus.clicked.connect(self.oper_minus)
        self.pushButton_Mul.clicked.connect(self.oper_mul)
        self.pushButton_Equal.clicked.connect(self.oper_equal)
        self.pushButton_Back.clicked.connect(self.back_forward)
        self.pushButton_AC.clicked.connect(self.user_defined)
        self.Cal.clicked.connect(self.find_result) #求解槽函数
        self.RandomEqu.clicked.connect(self.equation_generate)#随机等式槽函数
        self.Addtodata.clicked.connect(self.addtodata)#添加至等式库
        self.Result_List.itemClicked.connect(self.showresult)#显示对应结果

    #进入等式库
    def equation_select(self, MyWindow):
        self.clear_all()
        self.Addtodata.setEnabled(False)
        self.clear_all()
        self.ui_select.myfile = open('equation_data.txt','r+')
        self.ui_select.Equation_List.clear()
        self.ui_select.read_equation()
        MyWindow.show()

    #选择等式库的等式
    def get_equation(self):
        self.mode = 1
        self.Addtodata.setEnabled(False)
        equation_select = Equation()
        equation_select.set_text(self.ui_select.Equation_List.item(self.ui_select.Equation_List.currentRow()).text())
        self.operator = equation_select.operator_1
        for i in range(6):
            if equation_select.num_list[i] == 0 and i%2 ==0:
                self.Lcd_List[i].display('')
            else:
                self.Lcd_List[i].display(equation_select.num_list[i])
        if self.operator == '+':
            self.oper_1.setPixmap(QPixmap(":/Operator/oper/add.png"))
        elif self.operator == '-':
            self.oper_1.setPixmap(QPixmap(":/Operator/oper/minus.png"))
        else:
            self.oper_1.setPixmap(QPixmap(":/Operator/oper/mul.png"))
        self.oper_2.setPixmap(QPixmap(":/Operator/oper/equal.png"))
        self.ui_select.myfile.close()
        self.select_win.close()

    #求解结果
    def find_result(self):
        #还原初始化
        #self.clear_all()
        self.Result_List.clear()
        for lcd in self.Lcd_Result_List:
            lcd.display(0)
        lcd_num = []
        for lcd in self.Lcd_List:
            lcd_num.append(lcd.intValue())
        equation = Equation()
        equation.set_para(lcd_num, self.operator, '=')
        if self.MatchNum.currentIndex() == 0:
            answer_list = self.solution.move_one_match(equation)
        if self.MatchNum.currentIndex() == 1:
            answer_list = self.solution.move_two_match(equation)
        for answer in answer_list:
            item = QListWidgetItem(answer.rstrip())
            item.setTextAlignment(Qt.AlignHCenter)
            self.Result_List.addItem(item)
        #显示解的个数和难度
        if len(answer_list):
            self.label_solvenum_2.setText(str(len(answer_list)))
            self.label_difficulty_2.setText(str(Result_Search.cal_difficulty(equation, self.MatchNum.currentIndex())))
            if self.mode == 2 or self.mode == 3:
                self.Addtodata.setEnabled(True)
        else:
            self.label_solvenum_2.setText('No Answer :(')
            self.label_difficulty_2.setText('Infinnity')

    #显示结果
    def showresult(self):
        equation_select = Equation()
        equation_select.set_text(self.Result_List.item(self.Result_List.currentRow()).text())
        for i in range(6):
            if equation_select.num_list[i] == 0 and i%2 ==0 and self.Lcd_List[i].intValue() == 0:
                self.Lcd_Result_List[i].display('')
            else:
                self.Lcd_Result_List[i].display(equation_select.num_list[i])
        if equation_select.operator_1 == '+':
            self.oper_3.setPixmap(QPixmap(":/Operator/oper/add.png"))
            self.oper_4.setPixmap(QPixmap(":/Operator/oper/equal.png"))
        elif equation_select.operator_1 == '-':
            self.oper_3.setPixmap(QPixmap(":/Operator/oper/minus.png"))
            self.oper_4.setPixmap(QPixmap(":/Operator/oper/equal.png"))
        elif equation_select.operator_1 == '*':
            self.oper_3.setPixmap(QPixmap(":/Operator/oper/mul.png"))
            self.oper_4.setPixmap(QPixmap(":/Operator/oper/equal.png"))
        else:
            self.oper_3.setPixmap(QPixmap(":/Operator/oper/equal.png"))
            if equation_select.operator_2 == '+':
                self.oper_4.setPixmap(QPixmap(":/Operator/oper/add.png"))
            elif equation_select.operator_2 == '-':
                self.oper_4.setPixmap(QPixmap(":/Operator/oper/minus.png"))

    #随机出题
    def equation_generate(self):
        self.Addtodata.setEnabled(False)
        self.clear_all()
        self.mode = 3
        random_equation = Result_Search.equation_generate(self.MatchNum.currentIndex(), self.Equal_select.currentIndex())
        for i in range(6):
            if random_equation.num_list[i] == 0 and i%2 ==0:
                self.Lcd_List[i].display('')
            else:
                self.Lcd_List[i].display(random_equation.num_list[i])
        self.operator = random_equation.operator_1
        if random_equation.operator_1 == '+':
            self.oper_1.setPixmap(QPixmap(":/Operator/oper/add.png"))
        elif random_equation.operator_1 == '-':
            self.oper_1.setPixmap(QPixmap(":/Operator/oper/minus.png"))
        else:
            self.oper_1.setPixmap(QPixmap(":/Operator/oper/mul.png"))
        self.oper_2.setPixmap(QPixmap(":/Operator/oper/equal.png"))

    #添加至等式库
    def addtodata(self):
        self.myfile = open('equation_data.txt', 'a+')  # 等式库文件
        new_equation = Equation()
        lcd_num = []
        for lcd in self.Lcd_List:
            lcd_num.append(lcd.intValue())
        QMessageBox.information(None, '提示', '添加成功', QMessageBox.Yes)
        new_equation.set_para(lcd_num, self.operator, '=')
        self.myfile.write('\n'+ new_equation.equation_str)
        self.myfile.close()
        self.Addtodata.setEnabled(False)
    #自定义输入
    def user_defined(self):
        self.clear_all()
        self.Addtodata.setEnabled(False)
        self.mode = 2
        self.enable_matrix()
    #数字键able
    def enable_matrix(self):
        self.pushButton_0.setEnabled(True)
        self.pushButton_1.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(True)
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(True)
        self.pushButton_6.setEnabled(True)
        self.pushButton_7.setEnabled(True)
        self.pushButton_8.setEnabled(True)
        self.pushButton_9.setEnabled(True)
        self.pushButton_AC.setEnabled(True)
        self.pushButton_Back.setEnabled(True)
    #运算符unable
    def unable_oprt(self):
        self.pushButton_Add.setEnabled(False)
        self.pushButton_Minus.setEnabled(False)
        self.pushButton_Mul.setEnabled(False)
        self.pushButton_Equal.setEnabled(False)

    #数字键unable
    def unable_num(self):
        self.pushButton_0.setEnabled(False)
        self.pushButton_1.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        self.pushButton_8.setEnabled(False)
        self.pushButton_9.setEnabled(False)
    #运算符able
    def enable_oper(self):
        #应该输入+ - *
        if self.Count == 1 or self.Count == 2:
            self.pushButton_Mul.setEnabled(True)
            self.pushButton_Add.setEnabled(True)
            self.pushButton_Minus.setEnabled(True)
            self.pushButton_Equal.setEnabled(False)
        #应该输入等于号
        elif self.Count == 3 or self.Count == 4:
            self.pushButton_Equal.setEnabled(True)
            self.pushButton_Mul.setEnabled(False)
            self.pushButton_Add.setEnabled(False)
            self.pushButton_Minus.setEnabled(False)

    #数字键槽函数
    def num_input(self, num):
        self.Count3 = 0
        if self.Count % 2 == 1:
            self.Lcd_List[self.Count - 1].display(self.Lcd_List[self.Count].intValue())
            self.Lcd_List[self.Count].display(num)
        else:
            self.Lcd_List[self.Count + 1].display(num)
            self.Lcd_List[self.Count].display('')
        self.Count = self.Count + 1
        #应该输入运算符
        if self.Count == 2 or self.Count == 4 or self.Count == 6:
            self.unable_num()
        if self.Count <= 4:
            self.enable_oper()

    #运算符槽函数
    def oper_add(self):
        if self.Count % 2 == 1:
            self.Count = self.Count + 1
        self.Count3 = 1
        self.operator = '+'
        self.oper_1.setPixmap(QPixmap(":/Operator/oper/add.png"))
        self.Count2 = 1
        self.enable_matrix()
        self.unable_oprt()

    def oper_minus(self):
        #self.move_num()
        if self.Count % 2 == 1:
            self.Count = self.Count + 1
        self.Count3 = 1
        self.operator = '-'
        self.oper_1.setPixmap(QPixmap(":/Operator/oper/minus.png"))
        self.Count2 = 1
        self.enable_matrix()
        self.unable_oprt()

    def oper_mul(self):
        #self.move_num()
        if self.Count % 2 == 1:
            self.Count = self.Count + 1
        self.Count3 = 1
        self.operator = '*'
        self.oper_1.setPixmap(QPixmap(":/Operator/oper/mul.png"))
        self.Count2 = 1
        self.enable_matrix()
        self.unable_oprt()

    def oper_equal(self):
        if self.Count % 2 == 1:
            self.Count = self.Count + 1
        self.Count3 = 1
        self.oper_2.setPixmap(QPixmap(":/Operator/oper/equal.png"))
        self.Count2 = 0
        self.enable_matrix()
        self.unable_oprt()

    #回退键槽函数
    def back_forward(self):
        if self.Count ==0:#如果在第0位 返回无效
            return
        if self.Count3 == 0:#当前输入为数字
            self.Lcd_List[self.Count].display(0)
            self.Count = self.Count -1
            self.Lcd_List[self.Count].display(0)
            self.enable_matrix()
            if self.Count == 2 or self.Count == 4:#之前一个输入为运算符
                self.Count3 =1
        else:
            if self.Count2 == 1:
                self.oper_1.clear()
                self.Count2 = 0
            else:
                self.oper_2.clear()
                self.Count2 = 1
            self.enable_oper()
            self.Count3 = 0

    #清除所有
    def clear_all(self):
        for lcd_num in self.Lcd_List:
            lcd_num.display(0)
        for lcd_num in self.Lcd_Result_List:
            lcd_num.display(0)
        self.oper_1.clear()
        self.oper_2.clear()
        self.oper_3.clear()
        self.oper_4.clear()
        self.Result_List.clear()
        self.unable_num()
        self.unable_oprt()
        self.pushButton_Back.setEnabled(False)
        self.label_difficulty_2.clear()
        self.label_solvenum_2.clear()
        self.Count = 0
        self.Count2 = 0
        self.Count3 = 0



