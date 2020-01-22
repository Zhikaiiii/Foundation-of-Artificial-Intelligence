from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Equation_Select import Ui_Equation_Select as Ui_Select
#等式库的界面
#继承自Ui_Select

class Interface_Equation(Ui_Select):
    def __init__(self, MyWindow):
        super(Interface_Equation, self).__init__()
        self.setupUi(MyWindow)
        #样式设定
        self.Equation_List.setStyleSheet("QListWidget{background-color:transparent}" "QListWidget::item::selected{background-color:transparent}")
        self.Equation_List.setFocusPolicy(Qt.NoFocus)
        self.Equation_List.setFrameShape(QFrame.NoFrame)
        self.pushButton_confirm.setStyleSheet("QPushButton{border-image: url(./resource/button2.png)}" "QPushButton:hover{font:  Italic}")
        self.Equation_List_Back.setStyleSheet("QLabel{border-image: url(./resource/ListWidget.png)}")
        self.Equation_List.itemClicked.connect(self.show_selection) # 点击显示对应结果
        self.myfile= open('equation_data.txt','r+')

    #读入文件中的等式
    def read_equation(self):
        equation_list = self.myfile.readlines()
        for equation_exist in equation_list:
            item = QListWidgetItem(equation_exist.rstrip())
            item.setTextAlignment(Qt.AlignHCenter)
            self.Equation_List.addItem(item)

    #显示选中的等式
    def show_selection(self):
        self.equation_select.setText(self.Equation_List.item(self.Equation_List.currentRow()).text())

