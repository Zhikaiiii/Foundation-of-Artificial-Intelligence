# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Equation_Select.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Equation_Select(object):
    def setupUi(self, Equation_Select):
        Equation_Select.setObjectName("Equation_Select")
        Equation_Select.resize(339, 415)
        self.centralwidget = QtWidgets.QWidget(Equation_Select)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_confirm = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_confirm.setGeometry(QtCore.QRect(230, 350, 81, 41))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(170, 0, 3))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 0, 3))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        self.pushButton_confirm.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_confirm.setFont(font)
        self.pushButton_confirm.setObjectName("pushButton_confirm")
        self.Equation_List = QtWidgets.QListWidget(self.centralwidget)
        self.Equation_List.setGeometry(QtCore.QRect(80, 50, 191, 191))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        self.Equation_List.setFont(font)
        self.Equation_List.setObjectName("Equation_List")
        self.Equation_List_Back = QtWidgets.QLabel(self.centralwidget)
        self.Equation_List_Back.setGeometry(QtCore.QRect(70, 20, 211, 251))
        self.Equation_List_Back.setText("")
        self.Equation_List_Back.setObjectName("Equation_List_Back")
        self.equation_select = QtWidgets.QLabel(self.centralwidget)
        self.equation_select.setGeometry(QtCore.QRect(90, 320, 161, 21))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.equation_select.setFont(font)
        self.equation_select.setText("")
        self.equation_select.setAlignment(QtCore.Qt.AlignCenter)
        self.equation_select.setObjectName("equation_select")
        self.label_info = QtWidgets.QLabel(self.centralwidget)
        self.label_info.setGeometry(QtCore.QRect(30, 290, 151, 20))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(12)
        self.label_info.setFont(font)
        self.label_info.setObjectName("label_info")
        self.Equation_List_Back.raise_()
        self.pushButton_confirm.raise_()
        self.Equation_List.raise_()
        self.equation_select.raise_()
        self.label_info.raise_()
        Equation_Select.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Equation_Select)
        self.statusbar.setObjectName("statusbar")
        Equation_Select.setStatusBar(self.statusbar)

        self.retranslateUi(Equation_Select)
        QtCore.QMetaObject.connectSlotsByName(Equation_Select)

    def retranslateUi(self, Equation_Select):
        _translate = QtCore.QCoreApplication.translate
        Equation_Select.setWindowTitle(_translate("Equation_Select", "Please Select Equation"))
        self.pushButton_confirm.setText(_translate("Equation_Select", "确定"))
        self.label_info.setText(_translate("Equation_Select", "已选择等式如下）："))
