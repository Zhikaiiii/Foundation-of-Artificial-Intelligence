import sys,os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from Interface import Interface
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Interface(MainWindow)
    qss_style = '#MainWindow{border-image:url(./resource/background.jpg);}'
    MainWindow.setStyleSheet(qss_style)
    MainWindow.show()
    sys.exit(app.exec_())