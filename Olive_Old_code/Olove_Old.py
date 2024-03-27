import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5 import uic


form_window = uic.loadUiType('./Olove_Old.ui')[0]
second_window = uic.loadUiType('./camerawindow.ui')[0]
class second(QDialog, second_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.background_1.setPixmap(QPixmap("./UI/sea.jpg"))
        self.background_1.setScaledContents(True)

        self.background_2.setPixmap(QPixmap("./UI/uni.jpg"))
        self.background_2.setScaledContents(True)

        self.background_3.setPixmap(QPixmap("./UI/club.jpg"))
        self.background_3.setScaledContents(True)

        self.background_4.setPixmap(QPixmap("./UI/merry.jpg"))
        self.background_4.setScaledContents(True)

        self.makeuped.setPixmap(QPixmap("./imgs/makeuped.jpg"))
        self.makeuped.setScaledContents(True)
class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # -이미지 넣기-
        # 타이틀
        self.Olove_Old.setPixmap(QPixmap("./UI/Olove_Old.png"))
        self.Olove_Old.setScaledContents(True)

        # 메이크업 이미지
        self.make_up_1.setPixmap(QPixmap("./UI/디올 썸머룩.PNG"))
        self.make_up_1.setScaledContents(True)

        self.make_up_2.setPixmap(QPixmap("./UI/안유진.PNG"))
        self.make_up_2.setScaledContents(True)

        self.make_up_3.setPixmap(QPixmap("./UI/남자.png"))
        self.make_up_3.setScaledContents(True)

        self.make_up_4.setPixmap(QPixmap("./UI/군인.PNG"))
        self.make_up_4.setScaledContents(True)

        # 장소 이미지
        self.background_1.setPixmap(QPixmap("./UI/sea.jpg"))
        self.background_1.setScaledContents(True)

        self.background_2.setPixmap(QPixmap("./UI/uni.jpg"))
        self.background_2.setScaledContents(True)

        self.background_3.setPixmap(QPixmap("./UI/club.jpg"))
        self.background_3.setScaledContents(True)

        self.background_4.setPixmap(QPixmap("./UI/merry.jpg"))
        self.background_4.setScaledContents(True)

        # 배경
        self.Olove_Old_background.setPixmap(QPixmap("./UI/cosmetics.jpg"))
        self.Olove_Old_background.setScaledContents(True)

        # make_up_n 라벨에 클릭 이벤트 연결
        self.make_up_1.mousePressEvent = self.on_make_up_1_clicked
        self.make_up_2.mousePressEvent = self.on_make_up_2_clicked
        self.make_up_3.mousePressEvent = self.on_make_up_3_clicked
        self.make_up_4.mousePressEvent = self.on_make_up_4_clicked

        # background 라벨을 클릭할때
        self.background_1.mousePressEvent = self.on_background_1_clicked
        self.background_2.mousePressEvent = self.on_background_2_clicked
        self.background_3.mousePressEvent = self.on_background_3_clicked
        self.background_4.mousePressEvent = self.on_background_4_clicked

    # make_up_n 라벨 클릭 시 실행되는 함수
    def on_make_up_1_clicked(self, event):
        print("make_up_1 라벨이 클릭되었습니다.")

        secondwindow = second()
        secondwindow.exec_()
    def on_make_up_2_clicked(self, event):
        print("make_up_2 라벨이 클릭되었습니다.")
    def on_make_up_3_clicked(self, event):
        print("make_up_3 라벨이 클릭되었습니다.")
    def on_make_up_4_clicked(self, event):
        print("make_up_4 라벨이 클릭되었습니다.")


    # background_n 라벨 클릭 시 실행되는 함수
    def on_background_1_clicked(self, event):
        print("background_1 라벨이 클릭되었습니다.")
    def on_background_2_clicked(self, event):
        print("background_2 라벨이 클릭되었습니다.")
    def on_background_3_clicked(self, event):
        print("background_3 라벨이 클릭되었습니다.")
    def on_background_4_clicked(self, event):
        print("background_4 라벨이 클릭되었습니다.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())