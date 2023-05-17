# coding:utf-8
import sys
import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStackedWidget, QHBoxLayout, QLabel, QWidget, QVBoxLayout, QFileDialog, QListWidget, QListWidgetItem

from qfluentwidgets import (NavigationInterface, NavigationItemPosition, NavigationWidget, MessageBox, PrimaryPushButton, InfoBarPosition, 
                            isDarkTheme, setTheme, Theme, setThemeColor, NavigationToolButton, NavigationPanel, SwitchButton, InfoBar)
from qfluentwidgets import FluentIcon as FIF
from qframelesswindow import FramelessWindow, StandardTitleBar

import requests
import subprocess
import shutil

mlp_server_url = 'http://172.22.140.199:18924' + '/run'

def send(data):
    try:
        r = requests.post(mlp_server_url, data=data, timeout=0.5)
    except Exception as e:
        print(e)
    return

class Widget(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.filePath = r'../../video/people1'
        self.wsl_filePath = self.filePath.replace(':/', '/')
        self.wsl_filePath = self.wsl_filePath[0].lower() + self.wsl_filePath[1:]
        self.wsl_filePath = '/mnt/' + self.wsl_filePath
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setSpacing(20)
        self.verticalLayout.setContentsMargins(30, 30, 30, 30)
        self.update_button = PrimaryPushButton('刷新', self, FIF.UPDATE)
        self.verticalLayout.addWidget(self.update_button, 0, Qt.AlignRight)
        
        self.listWidget = QListWidget(self)
        self.listWidget.setObjectName(u"listWidget")
        self.verticalLayout.addWidget(self.listWidget)
        self.update_button.clicked.connect(self.refresh)
        self.setObjectName(text.replace(' ', '-'))
        self.setStyleSheet('Demo{background: white} QPushButton{padding: 5px 10px; font:15px "Microsoft YaHei"}')
        
    def error_message(self, title, content = ""):
        InfoBar.error(
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.BOTTOM_RIGHT,
            duration=-1,    # won't disappear automatically
            parent=self
        )

    def success_message(self, title, content = ""):
        # convenient class mothod
        InfoBar.success(
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )

    def get_data(self):
        try:
            name = self.listWidget.currentItem().text()
        except:
            self.error_message("请选择一个文件夹 😆")
        data = {'path' : self.wsl_filePath, 'name' : name}
        return data
    
    def message(self):
        match self.objectName():
            case "预处理":
                self.success_message('正在预处理', '请耐心等待 😆')
                    
            case "模型训练" | "模型展示":
                self.success_message('请稍后 😆')
    
    def refresh(self):
        self.listWidget.clear()
        match self.objectName():
            case "选择文件夹":
                for name in os.listdir(self.filePath):
                    item = QListWidgetItem(name)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.listWidget.addItem(item)
                    
            case "预处理" | "工具箱":
                for name in os.listdir(self.filePath):
                    if name.endswith(".mp4"):
                        item = QListWidgetItem(name)
                        item.setTextAlignment(Qt.AlignCenter)
                        self.listWidget.addItem(item)
                        
            case "模型训练":
                for name in os.listdir(self.filePath):
                    if not name.endswith(".mp4") and not name.endswith("_img"):
                        item = QListWidgetItem(name)
                        item.setTextAlignment(Qt.AlignCenter)
                        self.listWidget.addItem(item)
                        
            case "模型展示":
                for name in os.listdir(self.filePath):
                    if not name.endswith(".mp4") and not name.endswith("_img") and os.path.exists(os.path.join(self.filePath, name, "base.ingp")):
                        item = QListWidgetItem(name)
                        item.setTextAlignment(Qt.AlignCenter)
                        self.listWidget.addItem(item)

class preprocess(Widget):
    def __init__(self, text: str, parent=None):
        super().__init__(text = text, parent=parent)        
        self.hLayout = QHBoxLayout(self)
        self.hLayout.setSpacing(20)
        self.hLayout.setContentsMargins(30, 30, 30, 30)
        self.radioButton = SwitchButton('自适应采样', self)
        self.hLayout.addWidget(self.radioButton, 0, Qt.AlignHCenter)
        self.pushButton_2 = PrimaryPushButton('人体抠图', self)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.hLayout.addWidget(self.pushButton_2, 0, Qt.AlignCenter)
        self.pushButton_3 = PrimaryPushButton('人脸抠图', self)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.hLayout.addWidget(self.pushButton_3, 0, Qt.AlignCenter)
        self.pushButton_4 = PrimaryPushButton('计算位姿', self)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.hLayout.addWidget(self.pushButton_4, 0, Qt.AlignCenter)
        self.verticalLayout.addLayout(self.hLayout)
        self.pushButton_2.clicked.connect(self.matting)
        self.pushButton_3.clicked.connect(self.parsing)
        self.pushButton_4.clicked.connect(self.calculate)
        
    def matting(self):
        data = self.get_data()
        data['type'] = 'matting'
        data['train_flag'] = self.radioButton.isChecked()
        self.message()
        send(data)

    def parsing(self):
        data = self.get_data()
        data['type'] = 'parsing'
        data['train_flag'] = self.radioButton.isChecked()
        self.message()
        send(data)
            
    def calculate(self):
        data = self.get_data()
        data['type'] = 'calculate'
        data['train_flag'] = self.radioButton.isChecked()
        self.message()
        name = data['name']
        output_path = os.path.join(self.filePath, name[:-4])
        print("请等待colmap计算相机姿态，这通常需要几分钟时间")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        subprocess.call(f"python ./colmap2nerf.py --video_in {os.path.join(self.filePath, name)} --video_fps 3 --run_colmap --aabb_scale 16 --images {os.path.join(output_path, 'images')} \
                    --out {os.path.join(output_path, 'transforms.json')} --colmap_db {os.path.join(output_path, 'colmap.db')} --text {os.path.join(output_path, 'colmap_text')}", shell=True)
        print("预处理完成，可进行神经辐射场训练")

class train(Widget):
    def __init__(self, text: str, parent=None):
        super().__init__(text = text, parent=parent)  
        self.pushButton_5 = PrimaryPushButton('训练模型', self)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.verticalLayout.addWidget(self.pushButton_5)
        self.pushButton_5.clicked.connect(self.training)
        
    def training(self):
        self.message()
        os.system(r".\instant-ngp\instant-ngp.exe -m nerf -s " + os.path.join(self.filePath, self.listWidget.currentItem().text()))

class show(Widget):
    def __init__(self, text: str, parent=None):
        super().__init__(text = text, parent=parent)  
        self.pushButton_6 = PrimaryPushButton('展示模型', self)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.verticalLayout.addWidget(self.pushButton_6)
        self.pushButton_6.clicked.connect(self.render)

    def render(self):
        self.message()
        path = os.path.join(self.filePath, self.listWidget.currentItem().text(), "base.ingp")
        os.system(r".\instant-ngp\instant-ngp.exe --load_snapshot " + path)

class util(Widget):
    def __init__(self, text: str, parent=None):
        super().__init__(text = text, parent=parent)  
        self.hLayout = QHBoxLayout(self)
        self.hLayout.setSpacing(20)
        self.hLayout.setContentsMargins(30, 30, 30, 30)
        self.radioButton = SwitchButton('自适应采样', self)
        self.hLayout.addWidget(self.radioButton, 0, Qt.AlignHCenter)
        self.pushButton_7 = PrimaryPushButton('切分视频', self)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.hLayout.addWidget(self.pushButton_7, 0, Qt.AlignCenter)
        self.verticalLayout.addLayout(self.hLayout)
        self.pushButton_7.clicked.connect(self.split)

    def split(self):
        import cv2
        path = os.path.join(self.filePath, self.get_data()['name'])
        self.success_message("正在切分视频", "请稍后")
        cap = cv2.VideoCapture(path)
        new_path = path.split(".")[0] + "_img"
        fps = int(cap.get(5))
        frame_count = int(cap.get(7))
        if self.radioButton.isChecked():
            step = 50 if fps <= 30 else 100
            step = int(step / 2) if frame_count < 1300 else step
            step = 1 if path.__contains__("train") else step
        else:
            step = 1
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            
        for idx in range(0, frame_count, step):
            if step != 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            flag, img = cap.read()
            if flag:
                cv2.imwrite(os.path.join(new_path, str(idx*2 if frame_count < 1300 and frame_count > 200 else idx) + '.jpg'), img)
                
        self.success_message("切分完毕！")

class NavigationBar(QWidget):
    """ Navigation widget """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.hBoxLayout = QHBoxLayout(self)
        self.menuButton = NavigationToolButton(FIF.MENU, self)
        self.navigationPanel = NavigationPanel(parent, True)
        self.titleLabel = QLabel(self)

        self.navigationPanel.move(0, 31)
        self.hBoxLayout.setContentsMargins(5, 5, 5, 5)
        self.hBoxLayout.addWidget(self.menuButton)
        self.hBoxLayout.addWidget(self.titleLabel)

        self.menuButton.clicked.connect(self.showNavigationPanel)
        self.navigationPanel.setExpandWidth(260)
        self.navigationPanel.setMenuButtonVisible(True)
        self.navigationPanel.hide()

    def setTitle(self, title: str):
        self.titleLabel.setText(title)
        self.titleLabel.adjustSize()

    def showNavigationPanel(self):
        self.navigationPanel.show()
        self.navigationPanel.raise_()
        self.navigationPanel.expand()

    def addItem(self, routeKey, icon, text: str, onClick, selectable=True, position=NavigationItemPosition.TOP):
        def wrapper():
            onClick()
            self.setTitle(text)

        self.navigationPanel.addItem(
            routeKey, icon, text, wrapper, selectable, position)

    def addSeparator(self, position=NavigationItemPosition.TOP):
        self.navigationPanel.addSeparator(position)

    def setCurrentItem(self, routeKey: str):
        self.navigationPanel.setCurrentItem(routeKey)
        self.setTitle(self.navigationPanel.items[routeKey]._text)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.navigationPanel.resize(self.navigationPanel.width(), self.window().height() - 31)


class Window(FramelessWindow):

    def __init__(self):
        super().__init__()
        self.setTitleBar(StandardTitleBar(self))
        self.filePath = r'../../video/people1'
        # use dark theme mode
        setTheme(Theme.LIGHT)

        # change the theme color
        # setThemeColor('#0078d4')

        self.vBoxLayout = QVBoxLayout(self)
        self.navigationInterface = NavigationBar(self)
        self.stackWidget = QStackedWidget(self)

        # create sub interface
        self.select_widge = Widget('选择文件夹', self)
        self.preprocess_widge = preprocess('预处理', self)
        self.train_widge = train('模型训练', self)
        self.show_widge = show('模型展示', self)
        self.util_widge = util("工具箱", self)

        self.stackWidget.addWidget(self.select_widge)
        self.stackWidget.addWidget(self.preprocess_widge)
        self.stackWidget.addWidget(self.train_widge)
        self.stackWidget.addWidget(self.show_widge)
        self.stackWidget.addWidget(self.util_widge)

        # initialize layout
        self.initLayout()

        # add items to navigation interface
        self.initNavigation()

        self.initWindow()

    def initLayout(self):
        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, self.titleBar.height(), 0, 0)
        self.vBoxLayout.addWidget(self.navigationInterface)
        self.vBoxLayout.addWidget(self.stackWidget)
        self.vBoxLayout.setStretchFactor(self.stackWidget, 1)

    def initNavigation(self):
        self.navigationInterface.addItem(
            routeKey=self.select_widge.objectName(),
            icon=FIF.SEARCH,
            text='选择文件夹',
            onClick=lambda: self.switchTo(self.select_widge)
        )
        self.navigationInterface.addItem(
            routeKey=self.preprocess_widge.objectName(),
            icon=FIF.MUSIC,
            text='预处理',
            onClick=lambda: self.switchTo(self.preprocess_widge)
        )
        self.navigationInterface.addItem(
            routeKey=self.train_widge.objectName(),
            icon=FIF.VIDEO,
            text='模型训练',
            onClick=lambda: self.switchTo(self.train_widge)
        )

        self.navigationInterface.addSeparator()

        # add navigation items to scroll area
        self.navigationInterface.addItem(
            routeKey=self.show_widge.objectName(),
            icon=FIF.FOLDER,
            text='模型展示',
            onClick=lambda: self.switchTo(self.show_widge),
            position=NavigationItemPosition.SCROLL
        )
        
        self.navigationInterface.addItem(
            routeKey=self.util_widge.objectName(),
            icon=FIF.FOLDER,
            text='工具箱',
            onClick=lambda: self.switchTo(self.util_widge),
            position=NavigationItemPosition.SCROLL
        )

        self.stackWidget.currentChanged.connect(self.onCurrentInterfaceChanged)
        self.preprocess_widge.refresh()
        self.stackWidget.setCurrentIndex(1)

    def initWindow(self):
        self.resize(1280, 720)
        self.setWindowIcon(QIcon('resource/logo.png'))
        self.setWindowTitle('基于多分辨率哈希编码神经辐射场的三维人脸重建系统')
        self.titleBar.setAttribute(Qt.WA_StyledBackground)

        desktop = QApplication.screens()[0].availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)

        self.setQss()

    def setQss(self):
        color = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/{color}/demo.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def switchTo(self, widget):
        if widget.objectName() == "选择文件夹":
            self.filePath = QFileDialog.getExistingDirectory(
                self,  # 父窗口对象
                "选择路径",  # 标题
                r'../../video'  # 起始目录
                )
            if not self.filePath:
                self.filePath = widget.filePath
                return

        widget.filePath = self.filePath
        widget.wsl_filePath = self.filePath.replace(':/', '/')
        widget.wsl_filePath = widget.wsl_filePath[0].lower() + widget.wsl_filePath[1:]
        widget.wsl_filePath = '/mnt/' + widget.wsl_filePath
        widget.filePath = self.filePath
        widget.refresh()
        self.stackWidget.setCurrentWidget(widget)

    def onCurrentInterfaceChanged(self, index):
        widget = self.stackWidget.widget(index)
        self.navigationInterface.setCurrentItem(widget.objectName())

    def showMessageBox(self):
        w = MessageBox(
            'This is a help message',
            'You clicked a customized navigation widget. You can add more custom widgets by calling `NavigationInterface.addWidget()` 😉',
            self
        )
        w.exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    app.exec()
