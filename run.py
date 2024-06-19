from ui.window import Ui_MainWindow
import sys
import subprocess
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6 import QtWidgets
from PySide6.QtCore import QTimer, QThreadPool, Signal, QObject
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from face_detect_task import FaceDetectionTask, FaceRegisterTask, reset_face_registery, reset_record
from face_recognition_modules.common import RecognizeRecord
from PySide6.QtCore import Slot

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow, QObject):
    signal = Signal(QPixmap)
    text_signal = Signal(str)
    record_signal = Signal(RecognizeRecord)
    face_recognized_signal = Signal(bool)  # 新增信号

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 创建定时器
        self.timer = QTimer(self)
        self.is_playing = False
        # 创建信号
        self.signal.connect(self.update_frame)
        self.text_signal.connect(self.update_register_progress)
        self.record_signal.connect(self.update_record)
        self.face_recognized_signal.connect(self.handle_face_recognized)  # 连接信号
        # 创建线程池
        self.threadpool = QThreadPool()

        # 初始化变量
        self.is_playing = False

        # 清空注册库和记录库的选择
        self.check_clear_databases()

        # 手动指定注册文件夹路径并启动注册任务
        register_folder_path = "./pic_register"  # 替换为实际路径
        self.start_register_task(register_folder_path)

        # 直接打开本地摄像头
        self.open_usb_camera()

    def check_clear_databases(self):
        #choice = input("是否清空注册库和记录库? (Y/N): ").strip().upper()
        #if choice == 'Y':
        reset_face_registery()
        reset_record()
        #print("数据库和记录库已重置!")

    def start_register_task(self, folderpath):
        task = FaceRegisterTask(folderpath, self.text_signal)
        self.threadpool.start(task)

    def open_usb_camera(self):
        self.stop_playing()
        self.start_playing()

    def start_playing(self):
        if self.is_playing:
            self.stop_playing()
        self.task = FaceDetectionTask(self.signal, self.record_signal)
        self.threadpool.start(self.task)
        self.is_playing = True

    def stop_playing(self):
        if not self.is_playing:
            return
        self.task.stop()
        self.is_playing = False

    @Slot(QPixmap)
    def update_frame(self, pixmap):
        # 检查是否有新的图像
        if pixmap is not None:
            # 将QPixmap设置为标签的图像
            self.player.setPixmap(pixmap)

    @Slot(str)
    def update_register_progress(self, text):
        print(text)

    @Slot(RecognizeRecord)
    def update_record(self, record: RecognizeRecord):
        if record.person_name != 'Unknown':
            self.face_recognized_signal.emit(True)
        #print(f"{record.person_name} 在 {record.time} 被识别到, 已经识别 {record.count} 次")

    @Slot(bool)
    def handle_face_recognized(self, recognized):
        if recognized:
            print("已识别到注册人脸")
            global face_recognized
            face_recognized = True

if __name__ == "__main__":
    face_recognized = False
    app = QApplication(sys.argv)
    print("请开始身份验证... \n")
    window = MainWindow()
    window.show()

    # 检查是否识别到注册人脸
    while not face_recognized:
        app.processEvents()

    print("检测到注册人脸，身份验证通过! \n")
    subprocess.Popen([sys.executable, './loop.py'])
    #window.close()
    sys.exit(0)
