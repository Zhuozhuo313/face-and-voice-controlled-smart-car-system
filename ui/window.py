# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.5.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtWidgets import (QFrame, QLabel, QMainWindow,
    QPushButton, QStatusBar, QWidget, QLineEdit, QPlainTextEdit,
    QListWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(640, 360)
        MainWindow.setStyleSheet("""
            QWidget {
                font-family: '汉仪文黑-85W';
                font-size: 8pt;
            }
            QLabel, QLineEdit, QPushButton, QListWidget {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #5cacee;
                color: white;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1e90ff;
            }
            QPlainTextEdit {
                background-color: #ffffff;
                border: 1px solid #ddd;
            }
            QListWidget {
                background-color: #ffffff;
                color: #333;
            }
        """)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.player = QLabel(self.centralwidget)
        self.player.setObjectName(u"player")
        self.player.setGeometry(QRect(0, 0, 640, 360))
        self.player.setFrameShadow(QFrame.Plain)
        self.player.setMidLineWidth(0)
        player_default_stylesheet = """
        background-color: rgb(114, 114, 114);
        border: 1px solid rgb(0, 0, 0);
        """
        self.player.setStyleSheet(player_default_stylesheet)

        
        MainWindow.setCentralWidget(self.centralwidget)
        self.player.raise_()

        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"人脸身份验证", None))
        self.player.setText("")
    # retranslateUi

