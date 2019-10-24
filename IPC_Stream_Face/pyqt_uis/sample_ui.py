# -*- coding:utf-8 -*-
"""
视频流采样检测人脸，并以ui形式展现出来。
采样频率较低，以及计算性能问题，正常1080p处理速度大致2帧/秒
"""
from PyQt5.QtWidgets import QWidget, QMainWindow, QComboBox, QFrame, QVBoxLayout, QLabel
from PyQt5.Qt import QTimer, QImage, QBrush, QPalette
import numpy as np
import cv2
from facenet_wraper import Recon
from video_frame_sample import MStack, CaptureThread
from picture_check import check_each_part


class MyWindow(QMainWindow):
    def __init__(self, ipc_address: [str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("测试用")
        self.ipc_address = ipc_address
        frame = QFrame(self)
        self.setCentralWidget(frame)
        lay = QVBoxLayout(frame)
        frame.setLayout(lay)
        self.com_box = QComboBox(frame)
        self.com_box.addItems(ipc_address)
        self.widget = QWidget(frame)
        self.widget.setAutoFillBackground(True)
        self.label = QLabel(frame)
        lay.addWidget(self.com_box, 1)
        lay.addWidget(self.widget, 10)
        lay.addWidget(self.label, 1)
        remote_address = self.com_box.currentText()
        self.com_box.currentTextChanged.connect(self.item_change)

        self.Recon = Recon()

        self.stack = MStack()
        self.cap_thread = CaptureThread(remote_address, self.stack)
        self.cap_thread.start()

        self.tm = QTimer(self)
        self.tm.timeout.connect(self.get_image_sample)
        self.tm.start(125)

        self.last_frame = None

    def update_image(self, array: np.ndarray):
        im = cv2.transpose(array)
        im = im.transpose((1, 0, 2)).copy()
        image = QImage(im, im.shape[1], im.shape[0], QImage.Format_RGB888)
        palette = self.widget.palette()
        image = image.scaled(self.widget.geometry().width(), self.widget.geometry().height())
        palette.setBrush(QPalette.Background, QBrush(image))
        self.widget.setPalette(palette)
        self.widget.update()

    def item_change(self, text):
        self.cap_thread.set_src(text)

    def get_image_sample(self):
        frame, _ = self.stack.pop()
        if frame is not None:
            frame = frame[:, :, 0:3]
            if self.last_frame is not None:
                x0, y0, x1, y1 = check_each_part(self.last_frame, frame, 1.2)
                self.last_frame = frame.copy()
                if x0 != -1:
                    frame0 = frame[x0:x1, y0:y1]
                    ret = self.Recon.face_check(frame0)  # 识别效率0.5s
                    im = ret["src"]
                    kv = dict(ret["ret"])
                    label = ";".join([k+":%s" % v for k, v in kv.items()])
                    self.label.setText(label)
                    if x1-x0 == frame.shape[0] and y1-y0 == frame.shape[1]:
                        self.update_image(im)
                    else:
                        frame[x0:x1, y0:y1] = im
                        self.update_image(frame)
            else:
                self.last_frame = frame.copy()
                ret = self.Recon.face_check(frame)  # 识别效率0.5s
                im = ret["src"]
                kv = dict(ret["ret"])
                label = ";".join([k + ":%s" % v for k, v in kv.items()])
                self.label.setText(label)
                self.update_image(im)
