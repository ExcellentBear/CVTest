from PyQt5.Qt import QApplication
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from pyqt_uis.sample_ui import MyWindow


if __name__ == '__main__':
    src = [
        "rtmp://media3.sinovision.net:1935/live/livestream",
        "http://ivi.bupt.edu.cn/hls/cctv6hd.m3u8",  # 或者ip摄像头地址
    ]
    q = QApplication([])
    w = MyWindow(src, None)
    w.show()
    q.exec()
