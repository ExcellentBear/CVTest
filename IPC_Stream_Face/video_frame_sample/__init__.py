"""
非阻塞式，视频流的帧采集, 便于不定时采样
"""
import cv2
from threading import Lock, Thread
from typing import Optional
import numpy as np

__all__ = ["MStack", "CaptureThread"]


class MStack:
    def __init__(self, max_size=100):
        self.index = -1
        max_size = max_size if max_size > 1 else 1
        self.deque = [None]*max_size
        self.__max_size = max_size
        self.__mutex = Lock()

    def pop(self)->[Optional[np.ndarray], int]:
        with self.__mutex:
            ret = self.deque[self.index]
        return ret, self.index

    def fetch_last_pic(self, from_index, size=5)->[np.ndarray]:
        ret = []
        with self.__mutex:
            for _ in range(size):
                arr, _ = self.deque[from_index]
                if arr is not None:
                    ret.append(arr)
                from_index = (from_index - 1) % self.__max_size
        return ret

    def push(self, item: np.ndarray):
        with self.__mutex:
            self.index = (self.index + 1) % self.__max_size
            self.deque[self.index] = item


class CaptureThread(Thread):
    def __init__(self, src, frame_buff: MStack):
        super().__init__()
        self.frame_buff = frame_buff
        self.cap = cv2.VideoCapture(src)
        self.signal = False
        self.src = src

    def get(self)->[Optional[np.ndarray], int]:
        return self.frame_buff.pop()

    def set_src(self, src):
        if self.src == src:
            return
        self.src = src
        self.signal = True

    def run(self):
        while True:
                if self.signal:
                    self.signal = False
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.src)
                bl, frame = self.cap.read()
                if bl:
                    self.frame_buff.push(frame)
