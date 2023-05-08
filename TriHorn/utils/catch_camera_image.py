#-*-coding:utf-8-*-
import cv2
import threading
import numpy as np
from queue import Queue


class Camera(threading.Thread):
    def __init__(self, con):
        self.con = con
        self.cap = cv2.VideoCapture(0)  # 视频进行读取操作以及调用摄像头
        width = 320  # 宽度
        self.cap.set(3, width)
        height = 240  # 高度
        self.cap.set(4, height)
        self.frames = Queue()
        super().__init__()

    def run(self):
        self.con.acquire()
        print("执行摄像头调用")
        while self.cap.isOpened():  # 判断视频读取或者摄像头调用是否成功，成功则返回true。
            ret, frame = self.cap.read()  # 返回给ret的是摄像头调用是否成功的结果，返回给frame的为获取到的视频
            if ret is True:
                # cv2.imshow('frame', frame)
                if not self.frames.empty():
                    self.frames.get()
                self.frames.put(frame)
            self.con.notify()
            self.con.wait()
        self.con.release()

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def catch_camera_img_data(self):
        img = self.frames.get()
        return self.rgb2gray(img)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
