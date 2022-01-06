
import pyrealsense2 as rs
import cv2
import os
import sys
import numpy as np
sys.path.append(
    '/home/stefan/Documents/Masterarbeit/DRL/hindsight-experience-replay-ur5/utils')
from real_demo_CV import read_camera_params


class CameraWrapper():
    def __init__(self):
        self.latest_img = None
        self.latest_frame = None
        self.pipe = None
        self._config_cam()
        self.rvecs, self.tvecs, self.mtx, self.dist, self.ret = read_camera_params()

    def _config_cam(self):
        self.pipe = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self.pipe.start(config)

    def warmup_cam(self, warmup=50):

        warmup_counter = 0
        while warmup_counter < warmup:
            warmup_counter += 1
            frames = self.pipe.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned = align.process(frames)
        c_frames = aligned.get_color_frame()
        img = np.asanyarray(c_frames.get_data())
        self.latest_img = img

    def get_latest_img(self):
        frames = self.pipe.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned = align.process(frames)
        c_frames = aligned.get_color_frame()
        img = np.asanyarray(c_frames.get_data())
        self.latest_img = img
        return self.latest_img

    def stop_cam(self):
        self.pipe.stop()


''' for testing
if __name__ == "__main__":
    myCam = CameraWrapper()
    myCam.warmup_cam()

    counter = 0
    while counter < 10:
        img = myCam.get_latest_img()
        cv2.imshow('Live View', img)
        cv2.waitKey(0)
        counter+=1
'''
