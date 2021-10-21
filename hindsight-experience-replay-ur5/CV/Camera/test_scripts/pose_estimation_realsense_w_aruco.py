
import sys
import cv2
import pyrealsense2 as rs
import numpy as np
from utils import *


CAMERA_INFO_PATH = "../camera_info/"
# load camera params
rvecs, tvecs, mtx, dist, ret = read_camera_params()
# start pipline
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipe.start(config)
# set aruco dict type
dict_type = "DICT_5X5_100"
aruco_marker_type = ARUCO_DICT[dict_type]
warmup_counter = 0
try:
    while True:
        warmup_counter += 1
        frames = pipe.wait_for_frames()
        if warmup_counter > 50:
            align = rs.align(rs.stream.color)
            aligned = align.process(frames)
            c_frames = aligned.get_color_frame()
            img = np.asanyarray(c_frames.get_data())
            img = aruco_pose_estimation(img, aruco_dict_type=aruco_marker_type,
                                        matrix_coefficients=mtx, distortion_coefficients=dist)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            # cv2.imwrite("calibration_image_{}".format(str(counter-50)))
finally:
    pipe.stop()
