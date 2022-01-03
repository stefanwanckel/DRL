
import sys
import cv2
import pyrealsense2 as rs
import numpy as np
import pickle
import os
sys.path.append(os.path.abspath('../test_scripts/'))

from utils import *

np.set_printoptions(precision=3, suppress=True)

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
        if warmup_counter > 10:
            align = rs.align(rs.stream.color)
            aligned = align.process(frames)
            c_frames = aligned.get_color_frame()
            img = np.asanyarray(c_frames.get_data())
            img, marker_Transformations, rvec, tvec = aruco_pose_estimation(img, aruco_dict_type=aruco_marker_type,
                                                                            matrix_coefficients=mtx, distortion_coefficients=dist)
            cv2.imshow('img', img)

            for ID in marker_Transformations:
                print("[INFO] T-Matrix of markerID: {}".format(ID))
                print("*"*20)
                print(marker_Transformations[ID])
                print(rvec)
                print("*"*20)
            res = cv2.waitKey(0)
            if res % 256 == 32:
                pickle.dump(marker_Transformations, open(
                    "marker_transformation.p", "wb"))

finally:
    pipe.stop()
