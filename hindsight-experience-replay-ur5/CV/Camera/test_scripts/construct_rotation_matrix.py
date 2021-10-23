"""
1. get 3 images
2. detect aruco markers (2) and retrieve rvecs and tvecs
3. Build transformation matrices and invert them
4. get points in Aruco frame (retrieve t)
5. define get_custom_T_matrix()
"""

import sys
import cv2
import pyrealsense2 as rs
import numpy as np
from utils import *

# load intrinsic camera params
# load camera params
_, _, mtx, dist, ret = read_camera_params()

# start pipline
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipe.start(config)
# 1
# set aruco dict type
dict_type = "DICT_5X5_100"
aruco_marker_type = ARUCO_DICT[dict_type]
warmup_counter = 0
images_counter = 0
images_needed = 10
rvec = None
tvec = None
images = []
lst_transformations = []

try:
    while True:
        warmup_counter += 1
        frames = pipe.wait_for_frames()
        if warmup_counter > 20:

            align = rs.align(rs.stream.color)
            aligned = align.process(frames)
            c_frames = aligned.get_color_frame()
            img = np.asanyarray(c_frames.get_data())
            img_drawn, marker_Transformations, rvec, tvec = aruco_pose_estimation(
                img, aruco_dict_type=aruco_marker_type, matrix_coefficients=mtx, distortion_coefficients=dist)

            cv2.imshow('curr_img', img_drawn)

            res = cv2.waitKey(0)

            if res % 256 == 32:
                images_counter += 1
                print(f"[INFO] aquired image {images_counter}/{images_needed}")
                images.append(img)
                cv2.imwrite('drawn_img_'+str(images_counter)+'.png', img_drawn)
                lst_transformations.append(marker_Transformations)
                if images_counter == images_needed:
                    break

finally:
    pipe.stop()

# 3 only invert

lst_transformations_CF1_inv = []
lst_transformations_CF2_inv = []

for T_dict in lst_transformations:
    lst_transformations_CF1_inv.append(np.linalg.inv(T_dict[1]))
    lst_transformations_CF2_inv.append(np.linalg.inv(T_dict[2]))


# 4 get points
list_points_1 = []
list_points_2 = []
for T in lst_transformations_CF1_inv:
    list_points_1.append(T[:, 3])
for T in lst_transformations_CF2_inv:
    list_points_2.append(T[:, 3])

# 5
T_cf1_cf2 = get_custom_T_matrix(list_points_1, list_points_2)
np.set_printoptions(precision=2)
print("*"*20)
print("[INFO] T matrix from cf1 to cf2:")
print(T_cf1_cf2)
print("*"*20)
