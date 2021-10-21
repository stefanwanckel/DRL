import numpy as np
import argparse
import time
import cv2
import sys

from utils import ARUCO_DICT, aruco_display
import pyrealsense2 as rs
# set up realsense pipeline
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipe.start(config)
# aruco marker params
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_100"])
arucoParams = cv2.aruco.DetectorParameters_create()
try:
    while True:
        frames = pipe.wait_for_frames()
        if frames is not None:
            # print(sys.getsizeof(frames))
            align = rs.align(rs.stream.color)
            aligned = align.process(frames)

            c_frames = aligned.get_color_frame()
            frame = np.asanyarray(c_frames.get_data())

            corners, ids, rejected = cv2.aruco.detectMarkers(
                frame, arucoDict, parameters=arucoParams)
            # cv2.waitKey(0)
            detected_markers, center_points = aruco_display(
                corners, ids, rejected, frame)
            # print(type(detected_markers))
            # print(detected_markers.shape)
            cv2.imshow("Image", detected_markers)
            cv2.waitKey(1)

finally:
    cv2.destroyAllWindows()
    pipe.stop()


print("dummmy")
