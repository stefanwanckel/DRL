import pyrealsense2 as rs
import numpy as np
import cv2
import sys

CAL_IMAGES_PATH = '/home/stefan/Documents/Masterarbeit/DRL/hindsight-experience-replay-ur5/CV/Camera/extrinsic_calibration/corner_detection_images/'
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipe.start(config)
counter = 0
img_count = 0
img_max = 10

try:
    while True:
        counter += 1

        frames = pipe.wait_for_frames()
        if counter > 50:
            align = rs.align(rs.stream.color)
            aligned = align.process(frames)
            c_frames = aligned.get_color_frame()
            img = np.asanyarray(c_frames.get_data())

            cv2.imshow('img', img)

            res = cv2.waitKey(0)

            if res % 256 == 32:
                filename = CAL_IMAGES_PATH + \
                    'calib2_' + str(img_count+1) + '.jpg'
                cv2.imwrite(filename, img)
                print(f'[INFO] {img_count+1}/{img_max} acquired!')
                img_count += 1

            if img_count == img_max:
                print(f'[INFO] ------ All images acquired!')
                break
            # cv2.imwrite("calibration_image_{}".format(str(counter-50)))
finally:
    pipe.stop()


# align = rs.align(rs.stream.color)
# aligned = align.process(frames)
# c_frames = aligned.get_color_frame()
# img = np.asanyarray(c_frames.get_data())
# cv2.imshow('img', img)
cv2.waitKey(0)
