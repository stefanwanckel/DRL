import pyrealsense2 as rs
import numpy as np
import cv2
import sys

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipe.start(config)


try:
    for i in range(0, 100):
        frames = pipe.wait_for_frames()
        print(sys.getsizeof(frames))
        for f in frames:
            print(f.profile)
            print(f)
        align = rs.align(rs.stream.color)
        aligned = align.process(frames)
        c_frames = aligned.get_color_frame()
        img = np.asanyarray(c_frames.get_data())
        cv2.imshow('img', img)
finally:
    pipe.stop()


# align = rs.align(rs.stream.color)
# aligned = align.process(frames)
# c_frames = aligned.get_color_frame()
# img = np.asanyarray(c_frames.get_data())
# cv2.imshow('img', img)
cv2.waitKey(0)
