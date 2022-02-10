import cv2
import os
import sys
import numpy as np
import pyautogui
import time
sys.path.append('/home/stefan/Documents/Masterarbeit/DRL/hindsight-experience-replay-ur5/utils')


class VideoRecorder():
    def __init__(self, filename):
        self.output_path = os.path.join("Results", "simulation","reach")
        self.counter=0
        self.filename=filename
        self.fps = 15
        screen_size = pyautogui.size()
        self.y_offset = 48
        self.capture_size = tuple((screen_size[0],screen_size[1]-self.y_offset))
        self.setup_writer()

    def setup_writer(self):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(os.path.join(self.output_path, self.filename+".avi"), fourcc, self.fps, self.capture_size)
        
    
    def capture_frame(self):
        curr_img = pyautogui.screenshot()
        curr_img = np.array(curr_img)
        curr_img = curr_img[self.y_offset:,:,:]
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.output_path,"reach_"+str(self.counter)+".png"), curr_img)
        self.counter+=1

        self.writer.write(curr_img)

    def stop_recording(self):
        self.writer.release()

if __name__ == "__main__":
    VR = VideoRecorder()
    for i in range(0,100):
        VR.capture_frame()

    VR.stop_recording()
