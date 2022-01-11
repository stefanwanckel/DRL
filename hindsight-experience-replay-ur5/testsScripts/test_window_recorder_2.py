
import cv2
import numpy as np
import pyautogui
import os

output_path = os.path.join("test_images")

# display screen resolution, get it using pyautogui itself
SCREEN_SIZE = tuple(pyautogui.size())
# define the codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")
# frames per second
fps = 15
# create the video write object
#top has to be cropped
out = cv2.VideoWriter("output.avi", fourcc, fps, SCREEN_SIZE)
# the time you want to record in seconds
record_seconds = 1

for i in range(int(record_seconds * fps)):
    # make a screenshot
    #img = pyautogui.screenshot()
    # convert these pixels to a proper numpy array to work with OpenCV
    frame = np.array(img)
    frame = frame[24:,:,:]
    # convert colors from BGR to RGB    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # write the frame
    out.write(frame)
    # show the frame
    #cv2.imshow("screenshot", frame)
    cv2.imwrite(os.path.join(output_path,f"test_image_{str(i)}.png"), frame)
    # if the user clicks q, it exits
    if cv2.waitKey(1) == ord("q"):
        break

# make sure everything is closed when exited
cv2.destroyAllWindows()
out.release()

def recorder():
    output_path = os.path.join("testScripts","test_images")

    # display screen resolution, get it using pyautogui itself
    SCREEN_SIZE = tuple(pyautogui.size())
    # define the codec
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # frames per second
    fps = 15
    # create the video write object
    #top has to be cropped
    out = cv2.VideoWriter("output.avi", fourcc, fps, SCREEN_SIZE)
    # the time you want to record in seconds
    record_seconds = 1

    while True:
        # make a screenshot
        img = pyautogui.screenshot()
        # convert these pixels to a proper numpy array to work with OpenCV
        frame = np.array(img)
        frame = frame[24:,:,:]
        # convert colors from BGR to RGB    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # write the frame
        out.write(frame)
        # show the frame
        #cv2.imshow("screenshot", frame)
        cv2.imwrite(os.path.join(output_path,f"test_image_{str(i)}.png"), frame)
        # if the user clicks q, it exits
        if cv2.waitKey(1) == ord("q"):
            break

    # make sure everything is closed when exited
        cv2.destroyAllWindows()
        out.release()