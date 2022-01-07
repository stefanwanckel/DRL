import numpy as np
import cv2
import os
import glob


def write_video(file_path, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    #w, h = frames[0].size
    w = 1280
    h = 720
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(pil_to_cv(frame))

    writer.release()


if __name__ == "__main__":

    img_array = []
    task_name = "push"
    success_run = "1"
    image_path = os.path.join(task_name, task_name+"_success_"+success_run)
    print(image_path)
    for filename in glob.glob(image_path+"/*.svg"):
        img = cv2.imread(filename)
        height, widthfilename, layers = img.shape
        size = (width, height)
        img_array.append(img)
    fps = int(1/len(glob.glob(image_path+"/*.svg")))
    out = cv2.VideoWriter(
        'project.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
