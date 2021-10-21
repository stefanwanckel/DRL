import numpy as np
import cv2
import glob

CALIB_IMG_PATH = '/home/stefan/Documents/Masterarbeit/DRL/hindsight-experience-replay-ur5/CV/Camera/extrinsic_calibration/corner_detection_images/'
MARKUP_IMG_PATH = '/home/stefan/Documents/Masterarbeit/DRL/hindsight-experience-replay-ur5/CV/Camera/extrinsic_calibration/corner_detection_images_markup'
CAMERA_INFO_PATH = '/home/stefan/Documents/Masterarbeit/DRL/hindsight-experience-replay-ur5/CV/Camera/camera_info/'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7, 3), np.float32)
objp[:, :2] = 20*np.mgrid[0:7, 0:9].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob(CALIB_IMG_PATH + '*.jpg')


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7, 9), corners2, ret)
        cv2.imshow('img', img)

        new_name = MARKUP_IMG_PATH + fname.split('/')[-1]
        cv2.imwrite(new_name, img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

np.save(CAMERA_INFO_PATH+'ret', ret)
np.save(CAMERA_INFO_PATH+'mtx', mtx)
np.save(CAMERA_INFO_PATH+'dist', dist)
np.save(CAMERA_INFO_PATH+'rvecs', rvecs)
np.save(CAMERA_INFO_PATH+'tvecs', tvecs)
