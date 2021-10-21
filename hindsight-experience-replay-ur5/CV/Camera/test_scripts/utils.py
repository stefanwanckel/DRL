import numpy as np
import os
import cv2

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def read_camera_params():
    PARAMS_PATH = "/home/stefan/Documents/Masterarbeit/DRL/hindsight-experience-replay-ur5/CV/Camera/camera_info"
    rvecs = np.load(os.path.join(PARAMS_PATH, "rvecs.npy"))
    tvecs = np.load(os.path.join(PARAMS_PATH, "tvecs.npy"))
    mtx = np.load(os.path.join(PARAMS_PATH, "mtx.npy"))
    dist = np.load(os.path.join(PARAMS_PATH, "dist.npy"))
    ret = np.load(os.path.join(PARAMS_PATH, "ret.npy"))

    return rvecs, tvecs, mtx, dist, ret


def aruco_pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)
    marker_Transformations = {}
    if len(corners) > 0:
        for ID in list(ids):
            marker_Transformations[int(ID)] = []
    # If markers are detected
    if len(corners) > 0:
        for i, ID in enumerate(ids):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.1, matrix_coefficients,
                                                                           distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients,
                               distortion_coefficients, rvec, tvec, 0.05)
            marker_Transformations[int(
                ID)] = calculate_rotation_matrix(rvec, tvec)
    return frame, marker_Transformations


def calculate_rotation_matrix(rvec, tvec):
    """
    INPUT
    tvec:   translation vector. Offset to the marker from the camera position. 
            Already scaled to same unit used for the marker size.
    rvec:   rotation vector. 
            In angle-axis form:
                -direction of the vector is rotation axis,
                -magniture of vector is rotation magnitude
            Will be converted with Rodrigues formula to rotation matrix R.
            R⁻¹ * Marker_pose = Camera Pose
    OUTPUT
    T:      4x4 3D affine matrix containing the rotation and translation
    """
    R, _ = cv2.Rodrigues(rvec)
    R_inv = np.linalg.inv(R)

    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = tvec
    T[3, 3] = 1
    return T
