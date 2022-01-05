import pyrealsense2 as rs
import numpy as np
import time
import cv2
import sys
import matplotlib.pyplot as plt
import os

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


def get_goal_position(goal_marker_ID=2):

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
    while True:
        warmup_counter += 1
        frames = pipe.wait_for_frames()
        if warmup_counter > 50:
            align = rs.align(rs.stream.color)
            aligned = align.process(frames)
            c_frames = aligned.get_color_frame()
            img = np.asanyarray(c_frames.get_data())
            _, marker_Transformations, rvec, tvec = aruco_pose_estimation(img, aruco_dict_type=aruco_marker_type,
                                                                          matrix_coefficients=mtx, distortion_coefficients=dist)
            if marker_Transformations[int(goal_marker_ID)] is not None:
                pipe.stop()
                marker_pos_robot_frame = map_c_2_r(
                    marker_Transformations[int(goal_marker_ID)][:, 3])
                return marker_pos_robot_frame
                break
            else:
                print("ERROR: goal marker not detected. Aborting")
                sys.exit()


def map_c_2_r(marker_pos_camera_frame):
    T_matrix_path = os.path.join(
        "CV", "Camera", "extrinsic_calibration", "Camera_to_robot_transformation.npy")
    T = np.load(T_matrix_path)
    marker_pos_robot_frame = np.dot(T.T, marker_pos_camera_frame)
    return marker_pos_robot_frame


def get_object_position(object_marker_ID):

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
    object_center_aruco_f = np.array([0, 0, -0.025, 1])
    while True:
        warmup_counter += 1
        frames = pipe.wait_for_frames()
        if warmup_counter > 50:
            align = rs.align(rs.stream.color)
            aligned = align.process(frames)
            c_frames = aligned.get_color_frame()
            img = np.asanyarray(c_frames.get_data())
            img, marker_Transformations, rvec, tvec = aruco_pose_estimation(img, aruco_dict_type=aruco_marker_type,
                                                                            matrix_coefficients=mtx, distortion_coefficients=dist, actual_size=0.04)

            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            if marker_Transformations[int(object_marker_ID)] is not None:
                pipe.stop()
                marker_pos_robot_frame = map_c_2_r(
                    marker_Transformations[int(object_marker_ID)][:, 3])  # aruco [0,0,0]
                object_center_r_f = np.zeros(4)

                object_center_r_f[:3] = map_c_2_r(np.dot(
                    marker_Transformations[int(object_marker_ID)], object_center_aruco_f))[:3]

                return object_center_r_f
            else:
                print("ERROR: object marker not detected. Aborting")
                sys.exit()


def read_camera_params():
    PARAMS_PATH = "/home/stefan/Documents/Masterarbeit/DRL/hindsight-experience-replay-ur5/CV/Camera/camera_info"
    rvecs = np.load(os.path.join(PARAMS_PATH, "rvecs.npy"))
    tvecs = np.load(os.path.join(PARAMS_PATH, "tvecs.npy"))
    mtx = np.load(os.path.join(PARAMS_PATH, "mtx.npy"))
    dist = np.load(os.path.join(PARAMS_PATH, "dist.npy"))
    ret = np.load(os.path.join(PARAMS_PATH, "ret.npy"))

    return rvecs, tvecs, mtx, dist, ret


def aruco_pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, actual_size=0.1):
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
    rvec = None
    tvec = None
    # If markers are detected
    if len(corners) > 0:
        for i, ID in enumerate(ids):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], actual_size, matrix_coefficients,
                                                                           distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(
                frame, corners, ids=ids, borderColor=(143, 143, 250))
            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients,
                               distortion_coefficients, rvec, tvec, 0.05)
            marker_Transformations[int(
                ID)] = calculate_rotation_matrix(rvec, tvec)
    return frame, marker_Transformations, rvec, tvec


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


def recover_homogenous_affine_transformation(p, p_prime):
    '''
    Find the unique homogeneous affine transformation that
    maps a set of 3 points to another set of 3 points in 3D
    space:

        p_prime == np.dot(p, R) + t

    where `R` is an unknown rotation matrix, `t` is an unknown
    translation vector, and `p` and `p_prime` are the original
    and transformed set of points stored as row vectors:

        p       = np.array((p1,       p2,       p3))
        p_prime = np.array((p1_prime, p2_prime, p3_prime))

    The result of this function is an augmented 4-by-4
    matrix `A` that represents this affine transformation:

        np.column_stack((p_prime, (1, 1, 1))) == \
            np.dot(np.column_stack((p, (1, 1, 1))), A)

    Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q = p[1:] - p[0]
    Q_constructed = np.cross(*Q)

    Q_prime = p_prime[1:] - p_prime[0]
    Q_prime_constructed = np.cross(*Q_prime)

    Q_matrix = np.row_stack((Q, Q_constructed))
    Q_prime_matrix = np.row_stack((Q_prime, Q_prime_constructed))

    # P_1 x T   = P_2
    # T         = P_1⁻¹ x P_2

    Q_matrix_inv = np.linalg.inv(Q_matrix)

    R = np.dot(Q_matrix_inv, Q_prime_matrix)

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix

    return np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1)))
