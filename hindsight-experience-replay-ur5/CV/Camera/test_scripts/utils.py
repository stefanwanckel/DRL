import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
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


def show_robustness(lst_rvec, lst_tvec):
    assert len(lst_rvec) == len(lst_tvec)
    rvec_norm = []
    tvec_norm = []
    if len(lst_rvec) > 1:
        for i in range(1, len(lst_rvec)):
            rvec_norm.append(np.linalg.norm(lst_rvec[i]-lst_rvec[-1]))
            tvec_norm.append(np.linalg.norm(lst_tvec[i]-lst_tvec[-1]))

        fig, axs = plt.subplots(2)
        fig.suptitle('robustness of rvec and tvec')
        axs[0].set_title("rvec")
        axs[1].set_title("tvec")
        axs[0].plot(rvec_norm)
        axs[1].plot(tvec_norm)

        plt.pause(0.01)
        plt.show()


def get_custom_T_matrix(points_cf_1, points_cf_2):

    no_img = len(points_cf_1)

    # Points_1 = np.array(points_cf_1)[:, :3]
    # Points_2 = np.array(points_cf_2)[:, :3]
    Points_1 = np.array(points_cf_1)[:3]
    Points_2 = np.array(points_cf_2)[:3]

    # construct p
    b = Points_2.flatten()

  # construct A
    A = np.zeros((no_img*3, 9))
    for i in range(0, no_img):
        A[0+i*3, :3] = Points_1[i, :]
        A[0+i*3, 6] = 1

        A[1+i*3, 1] = Points_1[i, 0]
        A[1+i*3, 3] = Points_1[i, 1]
        A[1+i*3, 4] = Points_1[i, 2]
        A[1+i*3, 7] = 1

        A[2+i*3, 2] = Points_1[i, 0]
        A[2+i*3, 4] = Points_1[i, 1]
        A[2+i*3, 5] = Points_1[i, 2]
        A[2+i*3, 8] = 1

    x = np.linalg.solve(A, b)
    T_cf1_cf2 = np.zeros((4, 4))
    T_cf1_cf2[0, :3] = x[:3]
    T_cf1_cf2[:3, 3] = x[6:]
    T_cf1_cf2[3, :] = np.array([0, 0, 0, 1])
    T_cf1_cf2[1, 0] = x[1]
    T_cf1_cf2[1, 1:3] = x[3:5]
    T_cf1_cf2[2, 0] = x[2]
    T_cf1_cf2[2, 1:3] = x[4:6]

    return T_cf1_cf2


def get_custom_T_matrix_hack(points_cf_1, points_cf_2):

    no_img = len(points_cf_1)

    # Points_1 = np.array(points_cf_1)[:, :3]
    # Points_2 = np.array(points_cf_2)[:, :3]
    Points_1 = np.array(points_cf_1)
    Points_2 = np.array(points_cf_2)

    # construct p
    b = Points_2.flatten()

  # construct A
    A = np.zeros((no_img*3, 9))
    for i in range(0, no_img):
        A[0+i*3, :3] = Points_1[i, :]
        A[0+i*3, 6] = 1

        A[1+i*3, 1] = Points_1[i, 0]
        A[1+i*3, 3] = Points_1[i, 1]
        A[1+i*3, 4] = Points_1[i, 2]
        A[1+i*3, 7] = 1

        A[2+i*3, 2] = Points_1[i, 0]
        A[2+i*3, 4] = Points_1[i, 1]
        A[2+i*3, 5] = Points_1[i, 2]
        A[2+i*3, 8] = 1

    x = np.linalg.solve(A, b)
    T_cf1_cf2 = np.zeros((4, 4))
    T_cf1_cf2[0, :3] = x[:3]
    T_cf1_cf2[:3, 3] = x[6:]
    T_cf1_cf2[3, :] = np.array([0, 0, 0, 1])
    T_cf1_cf2[1, 0] = x[1]
    T_cf1_cf2[1, 1:3] = x[3:5]
    T_cf1_cf2[2, 0] = x[2]
    T_cf1_cf2[2, 1:3] = x[4:6]

    return T_cf1_cf2


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


# cf1 camera
# cf2 is robot
# milestones
# get 3 points and transformation matrix
# check difference of position and if the match the transformation taking matrix times camera aruco pos = Robot pos
# difference can be a correction factor
#
