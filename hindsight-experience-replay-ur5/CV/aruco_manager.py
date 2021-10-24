import cv2
import pyrealsense2 as rs
import rtde_receive
import rtde_control
import numpy as np

"""
THreading
https://gist.github.com/allskyee/7749b9318e914ca45eb0a1000a81bf56
https://docs.python.org/3/library/threading.html
"""


class ArucoManager:
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)
        self.Tc2r = None
        self.path2intrinsics = "Camera/camera_info/"
        self.read_camera_intrinsics()
        self.warmup_length = 50
        self._load_aruco_dict()
        self.aruco_dict_type = "DICT_5X5_100"
        self.aruco_marker = self.ARUCO_DICT[self.aruco_dict_type]
        self.joint_q = [-0.7866423765765589,
                        -1.8796035252013148,
                        -1.7409639358520508,
                        -1.0964625638774415,
                        1.5797905921936035,
                        -0.0025427977191370132]

    def get_Tc2r(self):

        rtde_c = rtde_control.RTDEControlInterface("192.168.178.15")
        rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.15")
        print("[INFO] AQUIRING ROBOT TO CAMERA TRANSFORMATION")
        print("[INFO] 1. move robot to idle position (jointq)")
        print("[INFO] 2. aquire an image with 3 aruco markers on it.")

        rtde_c.moveJ(joint_q)
        TCPpose = rtde_r.getActualTCPPose()
        print("[INFO] 1. move robot to idle position (jointq)")

        print("[INFO] starting camera")
        self._setup_camera()
        warmup_counter = 0
        while True:
            self.frame = self.wait_for_frames()
            if warmup_counter > self.warmup_length:
                align = rs.align(rs.stream.color)
                aligned = align.process(frames)
                c_frames = aligned.get_color_frame()
                img = np.asanyarray(c_frames.get_data())
                img, rvec, tvec =

        """
        jog ur5
        https://www.universal-robots.com/articles/ur/programming/how-to-jog-the-robot/
        teach mode for freedrive start and stop
        https://sdurobotics.gitlab.io/ur_rtde/api/api.html
        """
        pass

    def estimate_aruco_pose(self):
        pass

    def read_camera_intrinsics(self):
        self.rvecs = np.load(os.path.join(self.path2intrinsics, "rvecs.npy"))
        self.tvecs = np.load(os.path.join(self.path2intrinsics, "tvecs.npy"))
        self.mtx = np.load(os.path.join(self.path2intrinsics, "mtx.npy"))
        self.dist = np.load(os.path.join(self.path2intrinsics, "dist.npy"))
        self.ret = np.load(os.path.join(self.path2intrinsics, "ret.npy"))

    def wait_for_frames(self):

        self.frames = pipe.wait_for_frames()

############HELPER FUNCTIRONS#####
    def _setup_camera(self):
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = pipe.start(config)

    def _aruco_pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
        '''
        frame - Frame from the video stream
        matrix_coefficients - Intrinsic matrix of the calibrated camera
        distortion_coefficients - Distortion coefficients associated with your camera

        return:-
        frame - The frame with the axis drawn on it
        '''

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.Dictionary_get(self.aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters_create()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters,
                                                                    cameraMatrix=self.mtx,
                                                                    distCoeff=self.dist)
        print(f"[MARKER_DETECTION] {str(len(ids))} markers detected. ")
        lst_rvec = None
        lst_tvec = None
        # If markers are detected
        if len(corners) > 0:
            for i, ID in enumerate(ids):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.1, matrix_coefficients,
                                                                               distortion_coefficients)
                lst_rvec.append(rvec)
                lst_tvec.append(tvec)
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(
                    frame, corners, ids=ids, borderColor=(0, 255, 0))
                # Draw Axis
                cv2.aruco.drawAxis(frame, matrix_coefficients,
                                   distortion_coefficients, rvec, tvec, 0.05)

        return frame, lst_rvec, lst_tvec

    def _load_aruco_dict(self):
        self.ARUCO_DICT = ARUCO_DICT = {
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
