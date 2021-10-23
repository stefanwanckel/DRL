import cv2
import pyrealsense2 as rs


class ArucoManager:
    def __init__(self):
        self.Tc2r = None
        self.path2intrinsics = "Camera/camera_info/"
        self.read_camera_intrinsics()
        self.warmup_length = 50
        self.aruco_dict_type = "DICT_5X5_100"
        self.aruco_marker = ARUCO_DICT[self.aruco_dict_type]

    def get_Tc2r(self):
        pass

    def estimate_aruco_pose(self):
        pass

    def read_camera_intrinsics(self):
        self.rvecs = np.load(os.path.join(self.path2intrinsics, "rvecs.npy"))
        self.tvecs = np.load(os.path.join(self.path2intrinsics, "tvecs.npy"))
        self.mtx = np.load(os.path.join(self.path2intrinsics, "mtx.npy"))
        self.dist = np.load(os.path.join(self.path2intrinsics, "dist.npy"))
        self.ret = np.load(os.path.join(self.path2intrinsics, "ret.npy"))

    def setup_pipeline(self):
        self.pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = pipe.start(config)

    def start_camera():
        frames = pipe.wait_for_frames()
