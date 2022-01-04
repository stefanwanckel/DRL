# Visualisation
import pyrealsense2 as rs
from utils import *
from CV.Camera.test_scripts.utils import ARUCO_DICT, read_camera_params, aruco_pose_estimation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import argparse
import time
import cv2
import sys


def setup_vis_reach(nTests, startPos, SampleRange, axisLimitExtends, g_robotCF, goal_threshold):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    fig.suptitle("TCP Planar View REACH - Episode {}".format(nTests))
    fig.canvas.set_window_title("Episode No. {}".format(nTests))
    ax = fig.gca()

    ax1.set_xlim(startPos[0]-SampleRange-axisLimitExtends,
                 startPos[0]+SampleRange+axisLimitExtends)
    ax1.set_ylim(startPos[1]-SampleRange-axisLimitExtends,
                 startPos[1]+SampleRange+axisLimitExtends)
    ax2.set_xlim(startPos[0]-SampleRange-axisLimitExtends,
                 startPos[0]+SampleRange+axisLimitExtends)
    ax2.set_ylim(startPos[2]-SampleRange-axisLimitExtends,
                 startPos[2]+SampleRange+axisLimitExtends)
    ax1.grid()
    ax2.grid()
    ax1.set(adjustable='box', aspect='equal')
    ax2.set(adjustable='box', aspect='equal')
    ax1.set_title("X-Y TCP")
    ax2.set_title("X-Z TCP")
    ax1.set_xlabel("X-axis [m]")
    ax2.set_xlabel("X-axis [m]")
    ax1.set_ylabel("Y-axis [m]")
    ax2.set_ylabel("Z-axis [m]")

    ax1.plot(g_robotCF[0], g_robotCF[1], 'o', color="g")
    ax2.plot(g_robotCF[0], g_robotCF[2], 'o', color="g")
    graph, = ax1.plot([], [], 'o', color="r")
    sampleSpace_xy = Circle((startPos[0], startPos[1]),
                            radius=SampleRange+0.05, fill=False)
    successThreshold_xy = Circle(
        (g_robotCF[0], g_robotCF[1]), radius=goal_threshold, fill=False, ls="--")
    sampleSpace_xz = Circle((startPos[0], startPos[2]),
                            radius=SampleRange+0.05, fill=False)
    successThreshold_xz = Circle(
        (g_robotCF[0], g_robotCF[2]), radius=goal_threshold, fill=False, ls="--")
    ax1.add_patch(sampleSpace_xy)
    ax1.add_patch(successThreshold_xy)
    ax2.add_patch(sampleSpace_xz)
    ax2.add_patch(successThreshold_xz)

    return fig, (ax1, ax2)


def setup_vis_push(nTests, startPos, SampleRange, axisLimitExtends, g_robotCF, goal_threshold):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    fig.suptitle("TCP Planar View PUSH - Episode {}".format(nTests))
    fig.canvas.set_window_title("Episode No. {}".format(nTests))
    ax = fig.gca()

    ax1.set_xlim(startPos[0]-SampleRange-axisLimitExtends,
                 startPos[0]+SampleRange+axisLimitExtends)
    ax1.set_ylim(startPos[1]-SampleRange-axisLimitExtends,
                 startPos[1]+SampleRange+axisLimitExtends)
    ax2.set_xlim(startPos[0]-SampleRange-axisLimitExtends,
                 startPos[0]+SampleRange+axisLimitExtends)
    ax2.set_ylim(startPos[2]-SampleRange-axisLimitExtends,
                 startPos[2]+SampleRange+axisLimitExtends)
    ax1.grid()
    ax2.grid()
    ax1.set(adjustable='box', aspect='equal')
    ax2.set(adjustable='box', aspect='equal')
    ax1.set_title("PUSH X-Y TCP")
    ax2.set_title("PUSH X-Z TCP")
    ax1.set_xlabel("X-axis [m]")
    ax2.set_xlabel("X-axis [m]")
    ax1.set_ylabel("Y-axis [m]")
    ax2.set_ylabel("Z-axis [m]")

    ax1.plot(g_robotCF[0], g_robotCF[1], 'o', color="g")
    ax2.plot(g_robotCF[0], g_robotCF[2], 'o', color="g")
    graph, = ax1.plot([], [], 'o', color="r")
    sampleSpace_xy = Circle((startPos[0], startPos[1]),
                            radius=SampleRange+0.05, fill=False)
    successThreshold_xy = Circle(
        (g_robotCF[0], g_robotCF[1]), radius=goal_threshold, fill=False, ls="--")
    sampleSpace_xz = Circle((startPos[0], startPos[2]),
                            radius=SampleRange+0.05, fill=False)
    successThreshold_xz = Circle(
        (g_robotCF[0], g_robotCF[2]), radius=goal_threshold, fill=False, ls="--")
    ax1.add_patch(sampleSpace_xy)
    ax1.add_patch(successThreshold_xy)
    ax2.add_patch(sampleSpace_xz)
    ax2.add_patch(successThreshold_xz)

    return fig, (ax1, ax2)


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -
                     args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -
                     args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


def load_last_model(save_dir, env_name):
    search_dir = os.path.join(save_dir, env_name)
    #search_dir = "./saved_models/ur5_push_no_gripper-v1"
    files = os.listdir(search_dir)
    files_ext = [file for file in files if file.endswith(".pt")]
    files_sorted = sorted(files_ext,
                          key=lambda x: os.path.getmtime(
                              os.path.join(search_dir, x))
                          )
    assert len(files_sorted) > 0, "{} must be empty...".format(search_dir)
    return files_sorted[-1]


def load_last_archived_model(save_dir, env_name, project_dir):
    search_dir = os.path.join(save_dir, env_name, project_dir)
    #search_dir = "./saved_models/ur5_push_no_gripper-v1"
    files = os.listdir(search_dir)
    files_ext = [file for file in files if file.endswith(".pt")]
    files_sorted = sorted(files_ext,
                          key=lambda x: os.path.getmtime(
                              os.path.join(search_dir, x))
                          )
    assert len(files_sorted) > 0, "{} must be empty...".format(search_dir)
    return files_sorted[-1]


def get_demo_model_path(args, last_model, is_archived):
    """
    loads model depending on:
    Environment name
    if demo should be run with latest model
    """
    if args.env_name == "ur5_reach-v1":
        model_path = args.save_dir + args.env_name + '/Oct_24_1_2.pt'
    elif args.env_name == "ur5_push-v1":
        if last_model:
            if is_archived:
                model_path = os.path.join(
                    args.save_dir, args.env_name, args.project_dir, load_last_archived_model(args.save_dir, args.env_name, args.project_dir))
            else:
                model_path = os.path.join(
                    args.save_dir, args.env_name, load_last_model(args.save_dir, args.env_name))
            print("Last model name: ", load_last_model(
                args.save_dir, args.env_name))
            time.sleep(1)
        else:
            model_path = args.save_dir + args.env_name + \
                '/2021-12-10T04:00:18.657028_epoch_79.pt'
    elif args.env_name == "ur5_reach_no_gripper-v1":
        if last_model:
            if is_archived:
                model_path = os.path.join(
                    args.save_dir, args.env_name, args.project_dir, load_last_archived_model(args.save_dir, args.env_name, args.project_dir))
            else:
                model_path = os.path.join(
                    args.save_dir, args.env_name, load_last_model(args.save_dir, args.env_name))
            print("Last model name: ", load_last_model(
                args.save_dir, args.env_name))
            time.sleep(1)
        else:
            model_path = args.save_dir + args.env_name + \
                '/2021-12-10T04:00:18.657028_epoch_79.pt'
    elif args.env_name == "ur5_push_no_gripper-v1":
        if last_model:
            if is_archived:
                model_path = os.path.join(
                    args.save_dir, args.env_name, args.project_dir, load_last_archived_model(args.save_dir, args.env_name, args.project_dir))
                print("Last model name: ", load_last_archived_model(
                    args.save_dir, args.env_name, args.project_dir))
            else:
                model_path = os.path.join(
                    args.save_dir, args.env_name, load_last_model(args.save_dir, args.env_name))
                print("Last model name: ", load_last_model(
                    args.save_dir, args.env_name))
            time.sleep(1)
        else:
            model_path = args.save_dir + args.env_name + \
                '/2021-12-11T17:18:10.390514_epoch_18.pt'
    elif args.env_name == "ur5_pick_and_place-v1":
        if last_model:
            if is_archived:
                model_path = os.path.join(
                    args.save_dir, args.env_name, args.project_dir, load_last_archived_model(args.save_dir, args.env_name, args.project_dir))
                print("Last model name: ", load_last_archived_model(
                    args.save_dir, args.env_name, args.project_dir))
            else:
                model_path = os.path.join(
                    args.save_dir, args.env_name, load_last_model(args.save_dir, args.env_name))
                print("Last model name: ", load_last_model(
                    args.save_dir, args.env_name))
            time.sleep(1)
        else:
            model_path = args.save_dir + args.env_name + \
                '/2021-12-11T17:18:10.390514_epoch_18.pt'

    return model_path


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


def get_object_position(goal_marker_ID=1):

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
    object_aruco_center_offset = np.array([0, 0, -0.025, 0])
    while True:
        warmup_counter += 1
        frames = pipe.wait_for_frames()
        if warmup_counter > 50:
            align = rs.align(rs.stream.color)
            aligned = align.process(frames)
            c_frames = aligned.get_color_frame()
            img = np.asanyarray(c_frames.get_data())
            img, marker_Transformations, rvec, tvec = aruco_pose_estimation(img, aruco_dict_type=aruco_marker_type,
                                                                            matrix_coefficients=mtx, distortion_coefficients=dist, actual_size=0.05)

            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            if marker_Transformations[int(goal_marker_ID)] is not None:
                pipe.stop()
                marker_pos_robot_frame = map_c_2_r(
                    marker_Transformations[int(goal_marker_ID)][:, 3])
                return marker_pos_robot_frame+object_aruco_center_offset
                break
            else:
                print("ERROR: object marker not detected. Aborting")
                sys.exit()
