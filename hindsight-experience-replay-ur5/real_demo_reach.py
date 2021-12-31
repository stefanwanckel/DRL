# general imports
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import ur5_env_mjco
import gym
from arguments import get_args
from rl_modules.models import actor
from numpy.lib.function_base import _average_dispatcher
import torch
import urkin
import rtde_receive
import rtde_control
import time
import math
import numpy as np
import random
from collections import OrderedDict
import os
from utils import *
import pickle
import datetime
np.set_printoptions(precision=3, suppress=True)
# imports for robot control
# imports for sim environment and agent
# imports for visualization
ON_REAL_ROBOT = True
# get arguments for demo import model
args = get_args()
env_name = "ur5_reach_no_gripper-v1"
model_name = '/2021-12-31T10:45:23.477367_epoch_7.pt'
# model_path = args.save_dir + args.env_name + '/July30_reach_2_39.pt'
model_path = args.save_dir + env_name + "/ur5_reach_raw" + model_name

o_mean, o_std, g_mean, g_std, model, _, _, _ = torch.load(
    model_path, map_location=lambda storage, loc: storage)
# create gym environment and get first observation
env = gym.make(env_name)
observation = env.reset()
# get environment parameters
env_params = {'obs': observation['observation'].shape[0],
              'goal': observation['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              }
# setup agent with loaded model
actor_network = actor(env_params)
actor_network.load_state_dict(model)
actor_network.eval()

# setup conneciton to robot
if ON_REAL_ROBOT:
    rtde_c = rtde_control.RTDEControlInterface("192.168.178.232")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.232")

# move robot to start configuration
joint_q = [-0.7866423765765589,
           -1.8796035252013148,
           -1.7409639358520508,
           -1.0964625638774415,
           1.5797905921936035,
           -0.0025427977191370132]
# start motion was here
if ON_REAL_ROBOT:
    rtde_c.moveJ(joint_q)
    TCPpose = rtde_r.getActualTCPPose()
    startPos = TCPpose[0:3]
    orn = TCPpose[3:]
else:
    TCP_pos_path = os.path.join(
        "Results", "real_robot", "Reach_TCP_start_pose.json")
    with open(TCP_pos_path, "r") as f:
        TCPpose = json.load(f)
    startPos = TCPpose[0:3]
    orn = TCPpose[3:]

currPos = startPos
# setting parameters for robot motion
stepSize = 0.05
SampleRange = 0.20
goal_threshold = 0.05
# setup figure  limits
axisLimitExtends = 0.10

# Init lists
x = []
y = []
z = []
info = {}
nEvaluations = 1
INFO = OrderedDict()
for nTests in range(nEvaluations):

    # Move to start and get TCP pose
    if ON_REAL_ROBOT:
        rtde_c.moveJ(joint_q)
        TCPpose = rtde_r.getActualTCPPose()
        startPos = TCPpose[0:3]
        orn = TCPpose[3:]
    else:
        TCP_pos_path = os.path.join(
            "Results", "real_robot", "Reach_TCP_start_pose.json")
        with open(TCP_pos_path, "r") as f:
            TCPpose = json.load(f)
        startPos = TCPpose[0:3]
        orn = TCPpose[3:]
    # reset position
    currPos = startPos
    # reset x and y
    x = []
    y = []
    z = []

    observation = env.reset()
    # observation must be robot pos
    obs = np.hstack((startPos, np.zeros(7)))
    obs_sim = observation['observation']

    # difference between sim and real world coordinate frames (mit marcus)
    obs_diff = obs_sim-obs

    # sampling goal
    rndDisp = -SampleRange + 2*SampleRange*np.random.random(3)
    g = list(np.asarray(startPos + obs_diff[:3]) + rndDisp)
    g_robotCF = list(np.asarray(startPos) + rndDisp)
    while np.linalg.norm(np.array(g_robotCF)-g) < 1*SampleRange:
        g = list(np.asarray(startPos + obs_diff[:3]) + rndDisp)
        g_robotCF = list(np.asarray(startPos) + rndDisp)

    # plotting and setting up plot
    # setup figure
    fig, (ax1, ax2) = setup_vis(nTests, startPos, SampleRange,
                                axisLimitExtends, g_robotCF, goal_threshold)

    # logging
    info["timestep"] = []
    info["old_position"] = []
    info["action"] = []
    info["new_position"] = []
    info["actual_new_position"] = []
    info["goal"] = []
    info["distance_to_goal"] = []
    info["is_success"] = []
    info["displacement"] = []
    info["real_displacement"] = []

    # for t in range(env._max_episode_steps):
    n_timesteps = 50
    for t in range(n_timesteps):
        # exit this for loop. THis is necessary due to the
        # substeps condition exiting the substep loop but then continuing in the t range
        if len(info["is_success"]) > 0:
            if info["is_success"][-1]:
                break
        # add coordinate frame diff (mit marcus)
        obs = obs + obs_diff
        inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
        with torch.no_grad():
            pi = actor_network(inputs)

        action = pi.detach().numpy().squeeze()

        # ignore gripper at first
        action = action[0:-1]
        # move robot to new position which is old position plus stepsize times the action
        # Orn remains the same throughout
        n_substeps = 1
        for substeps in range(n_substeps):
            newPos = currPos + stepSize * action
            if ON_REAL_ROBOT:
                rtde_c.moveL(np.hstack((newPos, orn)), 0.2, 0.3)
                time.sleep(0.05)
                actual_newPos = np.array(rtde_r.getActualTCPPose()[0:3])
            else:
                actual_newPos = newPos

            x.append(actual_newPos[0])
            y.append(actual_newPos[1])
            z.append(actual_newPos[2])

            # fill info
            g_robotCF_np = np.array(g_robotCF)
            info["timestep"].append(t)
            info["old_position"].append(currPos)
            info["action"].append(action)
            info["new_position"].append(newPos)
            info["actual_new_position"].append(actual_newPos)
            info["goal"].append(g)
            info["distance_to_goal"].append(
                np.linalg.norm(actual_newPos-g_robotCF_np))
            info["is_success"].append(np.linalg.norm(
                actual_newPos-g_robotCF) < goal_threshold)
            info["displacement"].append(
                list(np.asarray(actual_newPos) - np.asarray(currPos)))
            info["real_displacement"].append(stepSize * action)

            # updating position
            obs[0:3] = actual_newPos
            currPos = newPos  # actual_newPos

            if t % 1 == 0:
                print("*"*20)
                for key in info:
                    # print("|{0:>20} : {0:>20}|".format(key,info[key]))
                    print(key, info[key][t])

            # plotting and setting up plot
            ax1.plot(x, y, 'o', color="r")
            ax2.plot(x, z, 'o', color="r")
            plt.pause(0.05)

            # checking for success
            if info["is_success"][-1] == True:
                print("Success! Norm is {}".format(
                    np.linalg.norm(actual_newPos-g_robotCF)))
                # saving plots
                fig_path = os.path.join("Results", "reach", "plane_view")
                plt.savefig(os.path.join(
                    fig_path, "plane_view_episode_"+str(nTests)+".svg"))
                INFO[nTests] = info
                if ON_REAL_ROBOT and nTests == range(nEvaluations)[-1]:
                    rtde_c.stopScript()
                break
    info_path = os.path.join("Results", "reach")
    with open(os.path.join(info_path, 'INFO.pkl'), 'wb') as f:
        pickle.dump(INFO, f)
today = datetime.date.today()
today = today.strftime("%d/%m/%Y")
time = datetime.datetime.now()
time = current_time = time.strftime("%H:%M:%S")
date_n_time = today + " " + time
settings = OrderedDict()
settings["date_n_time"] = date_n_time
settings["env_name"] = env_name
settings["model_name"] = model_name
settings["model_name"] = model_name
settings["model_path"] = model_path
settings["robot_start_pos"] = joint_q
settings["step_size"] = stepSize
settings["Sample_range"] = SampleRange
settings["goal_threshold"] = goal_threshold
settings["axis_limit_extends"] = axisLimitExtends
settings["n_evaluations"] = nEvaluations
settings["n_timesteps"] = n_timesteps
settings["n_substeps"] = n_substeps
with open(os.path.join(info_path, 'settings.txt'), 'w') as f:
    for key, val in settings.items():
        f.write(f"{key} : {val}\n")
    print("DONE")
