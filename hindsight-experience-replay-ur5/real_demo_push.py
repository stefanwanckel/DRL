# Imports
# visualization
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
# real robot
import urkin
import rtde_receive
import rtde_control
import ur5_env_mjco
# simulation
import gym
#RL and model
from rl_modules.models import actor
import torch
# MPI
from numpy.lib.function_base import _average_dispatcher
# CV

# general
import time
import math
import numpy as np
import random
import json
import os
# custom functions
from utils import *
from arguments import get_args

np.set_printoptions(precision=2, suppress=True)
# Bool deciders
ON_REAL_ROBOT = False

# get arguments for demo
args = get_args()
model_path = args.save_dir + args.env_name + '/modelur5.pt'

o_mean, o_std, g_mean, g_std, model = torch.load(
    model_path, map_location=lambda storage, loc: storage)
# create gym environment and get first observation
env = gym.make(args.env_name)
observation = env.reset()


# get environment parameters shapes to initialize torch model
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
    rtde_c = rtde_control.RTDEControlInterface("192.168.178.15")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.15")

# move robot to start configuration
joint_q = [-0.7866423765765589,
           -1.8796035252013148,
           -1.7409639358520508,
           -1.0964625638774415,
           1.5797905921936035,
           -0.0025427977191370132]

# move to joint jointq and get TCP pose
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

"""
Description of observation needed for the agent
ORDER OF CONCATENATION grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel

DESCRIPTION OF PARAMETERS
grip_pos            Gripper position in robot cartesian coordinate frame        len(3)
object_pos          object position in cartesian coordinate frame               len(3)
object_rel_pos      Difference between object position and gripper position     len(3)
gripper_state       The state of the gripper fingers. 
                    0 means closed, 1 means open                                len(2)
object_rot          Object orientation w.r.t. world coordinate frame. 
                    As Euler angles. Extracted from Transformation matrix
                    Use rotations.py library                                    len(3)
object_velp         Object velocity in cartesian coordinate frame.
                    Needs to be calculated using prior state.                   len(3)
grip_velp           Gripper velocity in robot cartesian coordinate frame
                    Can be directly taken from rtde_receive                     len(3)
object_velr         Rotational velocity around axes.
                    Will be 0 since no rotation will be expected
                    due to slow motion velocity                                 len(3)
gripper_vel         velocity of gripper fingers.
                    Set to 0 as grippers will be closed during the test         len(2)
"""

grip_pos = get_xpos_from_robot()
object_pos = get_object_pos_from_camera()
object_rel_pos = object_pos - grip_pos
gripper_state = [0, 0]
object_rot = NotImplementedError()
object_velp = NotImplementedError()
grip_velp = rtde_r.actual_TCP_speed()
object_velr = [0, 0, 0]
gripper_vel = [0, 0]

obs = np.concatenate([grip_pos, object_pos, object_rel_pos, gripper_state,
                     object_rot, object_velp, grip_velp, object_velr, gripper_vel])


print("the starting position is {} \nThe starting orientation is{}".format(startPos, orn))
currPos = startPos

# setting parameters for robot motion
stepSize = 0.025
SampleRange = 0.40
goal_threshold = 0.05

# setup figure  limits
axisLimitExtends = 0.10

# Init lists for visualization and info dict
x = []
y = []
z = []
info = {}

for nTests in range(3):
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
    for t in range(50):
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
        for substeps in range(10):
            newPos = currPos + stepSize * action
            if ON_REAL_ROBOT:
                rtde_c.moveL(np.hstack((newPos, orn)), 0.2, 0.3)
                time.sleep(0.05)
                actual_newPos = rtde_r.getActualTCPPose()[0:3]
            else:
                actual_newPos = newPos

            x.append(actual_newPos[0])
            y.append(actual_newPos[1])
            z.append(actual_newPos[2])

            # fill info
            info["timestep"].append(t)
            info["old_position"].append(currPos)
            info["action"].append(action)
            info["new_position"].append(newPos)
            info["actual_new_position"].append(actual_newPos)
            info["goal"].append(g)
            info["distance_to_goal"].append(
                np.linalg.norm(actual_newPos-g_robotCF))
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
                if ON_REAL_ROBOT:
                    rtde_c.stopScript()
                break

    print("DONE")
