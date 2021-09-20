# general imports
import seaborn as sns
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
import json
import os
np.set_printoptions(precision=12, suppress=True)
# imports for robot control
# imports for sim environment and agent
# imports for visualization
ON_REAL_ROBOT = False
# process the inputs


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


# get arguments for demo import model
args = get_args()
# model_path = args.save_dir + args.env_name + '/July30_reach_2_39.pt'
model_path = args.save_dir + args.env_name + '/reach_8.pt'

o_mean, o_std, g_mean, g_std, model = torch.load(
    model_path, map_location=lambda storage, loc: storage)
# create gym environment and get first observation
env = gym.make(args.env_name)
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
    rtde_c = rtde_control.RTDEControlInterface("192.168.178.15")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.15")

# move robot to start configuration
joint_q = [-0.7866423765765589,
           -1.8796035252013148,
           -1.7409639358520508,
           -1.0964625638774415,
           1.5797905921936035,
           -0.0025427977191370132]
if ON_REAL_ROBOT:
    rtde_c.moveJ(joint_q)

# get TCP pose
if ON_REAL_ROBOT:
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

print("the starting position is {} \nThe starting orientation is{}".format(startPos, orn))
currPos = startPos
# setting parameters for robot motion
stepSize = 0.01
SampleRange = 0.20
goal_threshold = 0.05
# setup figure  limits
axisLimitExtends = 0.10

# Init lists
x = []
y = []
z = []
info = {}

for nTests in range(2):
    # reset position
    currPos = startPos
    # reset x and y
    x = []
    y = []
    z = []
    # setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    fig.suptitle("TCP Planar View")
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
    ax1.plot(g_robotCF[0], g_robotCF[1], 'o', color="g")
    ax2.plot(g_robotCF[0], g_robotCF[2], 'o', color="g")
    graph, = ax1.plot([], [], 'o', color="r")
    sampleSpace_xy = Circle((startPos[0], startPos[1]),
                            radius=SampleRange, fill=False)
    successThreshold_xy = Circle(
        (g_robotCF[0], g_robotCF[1]), radius=goal_threshold, fill=False, ls="--")
    sampleSpace_xz = Circle((startPos[0], startPos[2]),
                            radius=SampleRange, fill=False)
    successThreshold_xz = Circle(
        (g_robotCF[0], g_robotCF[2]), radius=goal_threshold, fill=False, ls="--")
    ax1.add_patch(sampleSpace_xy)
    ax1.add_patch(successThreshold_xy)
    ax2.add_patch(sampleSpace_xz)
    ax2.add_patch(successThreshold_xz)

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
    # start animation
    plt.show()
    print("DONE")
