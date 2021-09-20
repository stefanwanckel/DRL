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
np.set_printoptions(precision=12, suppress=True)
# imports for robot control
# imports for sim environment and agent
# imports for visualization

# process the inputs
goal_threshold = 0.05


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


def animate(i):
    graph.set_data(x[:i+1], y[:i+1])
    return graph


# get arguments for demo import model
args = get_args()
#model_path = args.save_dir + args.env_name + '/July30_reach_2_39.pt'
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
rtde_c = rtde_control.RTDEControlInterface("192.168.178.15")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.15")


joint_q = [-0.7866423765765589,
           -1.8796035252013148,
           -1.7409639358520508,
           -1.0964625638774415,
           1.5797905921936035,
           -0.0025427977191370132]

# move robot to start configuration
rtde_c.moveJ(joint_q)

# get TCP pose
TCPpose = rtde_r.getActualTCPPose()
startPos = TCPpose[0:3]
orn = TCPpose[3:]
print("the starting position is {} \nThe starting orientation is{}".format(startPos, orn))
currPos = startPos
# setting parameters for robot motion
stepSize = 0.015
SampleRange = 0.20

# setup figure and limits
axisLimitExtends = 0.10
# sns.set_theme()
fig = plt.figure()
ax = fig.gca()
plt.axis("equal")
plt.xlim(startPos[0]-SampleRange-axisLimitExtends,
         startPos[0]+SampleRange+axisLimitExtends)
plt.ylim(startPos[1]-SampleRange-axisLimitExtends,
         startPos[1]+SampleRange+axisLimitExtends)
plt.title("X-Y TCP")
plt.xlabel("X-axis [m]")
plt.ylabel("Y-axis [m]")
# Init lists
x = []
y = []
info = {}

for nTests in range(1):
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
    # plotting and setting up plot
    plt.plot(g_robotCF[0], g_robotCF[1], 'o', color="g")
    graph, = plt.plot([], [], 'o', color="r")
    sampleSpace = Circle((startPos[0], startPos[1]),
                         radius=SampleRange, fill=False)
    successThreshold = Circle(
        (g_robotCF[0], g_robotCF[1]), radius=goal_threshold, fill=False, ls="--")
    ax.add_patch(sampleSpace)
    ax.add_patch(successThreshold)

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
        # add coordinate frame diff (mit marcus)
        obs = obs + obs_diff
        inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
        #inputs = np.concatenate([obs,g])
        with torch.no_grad():
            pi = actor_network(inputs)

        action = pi.detach().numpy().squeeze()

        # ignore gripper at first
        action = action[0:-1]
        # move robot to new position which is old position plus stepsize times the action
        # Orn remains the same throughout
        newPos = currPos + stepSize * action

        rtde_c.moveL(np.hstack((newPos, orn)), 0.2, 0.3)
        time.sleep(0.05)
        actual_newPos = rtde_r.getActualTCPPose()[0:3]
        #actual_newPos = newPos
        x.append(actual_newPos[0])
        y.append(actual_newPos[1])
        print("length of x: ", len(x))

        # fill info
        info["timestep"].append(t)
        info["old_position"].append(currPos)
        info["action"].append(action)
        info["new_position"].append(newPos)
        info["actual_new_position"].append(actual_newPos)
        info["goal"].append(g)
        info["distance_to_goal"].append(np.linalg.norm(obs[0:3]-g))
        info["is_success"].append(np.linalg.norm(
            obs[0:3]-g, ord=2) < goal_threshold)
        info["displacement"].append(
            list(np.asarray(actual_newPos) - np.asarray(currPos)))
        info["real_displacement"].append(stepSize * action)
        # updating position
        real_disp = stepSize * action

        # checking for success
        if info["is_success"][-1] == True:
            print("Success! Norm is {}".format(np.linalg.norm(obs[0:3]-g)))
            rtde_c.stopScript()
            break
        obs[0:3] = actual_newPos
        currPos = newPos  # actual_newPos

        if t % 1 == 0:
            print("*"*20)
            for key in info:
                #print("|{0:>20} : {0:>20}|".format(key,info[key]))
                print(key, info[key][t])
            # time.sleep(3)

        # plotting and setting up plot
        plt.plot(g_robotCF[0], g_robotCF[1], 'o', color="g")
        graph, = plt.plot(x, y, 'o', color="r")
        plt.pause(0.05)

    # start animation
    ani = FuncAnimation(fig, animate, interval=200)
    plt.show()
    print("DONE")
