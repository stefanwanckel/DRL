#general imports
import time
import math
import numpy as np
import random
np.set_printoptions(precision=2,suppress=True)
#imports for robot control
import rtde_control
import rtde_receive
import urkin
#imports for sim environment and agent
import torch
from numpy.lib.function_base import _average_dispatcher
from rl_modules.models import actor
from arguments import get_args
import gym
import ur5_env_mjco
#imports for visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import seaborn as sns

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def animate(i):
    graph.set_data(x[:i+1], y[:i+1])
    return graph

#get arguments for demo import model
args = get_args()
model_path = args.save_dir + args.env_name + '/July30_reach_2_39.pt'
o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
#create gym environment and get first observation
env = gym.make(args.env_name)
observation = env.reset()
#get environment parameters
env_params = {'obs': observation['observation'].shape[0], 
            'goal': observation['desired_goal'].shape[0], 
            'action': env.action_space.shape[0], 
            'action_max': env.action_space.high[0],
            }
#setup agent with loaded model
actor_network = actor(env_params)
actor_network.load_state_dict(model)
actor_network.eval()

#setup conneciton to robot
rtde_c = rtde_control.RTDEControlInterface("192.168.178.15")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.15")

joint_q = [-0.7866423765765589,
        -1.8796035252013148,
        -1.7409639358520508, 
        -1.0964625638774415,
        1.5797905921936035, 
        -0.0025427977191370132]

#move robot to start configuration
rtde_c.moveJ(joint_q)

#get TCP pose
TCPpose = rtde_r.getActualTCPPose()
startPos = TCPpose[0:3]
orn = TCPpose[3:]
print("the starting position is {} \nThe starting orientation is{}".format(startPos,orn))
currPos = startPos
#setting parameters for robot motion
stepSize = 0.01
SampleRange = 0.15

#setup figure and limits
axisLimitExtends = 0.10
#sns.set_theme()
fig = plt.figure()
ax = fig.gca()
plt.axis("equal")
plt.xlim(startPos[0]-SampleRange-axisLimitExtends, startPos[0]+SampleRange+axisLimitExtends)
plt.ylim(startPos[1]-SampleRange-axisLimitExtends, startPos[1]+SampleRange+axisLimitExtends)
plt.title("X-Y TCP")
plt.xlabel("X-axis [m]")
plt.ylabel("Y-axis [m]")
#Init lists
x= []
y= []
info = {}
for nTests in range(1):
    #observation = env.reset()
    #observation must be robot pos
    obs = np.hstack((startPos, np.zeros(7)))

    #sampling goal
    
    rndDisp =-SampleRange+ 2*SampleRange*np.random.random(3)
    g = list(np.asarray(startPos) + rndDisp)
    
    #plotting and setting up plot
    plt.plot(g[0],g[1],'o',color="g")
    graph, = plt.plot([], [], 'o',color="r")
    sampleSpace = Circle((startPos[0],startPos[1]), radius=SampleRange,fill=False)
    successThreshold = Circle((g[0],g[1]), radius=0.05,fill=False,ls="--")
    ax.add_patch(sampleSpace)
    ax.add_patch(successThreshold)

    #logging
    info["timestep"] = []
    info["old_position"] = []
    info["action"] = []
    info["new_position"] = []
    info["actual_new_position"] = []
    info["goal"] = []
    info["distance_to_goal"] = []
    info["is_success"] = []
    info["displacement"] = []
    n_substeps = range(10)

    #init tmp vars
    newPos = actual_newPos = np.zeros(3)

    for t in range(env._max_episode_steps):
        inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
        # inputs = np.concatenate([obs,g])
        # inputs = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
                pi = actor_network(inputs)
        action = pi.detach().numpy().squeeze()
        #ignore gripper at first
        action = action[0:-1]
        #move robot to new position which is old position plus stepsize times the action
        #Orn remains the same throughout
        for substeps in n_substeps:
            newPos = currPos + stepSize * action
            time.sleep(0.1)
            rtde_c.moveL(np.hstack((newPos,orn)),0.2,0.3)
            time.sleep(0.1)
            actual_newPos = rtde_r.getActualTCPPose()[0:3]
            time.sleep(0.1)
            x.append(actual_newPos[0])
            y.append(actual_newPos[1])
            currPos = actual_newPos
        #print("length of x: ",len(x))
        #fill info
        info["timestep"].append(t)
        info["old_position"].append(currPos)
        info["action"].append(action)
        info["new_position"].append(newPos)
        info["actual_new_position"].append(actual_newPos)
        info["goal"].append(g)
        info["distance_to_goal"].append(np.linalg.norm(obs[0:3]-g))
        info["is_success"].append(np.linalg.norm(obs[0:3]-g,ord=2)<0.05)
        info["displacement"].append(list(np.asarray(actual_newPos) - np.asarray(currPos)))
        #updating position
        actual_newPos = rtde_r.getActualTCPPose()[0:3]
        obs[0:3] = actual_newPos
        
        #checking for success
        if np.linalg.norm(obs[0:3]-g,ord=2)<0.05:
            rtde_c.stopScript()
            print("Success! Norm is {}".format(np.linalg.norm(obs[0:3]-g)))
            break

        if t%2==0:
            print("*"*20)
            for key in info:
                #print("|{0:>20} : {0:>20}|".format(key,info[key]))
                print(key,info[key][t])
            #time.sleep(3)
        


#start animation
ani = FuncAnimation(fig, animate, interval=200)   
plt.show()