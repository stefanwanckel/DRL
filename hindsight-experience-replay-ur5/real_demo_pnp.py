# general imports
import datetime
import os
import pickle
import random
import time
from collections import OrderedDict
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import ur5_env_mjco
from arguments import get_args
from rl_modules.models import actor
from utils.real_demo_CV import get_goal_position, get_object_position
from utils.model_loader import *
from utils.real_demo_visualization import setup_vis_push_w_image
from utils.real_demo_rg2 import map_2_rg2_action, map_joints_rg2_2_sim
from utils.safety_measures import check_table_collision
from gripper_control.ur5e_robot import Ur5eRobot
from CV.Camera.CameraWrapper import CameraWrapper
import cv2
np.set_printoptions(precision=3, suppress=True)
# imports for robot control
# imports for sim environment and agent
# imports for visualization
ON_REAL_ROBOT = True
folder_name = "test_" 
test_number = 1
fig_path = os.path.join("Results", "pick_and_place")
fig_folder_name = os.path.join(fig_path,folder_name+str(int(time.time())))
data_folder=os.path.join(fig_folder_name,"data")
if not os.path.exists(fig_folder_name):
    os.makedirs(fig_folder_name)
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
# get arguments for demo import model
args = get_args()
env_name = "ur5_pick_and_place_rg2-v1"
project_dir = "definitive"
model_name = '2022-01-14T21:56:50.800343_epoch_57.pt'
# model_path = args.save_dir + args.env_name + '/July30_reach_2_39.pt'
#model_path = args.save_dir + env_name + "/12-12-2021_2_sharpen" + model_name
model_path = os.path.join(args.save_dir,env_name,project_dir, model_name)

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
# rtde_c = rtde_control.RTDEControlInterface("192.168.178.232")
# rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.232")
robot_ip = "192.168.178.232"
config_file_path = os.path.join(
    "gripper_control", "ur5e_rg2_left_calibrated_rg2.yaml")
ur5e_robot = Ur5eRobot("ur5e", robot_ip, 50003, config_file_path, 0)
time.sleep(1)

# set up camera
myCam = CameraWrapper()
myCam.warmup_cam()
# move robot to start configuration
pnp_joint_q = np.deg2rad(
    np.array([-45.1,-127.4,-104.1,-38.7,90.4,-0.1]))
# pnp_joint_q = np.array([-0.7866423765765589,
# -1.8796035252013148,-1.7409639358520508,-1.0964625638774415,1.5797905921936035,-0.0025427977191370132])
#pnp_joint_q = np.deg2rad(
#    np.array([108.4,-66.0,107.5,-132.9,271.1,62.0]))
    #ur5e_robot.servo_joint_position(push_joint_q_start)
gripper_2_tcp_offset = np.array([0,0,-0.225, 0 ,0,0]) #position + orientation
gripper_xy_plane_offset = np.array([0.007,-0.007,0,0,0,0])
gripper_2_tcp_offset =gripper_2_tcp_offset+gripper_xy_plane_offset
TCPpose = np.array(ur5e_robot.receiver.getActualTCPPose())+gripper_2_tcp_offset
startPos = TCPpose[0:3]
orn = TCPpose[3:]

currPos = startPos
# setting parameters for robot motion
stepSize = 0.05
SampleRange = 0.30
goal_threshold = 0.07
# setup figure  limits
axisLimitExtends = 0.20
#goal marker is a list containing the goal for every evaluation episode
goal_marker_ID_lst = [9]
n_timesteps = 50


# Init lists
x = []
y = []
z = []
info = {}
nEvaluations = len(goal_marker_ID_lst)
n_substeps = 1

INFO = OrderedDict()

today = datetime.date.today()
today = today.strftime("%d/%m/%Y")
_time = datetime.datetime.now()
_time = current_time = time.strftime("%H:%M:%S")
date_n_time = today + " " + _time
settings = OrderedDict()
settings["date_n_time"] = date_n_time
settings["env_name"] = env_name
settings["model_name"] = model_name
settings["model_name"] = model_name
settings["model_path"] = model_path
settings["robot_start_pos"] = pnp_joint_q
settings["step_size"] = stepSize
settings["Sample_range"] = SampleRange
settings["goal_threshold"] = goal_threshold
settings["axis_limit_extends"] = axisLimitExtends
settings["n_evaluations"] = nEvaluations
settings["n_timesteps"] = n_timesteps
settings["n_substeps"] = n_substeps
with open(os.path.join(data_folder, 'settings.txt'), 'w') as f:
    for key, val in settings.items():
        f.write(f"{key} : {val}\n")
    print("DONE")


for nTests in range(nEvaluations):
    plt.close()
    goal_marker_ID = goal_marker_ID_lst[nTests]
    ur5e_robot.gripper.open()
    print("Please Place the object. Mind occlusion...")
    #time.sleep(3)
    ur5e_robot.gripper.grip(0.8,0.3, blocking=True, use_depth_compensation=False)
    # Move to start and get TCP pose
    ur5e_robot.servo_joint_position(pnp_joint_q)
    TCPpose = np.array(ur5e_robot.receiver.getActualTCPPose())+gripper_2_tcp_offset
    startPos = TCPpose[0:3]
    orn = TCPpose[3:]
    # reset position
    currPos = startPos
    # reset x and y
    x = []
    y = []
    z = []
    x_o = []
    y_o = []
    z_o = []


    
    observation = env.reset()
    obs_sim = observation['observation']
    obs_sim[9:] = np.zeros(10)
    # observation must be robot pos,object_pos and robot-object rel pos
    object_marker_ID = 6
    object_pos, img = get_object_position(myCam, object_marker_ID)
    object_pos = object_pos[:3]
    object_rel_pos = object_pos - startPos
    object_velp = np.zeros(3)
    grip_velp = np.zeros(3)
     #trying out a little hack to fix a mistake in training
    x.append(startPos[0])
    y.append(startPos[1])
    z.append(startPos[2])
    x_o.append(object_pos[0])
    y_o.append(object_pos[1])
    z_o.append(object_pos[2])

    """
    obs[9:10] need to be right moment arm joint and left moment arm joint respectively
    gripper_state = np.array(ur5e_robot.receiver.getActualQ())[-2:]
    #gripper_state =np.zeros(2)
    """

    gripper_state = np.array(ur5e_robot.gripper.get_joint_states()["l_moment_arm_joint"]["joint_position"])
    gripper_state = map_joints_rg2_2_sim(gripper_state)
    gripper_state = np.array([gripper_state,gripper_state])
    gripper_vel = np.zeros(2)
    obs = np.hstack((startPos, object_pos, object_rel_pos,
                    gripper_state, object_velp, grip_velp, gripper_vel))
    # difference between sim and real world coordinate frames (mit marcus)
    obs_diff = np.zeros(19)
    obs_diff[0:3] = obs_sim[0:3]-obs[0:3]
    obs_diff[3:6] = obs_sim[0:3]-obs[0:3]
    obs += obs_diff
    
    print('Please show the goal marker...')
    #time.sleep(5)
    goal = np.array(get_goal_position(myCam, goal_marker_ID)[:3])
    #goal = goal + np.array([0,0,0.25])#+ np.array(gripper_2_tcp_offset[:3])
    print('Goal is: ', goal)
    #goal = 0.001*np.array([67.8,-592.8,187.6])+np.array(gripper_2_tcp_offset[:3])
    # in real life we just place the goal somewhere and no sample is required
    g_robotCF = goal
    g = goal + obs_diff[:3]
    # plotting and setting up plot
    # setup figure
    fig, axs, ax1, ax2, ax_img = setup_vis_push_w_image(nTests, startPos, SampleRange,
                                                          axisLimitExtends, g_robotCF, goal_threshold)

    
    plt.savefig(os.path.join(fig_folder_name, "plane_view_episode_"+str(nTests)+ '_timestep_' + str(00) + ".svg"))
    cv2.imwrite(os.path.join(fig_folder_name, "img_"+str(nTests)+ '_timestep_' + str(00) + ".png"), img)

    # logging
    info["timestep"] = []
    info["obs"] = []
    info["old_position"] = []
    info["action"] = []
    info["rg2_action"] = []
    info["new_position"] = []
    info["actual_new_position"] = []
    info["goal"] = []
    info["distance_to_goal"] = []
    info["is_success"] = []
    info["displacement"] = []
    info["real_displacement"] = []
    info["object_pos"] = []

    # for t in range(env._max_episode_steps):
    fig_list= []
    # opacity_arr = np.linspace(1, 0.1, 20)
    # opacity_arr = np.concatenate([opacity_arr, np.ones(n_timesteps - opacity_arr.shape[0]) * 0.1])

    for t in range(1, n_timesteps+1):
        # exit this for loop. THis is necessary due to the
        # substeps condition exiting the substep loop but then continuing in the t range
        if len(info["is_success"]) > 0:
            if info["is_success"][-1]:
                break
        # add coordinate frame diff (mit marcus)
        inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
        with torch.no_grad():
            pi = actor_network(inputs)

        action = pi.detach().numpy().squeeze()

        # ignore gripper at first     
        gripper_action= action[-1]
        
        action = action[0:-1]
        # move robot to new position which is old position plus stepsize times the action
        # Orn remains the same throughout
        for substeps in range(n_substeps):
            newPos = currPos + stepSize * action
            if ON_REAL_ROBOT:

                relative_action = stepSize * action
                relative_action = check_table_collision(
                    currPos, relative_action)
                #move TCP linearly
                ur5e_robot.moveL_offset(np.hstack((relative_action, [0, 0, 0])))

                #move gripper
                rg2_action = map_2_rg2_action(gripper_action, ur5e_robot)
                print('rg2_action', rg2_action)
                rel_force = 0.15
                ur5e_robot.gripper.grip(rg2_action, rel_force, blocking=True, use_depth_compensation=False)

                time.sleep(0.05)
                actual_newPos = np.array(
                    ur5e_robot.receiver.getActualTCPPose()[0:3])+gripper_2_tcp_offset[0:3]
            else:
                actual_newPos = newPos

            x.append(actual_newPos[0])
            y.append(actual_newPos[1])
            z.append(actual_newPos[2])
            x_o.append(object_pos[0])
            y_o.append(object_pos[1])
            z_o.append(object_pos[2])

            # fill info
            g_robotCF_np = np.array(g_robotCF)
            info["timestep"].append(t)
            info["obs"].append(obs)
            info["old_position"].append(currPos)
            info["action"].append(action)
            info["rg2_action"].append(rg2_action)
            info["new_position"].append(newPos)
            info["actual_new_position"].append(actual_newPos)
            info["goal"].append(g)
            info["distance_to_goal"].append(
                np.linalg.norm(actual_newPos-g_robotCF_np))
            info["is_success"].append(np.linalg.norm(
                object_pos-g_robotCF) < goal_threshold)
            info["displacement"].append(
                list(np.asarray(actual_newPos) - np.asarray(currPos)))
            info["real_displacement"].append(stepSize * action)
            info["object_pos"].append(object_pos)
            print("*"*20)
            for key in info.keys():
                print(key,info[key][-1])
            print("*"*20)
            # updating observation
            object_marker_ID = 6
            currPos = actual_newPos  # new_Pos
            grip_pos = actual_newPos
            object_pos, img = get_object_position(myCam, object_marker_ID)
            object_pos = object_pos[:3]#+gripper_2_tcp_offset[:3]
            object_rel_pos = object_pos - actual_newPos
            object_velp = np.zeros(3)
            grip_velp = np.zeros(3)
            gripper_vel = np.zeros(2)
            gripper_state = np.array(ur5e_robot.gripper.get_joint_states()["l_moment_arm_joint"]["joint_position"])
            gripper_state = map_joints_rg2_2_sim(gripper_state)
            gripper_state = np.array([gripper_state,gripper_state])
            obs = np.hstack((grip_pos, object_pos, object_rel_pos,
                            gripper_state, object_velp, grip_velp, gripper_vel))
            obs += obs_diff
            print('obj rel position', object_rel_pos)
            # printing some episode-specific information
            #if t % 1 == 0:
                #print("*"*20)
                #for key in info:
                    # print("|{0:>20} : {0:>20}|".format(key,info[key]))
                    #print(key, info[key][t])

            # plotting and setting up plot
            ax1.plot(x, y, 'o', color="r", markersize=4)
            ax2.plot(x, z, 'o', color="r", markersize=4)
            ax1.plot(x_o, y_o, 'o', color="blue", alpha=0.2, markersize=4)
            ax2.plot(x_o, z_o, 'o', color="blue", alpha=0.2,  markersize=4)
            ax_img.imshow(img)
            
            plt.pause(0.05)
            
            plt.savefig(os.path.join(fig_folder_name, "plane_view_episode_"+str(nTests)+ '_timestep_' + str(t).zfill(2) + ".svg"))
            cv2.imwrite(os.path.join(fig_folder_name, "img_"+str(nTests)+ '_timestep_' + str(t).zfill(2) + ".png"), img)
            #if info["is_success"][-1] == True or t == n_timesteps-1:
                #curr_plt.savefig(os.path.join(fig_path, "plane_view_episode_"+str(nTests)+ '_timestep_' + str(i) + ".svg"))
            
            # checking for success
            if info["is_success"][-1] == True:
                print("Success! Norm is {}".format(
                    np.linalg.norm(object_pos-g_robotCF)))
                # saving plots

                
                if ON_REAL_ROBOT and nTests == range(nEvaluations)[-1]:
                    #ur5e_robot.controller.stopScript()
                    myCam.stop_cam()
                    plt.close() 
            INFO[nTests] = info      
            #info_path = os.path.join("Results", "pick_and_place")
            with open(os.path.join(data_folder, 'INFO.pkl'), 'wb') as f:
                pickle.dump(INFO, f)

            Plotting_data = {}
            Plotting_data["x"] = x
            Plotting_data["y"] = y
            Plotting_data["z"] = z
            Plotting_data["x_0"] = x_o
            Plotting_data["y_0"] = y_o
            Plotting_data["x_o"] = z_o
            with open(os.path.join(data_folder, 'Plotting_data.pkl'), 'wb') as f:
                pickle.dump(Plotting_data, f)


