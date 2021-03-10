import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

class ur5eEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.actionSpace = _make_action_space()
        self.observationSpace = _make_observation_space()
        self.ur5eUid = p.loadURDF("../resources/ur_e_description/urdf/ur5e.urdf")
        #self.objectUid = p.loadURDF("../resources/sphere_small/sphere_small.urdf",basePosition=state_object)
    
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING)
        posVel = _reshape_joint_states(self.ur5eUid)
        currPos = posVel[0:6]
        stepSize = 0.0001
        lstRevJointsIndices, endEffectorIndex = _get_revolute_joints_indices(self.ur5eUid)
        NewPosition = []
        for qi in action:
            NewPosition.append(currPos[qi]+action[qi]*stepSize)
        if not len(NewPosition)==6:
            print("error NewPosition has wrong length")
                
        #Motion and step
        p.setJointMotorControlArray(self.ur5eUid,lstRevJointsIndices,p.POSITION_CONTROL,NewPosition)
        p.stepSimulation()
        #reward and observation returns

        cartPosOrnEef = _get_cartesian_pos_and_orn(self.ur5eUid,endEffectorIndex)
        cartPosEef = cartPosOrnEef[0:3]
        SpherePosition = p.getBasePositionandOrientation(self.objectUid)[0]

        if np.linalg.norm(cartPosEef-SpherePosition)<0.1:
            done = True
            reward = 1
        else:
            done = False
            reward = -1
        eefCartPosOrn = _get_cartesian_pos_and_orn(self.ur5eUid)
        posVel = _reshape_joint_states(self.ur5eUid)
        observation =np.concatenate((eefCartPosOrn,posVel))
        
        return observation, reward, done


    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        p.setGravity(0,0,-10)
        #load plane
        planeUid = p.loadURDF("plane.urdf")
        #load robot and reset pose
        self.ur5eUid = p.loadURDF("../resources/ur_e_description/urdf/ur5e.urdf")
        #starting pose in joint space
        rest_pose = np.array([0.1,-0.2,0.3,0.1,-0.2,0.05])
        for i in range(6):
           p.resetJointState(self.ur5eUid,i, rest_pose[i])
        #load goal object as a small sphere on random point on xy plane at 0.05 height
        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        self.objectUid = p.loadURDF("../resources/sphere_small/sphere_small.urdf",basePosition=state_object)
        jointPosVel = _reshape_joint_states(ur5eUid)
        eeffCartPosOrn = _get_cartesian_pos_and_orn(ur5eUid,7)
        observation = np.concatenate((jointPosVel,eeffCartPosOrn))

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # render again
        return observation
        
    def render(self, mode='human', close=False):
        return
    def close(self):
        p.disconnect()


#Action space is made from joint position
def _make_action_space():
    jointLimit = 1

    LOW = -jointLimit * np.ones(6)
    HIGH = jointLimit * np.ones(6)
    return spaces.Box(low=LOW,high=HIGH)
#Observation spaces is made from joint position, velocities, cartesian Position and Orientation
def _make_observation_space():
    jointLimit = 2 * math.pi
    jointPosLow = -jointLimit * np.ones(6)
    jointPosHigh = jointLimit * np.ones(6)
    jointVelLow = -jointLimit * np.ones(6)
    jointVelHigh = jointLimit * np.ones(6)
    maxRange = 1.5
    cartPosLow = -maxRange * np.ones(3)
    cartPosHigh = maxRange * np.ones(3)
    ornLimit = 1
    ornLow = -ornLimit * np.ones(3)
    ornHigh = ornLimit * np.ones(3)

    LOW = np.concatenate((jointPosLow,jointVelLow,cartPosLow,ornLow),axis= 0)
    HIGH = np.concatenate((jointPosHigh,jointVelHigh,cartPosHigh,ornHigh),axis= 0)
    return spaces.Box(low=LOW,high=HIGH)

#gets joint angle and velocity from jointstate
def _reshape_joint_states(robotUid):
    lstRevJointsIndices, endEffectorIndex = _get_revolute_joints_indices(robotUid)
    jointStates = p.getJointStates(robotUid,range(p.getNumJoints(robotUid)))
    pos = []
    vel = []
    for rJindex in lstRevJointsIndices:
        currentJointState = jointStates[rJindex]
        pos.append(currentJointState[0])
        vel.append(currentJointState[1])
    pos = np.array(pos)
    vel = np.array(pos)
    posVel = np.concatenate((pos,vel))
    return posVel

def _get_cartesian_pos_and_orn(robotUid,endEffectorIndex):
    linkState = p.getLinkState(robotUid,endEffectorIndex,computeForwardKinematics=True)
    cartPos = np.array(linkState[0])
    cartOrn = np.array(linkState[1])
    cartPosOrn = np.concatenate((cartPos,cartOrn))
    return cartPosOrn

#Returns the Joint info on revolute joints and the end effector index
def _get_revolute_joints_indices(robotUid):
    lstRevJointsIndices = []
    for index,joint in enumerate(range(p.getNumJoints(robotUid))):
        if p.getJointInfo(robotUid,joint)[2] == 0:
            lstRevJointsIndices.append(index)
        if str(p.getJointInfo(robotUid,joint)[12],'utf-8') == "ee_link":
            endEffectorIndex = index
    print(lstRevJointsIndices)
    return lstRevJointsIndices, endEffectorIndex

env = gym.make('Ur5eMove-v0')
