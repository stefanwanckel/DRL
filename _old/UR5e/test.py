import math
import numpy as np
from gym import spaces
import gym
import pybullet as p

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
    print("This is the LOW: ",LOW)
    return spaces.Box(low=LOW,high=HIGH,dtype=np.float32)


def _reshape_joint_state(robotUid):
    lstJointIndices = range(p.getNumJoints(robotUid))
    jointStates = p.getJointStates(robotUid,lstJointIndices)

    return JointState

a = [1,2,3,4,5,6]
print(a[0:3])