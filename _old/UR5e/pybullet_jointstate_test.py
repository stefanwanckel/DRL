import pybullet as p
import time
import pybullet_data
import pprint
import numpy as np
import matplotlib.pyplot as plt

def control_joint_pos(body, joints, target, max_vel=None, max_force=None):
    if max_vel is None:
        p.setJointMotorControlArray(
                bodyIndex=body,
                jointIndices=joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target,
                forces=max_force)
    else:
        p.setJointMotorControlArray(
                bodyIndex=body,
                jointIndices=joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target,
                targetVelocities=max_vel,
                forces=max_force) 

def inverse_kinematics(body,eeIndex,targetPos,targetOrn=None):
    jointVector = []
    if targetOrn is None:
        jointVector = p.calculateInverseKinematics(
            bodyIndex = body,
            #bodyUniqueID = body,
            endEffectorLinkIndex = eeIndex,
            targetPosition = targetPos)
    else:
        jointVector = p.calculateInverseKinematics(
            bodyUniqueID = body,
            endEffectorLinkIndex = eeIndex,
            targetPosition = targetPos,
            targetOrientation = targetOrn)

    return jointVector

pp = pprint.PrettyPrinter(indent=4)

#Connecting to engine and loading models
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version or GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
ur5eStartPos = [0,0,1]
ur5eStartOrientation = p.getQuaternionFromEuler([0,0,0])
ur5eID = p.loadURDF("/home/stefan/Documents/Masterarbeit/UR5e/ur_e_description/urdf/ur5e.urdf",ur5eStartPos, ur5eStartOrientation)
#Getting info on ur5e to print
#get a list of all revolute joints
# ur5eID
lstRevJoints = []
lstRevJointsIndices = []
endEffectorIndex = None
for index,joint in enumerate(range(p.getNumJoints(ur5eID))):
    if p.getJointInfo(ur5eID,joint)[2] == 0:
        lstRevJoints.append(p.getJointInfo(ur5eID,joint))
        lstRevJointsIndices.append(index)
    if str(p.getJointInfo(ur5eID,joint)[12],'utf-8') == "ee_link":
        endEffectorIndex = index
print("revolute indices: ", lstRevJointsIndices)
print("lstRevJoints[lstRevJointsIndices]: ",lstRevJoints[lstRevJointsIndices])
#print("endEffectorIndex= ", endEffectorIndex)
# lstJointPos = []
# for joint in range(len(lstRevJoints)):
#     lstJointPos.append(lstRevJoints[joint][-3])
# a = p.getJointStates(ur5eID,range(p.getNumJoints(ur5eID)))
# a = np.array(a)
# b = a[:,0:2]
# print(a)
# print("this is b: ", b)
#Define parameters for motion
#goal = [-1,-1,-1,-1,-1,-1]
cartGoal = [0.25,-0.25,0.7]
#Provide max forces for every joint
max_force_magn = 500
max_forces = max_force_magn*np.ones((6,1))
max_forces = max_forces.flatten('F')
max_vel_magn = 0.1
max_vel = max_vel_magn*np.ones((6,1))
max_vel = max_vel.flatten('F')

# #Looping in step Simulation
# eeIndex = endEffectorIndex
# JointVector = p.calculateInverseKinematics(ur5eID,eeIndex,cartGoal,targetOrientation=None)
# arrJointVector = np.array(JointVector)
# error = []


# for i in range(150):
#     if i%10==0:
#         #print("i = ", i," : ",lstRevJoints[-1])
#         print("*******")
#         #print(p.getJointStates(ur5eID,lstRevJointsIndices))
#         print(p.getJointState(ur5eID,lstRevJointsIndices[-1]))
#         print("*******")
#     p.stepSimulation()
#     time.sleep(1./50.)
#     control_joint_pos(ur5eID,lstRevJointsIndices, JointVector, max_vel,max_forces)
# p.disconnect()
