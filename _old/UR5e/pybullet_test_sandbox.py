import pybullet as p
import time
import pybullet_data
import pprint
import numpy as np

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
    if targetOrn is None:
        p.calculateInverseKinematics(
            bodyUniqueID = body,
            endEffectorLinkIndex = eeIndex,
            targetPosition = targetPos)
    else:
        p.calculateInverseKinematics(
            bodyUniqueID = body,
            endEffectorLinkIndex = eeIndex,
            targetPosition = targetPos,
            targetOrientation = targetOrn)

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

lstRevJoints = []
endEffectorIndex = None
checker = False
for index,joint in enumerate(range(p.getNumJoints(ur5eID))):
    currJoint = p.getJointInfo(ur5eID,joint)
    pp.pprint(currJoint[12])
    if str(currJoint[12],'utf-8') == 'ee_link':
        checker = True
    print(checker)
    

pp.pprint(lstRevJoints)
lstJointPos = []
for joint in range(len(lstRevJoints)):
    lstJointPos.append(lstRevJoints[joint][-3])

#Define parameters for motion
goal = [-1,-1,-1,-1,-1,-1]
cartGoal = [1,0,0]
max_force_magn = 100
max_forces = max_force_magn*np.ones((6,1))
max_forces = max_forces.flatten('F')
max_vel_magn = 1
max_vel = max_vel_magn*np.ones((6,1))
max_vel = max_vel.flatten('F')

#Looping in step Simulation
lstJointIndices = range(0,len(lstRevJoints))
#eeIndex = lstJointIndices[-1]

for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)
    #inverse_kinematics(ur5eID,5,cartGoal,targetOrn=None)
    control_joint_pos(ur5eID,lstJointIndices, goal, max_vel,max_forces)
    print(p.getJointStates(ur5eID,lstJointIndices))
     #printing joint pos
    #if i%500==0:
    #  for j in lstJointIndices:
    #         pp.pprint(p.getJointStates(ur5eID,lstJointIndices)[j][0])


p.disconnect()