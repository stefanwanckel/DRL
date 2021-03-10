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
physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version or GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.82)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
ur5eID = p.loadURDF("/home/stefan/Documents/Masterarbeit/UR5e/ur_e_description/urdf/ur5e.urdf",cubeStartPos, cubeStartOrientation)
#Getting info on ur5e to print
#get a list of all revolute joints
lstRevJoints = []
lstRevJointsIndices = []
endEffectorIndex = None
for index,joint in enumerate(range(p.getNumJoints(ur5eID))):
    if p.getJointInfo(ur5eID,joint)[2] == 0:
        lstRevJoints.append(p.getJointInfo(ur5eID,joint))
        lstRevJointsIndices.append(index)
    if str(p.getJointInfo(ur5eID,joint)[12],'utf-8') == "ee_link":
        endEffectorIndex = index
print("endEffectorIndex= ", endEffectorIndex)
lstJointPos = []
for joint in range(len(lstRevJoints)):
    lstJointPos.append(lstRevJoints[joint][-3])

#Define parameters for motion
#goal = [-1,-1,-1,-1,-1,-1]
cartGoal = [0.5,0.5,1.8]
max_force_magn = range(20,600,10)
for force in max_force_magn:
    for count,jointIndex in enumerate(lstRevJointsIndices):
        p.resetJointState(ur5eID,jointIndex,lstJointPos[count])
    max_forces = max_force_magn[force]*np.ones((6,1))
    max_forces = max_forces.flatten('F')
    max_vel_magn = 0.1
    max_vel = max_vel_magn*np.ones((6,1))
    max_vel = max_vel.flatten('F')

    #Looping in step Simulation
    eeIndex = endEffectorIndex
    JointVector = p.calculateInverseKinematics(ur5eID,eeIndex,cartGoal,targetOrientation=None)
    arrJointVector = np.array(JointVector)
    error = []

    for i in range(1000):
        CurrentPos = []
        p.stepSimulation()
        time.sleep(1./240.)
        control_joint_pos(ur5eID,lstRevJointsIndices, JointVector, max_vel,max_forces)
        #printing joint pos
        #if i%50==0:
        s = p.getJointStates(ur5eID,lstRevJointsIndices)
        for lst in s:
            CurrentPos.append(lst[0])
        arrCurrentPos = np.array(CurrentPos)
        error.append(arrJointVector - arrCurrentPos)


    error = np.array(error)
    jointPlot = error[:,0]
    for joint in range(error.shape[1]):
        plt.plot(range(error.shape[0]),error[:,joint],label="joint_"+str(joint))
        plt.title("InvKin to " + str(cartGoal) + ". Max_force = " + str(max_force_magn))
    plt.legend()
    plt.savefig("./invkin/invkin_maxForce_"+str(max_force_magn[force]))
p.disconnect()