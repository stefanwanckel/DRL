# imports for robot control
import rtde_control
import rtde_receive
import urkin
import json
import time
import numpy as np
from gripper_control.ur5e_robot import Ur5eRobot
# setup connechiton to robot
robot_IP = "192.168.178.232"
# rtde_c = rtde_control.RTDEControlInterface(R_IP)
# rtde_r = rtde_receive.RTDEReceiveInterface(R_IP)
robot_namespace = "ur5e"
R = Ur5eRobot(robot_namespace, robot_ip=robot_IP, robot_port=50003,
              file="gripper_control/ur5e_rg2_left_calibrated.yaml", base_pose=0)

joint_q = [-0.7866423765765589,
           -1.8796035252013148,
           -1.7409639358520508,
           -1.0964625638774415,
           1.5797905921936035,
           -0.0025427977191370132]
push_joint_q = np.deg2rad(
    np.array([102.2, -42.2, 113.0, -161.5, 270.4, 1.6]))

# move robot to start configuration
# instead of using rtde_c we use the ur5erobot class
# R.setTCPoffset()
#R.moveL_offset([0, 0, -0.1, 0, 0, 0])
R.servo_joint_position(push_joint_q)
# rtde_c.moveJ(joint_q)
time.sleep(1)
