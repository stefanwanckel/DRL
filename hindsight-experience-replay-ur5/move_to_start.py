# imports for robot control
import rtde_control
import rtde_receive
import urkin
import json
import time
import numpy as np
# setup connechiton to robot
rtde_c = rtde_control.RTDEControlInterface("192.168.178.232")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.232")

joint_q = [-0.7866423765765589,
           -1.8796035252013148,
           -1.7409639358520508,
           -1.0964625638774415,
           1.5797905921936035,
           -0.0025427977191370132]
push_joint_q = np.deg2rad(
    np.array([-45.0, -146.3, -112.5, -11.44, 90.38, -0.12]))
# move robot to start configuration
rtde_c.moveJ(joint_q)
time.sleep(1)
data = rtde_r.getActualTCPPose()
print(data)

with open('Reach_TCP_start_pose.json', 'w') as f:
    json.dump(data, f)
