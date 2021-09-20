# imports for robot control
import rtde_control
import rtde_receive
import urkin
import json
import time
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
time.sleep(1)
data = rtde_r.getActualTCPPose()
print(data)

with open('Reach_TCP_start_pose.json', 'w') as f:
    json.dump(data, f)
