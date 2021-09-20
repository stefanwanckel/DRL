#imports for robot control
import rtde_control
import rtde_receive
import urkin
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