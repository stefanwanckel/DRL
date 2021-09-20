import rtde_control
import time
import rtde_receive
import numpy as np
import math
import urkin
rtde_c = rtde_control.RTDEControlInterface("192.168.178.15")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.15")
np.set_printoptions(precision=2,suppress=True)
import time
# # Parameters
# velocity = 0.5
# acceleration = 0.4
# dt = 1.0/500  # 2ms
# lookahead_time = 0.1
# gain = 300
joint_q = [-0.7866423765765589,
        -1.8796035252013148,
        -1.7409639358520508, 
        -1.0964625638774415,
        1.5797905921936035, 
        -0.0025427977191370132]

print(rtde_r.getActualQ())
#rtde_c.moveJ(bestSol)
# Move to initial joint position with a regular moveJ
rtde_c.moveJ(joint_q)
dx = 0.1
dy = 0.1
rtde_c.moveL([0.31436597646071746+dx, -0.5025529103443298+dy, 0.2881632357253061, -1.1484624048322778, 2.917830206626033, 0.0034119167505656076], 0.5, 0.3)


#rtde_c.servoStop()
rtde_c.stopScript()
