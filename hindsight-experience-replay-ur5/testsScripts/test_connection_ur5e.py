import math
import numpy as np
import rtde_control
import rtde_receive

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.15")
actual_q = np.array(rtde_r.getActualQ())*180/(2*math.pi)
rtde_c = rtde_control.RTDEControlInterface("192.168.178.15")
modified_q = actual_q
modified_q[0] = 0
from rtde_control import RTDEControlInterface as RTDEControl
rtde_c = RTDEControl("192.168.178.15", RTDEControl.FLAG_USE_EXT_UR_CAP)
rtde_c.moveL(modified_q, 0.1, 0.1)
print(actual_q)
