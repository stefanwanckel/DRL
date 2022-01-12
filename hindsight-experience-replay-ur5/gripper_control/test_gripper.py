
from ur5e_robot import Ur5eRobot
import os
import time
# rtde_c = rtde_control.RTDEControlInterface("192.168.178.232")
# rtde_r = rtde_receive.RTDEReceiveInterface("192.168.178.232")
robot_ip = "192.168.178.232"
config_file_path = "ur5e_rg2_left_calibrated_rg2.yaml"
ur5e_robot = Ur5eRobot("ur5e", robot_ip, 50003, config_file_path, 0)
time.sleep(1)
print(ur5e_robot.receiver.getActualTCPPose())

ur5e_robot.close_gripper()

# print(ur5e_robot.gripper.get_gripper_width())
# print(ur5e_robot.gripper.get_joint_states())
#ur5e_robot.gripper.grip(0.6, 1, blocking=True, use_depth_compensation=False)
# ur5e_robot.close_gripper()
