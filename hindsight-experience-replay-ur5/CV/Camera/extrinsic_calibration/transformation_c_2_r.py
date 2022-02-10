import pickle
import numpy as np
import os
import sys
# from utils.real_demo_CV import get_goal_position, get_object_position
# from CV.Camera.CameraWrapper import CameraWrapper

from CV.Camera.test_scripts.utils import *

_rg2=False
r_pos_1 = 0.001*np.array([336.2,-607.3,-231.5])
r_pos_2 = 0.001*np.array([541.1,-389.4,-231.2])
r_pos_4 = 0.001*np.array([138.1,-396.0,-233.8])
if _rg2:
    z_offset = np.array([0,0,-225])*0.001
    r_pos_1 =r_pos_1+z_offset
    r_pos_2 =r_pos_2+z_offset
    r_pos_4 =r_pos_4+z_offset
#r_pos_5 = 0.001*np.array([209.7, -259.4, -675.1])
lst_r_pos = np.array([r_pos_1, r_pos_2, r_pos_4])

marker_transformation = pickle.load(
    open("./marker_transformation.p", "rb"))
# extract t_vec

c_pos_1 = marker_transformation[1][:3, 3]
c_pos_2 = marker_transformation[2][:3, 3]
c_pos_4 = marker_transformation[4][:3, 3]
#c_pos_5 = marker_transformation[5][:, 3]
# remove affine 1

lst_c_pos = np.array([c_pos_1, c_pos_2, c_pos_4])

T_c_2_r = recover_homogenous_affine_transformation(lst_c_pos, lst_r_pos)

#T_c_2_r = T_c_2_r.T
example_point = np.array([0.008, 0.128, 0.744])
example_point_affine = np.concatenate((example_point, [1]))
r_pos_1_prime = np.dot(T_c_2_r.T, example_point_affine)

np.save("Camera_to_robot_transformation", T_c_2_r)
print("Camera_to_robot_transformation\n", T_c_2_r)
print("Sample POint:\n", r_pos_1_prime)
print("Done")

# 361, -493, -191

if __name__=='__main__':
    myCam = CameraWrapper()
    myCam.warmup_cam()

    obs,_ = get_object_position(myCam, 6)
    print('OBJECT: ', obs)