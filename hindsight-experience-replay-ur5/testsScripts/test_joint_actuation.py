import mujoco_py
import os
import time
import ur5_env_mjco
import utils
import numpy as np

MODEL_XML_PATH = os.path.join("assets","ur5", "reach.xml")

model = mujoco_py.load_model_from_path(MODEL_XML_PATH)
sim = mujoco_py.MjSim(model)
viewer=mujoco_py.MjViewer(sim)
state = sim.get_state()
jointNames = sim.model.joint_names
# 0:'robot0:shoulder_joint'
# 1:'robot0:shoulder_lift_joint'
# 2:'robot0:elbow_joint'
# 3:'robot0:wrist_1_joint'
# 4:'robot0:wrist_2_joint'
# 5:'robot0:wrist_3_joint'
# 6:'robot0:r_gripper_finger_joint'
# 7:'robot0:l_gripper_finger_joint'
viewer.render()
utils.reset_mocap_welds(sim)
sim.forward()
gripper_target = sim.data.get_site_xpos('robot0:grip')
gripper_rotation = np.array([1., 0., 1., 0.])
sim.data.set_mocap_pos('robot0:mocap', gripper_target)
sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
for _ in range(100000):
    sim.step()


#for _ in range(10000) :