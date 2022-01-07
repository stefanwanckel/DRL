import mujoco_py
import os
import time
import ur5_env_mjco
import utils
import numpy as np

MODEL_XML_PATH = os.path.join("assets", "ur5", "push_no_gripper.xml")


def set_robot_joints(sim, initial_qpos):
    for name, value in initial_qpos.items():
        sim.data.set_joint_qpos(name, value)
    # utils.reset_mocap_welds(sim)
    sim.forward()


model = mujoco_py.load_model_from_path(MODEL_XML_PATH)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
state = sim.get_state()
push_joint_q = np.deg2rad(
    np.array([90.7, -42.9, 112.1, -159.6, 270.4, -9.9]))
initial_qpos = {
    "robot0:shoulder_joint": push_joint_q[0],
    "robot0:shoulder_lift_joint": push_joint_q[1],
    "robot0:elbow_joint": push_joint_q[2],
    "robot0:wrist_1_joint": push_joint_q[3],
    "robot0:wrist_2_joint": push_joint_q[4],
    "robot0:wrist_3_joint": push_joint_q[5]
}

print(sim.model.joint_names)
print(state)

#shoulderPos = sim.get_body_xpos("")
viewer.cam.distance = 3.5
viewer.cam.azimuth = 132.
viewer.cam.elevation = -10.


set_robot_joints(sim, initial_qpos)
viewer.render()
show = True
while show:
    sim.forward()
