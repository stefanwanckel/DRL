import mujoco_py
import os
import time
import ur5_env_mjco
import numpy as np

MODEL_XML_PATH = os.path.join("assets", "ur5", "robot_RG2_no_truss.xml")


model = mujoco_py.load_model_from_path(MODEL_XML_PATH)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
state = sim.get_state()

# shoulderPos = sim.get_body_xpos("")
viewer.cam.distance = 1
viewer.cam.azimuth = 132.
viewer.cam.elevation = -10.
joint_names = sim.model.joint_names

show = True
i = 0
alpha = 1
joint_delta = 0.3
jointvalues_before = []
jointvalues_after = []
joint_dict = {}


def gripper_close(joint_delta, sim):
    new_rma_qpos = sim.data.get_joint_qpos(
        "right_moment_arm_joint")+joint_delta
    new_lma_qpos = sim.data.get_joint_qpos("left_moment_arm_joint")-joint_delta
    sim.data.set_joint_qpos('right_moment_arm_joint', new_rma_qpos)
    sim.data.set_joint_qpos('left_moment_arm_joint',  new_lma_qpos)

    new_rfj_qpos = sim.data.get_joint_qpos("r_finger_joint")-alpha*joint_delta
    new_lfj_qpos = sim.data.get_joint_qpos("l_finger_joint")+alpha*joint_delta
    sim.data.set_joint_qpos('r_finger_joint', new_rfj_qpos)
    sim.data.set_joint_qpos('l_finger_joint',  new_lfj_qpos)


timesteps_motion = [100]
for i in range(5000):
    if i == 0:
        for name in joint_names:
            jointvalues_before.append(sim.data.get_joint_qpos(name))

    if i in timesteps_motion:
        new_rma_qpos = sim.data.get_joint_qpos(
            "right_moment_arm_joint")+joint_delta
        new_lma_qpos = sim.data.get_joint_qpos(
            "left_moment_arm_joint")-joint_delta
        sim.data.set_joint_qpos('right_moment_arm_joint', new_rma_qpos)
        sim.data.set_joint_qpos('left_moment_arm_joint',  new_lma_qpos)

        # new_rfp_qpos = sim.data.get_joint_qpos(
        #     "r_finger_passive_joint")-joint_delta
        # new_lfp_qpos = sim.data.get_joint_qpos(
        #     "l_finger_passive_joint")+joint_delta

        # sim.data.set_joint_qpos('r_finger_passive_joint', new_rfp_qpos)
        # sim.data.set_joint_qpos('l_finger_passive_joint', new_lfp_qpos)

        print(sim.data.get_joint_qpos("r_finger_joint"))
        new_rfj_qpos = sim.data.get_joint_qpos(
            "r_finger_joint")-alpha*joint_delta
        new_lfj_qpos = sim.data.get_joint_qpos(
            "l_finger_joint")+alpha*joint_delta
        sim.data.set_joint_qpos('r_finger_joint', new_rfj_qpos)
        sim.data.set_joint_qpos('l_finger_joint',  new_lfj_qpos)
        print(sim.data.get_joint_qpos("r_finger_joint"))

    if i == 4800:
        for name in joint_names:
            jointvalues_after.append(sim.data.get_joint_qpos(name))

    sim.step()
    viewer.render()


for i, name in enumerate(joint_names):
    joint_dict[name] = (jointvalues_before[i], jointvalues_before[i])

for key in joint_dict.keys():
    print(key, joint_dict[key])
