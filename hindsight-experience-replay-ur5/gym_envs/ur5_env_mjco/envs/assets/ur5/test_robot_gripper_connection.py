import mujoco_py
import os
import time
import ur5_env_mjco
import numpy as np

MODEL_XML_PATH = os.path.join("assets", "ur5", "pick_and_place_RG2.xml")


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
joint_delta = -0.2
jointvalues_before = []
jointvalues_after = []
joint_dict = {}


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        a, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


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


timesteps_motion = [1000, 2000, 3000, 4000]


for i in range(5000):
    if i == 0:
        for name in joint_names:
            jointvalues_before.append(sim.data.get_joint_qpos(name))

    if i in timesteps_motion:

        # set gripper action
        pos_ctrl = [0, 0, 0]  # not considered in ctrl_set_action
        rot_ctrl = [0, 0, 0, 0]  # not considered in ctrl_set_action
        gripper_ctrl = [joint_delta, -joint_delta, -joint_delta,
                        joint_delta]  # fingerr, fingerl, armr, arml
        ctrl_action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # ctrl action only consideres the values after the seventh entry corresponging to gripper actuators (ignoring the 7 robot joints )
        ctrl_set_action(sim, ctrl_action)

    if i == 4800:
        for name in joint_names:
            jointvalues_after.append(sim.data.get_joint_qpos(name))

    sim.step()
    viewer.render()


for i, name in enumerate(joint_names):
    joint_dict[name] = (jointvalues_before[i], jointvalues_before[i])

for key in joint_dict.keys():
    print(key, joint_dict[key])
