import os
from gym import utils
from ur5_env_mjco.envs import ur5_env
import numpy as np

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('ur5', 'push_no_gripper.xml')


class Ur5PushEnv(ur5_env.Ur5Env, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        old_initial_qpos = {
            "robot0:shoulder_joint": -0.7866423765765589,
            "robot0:shoulder_lift_joint": -1.8796035252013148,
            "robot0:elbow_joint": -1.7409639358520508,
            "robot0:wrist_1_joint": -1.0964625638774415,
            "robot0:wrist_2_joint": 1.5797905921936035,
            "robot0:wrist_3_joint": -0.0025427977191370132
            # 'object0:joint': [0, 0, 0.4, 1., 0., 0., 1.]
        }
        # push_joint_q = np.deg2rad(
        #     np.array([90.3, -27.6, 150.5, -175.1, 299.4, -49.9]))
        push_joint_q = np.deg2rad(
            np.array([102.2, -42.2, 113.0, -161.5, 270.4, 1.6]))
        initial_qpos = {
            "robot0:shoulder_joint": push_joint_q[0],
            "robot0:shoulder_lift_joint": push_joint_q[1],
            "robot0:elbow_joint": push_joint_q[2],
            "robot0:wrist_1_joint": push_joint_q[3],
            "robot0:wrist_2_joint": push_joint_q[4],
            "robot0:wrist_3_joint": push_joint_q[5]
            # 'object0:joint': [0, 0, 0.4, 1., 0., 0., 1.]
        }
        ur5_env.Ur5Env.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=10,
            gripper_extra_height=0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.2, distance_threshold=0.025,
            initial_qpos=initial_qpos, reward_type=reward_type, table_height=0.4, max_pos_change=0.05, reduced=True)
        utils.EzPickle.__init__(self)
