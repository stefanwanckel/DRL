import os
from gym import utils
from ur5_env_mjco.envs import ur5_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('ur5', 'push_no_gripper.xml')


class Ur5PushEnv(ur5_env.Ur5Env, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            "robot0:shoulder_joint": -0.7866423765765589,
            "robot0:shoulder_lift_joint": -1.8796035252013148,
            "robot0:elbow_joint": -1.7409639358520508,
            "robot0:wrist_1_joint": -1.0964625638774415,
            "robot0:wrist_2_joint": 1.5797905921936035,
            "robot0:wrist_3_joint": -0.0025427977191370132
            # 'object0:joint': [0, 0, 0.4, 1., 0., 0., 1.]
        }
        ur5_env.Ur5Env.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.2, distance_threshold=0.025,
            initial_qpos=initial_qpos, reward_type=reward_type, table_height=0.4, max_pos_change=0.025, reduced=True)
        utils.EzPickle.__init__(self)
