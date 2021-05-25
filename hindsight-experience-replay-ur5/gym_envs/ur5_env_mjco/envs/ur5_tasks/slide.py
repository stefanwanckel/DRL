import os
import numpy as np

from gym import utils
from ur5_env_mjco.envs import ur5_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('ur5', 'slide.xml')


class Ur5SlideEnv(ur5_env.Ur5Env, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'object0:joint': [0.6, 0., 0.4, 1., 0., 0., 0.],
        }
        ur5_env.Ur5Env.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
