import os
from gym import utils
from ur5_env_mjco.envs import ur5_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('ur5', 'reach.xml')


class Ur5ReachEnv(ur5_env.Ur5Env, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            #"robot0:shoulder_lift_joint" : -0.5,
            #"robot0:elbow_joint" : 0.5
        }
        #target_range - gripper_extra_height = 0.15
        ur5_env.Ur5Env.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.20, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
