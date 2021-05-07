""" Register Gym environments """

import numpy as np
from gym.envs.registration import register


register(
    id='ur5_reach-v1',
    entry_point='ur5_env_mjco.envs.ur5_tasks.reach:Ur5ReachEnv',
    #kwargs=kwargs,
    max_episode_steps=50,
    )
