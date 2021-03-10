""" Register Gym environments """

import numpy as np
from gym.envs.registration import register


register(
    id='ur5e_reacher-v1',
    entry_point='gym_envs.ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=100,
    kwargs={
        'random_position' : False,
        'random_orientation': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-0.05, -0.025, -0.025, -0.025, -0.05, -0.0005],
        'action_max': [0.05, 0.025, 0.025, 0.025, 0.05, 0.0005],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },
    )


