""" Register Gym environments """

import numpy as np
from gym.envs.registration import register


register(
    id='ur5e_reacher-v1',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=4000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 5,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1
        },
    )

register(
    id='ur5e_reacher-v2',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=4000,
    kwargs={
        'eps' : 0.1,
        #'random_position' : False,
        #'random_orientation': False,
        #'moving_target': False,
        #'target_type': "sphere",
        #'goal_oriented' : False,
        #'obs_type' : 1,
        'reward_type' : 13,
        #'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'action_scale' : 1,
        #'alpha_reward': 0.1,
        #'reward_coeff': 1,
        #
        }
    )

register(
    id='ur5e_reacher-v3',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=4000,
    kwargs={
        'eps' : 0.1,
        #'random_position' : False,
        #'random_orientation': False,
        #'moving_target': False,
        #'target_type': "sphere",
        #'goal_oriented' : False,
        #'obs_type' : 1,
        'reward_type' : 1,
        #'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'action_scale' : 1,
        #'alpha_reward': 0.1,
        #'reward_coeff': 1,
        #
        }
    )

register(
    id='ur5e_reacher-v4',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=4000,
    kwargs={
        'eps' : 0.1,
        #'random_position' : False,
        #'random_orientation': False,
        #'moving_target': False,
        #'target_type': "sphere",
        #'goal_oriented' : False,
        #'obs_type' : 1,
        'reward_type' : 13,
        #'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-0.03, -0.03, -0.03, -0.03, -0.03, -0.03],
        'action_max': [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        'action_scale' : 1,
        #'alpha_reward': 0.1,
        #'reward_coeff': 1,
        #
        }
    )