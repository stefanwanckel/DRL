""" Register Gym environments """

import numpy as np
from gym.envs.registration import register


register(
    id='ur5e_reacher-v1',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=2000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 5
        },
    )

register(
    id='ur5e_reacher-v2',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=2000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 5
        }
    )

register(
    id='ur5e_reacher-v3',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=2000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 5
        }
    )

register(
    id='ur5e_reacher-v4',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=2000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 5,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 5
        }
    )

register(
    id='ur5e_reacher-v5',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=2000,
    kwargs={
        'random_position' : False,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 5,
        'action_mode' : "force"
        }
    )

register(
    id='ur5e_reacher-v6',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=200,
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
        'eps' : 0.01,
        'sim_rep' : 1
        }
    )

register(
    id='ur5e_reacher-v7',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=200,
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
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )
register(
    id='ur5e_reacher-v8',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
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
        'action_min': [-0.1, -.1, -.1, -.1, -.1, -.1],
        'action_max': [.1, .1, .1, .1, .1, .1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )

register(
    id='ur5e_reacher-v9',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=200,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 5,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-.1, -.1, -.1, -.1, -.1, -.1],
        'action_max': [.1, .1, .1, .1, .1, .1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )

register(
    id='ur5e_reacher-v10',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
       'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 5,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-.1, -.1, -.1, -.1, -.1, -.1],
        'action_max': [.1, .1, .1, .1, .1, .1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )


register(
    id='ur5e_reacher-v11',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=177,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )

register(
    id='ur5e_reacher-v13',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )

register(
    id='ur5e_reacher-v14',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )

register(
    id='ur5e_reacher-v15',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )

register(
    id='ur5e_reacher-v16',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )

register(
    id='ur5e_reacher-v17',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : False,
        'obs_type' : 1,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )

register(
    id='ur5e_reacher-v18',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )

register(
    id='ur5e_reacher-v19',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10
        }
    )

register(
    id='ur5e_reacher-v20',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 10,
        'action_mode':"force"
        }
    )

register(
    id='ur5e_reacher-v21',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 11,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.15,
        'sim_rep' : 5,
        'action_mode' : "set"
        }
    )

register(
    id='ur5e_reacher-v22',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.20,
        'sim_rep' : 5,
        'action_mode' : "set"
        }
    )

register(
    id='ur5e_reacher-v23',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=2000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 16,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.15,
        'sim_rep' : 10,
        'action_mode' : "set"
        }
    )

register(
    id='ur5e_reacher-v24',
    entry_point='ur5e_env.envs.ur5e_env:Ur5eEnv',
    max_episode_steps=1000,
    kwargs={
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.2,
        'sim_rep' : 10,
        'action_mode' : "set"
        }
    )