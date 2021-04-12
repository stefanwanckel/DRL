""" Log env kwargs in envs_list.csv file """

import pandas as pd



d = {
    
    'ur5e_reacher-v1':
    {
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 1
        },
        
    'ur5e_reacher-v2':
    {
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 1
        },

    'ur5e_reacher-v3':
    {
        'random_position' : True,
        'random_orientation': False,
        'moving_target': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        'action_scale': 1,
        'eps' : 0.1,
        'sim_rep' : 1
        },

    'ur5e_reacher-v4':
    {
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
        'sim_rep' : 1
        },
    
    'ur5e_reacher-v5':
    {
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
        },
    
    'ur5e_reacher-v6':
    {
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
        },
        
    'ur5e_reacher-v7':
    {
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
        },
        
    'ur5e_reacher-v8':
    {
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
        },
    'ur5e_reacher-v9':
    {
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
        },

    'ur5e_reacher-v10':
    {
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
        },

    'ur5e_reacher-v13':
    {
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

}

df = pd.DataFrame.from_dict(d, orient='index').reset_index()
df.rename(columns={'index': 'env_id'}, inplace=True)

print(df)

df.to_csv('gym_envs/ur5e_env/envs_list.csv', index=False)
