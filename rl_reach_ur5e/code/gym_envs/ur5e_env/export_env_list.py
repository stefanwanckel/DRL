""" Log env kwargs in envs_list.csv file """

import pandas as pd



d = {
    'widowx_reacher-v1':
    {
        'random_position' : False,
        'random_orientation': False,
        'target_type': "sphere",
        'goal_oriented' : False,
        'obs_type' : 1,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': '[-0.05, -0.025, -0.025, -0.025, -0.05, -0.0005]',
        'action_max': '[0.05, 0.025, 0.025, 0.025, 0.05, 0.0005]',
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'ur5e_reacher-v1':
    {
        'random_position' : False,
        'random_orientation': False,
        'target_type': "sphere",
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': '[-0.05, -0.025, -0.025, -0.025, -0.05, -0.0005]',
        'action_max': '[0.05, 0.025, 0.025, 0.025, 0.05, 0.0005]',
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v2':
    {
        'random_position' : False,
        'random_orientation': False,
        'target_type': "sphere",
        'goal_oriented' : False,
        'obs_type' : 2,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

}

df = pd.DataFrame.from_dict(d, orient='index').reset_index()
df.rename(columns={'index': 'env_id'}, inplace=True)

print(df)

df.to_csv('gym_envs/ur5e_env/envs_list.csv', index=False)
#df.to_csv('envs_list.csv', index=False)