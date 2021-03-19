import gym
import numpy as np
import time
import pprint
import yaml
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from stable_baselines3 import HER, DQN, SAC, DDPG, TD3
import ur5e_env

def Main():
    #define arguments for her
    env_id='ur5e_reacher-v1'
    model_class = DDPG
    goal_selection_strategy = 'future'
    env=gym.make(env_id)
    #define kwargs to be passed to HER and wrapped algo
    kwargs = {  #"n_timesteps":10000,
                "policy": 'MlpPolicy',
                "model_class": DDPG,
                "n_sampled_goal": 4,
                "goal_selection_strategy": 'future',
                "buffer_size": 1000000,
                #"ent_coef": 'auto',
                "batch_size": 256,
                "gamma": 0.95,
                "learning_rate": 0.001,
                "learning_starts": 1000,
                "online_sampling": True,
                #"normalize": True
                }
    #In the future, read hyperparams from her.yml
    #kwargs = read_hyperparameters(env_id)
                                    
    model = HER(env=env,**kwargs)
    total_n_steps = 1e6
    safe_freq = total_n_steps//10
    max_episode_length = 4000
    n_episodes = total_n_steps//max_episode_length

    model.learn(4000)
    model.save("./her_ur5e_model/model_3")

    model = HER.load('./her_ur5e_model/model_3', env=env)


    all_cumulative_rewards = []
    num_episodes = 5
    num_timesteps = 4800
    env.render()
    #each timestep lasts 1/240 s.
    for episode in range(num_episodes):
        obs = env.reset()
        epi_rewards = []
        for t in range(num_timesteps):

            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            #time.sleep(1/240)
            epi_rewards.append(reward)

            if t==num_timesteps-1:
                done = True
            if done:
                #pp.pprint(info)
                obs = env.reset()
                cumulative_reward = sum(epi_rewards)
                all_cumulative_rewards.append(cumulative_reward)
                print(
                "episode {} | cumulative reward : {}".format(
                    episode,
                    cumulative_reward))
    print("all_cumulative_rewards: ")
    pp.pprint(all_cumulative_rewards)


def read_hyperparameters(env_id) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Load hyperparameters from yaml file
    with open("hyperparams/her.yml", "r") as f:
    #with open("logs.monitor.csv", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]

        else:
            raise ValueError(f"Hyperparameters not found for her-{env_id}")

    return hyperparams

if __name__ == "__main__":
    Main()