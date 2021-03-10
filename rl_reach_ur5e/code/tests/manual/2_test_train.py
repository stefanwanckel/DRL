""" Simple training with Stable Baselines """


import os
import gym
import ur5e_env
from stable_baselines3 import PPO
#from stable_baselines3 import HER
#from stable_baselines3 import TD3
from stable_baselines3 import HER, DDPG, DQN, SAC, TD3
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper


LOG_DIR = "logs/test/"
os.makedirs(LOG_DIR, exist_ok=True)

env = gym.make('ur5e_reacher-v1')
env = Monitor(env, LOG_DIR)

#model = HER(MlpPolicy, env,model_class=TD3)
model_class = DDPG
#model = PPO('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy='future', 
#                        verbose=1)
model = HER(MlpPolicy, env,  model_class=model_class,verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save(LOG_DIR + "ur5e_reach-HER")
