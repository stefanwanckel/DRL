from stable_baselines3 import HER, DQN, SAC, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise#, AdaptiveParamNoiseSpec
#from TimeLimitWrapper import TimeLimitWrapper
from gym.wrappers.time_limit import TimeLimit
import ur5e_env
import gym
import numpy as np
import time
import pprint

pp = pprint.PrettyPrinter(indent=4)


#make environment and wrap
env=gym.make('ur5e_reacher-v1')
env = Monitor(env,filename="logs",allow_early_resets=True)

#***define model***
#hyperparams
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
model_class = DDPG#('MlpPolicy',env,verbose=1, action_noise=action_noise)
model=HER('MlpPolicy',
          env, 
          model_class, 
          n_sampled_goal=4, 
          goal_selection_strategy='future',
          verbose=1,
          #max_episode_steps=4800
          )
#print(model.get_parameter_list())

#train model
train = False                                  
if train:   
    model.learn(6000)
    model.save("./her_ur5e_model/model_3")

#load model, not really necessary
model = HER.load('./her_ur5e_model/model_3', env=env)

mean_reward, std_reward = evaluate_policy(model,
                                          env, 
                                          n_eval_episodes=5, 
                                          render=True, 
                                          return_episode_rewards=True
                                          )
print(mean_reward,std_reward)