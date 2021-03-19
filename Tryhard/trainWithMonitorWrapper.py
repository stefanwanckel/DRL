from stable_baselines3 import HER, DQN, SAC, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise#, AdaptiveParamNoiseSpec
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
#from TimeLimitWrapper import TimeLimitWrapper
from gym.wrappers.time_limit import TimeLimit
import ur5e_env
import gym
import numpy as np
import time
import pprint
import os
def Main():
    pp = pprint.PrettyPrinter(indent=4)

    #make environment and wrap
    env=gym.make('ur5e_reacher-v1')
    env = Monitor(env,filename="logs",allow_early_resets=True)
    #***define model***
    #hyperparams
    # n_actions = env.action_space.shape[-1]
    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model_class = DDPG
    #kwargs are the parameters for DDPG model init
    kwargs= {"device":"cuda",
            "action_noise":NormalActionNoise}
    model=HER('MlpPolicy',
            env, 
            model_class, 
            n_sampled_goal=4, 
            goal_selection_strategy='future',
            verbose=1,
            learning_rate = 0.005,
            online_sampling=True,
            #max_episode_steps=4800
            **kwargs
            )

    #train model
    train = False                                 
    if train:   
        model.learn(2*10e5)
        model.save("./her_ur5e_model/model_")

        #load model, not really necessary
    evaluate = True
    # if evaluate:
    #     model = HER.load('./her_ur5e_model/model_', env=env)
    #     mean_reward, std_reward = evaluate_policy(model,
    #                                         env, 
    #                                         n_eval_episodes=5, 
    #                                         render=True, 
    #                                         return_episode_rewards=True
    #                                         )
    


if __name__ == "__main__":
    Main()