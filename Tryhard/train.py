from stable_baselines3 import HER, DQN, SAC, DDPG, TD3
import ur5e_env
import gym
import numpy as np
import time
import pprint

pp = pprint.PrettyPrinter(indent=4)
model_class = DDPG
goal_selection_strategy = 'future'
env=gym.make('ur5e_reacher-v1')
model=HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                                                verbose=1,gamma=0.99)
train = False                                  
if train:   
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