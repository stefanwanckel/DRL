import ur5e_env
import gym
import time
from stable_baselines3.common.env_checker import check_env
import time


print("hello")

env = gym.make('ur5e_reacher-v1')

#print("any warnings?", check_env(env))
time.sleep(1)

env.render()
for episode in range(5):
    obs = env.reset()
    rewards = []

    for t in range(1000):
        if t%1==0:
            action = env.action_space.sample()
        # action = [0, 0, 0, 0, 0, 4]

        obs, reward, done, info = env.step(action)

        print("action: ", action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)
        print("timestep: ", t)

        rewards.append(reward)
        #time.sleep(1. / 240.)

    cumulative_reward = sum(rewards)
    print(
        "episode {} | cumulative reward : {}".format(
            episode,
            cumulative_reward))

