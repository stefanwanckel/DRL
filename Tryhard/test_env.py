import ur5e_env
import gym
import time
from stable_baselines3.common.env_checker import check_env
import time
import pprint

pp = pprint.PrettyPrinter(indent=4)
print(float(1e5))
print("hello")

env = gym.make('ur5e_reacher-v1')
#check_env(env)
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
        #for key in obs.keys():
        #    print(key,obs[key])
        # time.sleep(1. / 3.)
        # print("action: ", action)
        # pp.pprint(obs)
        # print("reward: ", reward)
        # print("done: ", done)
        # print("info: ", info)
        # print("timestep: ", t)
        # print("end_effector_pos: ",info["endeffector_pos"])
        # print("goal_pos: ",info['goal_pos'])

        rewards.append(reward)
        #time.sleep(1. / 240.)
