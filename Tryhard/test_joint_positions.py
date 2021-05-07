import ur5e_env
import gym
import pybullet as p

env = gym.make("ur5e_reacher-v1")
env.render()
env.reset()
env.test_arm()