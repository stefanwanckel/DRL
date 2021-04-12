
import gym
import ur5e_env
import time
from stable_baselines3.common.env_checker import check_env
import time
import pprint

env = gym.make('ur5e_reacher-v1')

#print("any warnings?", check_env(env))


env.render()
cart_position = [0.37756395,  0.32278168,  0.4931646 ]
position = [0.4064539 , -1.0803024 , 1.0832953 ,  0.56399596, -1.3985776 ,  1.]
env.set_joint_positions(position)  
print(env.get_end_effector_position())

# print("action: ", action)
# pp.pprint(obs)
# print("reward: ", reward)
# print("done: ", done)
# print("info: ", info)
# print("timestep: ", t)
# print("end_effector_pos: ",info["endeffector_pos"])
# print("goal_pos: ",info['goal_pos'])


