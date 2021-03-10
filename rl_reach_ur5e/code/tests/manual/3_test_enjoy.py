""" Simple policy evaluation with Stable Baselines """


import gym
import ur5e_env
import time
from stable_baselines3 import PPO
from gym.wrappers.monitoring.video_recorder import VideoRecorder
#Trigger = True
#recPath = "/home/stefan/Documents/Masterarbeit/rl_reach_ur5e/code/tests/manual/vids"
env = gym.make('ur5e_reacher-v1')
#rec = VideoRecorder(env,path='vid.mp4',enabled=True)
# model = PPO(MlpPolicy, env, verbose=1)
model = PPO.load("../../logs/exp_99/ppo/ur5e_reacher-v1_1/ur5e_reacher-v1.zip")

obs = env.reset()
env.render(mode="human")

for t in range(200):
    if t%10==0:
        print(t)
#    rec.capture_frame() 
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    time.sleep(0.01)

env.close()
#rec.close()
