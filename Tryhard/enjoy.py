import gym
import ur5e_env
from stable_baselines3 import HER, DDPG, DQN, SAC, TD3
import time
env = gym.make("ur5e_reacher-v13")
#model = HER.load('./logs/Results/rl_model_50000_steps-v16', env=env)
model = HER.load('./logs/her/ur5e_reacher-v13_2/rl_model_250000_steps', env=env)
env.render()
for episode in range(10):
    obs=env.reset()
    episodic_reward = 0
    for timestep in range(1000):
        #time.sleep(1/90)
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episodic_reward+=reward
        if reward > 0:
            print("Within epsilon")
        if timestep%100==0:
            print(" Reward at timestep {} is {}".format(str(timestep),str(reward)))
        if info["distance"]<0.1:
            print(timestep)
            print("within epsilon: Desired goal is achieved. Displaying info at last step:")
            print("distance:", info["distance"])
            print("goal_pos:", info["goal_pos"])
            print("endeffector_pos:", info["endeffector_pos"])

            time.sleep(4)
            obs = env.reset()
            
            break
            
    time.sleep(1)
    print(" Reward at episode {} is {}".format(str(episode),str(episodic_reward)))