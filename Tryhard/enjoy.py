import gym
import ur5e_env
from stable_baselines3 import HER, DDPG, DQN, SAC, TD3
env = gym.make("ur5e_reacher-v1")
model = HER.load('./logs/her/ur5e_reacher-v1_4/best_model.zip', env=env)
env.render()
for episode in range(10):
    obs=env.reset()
    episodic_reward = 0
    for timestep in range(2000):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episodic_reward+=reward
        if reward > 0:
            print("Within epsilon")
        if timestep%100==0:
            print(" Reward at timestep {} is {}".format(str(timestep),str(reward)))
        if done:
            obs = env.reset()
            break
    print(" Reward at episode {} is {}".format(str(episode),str(episodic_reward)))