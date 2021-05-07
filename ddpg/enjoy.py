import gym
from ddpg_tf2 import Agent

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = Agent(input_dims = env.observation_space.shape, env = env, n_actions = env.action_space.shape[0])

    n_games = 20
    #if evaluate  = True it means we dont add noise to the action
    evaluate = True
    agent.load_models()

    env.render()
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            #agent.remember(observation, action, reward, observation_, done)