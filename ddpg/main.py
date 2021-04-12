import gym 
import numpy as np
import matplotlib.pyplot as plt
import ur5e_env


from ddpg_tf2 import Agent

if __name__ == '__main__':
    env = gym.make('ur5e_reacher-v17')
    agent = Agent(input_dims = env.observation_space.shape,
                  env = env,
                  n_actions = env.action_space.shape[0],
                  noise = 0.2,
                  batch_size = 128
                  )

    n_episodes = 250

    figure_file = 'tmp/ddpg/ur5e_reacher-v1.png'

    best_score = env.reward_range[0]
    score_history = []
    dist_history = []
    load_trained_model = False
    improve_model = True
    observation = env.reset()
    
    if load_trained_model:
        #until agent.learn() only needed for initialization
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps+=1
        agent.learn()
        agent.load_models()
        evaluate = True
    else: 
        evaluate = False

    #if not improve_model:
    env.render()

    for i in range(n_episodes):
        observation = env.reset()
        
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if improve_model:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if improve_model:
                agent.save_models()
        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    env.close()
    
    if improve_model:
        x = [i+1 for i in range(n_episodes)]
        #plot_learning_curve(x,score_history, figure_file)
        plt.plot(x,score_history)
        plt.savefig(figure_file)