import torch
from numpy.lib.function_base import _average_dispatcher
from rl_modules.models import actor
from utils.model_loader import load_last_model, get_demo_model_path
from arguments import get_args
import gym
import numpy as np
import ur5_env_mjco
import os
import time
# process the inputs

np.set_printoptions(2)


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -
                     args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -
                     args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


if __name__ == '__main__':
    args = get_args()

    dash = "-"*42
    for i, arg in enumerate(vars(args)):
        if i == 0:
            print("Starting demo with these arguments:")
            print(dash)
            print("{:<25s}{:<15s}".format("PARAMS", "VALUE"))
            print(dash)
        if getattr(args, arg) is not None:
            print("|{:<22s} | {:<15}|".format(arg, getattr(args, arg)))
        if i == len(vars(args))-1:
            print(dash)
    # load the model from file
    # commented out code is in case of use of models with saved actor_network only
    last_model = True
    is_archived = False
    if args.project_dir is not None:
        is_archived = True
    model_path = get_demo_model_path(args, last_model, is_archived)
    o_mean, o_std, g_mean, g_std, model, _, _, _ = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    # o_mean, o_std, g_mean, g_std, model = torch.load(
    #     model_path, map_location=lambda storage, loc: storage)
    # create the environment

    env = gym.make(args.env_name)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    # needed for evaluation part.
    actor_network.eval()
    lstGoals = []
    success_counter = 0
    for i in range(args.demo_length):
        observation = env.reset()

        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        lstGoals.append(g)

        for t in range(env._max_episode_steps):  # env._max_episode_steps):

            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
            if t == 0 and i == 0:
                print(dash)
                print("{:<25s}{:<15s}".format("ENV_INIT", "VALUE"))
                for key in env.inits:
                    print("|{:<22s} | {:<15}|".format(key, env.inits[key]))
                print(dash)
        if info['is_success'] == 1:
            success_counter += 1
        print('Episode-No.: {} \n\t is success: {},\t overall success : {}/{}'.format(
            i, info['is_success'], success_counter, args.demo_length))
        #print('Episode {} has goal {}'.format(i, lstGoals[i]))
