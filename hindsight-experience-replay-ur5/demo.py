import torch
from numpy.lib.function_base import _average_dispatcher
from rl_modules.models import actor
from utils import load_last_model
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
    last_model = False
    # load the model param
    print(vars(args))
    dash = "-"*42
    for i, arg in enumerate(vars(args)):
        if i == 0:
            print("Starting demo with these arguments:")
            print(dash)
            print("{:<25s}{:<15s}".format("PARAMS", "VALUE"))
            print(dash)
        print("|{:<22s} | {:<15}|".format(arg, getattr(args, arg)))
        if i == len(vars(args))-1:
            print(dash)
    if args.env_name == "ur5_reach-v1":
        model_path = args.save_dir + args.env_name + '/Oct_24_1_2.pt'
    elif args.env_name == "ur5_push-v1":
        if last_model:
            model_path = os.path.join(
                args.save_dir, args.env_name, load_last_model(args.save_dir, args.env_name))
            print("Last model name: ", load_last_model(
                args.save_dir, args.env_name))
            time.sleep(1)
        else:
            model_path = args.save_dir + args.env_name + \
                '/2021-12-10T04:00:18.657028_epoch_79.pt'
    elif args.env_name == "ur5_reach_no_gripper-v1":
        if last_model:
            model_path = os.path.join(
                args.save_dir, args.env_name, load_last_model(args.save_dir, args.env_name))
            print("Last model name: ", load_last_model(
                args.save_dir, args.env_name))
            time.sleep(1)
        else:
            model_path = args.save_dir + args.env_name + \
                '/2021-12-10T04:00:18.657028_epoch_79.pt'
    elif args.env_name == "ur5_push_no_gripper-v1":
        if last_model:
            model_path = os.path.join(
                args.save_dir, args.env_name, load_last_model(args.save_dir, args.env_name))
            print("Last model name: ", load_last_model(
                args.save_dir, args.env_name))
            time.sleep(1)
        else:
            model_path = args.save_dir + args.env_name + \
                '/2021-12-11T17:18:10.390514_epoch_18.pt'
    # o_mean, o_std, g_mean, g_std, model, _, _, _ = torch.load(
    #     model_path, map_location=lambda storage, loc: storage)
    o_mean, o_std, g_mean, g_std, model = torch.load(
        model_path, map_location=lambda storage, loc: storage)
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
                print("{:<25s}{:<15s}".format("ENV_ARGS", "VALUE"))
                for key in info["args"]:
                    print("|{:<22s} | {:<15}|".format(key, info["args"][key]))
                print(dash)

        print('the episode is: {}, is success: {}.'.format(
            i, info['is_success']))
        print('Episode {} has goal {}'.format(i, lstGoals[i]))
