import os
import numpy as np
import torch
import time


def load_last_model(save_dir, env_name):
    search_dir = os.path.join(save_dir, env_name)
    #search_dir = "./saved_models/ur5_push_no_gripper-v1"
    files = os.listdir(search_dir)
    files_ext = [file for file in files if file.endswith(".pt")]
    files_sorted = sorted(files_ext,
                          key=lambda x: os.path.getmtime(
                              os.path.join(search_dir, x))
                          )
    assert len(files_sorted) > 0, "{} must be empty...".format(search_dir)
    return files_sorted[-1]


def load_last_archived_model(save_dir, env_name, project_dir):
    search_dir = os.path.join(save_dir, env_name, project_dir)
    #search_dir = "./saved_models/ur5_push_no_gripper-v1"
    files = os.listdir(search_dir)
    files_ext = [file for file in files if file.endswith(".pt")]
    files_sorted = sorted(files_ext,
                          key=lambda x: os.path.getmtime(
                              os.path.join(search_dir, x))
                          )
    assert len(files_sorted) > 0, "{} must be empty...".format(search_dir)
    return files_sorted[-1]


def get_demo_model_path(args, last_model, is_archived):
    """
    loads model depending on:
    Environment name
    if demo should be run with latest model
    """
    if args.env_name == "ur5_reach-v1":
        model_path = args.save_dir + args.env_name + '/Oct_24_1_2.pt'
    elif args.env_name == "ur5_push-v1":
        if last_model:
            if is_archived:
                model_path = os.path.join(
                    args.save_dir, args.env_name, args.project_dir, load_last_archived_model(args.save_dir, args.env_name, args.project_dir))
            else:
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
            if is_archived:
                model_path = os.path.join(
                    args.save_dir, args.env_name, args.project_dir, load_last_archived_model(args.save_dir, args.env_name, args.project_dir))
            else:
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
            if is_archived:
                model_path = os.path.join(
                    args.save_dir, args.env_name, args.project_dir, load_last_archived_model(args.save_dir, args.env_name, args.project_dir))
                print("Last model name: ", load_last_archived_model(
                    args.save_dir, args.env_name, args.project_dir))
            else:
                model_path = os.path.join(
                    args.save_dir, args.env_name, load_last_model(args.save_dir, args.env_name))
                print("Last model name: ", load_last_model(
                    args.save_dir, args.env_name))
            time.sleep(1)
        else:
            model_path = args.save_dir + args.env_name + \
                '/2021-12-11T17:18:10.390514_epoch_18.pt'
    elif args.env_name == "ur5_pick_and_place-v1":
        if last_model:
            if is_archived:
                model_path = os.path.join(
                    args.save_dir, args.env_name, args.project_dir, load_last_archived_model(args.save_dir, args.env_name, args.project_dir))
                print("Last model name: ", load_last_archived_model(
                    args.save_dir, args.env_name, args.project_dir))
            else:
                model_path = os.path.join(
                    args.save_dir, args.env_name, load_last_model(args.save_dir, args.env_name))
                print("Last model name: ", load_last_model(
                    args.save_dir, args.env_name))
            time.sleep(1)
        else:
            model_path = args.save_dir + args.env_name + \
                '/2021-12-11T17:18:10.390514_epoch_18.pt'

    return model_path


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
