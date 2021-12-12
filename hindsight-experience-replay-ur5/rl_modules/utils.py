import torch
import os


def load_saved_state_dicts(save_dir, env_name):

    search_dir = os.path.join(save_dir, env_name)
    files = os.listdir(search_dir)
    files_ext = [file for file in files if file.endswith(".pt")]
    files_sorted = sorted(files_ext,
                          key=lambda x: os.path.getmtime(
                              os.path.join(search_dir, x))
                          )
    latest_model = files_sorted[-1]
    model_path = os.path.join(save_dir, env_name, latest_model)
    _, _,  _, _, actor_model, critic_model, actor_target_model, critic_target_model = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    saved_dicts = {
        'actor': actor_model,
        'critic': critic_model,
        'actor_target': actor_target_model,
        'critic_target': critic_target_model
    }
    return saved_dicts
