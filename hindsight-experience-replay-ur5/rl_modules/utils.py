import torch
import os


def load_saved_state_dicts(save_dir, env_name, rank):
    latest = False
    if latest:
        search_dir = os.path.join(save_dir, env_name)
        files = os.listdir(search_dir)
        files_ext = [file for file in files if file.endswith(".pt")]
        files_sorted = sorted(files_ext,
                              key=lambda x: os.path.getmtime(
                                  os.path.join(search_dir, x))
                              )
        latest_model = files_sorted[-1]

        model_path = os.path.join(save_dir, env_name, latest_model)
    else:
        dash = "-"*42
        model_name = "2021-12-12T20:22:04.220598_epoch_19.pt"
        project_path = "12-12-2021_1_reduced_obs"
        log_name = "ur5_push_no_gripper_1.log"
        if rank == 0:
            print("Continuing from {} ".format(model_name))
            print("log file from last prior training:")
            print("")
            with open(os.path.join(save_dir, env_name, project_path, log_name), "r") as log:
                lines = log.readlines()
                for line in lines:
                    print(line)
            print(dash)
        model_path = os.path.join(
            save_dir, env_name, project_path, model_name)

    _, _,  _, _, actor_model, critic_model, actor_target_model, critic_target_model = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    saved_dicts = {
        'actor': actor_model,
        'critic': critic_model,
        'actor_target': actor_target_model,
        'critic_target': critic_target_model
    }
    return saved_dicts
