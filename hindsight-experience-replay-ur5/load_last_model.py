def load_last_model(save_dir, env_name):
    import os
    search_dir = os.path.join(save_dir, env_name)
    #search_dir = "./saved_models/ur5_push_no_gripper-v1"
    files = os.listdir(search_dir)
    files_sorted = sorted(files,
                          key=lambda x: os.path.getmtime(
                              os.path.join(search_dir, x))
                          )
    return files_sorted[-1]


print(load_last_model("saved_models", "ur5_push_no_gripper-v1"))
