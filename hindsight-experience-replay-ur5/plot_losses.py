import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns
import os

save_dir = "saved_models"
env_name = "ur5_push_no_gripper-v1"
search_dir = os.path.join(save_dir, env_name)
latest = True
if latest:
    files = os.listdir(search_dir)
    files_ext = [file for file in files if file.endswith(".npz")]
    files_sorted = sorted(files_ext,
                          key=lambda x: os.path.getmtime(
                              os.path.join(search_dir, x))
                          )
    loss_file = files_sorted[-1]
else:
    loss_file = "INSERT_NAME"
print("using loss from ", os.path.join(search_dir, loss_file))
# actor_loss, critic_loss = np.load(os.path.join(search_dir, latest_loss))
data = np.load(os.path.join(search_dir, loss_file))
actor_loss = data["actor_loss"]
critic_loss = data["critic_loss"]

sns.set_theme()
# updates per epoch is n_batches*n_cycles
updates_per_epoch = 2000
x = (1/updates_per_epoch)*np.linspace(1,
                                      actor_loss.shape[0], actor_loss.shape[0])
plt.figure(figsize=(16, 12))
plt.plot(x, -actor_loss, label="actor loss")
plt.plot(x, -critic_loss, label="critic loss")
plt.title("training losses of {}".format(env_name))
plt.xlabel("n epochs")
plt.ylabel("loss")
plt.legend()
# plt.tight_layout()
# title = "Results/figures/" + "success_rate_ur5" + saveName + ".png"
# plt.savefig(os.path.join(title))
# title = "Results/figures/" + "success_rate_ur5" + saveName + ".svg"
# plt.savefig(os.path.join(title))
plt.show()
