# Visualisation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


def setup_vis(nTests, startPos, SampleRange, axisLimitExtends, g_robotCF, goal_threshold):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    fig.suptitle("TCP Planar View - Episode {}".format(nTests))
    fig.canvas.set_window_title("Episode No. {}".format(nTests))
    ax = fig.gca()

    ax1.set_xlim(startPos[0]-SampleRange-axisLimitExtends,
                 startPos[0]+SampleRange+axisLimitExtends)
    ax1.set_ylim(startPos[1]-SampleRange-axisLimitExtends,
                 startPos[1]+SampleRange+axisLimitExtends)
    ax2.set_xlim(startPos[0]-SampleRange-axisLimitExtends,
                 startPos[0]+SampleRange+axisLimitExtends)
    ax2.set_ylim(startPos[2]-SampleRange-axisLimitExtends,
                 startPos[2]+SampleRange+axisLimitExtends)
    ax1.grid()
    ax2.grid()
    ax1.set(adjustable='box', aspect='equal')
    ax2.set(adjustable='box', aspect='equal')
    ax1.set_title("X-Y TCP")
    ax2.set_title("X-Z TCP")
    ax1.set_xlabel("X-axis [m]")
    ax2.set_xlabel("X-axis [m]")
    ax1.set_ylabel("Y-axis [m]")
    ax2.set_ylabel("Z-axis [m]")

    ax1.plot(g_robotCF[0], g_robotCF[1], 'o', color="g")
    ax2.plot(g_robotCF[0], g_robotCF[2], 'o', color="g")
    graph, = ax1.plot([], [], 'o', color="r")
    sampleSpace_xy = Circle((startPos[0], startPos[1]),
                            radius=SampleRange, fill=False)
    successThreshold_xy = Circle(
        (g_robotCF[0], g_robotCF[1]), radius=goal_threshold, fill=False, ls="--")
    sampleSpace_xz = Circle((startPos[0], startPos[2]),
                            radius=SampleRange, fill=False)
    successThreshold_xz = Circle(
        (g_robotCF[0], g_robotCF[2]), radius=goal_threshold, fill=False, ls="--")
    ax1.add_patch(sampleSpace_xy)
    ax1.add_patch(successThreshold_xy)
    ax2.add_patch(sampleSpace_xz)
    ax2.add_patch(successThreshold_xz)

    return fig, (ax1, ax2)


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
