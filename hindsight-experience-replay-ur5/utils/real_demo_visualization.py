from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib


def setup_vis_reach(nTests, startPos, SampleRange, axisLimitExtends, g_robotCF, goal_threshold):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    fig.suptitle("TCP Planar View REACH - Episode {}".format(nTests))
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
                            radius=SampleRange+0.05, fill=False)
    successThreshold_xy = Circle(
        (g_robotCF[0], g_robotCF[1]), radius=goal_threshold, fill=False, ls="--")
    sampleSpace_xz = Circle((startPos[0], startPos[2]),
                            radius=SampleRange+0.05, fill=False)
    successThreshold_xz = Circle(
        (g_robotCF[0], g_robotCF[2]), radius=goal_threshold, fill=False, ls="--")
    ax1.add_patch(sampleSpace_xy)
    ax1.add_patch(successThreshold_xy)
    ax2.add_patch(sampleSpace_xz)
    ax2.add_patch(successThreshold_xz)

    return fig, (ax1, ax2)


def setup_vis_push(nTests, startPos, SampleRange, axisLimitExtends, g_robotCF, goal_threshold):

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2)
    fig.tight_layout()
    fig.suptitle("TCP Planar View PUSH - Episode {}".format(nTests))
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
    ax1.set_title("PUSH X-Y TCP")
    ax2.set_title("PUSH X-Z TCP")
    ax1.set_xlabel("X-axis [m]")
    ax2.set_xlabel("X-axis [m]")
    ax1.set_ylabel("Y-axis [m]")
    ax2.set_ylabel("Z-axis [m]")

    ax1.plot(g_robotCF[0], g_robotCF[1], 'o', color="g")
    ax2.plot(g_robotCF[0], g_robotCF[2], 'o', color="g")
    graph, = ax1.plot([], [], 'o', color="r")
    sampleSpace_xy = Circle((startPos[0], startPos[1]),
                            radius=SampleRange+0.05, fill=False)
    successThreshold_xy = Circle(
        (g_robotCF[0], g_robotCF[1]), radius=goal_threshold, fill=False, ls="--")
    sampleSpace_xz = Circle((startPos[0], startPos[2]),
                            radius=SampleRange+0.05, fill=False)
    successThreshold_xz = Circle(
        (g_robotCF[0], g_robotCF[2]), radius=goal_threshold, fill=False, ls="--")
    ax1.add_patch(sampleSpace_xy)
    ax1.add_patch(successThreshold_xy)
    ax2.add_patch(sampleSpace_xz)
    ax2.add_patch(successThreshold_xz)

    return fig, (ax1, ax2)


def setup_vis_push_w_image(nTests, startPos, SampleRange, axisLimitExtends, g_robotCF, goal_threshold):

    title_fontsize = 8
    tic_fontsize = 8
    font = {'family': 'normal',
            'weight': 'normal',
            'size': tic_fontsize}
    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(4, 4)
    gs = GridSpec(4, 4, figure=fig, hspace=0.5, height_ratios=[3, 1, 1, 1])

    # fig.tight_layout()
    fig.suptitle("TCP Planar View Pick and Place ")
    fig.canvas.set_window_title("Episode No. {}".format(nTests))
    plt.axis('off')

    ax1 = fig.add_subplot(gs[2:, :2])
    ax2 = fig.add_subplot(gs[2:, 2:])
    ax_img = fig.add_subplot(gs[:2, :])
    ax_img.axis('off')

    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')

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
    ax1.set_title("PUSH X-Y TCP", fontsize=title_fontsize)
    ax2.set_title("PUSH X-Z TCP", fontsize=title_fontsize)
    ax1.set_xlabel("X-axis [m]", fontsize=title_fontsize)
    ax2.set_xlabel("X-axis [m]", fontsize=title_fontsize)
    ax1.set_ylabel("Y-axis [m]", fontsize=title_fontsize)
    ax2.set_ylabel("Z-axis [m]", fontsize=title_fontsize)

    ax1.plot(g_robotCF[0], g_robotCF[1], 'o', color="g", markersize=4)
    ax2.plot(g_robotCF[0], g_robotCF[2], 'o', color="g", markersize=4)
    graph, = ax1.plot([], [], 'o', color="r")
    sampleSpace_xy = Circle((startPos[0], startPos[1]),
                            radius=SampleRange+0.05, fill=False)
    successThreshold_xy = Circle(
        (g_robotCF[0], g_robotCF[1]), radius=goal_threshold, fill=False, ls="--", color="gray")
    sampleSpace_xz = Circle((startPos[0], startPos[2]),
                            radius=SampleRange+0.05, fill=False)
    successThreshold_xz = Circle(
        (g_robotCF[0], g_robotCF[2]), radius=goal_threshold, fill=False, ls="--", color="gray")
    ax1.add_patch(sampleSpace_xy)
    ax1.add_patch(successThreshold_xy)
    ax2.add_patch(sampleSpace_xz)
    ax2.add_patch(successThreshold_xz)

    return fig, axs, ax1, ax2, ax_img
