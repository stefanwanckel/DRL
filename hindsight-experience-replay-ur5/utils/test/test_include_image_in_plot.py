from real_demo_visualization import setup_vis_push_test
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

img = cv2.imread('test_image.jpg')

fig, axs, ax1, ax2, ax_img = setup_vis_push_test(
    nTests=1,
    startPos=[0, 0, 0],
    SampleRange=0.30,
    axisLimitExtends=0.10,
    g_robotCF=[0.1, 0, 0],
    goal_threshold=0.05)

x = y = z = [0, 0.1, 0.2, 0.15]
ax1.plot(x, y, 'o', color="r")
ax2.plot(x, z, 'o', color="r")

ax_img.imshow(img)
ax_img.axis('off')
plt.show()
