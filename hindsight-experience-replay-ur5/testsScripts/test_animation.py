import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_data = []
y_data = []

fig, ax = plt.subplots()
ax.set_xlim(0.22,0.52)
ax.set_ylim(-0.66-0.15,0.66+15)

line, = ax.scatter(0.376,-0.66)

def animation_frame(i):
    rndDisp =-0.15+ 0.3*np.random.random(3)
    g_x = 0.37 + rndDisp
    g_y = -0.66 + rndDisp
    x_data.append(g_x)
    y_data.append(g_y)
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    return line,

animation = FuncAnimation(fig, func=animation_frame,frames=np.arange(0,10,0.01),interval=10)
plt.show()