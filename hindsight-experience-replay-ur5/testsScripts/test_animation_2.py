import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = np.arange(10)
y = np.random.random(10)

fig = plt.figure()
plt.xlim(0, 10)
plt.ylim(0, 1)
graph, = plt.plot([], [], 'o')


def animate(i):
    graph.set_data(x[:i+1], y[:i+1])
    return graph


ani = FuncAnimation(fig, animate, frames=10, interval=200)
plt.show()
