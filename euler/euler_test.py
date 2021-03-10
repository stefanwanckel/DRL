from sympy import *
from sympy.abc import x
from matplotlib import pyplot as plt
import numpy as np
t = np.arange(0,10,0.1)
C = 1
lam = 1
thisDict = {
    "time" : [np.max(t),np.min(t)],
    "constant": C,
    "lambda" : lam
}

f = C*np.exp(-lam*t)

plt.plot(t,f,label='exact')
title = str(thisDict.items())
plt.title(title)
plt.xlabel('t')
plt.ylabel('f(t)')

plt.legend()
plt.show()