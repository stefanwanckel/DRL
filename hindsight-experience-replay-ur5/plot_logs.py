import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns
import os

#import log
logName = "reach.log"
saveName = "reach_adapted"
with open(os.path.join("Results/logs",logName),"r") as plog:
    lines = [line.split() for line in plog]
#get epochs, success, std from log rows
epochs = []
successes = []
stds = []
#remove "," from string and cast to float
for splitLine in lines:
    epochs.append(float(splitLine[4].replace(",","")))
    successes.append(float(splitLine[9].replace(",","")))
    stds.append(float(splitLine[-1].replace(",","")))
#make np arrays
epochs = np.array(epochs)
successes = np.array(successes)
stds = np.array(stds)
#plot
sns.set_theme()
plt.figure(figsize=(16,12))
plt.plot(epochs,successes,label="mean success rate")
#plt.fill_between(epochs, successes+stds,successes-stds, label="1 standard deviation",alpha=0.2)
plt.title("DDPG + HER: {} task success rate in training for UR5 over 1 seed(s)".format(logName.split(".")[0]))
plt.xlabel("n epochs")
plt.ylabel("mean success rate")
plt.legend()
#plt.tight_layout()
title = "Results/figures/" +"success_rate_ur5" + saveName + ".png"
plt.savefig(os.path.join(title))
title = "Results/figures/" +"success_rate_ur5" + saveName + ".svg"
plt.savefig(os.path.join(title))