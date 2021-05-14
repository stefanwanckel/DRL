import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns

#import log
with open("push.log","r") as plog:
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
plt.fill_between(epochs, successes+stds,successes-stds, label="1 standard deviation",alpha=0.2)
plt.title("DDPG + HER: push task success rate in training for UR5 over 5 seeds")
plt.xlabel("n epochs")
plt.ylabel("mean success rate")
plt.legend()
#plt.tight_layout()
plt.savefig("figures/sucess_rate_ur5_push-v1.png",dpi = 400)
plt.savefig("figures/sucess_rate_ur5_push-v1.svg")