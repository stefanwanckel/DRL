import matplotlib.pyplot as plt
import os
import numpy as np

importPath = "logs/Results"
exportPath = os.path.join(importPath, "Plots")

evalFiles = sorted([item for item in os.listdir(importPath) if ".npz" in item])

for i, cE in enumerate(evalFiles):
    currEval = np.load(os.path.join(importPath,cE))
    Ep_lengths = currEval['ep_lengths']
    Results = currEval['results']
    Timesteps = currEval['timesteps']
    plotName = cE.split('.')[0]


    plt.figure()
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    plt.title("learning for {}".format(cE))
    # plt.plot(Timesteps,Results[:,0],label="eval1")
    # plt.plot(Timesteps,Results[:,1],label="eval2")
    # plt.plot(Timesteps,Results[:,2],label="eval3")
    plt.plot(Timesteps,np.mean(Results,axis=1),label="mean",linewidth=2)
    plt.scatter(Timesteps,np.mean(Results,axis=1),label="mean_points")
    plt.legend()
    plt.savefig(os.path.join(exportPath,plotName),dpi= 300)


