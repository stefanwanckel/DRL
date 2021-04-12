import numpy as np

with np.load('logs/her/ur5e_reacher-v1_3/evaluations.npz') as data:
    #    print("timesteps: ",data["timesteps"])
    #    print("results: ",data["results"])
    #    print("ep_lengths: ",data["ep_lengths"])
    print("timesteps: ",data["timesteps"].shape)
    print("results: ",data["results"].shape)
    print("ep_lengths: ",data["ep_lengths"].shape)