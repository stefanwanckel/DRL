import mujoco_py
import os
path =  os.path.join("urdf","ur5e.urdf")
model = mujoco_py.load_model_from_path(path)
sim = mujoco_py.MjSim(model)
