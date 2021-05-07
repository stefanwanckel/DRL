import mujoco_py
import os
import time

MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')

model = mujoco_py.load_model_from_path(MODEL_XML_PATH)
sim = mujoco_py.MjSim(model)
mujoco_py.MjViewer(sim).render()
while True:
    sim.step()