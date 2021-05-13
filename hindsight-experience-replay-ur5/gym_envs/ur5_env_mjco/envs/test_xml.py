import mujoco_py
import os
import time
import ur5_env_mjco

MODEL_XML_PATH = os.path.join("assets","ur5", "reach.xml")

model = mujoco_py.load_model_from_path(MODEL_XML_PATH)
sim = mujoco_py.MjSim(model)
viewer=mujoco_py.MjViewer(sim)
state = sim.get_state()
print(sim.model.joint_names)

print(state)
#shoulderPos = sim.get_body_xpos("")
viewer.cam.distance = 3.5
viewer.cam.azimuth = 132.
viewer.cam.elevation = -10.
viewer.render()
#for _ in range(10000) :
while True:   
    sim.step()