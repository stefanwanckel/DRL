import pybullet as p
import pybullet_data
# Can alternatively pass in p.DIRECT 
client = p.connect(p.GUI)
p.setGravity(0, 0, -10, physicsClientId=client) 

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
#carId = p.loadURDF(“racecar/racecar.urdf”, basePosition=[0,0,0.2])
for _ in range(100): 
    # pos, ori = p.getBasePositionAndOrientation(__A__)
    # p.applyExternalForce(__B__, 0, [50, 0, 0], __C__, p.WORLD_FRAME)
    p.stepSimulation()