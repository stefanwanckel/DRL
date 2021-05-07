import mujoco_py
import gym
import time


class myEnv(gym.GoalEnv):
    def __init(self):
        self.a = "test"
        super(myEnv,self).__init__()
        self.observation_space = None
    

    def step(self,action,reaction):
        pass

    def reset(self):

        return print("it overwrites")
sampleEnv = myEnv()
sampleEnv.reset()
