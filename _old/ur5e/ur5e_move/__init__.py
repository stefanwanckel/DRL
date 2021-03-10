from gym.envs.registration import register
from gym.envs.robotics.ur5e-move.ur5e_move.envs.ur5e_move_env import MyEnv


register(
    id='gym_ur5e:ur5e-v0', 
    entry_point='ur5e_move.envs:Ur5eMoveEnv'
)