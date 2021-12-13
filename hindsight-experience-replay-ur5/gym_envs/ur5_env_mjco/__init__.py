""" Register Gym environments """

import numpy as np
from gym.envs.registration import register


register(
    id='ur5_reach-v1',
    entry_point='ur5_env_mjco.envs.ur5_tasks.reach:Ur5ReachEnv',
    # kwargs=kwargs,
    max_episode_steps=100,
)
register(
    id='ur5_reach_no_gripper-v1',
    entry_point='ur5_env_mjco.envs.ur5_tasks.reach_no_gripper:Ur5ReachEnv',
    # kwargs=kwargs,
    max_episode_steps=100,
)

register(
    id='ur5_push-v1',
    entry_point='ur5_env_mjco.envs.ur5_tasks.push:Ur5PushEnv',
    # kwargs=kwargs,
    max_episode_steps=100,
)

register(
    id='ur5_push_no_gripper-v1',
    entry_point='ur5_env_mjco.envs.ur5_tasks.push_no_gripper:Ur5PushEnv',
    # kwargs=kwargs,
    max_episode_steps=100,
)

register(
    id='ur5_pick_and_place-v1',
    entry_point='ur5_env_mjco.envs.ur5_tasks.pick_and_place:Ur5PickAndPlaceEnv',
    # kwargs=kwargs,
    max_episode_steps=100,
)

register(
    id='ur5_slide-v1',
    entry_point='ur5_env_mjco.envs.ur5_tasks.slide:Ur5SlideEnv',
    # kwargs=kwargs,
    max_episode_steps=100,
)
