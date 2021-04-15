"""
Implements the Gym training environment in Pybullet
WidowX MK-II robot manipulator reaching a target position
IMPORTANT: THIS IS THE UR5E ENVIRONMENT. FOR SIMPLICITY, USING SAME FILENAMES AS FOR WIDOWX
"""

import os
import gym
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
import time


# Initial joint angles
RESET_VALUES_CART = [0.1,0.1,0.3]
#RESET_VALUES = [ 4.06453904e-01, -1.08030241e+00,  1.08329535e+00,  5.63995981e-01,-1.39857752e+00,  1]
RESET_VALUES = [
    0.015339807878856412,
    -1.2931458041875956,
    1.0109710760673565,
    -1.3537670644267164,
    -0.07158577010132992,
    0]
# Global Variables to use instead of __init__
# MIN_GOAL_COORDS = np.array([-0.45, -0.45, 0.45])
# MAX_GOAL_COORDS = np.array([0.45, 0.45, 0.1])
MIN_GOAL_COORDS = np.array([-0.1, -0.1, 0.1])
MAX_GOAL_COORDS = np.array([0.6, 0.6, 0.4])
MIN_END_EFF_COORDS = np.array([-0.90, -0.90, 0.10])
MAX_END_EFF_COORDS = np.array([0.90, 0.90, 0.90])
MIN_GOAL_ORIENTATION = np.array([-np.pi, -np.pi, -np.pi])
MAX_GOAL_ORIENTATION = np.array([np.pi, np.pi, np.pi])

#FIXED_GOAL_COORDS  = np.array([0.1, .4, 0.5])
FIXED_GOAL_COORDS_SPHERE = np.array([0.1, 0.4, 0.5])
FIXED_GOAL_COORDS_ARROW = np.array([0.1, 0.4, 0.5])
FIXED_GOAL_COORDS_MOVING = np.array([0.1, 0.4, 0.5])
FIXED_GOAL_ORIENTATION  = np.array([-np.pi/4, 0, -np.pi/2])
ARROW_OBJECT_ORIENTATION_CORRECTION = np.array([np.pi/2, 0, 0])

##Marius
PYBULLET_ACTION_MIN = [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05]
PYBULLET_ACTION_MAX = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
##Wir
# PYBULLET_ACTION_MIN = [-1, -1, -1, -1, -1, -1]
# PYBULLET_ACTION_MAX = [1, 1, 1, 1, 1, 1]
#Pierre
# PYBULLET_ACTION_MIN = [-0.05, -0.025, -0.025, -0.025, -0.05, 0]
# PYBULLET_ACTION_MAX = [0.05, 0.025, 0.025, 0.025, 0.05, 0.025]

class Ur5eEnv(gym.GoalEnv):
    """ Ur5e reacher Gym environment """


    def __init__(self,
                random_position,
                random_orientation,
                moving_target,
                target_type,
                goal_oriented,
                obs_type,
                reward_type,
                action_type,
                joint_limits,
                action_min,
                action_max,
                alpha_reward,
                reward_coeff,
                action_scale,
                eps,
                sim_rep,
                action_mode="set"
                ):
        """
        Initialise the environment
        """
        self.is_rendered = False
    
        self.random_position = random_position
        self.random_orientation = random_orientation
        self.moving_target = moving_target
        self.target_type = target_type
        self.reward_type = reward_type
        self.action_type = action_type
        self.obs_type = obs_type
        self.joint_limits = joint_limits
        self.goal_oriented = goal_oriented
        self.alpha_reward = alpha_reward
        self.action_min = action_scale*np.array(action_min)
        self.action_max = action_scale*np.array(action_max)
        self.reward_coeff = reward_coeff

        self.endeffector_pos = None
        self.old_endeffector_pos = None
        self.endeffector_orient = None
        self.old_endeffector_orient = None
        self.torso_pos = None
        self.torso_orient = None
        self.end_torso_pos = None
        self.end_goal_pos = None
        self.end_torso_orient = None
        self.end_goal_orient = None
        self.joint_positions = None
        self.delta_orient = None
        self.delta_endeff_orient = None
        self.goal_orient = None
        self.target_object_orient = None
        self.reward = None
        self.obs = None

        self.action = np.zeros(6)
        self.pybullet_action = np.zeros(6)
        self.pybullet_action_min = action_scale*np.array(PYBULLET_ACTION_MIN)
        self.pybullet_action_max = action_scale*np.array(PYBULLET_ACTION_MAX)
        self.new_joint_positions = None
        self.dist = 0
        self.old_dist = 0
        self.term1 = 0
        self.term2 = 0
        self.delta_pos = 0
        self.delta_dist = 0
        self.target_object_orient = None
        self.goal_pos = FIXED_GOAL_COORDS_SPHERE
        self.eps = eps
        self.sim_rep = sim_rep
        self.action_mode= action_mode

        # Initialise goal orientation
        # if self.random_orientation:
        #     self.goal_orient = self.sample_random_orientation()
        # else:
        

        # Define action space
        self.action_space = spaces.Box(
                low=np.float32(self.action_min),
                high=np.float32(self.action_max),
                dtype=np.float32)

        # Define observation space
        if self.joint_limits == "small":
            self.joint_min = np.array([-3.2, -3.2, -3.2, -3.2, -3.2, -3.2])
            self.joint_max = np.array([3.2, 3.2, 3.2, 3.2, 3.2, 3.2])
        elif self.joint_limits == "large":
            self.joint_min = 2*np.array([-3.2, -3.2, -3.2, -3.2, -3.2, -3.2])
            self.joint_max = 2*np.array([3.2, 3.2, 3.2, 3.2, 3.2, 3.2])

        #Define observations 
        if self.obs_type == 1:
            self.obs_space_low = np.float32(
                np.concatenate((MIN_END_EFF_COORDS, self.joint_min), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate((MAX_END_EFF_COORDS, self.joint_max), axis=0))

        elif self.obs_type == 2:
            self.obs_space_low = np.float32(
                np.concatenate((MIN_GOAL_COORDS, self.joint_min), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate((MAX_GOAL_COORDS, self.joint_max), axis=0))

        elif self.obs_type == 3:
            self.obs_space_low = np.float32(
                np.concatenate(([-1.0]*6, self.joint_min), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate(([1.0]*6, self.joint_max), axis=0))

        elif self.obs_type == 4:
            self.obs_space_low = np.float32(
                np.concatenate(([-1.0]*3, self.joint_min), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate(([1.0]*3, self.joint_max), axis=0))

        elif self.obs_type == 5:
            self.obs_space_low = np.float32(
                np.concatenate(([-1.0]*6, MIN_GOAL_COORDS, self.joint_min), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate(([1.0]*6, MAX_GOAL_COORDS, self.joint_max), axis=0))

        elif self.obs_type == 6:
            self.obs_space_low = np.float32(
                np.concatenate((
                    [-1.0]*6,
                    [-2*np.pi]*6,
                    MIN_GOAL_COORDS,
                    MIN_GOAL_ORIENTATION,
                    MIN_END_EFF_COORDS,
                    [-np.pi]*3,
                    self.joint_min
                    ), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate((
                    [1.0]*6,
                    [2*np.pi]*6,
                    MAX_GOAL_COORDS,
                    MAX_GOAL_ORIENTATION,
                    MAX_END_EFF_COORDS,
                    [np.pi]*3,
                    self.joint_max
                    ), axis=0))

        self.observation_space = spaces.Box(
                    low=self.obs_space_low,
                    high=self.obs_space_high,
                    dtype=np.float32)
        #IMPORTANT FOR HER
        if self.goal_oriented:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(
                    low=np.float32(MIN_GOAL_COORDS),
                    high=np.float32(MAX_GOAL_COORDS),
                    dtype=np.float32),
                achieved_goal=spaces.Box(
                    low=np.float32(MIN_END_EFF_COORDS),
                    high=np.float32(MAX_END_EFF_COORDS),
                    dtype=np.float32),
                observation=self.observation_space))

        # Connect to physics client. By default, do not render
        self.physics_client = p.connect(p.DIRECT)

        # Load URDFs
        self.create_world()

        # reset environment
        self.reset()


    def sample_random_position(self):
        """ Sample random target position """
        
        return np.random.uniform(low=MIN_GOAL_COORDS, high=MAX_GOAL_COORDS)

    def sample_random_orientation(self):
        """ Sample random target orientation """
        
        return np.random.uniform(low=MIN_GOAL_ORIENTATION, high=MAX_GOAL_ORIENTATION)

    def create_world(self):
        """ Setup camera and load URDFs"""

        # Initialise camera angle
        p.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=15,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5],
            physicsClientId=self.physics_client)

        # Load robot, target object and plane urdf
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        path = os.path.abspath(os.path.dirname(__file__))
        self.arm = p.loadURDF(
            os.path.join(
                path,
                "URDFs/ur_e_description/urdf/ur5e.urdf"),
            useFixedBase=True,basePosition=[0,0,0])

        if self.target_type == "arrow":
            self.target_object = p.loadURDF(
                os.path.join(
                    path,
                    "URDFs/arrow.urdf"),
                useFixedBase=True)
        elif self.target_type == "sphere":
            self.target_object = p.loadURDF(
                os.path.join(
                    path,
                    "URDFs/sphere.urdf"),
                useFixedBase=True)

        self.plane = p.loadURDF('plane.urdf')

        # Set gravity
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        #p.setGravity(0, 0, 0, physicsClientId=self.physics_client)
        #self._force_joint_positions(RESET_VALUES)

    def reset(self):
        """
        Reset robot and goal at the beginning of an episode.
        Returns observation
        """
        # Initialise goal position
        if self.random_position:
            self.goal_pos = self.sample_random_position()
        else:
            if self.moving_target:
                # deepcopy is necessary to avoid changing the value of FIXED_GOAL_COORDS_MOVING
                self.goal_pos = copy.deepcopy(FIXED_GOAL_COORDS_MOVING)
            else:
                if self.target_type == "arrow":
                    self.goal_pos = FIXED_GOAL_COORDS_ARROW
                elif self.target_type == "sphere":
                    self.goal_pos = FIXED_GOAL_COORDS_SPHERE

        # Initialise goal orientation
        if self.random_orientation:
            self.goal_orient = self.sample_random_orientation()
        else:
            self.goal_orient = FIXED_GOAL_ORIENTATION
        
        # Correct the orientation of the target object for consistency with rendering
        # in Pybullet (This is due to the arrow's STL being oriented along a different axis)
        self.target_object_orient = self.goal_orient + ARROW_OBJECT_ORIENTATION_CORRECTION

        # Reset robot at the origin and move the target object to the goal position and orientation
        p.resetBasePositionAndOrientation(
            self.arm, [0, 0, 0], p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
        p.resetBasePositionAndOrientation(
            self.target_object, self.goal_pos, p.getQuaternionFromEuler(self.target_object_orient))

        # Reset joint at initial angles
        self._force_joint_positions(RESET_VALUES)

        # Get observation
        if self.obs_type == 1:
            self.obs = self._get_obs1()
        elif self.obs_type == 2:
            self.obs = self._get_obs2()
        elif self.obs_type == 3:
            self.obs = self._get_obs3()
        elif self.obs_type == 4:
            self.obs = self._get_obs4()
        elif self.obs_type == 5:
            self.obs = self._get_obs5()
        elif self.obs_type == 6:
            self.obs = self._get_obs6()

        # update observation if goal oriented environment
        if self.goal_oriented:
            self.obs = self._get_goal_oriented_obs()

        return self.obs

    def step(self, action):
        """
        Execute the action and return obs, reward, episode_over, info (tuple)
        Parameters
        ----------
        action (array)
        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (array)
            reward (float)
            episode_over (bool)
            info (dict)
        """

        # get distance and end effector position before taking the action
        self.old_dist = np.linalg.norm(self.endeffector_pos - self.goal_pos)
        self.old_endeffector_pos = self.endeffector_pos
        self.old_orient = np.linalg.norm(self.endeffector_orient - self.goal_orient)
        self.old_endeffector_orient = self.endeffector_orient

        # Update target position and move the target object
        if self.moving_target:
            self.goal_pos[1] += TARGET_SPEED
            p.resetBasePositionAndOrientation(
                self.target_object,
                self.goal_pos,
                p.getQuaternionFromEuler(self.target_object_orient)) 

        # take action
        self.action = np.array(action, dtype=np.float32)

        # Scale action to pybullet range
        self._scale_action_pybullet()

        self._take_action()

      # get observation
        if self.obs_type == 1:
            self.obs = self._get_obs1()
        elif self.obs_type == 2:
            self.obs = self._get_obs2()
        elif self.obs_type == 3:
            self.obs = self._get_obs3()
        elif self.obs_type == 4:
            self.obs = self._get_obs4()
        elif self.obs_type == 5:
            self.obs = self._get_obs5()
        elif self.obs_type == 6:
            self.obs = self._get_obs6()  

        # update observation if goal oriented environment
        if self.goal_oriented:
            self.obs = self._get_goal_oriented_obs()

        # get new distance
        self.dist = np.linalg.norm(self.endeffector_pos - self.goal_pos)
        self.orient = np.linalg.norm(self.endeffector_orient - self.goal_orient)

        # get reward
        if self.reward_type == 1:
            self.reward = self._get_reward1()
        elif self.reward_type == 2:
            self.reward = self._get_reward2()
        elif self.reward_type == 3:
            self.reward = self._get_reward3()
        elif self.reward_type == 4:
            self.reward = self._get_reward4()
        elif self.reward_type == 5:
            self.reward = self._get_reward5()
        elif self.reward_type == 6:
            self.reward = self._get_reward6()
        elif self.reward_type == 7:
            self.reward = self._get_reward7()
        elif self.reward_type == 8:
            self.reward = self._get_reward8()
        elif self.reward_type == 9:
            self.reward = self._get_reward9()
        elif self.reward_type == 10:
            self.reward = self._get_reward10()
        elif self.reward_type == 11:
            self.reward = self._get_reward11()
        elif self.reward_type == 12:
            self.reward = self._get_reward12()
        elif self.reward_type == 13:
            self.reward = self._get_reward13()
        elif self.reward_type == 14:
            self.reward = self._get_reward14()
        elif self.reward_type == 15:
            self.reward = self._get_reward15()
        elif self.reward_type == 16:
            self.reward = self._get_reward16()
        elif self.reward_type == 17:
            self.reward = self._get_reward17()
        elif self.reward_type == 18:
            self.reward = self._get_reward18()
        elif self.reward_type == 19:
            self.reward = self._get_reward19()

        # Apply reward coefficient
        self.reward *= self.reward_coeff

        # Create info
        self.delta_dist = self.old_dist - self.dist
        self.delta_pos = np.linalg.norm(self.old_endeffector_pos - self.endeffector_pos)
        self.delta_orient = self.old_orient - self.orient
        self.delta_orient = np.linalg.norm(self.old_endeffector_orient - self.endeffector_orient)

        info = {}
        info['distance'] = self.dist
        info['goal_pos'] = self.goal_pos
        info['endeffector_pos'] = self.endeffector_pos
        info['orientation'] = self.orient
        info['goal_orient'] = self.goal_orient
        info['endeffector_orient'] = self.endeffector_orient
        info['joint_pos'] = self.joint_positions
        info['joint_min'] = self.joint_min
        info['joint_max'] = self.joint_max
        info['term1'] = self.term1
        info['term2'] = self.term2
        info['action'] = self.action
        info['action_min'] = self.action_min
        info['action_max'] = self.action_max
        info['pybullet_action'] = self.pybullet_action
        info['pybullet_action_min'] = self.pybullet_action_min
        info['pybullet_action_max'] = self.pybullet_action_max
        # According to the Pybullet documentation, 1 timestep = 240 Hz
        info['vel_dist'] = self.delta_dist * 240
        info['vel_pos'] = self.delta_pos * 240

        # Create "episode_over": never end episode prematurily
        episode_over = False
        # if self.dist < self.eps:
        #     episode_over = True

        return self.obs, self.reward, episode_over, info

    def render(self, mode='human'):
        """ Render Pybullet simulation """
        if not self.is_rendered:
            p.disconnect(self.physics_client)
            self.physics_client = p.connect(p.GUI)
            self.create_world()
            self.is_rendered = True


# helper functions in RESET and STEP
    def _get_general_obs(self):
        """ Get information for generating observation array """
        self.endeffector_pos = self._get_end_effector_position()
        self.endeffector_orient = self._get_end_effector_orientation()
        self.torso_pos = self._get_torso_position()
        self.torso_orient = self._get_torso_orientation()
        self.end_torso_pos = self.endeffector_pos - self.torso_pos
        self.end_goal_pos = self.endeffector_pos - self.goal_pos
        self.end_torso_orient = self.endeffector_orient - self.torso_orient
        self.end_goal_orient = self.endeffector_orient - self.goal_orient
        self.joint_positions = self.get_joint_positions()

    def _get_obs1(self):
        """ Returns observation #1 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.endeffector_pos, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs2(self):
        """ Returns observation #2 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.goal_pos, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs3(self):
        """ Returns observation #3 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.end_torso_pos, self.end_goal_pos, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs4(self):
        """ Returns observation #4 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.end_goal_pos, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs5(self):
        """ Returns observation #5 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.end_torso_pos, self.end_goal_pos, self.goal_pos, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs6(self):
        """ Returns observation #6 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [
                self.end_torso_pos,
                self.end_goal_pos,
                self.end_torso_orient,
                self.end_goal_orient,
                self.goal_pos,
                self.goal_orient,
                self.endeffector_pos,
                self.endeffector_orient,
                self.joint_positions
                ]).ravel()

        return robot_obs

        robot_obs = np.concatenate(
            [self.endeffector_pos, self.joint_positions]).ravel()

        return robot_obs
#which range should be used here?
    def get_joint_positions(self):
        """ Return current joint position """
        return np.array([x[0] for x in p.getJointStates(self.arm, range(1,7))])

    def _get_end_effector_position(self):
        """ Get end effector coordinates """
        return np.array(p.getLinkState(
                self.arm,
                6,
                computeForwardKinematics=True)
            [0])

    def _get_end_effector_orientation(self):
        """ Get end effector orientation """
        orient_quat = p.getLinkState(self.arm, 6, computeForwardKinematics=True)[1]
        orient_euler = p.getEulerFromQuaternion(orient_quat)
        return np.array(orient_euler)

    def _get_torso_position(self):
        """ Get torso coordinates """
        return np.array(p.getLinkState(
                self.arm,
                0,
                computeForwardKinematics=True)
            [0])

    def _get_torso_orientation(self):
        """ Get torso orientation """
        orient_quat = p.getLinkState(self.arm, 0, computeForwardKinematics=True)[1]
        orient_euler = p.getEulerFromQuaternion(orient_quat)
        return np.array(orient_euler)

    def _get_goal_oriented_obs(self):
        """ return goal_oriented observation """
        obs = {}
        obs['observation'] = self.obs
        obs['achieved_goal'] = self.endeffector_pos
        obs['desired_goal'] = self.goal_pos
        return obs

    def _take_action(self):
        """ select action #1 (increments from previous joint position """
        # Update the new joint position with the action
        self.new_joint_positions = self.joint_positions + self.pybullet_action

        # Clip the joint position to fit the joint's allowed boundaries
        self.new_joint_positions = np.clip(
            np.array(self.new_joint_positions),
            self.joint_min,
            self.joint_max)

        # Instantaneously reset the joint position (no torque applied)
        #changed by me to set instead of force
        if self.action_mode == "set":
            self._set_joint_positions(self.new_joint_positions)
        elif self.action_mode == "force":
            self._force_joint_positions(self.new_joint_positions)

    def _normalize_scalar(self, var, old_min, old_max, new_min, new_max):
        """ Normalize scalar var from one range to another """
        return var / (old_max - old_min) * (new_max - new_min)
    
    def _scale_action_pybullet(self):
        """ Scale action to Pybullet action range """
        for i in range(6):
            self.pybullet_action[i] = self._normalize_scalar(
                self.action[i],
                self.action_min[i],
                self.action_max[i],
                self.pybullet_action_min[i],
                self.pybullet_action_max[i])

    def _get_reward1(self):
        """ Compute reward function 1 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward2(self):
        """ Compute reward function 2 (dense) """
        self.term1 = - self.dist
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward3(self):
        """ Compute reward function 3 (dense) """
        self.term1 = - self.dist ** 3
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward4(self):
        """ Compute reward function 4 (dense) """
        self.term1 = - self.dist ** 4
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward5(self):
        """ Compute reward function 5 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha_reward * np.linalg.norm(self.action)
        rew = self.term1 + self.term2
        return rew

    def _get_reward6(self):
        """ Compute reward function 6 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha_reward * np.linalg.norm(self.action) / self.dist ** 2
        rew = self.term1 + self.term2
        return rew

    def _get_reward7(self):
        """ Compute reward function 7 (dense) """
        self.term1 = self.delta_dist
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward8(self):
        """ Compute reward function 8 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = self.alpha_reward * abs(self.delta_dist / self.dist)
        rew = self.term1 + self.term2
        return rew

    def _get_reward9(self):
        """ Compute reward function 9 (dense) """
        self.term1 = self.delta_pos
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward10(self):
        """ Compute reward function 10 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha_reward * self.delta_pos / self.dist
        rew = self.term1 + self.term2
        return rew

    def _get_reward11(self):
        """ Compute reward function 11 (sparse) """
        if self.dist >= 0.001:
            self.term1 = -1
        else:
            self.term1 = 0
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward12(self):
        """ Compute reward function 12 (sparse) """
        if self.dist >= 0.001:
            self.term1 = 0
        else:
            self.term1 = 1
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward13(self):
        """ Compute reward function 13 (sparse) """
        if self.dist >= self.eps:
            self.term1 = -0.02
        else:
            self.term1 = 10
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward14(self):
        """ Compute reward function 14 (sparse) """
        if self.dist >= 0.001:
            self.term1 = -0.001
        else:
            self.term1 = 10
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward15(self):
        """ Compute reward function 15 (sparse + dense) """
        if self.dist >= self.eps:
            self.term1 = - self.dist
        else:
            self.term1 = 1
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward16(self):
        """ Compute reward function 16 (sparse + dense) """
        if self.dist >= 0.001:
            self.term1 = self.delta_dist
        else:
            self.term1 = self.delta_dist + 10
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward17(self):
        """ Compute reward function 17 (dense) """
        self.term1 = - self.orient ** 2
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward18(self):
        """ Compute reward function 18 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha_reward * self.orient ** 2
        rew = self.term1 + self.term2
        return rew

    def _get_reward19(self):
        """ Compute reward function 19 (sparse + dense) """
        if self.dist >= 0.001 and self.orient >= 0.001:
            self.term1 = - self.dist **2 - self.alpha_reward * self.orient ** 2
        else:
            self.term1 = 1
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew


    def compute_reward(self, achieved_goal, goal, info):
        """ Function necessary for goal Env"""
        return - (np.linalg.norm(achieved_goal - goal, axis=1)**2) # STEFAN: modified added axis=1

    def _set_joint_positions(self, joint_positions):
        """ Position control (not reset) """
        # In Pybullet, gripper halves are controlled separately
        #joint_positions = list(joint_positions) + [joint_positions[-1]]
        p.setJointMotorControlArray(
            self.arm,
            [1, 2, 3, 4, 5, 6],
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions
        )
        for _ in range(self.sim_rep):
            p.stepSimulation()

    def _force_joint_positions(self, joint_positions):
        """ Instantaneous reset of the joint angles (not position control) """
        for i in range(6):
            p.resetJointState(
                self.arm,
                i+1,
                joint_positions[i]
            )
        # In Pybullet, gripper halves are controlled separately
        #We dont have a gripper
        # for i in range(8, 10):
        #     p.resetJointState(
        #         self.arm,
        #         i,
        #         joint_positions[-1]
        #     )
    def get_revolute_joints_indices(self):
        lstRevJointsIndices = []
        #print("Number of Joints", p.getNumJoints(self.arm))
        for index,joint in enumerate(range(p.getNumJoints(self.arm))):
            #print(index, p.getJointInfo(self.arm,joint)[2],p.getJointInfo(self.arm,joint)[1])
            if p.getJointInfo(self.arm,joint)[2] == 0:
                lstRevJointsIndices.append(index)
            if str(p.getJointInfo(self.arm,joint)[12],'utf-8') == "ee_link":
                endEffectorIndex = index
                #print(endEffectorIndex)
        #print(lstRevJointsIndices)
        return lstRevJointsIndices, endEffectorIndex

    def calc_inv_kin(self):
        #reset_joint_angles = p.calculateInverseKinematics(self.arm,self.get_revolute_joints_indices()[1],RESET_VALUES_CART)
        #taking joint 6 as gripper because no gripper
        reset_joint_angles = p.calculateInverseKinematics(self.arm,7,RESET_VALUES_CART)
        return reset_joint_angles

    def set_joint_positions(self, joint_positions):
        """ Position control (not reset) """
        # In Pybullet, gripper halves are controlled separately
        #joint_positions = list(joint_positions) + [joint_positions[-1]]
        p.setJointMotorControlArray(
            self.arm,
            [1, 2, 3, 4, 5, 6],
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions
        )
        for _ in range(self.sim_rep):
            p.stepSimulation()
    def get_end_effector_position(self):
        """ Get end effector coordinates """
        return np.array(p.getLinkState(
                self.arm,
                6,
                computeForwardKinematics=True)
            [0])


    def test_arm(self):
        #print info and state before motion
        iAndS_old = self._get_joint_list_info_and_state_as_dict()
        self._print_info_and_state(iAndS_old)
        #define position, motion type and do a simulation step
        joint2Move = 1
        actionSpike = 0.1
        deltaAction = np.zeros(6) 
        for i in range(0,deltaAction.shape[0]):
            if i+1 == joint2Move:
                deltaAction[i]=actionSpike
                break

        new_joint_positions = np.array(RESET_VALUES) + deltaAction
        p.setJointMotorControlArray(
            self.arm,
            [1, 2, 3, 4, 5, 6],
            controlMode=p.POSITION_CONTROL,
            targetPositions=new_joint_positions,
            forces=10000*np.ones(6)
        )
        for i in range(100):
            p.stepSimulation()
            #time.sleep(1/40)

        iAndS_new = self._get_joint_list_info_and_state_as_dict()
        self._print_info_and_state(iAndS_new)
        for i,_ in enumerate(iAndS_new):
            currDeltaPos = iAndS_new[i]["jointPosition"]-iAndS_old[i]["jointPosition"]
            print(i,currDeltaPos)

    def _get_joint_list_info_and_state_as_dict(self):

        jointType = {
                    0: "JOINT_REVOLUTE",
                    1: "JOINT_PRISMATIC",
                    2: "JOINT_SPHERICAL",
                    3: "JOINT_PLANAR",
                    4: "JOINT_FIXED"
                    }
        #initialize jointList which is a list of dictionaries for every joint
        lstJointInfoAndState = []
        #get total number of joints in robot id
        n_joints = p.getNumJoints(self.arm)
        print("Total number of joints is ",n_joints)
        
        for i in range(0,n_joints):
            #get current joint
            currJointInfo = p.getJointInfo(self.arm, i)
            currJointState = p.getJointState(self.arm,i)
            #write current joint info to dictionary
            currJointInfo = {
                        "jointIndex": currJointInfo[0],
                        "jointName": currJointInfo[1],
                        "jointType": jointType[currJointInfo[2]],
                        "qIndex": currJointInfo[3],
                        "uIndex": currJointInfo[4],
                        "jointLowerLimit" : currJointInfo[8],
                        "jointUpperLimit" : currJointInfo[9],
                        "jointAxis" : currJointInfo[13],
                        "parentFramePos" : currJointInfo[14],
                        "parentIndex" : currJointInfo[15]
                        }

            currJointState = {
                        "jointPosition" : currJointState[0],
                        "jointVelocity" : currJointState[1],
                        "jointReactionForces" : currJointState[2],
                        "appliedJointMotorTorque" : currJointState[3]
                             }
            currJointInfoAndState = currJointInfo.copy()
            currJointInfoAndState.update(currJointState)
            lstJointInfoAndState.append(currJointInfoAndState)

        return lstJointInfoAndState

    def _print_info_and_state(self, infoAndState):

        for i, jointInfoAndState in enumerate(infoAndState):
            print("***** Info and state to joint ",i, " *****")
            for key in jointInfoAndState:
                print(key, jointInfoAndState[key])
    
    # def get_link_state(self)

    #     currLinkState = p.getLinkState
    #     currLinkState = {
    #                     "LinkWorldPosition" : 
    #     }

    
                