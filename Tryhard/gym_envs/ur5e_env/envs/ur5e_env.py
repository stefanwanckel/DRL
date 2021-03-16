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
RESET_VALUES = [ 4.06453904e-01, -1.08030241e+00,  1.08329535e+00,  5.63995981e-01,
       -1.39857752e+00,  7.76366484e-17]
# Global Variables to use instead of __init__
MIN_GOAL_ORIENTATION = np.array([-np.pi, -np.pi, -np.pi])
MAX_GOAL_ORIENTATION = np.array([np.pi, np.pi, np.pi])
MIN_GOAL_COORDS = np.array([-0.1, -0.1, 0.1])
MAX_GOAL_COORDS = np.array([0.6, 0.6, 0.4])
#MIN_GOAL_COORDS = np.array([-.5, -.5, 0.1])
#MAX_GOAL_COORDS = np.array([.5, .5, .5])
MIN_END_EFF_COORDS = np.array([-.90, -.90, 0.10])
MAX_END_EFF_COORDS = np.array([.90, .90, .90])
FIXED_GOAL_COORDS  = np.array([0.1, .4, 0.5])
FIXED_GOAL_ORIENTATION  = np.array([-np.pi/4, 0, -np.pi/2])
ARROW_OBJECT_ORIENTATION_CORRECTION = np.array([np.pi/2, 0, 0])
JOINT_LIMITS = "small"
ACTION_MIN = [-0.03, -0.03, -0.03, -0.03, -0.03, -0.03]
ACTION_MAX = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
PYBULLET_ACTION_MIN = [-0.03, -0.03, -0.03, -0.03, -0.03, -0.03]
PYBULLET_ACTION_MAX = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
GOAL_ORIENTED = True
REWARD_TYPE = 13
ACTION_SCALE = 1
RANDOM_GOAL = True

class Ur5eEnv(gym.GoalEnv):
    """ Ur5e reacher Gym environment """


    def __init__(self):
        """
        Initialise the environment
        """
        self.is_rendered = False
        self.joint_limits = JOINT_LIMITS
        self.action_min = ACTION_SCALE*np.array(ACTION_MIN)
        self.action_max = ACTION_SCALE*np.array(ACTION_MAX)
        #self.alpha_reward = alpha_reward
        self.reward_type = REWARD_TYPE
        self.goal_oriented = GOAL_ORIENTED
        self.endeffector_pos = None
        self.old_endeffector_pos = None
        self.endeffector_orient = None
        self.torso_pos = None
        self.torso_orient = None
        self.end_torso_pos = None
        self.end_goal_pos = None
        self.end_torso_orient = None
        self.end_goal_orient = None
        self.joint_positions = None
        self.reward = None
        self.obs = None
        self.action = np.zeros(6)
        self.pybullet_action = np.zeros(6)
        self.pybullet_action_min = ACTION_SCALE*np.array(PYBULLET_ACTION_MIN)
        self.pybullet_action_max = ACTION_SCALE*np.array(PYBULLET_ACTION_MAX)
        self.new_joint_positions = None
        self.dist = 0
        self.old_dist = 0
        self.term1 = 0
        self.term2 = 0
        self.delta_pos = 0
        self.delta_dist = 0
        self.target_object_orient = None
        self.goal_pos = FIXED_GOAL_COORDS
        self.random_goal = RANDOM_GOAL
        # Initialise goal orientation
        # if self.random_orientation:
        #     self.goal_orient = self.sample_random_orientation()
        # else:
        self.goal_orient = FIXED_GOAL_ORIENTATION

        # Define action space
        self.action_space = spaces.Box(
                low=np.float32(self.action_min),
                high=np.float32(self.action_max),
                dtype=np.float32)

        # Define observation space
        if self.joint_limits == "small":
            self.joint_min = np.array([-3.1, -1.6, -1.6, -1.8, -3.1, 0.0])
            self.joint_max = np.array([3.1, 1.6, 1.6, 1.8, 3.1, 0.0])
        elif self.joint_limits == "large":
            self.joint_min = 2*np.array([-3.2, -3.2, -3.2, -3.2, -3.2, -3.2])
            self.joint_max = 2*np.array([3.2, 3.2, 3.2, 3.2, 3.2, 3.2])

        self.obs_space_low = np.float32(
            np.concatenate((MIN_END_EFF_COORDS, self.joint_min), axis=0))
        self.obs_space_high = np.float32(
            np.concatenate((MAX_END_EFF_COORDS, self.joint_max), axis=0))

        self.observation_space = spaces.Box(
                    low=self.obs_space_low,
                    high=self.obs_space_high,
                    dtype=np.float32)

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
        #only "+" is neded since min_goal_coords is negative
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

        self.target_object = p.loadURDF(
            os.path.join(
                path,
                "URDFs/sphere.urdf"),
            useFixedBase=True)
        self.plane = p.loadURDF('plane.urdf')
        #self.box = p.loadURDF('URDFs/cube/cube_small.urdf',useFixedBase=True,basePosition=[0,0,0])
        # Reset robot at the origin and move the target object to the goal position and orientation
        # Note: the arrow's STL is oriented  along a different axis and its
        # orientation vector must be corrected (for consistency with the Pybullet rendering)
        #p.resetBasePositionAndOrientation(
        #    self.arm, [0, 0, 0], p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
        p.resetBasePositionAndOrientation(
            self.target_object, self.goal_pos, p.getQuaternionFromEuler([0, 0, 1]))

        # Set gravity
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        # Reset joint at initial angles
        #self.reset_values = self.calc_inv_kin()
        self._force_joint_positions(RESET_VALUES)

    def reset(self):
        """
        Reset robot and goal at the beginning of an episode.
        Returns observation
        """
        #reset values for robot are defined earlier to make goal position around it
        #self.reset_values = self.calc_inv_kin()
        # Initialise goal position
        if self.random_goal:
            self.goal_pos = self.sample_random_position()
        else:
            self.goal_pos = FIXED_GOAL_COORDS

        # Reset robot at the origin and move the target object to the goal position and orientation
        # Note: the arrow's STL is oriented  along a different axis and its
        # orientation vector must be corrected (for consistency with the Pybullet rendering)
        #p.resetBasePositionAndOrientation(
            #self.arm, [0, 0, 0], p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
        p.resetBasePositionAndOrientation(
            self.target_object, self.goal_pos, p.getQuaternionFromEuler([0, 0, 1]))

        # Reset joint at initial angles
       
        self._force_joint_positions(RESET_VALUES)

        # Get observation
        self.obs = self._get_obs()
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

        # take action
        self.action = np.array(action, dtype=np.float32)

        # Scale action to pybullet range
        self._scale_action_pybullet()

        self._take_action()

        # get observation
        self.obs = self._get_obs()

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
        # if self.new_distance < 0.0005:
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
        self.joint_positions = self._get_joint_positions()

    def _get_obs(self):
        """ Returns observation #1 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.endeffector_pos, self.joint_positions]).ravel()

        return robot_obs

    def _get_joint_positions(self):
        """ Return current joint position """
        return np.array([x[0] for x in p.getJointStates(self.arm, range(1,7))])

    def _get_end_effector_position(self):
        """ Get end effector coordinates """
        return np.array(p.getLinkState(
                self.arm,
                7,
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
        self._set_joint_positions(self.new_joint_positions)

    def _normalize_scalar(self, var, old_min, old_max, new_min, new_max):
        """ Normalize scalar var from one range to another """
        return ((new_max - new_min) * (var - old_min) / (old_max - old_min)) + new_min

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
        if self.dist >= 0.1:
            self.term1 = -0.02
        else:
            self.term1 = 1
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
        if self.dist >= 0.001:
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
        return - (np.linalg.norm(achieved_goal - goal)**2)

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
        for _ in range(5):
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

    #def goal_pos_around_end_effector(self):
        


# myRobot = Ur5eEnv()
# print(myRobot.get_revolute_joints_indices())
