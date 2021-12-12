import numpy as np
import math
from ur5_env_mjco.envs import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class Ur5Env(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, table_height=None, max_pos_change=0.05
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.table_height = table_height
        self.max_pos_change = max_pos_change

        super(Ur5Env, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            if self.has_object:
                pass
            else:
                if "no_gripper" in self.model_path:
                    pass
                else:
                    self.sim.data.set_joint_qpos(
                        'robot0:l_gripper_finger_joint', 0.)
                    self.sim.data.set_joint_qpos(
                        'robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= self.max_pos_change  # limit maximum change in position
        if self.has_object:
            if "no_gripper" in self.model_path:
                rot_ctrl = [0, 0., -1., 1.]
            else:
                rot_ctrl = [1., 0., 0., 0.]
        else:
            if "no_gripper" in self.model_path:
                rot_ctrl = [0, 0., -1., 1.]
            else:
                rot_ctrl = [1., 0., 0., 0.]
        # rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        # for gripper
        utils.ctrl_set_action(self.sim, action)
        # for rest of robot
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(
                self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(
                0)
        if "no_gripper" in self.model_path:
            gripper_state = np.array([0, 0])
            gripper_vel = np.array([0, 0])
            # gripper_state = np.zeros(0)
            # gripper_vel = np.zeros(0)
        else:
            gripper_state = robot_qpos[-2:]
            gripper_vel = robot_qvel[-2:] * dt

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(
            ), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        if self.has_object:
            body_id = self.sim.model.body_name2id('robot0:wrist_3_link')
        else:
            if "no_gripper" in self.model_path:
                body_id = self.sim.model.body_name2id('dummy_gripper')
            else:
                body_id = self.sim.model.body_name2id('robot0:gripper_link')

        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = -180.
        self.viewer.cam.elevation = -25.
        self.viewer._video_path = "/home/stefan/Documents/Masterarbeit/DRL/video_%07d.mp4"
        self.viewer._image_path = "/home/stefan/Documents/Masterarbeit/DRL/frame_%07d.png"

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos -
                        self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[site_id]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            # object_xpos = self.initial_gripper_xpos[:2]
            # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.8*self.obj_range:
            #     # [-0.34248927  0.4682842 ]
            #     object_xpos = self.initial_gripper_xpos[:2] + [-0.5, 0]
            #     print(object_xpos)
            #     # object_xpos = object_xpos + \
            #     #     self.np_random.uniform(-self.obj_range,
            #     #                            self.obj_range, size=2)
            #     print(np.linalg.norm(object_xpos -
            #           self.initial_gripper_xpos[:2]))
            # IMPORTANT: Apparently, self.set_joint_qpos sets the joints relative to its OWN coordinate frame.
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')

            object_qpos[:3] = np.array([-0.1, -0.1, 0])
            #print("object_qpos: ", object_qpos)
            assert object_qpos.shape == (7,)
            # object_qpos[:2] += self.np_random.uniform(-self.obj_range,
            #                                           self.obj_range, size=2)
            # object_qpos[2] = 0.5  # self.table_height  # + self.height_offset
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:

            #goal = self.initial_gripper_xpos[:3]
            goal = np.array([-0.33,  0.48,  0.73])
            goal[2] = 0.51
            #print("goal_prior: ", goal)
            object_xpos = self.sim.data.get_site_xpos('object0')
            new_goal = goal
            while np.linalg.norm(object_xpos - new_goal) < 0.8*self.target_range:
                new_goal[:2] = goal[:2]+self.np_random.uniform(-self.target_range,
                                                               self.target_range, size=2)
            goal = new_goal

            #print("goal_posterior: ", goal)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            #goal = object_qpos[:3]

            # goal += self.target_offset
            if self.table_height is not None:
                goal[2] = 0.51
            else:
                goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                # if self.target_in_the_air:
                goal[2] += self.np_random.uniform(0.1, 0.2)
        else:
            goal = self.initial_gripper_xpos[:3] + \
                self.np_random.uniform(-self.target_range,
                                       self.target_range, size=3)

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        # gripper_target = np.array([0.75,0.,0.4 + self.gripper_extra_height]) #+ self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = self.sim.data.get_site_xpos(
            'robot0:grip') + [0, 0, self.gripper_extra_height]
        #gripper_rotation = np.array([1., 0., 1., 0.])
        #rot_eul = [math.pi,math.pi/2,0]
        #gripper_rotation = rotations.euler2quat(rot_eul)
        if self.has_object:
            if "no_gripper" in self.model_path:
                gripper_rotation = np.array([0., 0., -1., 1.])
            else:
                gripper_rotation = np.array([1., 0., 0., 0.])
        else:
            if "no_gripper" in self.model_path:
                gripper_rotation = np.array([0., 0., -1., 1.])
            else:
                gripper_rotation = np.array([1., 0., 0., 0.])

        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        # for _ in range(100):
        self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos(
            'robot0:grip').copy()

    def render(self, mode='human', width=500, height=500):
        return super(Ur5Env, self).render(mode, width, height)
