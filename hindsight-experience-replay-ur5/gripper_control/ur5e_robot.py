#!/usr/bin/env python
import os
import time
from enum import IntEnum, unique
import numpy as np
#import rospy
import yaml
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_io import RTDEIOInterface as RTDEIO
from rtde_receive import RTDEReceiveInterface as RTDEReceive
#from sensor_msgs.msg import JointState


@unique
class OutputDoubleRegisters(IntEnum):
    # registers to set values for grip
    RG2_OUTPUT_ACTUAL_WIDTH = 18


@unique
class InputIntRegisters(IntEnum):
    # register to write input state for rg2 gripper
    RG2_INPUT_STATE_COMMAND = 18


@unique
class OutputIntRegisters(IntEnum):
    # register to read output state from rg2 gripper
    RG2_OUTPUT_STATE_COMMAND = 18
    RG2_GRIP_DETECTED = 19  # 1 or 0 (True or False)


@unique
class InputDoubleRegisters(IntEnum):
    # registers to set values for grip
    RG2_INPUT_TARGET_WIDTH = 18
    RG2_INPUT_TARGET_FORCE = 19

# add ur5erobot class to real_demo_pick_n_place
# add yaml file (jason)


class Ur5eRobot:
    def __init__(self, robot_namespace, robot_ip, robot_port, file, base_pose, control_mode="local"):

        self.robot_ns = robot_namespace
        self.robot_ip = robot_ip
        self.robot_port = int(robot_port)
        self.log = getattr(self, 'log_default')
        self.max_joint_velocity = 3
        self.max_joint_acceleration = 0.2

        # connect to the controller on the UR, use External UR CAP
        # necessary to use the functions of external UR CAPs, such as grippers
        # also allows control, manipulation from the teach pendant
        if control_mode == "local":
            # self.controller = RTDEControl(
            #     self.robot_ip, RTDEControl.FLAG_USE_EXT_UR_CAP, self.robot_port)
            self.controller = RTDEControl(
                self.robot_ip, RTDEControl.FLAG_USE_EXT_UR_CAP)
        elif control_mode == "remote":
            self.controller = RTDEControl(self.robot_ip, self.robot_port)
        else:
            self.log("Chosse a valid control mode: local or remote")

        self.receiver = RTDEReceive(self.robot_ip, variables=[])
        self.io_controller = RTDEIO(self.robot_ip)
        # read config
        self.config = {}
        with open(os.path.join('{}.yaml'.format(file.split('.')[0])), 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        if self.config['has_gripper']:
            # TODO: should also specify which gripper in yaml
            self.gripper = RG2(self, self.config)
            # init by opening the gripper
            self.open_gripper()
        else:
            self.gripper = None

        # we are using the RG2 gripper currently
        # self.gripper = RG2(self)

        # self.publisher = rospy.Publisher(
        #     f'{self.robot_ns}/joint_states/', JointState, queue_size=10)

        # TODO currently we have no idle state. Therefore, when robot is standing still, watchdog is not kicked. \
        #  but we can only use it when it is kicked with min freq...
        # set watchdog for safety. watchdog is kicked in control loop
        # self.controller.setWatchdog(min_frequency=10.0)

        # get config
        #self.max_joint_velocity = rospy.get_param('max_joint_velocity', 3)
        # self.max_joint_acceleration = rospy.get_param(
        #    'max_joint_acceleration', 0.2)

        self.joint_names = self.config['actuated_joints']

        # position of robot base
        self.base_pose = base_pose

        self.ft_sensor_names = {
            'force_x': 0,
            'force_y': 0,
            'force_z': 0,
            'torque_x': 0,
            'torque_y': 0,
            'torque_z': 0,
        }

    def log_default(self, msg):
        print(msg)

    def set_log_function(self, log_function):
        self.log = log_function

    def is_running_safe(self):
        """
        Check if controller is connected and program is running
        @return:
        """

        return self.controller.isConnected() and self.controller.isProgramRunning()

    def reconnect(self):
        """
        Reconnect to robot modules, can be used after
        """
        if not self.controller.isConnected():
            self.log('Reconnecting to controller')
            self.controller.reconnect()
        if not self.receiver.isConnected():
            self.log('Reconnecting to receiver')
            self.receiver.reconnect()

        # also check if script is running
        if not self.controller.isProgramRunning():
            self.log('Is Connected, but script is not running!')

        return True

    def disconnect(self):
        """
        Should be called to safely disconnect from the robot at the end of the program
        :return:
        """
        # stop script
        self.controller.stopScript()
        # disconnect controller and receiver
        self.controller.disconnect()
        self.receiver.disconnect()

    def _control_loop(self) -> bool:
        """
        This should be called as much as possible when moving the robot
        :return: True if everything is ok, False otherwise
        """
        # if not self.controller.kickWatchdog():
        #     return False
        if not self.is_running_safe():
            return False
        return True

    def follow_joint_trajectory(self, dt, joint_trajectory):
        # based on https://sdurobotics.gitlab.io/ur_rtde/examples/examples.html#servoj-example
        # Parameters
        lookahead_time = 0.1
        gain = 300  # controller gain

        # Move to initial joint position with a regular moveJ
        self.controller.moveJ(joint_trajectory[0], True)

        # Execute 500Hz control loop for 2 seconds, each cycle is 2ms
        for q in joint_trajectory:
            start = time.time()
            self.controller.servoJ(
                q,
                self.max_joint_velocity,
                self.max_joint_velocity,
                dt,  # duration this call is blocking, can be shorter if q value is reached
                lookahead_time,
                gain)
            end = time.time()
            duration = end - start
            if duration < dt:
                time.sleep(dt - duration)

        self.controller.servoStop()

    def servo_joint_position(self, joint_position):
        """
        This will change the joint target positions for the position controller and it will
        try to move there as fast as possible.
        :param joint_position: joint position of thr robot
        :return:
        """
        self.controller.moveJ(
            joint_position, self.max_joint_velocity, self.max_joint_acceleration, False)
        while not self.controller.isSteady():
            self._control_loop()

    def moveL_offset(self, offset):
        """
        Move linear in tool space from current pose to specified offset
        """
        # get current tcp values
        current_tcp_pose = np.array(self.receiver.getActualTCPPose())

        target_tcp_pose = current_tcp_pose + np.array(offset)
        self.log(f'moveL to {target_tcp_pose}')

        self.controller.moveL(target_tcp_pose, 0.2, 0.1, True)

        while not self.controller.isSteady():
            self._control_loop()

    def move_until_contact(self, speed, contact_direction):

        self.controller.moveUntilContact(speed, contact_direction)

    def enter_force_mode(self, task_frame, compliance, ft, limits, duration):
        """
        Enter force mode with the UR5e
        @param task_frame: force frame relative to the base frame
        @param compliance:
        @param ft:
        @param limits:
        @param duration:
        @return:
        """
        t_start = time.time()
        self.controller.forceModeSetDamping(0.0025)
        self.log(f'Entering Force Mode for {duration}s')
        result = self.controller.forceMode(
            task_frame, compliance, ft, 2, limits)

        # block for that duration
        while duration > time.time() - t_start and \
                self.controller.isJointsWithinSafetyLimits(self.receiver.getActualQ()):
            pass
        self.controller.forceModeStop()

    def stop_force_mode(self):

        self.controller.forceModeStop()

    def servoJ_offset(self, offset):
        """
        Move linear in joint space from current joint pose to specified offset
        """
        q = np.array(self.receiver.getActualQ())

        target_q = np.array(q) + np.array(offset)

        return self.servo_joint_position(target_q)

    def grip(self, gripper_width, gripper_force=1, block=True, use_depth_compensation=False):
        """
        Use the gripper with options depending on gripper
        """
        self.gripper.grip(gripper_width,
                          force=gripper_force,
                          blocking=block,
                          use_depth_compensation=use_depth_compensation)

    def open_gripper(self):

        self.gripper.open()

    def close_gripper(self):

        self.gripper.close()

    def get_joint_states(self):
        joint_positions = self.receiver.getActualQ()
        joint_velocities = self.receiver.getActualQd()
        joint_states = {'arm_joint_positions': joint_positions}
        for i, joint in enumerate(self.joint_names):
            joint_states[joint] = {
                'joint_position': joint_positions[i],
                'joint_velocity': joint_velocities[i],
                'index': i
            }
        if self.gripper:
            joint_states = {**joint_states, **self.gripper.get_joint_states()}
        return joint_states

    def get_ft_sensor(self):
        """
        Get the internal FT sensor reading
        @return: dictionary of ft sensor names and values
        """

        ft_values = self.receiver.getActualTCPForce()
        return {key: value for key, value in zip(self.ft_sensor_names, ft_values)}

    def zero_ft_sensor(self):

        self.controller.zeroFtSensor()
        self.controller.setTcp([0] * 6)
        # self.controller.setPayload(0, [0, 0, 0])

    def set_payload(self, mass, center_pos):

        self.controller.setPayload(mass, center_pos)
        self.log(
            f'New payload was set. \n Mass: {mass}kg \n Center: {center_pos}')


# ---- END EFFECTORS ----- #


class BaseGripper:
    """
    Template class for any gripper, need to implement corresponding methods for binary or non-binary gripper
    """

    def __init__(self, robot, is_binary):
        self.robot = robot
        self.log = self.robot.log
        self.is_binary = is_binary

    def log_default(self, msg):
        print(msg)

    def set_log_function(self, log_function):
        self.log = log_function

    def grip(self, width, force, blocking=True, use_depth_compensation=False):
        if self.is_binary:
            self.log('Use the open() and close() methods for a binary gripper')
        return None

    def get_gripper_width(self):
        if self.is_binary:
            self.log('Use the open() and close() methods for a binary gripper')
        return None

    def open(self):
        self.log('Not implemented!')

    def close(self):
        if not self.is_binary:
            self.log('Use the grip() method for a continuous gripper')

    def is_open(self):
        if not self.is_binary:
            self.log('Use the () method for a continuous gripper')


class RG2(BaseGripper):
    """
    The RG2 gripper is controlled via the UR script.
    To be used with the ur_rtde API, it is necessary to control the UR in Local mode and the external URCap.
    Communication with the OnRobot UrCap is done over registers, which are read in the control.script running on the UR.
    """
    @unique
    class RG2InputGripCommands(IntEnum):

        # different possible states
        NO_COMMAND = 0
        NORMAL_GRIP = 1
        NO_BLOCK_GRIP = 2
        DEPTH_COMPENSATION_GRIP = 3
        NO_BLOCK_DEPTH_COMPENSATION_GRIP = 4

    @unique
    class RG2OutputCommands(IntEnum):
        # output messages from rg2
        IDLE = 0
        RUNNING = 1
        DONE = 2

    def __init__(self, robot, config):
        super().__init__(robot, is_binary=False)
        self.robot = robot
        self.io_controller = self.robot.io_controller
        self.receiver = self.robot.receiver
        self.config = config

        self.joint_names = self.config['gripper_joints']
        self.max_width = self.config['gripper_max_width']
        self.max_force = self.config['gripper_max_force']

    def grip(self, width, force, blocking=True, use_depth_compensation=False):

        # multiply with actual max gripper width and force
        target_width = self.max_width * width
        taregt_force = self.max_force * force

        self.log('Starting Grip with:')
        # self.log(f'{width=}. {force=},{blocking=},{use_depth_compensation=}')

        # set values for grip
        self.io_controller.setInputDoubleRegister(
            InputDoubleRegisters.RG2_INPUT_TARGET_WIDTH, target_width)
        self.io_controller.setInputDoubleRegister(
            InputDoubleRegisters.RG2_INPUT_TARGET_FORCE, taregt_force)

        command = RG2.RG2InputGripCommands.NO_COMMAND
        # inform rg2 to do grip
        if blocking and not use_depth_compensation:
            command = RG2.RG2InputGripCommands.NORMAL_GRIP
        elif not blocking and not use_depth_compensation:
            command = RG2.RG2InputGripCommands.NO_BLOCK_GRIP
        elif blocking and use_depth_compensation:
            command = RG2.RG2InputGripCommands.DEPTH_COMPENSATION_GRIP
        elif not blocking and use_depth_compensation:
            command = RG2.RG2InputGripCommands.NO_BLOCK_DEPTH_COMPENSATION_GRIP

        # send command
        self.io_controller.setInputIntRegister(
            InputIntRegisters.RG2_INPUT_STATE_COMMAND, command)

        # wait until received feedback. if not blocking, should be immediate, otherwise until grip is done
        while self.receiver.getOutputIntRegister(OutputIntRegisters.RG2_OUTPUT_STATE_COMMAND) != \
                RG2.RG2OutputCommands.DONE:
            pass

        # set the state register for the gripper back to no command
        self.io_controller.setInputIntRegister(InputIntRegisters.RG2_INPUT_STATE_COMMAND,
                                               RG2.RG2InputGripCommands.NO_COMMAND)

    def get_gripper_width(self):
        return self.receiver.getOutputDoubleRegister(OutputDoubleRegisters.RG2_OUTPUT_ACTUAL_WIDTH)

    def grip_detected(self):
        message = self.receiver.getOutputIntRegister(
            OutputIntRegisters.RG2_GRIP_DETECTED)
        if message == 0:
            return False
        else:
            return True

    def get_joint_states(self):

        current_width = self.get_gripper_width()

        # get joint values for open and close and coupling
        pos_closed = np.array(self.config['gripper_closed'])
        pos_open = np.array(self.config['gripper_open'])
        joint_coupling = np.array(self.config['gripper_joint_coupling'])

        current_joints = pos_closed * joint_coupling + (current_width / self.max_width *
                                                        (pos_open - pos_closed)) * joint_coupling
        joint_values = {
            joint_name: {
                'joint_position': current_joints[i],
                'index': 6 + i
            }
            for i, joint_name in enumerate(self.joint_names)
        }

        return joint_values

    def open(self):
        return self.grip(width=90, force=40)

    def close(self):
        return self.grip(width=0, force=40)


class SchunkCoactGripper(BaseGripper):
    """
    The Schunk Coact UREK Gripper is controlled with digital IOs.
    """

    def __init__(self, robot):
        super().__init__(robot, is_binary=False)
        self.robot = robot
        self.io_controller = self.robot.io_controller

    def close_gripper(self):
        """
        Opens the gripper
        :return:
        """
        self.io_controller.setStandardDigitalOut(0, False)
        self.io_controller.setStandardDigitalOut(1, True)

    def open_gripper(self):
        """
        Opens the gripper
        :return:
        """
        self.io_controller.setStandardDigitalOut(0, True)
        self.io_controller.setStandardDigitalOut(1, False)

    def gripper_light_off(self):
        """
        Turns off the light of the Gripper
        :return:
        """
        self.io_controller.setStandardDigitalOut(2, False)
        self.io_controller.setStandardDigitalOut(3, False)

    def gripper_light_green(self):
        """
        Turns on the green light of the gripper.
        :return:
        """
        self.io_controller.setStandardDigitalOut(2, False)
        self.io_controller.setStandardDigitalOut(3, True)

    def gripper_light_yellow(self):
        """
        Turns on the yellow light of the gripper.
        :return:
        """
        self.io_controller.setStandardDigitalOut(2, True)
        self.io_controller.setStandardDigitalOut(3, False)

    def gripper_light_red(self):
        """
        Turns on the red light of the gripper.
        :return:
        """
        self.io_controller.setStandardDigitalOut(2, True)
        self.io_controller.setStandardDigitalOut(3, True)
