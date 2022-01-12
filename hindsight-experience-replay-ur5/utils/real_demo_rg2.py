def map_2_rg2_action(gripper_action, ur5e_robot):
    curr_width = ur5e_robot.gripper.get_gripper_width()
    max_width = ur5e_robot.gripper.max_width # 110 mm

    gripper_action = -gripper_action # we want positive to close, subtract width
    max_increment = 20 # mm
    increment = gripper_action*max_increment
    rg2_action = curr_width + increment
    rg2_action = rg2_action / max_width
    return rg2_action #new width in [0,1]*max_width
