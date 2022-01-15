def map_2_rg2_action(gripper_action, ur5e_robot):
    curr_width = ur5e_robot.gripper.get_gripper_width()
    max_width = ur5e_robot.gripper.max_width # 110 mm

    gripper_action = -gripper_action # we want positive to close, subtract width
    max_increment = max_width # mm
    increment = gripper_action*max_increment
    rg2_action = curr_width + increment
    rg2_action = rg2_action / max_width
    return rg2_action #new width in [0,1]*max_width

def map_joints_rg2_2_sim(rg2_val):
    rg2_closed = -0.41
    rg2_open = 0.45
    sim_closed = 0.2
    sim_open = -0.32

    slope = (sim_closed - sim_open) / (rg2_closed - rg2_open)
    sim_val = sim_open + slope * (rg2_val - rg2_open)
    return sim_val