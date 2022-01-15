import numpy as np


def check_table_collision(grip_pos, relative_action):
    """
    If moved further than allowed distance, we overwrite the relative action with the difference
    between the minimally allowed z position and the current gripper position
    """
    min_allowed_z_pos = np.array([-0.265])
    curr_grip_z_pos = grip_pos[2]
    next_grip_z_pos = grip_pos[2] + relative_action[2]

    if next_grip_z_pos < min_allowed_z_pos:
        relative_action[2] = min_allowed_z_pos - curr_grip_z_pos

    return relative_action

def check_table_collision_rg2(grip_pos, relative_action):
    """
    If moved further than allowed distance, we overwrite the relative action with the difference
    between the minimally allowed z position and the current gripper position
    """
    min_allowed_z_pos = np.array([-0.0464])
    curr_grip_z_pos = grip_pos[2]
    next_grip_z_pos = grip_pos[2] + relative_action[2]

    if next_grip_z_pos < min_allowed_z_pos:
        relative_action[2] = min_allowed_z_pos - curr_grip_z_pos

    return relative_action
