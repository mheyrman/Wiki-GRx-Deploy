import torch
import numpy
import os
import sys

"""
Utility functions for deploying Radian models.
"""

def import_reference_motions():
    motion_dir = os.path.join(os.path.dirname(__file__), '..', 'n1_test')

    motions = {}

    for file in os.listdir(motion_dir):
        if file.endswith('.pt'):
            data = torch.load(os.path.join(motion_dir, file))
            root_pos = data[..., 0:3]
            root_quat = data[..., 3:7]
            root_vel = data[..., 7:10]
            root_ang_vel = data[..., 10:13]
            root_proj_grav = data[..., 13:16]
            dof_joint_angles = data[..., 16:55]

            motion_data = torch.cat([
                dof_joint_angles,
                root_proj_grav,
                root_vel,
                root_ang_vel,
            ], dim=1).float()

            motions[file] = motion_data.numpy()
    

    return motions

def joint_mj_to_pol(joint_data):
    if len(joint_data.shape) == 1:
        joint_data = joint_data.unsqueeze(0)

    # Swap the joint data according to the specified order
    swapped_data = torch.cat((
        joint_data[..., 0],     # LHP: 0 -> 0
        joint_data[..., 6],     # RHP: 6 -> 1
        joint_data[..., 12],    # WY: 12 -> 2
        joint_data[..., 1],     # LHR: 1 -> 3
        joint_data[..., 7],     # RHR: 7 -> 4
        joint_data[..., 2],     # LHY: 2 -> 5
        joint_data[..., 8],     # RHY: 8 -> 6
        joint_data[..., 13],    # LSP: 13 -> 7
        joint_data[..., 18],    # RSP: 18 -> 8
        joint_data[..., 3],     # LKP: 3 -> 9
        joint_data[..., 9],     # RKP: 9 -> 10
        joint_data[..., 14],    # LSR: 14 -> 11
        joint_data[..., 19],    # RSR: 19 -> 12
        joint_data[..., 4],     # LAR: 4 -> 13
        joint_data[..., 10],    # RAR: 10 -> 14
        joint_data[..., 15],    # LSY: 15 -> 15
        joint_data[..., 20],    # RSY: 20 -> 16
        joint_data[..., 5],     # LAP: 5 -> 17
        joint_data[..., 11],    # RAP: 11 -> 18
        joint_data[..., 16],    # LEP: 16 -> 19
        joint_data[..., 21],    # REP: 21 -> 20
        joint_data[..., 17],    # LWY: 17 -> 21
        joint_data[..., 22],    # RWY: 22 -> 22
    ), dim=0).unsqueeze(0)

    return swapped_data

def joint_pol_to_mj(joint_data):
    if len(joint_data.shape) == 1:
        joint_data = joint_data.unsqueeze(0)

    swapped_data = torch.cat((
    joint_data[..., 0],   # LHP
    joint_data[..., 3],   # LHR
    joint_data[..., 5],   # LHY
    joint_data[..., 9],   # LKP
    joint_data[..., 13],  # LAR
    joint_data[..., 17],  # LAP
    joint_data[..., 1],   # RHP
    joint_data[..., 4],   # RHR
    joint_data[..., 6],   # RHY
    joint_data[..., 10],  # RKP
    joint_data[..., 14],  # RAR
    joint_data[..., 18],  # RAP
    joint_data[..., 2],   # WY
    joint_data[..., 7],   # LSP
    joint_data[..., 11],  # LSR
    joint_data[..., 15],  # LSY
    joint_data[..., 19],  # LEP
    joint_data[..., 21],  # LWY
    joint_data[..., 8],   # RSP
    joint_data[..., 12],  # RSR
    joint_data[..., 16],  # RSY
    joint_data[..., 20],  # REP
    joint_data[..., 22],  # RWY
    ), dim=0).unsqueeze(0)

    return swapped_data

# Function to rotate quaternion inversely
def quat_rotate_inverse(q, v):
    """
    Rotate xyzw quaternion by vector v inversely
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c