import torch
import numpy as np
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
    # Expect numpy arrays. If 1D, make it (1, D)
    joint_data = np.asarray(joint_data)
    if joint_data.ndim == 1:
        joint_data = joint_data[np.newaxis, :]

    idx = [
        0, 6, 12, 1, 7, 2,
        8, 13, 18, 3, 9, 14,
        19, 4, 10, 15, 20, 5,
        11, 16, 21, 17, 22
    ]
    swapped_data = joint_data[..., idx].astype(np.float32)
    return swapped_data

def joint_pol_to_mj(joint_data):
    joint_data = np.asarray(joint_data)
    if joint_data.ndim == 1:
        joint_data = joint_data[np.newaxis, :]

    idx = [
        0, 3, 5, 9, 13, 17,
        1, 4, 6, 10, 14, 18,
        2, 7, 11, 15, 19, 21,
        8, 12, 16, 20, 22
    ]
    swapped_data = joint_data[..., idx].astype(np.float32)
    return swapped_data

def quat_rotate_inverse(q, v):
    """
    Rotate xyzw quaternion q by vector v inversely using numpy.
    q: array shape (N,4) or (4,) with order [x,y,z,w]
    v: array shape (N,3) or (3,)
    returns: rotated vectors shape (N,3)
    """
    q = np.asarray(q)
    v = np.asarray(v)
    if q.ndim == 1:
        q = q[np.newaxis, :]
    if v.ndim == 1:
        v = v[np.newaxis, :]

    N = q.shape[0]
    q_w = q[:, -1]                 # (N,)
    q_vec = q[:, :3]               # (N,3)

    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]                       # (N,3)
    b = np.cross(q_vec, v) * (2.0 * q_w)[:, np.newaxis]               # (N,3)
    dot = np.matmul(q_vec.reshape(N, 1, 3), v.reshape(N, 3, 1)).squeeze(-1)  # (N,)
    c = q_vec * (2.0 * dot)[:, np.newaxis]                             # (N,3)

    return a - b + c
