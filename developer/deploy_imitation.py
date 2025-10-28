import os
import numpy as np
import torch
import random
import onnxruntime as ort

# Fourier runtime imports
from ischedule import run_loop, schedule
import fourier_grx.sdk.developer as fourier_grx

# Helper functions
import deploy_utils as dpu

# Constants
POLICY_PATH = "policy_imitation.onnx"
G_DOWN = torch.Tensor([0.0, 0.0, -1.0]).unsqueeze(0).float()
CONTROL_FREQ = 50                       # Hz
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQ   # seconds

class Controller:
    def __init__(self, policy_file_path: str) -> None:
        self.control_system = fourier_grx.ControlSystem()

        self.policy_file_path = None
        self.policy_ort = None
        self.policy_action = None
        self.obs_buf_stack = None
        
        self.load_policy_model(policy_file_path)

        # - Robot & state variables
        self.state_dict = None
        self.num_dof = 6 + 6 + 1 + 5 + 5
        self.joint_idx = np.array([
            0, 1, 2, 3, 4, 5,       # left leg
            6, 7, 8, 9, 10, 11,     # right leg
            12,                     # waist
            13, 14, 15, 16, 17,     # left arm
            18, 19, 20, 21, 22,     # right arm
        ])

        self.def_dof_pos = np.array([
            -0.2468, 0.0, 0.0, 0.5181, 0.0, -0.2408,    # left leg
            -0.2468, 0.0, 0.0, 0.5181, 0.0, -0.2408,    # right leg
            0.0,                                        # waist
            0.0, 0.0, 0.0, 0.0, 0.0,                    # left arm
            0.0, 0.0, 0.0, 0.0, 0.0,                    # right arm
        ])

        self.action_clip_max = np.array([
            2.618, 1.571, 1.571, 2.356, 0.436, 0.785,   # left leg
            2.618, 0.262, 1.571, 2.356, 0.436, 0.785,   # right leg
            2.618,                                      # waist
            2.966, 0.174, 1.834, 0.349, 1.832,          # left arm
            2.966, 0.174, 1.834, 0.349, 1.832,          # right arm
        ]) + np.array([
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,               # left leg
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,               # right leg
            0.5,                                        # waist
            0.5, 0.5, 0.5, 0.5, 0.5,                    # left arm
            0.5, 0.5, 0.5, 0.5, 0.5,                    # right arm
        ])

        self.action_clip_min = np.array([
            -2.618, -0.262, -1.571, -0.087, -0.436, -0.785,     # left leg
            -2.618, -1.571, -1.571, -0.087, -0.436, -0.785,     # right leg
            -2.618,                                             # waist
            -2.966, -1.396, -1.834, -1.396, -1.832,             # left arm
            -2.966, -1.396, -1.834, -1.396, -1.832,             # right arm
        ]) - np.array([
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,                       # left leg
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,                       # right leg
            0.5,                                                # waist
            0.5, 0.5, 0.5, 0.5, 0.5,                            # left arm
            0.5, 0.5, 0.5, 0.5, 0.5,                            # right arm
        ])

        # - Control parameters
        self.dof_control_mode = np.array([
            # left leg
            fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD,
            fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD,
            # right leg
            fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD,
            fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD,
            # waist
            fourier_grx.JointControlMode.PD,
            # left arm
            fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD,
            fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD,
            # right arm
            fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD,
            fourier_grx.JointControlMode.PD, fourier_grx.JointControlMode.PD,
        ])
        # - default PD
        self.dof_target_kp = np.array([
            180.0, 120.0, 90.0, 120.0, 45.0, 45.0,  # left leg
            180.0, 120.0, 90.0, 120.0, 45.0, 45.0,  # right leg
            90.0,                                   # waist
            90.0, 45.0, 45.0, 45.0, 45.0,           # left arm
            90.0, 45.0, 45.0, 45.0, 45.0,           # right arm
        ])
        self.dof_target_kd = np.array([
            10.0, 10.0, 8.0, 8.0, 2.5, 2.5,         # left leg
            10.0, 10.0, 8.0, 8.0, 2.5, 2.5,         # right leg
            8.0,                                    # waist
            8.0, 2.5, 2.5, 2.5, 2.5,                # left arm
            8.0, 2.5, 2.5, 2.5, 2.5,                # right arm
        ])
        # - test PD
        self.dof_target_kp = np.array([
            120.0, 80.0, 60.0, 80.0, 35.0, 45.0,    # left leg
            120.0, 80.0, 60.0, 80.0, 35.0, 45.0,    # right leg
            60.0,                                   # waist
            45.0, 22.5, 22.5, 22.5, 22.5,           # left arm
            45.0, 22.5, 22.5, 22.5, 22.5,           # right arm
        ])
        self.dof_target_kd = np.array([
            10.0, 8.0, 4.0, 4.0, 2.5, 2.5,          # left leg
            10.0, 8.0, 4.0, 4.0, 2.5, 2.5,          # right leg
            4.0,                                    # waist
            10.0, 5.0, 5.0, 5.0, 5.0,               # left arm
            10.0, 5.0, 5.0, 5.0, 5.0,               # right arm
        ])
        self.dof_target_positions = np.zeros(self.num_dof, dtype=np.float32)

        # - Observation data
        self.hist_horizon = 5
        self.num_prop_obs = 75
        self.num_ref_obs_per_frame = 39 + 3 + 3 + 3
        self.num_ref_obs = self.hist_horizon * self.num_ref_obs_per_frame
        self.num_obs = self.num_prop_obs + self.num_ref_obs

        self.policy_action = np.zeros(self.num_dof, dtype=np.float32)
        
        # - Reference motion data
        self.ref_motions = dpu.import_reference_motions()

        self.ref_obs_buffer = np.zeros([1, self.num_ref_obs], dtype=np.float32)
        self.m_key = None # random.choice(list(self.ref_motions.keys()))
        
        # TEMPORARY: prompt user to select a motion to play, play it, then exit
        while True:
            try:
                cmd = input("Enter motion key (or 'list', 'list | grep {substr}'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\033[91m\nInterrupted. Please enter a valid key to continue.\033[0m")
                continue

            if cmd == "list":
                print("Available keys:")
                for k in sorted(self.ref_motions.keys()):
                    print(k)
                continue

            if cmd.startswith("list | grep "):
                substr = cmd[len("list | grep "):].strip()
                matches = [k for k in sorted(self.ref_motions.keys()) if substr in k]
                if matches:
                    print(f"Keys containing '{substr}':")
                    for k in matches:
                        print(k)
                else:
                    print(f"No keys contain '{substr}'.")
                continue

            if cmd in self.ref_motions:
                self.m_key = cmd
                print(f"\033[96mSelected motion: {self.m_key}\033[0m")
                break

            print("\033[31mInvalid input. Type 'list' to see all keys, or 'list | grep {substr}' to filter.\033[0m")

        self.m_val = self.ref_motions[self.m_key]
        self.m_index = 0


    def load_policy_model(self, policy_file_path: str, map_location='cpu'):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.policy_file_path = os.path.join(cur_path, policy_file_path)
        self.policy_ort = ort.InferenceSession(self.policy_file_path)
        print(f"Policy model loaded from {self.policy_file_path}")


    def run(self):
        state_dict = self.control_system.robot_control_loop_get_state()

        imu_measured_quat = state_dict.get("imu_quat", [0, 0, 0, 1])
        imu_measured_angular_velocity = state_dict.get("imu_angular_velocity", [0, 0, 0])
        joint_measured_position = state_dict.get("joint_position", [0] * self.num_dof)
        joint_measured_velocity = state_dict.get("joint_velocity", [0] * self.num_dof)

        imu_measured_quat = np.deg2rad(imu_measured_quat)
        imu_measured_angular_velocity = np.deg2rad(imu_measured_angular_velocity)
        joint_measured_position_deg = np.zeros(self.num_dof)
        joint_measured_velocity = np.zeros(self.num_dof)
        for i in range(self.num_dof):
            joint_measured_position_deg[i] = np.deg2rad(joint_measured_position[i])
            joint_measured_velocity[i] = np.deg2rad(joint_measured_velocity[i])

        # - Build proprioceptive observations
        proj_grav = dpu.quat_rotate_inverse(imu_measured_quat[None], G_DOWN)[0]
        ang_vel = imu_measured_angular_velocity
        q = joint_measured_position - self.def_dof_pos
        q = dpu.joint_mj_to_pol(q)
        dq = joint_measured_velocity
        dq = dpu.joint_mj_to_pol(dq)

        # - Build reference observations
        if self.m_index >= self.m_val.shape[0]:
            # crash out when motion ends
            print("\033[93mReference motion ended. Exiting control loop.\033[0m")
            os._exit(0)

        self.ref_obs_buffer = np.roll(self.ref_obs_buffer, -self.num_ref_obs_per_frame, axis=1)
        step_m_val = self.m_val[self.m_index, ...].reshape(-1)
        self.ref_obs_buffer[0, -self.num_ref_obs_per_frame:] = step_m_val
        self.m_index += 1

        # - Combine observations
        prop_obs_list = []
        prop_obs_list.append(ang_vel[None])
        prop_obs_list.append(proj_grav)
        prop_obs_list.append(q)
        prop_obs_list.append(dq)
        prop_obs_list.append(self.policy_action[None])

        prop_obs = np.concatenate(prop_obs_list, axis=1)
        input = np.concatenate([self.ref_obs_buffer.reshape(-1)[None], prop_obs], axis=1)
        input = input.astype(np.float32)

        # - Get policy action
        output = self.policy_ort.run(None, {'obs': input})

        # store for next observation
        self.policy_action = output[0].squeeze(0)

        output = dpu.joint_pol_to_mj(output[0]) # swap from IsaacLab to expected robot joint order

        action = output
        action = np.clip(
            action,
            a_min=self.action_clip_min[np.newaxis, :],
            a_max=self.action_clip_max[np.newaxis, :],
        )

        self.dof_target_positions = (action + self.def_dof_pos).squeeze(0)

        self.dof_target_positions = np.rad2deg(self.dof_target_positions)

        # Set control
        """
        Robot Control:
        - control_mode
        - pd_control_kp
        - pd_control_kd
        - position [rad]
        """
        control_dict = {
            "control_mode": self.dof_control_mode,
            "pd_control_kp": self.dof_target_kp,
            "pd_control_kd": self.dof_target_kd,
            "position": self.dof_target_positions,
        }

        print("\033[92mInputting to robot\033[0m")

        # output control
        self.control_system.robot_control_loop_set_control(control_dict=control_dict)

        print("\033[92mControl input sent\033[0m")

if __name__ == "__main__":
    policy_file_path = POLICY_PATH
    controller = Controller(policy_file_path)

    controller.control_system.developer_mode(servo_on=True, control_frequency=100)
    
    schedule(controller.run, interval=CONTROL_PERIOD_S)
    
    run_loop()
