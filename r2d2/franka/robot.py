# ROBOT SPECIFIC IMPORTS
from polymetis import RobotInterface, GripperInterface
from r2d2.robot_ik.robot_ik_solver import RobotIKSolver
import grpc

# UTILITY SPECIFIC IMPORTS
from r2d2.misc.transformations import euler_to_quat, quat_to_euler, add_poses, pose_diff, add_quats
from r2d2.misc.subprocess_utils import run_terminal_command, run_threaded_command
from r2d2.misc.parameters import sudo_password
from r2d2.camera_utils.wrappers.iris_multi_cam_wrapper import IrisMultiCameraWrapper
import numpy as np
import torch
import time
import os

class FrankaRobot:

    def launch_controller(self):
        try: self.kill_controller()
        except: pass

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._robot_process = run_terminal_command(
            'echo ' + sudo_password + ' | sudo -S ' + 'bash ' + dir_path + '/launch_robot.sh')
        # self._gripper_process = run_terminal_command(
        #     'echo ' + sudo_password + ' | sudo -S ' + 'bash ' + dir_path + '/launch_gripper.sh')
        self._server_launched = True
        time.sleep(5)

    def launch_robot(self):
        self._robot = RobotInterface(ip_address="localhost")
        # self._gripper = GripperInterface(ip_address="localhost")
        # self._max_gripper_width = self._gripper.metadata.max_width
        self._ik_solver = RobotIKSolver()
        self.camera_reader = IrisMultiCameraWrapper()
        self.ideal_position = []
        self.init_ideal = False

    def kill_controller(self):
        self._robot_process.kill()
        # self._gripper_process.kill()

    def read_cameras(self):
        camera_read = self.camera_reader.read_cameras()
        print("---CAM READ---")
        print(camera_read)
        camera_feed = pickle.dumps(camera_read)
        # camera_feed = [c.tolist() for c in camera_feed]
        return camera_feed

    def update_command(self, command, action_space='cartesian_velocity', blocking=False):
        action_dict = self.create_action_dict(command, action_space=action_space)
        
        self.update_joints(action_dict['joint_position'], velocity=False, blocking=blocking)
        # self.update_gripper(action_dict['gripper_position'], velocity=False, blocking=blocking)
        post_robot_state = self.get_robot_state()[0]
        action_dict["post_action_cartestian_pos"] = post_robot_state["cartesian_position"]
        action_dict["post_action_joint_pos"] = post_robot_state["joint_positions"]
        return action_dict

    def update_pose(self, command, velocity=False, blocking=False):
        if blocking:
            if velocity:
                curr_pose = self.get_ee_pose()
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(command)
                command = add_poses(cartesian_delta, curr_pose)

            pos = torch.Tensor(command[:3])
            quat = torch.Tensor(euler_to_quat(command[3:6]))

            if self._robot.is_running_policy():
                self._robot.terminate_current_policy()
            try: self._robot.move_to_ee_pose(pos, quat)
            except grpc.RpcError: pass
        else:
            if not velocity:
                curr_pose = self.get_ee_pose()
                cartesian_delta = pose_diff(command, curr_pose)
                command = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
            
            robot_state = self.get_robot_state()[0]
            joint_velocity = self._ik_solver.cartesian_velocity_to_joint_velocity(command, robot_state=robot_state)
            
            self.update_joints(joint_velocity, velocity=True, blocking=False)

    def update_joints(self, command, velocity=False, blocking=False, cartesian_noise=None):
        if cartesian_noise is not None: command = self.add_noise_to_joints(command, cartesian_noise)
        command = torch.Tensor(command)
        
        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self._robot.get_joint_positions()

        def helper_non_blocking():
            if not self._robot.is_running_policy():
                self._robot.start_cartesian_impedance()
            try: self._robot.update_desired_joint_positions(command)
            except grpc.RpcError: pass

        if blocking:
            if self._robot.is_running_policy():
                self._robot.terminate_current_policy()
            try: self._robot.move_to_joint_positions(command)
            except grpc.RpcError: pass
            self._robot.start_cartesian_impedance()
        else:
            run_threaded_command(helper_non_blocking)

    def update_gripper(self, command, velocity=True, blocking=False):
        raise Exception("cant update gripper, no gripper attached")
        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            command = gripper_delta + self.get_gripper_position()
        
        command = float(np.clip(command, 0, 1))
        raise Exception("Gripper not attached")
        # self._gripper.goto(width=self._max_gripper_width * (1 - command),
        #     speed=0.05, force=0.1, blocking=blocking)

    def add_noise_to_joints(self, original_joints, cartesian_noise):
        original_joints = torch.Tensor(original_joints)

        pos, quat = self._robot.robot_model.forward_kinematics(original_joints)
        curr_pose = pos.tolist() + quat_to_euler(quat).tolist()
        new_pose = add_poses(cartesian_noise, curr_pose)

        new_pos = torch.Tensor(new_pose[:3])
        new_quat = torch.Tensor(euler_to_quat(new_pose[3:]))

        noisy_joints, success = self._robot.solve_inverse_kinematics(
            new_pos, new_quat, original_joints)

        if success: desired_joints = noisy_joints
        else: desired_joints = original_joints
        
        return desired_joints.tolist()

    def get_joint_positions(self):
        return self._robot.get_joint_positions().tolist()

    def get_joint_velocities(self):
        return self._robot.get_joint_velocities().tolist()

    def get_gripper_position(self):
        raise Exception("gripper not attached, can't fetch position")
        return 1 - (self._gripper.get_state().width / self._max_gripper_width)

    def get_ee_pose(self):
        pos, quat = self._robot.get_ee_pose()
        angle = quat_to_euler(quat.numpy())
        return np.concatenate([pos, angle]).tolist()

    def get_robot_state(self):
        robot_state = self._robot.get_robot_state()
        gripper_position = 0 #None #self.get_gripper_position()
        pos, quat = self._robot.robot_model.forward_kinematics(torch.Tensor(robot_state.joint_positions))
        cartesian_position = pos.tolist() + quat_to_euler(quat.numpy()).tolist()

        state_dict = {
                'cartesian_position': cartesian_position,
                'gripper_position': gripper_position,
                'joint_positions': list(robot_state.joint_positions),
                'joint_velocities': list(robot_state.joint_velocities),
                'joint_torques_computed': list(robot_state.joint_torques_computed),
                'prev_joint_torques_computed': list(robot_state.prev_joint_torques_computed),
                'prev_joint_torques_computed_safened': list(robot_state.prev_joint_torques_computed_safened),
                'motor_torques_measured': list(robot_state.motor_torques_measured),
                'prev_controller_latency_ms': robot_state.prev_controller_latency_ms,
                'prev_command_successful': robot_state.prev_command_successful}

        timestamp_dict = {'robot_timestamp_seconds': robot_state.timestamp.seconds,
                          'robot_timestamp_nanos': robot_state.timestamp.nanos}
       
        return state_dict, timestamp_dict

    def create_action_dict(self, action, action_space, robot_state=None):
        assert action_space in ['cartesian_position', 'joint_position', 'cartesian_velocity', 'joint_velocity']
        if robot_state is None: robot_state = self.get_robot_state()[0]
        robot_state['gripper_position'] = 0
        action[-1] = 0
        action_dict = {'robot_state': robot_state}
        velocity = 'velocity' in action_space
        action_dict['gripper_position'] = 0
        action_dict['gripper_delta'] = 0

        # if velocity:
        #     action_dict['gripper_velocity'] = action[-1]
        #     gripper_delta = self._ik_solver.gripper_velocity_to_delta(action[-1])
        #     gripper_position = robot_state['gripper_position'] + gripper_delta
        #     action_dict['gripper_position'] = float(np.clip(gripper_position, 0, 1))
        # else:
        #     action_dict['gripper_position'] = float(np.clip(action[-1], 0, 1))
        #     gripper_delta = action_dict['gripper_position'] - robot_state['gripper_position']
        #     gripper_velocity = self._ik_solver.gripper_delta_to_velocity(gripper_delta)
        #     action_dict['gripper_delta'] = gripper_velocity

        if 'cartesian' in action_space:
            if velocity:
                action_dict['cartesian_velocity'] = action[:-1]
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(action[:-1])
                action_dict['cartesian_position'] = add_poses(cartesian_delta, robot_state['cartesian_position']).tolist()
 
                # joint_delta
                if self.init_ideal == False:
                    self.ideal_position = np.array(robot_state['cartesian_position'])
                    self.init_ideal = True
                self.ideal_position = add_poses(cartesian_delta, self.ideal_position)
                action_dict['ideal_cartesian_position'] = self.ideal_position.tolist()
            else:
                action_dict['cartesian_position'] = action[:-1]
                cartesian_delta = pose_diff(action[:-1], robot_state['cartesian_position'])
                cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
                action_dict['cartesian_velocity'] = cartesian_velocity.tolist()
            
            action_dict['joint_velocity'] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                action_dict['cartesian_velocity'], robot_state=robot_state).tolist()
            joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict['joint_velocity'])

            action_dict['joint_position'] = (joint_delta + np.array(robot_state['joint_positions'])).tolist()
            
            
        if 'joint' in action_space:
            # NOTE: Joint to Cartesian has undefined dynamics due to IK
            if velocity:
                action_dict['joint_velocity'] = action[:-1]
                joint_delta = self._ik_solver.joint_velocity_to_delta(action[:-1])
                action_dict['joint_position'] = (joint_delta + np.array(robot_state['joint_positions'])).tolist()
            else:
                action_dict['joint_position'] = action[:-1]
                joint_delta = np.array(action[:-1]) - np.array(robot_state['joint_positions'])
                joint_velocity = self._ik_solver.joint_delta_to_velocity(joint_delta)
                action_dict['joint_velocity'] = joint_velocity.tolist()
        
        # print(action_dict['joint_position'])
        # print("----------------")
        # print(action_dict['joint_position'])
        # print(joint_delta)
        # raise Exception("debug")
        return action_dict
