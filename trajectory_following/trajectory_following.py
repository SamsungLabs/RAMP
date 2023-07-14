"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Vasileios Vasilopoulos (vasileios.v@samsung.com; vasilis.vasilop@gmail.com)
Jinwook Huh (jinwook.huh@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""

# General imports
import time
import os
import numpy as np
import logging
from typing import List
import open3d as o3d

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as Functional

# Import C-SDF
from csdf.pointcloud_sdf import PointCloud_CSDF

# Import differentiable-robot-model functionality
from differentiable_robot_model.robot_model import DifferentiableRobotModel


log = logging.getLogger('TRAJECTORY FOLLOWING')


LINK_FIXED = 'panda_link0'

LINK_EE = 'panda_hand'

LINK_SKELETON = [
    'panda_link1',
    'panda_link3',
    'panda_link4',
    'panda_link5',
    'panda_link7',
    'panda_hand',
]

JOINT_LIMITS = [
    np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
]


robot_urdf_location = os.path.join('resources',"panda/panda.urdf",)

class TrajectoryFollower(nn.Module):
    def __init__(
        self,
        joint_limits: List[np.ndarray],
        trajectory: np.ndarray = None,
        robot_urdf_location: str = './robot.urdf',
        link_fixed: str = 'fixed_link',
        link_ee: str = 'ee_link',
        link_skeleton: List[str] = ['fixed_link','ee_link'],
        control_points_number: int = 70,
        device: str = 'cuda',
    ):
        super(TrajectoryFollower, self).__init__()
        
        # Define robot's number of DOF
        self._njoints = joint_limits[0].shape[0]

        # Define number of control points
        self._control_points_number = control_points_number

        # Store device
        self._device = device

        # Define lower and upper joint limits
        self._joint_limits_lower = torch.from_numpy(joint_limits[0]).double().to(self._device)
        self._joint_limits_upper = torch.from_numpy(joint_limits[1]).double().to(self._device)
        
        # Register fixed link, end effector link and link skeleton
        self._link_fixed = link_fixed
        self._link_ee = link_ee
        self._link_skeleton = link_skeleton
        
        # Define a null gripper state
        self._gripper_state = torch.Tensor([0.0, 0.0]).to(self._device)
        
        # Store trajectory
        if trajectory is not None:
            self._trajectory = torch.from_numpy(trajectory).double().to(self._device)
        
        # Set up differentiable robot model
        self.differentiable_model = DifferentiableRobotModel(robot_urdf_location, "robot_following", device=self._device)
        self.differentiable_model.eval()
        self.differentiable_model.to(self._device)
        
        # Set up C-SDF - Initialize with a random point cloud
        self.csdf = PointCloud_CSDF(np.random.rand(100000,3), device=self._device)
        self.csdf.eval()
        self.csdf.to(self._device)

        # Initialize consider obstacle collisions flag
        self._consider_obstacle_collisions = True

        # Initialize object control points
        self._grasped_object_nominal_control_points = None
        self._grasped_object_grasp_T_object = None
        
    def compute_ee_pose(self, state: torch.Tensor) -> torch.Tensor:
        
        """
        Receives a robot configuration and computes the end effector pose as a tensor.
        
        :param state: Current joint configuration (BATCH_SIZE x N_STATE)
        :returns: End effector pose (BATCH_SIZE x 4 x 4)
        """
        
        batch_size = state.shape[0]
        
        # Find link locations after stacking robot configuration with gripper state
        augmented_robot_state = torch.cat((state, torch.tile(self._gripper_state, (batch_size, 1))), dim=1)
        link_transformations = self.differentiable_model.compute_forward_kinematics_all_links(augmented_robot_state)

        # Find end effector pose
        dummy_row = torch.zeros((batch_size, 1, 4)).to(self._device)
        dummy_row[:, :, 3] = 1.0
        ee_pose = torch.cat((link_transformations[self._link_skeleton[-1]][1], link_transformations[self._link_skeleton[-1]][0].unsqueeze(2)), dim = 2)
        ee_pose = torch.cat((ee_pose, dummy_row), dim = 1)
        
        return ee_pose
    
    def _get_skeleton_control_points(self, state: torch.Tensor) -> torch.Tensor:
        
        """
        Receives a robot configuration and returns a list of all skeleton control points on the manipulator.
        
        :param state: Current joint configuration (BATCH_SIZE x N_STATE)
        :returns: List of control points on the robot manipulator (BATCH_SIZE x CONTROL_POINTS x 3)
        """
        
        batch_size = state.shape[0]
        
        # Find link locations after stacking robot configuration with gripper state
        augmented_robot_state = torch.cat((state, torch.tile(self._gripper_state, (batch_size, 1))), dim=1)
        link_transformations = self.differentiable_model.compute_forward_kinematics_all_links(augmented_robot_state)
        
        # Initialize skeleton for control points - tensor should be BATCH_SIZE x 1 x 3
        skeleton_control_point_locations = torch.zeros((batch_size, len(self._link_skeleton), 3)).to(self._device)
        
        # Find skeleton control points
        for link_idx in range(len(self._link_skeleton)):
            skeleton_control_point_locations[:, link_idx, :] = link_transformations[self._link_skeleton[link_idx]][0]

        # Find end effector pose
        dummy_row = torch.zeros((batch_size, 1, 4)).to(self._device)
        dummy_row[:, :, 3] = 1.0
        ee_pose = torch.cat((link_transformations[self._link_skeleton[-1]][1], link_transformations[self._link_skeleton[-1]][0].unsqueeze(2)), dim = 2)
        ee_pose = torch.cat((ee_pose, dummy_row), dim = 1)

        # Compute grasped object control points
        if self._grasped_object_grasp_T_object is not None:
            object_pose = ee_pose[:, ] @ self._grasped_object_grasp_T_object
            object_control_points = object_pose @ torch.hstack((
                self._grasped_object_nominal_control_points,
                torch.ones((self._grasped_object_nominal_control_points.shape[0],1)).to(device = self._device)
            )).transpose(0,1)
            object_control_points = object_control_points.transpose(1,2)[:, :, :3]
            skeleton_control_point_locations = torch.cat((skeleton_control_point_locations, object_control_points), dim=1)
        
        return skeleton_control_point_locations
    
    def _get_control_points(self, state: torch.Tensor) -> torch.Tensor:
        
        """
        Receives a robot configuration and returns a list of all control points on the manipulator.
        
        :param state: Current joint configuration (BATCH_SIZE x N_STATE)
        :returns: List of control points on the robot manipulator (BATCH_SIZE x CONTROL_POINTS x 3)
        """
        
        # Find skeleton control points
        skeleton_control_point_locations = self._get_skeleton_control_points(state)
        
        # Augment control points based on the skeleton
        control_point_locations = Functional.interpolate(skeleton_control_point_locations.transpose(1,2), size=self._control_points_number, mode='linear', align_corners=True).transpose(1,2)
        
        return skeleton_control_point_locations, control_point_locations
    
    def update_trajectory(
        self,
        trajectory: np.ndarray,
        consider_obstacle_collisions: bool = True,
    ):

        """
        Update the trajectory to follow.
        
        :param trajectory: Trajectory (N x 3)
        :param consider_obstacle_collisions: Flag to consider or ignore obstacle collisions
        """

        # Update trajectory
        trajectory = torch.from_numpy(trajectory).double().unsqueeze(0).to(self._device)

        # Interpolate trajectory
        interpolated_trajectory = Functional.interpolate(trajectory.transpose(1,2), size=500, mode='linear', align_corners=True).transpose(1,2)
        self._trajectory = interpolated_trajectory[0]

        # Compute the skeleton control point locations for all configurations in the trajectory
        self._trajectory_skeleton_control_points = self._get_skeleton_control_points(self._trajectory)

        # Compute distances of skeleton control points to scene point cloud
        self._trajectory_skeleton_control_points_distances = self.csdf.compute_distances(self._trajectory_skeleton_control_points)

        # Consider obstacle collisions or not
        self._consider_obstacle_collisions = consider_obstacle_collisions
    
    def attractive_potential(
        self,
        state: torch.Tensor,
        skeleton_control_points: torch.Tensor,
        sdf_value: torch.Tensor,
    ) -> torch.Tensor:
        
        """
        Compute the attractive potential.
        
        :param state: Joint configurations (BATCH_SIZE x N_STATE)
        :param skeleton_control_points: Skeleton control points (BATCH_SIZE x CONTROL_POINTS x 3)
        :param sdf_value: SDF value for each configuration (BATCH_SIZE)
        :returns: Attractive potential to the trajectory (BATCH_SIZE)
        """

        # Find the trajectory indices that lie within the given SDF value
        distance_diff = self._trajectory_skeleton_control_points - skeleton_control_points
        distance_diff_norm = torch.linalg.norm(distance_diff, dim=2)
        if self._consider_obstacle_collisions:
            valid_waypoints = torch.all(distance_diff_norm <= sdf_value, dim=1)
        else:
            valid_waypoints = torch.all(distance_diff_norm <= 0.1, dim=1)

        # Pick as goal the furthest valid waypoint
        if len(self._trajectory[valid_waypoints]) > 0:
            goal = self._trajectory[valid_waypoints][-1]
        else:
            goal = state
        
        # Attractive potential is just the distance to this goal
        dist = torch.linalg.norm(state - goal, dim=-1)
        
        return dist**2
    
    def implicit_obstacles(
        self,
        control_points: torch.Tensor,
        collision_threshold: float = 0.08,
    ) -> torch.Tensor:
        
        """
        Compute the repulsive potential.
        
        :param control_points: Control points (BATCH_SIZE x CONTROL_POINTS x 3)
        :param collision_threshold: Threshold below which a configuration is considered to be in collision (0.08 by default)
        :returns: SDF values for each configuration (BATCH_SIZE)
        """
        
        # Evaluate C-SDF based on these points
        sdf_values = self.csdf.forward(control_points) - collision_threshold

        return sdf_values
    
    def update_obstacle_pcd(
        self,
        pcd: np.ndarray,
    ):

        """
        Update the total point cloud used for obstacle avoidance.
        
        :param pcd: Point cloud (N x 3)
        """

        # Update point cloud in SDF
        self.csdf.update_pcd(pcd)

    def attach_to_gripper(
        self,
        object_geometry,
        world_T_grasp: np.ndarray,
        object_name: str,
        object_type: str,
        world_T_object: np.ndarray = None,
    ) -> bool:

        """
        Attach object to gripper and consider the whole arm+object pair for planning.
        
        :param object_geometry: Object geometry (path to desired file for "mesh", or num_points x 6 numpy.ndarray for "pcd")
        :param world_T_grasp: Grasp pose of the gripper
        :param object_name: Name of the object to be updated
        :param object_type: Type of the object to be updated ("mesh" or "pcd")
        :param world_T_object: Pose of the object in world frame (not needed for pointclouds)
        :returns: True if object is successfully attached, False otherwise
        """

        # Add mesh object to collision checker
        if object_type == "mesh":
            # Read the mesh
            object_mesh = o3d.io.read_triangle_mesh(object_geometry)

            # Sample points on the mesh's surface
            object_nominal_control_points = np.asarray(object_mesh.sample_points_uniformly(number_of_points=20).points, dtype= np.float32)
            self._grasped_object_nominal_control_points = torch.from_numpy(object_nominal_control_points).to(self._device)

            # Construct tensor describing grasp_T_object
            grasped_object_grasp_T_object = np.linalg.inv(world_T_grasp) @ world_T_object
            self._grasped_object_grasp_T_object = torch.from_numpy(grasped_object_grasp_T_object).to(self._device)
        
        self._trajectory_skeleton_control_points = self._get_skeleton_control_points(self._trajectory)
        self._trajectory_skeleton_control_points_distances = self.csdf.compute_distances(self._trajectory_skeleton_control_points)
        
        return True
    
    def detach_from_gripper(
        self,
        object_name: str,
        to: np.ndarray = None
    ) -> bool:

        """
        Detach object from gripper and no longer consider the whole arm+object pair for planning.
        
        :param object_name: Name of the object to be detached from the gripper
        :param to: Detach object to a desired pose in the world frame
        :returns: True if object is successfully detached, False otherwise
        """

        # Detach mesh object from collision checker
        self._grasped_object_nominal_control_points = None
        self._grasped_object_grasp_T_object = None

        self._trajectory_skeleton_control_points = self._get_skeleton_control_points(self._trajectory)
        self._trajectory_skeleton_control_points_distances = self.csdf.compute_distances(self._trajectory_skeleton_control_points)

        return True
    
    def forward(
        self, 
        current_ja: torch.Tensor,
    ) -> torch.Tensor:
        
        """
        Given the current joint configuration, compute the value of the potential field that tracks the trajectory while avoiding obstacles.
        
        :param current_ja: The start joint configuration
        :returns: The value of the potential field at this particular configuration
        """

        # Compute control point locations through FK for given state
        since = time.time()
        skeleton_control_points, control_points = self._get_control_points(current_ja)
        # log.info(f"Control points computed in: {time.time()-since}")

        # Compute SDF value
        since = time.time()
        sdf_value = self.implicit_obstacles(control_points)
        # log.info(f"SDF computed in: {time.time()-since}")

        since = time.time()
        attractive_potential = self.attractive_potential(current_ja, skeleton_control_points, sdf_value)
        # log.info(f"Attractive potential computed in: {time.time()-since}")
        
        # Define potential field value
        if self._consider_obstacle_collisions:
            # Main potential field definition
            potential = torch.div(1.0 + 10.0 * attractive_potential, 1.0 + 2.0 * sdf_value)
        else:
            # Here the robot does not care about obstacles
            potential = attractive_potential

        return potential


    def follow_trajectory(self, current_joint_angles):

        """
        Main method for trajectory following.

        :param current_joint_angles: Current robot configuration
        :returns: Computed joint velocity commands
        """

        # Define maximum joint speed and control gain
        MAX_SPEED = 0.3
        CONTROL_GAIN = 0.3

        # Extract current joint states
        current_joint_angles_tensor = torch.tensor(current_joint_angles).unsqueeze(0).to('cuda')
        current_joint_angles_tensor.requires_grad = True
        
        # Find the value of the potential field
        potential_field = self.forward(current_joint_angles_tensor)
        
        # Compute the gradient
        potential_field.backward()
        potential_field_grad = current_joint_angles_tensor.grad
        
        # Find current velocity command
        current_velocity_command = -CONTROL_GAIN * potential_field_grad
        if torch.linalg.norm(current_velocity_command) > MAX_SPEED:
            current_velocity_command = MAX_SPEED * current_velocity_command / torch.linalg.norm(current_velocity_command)
        current_velocity_command = current_velocity_command.cpu().numpy()

        # Define velocity command
        velocity_command = current_velocity_command

        return velocity_command


def planner_test():
    # Candidate poses
    pose1 = np.array([-0.47332507,  1.13872886,  1.30867887, -2.30050802,  2.07975602,  2.64635682, -2.65230727])
    pose2 = np.array([ 2.09151435, -0.54573566, -0.99544001, -2.25478268,  2.02075601,  2.74072695, -1.75231826])
    pose3 = np.array([ 2.17793441, -0.48076588, -0.856754,   -1.67240107,  0.553756,    2.79897308, -0.10493574])
    pose4 = np.array([ 0.45744711,  0.70788223,  0.71865666, -0.27235043,  0.553756,    2.09835196, -0.01765767])
    pose5 = np.array([ 1.52491331, -0.45537129, -0.08102775, -1.83516145,  0.553756,    2.91463614,  0.20733792])

    # Test planning time
    start_time = time.time()
    planner = TrajectoryFollower()
    print("planning time : ", time.time()-start_time)


def main():
    planner_test()

if __name__ == '__main__':
    main()
