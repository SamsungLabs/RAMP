"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Vasileios Vasilopoulos (vasileios.v@samsung.com; vasilis.vasilop@gmail.com)
Suveer Garg (suveer.garg@samsung.com)

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
import numpy as np
import logging
from typing import List, Optional
import open3d as o3d

# Torch imports
import torch

# Import MPPI and cost modules
import mppi_planning.mppi_planning as mppi
from mppi_planning.cost import PathLengthCost, CollisionCost, ConvergenceCost, ManipulabilityCost

# Import C-SDF
from csdf.pointcloud_sdf import PointCloud_CSDF

# Import pytorch-kinematics functionality
import pytorch_kinematics as pk

log = logging.getLogger('MPPI TRAJECTORY PLANNING')


class TrajectoryPlanner:
    def __init__(
        self,
        joint_limits: List[np.ndarray],
        robot_urdf_location: str = 'resources/panda/panda.urdf',
        scene_urdf_location: str = 'resources/environment/environment.urdf',
        control_points_location: str = 'resources/panda_control_points/control_points.json',
        link_fixed: str = 'fixed_link',
        link_ee: str = 'ee_link',
        link_skeleton: List[str] = ['fixed_link','ee_link'],
        control_points_number: int = 30,
        device: str = 'cuda',
    ): 
        # Define robot's number of DOF
        self._njoints = joint_limits[0].shape[0]

        # Control points location
        self._control_points_location = control_points_location

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
        
        # Default gripper state - set to [0.0, 0.0]
        self._gripper_state = torch.Tensor([0.0, 0.0]).to(self._device)

        # Set up differentiable FK
        self.differentiable_model = pk.build_serial_chain_from_urdf(open(robot_urdf_location).read(), self._link_ee)
        self.differentiable_model = self.differentiable_model.to(dtype = torch.double, device = self._device)
        
        # Set up C-SDF - Initialize with a random point cloud
        self.csdf = PointCloud_CSDF(np.random.rand(100000,3), device=self._device)
        self.csdf.eval()
        self.csdf.to(self._device)

        # Indicate whether to consider collisions or not
        self._consider_obstacle_collisions = True

        # Initialize object control points
        self._grasped_object_nominal_control_points = None
        self._grasped_object_grasp_T_object = None
    
    def _mppi_dynamics(
        self,
        state: torch.Tensor,
        perturbed_action: torch.Tensor,
    ) -> torch.Tensor:

        """
        MPPI dynamics - Here the actions are waypoint displacements.
        
        :param state: Current joint configuration (BATCH_SIZE x N_STATE)
        :param perturbed_action: Current waypoint displacement (BATCH_SIZE x N_CONTROL)
        :returns: New state after applying the desired action (BATCH_SIZE x N_STATE)
        """

        # Define the action (waypoint displacement)
        u = perturbed_action

        # Add the displacement and clamp
        state = torch.clamp(state+u, self._joint_limits_lower, self._joint_limits_upper)

        return state
    
    def _mppi_running_cost(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.float:

        """
        MPPI running cost - cost at each waypoint.
        
        :param state: Current joint configuration (BATCH_SIZE x N_STATE)
        :param action: Current waypoint displacement (BATCH_SIZE x N_CONTROL)
        :returns: Cost after applying this action while being at this state
        """
        
        # Compute individual costs
        with torch.no_grad():
            cost_path_length = self._cost_path_length.forward(state, action)
            cost_collision = self._cost_collision.forward(state, self._grasped_object_nominal_control_points, self._grasped_object_grasp_T_object) if self._consider_obstacle_collisions else 0.0
        
        # Evaluate cost
        cost = cost_path_length + cost_collision
        return cost

    def _mppi_terminal_cost(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.float:

        """
        MPPI terminal cost - cost of terminal state.
        
        :param state: Current joint configuration (BATCH_SIZE x N_STATE)
        :param action: Current waypoint displacement (BATCH_SIZE x N_CONTROL)
        :returns: Cost after applying this action while being at this state
        """
        
        # Compute individual costs
        cost_convergence = self._cost_convergence.forward(state)
        
        # Evaluate cost
        cost = cost_convergence
        return cost

    def instantiate_mppi_ja_to_ja(
        self, 
        start_ja: List,
        goal_ja: List,
        mppi_control_limits: List[np.ndarray],
        mppi_nsamples: int = 1000,
        mppi_covariance: float = 0.5,
        mppi_lambda: float = 1.0,
        mppi_cost_weight_convergence = 1000.0,
        mppi_cost_weight_path_length = 200.0,
        mppi_cost_weight_collision = 500.0,
        mppi_consider_obstacle_collisions: bool = True,
    ) -> List:
        
        """
        Given the current joint configuration and a goal joint configuration, instantiate an MPPI object that plans for a joint space trajectory between them.
        
        :param start_ja: The start joint configuration
        :param goal_ja: The goal joint configuration
        :param mppi_control_limits: Limits for MPPI control (max joint angle displacement between two timesteps)
        :param mppi_nsamples: Number of rollout trajectories
        :param mppi_covariance: Control noise covariance value
        :param mppi_lambda: Temperature, positive scalar where larger values will allow more exploration
        :param mppi_cost_weight_convergence: Weight for target convergence cost
        :param mppi_cost_weight_path_length: Weight for path length cost
        :param mppi_cost_weight_collision: Weight for collision cost
        :param mppi_consider_obstacle_collisions: True if collisions with obstacles need to be checked, False otherwise
        :returns: The target joint angles (ROBOT_DOF)
        """
        
        # Set collision checking
        self._consider_obstacle_collisions = mppi_consider_obstacle_collisions

        # Store weights
        self._mppi_cost_weight_convergence = mppi_cost_weight_convergence
        self._mppi_cost_weight_path_length = mppi_cost_weight_path_length
        self._mppi_cost_weight_collision = mppi_cost_weight_collision

        # Store start and goal configurations
        self._start = np.array(start_ja)
        self._goal = np.array(goal_ja)
        self._start_tensor = torch.from_numpy(self._start).double().to(self._device)
        self._goal_tensor = torch.from_numpy(self._goal).double().to(self._device)
        
        # Control limits tensors
        mppi_control_limits_lower = torch.from_numpy(mppi_control_limits[0]).double().to(self._device)
        mppi_control_limits_upper = torch.from_numpy(mppi_control_limits[1]).double().to(self._device)

        # Noise covariance
        noise_sigma = torch.from_numpy(mppi_covariance * np.eye(self._njoints)).double().to(self._device)
        
        # Define cost functions for MPPI
        # 0) Convergence cost
        self._cost_convergence = ConvergenceCost(
            weight = self._mppi_cost_weight_convergence,
            target = self._goal_tensor,
            device = self._device,
        )

        # 1) Path length cost
        self._cost_path_length = PathLengthCost(
            weight = self._mppi_cost_weight_path_length,
            device = self._device,
        )
        
        # 2) Collision cost
        self._cost_collision = CollisionCost(
            weight = self._mppi_cost_weight_collision,
            differentiable_model = self.differentiable_model,
            csdf = self.csdf,
            control_points_number = self._control_points_number,
            link_fixed = self._link_fixed,
            link_skeleton = self._link_skeleton,
            gripper_state = self._gripper_state,
            control_points_json = self._control_points_location,
            device = self._device,
        )

        # Define MPPI object
        self._mppi_obj = mppi.MPPI(
            self._mppi_dynamics,
            self._mppi_running_cost,
            self._njoints,
            noise_sigma,
            self._goal_tensor,
            terminal_state_cost = self._mppi_terminal_cost,
            num_samples = mppi_nsamples,
            device = self._device,
            temperature = mppi_lambda,
            u_min = mppi_control_limits_lower,
            u_max = mppi_control_limits_upper,
        )

        return goal_ja
    
    def update_goal_ja(self, goal_ja: List) -> List:
        """
        :param goal_ja: The new goal joint configuration
        """

        # Update goal and goal tensor
        self._goal = np.array(goal_ja)
        self._goal_tensor = torch.from_numpy(self._goal).double().to(self._device)

        # Update convergence cost
        self._cost_convergence.update_target(self._goal_tensor)

        # Update MPPI object
        self._mppi_obj.update_goal(self._goal_tensor)

        return goal_ja

    def get_mppi_rollout(
        self, 
        current_ja: List,
    ) -> Optional[np.ndarray]:
        
        """
        Given the current joint configuration, get the rollout based on an instantiated MPPI object.
        
        :param current_ja: The start joint configuration
        :returns: A path in joint space, described by N waypoints for the joint angles (N x ROBOT_DOF np.ndarray)
        """

        # Store current_configuration
        current_ja_tensor = torch.from_numpy(np.array(current_ja)).double().to(self._device)

        # Get trajectory rollout
        rollout, _ = self._mppi_obj.command(current_ja_tensor)

        return rollout.cpu().numpy()
    
    def update_obstacle_pcd(
        self,
        pcd: np.ndarray,
    ):

        """
        Update the total point cloud used for obstacle avoidance.
        
        :param pcd: Point cloud (N x 3)
        :param robot_T_camera: Transform from camera to robot base (4*4)
        """

        # Update point cloud in SDF
        self.csdf.update_pcd(pcd)
        return

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
            object_nominal_control_points = np.asarray(object_mesh.get_axis_aligned_bounding_box().get_box_points())
            self._grasped_object_nominal_control_points = torch.from_numpy(object_nominal_control_points).to(self._device)

            # Construct tensor describing grasp_T_object
            grasped_object_grasp_T_object = np.linalg.inv(world_T_grasp) @ world_T_object
            self._grasped_object_grasp_T_object = torch.from_numpy(grasped_object_grasp_T_object).to(self._device)

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
        return True


def planner_test():
    # Candidate poses
    pose1 = np.array([-0.47332507,  1.13872886,  1.30867887, -2.30050802,  2.07975602,  2.64635682, -2.65230727])
    pose2 = np.array([ 2.09151435, -0.54573566, -0.99544001, -2.25478268,  2.02075601,  2.74072695, -1.75231826])
    pose3 = np.array([ 2.17793441, -0.48076588, -0.856754,   -1.67240107,  0.553756,    2.79897308, -0.10493574])
    pose4 = np.array([ 0.45744711,  0.70788223,  0.71865666, -0.27235043,  0.553756,    2.09835196, -0.01765767])
    pose5 = np.array([ 1.52491331, -0.45537129, -0.08102775, -1.83516145,  0.553756,    2.91463614,  0.20733792])

    # Test planning time
    start_time = time.time()
    planner = TrajectoryPlanner()
    print("planning time : ", time.time()-start_time)


def main():
    planner_test()

if __name__ == '__main__':
    main()
