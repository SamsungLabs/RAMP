#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import time
import torch
import torch.nn as nn
import torch.nn.functional as Functional

from .gaussian_projection import GaussianProjection

class CollisionCost(nn.Module):
    def __init__(
        self,
        weight: float,
        differentiable_model: object,
        csdf: object,
        control_points_number: int,
        link_fixed: str,
        link_skeleton: str,
        gripper_state: torch.Tensor,
        distance_threshold: float = 0.03,
        gaussian_params: dict = {'n':0,'c':0,'s':0,'r':0},
        device: torch.device = torch.device('cpu'),
        float_dtype: torch.dtype = torch.float32,
    ):
        super(CollisionCost, self).__init__()
        self._device = device
        self._float_dtype = float_dtype
        self._weight = torch.as_tensor(weight, device=self._device, dtype=self._float_dtype)
        self._proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        
        self._differentiable_model = differentiable_model
        self._csdf = csdf
        self._distance_threshold = distance_threshold
        
        self._control_points_number = control_points_number
        self._link_fixed = link_fixed
        self._link_skeleton = link_skeleton
        self._gripper_state = gripper_state
    
    def _get_control_points(self, state: torch.Tensor) -> torch.Tensor:
        
        """
        Receives a robot configuration and returns a list of all control points on the manipulator.
        
        :param state: Current joint configuration (BATCH_SIZE x N_STATE)
        :returns: List of control points on the robot manipulator (BATCH_SIZE x CONTROL_POINTS x 3)
        """
        
        batch_size = state.shape[0]
        
        # Find link locations after stacking robot configuration with gripper state
        augmented_robot_state = torch.cat((state, torch.tile(self._gripper_state, (batch_size, 1))), dim=1)
        link_transformations = self._differentiable_model.compute_forward_kinematics_all_links(augmented_robot_state)
        
        # Initialize skeleton for control points - tensor should be BATCH_SIZE x 1 x 3
        skeleton_control_point_locations = torch.zeros((batch_size, len(self._link_skeleton), 3)).to(self._device)
        
        # Find skeleton control points
        for link_idx in range(len(self._link_skeleton)):
            skeleton_control_point_locations[:, link_idx, :] = link_transformations[self._link_skeleton[link_idx]][0]

        # Find end effector poses
        dummy_row = torch.zeros((batch_size, 1, 4)).to(self._device)
        dummy_row[:, :, 3] = 1.0
        self.ee_pose = torch.cat((link_transformations[self._link_skeleton[-1]][1], link_transformations[self._link_skeleton[-1]][0].unsqueeze(2)), dim = 2)
        self.ee_pose = torch.cat((self.ee_pose, dummy_row), dim = 1)
        
        # Augment control points based on the skeleton
        control_point_locations = Functional.interpolate(skeleton_control_point_locations.transpose(1,2), size=self._control_points_number, mode='linear', align_corners=True).transpose(1,2)
        
        return control_point_locations
    
    def forward(
        self,
        state: torch.Tensor,
        grasped_object_nominal_control_points: torch.Tensor = None,
        grasped_object_grasp_T_object: torch.Tensor = None,
    ) -> torch.Tensor:
        
        """
        Computation of collision cost for a batch of robot configurations.
        
        :param state: Joint configuration (BATCH_SIZE x N_STATE)
        :param grasped_object_nominal_control_points: Nominal control points for grasped object in object frame (N_POINTS x 3)
        :param grasped_object_grasp_T_object: 4x4 transform from robot's gripper to object frame (4 x 4)
        :returns: Collision cost (BATCH_SIZE)
        """
        
        # Compute robot control point locations through FK for given state
        robot_control_points = self._get_control_points(state)

        # Compute grasped object control points
        if grasped_object_grasp_T_object is not None:
            object_pose = self.ee_pose[:, ] @ grasped_object_grasp_T_object.to(dtype = self._float_dtype)
            object_control_points = object_pose @ torch.hstack((
                grasped_object_nominal_control_points.to(dtype = self._float_dtype),
                torch.ones((grasped_object_nominal_control_points.shape[0],1)).to(dtype = self._float_dtype, device = self._device)
            )).transpose(0,1)
            object_control_points = object_control_points.transpose(1,2)[:, :, :3]
            control_points = torch.cat((robot_control_points, object_control_points), dim=1)
        else:
            control_points = robot_control_points
        
        # Evaluate C-SDF of these points
        sdf_values = self._csdf.forward(control_points)
        
        # Evaluate collision cost
        cost_sdf = torch.zeros_like(sdf_values)
        positive = sdf_values > self._distance_threshold
        negative = sdf_values <= self._distance_threshold
        cost_sdf[positive] = torch.div(torch.tensor(self._distance_threshold), sdf_values[positive])
        cost_sdf[negative] = torch.tensor(1.0)
        cost = self._weight * self._proj_gaussian(cost_sdf)

        return cost.to(self._device)