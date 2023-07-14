"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Vasileios Vasilopoulos (vasileios.v@samsung.com; vasilis.vasilop@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""

# General imports
import logging
import numpy as np
import random
import time

# NN imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from csdf.utils.chamfer_distance import ChamferDistance


class PointCloud_CSDF(nn.Module):
    def __init__(
        self,
        pcd: np.ndarray,
        sphere_radius: float = 0.02,
        device: str = 'cuda'
    ):
        
        """
        Brute force estimation of SDF for a point cloud.
        
        :param pcd: N x 3 numpy array describing the point cloud
        :param sphere_radius: Radius of sphere enclosing each point in the point cloud
        :param device: Device to load the point cloud
        """
        
        super().__init__()
        
        self._device = device
        self._sphere_radius = sphere_radius
        
        self.pcd = torch.from_numpy(pcd).float().to(self._device)
        
        self.chamfer_distance = ChamferDistance()
    
    def update_pcd(self, pcd: np.ndarray):

        """
        Function for updating the internal point cloud.
        
        :param pcd: N x 3 numpy array describing the point cloud
        """

        self.pcd = torch.from_numpy(pcd).float().to(self._device)
    
    def compute_distances(self, x: torch.Tensor):

        """
        Function for computing the distances of passed points to the internally saved point cloud.
        
        :param x: batch_size x num_points x 3 query points
        :returns: batch_size x num_points distance values
        """

        # Save shape features of input (batch_size x num_points x 3)
        batch_size = x.shape[0]
        num_points = x.shape[1]
        
        # Compute distance (ChamferDistance returns the squared distance between the point clouds)
        dist_x_to_pcd, _ = self.chamfer_distance(x.reshape(-1, 3).unsqueeze(0), self.pcd.unsqueeze(0))
        dist_x_to_pcd = torch.sqrt(dist_x_to_pcd.reshape((batch_size, num_points))) - self._sphere_radius

        return dist_x_to_pcd
        
    def forward(self, x: torch.Tensor):

        """
        Function for returning C-SDF values of passed points to the internally saved point cloud.
        
        :param x: batch_size x num_points x 3 query points
        :returns: batch_size C-SDF values
        """

        # Compute distances
        dist_x_to_pcd = self.compute_distances(x)
        
        # Compute C-SDF values
        csdf_values = torch.min(dist_x_to_pcd, axis = 1).values

        return csdf_values


if __name__ == "__main__":
    DEVICE = "cuda"
    CONTROL_POINTS = 70
    
    # pcd = np.load('/home/vasileiosv/scene_pcd.npy')
    pcd = np.random.rand(100000,3)

    model = PointCloud_CSDF(pcd, device=DEVICE)
    model.eval()
    model.to(DEVICE)
    
    since = time.time()
    points = torch.rand((500, CONTROL_POINTS, 3), device=DEVICE, requires_grad=True)

    # with torch.no_grad():
    sdf_values = model(points)
    print(f'Total time to compute the SDF value: {time.time()-since}')
    
    total_sdf_value = sdf_values.sum()
    since = time.time()
    total_sdf_value.backward()
    sdf_gradient = points.grad
    sdf_gradient = sdf_gradient[torch.nonzero(sdf_gradient).data[0][0]]
    print(f'Total time to compute the SDF gradient: {time.time()-since}')
    print(f'SDF values: {sdf_values.cpu().detach().numpy()}')
    print(f'SDF gradient: {sdf_gradient.cpu().detach().numpy()}')
