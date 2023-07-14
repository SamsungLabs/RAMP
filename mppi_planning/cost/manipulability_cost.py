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

import torch
import torch.nn as nn

from .gaussian_projection import GaussianProjection


class ManipulabilityCost(nn.Module):
    def __init__(
        self,
        weight: float,
        ndofs: int,
        differentiable_model: object,
        link_ee: str,
        gripper_state: torch.Tensor,
        thresh: float = 0.1,
        gaussian_params: dict = {'n':0,'c':0,'s':0,'r':0},
        device: torch.device = torch.device('cpu'),
        float_dtype: torch.dtype = torch.float32,
    ):
        super(ManipulabilityCost, self).__init__() 
        self._device = device
        self._float_dtype = float_dtype
        self._weight = torch.as_tensor(weight, device=self._device, dtype=self._float_dtype)
        self._proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)

        self._ndofs = ndofs
        self._differentiable_model = differentiable_model
        self._link_ee = link_ee
        self._gripper_state = gripper_state
        self._thresh = thresh
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        
        """
        Computation of manipulability costs for a batch of robot configurations.
        
        :param state: Joint configuration (BATCH_SIZE x N_STATE)
        :returns: Manipulability cost (BATCH_SIZE)
        """
        
        batch_size = state.shape[0]
        
        # Find jacobians after stacking robot configuration with gripper state
        augmented_robot_state = torch.cat((state, torch.tile(self._gripper_state, (batch_size, 1))), dim=1)
        jac_batch_linear, jac_batch_angular = self._differentiable_model.compute_endeffector_jacobian(augmented_robot_state, self._link_ee)
        jac_batch = torch.cat((jac_batch_linear, jac_batch_angular), dim=1)
        
        with torch.cuda.amp.autocast(enabled=False):
            J_J_t = torch.matmul(jac_batch, jac_batch.transpose(-2,-1))
            score = torch.sqrt(torch.det(J_J_t))
        score[score != score] = 0.0
        
        score[score > self._thresh] = self._thresh
        score = (self._thresh - score) / self._thresh

        cost = self._weight * self._proj_gaussian(score) 
        
        return cost.to(self._device)