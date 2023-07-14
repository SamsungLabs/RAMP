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
# import torch.nn.functional as F

from .gaussian_projection import GaussianProjection

class ConvergenceCost(nn.Module):
    def __init__(
        self,
        weight: float,
        target: torch.Tensor,
        gaussian_params: dict = {'n':0,'c':0,'s':0,'r':0},
        device: torch.device = torch.device('cpu'),
        float_dtype: torch.dtype = torch.float32,
    ):

        super(ConvergenceCost, self).__init__()
        self._device = device
        self._float_dtype = float_dtype
        self._weight = torch.as_tensor(weight, device=self._device, dtype=self._float_dtype)
        self._proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        
        self._target = target

    def update_target(self, target: torch.Tensor):
        """
        :param target: New target that will overwrite the current goal
        """

        # Update target
        self._target = target

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        
        """
        Computation of convergence cost for a batch of robot configurations.

        :param state: Joint configuration (NUM_ROLLOUTS x HORIZON x NDOF)
        :returns: Convergence cost (NUM_ROLLOUTS)
        """
        
        cost = self._weight * self._proj_gaussian(torch.linalg.norm(state[:, -1]-self._target, ord=2, dim=1))

        return cost.to(self._device)