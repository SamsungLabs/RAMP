# Copyright (c) 2023 University of Michigan ARM Lab
# Modified by: Vasileios Vasilopoulos (vasileios.v@samsung.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools
import logging

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

logger = logging.getLogger("MPPI PLANNING")


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray


def squeeze_n(v, n_squeeze):
    for _ in range(n_squeeze):
        v = v.squeeze(0)
    return v


# from arm_pytorch_utilities, standalone since that package is not on pypi yet
def handle_batch_input(n):
    def _handle_batch_input(func):
        """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
            batch_dims = []
            for arg in args:
                if is_tensor_like(arg):
                    if len(arg.shape) > n:
                        # last dimension is type dependent; all previous ones are batches
                        batch_dims = arg.shape[:-(n - 1)]
                        break
                    elif len(arg.shape) < n:
                        n_batch_dims_to_add = n - len(arg.shape)
                        batch_ones_to_add = [1] * n_batch_dims_to_add
                        args = [v.view(*batch_ones_to_add, *v.shape) if is_tensor_like(v) else v for v in args]
                        ret = func(*args, **kwargs)
                        if isinstance(ret, tuple):
                            ret = [squeeze_n(v, n_batch_dims_to_add) if is_tensor_like(v) else v for v in ret]
                            return ret
                        else:
                            if is_tensor_like(ret):
                                return squeeze_n(ret, n_batch_dims_to_add)
                            else:
                                return ret
            # no batches; just return normally
            if not batch_dims:
                return func(*args, **kwargs)

            # reduce all batch dimensions down to the first one
            args = [v.view(-1, *v.shape[-(n - 1):]) if (is_tensor_like(v) and len(v.shape) > 2) else v for v in args]
            ret = func(*args, **kwargs)
            # restore original batch dimensions; keep variable dimension (nx)
            if type(ret) is tuple:
                ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                    v.view(*batch_dims, *v.shape[-(n - 1):]) if len(v.shape) == n else v.view(*batch_dims)) for v in
                       ret]
            else:
                if is_tensor_like(ret):
                    if len(ret.shape) == n:
                        ret = ret.view(*batch_dims, *ret.shape[-(n - 1):])
                    else:
                        ret = ret.view(*batch_dims)
            return ret

        return wrapper

    return _handle_batch_input


class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(
        self,
        dynamics,
        running_cost,
        nx,
        noise_sigma,
        goal,
        num_samples=100,
        temperature=1.,
        terminal_state_cost=None,
        u_min=None,
        u_max=None,
        waypoint_density=10,
        action_smoothing=0.5,
        cov_smoothing=0.0,
        noise_mu=None,
        step_dependent_dynamics=False,
        sample_null_noise=True,
        noise_abs_cost=False,
        device="cpu",
    ):

        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param goal: Desired target to reach (nx)
        :param num_samples: K, number of trajectories to sample
        :param temperature: temperature, positive scalar where larger values will allow more exploration
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param waypoint_density: Number of state waypoints per unit length
        :param action_smoothing: Smoothing between previous and next control rollout
        :param cov_smoothing: Smoothing between previous and next used covariance
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param sample_null_noise: Whether to explicitly sample a zero noise
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        :param device: pytorch device
        """

        # Basic MPPI parameters
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = 15  # HORIZON - initialization (to be dynamically updated later)
        self.waypoint_density = waypoint_density

        # Dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]

        # Temperature (lambda)
        self.temperature = temperature

        # Noise mean value
        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        # Handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)
        
        # Goal
        self.goal = goal

        # Bounds and scale
        self.u_min = u_min
        self.u_max = u_max

        # Make sure if any of the bounds is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)
        
        # Noise parameters and distribution
        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        
        # Action and covariance smoothing
        self.action_smoothing = action_smoothing
        self.cov_smoothing = cov_smoothing
        
        # MPPI dynamics and running cost parameters
        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_noise = sample_null_noise
        self.noise_abs_cost = noise_abs_cost
        self.state = None

        # Sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

        # Number of times the command action has been called
        self.num_steps = 0

    @handle_batch_input(n=2)
    def _dynamics(self, state, u, t):
        return self.F(state, u, t) if self.step_dependency else self.F(state, u)

    @handle_batch_input(n=2)
    def _running_cost(self, state, u):
        return self.running_cost(state, u)
    
    def update_goal(self, goal):
        """
        :param goal: New target that will overwrite the current goal (nx)
        """

        # Update goal
        self.goal = goal

    def command(self, state):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns: A tuple of (state_rollout, action_rollout) from the current state
        """

        return self._command(state)

    def _command(self, state):

        # Register state
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)

        # Initialize action and state rollouts
        if self.num_steps == 0:
            initial_state_rollout = torch.zeros((2,self.nx)).to(dtype=self.dtype, device=self.d)
            initial_state_rollout[0] = self.state
            initial_state_rollout[1] = self.goal
            num_waypoints = max(int(torch.floor(torch.linalg.norm(self.goal - self.state) * self.waypoint_density)), 2)
            initial_state_rollout_interpolated = torch.nn.functional.interpolate(initial_state_rollout.unsqueeze(0).transpose(1,2), size=num_waypoints, mode='linear', align_corners=True).transpose(1,2).squeeze(0)
            self.state_rollout = initial_state_rollout_interpolated
            self.action_rollout = torch.roll(self.state_rollout, -1, dims=0) - self.state_rollout
            self.action_rollout[-1] = 0.0
        else:
            # Upsample current trajectory
            interpolated_state_rollout = torch.nn.functional.interpolate(self.state_rollout.unsqueeze(0).transpose(1,2), size=500, mode='linear', align_corners=True).transpose(1,2).squeeze(0)
            
            # Find waypoint that is closest to the current configuration
            distance_diff = self.state - interpolated_state_rollout
            distance_diff_norm = torch.linalg.norm(distance_diff, dim=1)
            distance_diff_norm_min = torch.min(distance_diff_norm, axis=0)
            closest_waypoint_idx = distance_diff_norm_min.indices
            closest_waypoint = interpolated_state_rollout[closest_waypoint_idx]
            if len(closest_waypoint.shape) > 1:
                closest_waypoint_idx = closest_waypoint_idx[0]
                closest_waypoint = closest_waypoint[0]
            
            # Construct new interpolated trajectory
            new_interpolated_state_rollout = torch.vstack((self.state, interpolated_state_rollout[closest_waypoint_idx:]))
            new_interpolated_state_rollout = torch.vstack((new_interpolated_state_rollout, self.goal))

            # Downsample trajectory
            self.state_rollout = self._downsample_trajectory(new_interpolated_state_rollout, 1.0/self.waypoint_density)
            self.action_rollout = torch.roll(self.state_rollout, -1, dims=0) - self.state_rollout
            self.action_rollout[-1] = 0.0

        # Sample trajectories and compute total cost of all trajectories
        cost_total = self._compute_total_cost_batch()

        # Compute weights omega
        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.temperature)
        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        
        # Compute new action rollout as the weighted mean, and apply smoothing
        new_action_rollout = self.action_rollout + torch.sum(self.omega.view(-1, 1, 1) * self.noise, dim=0)
        self.action_rollout = (1.0 - self.action_smoothing) * self.action_rollout + self.action_smoothing * new_action_rollout

        # Update covariance - taken from STORM
        weighted_noise = self.omega * (self.noise ** 2).T
        cov_update = torch.diag(torch.mean(torch.sum(weighted_noise.T, dim=0), dim=0))
        self.noise_sigma = (1.0 - self.cov_smoothing) * self.noise_sigma + self.cov_smoothing * cov_update
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        # Compute action rollout to return
        action_rollout = self.action_rollout

        # Compute the state rollout and add the goal at the end
        self.state_rollout = self.get_rollout(self.state, action_rollout)[0]
        state_rollout = torch.vstack((self.state_rollout, self.goal))

        # Upsample and downsample the trajectory
        interpolated_state_rollout = torch.nn.functional.interpolate(state_rollout.unsqueeze(0).transpose(1,2), size=500, mode='linear', align_corners=True).transpose(1,2).squeeze(0)
        state_rollout = self._downsample_trajectory(interpolated_state_rollout, 1.0/self.waypoint_density)

        self.num_steps += 1

        return state_rollout, action_rollout

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.action_rollout = self.noise_dist.sample((self.T,))

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        # Initialize total cost
        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)

        # Allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        # Actions is K x T x nu
        # States is K x T x nx
        states = torch.zeros((K, T, self.nx), device=self.d, dtype=self.dtype)
        actions = torch.zeros((K, T, self.nu), device=self.d, dtype=self.dtype)
        current_state = state
        for t in range(T):
            u = perturbed_actions[:, t]
            current_state = self._dynamics(current_state, u, t)
            states[:, t] = current_state
            actions[:, t] = u

        cost_total += self._running_cost(states.view(-1, self.nx), actions.view(-1, self.nu)).view(K, T).sum(dim=1)

        # Action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_total += c
        return cost_total, states, actions

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.action_rollout.shape[0]))
        # broadcast own control to noise over samples; now it's K x T x nu
        if self.sample_null_noise:
            self.noise[self.K - 1] = 0
        self.perturbed_action = self.action_rollout + self.noise
        # naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.action_rollout
        if self.noise_abs_cost:
            action_cost = self.temperature * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.temperature * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.temperature * self.noise @ self.noise_sigma_inv  # Like original paper

        self.cost_total, self.states, self.actions = self._compute_rollout_costs(self.perturbed_action)

        # Action perturbation cost
        perturbation_cost = torch.abs(torch.sum(self.action_rollout * action_cost, dim=(1, 2)))
        self.cost_total += perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            action_bounded = torch.max(torch.min(action, self.u_max), self.u_min)
        return action_bounded
    
    def _downsample_trajectory(self, trajectory, path_interval):
        # Remove based on path_interval
        idx = 0
        new_path_points = []
        new_path_points.append(trajectory[idx,:])
        while(1):
            lengths = torch.linalg.norm(trajectory[idx:,:] - new_path_points[-1], axis = 1)

            if idx == (trajectory.shape[0]-1):
                break
            else:
                if torch.where(lengths > path_interval)[0].shape[0] == 0:
                    new_path_points.append(trajectory[-1,:])
                    break
                else:
                    interval = torch.where(lengths > path_interval)[0][0]
                    if interval > 1:
                        idx = idx + interval - 1
                    else:
                        idx = idx + interval
                    new_path_points.append(trajectory[idx,:])

        return torch.stack(new_path_points, dim = 0)

    def get_rollout(self, state, action_sequence, num_rollouts=1):

        """
        :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
        :param action_sequence: Action sequence (T x nu)
        :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic dynamics
        :returns: states (num_rollouts x T x nx vector of trajectories)
        """

        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        T = action_sequence.shape[0]
        states = torch.zeros((num_rollouts, T + 1, self.nx), dtype=self.action_rollout.dtype, device=self.action_rollout.device)
        states[:, 0] = state
        for t in range(T):
            states[:, t + 1] = self._dynamics(states[:, t].view(num_rollouts, -1), action_sequence[t].view(num_rollouts, -1), t)
        return states[:, 1:]
