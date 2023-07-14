# RAMP

This package includes the code for [RAMP: Hierarchical Reactive Motion Planning for Manipulation Tasks Using Implicit Signed Distance Functions](https://arxiv.org/abs/2305.10534), developed by Samsung Research (Samsung AI Center - New York) and presented at IROS 2023.

For simulation demonstrations, we use the [Isaac Gym](https://developer.nvidia.com/isaac-gym) physics simulation environment from NVIDIA, as well as modified environment generation code from [SceneCollisionNet](https://github.com/NVlabs/SceneCollisionNet), included in the [sim_env](sim_env) folder. For the computation of differentiable forward kinematics, the package uses [differentiable-robot-model](https://github.com/facebookresearch/differentiable-robot-model) from Meta Research.

For more information, please check our [project page](https://samsunglabs.github.io/RAMP-project-page/).

## Installation and running

### Environment setup

1. Create a python virtual environment inside the repo, source it and update pip:
        
	```bash
    python3 -m venv pyvenv
    source pyvenv/bin/activate
    pip install --upgrade pip
    ```

2. Install the requirements. **When installing PyTorch, make sure that the PyTorch version matches your installed CUDA version by updating `--extra-index-url`, for example: https://download.pytorch.org/whl/cu118 for CUDA 11.8. You can check your CUDA version by running: `nvcc --version`.**

	```bash
    pip install -r requirements.txt
	```

3. Install PyTorch3D with GPU support:

	```bash
    pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
	```

4. Download [Isaac Gym](https://developer.nvidia.com/isaac-gym/download) and copy the `isaacgym` folder into `extern` (Prerequisites:Ubuntu 18.04, or 20.04. Python 3.6, 3.7, or 3.8).


### Setup all packages

Install all packages by running:
```bash
pip install -e .
```

### Run without any arguments
Run the simulation:
```bash
python -m test_ramp_simulation
```

### Run with optional arguments
Optionally, you can visualize the start (green) and goal (red) sphere markers with the `--markers` flag and/or you can specify an experiment to run with the `--experiment` flag. For demonstration purposes, we have included 5 static environment scenarios (numbered 0-4) and 5 dynamic environment scenarios (numbered 10-14). The full list of all available arguments is included near the top of the [main script](test_ramp_simulation.py).
```bash
python -m test_ramp_simulation --markers True --experiment 10
```

## Known Issues 
1. If you see the error `WARNING: lavapipe is not a conformant vulkan implementation, testing use only.`, try the following command:
    ```bash
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    ```

## Use Outside of Isaac Gym Environment - API Description
In this package, RAMP is wrapped around an Isaac-Gym simulation environment. To use the algorithm for your own application:
1. You need to copy over to your project the [mppi_planning](mppi_planning) (which includes the trajectory generator) and [trajectory_following](trajectory_following) (which includes the trajectory follower) folders.
2. You need to instantiate a `TrajectoryPlanner` (see [trajectory_planning](mppi_planning/trajectory_planning.py)) class, for example:
    ```python
    # Robot parameters
    JOINT_LIMITS = [
        np.array([-2.8973, -1.7628, -2.8973, -
                    3.0718, -2.8973, -0.0175, -2.8973]),
        np.array([2.8973, 1.7628, 2.8973, -
                    0.0698, 2.8973, 3.7525, 2.8973])
    ]

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

    robot_urdf_location = 'resources/panda/panda.urdf'
    scene_urdf_location = 'resources/environment/environment.urdf'

    # Instantiate trajectory planner
    self.trajectory_planner = TrajectoryPlanner(
        joint_limits=JOINT_LIMITS,
        robot_urdf_location=robot_urdf_location,
        scene_urdf_location=scene_urdf_location,
        link_fixed=LINK_FIXED,
        link_ee=LINK_EE,
        link_skeleton=LINK_SKELETON,
    )
    ```
3. You need to instantiate a `TrajectoryFollower` (see [trajectory_following](trajectory_following/trajectory_following.py)) class, for example:
    ```python
    # Trajectory Follower initialization
    trajectory_follower = TrajectoryFollower(
        joint_limits = JOINT_LIMITS,
        robot_urdf_location = robot_urdf_location,
        link_fixed = LINK_FIXED,
        link_ee = LINK_EE,
        link_skeleton = LINK_SKELETON,
    )
    ```

4. With the **trajectory planner** object, you can instantiate a motion planning problem for RAMP by calling the `instantiate_mppi_ja_to_ja` method of `TrajectoryPlanner` and passing the required parameters as well as the current and target joint angles, for example:
    ```python
    # MPPI parameters
    N_JOINTS = 7
    mppi_control_limits = [
        -0.05 * np.ones(N_JOINTS),
        0.05 * np.ones(N_JOINTS)
    ]
    mppi_nsamples = 500
    mppi_covariance = 0.005
    mppi_lambda = 1.0

    # Instantiate MPPI object
    self.trajectory_planner.instantiate_mppi_ja_to_ja(
        current_joint_angles,
        target_joint_angles,
        mppi_control_limits=mppi_control_limits,
        mppi_nsamples=mppi_nsamples,
        mppi_covariance=mppi_covariance,
        mppi_lambda=mppi_lambda,
    )
    ```

    Then, we offer the following functionalities:
    
    1. You can update the obstacle point cloud used for planning by calling the `update_obstacle_pcd` method of `TrajectoryPlanner`, for example:
        ```python
        self.trajectory_planner.update_obstacle_pcd(pcd=pcd)
        ```
    
    2. You can run an MPC iteration to get the current trajectory by calling the `get_mppi_rollout` method of `TrajectoryPlanner`, for example:
        ```python
        trajectory = self.trajectory_planner.get_mppi_rollout(current_joint_angles)
        ```
    
    3. You can update the current target without instantiating a new motion planning problem (the most recent trajectory will be used to warm-start the search) by calling the `update_goal_ja` method of `TrajectoryPlanner`, for example:
        ```python
        self.trajectory_planner.update_goal_ja(new_target_joint_angles)
        ```

5. With the **trajectory follower** object:
    1. You can update the currently followed trajectory when needed with the `update_trajectory` method of `TrajectoryFollower`, for example:
        ```python
        trajectory_follower.update_trajectory(trajectory)
        ```

    2. You can update the obstacle point cloud used for obstacle avoidance by calling the `update_obstacle_pcd` method of `TrajectoryFollower`, for example:
        ```python
        trajectory_follower.update_obstacle_pcd(new_pcd)
        ```

    3. You can extract the commanded joint velocities at each control iteration by calling the `follow_trajectory` method of `TrajectoryFollower`, for example:
        ```python
        velocity_command = trajectory_follower.follow_trajectory(current_joint_angles)
        ```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/) (CC BY-NC).

## Citation

If you find this work useful, please consider citing:

```
@inproceedings{vasilopoulos2023ramp,
  title={{RAMP: Hierarchical Reactive Motion Planning for Manipulation Tasks Using Implicit Signed Distance Functions}},
  author={Vasilopoulos, Vasileios and Garg, Suveer and Piacenza, Pedro and Huh, Jinwook and Isler, Volkan},
  booktitle={{IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}},
  year={2023}
}
```
