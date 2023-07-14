"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Jinwook Huh (jinwook.huh@samsung.com)
Vasileios Vasilopoulos (vasileios.v@samsung.com; vasilis.vasilop@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""

import argparse
import random
import open3d as o3d
import trimesh
import numpy as np
from sim_env import RobotEnvironment
from sim_env import RAMPPolicy
from trajectory_following.trajectory_following import (
    LINK_FIXED,
    LINK_EE,
    LINK_SKELETON,
    JOINT_LIMITS,
    robot_urdf_location,
    TrajectoryFollower,
)
from sim_env.constants import ROBOT_LABEL, START_SPHERE_LABEL, GOAL_SPHERE_LABEL
import trimesh.transformations as tra

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Toggle the visualization of start-goal markers
    parser.add_argument("--markers", type=bool, default=False)

    # Determine the experiment we want to run using this convention:
    # experiment = 01 : Experiment 0 (static environment), Scene 1
    # experiment = 02 : Experiment 0 (static environment), Scene 2
    # experiment = 10 : Experiment 1 (dynamic environment), Scene 0 ... etc
    parser.add_argument("--experiment", type=int, default=0)

    # Number of start-goal configuration pairs that we want to evaluate the algorithm on
    parser.add_argument("--test_config_pairs_num", type=int, default=101)

    # Minimum distance between any pair of start-goal configurations sampled for evaluation
    parser.add_argument("--test_config_min_dist", type=float, default=0.3)

    # Standard arguments needed for the simulation
    parser.add_argument("--compute-device-id", type=int, default=0)
    parser.add_argument("--graphics-device-id", type=int, default=1)
    parser.add_argument("--headless", action="store_true", default=False)

    # Robot URDF location arguments
    parser.add_argument("--robot-asset-root", type=str, default="resources/")
    parser.add_argument("--robot-asset-file", type=str, default="panda/panda.urdf")
    parser.add_argument("--ee-link-name", type=str, default="panda_hand")

    # Other objects location and convex decomposition cache location arguments
    parser.add_argument("--dataset-root", type=str, default="output_shapenet")
    parser.add_argument("--urdf-cache", type=str, default="datasets/scene_cache")
    parser.add_argument("--num-objects", type=int, default=2)

    # Toggle visualization of feasible configurations
    parser.add_argument("--invisible_feasible_points", action="store_true", default=True)

    # Simulation camera properties
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--mppi-cam", type=str, default="ws")

    # Number of points in observed point cloud from the camera
    parser.add_argument("--npoints", type=int, default=2048)

    # Nominal main simulation loop frequency
    parser.add_argument("--control-frequency", type=float, default=20)

    # Distance threshold for switching to another goal
    parser.add_argument("--mppi-transition-threshold", type=float, default=0.15)

    # Trajectory generation update frequency in Hz
    parser.add_argument("--mppi-update-frequency", type=float, default=2.0)

    # Name of log file
    parser.add_argument("--log-file", type=str)

    args = parser.parse_args()
    # rollout(args)


    # Use the arguments to construct the simulation environment
    env = RobotEnvironment(args)

    # RAMP policy - Trajectory Planner initialization
    policy = RAMPPolicy(
        transition_threshold=args.mppi_transition_threshold,
        cam_type=args.mppi_cam,
        device=args.compute_device_id,
        log_file=args.log_file,
    )

    # Trajectory Follower initialization
    trajectory_follower = TrajectoryFollower(
        joint_limits = JOINT_LIMITS,
        robot_urdf_location = robot_urdf_location,
        link_fixed = LINK_FIXED,
        link_ee = LINK_EE,
        link_skeleton = LINK_SKELETON,
    )

    # Variable initialization
    next_obs, info = env.step()
    current_joint_angles = next_obs['robot_q']
    previous_update_time = 0.0
    current_time = info["time"]
    action_needed = True

    # Construct static point clouds
    static_pcds = []
    for i, (obj_name, obj_info) in enumerate(
        env.scene_manager.objs.items()
    ):
        # Find object pose
        object_pose = env.scene_manager.objs[obj_name]['pose'].copy()
        if obj_name == 'table':
            rotated_object_pose = tra.euler_matrix(-np.pi / 2, 0, 0) @ object_pose
            object_pose[:3, :3] = rotated_object_pose[:3, :3]

        # Transform the mesh based on its pose
        transformed_mesh = env.scene_manager.objs[obj_name]['mesh'].copy().apply_transform(object_pose)

        # Get point cloud and append to list
        transformed_mesh_pcd = trimesh.sample.sample_surface(transformed_mesh, 2000)[0]
        static_pcds.append(transformed_mesh_pcd)
    static_pcds = np.concatenate(static_pcds)

    # Store camera pose
    camera_pose = tra.euler_matrix(np.pi / 2, 0, 0)@ next_obs["camera_pose"][1]

    # Point cloud labels of interest
    label_map = {
        "robot": ROBOT_LABEL,
        "start_sphere": START_SPHERE_LABEL,
        "goal_sphere": GOAL_SPHERE_LABEL,
    }
   
    while True:
        # Get point cloud in camera frame and associated point cloud labels
        complete_pc_camera_frame = next_obs["pc"][1]
        complete_pc_labels = next_obs["pc_label"][1]

        # Get point cloud labels that don't correspond to the robot or the sphere markers
        scene_pc_mask = (complete_pc_labels != label_map["robot"]) & (complete_pc_labels != label_map["start_sphere"]) & (complete_pc_labels != label_map["goal_sphere"])

        # Get point cloud labels that correspond to the robot
        robot_pc_mask = (complete_pc_labels == label_map["robot"])

        # Transform point clouds into world frame (z up)
        complete_pc_world_frame = tra.transform_points(complete_pc_camera_frame, camera_pose)
        scene_pc = np.concatenate([static_pcds, complete_pc_world_frame[scene_pc_mask,:]])
        robot_pc = complete_pc_world_frame[robot_pc_mask, :]

        # Update the trajectory if there is no current trajectory or based on a specified frequency
        if action_needed or (current_time-previous_update_time > 1.0/args.mppi_update_frequency):
            # Get trajectory with the most recent scene point cloud
            trajectory = policy.get_action(next_obs, scene_pc)

            # Update flag and update time
            previous_update_time = current_time
            action_needed = False
        
        # Update the scene point cloud used by the trajectory follower
        trajectory_follower.update_obstacle_pcd(scene_pc)
        
        # Compute velocity command using the trajectory follower
        if trajectory is not None:
            # Update trajectory follower with the most recent trajectory
            augmented_trajectory = trajectory[:,:7]
            trajectory_follower.update_trajectory(augmented_trajectory)

            # Compute velocity command
            velocity_command = trajectory_follower.follow_trajectory(current_joint_angles[:7])

            # Step the environment
            next_obs, info = env.step(velocity_command)
            current_time = info["time"]
        else:
            # Step the environment and raise the flag that a new trajectory is needed 
            next_obs, info = env.step()
            action_needed = True
        
        # Visualize point cloud if necessary
        # pcdo3d = o3d.geometry.PointCloud()
        # pcdo3d.points = o3d.utility.Vector3dVector(scene_pc)
        # o3d_axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([pcdo3d, o3d_axis])

        # Update robot joints
        current_joint_angles = next_obs['robot_q']
        
        # Evaluate safety distance
        env.get_min_dist(scene_pc, robot_pc)
        if info["done"]:
            break
