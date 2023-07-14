import copy
import os
import numpy as np
import math
import trimesh.transformations as tra
from autolab_core import Logger
from isaacgym import gymapi
from isaacgym import gymutil
from .utils import get_valid_configs, get_all_valid_positions
from .scene import SceneManager
from . import utils
from .constants import (
    DEPTH_CLIP_RANGE,
    ROBOT_LABEL,
    START_SPHERE_LABEL,
    GOAL_SPHERE_LABEL,
    ROBOT_Q_INIT,
    TABLE_LABEL,
)
import torch
from csdf.pointcloud_sdf import PointCloud_CSDF


class CameraObservation:
    __slots__ = (
        "cam_pose",
        "proj_matrix",
        "rgb",
        "depth",
        "segmentation",
        "pc",
    )

    def __init__(
        self,
        cam_pose=None,
        proj_matrix=None,
        rgb=None,
        depth=None,
        segmentation=None,
        pc=None,
    ):
        self.cam_pose = cam_pose
        self.proj_matrix = proj_matrix
        self.rgb = rgb
        self.depth = depth
        self.segmentation = segmentation
        self.pc = pc


class RobotEnvironment(object):
    logger = Logger.get_logger("RobotEnvironment")

    def __init__(self, args):
        self.args = args
        self._gym = gymapi.acquire_gym()
        self._create_sim()
        self._create_env()
        self.done_plan = False
        # Create viewer if not headless
        if args.headless:
            self.viewer = None
        else:
            self.logger.info("adding viewer")
            self.viewer = self._gym.create_viewer(
                self._sim, gymapi.CameraProperties()
            )
            if self.viewer is None:
                raise ValueError("failed to create viewer!!!")

            cam_pos = gymapi.Vec3(3, 2.0, 0)
            cam_target = gymapi.Vec3(-0.0, 0, 0)

            self._gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target
            )

        # Show feasible positions
        if not args.invisible_feasible_points:
            self._show_feasible_configs()

        # Set up interactions with viewer
        if self.viewer is not None:
            self._gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_P, "pause"
            )
            self._gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_N, "next_target"
            )

        self.playing = True
        self.last_refresh = -1
        self.current_plan = None
        self.current_plan_index = 0
        # Since stepping with action=None happens before anything, self._next_target with set this to 0
        self.target_idx = 0
        # Variables for metrics, requires measuring time for timeouts
        self.iteration_is_collision_free = True
        self.iteration_has_timed_out = False
        self.total_iterations = -1
        self.succesful_iterations = 0
        self.num_collision_free_iterations = 0
        self.iteration_start_time = None
        self.iteration_timeout = 100.0
        self.iteration_durations = []
        self.iteration_safe_min_distances = []
        self.csdf = PointCloud_CSDF(np.random.rand(100000,3), device='cuda')
        self.min_dist = 10000
        self.min_dist_array = []

    def _create_sim(self):
        """Creates the simulator"""
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        self.sim_dt = 1.0 / 60.0

        sim_params.substeps = 6  # 6
        sim_params.gravity = gymapi.Vec3(0, -9.8, 0)
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.num_position_iterations = 8  # 8
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.bounce_threshold_velocity = 7.0
        # sim_params.use_gpu_pipeline = True

        self._sim = self._gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )

    def _create_env(self):
        spacing = 1.0
        self._gym.add_ground(self._sim, gymapi.PlaneParams())

        # Creating Env
        self._env = self._gym.create_env(
            self._sim,
            gymapi.Vec3(-spacing, 0.0, -spacing),
            gymapi.Vec3(spacing, spacing, spacing),
            1,
        )

        # Retrieve collision-free start-goal configurations for this scene/experiment
        # 00 for experiment 0, 01 for experiment 1, etc
        # For dynamic scenes 00 for experiment 10, 01 for experiment 11, etc
        scene_str = f'0{self.args.experiment}' if self.args.experiment < 10 else f'0{self.args.experiment-10}'
        self.goal_qs, self.goal_pos = get_valid_configs(
            scenario=scene_str,
            num_configs=self.args.test_config_pairs_num,
            min_distance=self.args.test_config_min_dist,
        )

        # Creating Robot
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0, 0.0, 0)
        robot_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
        self._robot_actor = self._gym.create_actor(
            self._env,
            self._load_robot_asset(),
            robot_pose,
            "robot",
            0,
            0,
            ROBOT_LABEL,  # Segmentation id
        )
        # Set robot dof properties and initial state
        robot_dof_props = self._gym.get_actor_dof_properties(
            self._env, self._robot_actor)
        self._set_robot_dof_props(robot_dof_props)
        robot_dof_states = self._gym.get_actor_dof_states(
            self._env, self._robot_actor, gymapi.STATE_NONE)
        self._gym.set_actor_dof_states(
            self._env, self._robot_actor, robot_dof_states, gymapi.STATE_POS)
        self._set_robot_targets(ROBOT_Q_INIT)

        # Create Object Actors
        self._objects = self._arrange_objects()

        # Adding cameras.
        hand_handle = self._gym.find_actor_rigid_body_handle(
            self._env, self._robot_actor, self.args.ee_link_name
        )
        external_transform = tra.euler_matrix(0, 0, np.pi / 9).dot(
            tra.euler_matrix(0, np.pi / 2, 0)
        )
        external_transform[0, 3] = 2
        external_transform[1, 3] = 1
        external_transform[2, 3] = 0

        self._cameras = self._create_cameras(
            robot_body_handle=hand_handle,
            fov=60.0,
            width=self.args.camera_width,
            height=self.args.camera_height,
            external_camera_transform=external_transform,
        )

    def _arrange_objects(self):
        actor_handles = {}

        # Add table
        table_dims = gymapi.Vec3(1.0, 0.2, 1.6)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(
            0.5 * table_dims.x + 0.1,  # Move table away from Robot.
            0.5 * table_dims.y + 0.001,
            0.0,
        )
        table_asset = self._gym.create_box(
            self._sim,
            table_dims.x,
            table_dims.y,
            table_dims.z,
            self._asset_options(static=True),
        )
        actor_handles["table"] = self._gym.create_actor(
            self._env,
            table_asset,
            table_pose,
            "table",
            0,
            0,
            TABLE_LABEL,
        )

        # Arrange scene using scene manager
        self.scene_manager = SceneManager(self.args.dataset_root)
        self.scene_manager.table_bounds = np.array(
            [
                [
                    table_pose.p.x - 0.4 * table_dims.x,
                    table_pose.p.z - 0.2 * table_dims.z,
                    table_pose.p.y + 0.5 * table_dims.y,
                ],
                [
                    table_pose.p.x + 0.1 * table_dims.x,
                    table_pose.p.z + 0.2 * table_dims.z,
                    table_pose.p.y + 0.5 * table_dims.y,
                ],
            ]
        )
        self.scene_manager.arrange_scene(
            self.args.num_objects, self.args.experiment, max_attempts=10)
        if len(self.scene_manager.objs) == 0:
            raise ValueError("Environment does not have any objects!")

        for i, (obj_name, obj_info) in enumerate(
            self.scene_manager.objs.items()
        ):
            if obj_name == "table":
                continue
            else:
                # Write urdf for mesh
                urdf_path = utils.write_urdf(
                    obj_name,
                    os.path.abspath(obj_info["mesh"].metadata["path"]),
                    self.args.urdf_cache,
                )

                # Set vhacd params
                asset_options = self._asset_options(static=False)
                asset_options.density = 300
                asset_options.linear_damping = 20
                asset_options.angular_damping = 20
                asset_options.vhacd_enabled = True
                asset_options.vhacd_params = gymapi.VhacdParams()
                asset_options.vhacd_params.resolution = 1000000
                asset_options.vhacd_params.concavity = 0.00001
                asset_options.vhacd_params.convex_hull_downsampling = 16
                asset_options.vhacd_params.plane_downsampling = 16
                asset_options.vhacd_params.alpha = 0.15
                asset_options.vhacd_params.beta = 0.15
                asset_options.vhacd_params.mode = 0
                asset_options.vhacd_params.pca = 0
                asset_options.vhacd_params.max_num_vertices_per_ch = 128
                asset_options.vhacd_params.min_volume_per_ch = 0.0001

                asset = self._gym.load_asset(
                    self._sim,
                    "",
                    urdf_path,
                    asset_options,
                )

            obj_pose = tra.euler_matrix(-np.pi / 2, 0, 0) @ obj_info["pose"]
            asset_transform = gymapi.Transform()
            asset_transform.p = gymapi.Vec3(
                obj_pose[0, 3],
                obj_pose[1, 3],
                obj_pose[2, 3],
            )

            asset_transform.r = utils.from_rpy(
                *tra.euler_from_matrix(obj_pose)
            )
            actor_handles[obj_name] = self._gym.create_actor(
                self._env,
                asset,
                asset_transform,
                obj_name,
                0,
                0,
                i + 1,
            )

            self._gym.set_rigid_body_color(
                self._env,
                actor_handles[obj_name],
                0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                # gymapi.Vec3(*obj_info["mesh"].visual.face_colors[0, :3]),
                # Make all added objs color grey
                gymapi.Vec3(0.35, 0.35, 0.35),
            )

        for _, actor_handle in actor_handles.items():
            props = self._gym.get_actor_rigid_shape_properties(
                self._env,
                actor_handle,
            )
            props[0].restitution = 0

            # increase the friction to prevent object slippage.
            props[0].rolling_friction = 0.2
            props[0].torsion_friction = 0.2
            self._gym.set_actor_rigid_shape_properties(
                self._env, actor_handle, props
            )

        # Create here the start and goal sphere actors for visualization purposes
        # We use a different collision group for these so they don't interact with the rest of the scene
        if self.args.markers:
            ball_radius = 0.03
            asset_options = self._asset_options()
            asset_options.disable_gravity = True
            # Create start ball actor in a separate collision group (1) with segmentation mask START_SPHERE_LABEL
            start_ball = self._gym.create_sphere(
                self._sim, ball_radius, asset_options)
            self.start_ball_handle = self._gym.create_actor(
                self._env, start_ball, gymapi.Transform(), "start_ball", group = 1, filter = 0, segmentationId = START_SPHERE_LABEL)
            actor_handles["start_marker"] = self.start_ball_handle
            self._gym.set_rigid_body_color(
                self._env, self.start_ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 1.0, 0.0))
            # Create start ball actor in a separate collision group (2) with segmentation mask GOAL_SPHERE_LABEL
            goal_ball = self._gym.create_sphere(
                self._sim, ball_radius, asset_options)
            self.goal_ball_handle = self._gym.create_actor(
                self._env, goal_ball, gymapi.Transform(), "goal_ball", group = 2, filter = 0, segmentationId = GOAL_SPHERE_LABEL)
            actor_handles["goal_marker"] = self.goal_ball_handle
            self._gym.set_rigid_body_color(
                self._env, self.goal_ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0))

        # If experiment 1, add dynamic obstacle
        if self.args.experiment >= 10 and self.args.experiment <= 19:
            # create ball
            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = True
            ball_asset = self._gym.create_sphere(
                self._sim, 0.035, asset_options)
            pose = gymapi.Transform()
            pose.r = gymapi.Quat(0, 0, 0, 1)
            pose.p = gymapi.Vec3(0.60, 0.55, 0.0)
            # create ball actor in same collision group as other obstacles with segmentation mask 100
            ball_handle = self._gym.create_actor(
                self._env, ball_asset, pose, "dynamic_obstacle", group = 0, filter = 0, segmentationId = 100)
            actor_handles["dynamic_obstacle"] = ball_handle
            self._gym.set_rigid_body_color(
                self._env, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 1.0))
            self.dyn_obstacle_velocity = None

        return actor_handles

    def _show_feasible_configs(self):
        ### UNCOMMENT BELOW BLOCK TO SEE ALL THE POSITIONS WE TEST ###
        # set_robot_pos = []
        # for i in range(12):
        #     for j in range(10):
        #         set_robot_pos.append([0.25+i*0.05, 0.45, j * 0.05])
        #         set_robot_pos.append([0.25+i*0.05, 0.45, -1 * j * 0.05])

        all_valid_pos = get_all_valid_positions(self.args.experiment)
        for pos in all_valid_pos:
            this_pose = gymapi.Transform()
            this_pose.p = gymapi.Vec3(pos[0], 0.45, -pos[1])
            self._show_sphere(radius=0.01, pose=this_pose, color=(0, 1, 0))

    def _show_frame(self, pose: gymapi.Vec3, scale=1.0):
        axes_geom = gymutil.AxesGeometry(scale)
        gymutil.draw_lines(axes_geom, self._gym, self.viewer, self._env, pose)

    def _show_sphere(self, radius, pose: gymapi.Vec3, color):
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        sphere_geom = gymutil.WireframeSphereGeometry(
            radius, 16, 16, sphere_pose, color)
        gymutil.draw_lines(sphere_geom, self._gym,
                           self.viewer, self._env, pose)

    def _clear_drawn_lines(self):
        self._gym.clear_lines(self.viewer)

    def _asset_options(self, static=True):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = static
        asset_options.flip_visual_attachments = False
        asset_options.thickness = 0.001
        return asset_options

    def _load_robot_asset(self):
        options = self._asset_options()
        options.flip_visual_attachments = True
        options.disable_gravity = True
        options.armature = 0.01

        return self._gym.load_asset(
            self._sim,
            self.args.robot_asset_root,
            self.args.robot_asset_file,
            options,
        )

    def _set_robot_joints_pos(self, q):
        robot_dof_states = self._gym.get_actor_dof_states(
            self._env, self._robot_actor, gymapi.STATE_NONE)
        for i, dof_state in enumerate(robot_dof_states):
            dof_state["pos"] = q[i]
        self._gym.set_actor_dof_states(
            self._env, self._robot_actor, robot_dof_states, gymapi.STATE_POS
        )

    def _set_robot_targets(self, q):
        target = np.zeros(9, dtype=[("pos", ">f4"), ("vel", ">f4")])

        # the arm
        # target["pos"][:-2] = q[:-2]

        # the fingers (positive if opening, negative if closing)
        # target["vel"][-2:] = -0.1 if q[-1] < 0.02 else 0.1
        if len(q) == 9:
            target["pos"][:-2] = q[:-2]
            target["vel"][-2:] = 0.0
            self._gym.set_actor_dof_position_targets(
                self._env,
                self._robot_actor,
                target["pos"],
            )
        else:
            target['vel'][:7] = q

        self._gym.set_actor_dof_velocity_targets(
            self._env,
            self._robot_actor,
            target["vel"],
        )

    def _set_robot_dof_props(self, robot_dof_props):
        stiffness = [0]*9
        damping = [5]*9
        effort = [87, 87, 87, 87, 12, 12, 12, 1400, 1400]

        for i in range(9):
            robot_dof_props["stiffness"][i] = stiffness[i]
            robot_dof_props["damping"][i] = damping[i]
            robot_dof_props["effort"][i] = effort[i]
        robot_dof_props["driveMode"][:] = gymapi.DOF_MODE_VEL
        self._gym.set_actor_dof_properties(
            self._env, self._robot_actor, robot_dof_props)

    def _create_cameras(
        self,
        robot_body_handle,
        fov,
        width,
        height,
        external_camera_transform,
    ):

        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = fov
        camera_props.height = height
        camera_props.width = width
        camera_props.use_collision_geometry = False
        wrist_camera_handle = self._gym.create_camera_sensor(
            self._env, camera_props
        )
        self.num_cameras = 1
        if external_camera_transform is not None:
            self.num_cameras = 2
            external_camera_handle = self._gym.create_camera_sensor(
                self._env, camera_props
            )
            external_q = tra.quaternion_from_matrix(external_camera_transform)
            external_q = np.roll(external_q, -1)
            external_t = external_camera_transform[:3, 3]
            self._gym.set_camera_transform(
                external_camera_handle,
                self._env,
                gymapi.Transform(
                    gymapi.Vec3(*external_t), gymapi.Quat(*external_q)
                ),
            )
        else:
            external_camera_handle = None

        offset_q = tra.quaternion_from_matrix(
            tra.euler_matrix(0, 0, np.pi / 2) @ tra.euler_matrix(np.pi, 0, 0)
        )
        offset_q = np.roll(offset_q, -1)
        offset_t = np.array([0.06, 0.0, 0.01])

        self._gym.attach_camera_to_body(
            wrist_camera_handle,
            self._env,
            robot_body_handle,
            gymapi.Transform(gymapi.Vec3(*offset_t), gymapi.Quat(*offset_q)),
            gymapi.FOLLOW_TRANSFORM,
        )
        cameras = [wrist_camera_handle]
        if external_camera_handle is not None:
            cameras.append(external_camera_handle)
        for c in cameras:
            if c == -1:
                raise ValueError("camera is not instantiated properly")
        return cameras

    # Get keyboard events from gym
    def step_loop_event(self):
        if self.viewer is None:
            return

        for evt in self._gym.query_viewer_action_events(self.viewer):
            if evt.action == "pause" and evt.value > 0:
                self.playing = not self.playing
                return "pause"
            elif evt.action == "next_target" and evt.value > 0:
                return "next_target"

    def step(self, action=None):
        # Set robot plan
        done = False
        if action is None or self.iteration_has_timed_out:
            done = self._next_target()
        self._set_plan(action)

        # Step simulator
        (
            t,
            env_states,
            robot_contacts,
            self_collisions,
            keyboard_events,
        ) = self.main_loop_step(self.args.control_frequency)

        # self.logger.info(f"This iteration duration in sim time = {t-self.iteration_start_time}")

        # Check if we need to timeout this iteration
        if (t-self.iteration_start_time) > self.iteration_timeout:
            print(f" ######### This iteration has timed out! ######### ")
            self.iteration_has_timed_out = True

        info = {
            "time": t,
            "robot_contacts": robot_contacts,
            "self_collisions": self_collisions,
            "all_gym_states": env_states,
            "done": done,
        }

        # Update scene manager
        # if action is not None: #We are trying to reset the scene
        for ok, op in env_states[-1].items():
            if ok == "robot" or ok == "dynamic_obstacle" or ok == "start_ball" or ok == "goal_ball":
                continue
            self.scene_manager.set_object_pose(
                ok, tra.euler_matrix(np.pi / 2, 0, 0) @ op
            )

        # Create observation

        obs = {
            "robot_q": np.array(list(env_states[-1]["robot"].values())),
            "new_target": self.new_target_available
        }
        if not done:  # Otherwise index will be out of range
            obs.update(
                {
                    "goal_q": self.goal_qs[self.target_idx],
                }
            )
        obs.update(self._build_pc_observation())
        # Update observation with keyboard events
        obs["keyboard_next_target_event"] = int(
            keyboard_events is not None and "next_target" in keyboard_events
        )
        self.new_target_available = False
        return obs, info

  # Function for choosing new target
    def _next_target(self) -> bool:
        """
        This function returns True if we are done going through all targets.
        If not, we advance to the next target on the list and return False.
        """
        self.logger.info("=============== NEXT TARGET ===============")
        self.target_idx += 1
        self.new_target_available = True

        # Handle metrics variables
        sim_time_now = self._gym.get_sim_time(self._sim)
        self.total_iterations += 1
        # Record self.min_dist
        if self.min_dist != 10000:
            self.min_dist_array.append(self.min_dist)
        self.min_dist = 10000

        if self.target_idx != 1:
            # Was the iteration successful? How long it took?
            if not self.iteration_has_timed_out:
                self.succesful_iterations += 1
                self.iteration_durations.append(sim_time_now - self.iteration_start_time)
            # Was the iteration collision free?
            if self.iteration_is_collision_free:
                self.num_collision_free_iterations += 1
        # Reset relevant metric variables for next iteration
        self.iteration_is_collision_free = True
        self.iteration_has_timed_out = False
        self.iteration_start_time = sim_time_now

        if self.target_idx != 1:
            print("==================================================================")
            print(f"\tTotal iterations\t = \t\t {self.total_iterations}")

            print(
                f"\tSuccesful iterations\t = \t\t {self.succesful_iterations}")
            success_rate = self.succesful_iterations * 100.0 / self.total_iterations
            print(f"\tSuccess rate [%]\t = \t\t {success_rate}")

            print(
                f"\tNumber of collision free iterations\t = \t\t {self.num_collision_free_iterations}")
            if self.succesful_iterations > 0:
                collision_free_rate = self.num_collision_free_iterations * \
                    100.0 / self.total_iterations
                print(
                    f"\tCollision free iterations [%]\t = \t\t {collision_free_rate}")
            else:
                print(f"\tCollision free iterations [%]\t = \t\t Undetermined")

            time_avg = np.average(np.array(self.iteration_durations))
            time_std_dev = np.std(np.array(self.iteration_durations))
            print(
                f"\tAverage time to goal [t]\t = \t\t {time_avg} +- {time_std_dev}")
            safety_avg = np.average(np.array(self.min_dist_array))
            safety_std_dev = np.std(np.array(self.min_dist_array))
            print(
                f"\tAverage safety distance [m]\t = \t\t {safety_avg} +- {safety_std_dev}")
            self.min_dist_array
            print("==================================================================")

        if self.goal_qs is not None and len(self.goal_qs) == self.target_idx:
            self.logger.info(
                "=== We are done going through all start-goal configs ===")
            return True
        
        if self.target_idx == 1:
            start_pos = self.goal_pos[self.target_idx]
            goal_pos = self.goal_pos[self.target_idx]
        else:
            start_pos = self.goal_pos[self.target_idx-1]
            goal_pos = self.goal_pos[self.target_idx]
        if self.args.markers:
            self._set_object_state(self.start_ball_handle, zero_out_velocity=True, pos=gymapi.Vec3(
                start_pos[0], start_pos[2], -start_pos[1]))
            self._set_object_state(self.goal_ball_handle, zero_out_velocity=True, pos=gymapi.Vec3(
                goal_pos[0], goal_pos[2], -goal_pos[1]))

        return False

    def reset_scene(self, robot_q=None):
        self.scene_manager.reset_objs_to_initial_poses()
        for _, (obj_name, obj_info) in enumerate(self.scene_manager.objs.items()):
            if obj_name == "table":
                continue
            else:
                obj_pose = tra.euler_matrix(-np.pi / 2,
                                            0, 0) @ obj_info["pose"]
                self._set_object_state(
                    self._objects[obj_name],
                    zero_out_velocity=True,
                    pos=gymapi.Vec3(
                        obj_pose[0, 3], obj_pose[1, 3], obj_pose[2, 3]),
                    quat=utils.from_rpy(*tra.euler_from_matrix(obj_pose)),
                )
        if robot_q is not None:
            robot_dof_states = self._gym.get_actor_dof_states(
                self._env, self._robot_actor, gymapi.STATE_NONE)
            for i, dof_state in enumerate(robot_dof_states):
                dof_state["pos"] = robot_q[i]
            self._gym.set_actor_dof_states(
                self._env, self._robot_actor, robot_dof_states, gymapi.STATE_POS
            )
            self._set_robot_targets(robot_q)

    def main_loop_step(self, camera_render_freq=1000):
        robot_obj_contacts = {}
        robot_self_collision = False
        events = set()
        env_states = []
        while True:
            output = self._main_loop_step(camera_render_freq)
            if output is not None:
                # aggregating contact info and self collision info.
                (
                    t,
                    env_state,
                    obj_contacts,
                    self_collision,
                    new_event,
                ) = output
                env_states.append(env_state)
                events.add(new_event)
                for k, force_magnitude in obj_contacts.items():
                    if k not in robot_obj_contacts:
                        robot_obj_contacts[k] = force_magnitude
                    else:
                        robot_obj_contacts[k] = max(
                            robot_obj_contacts[k], force_magnitude
                        )
                robot_self_collision |= self_collision

            # We have simulated enough steps.
            if (self._has_observation()):
                break

        return (
            t,
            env_states,
            robot_obj_contacts,
            robot_self_collision,
            events,
        )

    def _main_loop_step(self, camera_render_freq):
        # Step the physics
        t = self._gym.get_sim_time(self._sim)
        self._gym.simulate(self._sim)
        self._gym.fetch_results(self._sim, True)
        event = self.step_loop_event()

        # Below are some hacks to circumvent physics simulation
        # problems.
        # Allow time for the objects to settle within 1 second.
        if t < 1.0:
            pass
        elif t > 1 and t < 2:  # Freeze all the objects.
            for _, actor in self._objects.items():
                self._set_object_state(
                    actor,
                    zero_out_velocity=True,
                )
        elif t < 2.5:
            pass
        else:
            (
                robot_obj_contacts,
                robot_self_collision,
            ) = self._get_robot_contacts()

            # Check if any object still moves after freezing process.
            # and throw them out of the table if they move. Problem solved!!!
            if t < 3:
                object_handles = self._get_unsettled_objects()
                for handle in object_handles:
                    self.logger.warn(
                        "Throwing out object {} from env".format(handle)
                    )
                    self._set_object_state(
                        handle,
                        zero_out_velocity=True,
                        pos=gymapi.Vec3(np.random.uniform(
                            2, 5), 0.5, np.random.uniform(2, 5)),
                    )
            # If experiment = 1X, control speed of the ball to bounce around
            if self.args.experiment >= 10 and self.args.experiment <= 19:
                ball_handle = self._objects["dynamic_obstacle"]
                state = self._gym.get_actor_rigid_body_states(
                    self._env, ball_handle, gymapi.STATE_ALL)
                position = state["pose"]["p"][0]
                if self.dyn_obstacle_velocity is None:
                    self.dyn_obstacle_velocity = [0.0, 0.0, 0.05]
                if position[2] >= 0.5 or position[2] <= -0.5:
                    self.dyn_obstacle_velocity[2] *= -1
                self._set_object_state(
                    ball_handle,
                    zero_out_velocity=False,
                    velocity=self.dyn_obstacle_velocity,
                )

        if (
            self.viewer is not None
            or t - self.last_refresh > 1.0 / camera_render_freq
        ):
            self._gym.step_graphics(self._sim)

        if t >= 3 and t - self.last_refresh > 2.5 / camera_render_freq:  # 1.0
            self.last_refresh = t
            self._gym.render_all_camera_sensors(self._sim)
            self._observe_all_cameras()

        # Step rendering
        if self.viewer is not None:
            self._gym.draw_viewer(self.viewer, self._sim, False)
        self._gym.sync_frame_time(self._sim)

        env_state = self._get_gym_state()
        if t > 3 and self.playing:
            # print("call get_next_q_in_plan")
            next_q = self._get_next_q_in_plan(
                np.array(list(env_state["robot"].values()))
            )
            if next_q is not None:
                self._set_robot_targets(next_q)
        if t < 3:
            return None

        return (
            t,
            env_state,
            robot_obj_contacts,
            robot_self_collision,
            event,
        )

    def _get_gym_state(self):
        gym_state = {}

        for i in range(self._gym.get_actor_count(self._env)):
            actor_name = self._gym.get_actor_name(self._env, i)
            dof_dict = self._gym.get_actor_rigid_body_dict(self._env, i)
            body_states = self._gym.get_actor_rigid_body_states(
                self._env,
                i,
                gymapi.STATE_ALL,
            )
            if actor_name == "robot":
                robot_dof_names = self._gym.get_actor_dof_names(self._env, i)
                dof_states = self._gym.get_actor_dof_states(
                    self._env, i, gymapi.STATE_POS
                )["pos"]
                assert len(dof_states) == len(robot_dof_names)
                links_pose = {
                    name: state
                    for name, state in zip(robot_dof_names, dof_states)
                }
            else:
                links_pose = {
                    k: utils.gym_pose_to_matrix(body_states["pose"][:][v])
                    for k, v in dof_dict.items()
                }
            if len(links_pose) == 1:
                gym_state[actor_name] = list(links_pose.values())[0]
            else:
                gym_state[actor_name] = links_pose

        return gym_state

    def _get_robot_contacts(self):
        # TODO: calculate dicts during init?
        link_names = list(
            self._gym.get_actor_rigid_body_dict(
                self._env, self._robot_actor
            ).keys()
        )
        robot_link_dict = {
            self._gym.find_actor_rigid_body_index(
                self._env, self._robot_actor, link_name, gymapi.DOMAIN_ENV
            ): link_name
            for link_name in link_names
        }

        obj_dict = {
            self._gym.find_actor_rigid_body_index(
                self._env,
                obj_actor,
                list(
                    self._gym.get_actor_rigid_body_dict(
                        self._env, obj_actor
                    ).keys()
                )[0],
                gymapi.DOMAIN_ENV,
            ): obj_name
            for obj_name, obj_actor in self._objects.items()
        }

        contacts = self._gym.get_env_rigid_contacts(self._env)
        obj_contacts_dict = {}
        self_collision = False
        for contact in contacts:
            if (
                contact["body0"] not in robot_link_dict
                and contact["body1"] not in robot_link_dict
            ):
                continue

            def get_name(index):
                if index in obj_dict:
                    return obj_dict[index]
                elif index in robot_link_dict:
                    return robot_link_dict[index]
                else:
                    if index == -1:
                        return None
                    raise ValueError(
                        "index {} can't be found in either robot dict {} or obj dict {}".format(
                            index, robot_link_dict, obj_dict
                        )
                    )

            names = [get_name(contact[name]) for name in ["body0", "body1"]]
            if names[0] is None or names[1] is None:
                name = "ground"
            elif names[0] in self._objects:
                name = names[0]
            elif names[1] in self._objects:
                name = names[1]
            else:
                # Make sure self collision is not because of fingers touching.
                if (
                    names[0].find("finger") >= 0
                    and names[1].find("finger") >= 0
                ):
                    continue
                self_collision = True
                continue

            if name not in obj_contacts_dict:
                obj_contacts_dict[name] = -1e9
            obj_contacts_dict[name] = max(
                obj_contacts_dict[name], contact["lambda"]
            )

        new_color = np.asarray([0, 0, 0])
        # if self_collision:
        #     new_color[2] = 1
        if len(obj_contacts_dict) > 0:
            new_color[0] = 1
            self.iteration_is_collision_free = False
        if np.all(new_color == 0):
            new_color = [1, 1, 1]

        self._set_object_color("table", new_color)

        return obj_contacts_dict, self_collision

    def _get_unsettled_objects(self):
        unsettled_objets = []
        for actor_name, actor in self._objects.items():

            body_state = self._gym.get_actor_rigid_body_states(
                self._env, actor, gymapi.STATE_ALL
            )

            linear_vel = [body_state["vel"]["linear"][0][i] for i in range(3)]
            if np.max(np.abs(linear_vel)) >= 0.2:
                if actor_name != 'dynamic_obstacle':
                    unsettled_objets.append(actor)

        return unsettled_objets

    def _set_object_state(
        self,
        actor,
        zero_out_velocity,
        pos=None,
        quat=None,
        velocity=None,
    ):
        body_state = self._gym.get_actor_rigid_body_states(
            self._env, actor, gymapi.STATE_ALL
        )
        # print(body_state)
        if zero_out_velocity:
            if velocity is not None:
                raise ValueError(
                    "if zero_out_velocity is False, velocity needs to be None"
                )
            body_state["vel"].fill(0)
        else:
            body_state["vel"]["linear"] = (
                velocity[0],
                velocity[1],
                velocity[2],
            )
        if pos is not None:
            body_state["pose"]["p"] = (pos.x, pos.y, pos.z)
        if quat is not None:
            body_state["pose"]["r"] = (quat.x, quat.y, quat.z, quat.w)
        self._gym.set_actor_rigid_body_states(
            self._env,
            actor,
            body_state,
            gymapi.STATE_ALL,
        )

    def _set_object_color(self, obj_name, color):
        if obj_name == "robot":
            actor_handle = self._robot_actor
        else:
            actor_handle = self._objects[obj_name]

        num_bodies = self._gym.get_actor_rigid_body_count(
            self._env, actor_handle
        )
        for n in range(num_bodies):
            self._gym.set_rigid_body_color(
                self._env,
                actor_handle,
                n,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*color),
            )

    def _has_observation(self):
        return (
            abs(
                self._gym.get_sim_time(self._sim)
                - self.last_refresh
                - self.sim_dt
            )
            < 1e-5
        )

    def _observe_all_cameras(self):
        self.current_observations = []
        for camera_handle in self._cameras:
            camera_pose = self._gym.get_camera_transform(
                self._sim, self._env, camera_handle
            )

            proj_matrix = self._gym.get_camera_proj_matrix(
                self._sim, self._env, camera_handle
            )

            q = camera_pose.r
            p = camera_pose.p
            camera_pose = utils.gym_pose_to_matrix(
                {"r": [q.x, q.y, q.z, q.w], "p": [p.x, p.y, p.z]}
            )
            camera_pose = camera_pose.dot(tra.euler_matrix(np.pi, 0, 0))

            color_image = self._gym.get_camera_image(
                self._sim,
                self._env,
                camera_handle,
                gymapi.IMAGE_COLOR,
            )
            color_image = np.reshape(color_image, [480, 640, 4])[:, :, :3]

            depth_image = -self._gym.get_camera_image(
                self._sim,
                self._env,
                camera_handle,
                gymapi.IMAGE_DEPTH,
            )
            depth_image[depth_image == np.inf] = 0
            depth_image[depth_image > DEPTH_CLIP_RANGE] = 0
            segmentation = self._gym.get_camera_image(
                self._sim,
                self._env,
                camera_handle,
                gymapi.IMAGE_SEGMENTATION,
            )

            self.current_observations.append(
                CameraObservation(
                    camera_pose,
                    proj_matrix,
                    color_image,
                    depth_image,
                    segmentation,
                )
            )

    def _build_pc_observation(self):
        """
        Puts point clouds from camera and robot link positions into one point
        cloud.
        Returns:
          env_xyzs: (num_env, npoints, 3), Joint pc for point
            cloud of all the cameras in each env.
          env_labels: (num_env, num_points) label for each point of the joint point cloud of each env.
          depth_images: if return_depth_images is True, return depth images for each env
          otherwise returns None.
        """
        fxs = []
        fys = []
        camera_poses = []
        label_images = []
        depth_images = []

        for obs in self.current_observations:
            depth_images.append(obs.depth.copy())
            camera_poses.append(obs.cam_pose.copy())
            fxs.append(obs.proj_matrix[0, 0])
            fys.append(obs.proj_matrix[1, 1])
            label_images.append(obs.segmentation.copy().flatten())

        label_images = np.asarray(label_images, dtype=np.uint32)
        output_camera_poses = np.asarray(camera_poses, dtype=np.float32)
        depth_images = np.asarray(depth_images, dtype=np.float32)
        num_cameras, height, width = depth_images.shape

        if not hasattr(self, "_input_x"):
            fxs = 2.0 / np.asarray(fxs).reshape(-1, 1, 1)
            fys = 2.0 / np.asarray(fys).reshape(-1, 1, 1)
            self._input_x = (np.arange(width) - (width / 2)) / width
            self._input_y = (np.arange(height) - (height / 2)) / height
            self._input_x, self._input_y = np.meshgrid(
                self._input_x, self._input_y
            )
            self._input_x = fxs * np.repeat(
                self._input_x[None, ...], num_cameras, axis=0
            )
            self._input_y = fys * np.repeat(
                self._input_y[None, ...], num_cameras, axis=0
            )

        output_x = depth_images * self._input_x
        output_y = depth_images * self._input_y

        cam_xyzs = np.stack(
            (output_x, output_y, depth_images), axis=-1
        ).reshape([-1, height * width, 3])
        cam_valid_depth = cam_xyzs[:, :, 2] > 0.001

        pcs = []
        labels = []
        for cam_pc, cam_label, cam_valid in zip(
            cam_xyzs, label_images, cam_valid_depth
        ):
            valid_index = np.where(cam_valid)[0]
            if np.any(valid_index):
                mask = np.random.choice(
                    valid_index,
                    size=self.args.npoints,
                    replace=len(valid_index) < self.args.npoints,
                )
                pcs.append(cam_pc[mask, :])
                label = cam_label[mask].copy()
                # label[label == self._objects[self._target_name]] = 1   #There are no more target objs
                # label[np.logical_and(label > 1, label < ROBOT_LABEL)] = 2
                labels.append(label)
            else:
                pcs.append(np.zeros((self.args.npoints, cam_pc.shape[-1])))
                labels.append(np.zeros(self.args.npoints))

        return {
            "pc": np.asarray(pcs).reshape(num_cameras, -1, 3),
            "pc_label": np.asarray(labels).reshape(num_cameras, -1),
            "depth_image": depth_images,
            "camera_pose": output_camera_poses,
        }

    def _set_plan(self, plan):
        self.current_plan = (
            [copy.deepcopy(q) for q in plan] if plan is not None else None
        )
        self.current_plan_index = 0
        self.num_steps_for_q = 0

    # max_steps_per_q=40 original value from SCN
    def _get_next_q_in_plan(self, cur_q, max_steps_per_q=40):
        self.done_plan = False
        if self.current_plan is None:
            return None
        if self.current_plan_index >= len(self.current_plan):
            return None
        self.num_steps_for_q += 1

        while True:
            if isinstance(self.current_plan[self.current_plan_index], tuple):
                current_action = self.current_plan[self.current_plan_index][0]
                super_accurate = self.current_plan[self.current_plan_index][1]
            else:
                current_action = self.current_plan[self.current_plan_index]
                super_accurate = (
                    self.current_plan_index == len(self.current_plan) - 1
                )

            delta_q = cur_q[:7] - current_action[:7]
            delta_q = np.max(np.abs(delta_q))

            if super_accurate:
                move_forward = delta_q < 0.001
                if move_forward:
                    self.done_plan = True
            else:
                move_forward = delta_q < 0.01

            if not move_forward and not super_accurate:
                move_forward = self.num_steps_for_q > max_steps_per_q

            if move_forward:
                self.current_plan_index += 1
                self.num_steps_for_q = 0
                if self.current_plan_index == len(self.current_plan):
                    self.current_plan_index -= 1
                    break
            else:
                break

        return current_action

    def __del__(self):
        if self.viewer is not None:
            self._gym.destroy_viewer(self.viewer)
        self._gym.destroy_sim(self._sim)

    def get_min_dist(self, scene_pcd, robot_pcd):
        self.csdf.update_pcd(scene_pcd)

        # Filter robot outside the table
        robot_pcd = robot_pcd[robot_pcd[:, 0] > 0.1, :]

        min_distance = self.csdf.forward(torch.from_numpy(robot_pcd).float().to('cuda').unsqueeze(0))
        # print(f"=========== MIN DISTANCE: {min_distance} ==================== ")

        min_distance = min_distance[0].cpu().numpy()
        if min_distance < self.min_dist:
            self.min_dist = min_distance
