import logging
import numpy as np
from autolab_core import Logger
from mppi_planning.trajectory_planning import TrajectoryPlanner


np.set_printoptions(suppress=True)


class InfeasibleTargetException(Exception):
    pass


class RAMPPolicy:
    def __init__(
        self,
        transition_threshold=0.05,
        cam_type="ws",
        device=0,
        log_file=None,
    ):
        """
        Args:
          transition_threshold: float, threshold for moving between states
          cam_type: "ws" or "hand" determines which camera to use
          device: int, compute device for rollouts
          safe: bool, pause before grasping and placing and wait for confirmation
          log_file: str, path to a logging file
        """
        self.transition_threshold = transition_threshold
        self.device = device

        if cam_type not in ["ws", "hand"]:
            raise ValueError("Invalid cam_type (ws or hand)")
        self.cam_type = 0 if cam_type == "hand" else 1

        self.logger = Logger.get_logger(
            "MPPIPolicy", log_file=log_file, log_level=logging.DEBUG
        )
        self.q_goal = None
        self.prev_rollout_lens = []

        # RAMP
        self.trajectory_planner = None
        self.mode_computing_grasp = True
        self.bool_new_target = True

    def get_action(self, obs, pcd):
        if (
            "keyboard_next_target_event" in obs
            and obs["keyboard_next_target_event"]
        ):
            self.logger.debug("Keyboard event detected: Next target")
            return None

        self.robot_q = obs["robot_q"].astype(np.float64).copy()
        self.q_goal = obs["goal_q"]

        # Check to see if we are within threshold of our goal; if so, reset robot a new q_init
        distance_to_target = np.linalg.norm(
            self.robot_q[:7] - self.q_goal[:7], ord=np.inf, axis=-1).min()
        self.logger.debug(f"Joint Dist to Target: {distance_to_target}")
        if distance_to_target < self.transition_threshold:
            return None

        # RAMP Rollouts
        """
        The high-level interface to the panda robot.
        """
        if self.trajectory_planner is None or obs["new_target"]:
            self.logger.info("Instantiating New trajectory planner!")
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

            N_JOINTS = 7

            # MPPI parameters
            mppi_control_limits = [
                -0.05 * np.ones(N_JOINTS),
                0.05 * np.ones(N_JOINTS)
            ]
            mppi_nsamples = 500
            mppi_covariance = 0.005
            mppi_lambda = 1.0


            self.trajectory_planner = TrajectoryPlanner(
                joint_limits=JOINT_LIMITS,
                # robot_urdf_location=robot_urdf_location,
                # scene_urdf_location=scene_urdf_location,
                link_fixed=LINK_FIXED,
                link_ee=LINK_EE,
                link_skeleton=LINK_SKELETON,
            )
            # sol = planner.plan_ja_to_ja(pose1, pose3)
            # print("planning time : ", time.time()-start_time)

            _goal = self.q_goal[:7]
            _start = self.robot_q[:7]

            current_joint_angles = _start
            ja = _goal
            # Instantiate MPPI object
            self.trajectory_planner.instantiate_mppi_ja_to_ja(
                current_joint_angles,
                ja,
                mppi_control_limits=mppi_control_limits,
                mppi_nsamples=mppi_nsamples,
                mppi_covariance=mppi_covariance,
                mppi_lambda=mppi_lambda,
            )

        # pcd = obs['pc'][obs['pc_label'] < 5,:]
        # self.scene_collision_checker.scene_pc

        # pcd = self.scene_collision_checker.scene_pc[self.scene_collision_checker.scene_pc_mask, :]

        # o3d_pcd = o3d.geometry.PointCloud()
        # o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
        # o3d_axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([o3d_pcd, o3d_axis])

        self.trajectory_planner.update_obstacle_pcd(pcd=pcd)

        _goal = self.q_goal[:7]
        _start = self.robot_q[:7]
        current_joint_angles = _start
        ja = _goal
        traj_a = self.trajectory_planner.get_mppi_rollout(current_joint_angles)

        traj_action = np.concatenate(
            (traj_a, np.ones((np.size(traj_a, 0), 2))*0.04), axis=1)

        # if (self.state == 1) or (self.state == 2) or (self.state == 3):
        #     traj_action[:,-2:] = 0

        return traj_action
