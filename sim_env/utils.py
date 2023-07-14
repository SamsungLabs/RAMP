import os
from typing import List

import numpy as np
import trimesh
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation as R
from isaacgym import gymapi


def get_T(
        x: float = 0.0, 
        y: float = 0.0, 
        z: float = 0.0, 
        seq: str = None,
        angles: List[float] = [],
    ) -> np.array:
    """
    Returns a 4x4 homogeneous transformation matrix.
    :param x: translation in x coordinate
    :param y: translation in y coordinate
    :param z: translation in z coordinate
    :param seq: string of euler axes to define rotation matrix
    :param angles: list of angles of rotation for the axes defined in seq
    :returns: 4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3,3] = np.array([x,y,z])
    if seq is not None:
        T[:3,:3] = R.from_euler(seq, angles, degrees=True).as_matrix()
    return T

def get_valid_configs(scenario: str, num_configs: int=1, min_distance: float = 0.3):
    set_configs = []
    set_positions = []
    scenario_data = np.load(f"./resources/targets/scenario_{scenario}_feasible_configs.npz")
    # scenario_01_feasible_configs
    set_free_configuration = scenario_data['set_free_configuration']
    # set_free_configuration = set_free_configuration.squeeze(1)
    set_free_pose = scenario_data['set_free_pose']
    # Random Start and goal configuration generation
    while len(set_configs) < num_configs:

        if len(set_configs)==0:
            rnd_idx = np.random.choice(np.size(set_free_configuration,0), 1, replace=False)
            set_configs.append(set_free_configuration[rnd_idx[0],:])
            set_positions.append(set_free_pose[rnd_idx[0],:])
        else:
            while True: 
                rnd_idx = np.random.choice(np.size(set_free_configuration,0), 1, replace=False)
                if np.linalg.norm(set_configs[-1] - set_free_configuration[rnd_idx[0],:]) > min_distance:
                    set_configs.append(set_free_configuration[rnd_idx[0],:])
                    set_positions.append(set_free_pose[rnd_idx[0],:])
                    break
    return set_configs, set_positions        

def get_all_valid_positions(experiment):
    scenario_data = np.load(f"./resources/targets/scenario_0{experiment}_feasible_configs.npz")
    return scenario_data['set_free_pose']

def pose_matrix_to_gym_pose(pose_matrix: np.ndarray):
    assert pose_matrix.shape == (4,4)
    gym_pose = gymapi.Transform()
    gym_pose.p = gymapi.Vec3(pose_matrix[0,3], pose_matrix[1,3], pose_matrix[2,3])
    gym_pose.r = gymapi.Quat(R.from_matrix(pose_matrix[:3,:3].as_quat()))
    return gym_pose

def gym_pose_to_matrix(gym_pose: gymapi.Transform):
    pose_matrix = np.eye(4)
    pose_matrix[:3,3] = [gym_pose.p.x, gym_pose.p.y, gym_pose.p.z]
    pose_matrix[:3,:3] = R.from_quat([gym_pose.r.x, gym_pose.r.y, gym_pose.r.z, gym_pose.r.w]).as_matrix()
    return pose_matrix

def from_rpy(a, e, th):
    rt = tra.euler_matrix(a, e, th)
    q = tra.quaternion_from_matrix(rt)
    q = np.roll(q, -1)
    return gymapi.Quat(q[0], q[1], q[2], q[3])

def gym_pose_to_matrix(pose):
    q = [pose["r"][3], pose["r"][0], pose["r"][1], pose["r"][2]]
    trans = tra.quaternion_matrix(q)
    trans[:3, 3] = [pose["p"][i] for i in range(3)]

    return trans

def write_urdf(
    obj_name,
    obj_path,
    output_folder,
):
    content = open("resources/urdf.template").read()
    content = content.replace("NAME", obj_name)
    content = content.replace("MEAN_X", "0.0")
    content = content.replace("MEAN_Y", "0.0")
    content = content.replace("MEAN_Z", "0.0")
    content = content.replace("SCALE", "1.0")
    content = content.replace("COLLISION_OBJ", obj_path)
    content = content.replace("GEOMETRY_OBJ", obj_path)
    urdf_path = os.path.abspath(
        os.path.join(output_folder, obj_name + ".urdf")
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    open(urdf_path, "w").write(content)
    return urdf_path
