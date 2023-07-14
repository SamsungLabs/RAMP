import os
from collections.abc import Iterable

import h5py
import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tra
from autolab_core import ColorImage, DepthImage
from .utils import get_T
from copy import deepcopy


class ObjectPlacementNotFound(Exception):
    pass


class SceneManager:
    def __init__(self, dataset_folder, renderer=None):
        self._dataset_path = dataset_folder
        obj_info = h5py.File(
            os.path.join(self._dataset_path, "object_info.hdf5"), "r"
        )
        self.mesh_info = obj_info["meshes"]
        self.categories = ['TV', 'Bowl']

        self._collision_manager = trimesh.collision.CollisionManager()
        if renderer is not None and not isinstance(renderer, SceneRenderer):
            raise ValueError("renderer must be of type SceneRenderer")
        self._renderer = renderer

        self.objs = {}

        self._gravity_axis = 2
        self._table_dims = np.array([1.0, 1.6, 0.2])
        self._table_pose = np.eye(4)
        self.valid_experiments = [0,1,2,3,4,5,10,11,12,13,14,15]

    @property
    def camera_pose(self):
        if self._renderer is None:
            raise ValueError("SceneManager does not contain a renderer!")
        return self._renderer.camera_pose

    @camera_pose.setter
    def camera_pose(self, cam_pose):
        if self._renderer is None:
            raise ValueError("SceneManager does not contain a renderer!")
        self._renderer.camera_pose = cam_pose

    @property
    def table_bounds(self):
        if not hasattr(self, "_table_bounds"):
            lbs = self._table_pose[:3, 3] - 0.5 * self._table_dims
            ubs = self._table_pose[:3, 3] + 0.5 * self._table_dims
            lbs[self._gravity_axis] = ubs[self._gravity_axis]
            ubs[self._gravity_axis] += 0.001
            lbs[self._gravity_axis] += 0.001
            self._table_bounds = (lbs, ubs)
        return self._table_bounds

    @table_bounds.setter
    def table_bounds(self, bounds):
        if not isinstance(bounds, np.ndarray):
            bounds = np.asarray(bounds)
        if bounds.shape != (2, 3):
            raise ValueError("Bounds is incorrect shape, should be (2, 3)")
        self._table_bounds = bounds

    def collides(self):
        return self._collision_manager.in_collision_internal()

    def min_distance(self, obj_manager):
        return self._collision_manager.min_distance_other(obj_manager)

    def add_object(self, name, mesh, info={}, pose=None, color=None):
        if name in self.objs:
            raise ValueError(
                "Duplicate name: object {} already exists".format(name)
            )

        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        color = (
            np.asarray((0.7, 0.7, 0.7)) if color is None else np.asarray(color)
        )
        mesh.visual.face_colors = np.tile(
            np.reshape(color, [1, 3]), [mesh.faces.shape[0], 1]
        )
        self.objs[name] = {"mesh": mesh, "pose": pose}
        if "grasps" in info:
            self.objs[name]["grasps"] = info["grasps"][()]
        self._collision_manager.add_object(
            name,
            mesh,
            transform=pose,
        )
        if self._renderer is not None:
            self._renderer.add_object(name, mesh, pose)

        return True

    def remove_object(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        self._collision_manager.remove_object(name)
        if self._renderer is not None:
            self._renderer.remove_object(name)
        del self.objs[name]

    def sample_obj(self, cat=None, obj=None):
        if cat is None:
            cat = np.random.choice(self.categories)
        elif isinstance(cat, Iterable):
            cat = np.random.choice(cat)
        if obj is None:
            if cat =='TV':
                obj = b'2213ac0d768e2f2b44c73a59ff49c982_0.017236322538757037'
            elif cat =='Bowl':
                obj = b'4be4184845972fba5ea36d5a57a5a3bb_0.0006581313232805561'
        try:
            mesh_path = os.path.join(
                self._dataset_path, self.mesh_info[obj]["path"].asstr()[()]
            )
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh.metadata["key"] = obj
            mesh.metadata["path"] = mesh_path
            info = self.mesh_info[obj]
        except (ValueError, TypeError):
            mesh = None
            info = None

        return mesh, info


    def place_obj(self, obj_id, mesh, info, max_attempts=10):
        self.add_object(obj_id, mesh, info=info)
        for _ in range(max_attempts):
            rand_stp = self._random_object_pose(info, *self.table_bounds)
            self.set_object_pose(obj_id, rand_stp)
            if not self.collides():
                return True

        self.remove_object(obj_id)
        return False


    def sample_and_place_obj(self, obj_id, max_attempts=10):
        for _ in range(max_attempts):
            obj_mesh, obj_info = self.sample_obj(cat=["Mug", "Book"])
            if not obj_mesh:
                continue
            if self.place_obj(obj_id, obj_mesh, obj_info):
                break
            else:
                continue

    def arrange_scene(self, num_objects, experiment, max_attempts=10):

        # Create and add table mesh
        table_mesh = trimesh.creation.box(self._table_dims)
        table_mesh.metadata["key"] = "table"
        self.add_object(
            name="table",
            mesh=table_mesh,
            pose=self._table_pose,
        )
        if experiment not in self.valid_experiments:
            raise ValueError(f"experiment id {experiment} is not defined or hasn't been added to self.valid_experiments")
        if experiment >= 10 and experiment <=19:
            experiment -= 10

        if experiment == 5:
            #Scene without any objects at all
            self.objs_initial = deepcopy(self.objs)
            return
        elif experiment == 100:
             # [SceneCollisionNet scenes] Sample and add objects randomly
            for i in range(num_objects):
                obj_id = "obj_{:d}".format(i + 1)
                self.sample_and_place_obj(obj_id, max_attempts=max_attempts)
        else:
            self.setup_experiment(experiment)
        #Make a backup of the scene setup
        self.objs_initial = deepcopy(self.objs)
        

    def setup_experiment(self, experiment):

        nominal_tv_pose, obj_mesh, obj_info = self.get_default_TV_pose()
        if experiment == 0:
            poses = self.get_scene1_poses(nominal_tv_pose)
        elif experiment == 1:
            poses = self.get_scene2_poses(nominal_tv_pose)
        elif experiment == 2:
            poses = self.get_scene3_poses(nominal_tv_pose)
        elif experiment == 3:
            poses = self.get_scene4_poses(nominal_tv_pose)
        elif experiment == 4:
            poses = self.get_scene6_poses(nominal_tv_pose)
        else:
            raise ValueError(f"No definition for experiment #{experiment}")
            
        #Place TVs as defined by objects
        for idx, pose in enumerate(poses):
            self.add_object(f"obj_{idx+1}", obj_mesh, info=obj_info)
            self.set_object_pose(f"obj_{idx+1}", pose)
        if not self.collides():
            return True
        else:
            print(f"Collision when setting up TV# {idx+1} for experiment {experiment}")
            for id in [f"obj_{id+1}" for id in range(len(poses))]:
                self.remove_object(id)
        return False


    def get_default_TV_pose(self):
        obj_mesh, obj_info = self.sample_obj(cat=["TV"])
        stps, _ = obj_info["stps"][()], obj_info["probs"][()]
        lbs, ubs = self._table_bounds
        pose = stps[0].copy()
        pose[:3, 3] += (lbs+ubs)/2.0
        pose[:3, 3] += np.array([0.15, 0.45, -0.21])
        pose = get_T(seq="X", angles=[90]) @ pose @ get_T(seq="Z", angles=[-90])
        return pose, obj_mesh, obj_info

    def get_scene1_poses(self, nominal_pose):
        pose1 = get_T(x=-0.35, y=-0.35) @ nominal_pose @ get_T(seq="Z", angles=[-55])
        pose2 = get_T(x=-0.13, y=-0.10) @  nominal_pose @ get_T(seq="Z", angles=[-28])
        pose3 = get_T(x=-0.05, y=0.2) @  nominal_pose @ get_T(seq="Z", angles=[0])
        pose4 = get_T(x=0.08, y=0.50) @ nominal_pose @ get_T(seq="Z", angles=[-45]) 
        return [pose1, pose2, pose3, pose4]

    def get_scene2_poses(self, nominal_pose):
        pose1 = get_T(x=0.05, y=-0.36) @ nominal_pose @ get_T(seq="Z", angles=[50])
        pose2 = get_T(x=-0.20, y=-0.15) @  nominal_pose @ get_T(seq="Z", angles=[50])
        pose3 = get_T(x=-0.20, y=0.15) @  nominal_pose @ get_T(seq="Z", angles=[-50])
        pose4 = get_T(x=0.05, y=0.36) @ nominal_pose @ get_T(seq="Z", angles=[-50]) 
        return [pose1, pose2, pose3, pose4]
    
    def get_scene3_poses(self, nominal_pose):
        pose1 = get_T(x=0.0) @ nominal_pose
        pose2 = get_T(x=0.3) @ nominal_pose
        return [pose1, pose2]
    
    def get_scene4_poses(self, nominal_pose):
        nominal_pose = np.array([[ 0.76888413, -0.63938814, -0.00002232,  0],                                                                                                                                                       
                                [ 0.63938807,  0.76888403,  0.00049378,  0],                                                       
                                [-0.00029855, -0.00039393,  0.99999988,  0.0],                                                       
                                [ 0.,          0.,          0.,          1.]])  
        pose1 = get_T(x = 0.54, y = -0.33, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[-2.19*180/np.pi]) 
        pose2 = get_T(x = 0.53, y = -0.058, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[-0.30*180/np.pi]) 
        pose3 = get_T(x = 0.74, y = 0.29, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[1.173*180/np.pi]) 
        pose4 = get_T(x = 0.73, y = 0.14, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[1.78*180/np.pi]) 

        return [pose1, pose2, pose3, pose4]

    def get_scene5_poses(self, nominal_pose):
        nominal_pose = np.array([[ 0.76888413, -0.63938814, -0.00002232,  0],                                                                                                                                                       
                                [ 0.63938807,  0.76888403,  0.00049378,  0],                                                       
                                [-0.00029855, -0.00039393,  0.99999988,  0.0],                                                       
                                [ 0.,          0.,          0.,          1.]])  
        pose1 = get_T(x = 0.84, y = -0.07, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[-0.72*180/np.pi]) 
        pose2 = get_T(x = 0.73, y = -0.26, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[-1.47*180/np.pi]) 
        pose3 = get_T(x = 0.66, y = 0.21, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[2.10*180/np.pi]) 
        pose4 = get_T(x = 0.82, y = 0.11, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[-0.51*180/np.pi]) 

        return [pose1, pose2, pose3, pose4]


    def get_scene6_poses(self, nominal_pose):
        nominal_pose = np.array([[ 0.76888413, -0.63938814, -0.00002232,  0],                                                                                                                                                       
                                [ 0.63938807,  0.76888403,  0.00049378,  0],                                                       
                                [-0.00029855, -0.00039393,  0.99999988,  0.0],                                                       
                                [ 0.,          0.,          0.,          1.]])  
        pose1 = get_T(x = 0.73, y = -0.33, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[-2.65*180/np.pi]) 
        pose2 = get_T(x = 0.95, y = -0.01, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[-2.34*180/np.pi]) 
        pose3 = get_T(x = 0.58, y = -0.15, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[0.15*180/np.pi]) 
        pose4 = get_T(x = 0.66, y = 0.23, z = 0.32) @ nominal_pose @ get_T(seq="Z", angles=[-0.52*180/np.pi]) 

        return [pose1, pose2, pose3, pose4]


    def get_object_pose(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        return self.objs[name]["pose"]

    def set_object_pose(self, name, pose):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        self.objs[name]["pose"] = pose
        self._collision_manager.set_transform(
            name,
            pose,
        )
        if self._renderer is not None:
            self._renderer.set_object_pose(name, pose)

    def render_points(self):
        if self._renderer is not None:
            return self._renderer.render_points()

    def reset(self):
        if self._renderer is not None:
            self._renderer.reset()

        for name in self.objs:
            self._collision_manager.remove_object(name)

        self.objs = {}

    def reset_objs_to_initial_poses(self):
        self.objs = deepcopy(self.objs_initial)

    def _random_object_pose(self, obj_info, lbs, ubs):
        stps, probs = obj_info["stps"][()], obj_info["probs"][()]
        pose = stps[np.random.choice(len(stps), p=probs)].copy()
        pose[:3, 3] += np.random.uniform(lbs, ubs)
        z_rot = tra.rotation_matrix(
            2 * np.pi * np.random.rand(), [0, 0, 1], point=pose[:3, 3]
        )
        return z_rot @ pose


class SceneRenderer:
    def __init__(self):

        self._scene = pyrender.Scene()
        self._node_dict = {}
        self._camera_intr = None
        self._camera_node = None
        self._light_node = None
        self._renderer = None

    def create_camera(self, intr, znear, zfar):
        cam = pyrender.IntrinsicsCamera(
            intr.fx, intr.fy, intr.cx, intr.cy, znear, zfar
        )
        self._camera_intr = intr
        self._camera_node = pyrender.Node(camera=cam, matrix=np.eye(4))
        self._scene.add_node(self._camera_node)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        self._light_node = pyrender.Node(light=light, matrix=np.eye(4))
        self._scene.add_node(self._light_node)
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=intr.width,
            viewport_height=intr.height,
            point_size=5.0,
        )

    @property
    def camera_pose(self):
        if self._camera_node is None:
            return None
        return self._camera_node.matrix

    @camera_pose.setter
    def camera_pose(self, cam_pose):
        if self._camera_node is None:
            raise ValueError("No camera in scene!")
        self._scene.set_pose(self._camera_node, cam_pose)
        self._scene.set_pose(self._light_node, cam_pose)

    def render_rgbd(self, depth_only=False):

        if depth_only:
            depth = self._renderer.render(
                self._scene, pyrender.RenderFlags.DEPTH_ONLY
            )
            color = None
            depth = DepthImage(depth, frame="camera")
        else:
            color, depth = self._renderer.render(self._scene)
            color = ColorImage(color, frame="camera")
            depth = DepthImage(depth, frame="camera")

        return color, depth

    def render_segmentation(self, full_depth=None):
        if full_depth is None:
            _, full_depth = self.render_rgbd(depth_only=True)

        self.hide_objects()
        output = np.zeros(full_depth.data.shape, dtype=np.uint8)
        for i, obj_name in enumerate(self._node_dict):
            self._node_dict[obj_name].mesh.is_visible = True
            _, depth = self.render_rgbd(depth_only=True)
            mask = np.logical_and(
                (np.abs(depth.data - full_depth.data) < 1e-6),
                np.abs(full_depth.data) > 0,
            )
            if np.any(output[mask] != 0):
                raise ValueError("wrong label")
            output[mask] = i + 1
            self._node_dict[obj_name].mesh.is_visible = False
        self.show_objects()

        return output, ["BACKGROUND"] + list(self._node_dict.keys())

    def render_points(self):
        _, depth = self.render_rgbd(depth_only=True)
        point_norm_cloud = depth.point_normal_cloud(self._camera_intr)

        pts = point_norm_cloud.points.data.T.reshape(
            depth.height, depth.width, 3
        )
        norms = point_norm_cloud.normals.data.T.reshape(
            depth.height, depth.width, 3
        )
        # cp = self.get_camera_pose()
        # import pdb
        # pdb.set_trace()
        # cp = self.camera_pose()
        cp = self._camera_node.matrix
        cp[:, 1:3] *= -1

        pt_mask = np.logical_and(
            np.linalg.norm(pts, axis=-1) != 0.0,
            np.linalg.norm(norms, axis=-1) != 0.0,
        )
        pts = tra.transform_points(pts[pt_mask], cp)
        return pts.astype(np.float32)

    def add_object(self, name, mesh, pose=None):
        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        node = pyrender.Node(
            name=name,
            mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False),
            matrix=pose,
        )
        self._node_dict[name] = node
        self._scene.add_node(node)

    def add_points(self, points, name, pose=None, color=None, radius=0.005):
        points = np.asanyarray(points)
        if points.ndim == 1:
            points = np.array([points])

        if pose is None:
            pose = np.eye(4)
        else:
            pose = pose.matrix

        color = (
            np.asanyarray(color, dtype=np.float) if color is not None else None
        )

        # If color specified per point, use sprites
        if color is not None and color.ndim > 1:
            self._renderer.point_size = 1000 * radius
            m = pyrender.Mesh.from_points(points, colors=color)
        # otherwise, we can make pretty spheres
        else:
            mesh = trimesh.creation.uv_sphere(radius, [20, 20])
            if color is not None:
                mesh.visual.vertex_colors = color
            poses = None
            poses = np.tile(np.eye(4), (len(points), 1)).reshape(
                len(points), 4, 4
            )
            poses[:, :3, 3::4] = points[:, :, None]
            m = pyrender.Mesh.from_trimesh(mesh, poses=poses)

        node = pyrender.Node(mesh=m, name=name, matrix=pose)
        self._node_dict[name] = node
        self._scene.add_node(node)

    def set_object_pose(self, name, pose):
        self._scene.set_pose(self._node_dict[name], pose)

    def has_object(self, name):
        return name in self._node_dict

    def remove_object(self, name):
        self._scene.remove_node(self._node_dict[name])
        del self._node_dict[name]

    def show_objects(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.is_visible = True

    def toggle_wireframe(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.primitives[0].material.wireframe ^= True

    def hide_objects(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.is_visible = False

    def reset(self):
        for name in self._node_dict:
            self._scene.remove_node(self._node_dict[name])
        self._node_dict = {}
