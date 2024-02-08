"""Helper classes and functions to sample grasps for a given object mesh."""

import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import tensorflow.compat.v1 as tf
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm
from transforms3d.euler import euler2mat


class Object:
    """Represents a graspable object."""

    def __init__(self, filename):
        """Constructor.

        :param filename: Mesh to load
        :param scale: Scaling factor
        """
        self.mesh = trimesh.load(filename)
        self.scale = 1.0

        # print(filename)
        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object("object", self.mesh)

    def rescale(self, scale=1.0):
        """Set scale of object mesh.

        :param scale
        """
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def resize(self, size=1.0):
        """Set longest of all three lengths in Cartesian space.

        :param size
        """
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)

    def in_collision_with(self, mesh, transform):
        """Check whether the object is in collision with the provided mesh.

        :param mesh:
        :param transform:
        :return: boolean value
        """
        return self.collision_manager.in_collision_single(mesh, transform=transform)


class PandaGripper:
    """An object representing a Franka Panda gripper."""

    def __init__(
        self,
        q=None,
        num_contact_points_per_finger=10,
        root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ):
        """Create a Franka Panda parallel-yaw gripper object.

        Keyword Arguments:
            q {list of int} -- opening configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
        """
        self.joint_limits = [0.0, 0.04]
        self.root_folder = root_folder

        self.default_pregrasp_configuration = 0.04
        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        fn_base = os.path.join(root_folder, "gripper_models/panda_gripper/hand.stl")
        fn_finger = os.path.join(root_folder, "gripper_models/panda_gripper/finger.stl")

        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.contact_ray_origins = []
        self.contact_ray_directions = []

        # coords_path = os.path.join(root_folder, 'gripper_control_points/panda_gripper_coords.npy')
        with open(
            os.path.join(
                root_folder, "gripper_control_points/panda_gripper_coords.pickle"
            ),
            "rb",
        ) as f:
            self.finger_coords = pickle.load(f, encoding="latin1")
        finger_direction = (
            self.finger_coords["gripper_right_center_flat"]
            - self.finger_coords["gripper_left_center_flat"]
        )
        self.contact_ray_origins.append(
            np.r_[self.finger_coords["gripper_left_center_flat"], 1]
        )
        self.contact_ray_origins.append(
            np.r_[self.finger_coords["gripper_right_center_flat"], 1]
        )
        self.contact_ray_directions.append(
            finger_direction / np.linalg.norm(finger_direction)
        )
        self.contact_ray_directions.append(
            -finger_direction / np.linalg.norm(finger_direction)
        )

        self.contact_ray_origins = np.array(self.contact_ray_origins)
        self.contact_ray_directions = np.array(self.contact_ray_directions)

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.

        Returns:
            list of trimesh -- visual meshes
        """
        return [self.finger_l, self.finger_r, self.base]

    def get_closing_rays_contact(self, transform):
        """Get an array of rays defining the contact locations and directions on the hand.

        Arguments:
            transform {[nump.array]} -- a 4x4 homogeneous matrix
            contact_ray_origin {[nump.array]} -- a 4x1 homogeneous vector
            contact_ray_direction {[nump.array]} -- a 4x1 homogeneous vector

        Returns:
            numpy.array -- transformed rays (origin and direction)
        """
        return (
            transform[:3, :].dot(self.contact_ray_origins.T).T,
            transform[:3, :3].dot(self.contact_ray_directions.T).T,
        )

    def get_control_point_tensor(
        self, batch_size, use_tf=True, symmetric=False, convex_hull=True
    ):
        """
        Outputs a 5 point gripper representation of shape (batch_size x 5 x 3).

        Arguments:
            batch_size {int} -- batch size

        Keyword Arguments:
            use_tf {bool} -- outputing a tf tensor instead of a numpy array (default: {True})
            symmetric {bool} -- Output the symmetric control point configuration of the gripper (default: {False})
            convex_hull {bool} -- Return control points according to the convex hull panda gripper model (default: {True})

        Returns:
            np.ndarray -- control points of the panda gripper
        """

        control_points = np.load(
            os.path.join(self.root_folder, "gripper_control_points/panda.npy")
        )[:, :3]
        if symmetric:
            control_points = [
                [0, 0, 0],
                control_points[1, :],
                control_points[0, :],
                control_points[-1, :],
                control_points[-2, :],
            ]
        else:
            control_points = [
                [0, 0, 0],
                control_points[0, :],
                control_points[1, :],
                control_points[-2, :],
                control_points[-1, :],
            ]

        control_points = np.asarray(control_points, dtype=np.float32)
        if not convex_hull:
            # actual depth of the gripper different from convex collision model
            control_points[1:3, 2] = 0.0584
        control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])

        if use_tf:
            return tf.convert_to_tensor(control_points)

        return control_points


class XArmGripper:
    """An object representing a UFactory XArm gripper."""

    def __init__(self, root_folder=Path(__file__).resolve().parents[1]):
        mesh_dir = Path(root_folder) / "gripper_models/xarm_gripper"

        self.load_gripper_meshes(mesh_dir)

        self.joint_limits = [0.0, 0.0453556139430441]

    def load_gripper_meshes(self, mesh_dir: Path):
        """Load gripper meshes and apply relative transform to gripper hand link"""
        self.camera = o3d.io.read_triangle_model(
            str(mesh_dir / "d435_with_tilt_cam_stand.STL")
        )
        self.hand = o3d.io.read_triangle_model(str(mesh_dir / "base_link.STL"))
        self.finger_l = o3d.io.read_triangle_model(
            str(mesh_dir / "left_finger_black.glb")
        )
        self.finger_r = o3d.io.read_triangle_model(
            str(mesh_dir / "right_finger_black.glb")
        )

        T_eef_camera = np.eye(4)
        T_eef_camera[:3, :3] = euler2mat(0, 0, np.pi)
        T_eef_gripper = np.eye(4)
        T_eef_gripper[:3, -1] = [0, 0, 0.005]
        T_gripper_camera = np.linalg.inv(T_eef_gripper) @ T_eef_camera
        self.camera = transform_geometry(self.camera, T_gripper_camera)

        # finger_pts are the 4 corners of the square finger contact pad
        self.finger_l_pts = np.array([
            [0.01475, -0.026003, 0.022253],
            [-0.01475, -0.026003, 0.022253],
            [-0.01475, -0.026003, 0.059753],
            [0.01475, -0.026003, 0.059753],
        ])
        self.finger_r_pts = self.finger_l_pts * [1, -1, 1]  # invert y coords
        T_gripper_finger_l = np.eye(4)
        T_gripper_finger_l[:3, -1] = [0, 0.02682323, 0.11348719]
        self.finger_l_pts = transform_points(self.finger_l_pts, T_gripper_finger_l)
        self.finger_l = transform_geometry(self.finger_l, T_gripper_finger_l)
        self.finger_l_joint_axis = np.array([0, 0.96221329, -0.27229686])
        T_gripper_finger_r = np.eye(4)
        T_gripper_finger_r[:3, -1] = [0, -0.02682323, 0.11348719]
        self.finger_r_pts = transform_points(self.finger_r_pts, T_gripper_finger_r)
        self.finger_r = transform_geometry(self.finger_r, T_gripper_finger_r)
        self.finger_r_joint_axis = np.array([0, -0.96221329, -0.27229686])

        self.gripper_opening_to_q_val = 2 * np.cos(
            np.arctan2(self.finger_l_joint_axis[2], self.finger_l_joint_axis[1])
        )
        self.gripper_width = 0.086

    def get_mesh(self, q=0.0, pose=np.eye(4)) -> rendering.TriangleMeshModel:
        """Return gripper mesh at given q value and pose

        :param q: gripper finger joint value
        :param pose: gripper pose from world frame to xarm_gripper_base_link
        :return mesh: gripper mesh, rendering.TriangleMeshModel
        """
        T = np.eye(4)
        T[:3, -1] = self.finger_l_joint_axis * q
        finger_l = transform_geometry(self.finger_l, T)

        T = np.eye(4)
        T[:3, -1] = self.finger_r_joint_axis * q
        finger_r = transform_geometry(self.finger_r, T)

        gripper = merge_geometries([self.camera, self.hand, finger_l, finger_r])
        return transform_geometry(gripper, pose)

    def get_control_points(
        self, q=0.0, pose=np.eye(4), use_tf=False, symmetric=False
    ) -> np.ndarray:
        """Return the 5 control points of gripper representation
        Control point order indices are shown below (symmetric=True is in parentheses)
                  * 0 (0)
                  |            y <---*
        1 (2) *-------* 2 (1)        |
              |       |              v
        3 (4) *       * 4 (3)         z
            left    right

        :param q: gripper finger joint value
        :param pose: gripper pose from world frame to xarm_gripper_base_link
        :return pts: [batch_size, 5, 3] np.floating np.ndarray
        """
        q = np.array([q]).reshape(-1)
        pose = np.array([pose]).reshape(-1, 4, 4)
        assert len(q) == len(pose), f"{q.shape = } {pose.shape = }"

        T = np.tile(np.eye(4), (len(q), 1, 1))
        T[..., :3, -1] = self.finger_l_joint_axis * q[:, None]
        finger_l_pts = transform_points_batch(
            self.finger_l_pts.reshape(2, 2, 3).mean(1), T
        )

        T = np.tile(np.eye(4), (len(q), 1, 1))
        T[..., :3, -1] = self.finger_r_joint_axis * q[:, None]
        finger_r_pts = transform_points_batch(
            self.finger_r_pts.reshape(2, 2, 3).mean(1), T
        )

        if symmetric:
            control_points = np.stack([finger_r_pts, finger_l_pts], axis=-2).reshape(
                -1, 4, 3
            )
        else:
            control_points = np.stack([finger_l_pts, finger_r_pts], axis=-2).reshape(
                -1, 4, 3
            )
        control_points = np.concatenate(
            [np.zeros((len(q), 1, 3)), control_points], axis=-2
        )
        control_points = transform_points_batch(control_points, pose)

        if use_tf:
            return tf.convert_to_tensor(control_points)

        return control_points

    def get_control_points_lineset(
        self, control_points: np.ndarray
    ) -> o3d.geometry.LineSet:
        """Contruct a LineSet for visualizing control_points
        :param control_points: [batch_size, 5, 3] np.floating np.ndarray
        """
        control_points = np.array(control_points).reshape(-1, 5, 3)
        batch_size = len(control_points)

        # Add mid point
        control_points = np.concatenate(
            [control_points[:, 1:3].mean(1, keepdims=True), control_points], axis=-2
        )

        lines = np.array([[0, 1], [2, 3], [2, 4], [3, 5]])
        lines = np.tile(lines, (batch_size, 1, 1))
        lines += np.arange(batch_size)[:, None, None] * 6

        lineset = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(control_points.reshape(-1, 3)),
            o3d.utility.Vector2iVector(lines.reshape(-1, 2)),
        )
        return lineset

    def visualize_grasps(
        self,
        vis: "O3DGUIVisualizer",
        pred_grasps_cam: Union[np.ndarray, Dict[int, np.ndarray]],
        scores: Union[np.ndarray, Dict[int, np.ndarray]],
        q_vals: Union[np.ndarray, Dict[int, np.ndarray]],
        cam_pose: np.ndarray = np.eye(4),
        group_prefix="CGN",
    ):
        """Visualize grasps using O3DGUIVisualizer

        :param vis: O3DGUIVisualizer instance
        :param pred_grasps_cam: predicted grasp poses in camera frame.
                                [n_grasps, 4, 4] np.floating np.ndarray
        :param scores: predicted grasp pose confidence scores.
                       [n_grasps,] np.floating np.ndarray
        :param q_vals: predicted gripper joint values
                       [n_grasps,] np.floating np.ndarray
        :param cam_pose: camera pose in world frame, [4, 4] np.floating np.ndarray
        :param group_prefix: prefix for visualization geometry name
        """
        if isinstance(pred_grasps_cam, np.ndarray):
            pred_grasps_cam = {1: pred_grasps_cam}
            scores = {1: scores}
            q_vals = {1: q_vals}

        for seg_id in pred_grasps_cam:
            max_score_idx = scores[seg_id].argmax()

            pred_grasps_world = cam_pose @ pred_grasps_cam[seg_id]

            vis.add_geometry(
                f"{group_prefix}/obj_{seg_id}/mesh",
                self.get_mesh(
                    q_vals[seg_id][max_score_idx], pred_grasps_world[max_score_idx]
                ),
            )

            control_points = self.get_control_points(q_vals[seg_id], pred_grasps_world)
            vis.add_geometry(
                f"{group_prefix}/obj_{seg_id}/pts",
                self.get_control_points_lineset(control_points),
                show=False,
            )

    def convert_grasp_poses_from_panda(
        self,
        obj_grasp_poses: Union[np.ndarray, Dict[int, np.ndarray]],
        obj_gripper_openings: Optional[Union[np.ndarray, Dict[int, np.ndarray]]] = None,
    ) -> np.ndarray:
        """Convert a Panda gripper grasp pose to XArm gripper

        :param grasp_poses: gripper pose from world frame to panda_hand
                            approaching +z, baseline +x
        :param gripper_openings: gripper opening widths
        :return grasp_poses: gripper pose from world frame to xarm_gripper_base_link
                             approaching +z, baseline +y
                             [batch_size, 4, 4] np.floating np.ndarray
        :return q_vals: gripper joint value, [batch_size,] np.floating np.ndarray
        """
        is_grasp_pose_dict = True
        if isinstance(obj_grasp_poses, np.ndarray):
            is_grasp_pose_dict = False
            obj_grasp_poses = {1: obj_grasp_poses}
            if obj_gripper_openings is not None:
                obj_gripper_openings = {1: obj_gripper_openings}

        xarm_obj_grasp_poses = {}
        xarm_obj_q_vals = {}
        for seg_id in obj_grasp_poses:
            grasp_poses = np.array([obj_grasp_poses[seg_id]]).reshape(-1, 4, 4)
            num_grasps = len(grasp_poses)
            if obj_gripper_openings is None:
                gripper_openings = [self.gripper_width] * num_grasps
            else:
                gripper_openings = np.array([obj_gripper_openings[seg_id]]).reshape(-1)
            assert (
                len(gripper_openings) == num_grasps
            ), f"{gripper_openings.shape = } {grasp_poses.shape = }"

            # Convert gripper width to joint value
            xarm_obj_q_vals[seg_id] = q_vals = (
                gripper_openings / self.gripper_opening_to_q_val
            )

            # Get XArm gripper contact pad center (tcp) z coord in gripper frame
            T = np.tile(np.eye(4), (num_grasps, 1, 1))
            T[..., :3, -1] = self.finger_l_joint_axis * q_vals[:, None]
            tcp_pts_z = transform_points_batch(
                self.finger_l_pts.reshape(2, 2, 3).mean(1), T
            ).mean(1)[:, 2]

            # Panda gripper contact pad center (tcp) z coord in gripper frame
            panda_tcp_pts_z = 0.1034

            T = np.eye(4)
            T[:3, :3] = euler2mat(0, 0, -np.pi / 2)
            T = np.tile(T, (num_grasps, 1, 1))
            T[..., 2, -1] = panda_tcp_pts_z - tcp_pts_z

            xarm_obj_grasp_poses[seg_id] = grasp_poses @ T

        if not is_grasp_pose_dict:
            return xarm_obj_grasp_poses[1], xarm_obj_q_vals[1]

        return xarm_obj_grasp_poses, xarm_obj_q_vals


def create_gripper(
    name,
    configuration=None,
    root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
):
    """Create a gripper object.

    Arguments:
        name {str} -- name of the gripper

    Keyword Arguments:
        configuration {list of float} -- configuration (default: {None})
        root_folder {str} -- base folder for model files (default: {''})

    Raises:
        Exception: If the gripper name is unknown.

    Returns:
        [type] -- gripper object
    """
    if name.lower() == "panda":
        return PandaGripper(q=configuration, root_folder=root_folder)
    else:
        raise Exception("Unknown gripper: {}".format(name))


def in_collision_with_gripper(
    object_mesh, gripper_transforms, gripper_name, silent=False
):
    """Check collision of object with gripper.

    Arguments:
        object_mesh {trimesh} -- mesh of object
        gripper_transforms {list of numpy.array} -- homogeneous matrices of gripper
        gripper_name {str} -- name of gripper

    Keyword Arguments:
        silent {bool} -- verbosity (default: {False})

    Returns:
        [list of bool] -- Which gripper poses are in collision with object mesh
    """
    manager = trimesh.collision.CollisionManager()
    manager.add_object("object", object_mesh)
    gripper_meshes = [create_gripper(gripper_name).hand]
    min_distance = []
    for tf in tqdm(gripper_transforms, disable=silent):
        min_distance.append(
            np.min([
                manager.min_distance_single(gripper_mesh, transform=tf)
                for gripper_mesh in gripper_meshes
            ])
        )

    return [d == 0 for d in min_distance], min_distance


def grasp_contact_location(
    transforms, successfuls, collisions, object_mesh, gripper_name="panda", silent=False
):
    """Computes grasp contacts on objects and normals, offsets, directions

    Arguments:
        transforms {[type]} -- grasp poses
        collisions {[type]} -- collision information
        object_mesh {trimesh} -- object mesh

    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})

    Returns:
        list of dicts of contact information per grasp ray
    """
    res = []
    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True
        )
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
    for p, colliding, outcome in tqdm(
        zip(transforms, collisions, successfuls), total=len(transforms), disable=silent
    ):
        contact_dict = {}
        contact_dict["collisions"] = 0
        contact_dict["valid_locations"] = 0
        contact_dict["successful"] = outcome
        contact_dict["grasp_transform"] = p
        contact_dict["contact_points"] = []
        contact_dict["contact_directions"] = []
        contact_dict["contact_face_normals"] = []
        contact_dict["contact_offsets"] = []

        if colliding:
            contact_dict["collisions"] = 1
        else:
            ray_origins, ray_directions = gripper.get_closing_rays_contact(p)

            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False
            )

            if len(locations) > 0:
                # this depends on the width of the gripper
                valid_locations = (
                    np.linalg.norm(ray_origins[index_ray] - locations, axis=1)
                    <= 2.0 * gripper.q
                )

                if sum(valid_locations) > 1:
                    contact_dict["valid_locations"] = 1
                    contact_dict["contact_points"] = locations[valid_locations]
                    contact_dict["contact_face_normals"] = object_mesh.face_normals[
                        index_tri[valid_locations]
                    ]
                    contact_dict["contact_directions"] = ray_directions[
                        index_ray[valid_locations]
                    ]
                    contact_dict["contact_offsets"] = np.linalg.norm(
                        ray_origins[index_ray[valid_locations]]
                        - locations[valid_locations],
                        axis=1,
                    )
                    # dot_prods = (contact_dict['contact_face_normals'] * contact_dict['contact_directions']).sum(axis=1)
                    # contact_dict['contact_cosine_angles'] = np.cos(dot_prods)
                    res.append(contact_dict)

    return res


def transform_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Transform points by 4x4 transformation matrix H
    :return out: same shape as pts
    """
    assert H.shape == (4, 4), H.shape
    assert pts.shape[-1] == 3, pts.shape

    return pts @ H[:3, :3].T + H[:3, 3]


def transform_points_batch(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Transform points by Bx4x4 transformation matrix H

    [3,], [4, 4] => [3,]
    [P, 3], [4, 4] => [P, 3]
    [H, W, 3], [4, 4] => [H, W, 3]
    [N, H, W, 3], [4, 4] => [N, H, W, 3]
    [P, 3], [B, 4, 4] => [B, P, 3]
    [B, P, 3], [B, 4, 4] => [B, P, 3]
    [H, W, 3], [B, 4, 4] => [B, H, W, 3]  # (H != B)
    [B, H, W, 3], [B, 4, 4] => [B, H, W, 3]
    [N, H, W, 3], [B, 4, 4] => [B, N, H, W, 3]  # (N != B)
    [B, N, H, W, 3], [B, 4, 4] => [B, N, H, W, 3]
    [B, N, 3], [B, N, 4, 4] => [B, N, 3]
    [B, N, P, 3], [B, N, 4, 4] => [B, N, P, 3]
    [B, N, H, W, 3], [B, N, 4, 4] => [B, N, H, W, 3]
    """
    assert H.shape[-2:] == (4, 4), H.shape
    assert pts.shape[-1] == 3, pts.shape

    batch_shape = H.shape[:-2]
    pts_shape = batch_shape + (-1, 3)
    out_pts_shape = pts.shape
    if batch_shape != pts.shape[: len(batch_shape)] or pts.ndim < H.ndim < 4:
        pts_shape = (-1, 3)
        out_pts_shape = batch_shape + out_pts_shape

    H = H.swapaxes(-1, -2)
    return (pts.reshape(pts_shape) @ H[..., :3, :3] + H[..., [3], :3]).reshape(
        out_pts_shape
    )


O3D_GEOMETRIES = (
    o3d.geometry.Geometry3D,
    o3d.t.geometry.Geometry,
    rendering.TriangleMeshModel,
)
ANY_O3D_GEOMETRY = Union[O3D_GEOMETRIES]


def transform_geometry(geometry: ANY_O3D_GEOMETRY, T: np.ndarray) -> ANY_O3D_GEOMETRY:
    """Apply transformation to o3d geometry, always returns a copy

    :param T: transformation matrix, [4, 4] np.floating np.ndarray
    """
    if isinstance(geometry, rendering.TriangleMeshModel):
        out_geometry = rendering.TriangleMeshModel()
        out_geometry.meshes = [
            rendering.TriangleMeshModel.MeshInfo(
                deepcopy(mesh_info.mesh).transform(T),
                mesh_info.mesh_name,
                mesh_info.material_idx,
            )
            for mesh_info in geometry.meshes
        ]
        out_geometry.materials = geometry.materials
    elif isinstance(geometry, (o3d.geometry.Geometry3D, o3d.t.geometry.Geometry)):
        out_geometry = deepcopy(geometry).transform(T)
    else:
        raise TypeError(f"Unknown o3d geometry type: {type(geometry)}")
    return out_geometry


O3D_GEOMETRY_LIST = Union[tuple(List[t] for t in O3D_GEOMETRIES)]


def merge_geometries(geometries: O3D_GEOMETRY_LIST) -> ANY_O3D_GEOMETRY:
    """Merge a list of o3d geometries, must be of same type"""
    geometry_types = set([type(geometry) for geometry in geometries])
    assert len(geometry_types) == 1, f"Not the same geometry type: {geometry_types = }"

    merged_geometry = next(iter(geometry_types))()
    for i, geometry in enumerate(geometries):
        if isinstance(geometry, rendering.TriangleMeshModel):
            num_materials = len(merged_geometry.materials)
            merged_geometry.meshes += [
                rendering.TriangleMeshModel.MeshInfo(
                    deepcopy(mesh_info.mesh),
                    f"mesh_{i}_{mesh_info.mesh_name}".strip("_"),
                    mesh_info.material_idx + num_materials,
                )
                for mesh_info in geometry.meshes
            ]
            merged_geometry.materials += geometry.materials
        elif isinstance(geometry, (o3d.geometry.Geometry3D, o3d.t.geometry.Geometry)):
            merged_geometry += geometry
        else:
            raise TypeError(f"Unknown o3d geometry type: {type(geometry)}")
    return merged_geometry
