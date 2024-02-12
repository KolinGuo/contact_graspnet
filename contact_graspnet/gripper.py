from __future__ import annotations

import abc
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering  # type: ignore
import tensorflow.compat.v1 as tf  # type: ignore
import trimesh
import trimesh.transformations as tra
from transforms3d.euler import euler2mat

from .mesh_utils import (
    merge_geometries,
    transform_geometry,
    transform_points,
    transform_points_batch,
)


class Gripper(abc.ABC):
    """Abastract base class for different gripper"""

    @abc.abstractmethod
    def convert_grasp_poses(
        self,
        pred_grasp_poses: np.ndarray | dict[int, np.ndarray],
        pred_gripper_openings: Optional[
            float | np.ndarray | dict[int, float | np.ndarray]
        ] = None,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[dict[int, np.ndarray], dict[int, np.ndarray]]
    ):
        """Converts grasp poses to align with URDF frame orientation
        which has approaching +z, baseline +y

        Gripper pose orientation is shown below
                  * gripper link origin
                  |            y <---*
              *-------*              |
              |       |              v
              *       *              z
            left    right

        All params and returns can be {seg_id: np.ndarray} or np.ndarray.

        :param pred_grasp_poses: CGN predicted gripper pose of panda_hand
                                 approaching +z, baseline +x
                                 [n_grasps, 4, 4] or [4, 4] np.floating np.ndarray
        :param pred_gripper_openings: CGN predicted gripper opening widths
                                      [n_grasps,] np.floating np.ndarray or float
                                      If None, use self.gripper_width.
        :return grasp_poses: converted gripper pose to align with URDF frame orientation
                             approaching +z, baseline +y
                             [n_grasps, 4, 4] np.floating np.ndarray
        :return q_vals: gripper joint value, [n_grasps,] np.floating np.ndarray
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_mesh(
        self, q: float = 0.0, pose: np.ndarray = np.eye(4)
    ) -> rendering.TriangleMeshModel:
        """Return gripper mesh at given q value and pose

        :param q: gripper finger joint value
        :param pose: gripper pose from world frame to gripper base link
        :return: gripper mesh, rendering.TriangleMeshModel
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_control_points(
        self,
        q: float | np.ndarray,
        pose: np.ndarray = np.eye(4),
        *,
        use_tf=False,
        symmetric=False,
    ) -> np.ndarray | tf.Tensor:
        """
        Return the 5 control points of gripper representation at given q value and pose.
        Control point order indices are shown below (symmetric=True is in parentheses)
                  * 0 (0)
                  |            y <---*
        1 (2) *-------* 2 (1)        |
              |       |              v
        3 (4) *       * 4 (3)         z
            left    right

        :param q: gripper finger joint value
        :param pose: gripper pose from world frame to gripper base link
        :return: [batch_size, 5, 3] np.floating np.ndarray
        """
        raise NotImplementedError

    @staticmethod
    def get_control_points_lineset(control_points: np.ndarray) -> o3d.geometry.LineSet:
        """Contruct a LineSet for visualizing control_points
        Control point order indices are shown below (symmetric=True is in parentheses)
                  * 0 (0)
                  |            y <---*
        1 (2) *-------* 2 (1)        |
              |       |              v
        3 (4) *       * 4 (3)         z
            left    right

        :param control_points: [batch_size, 5, 3] np.floating np.ndarray
        :return: o3d.geometry.LineSet for visualization
        """
        control_points = np.asanyarray(control_points).reshape(-1, 5, 3)
        batch_size = len(control_points)

        # Add mid point
        control_points = np.concatenate(
            [control_points[:, 1:3].mean(1, keepdims=True), control_points], axis=-2
        )
        # Now the order is:
        #           * 1 (1)
        #           |            y <---*
        # 2 (3) *---*---* 3 (2)        |
        #       |   0   |              v
        # 4 (5) *       * 5 (4)        z
        #     left    right

        lines = np.array([[0, 1], [2, 3], [2, 4], [3, 5]])
        lines = np.tile(lines, (batch_size, 1, 1))
        lines += np.arange(batch_size)[:, None, None] * 6

        return o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(control_points.reshape(-1, 3)),
            o3d.utility.Vector2iVector(lines.reshape(-1, 2)),
        )


class PandaGripper(Gripper):
    """An object representing a Franka Panda gripper."""

    def __init__(
        self,
        q=None,
        num_contact_points_per_finger=10,
        root_folder=Path(__file__).resolve().parent,
    ):
        """Create a Franka Panda parallel-yaw gripper object.

        Keyword Arguments:
            q {list of int} -- opening configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
        """
        mesh_dir = Path(root_folder) / "gripper_models/panda_gripper"
        self.control_pts_dir = Path(root_folder) / "gripper_control_points"

        self.joint_limits = [0.0, 0.04]
        self.root_folder = root_folder
        self.gripper_width = 0.08

        if q is None:
            q = self.joint_limits[-1]
        self.q = q

        # Load gripper meshes
        self.base = trimesh.load(mesh_dir / "hand_cgn.stl")
        self.finger_l: trimesh.Trimesh = trimesh.load(mesh_dir / "finger_cgn.stl")  # type: ignore
        self.finger_r: trimesh.Trimesh = self.finger_l.copy()  # type: ignore

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))  # type: ignore
        self.finger_l.apply_translation([+q, 0, 0.0584])  # type: ignore
        self.finger_r.apply_translation([-q, 0, 0.0584])  # type: ignore

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        # For visualization with Open3D
        self.hand_o3d = o3d.io.read_triangle_model(str(mesh_dir / "hand.glb"))
        # self.finger_l_o3d = o3d.io.read_triangle_model(str(mesh_dir / "finger.glb"))
        # self.finger_r_o3d = o3d.io.read_triangle_model(str(mesh_dir / "finger.glb"))
        # finger_pts are the 4 corners of the square finger contact pad
        # See franka_description/meshes/visual/finger.dae
        # self.finger_l_pts = np.array([
        #     [0.008763, 0.000011, 0.036266],
        #     [-0.008772, 0.000011, 0.036266],
        #     [-0.008772, 0.000011, 0.053746],
        #     [0.008763, 0.000011, 0.053746],
        # ])
        # self.finger_r_pts = self.finger_l_pts * [-1, -1, 1]  # invert xy coords
        # --- After conver hull --- #
        # self.hand_o3d = o3d.io.read_triangle_model(str(mesh_dir / "hand.stl"))
        self.finger_l_o3d = o3d.io.read_triangle_model(str(mesh_dir / "finger.stl"))
        self.finger_r_o3d = o3d.io.read_triangle_model(str(mesh_dir / "finger.stl"))
        # See franka_description/meshes/collision/finger.stl (after convex hull)
        self.finger_l_pts = np.array([
            [0.008693, -0.000133, 0.000147],
            [-0.008623, -0.000133, 0.000147],
            [-0.008623, -0.000133, 0.053725],
            [0.008693, -0.000133, 0.053725],
        ])
        self.finger_r_pts = self.finger_l_pts.copy()
        T_gripper_finger_l = np.eye(4)
        T_gripper_finger_l[:3, -1] = [0, 0, 0.0584]  # panda_finger_joint1
        self.finger_l_pts = transform_points(self.finger_l_pts, T_gripper_finger_l)
        self.finger_l_o3d = transform_geometry(self.finger_l_o3d, T_gripper_finger_l)
        T_gripper_finger_r = np.eye(4)
        T_gripper_finger_r[:3, :3] = euler2mat(0, 0, np.pi)  # invert xy coords
        T_gripper_finger_r[:3, -1] = [0, 0, 0.0584]  # panda_finger_joint2
        self.finger_r_pts = transform_points(self.finger_r_pts, T_gripper_finger_r)
        self.finger_r_o3d = transform_geometry(self.finger_r_o3d, T_gripper_finger_r)
        # Finger joint axes (frame orientation is after self.convert_grasp_poses())
        self.finger_l_joint_axis = np.array([0, 1.0, 0])
        self.finger_r_joint_axis = np.array([0, -1.0, 0])

        self.contact_ray_origins = []  # type: ignore
        self.contact_ray_directions = []  # type: ignore

        with (self.control_pts_dir / "panda_gripper_coords.pickle").open("rb") as f:
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

        self.contact_ray_origins: np.ndarray = np.array(self.contact_ray_origins)
        self.contact_ray_directions: np.ndarray = np.array(self.contact_ray_directions)

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
        self, batch_size: int, use_tf=True, symmetric=False, convex_hull=True
    ) -> np.ndarray | tf.Tensor:
        """
        Outputs a 5 point gripper representation of shape (batch_size x 5 x 3).

        :param batch_size: batch size
        :param use_tf: outputing a tf tensor instead of a numpy array (default: {True})
        :param symmetric: Output the symmetric control point configuration of
                          the gripper (default: {False})
        :param convex_hull: Return control points according to the convex hull
                            panda gripper model (default: {True})
        :return: control points of the panda gripper
        """

        control_points = np.load(self.control_pts_dir / "panda.npy")[:, :3]
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

    def convert_grasp_poses(
        self,
        pred_grasp_poses: np.ndarray | dict[int, np.ndarray],
        pred_gripper_openings: Optional[
            float | np.ndarray | dict[int, float | np.ndarray]
        ] = None,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[dict[int, np.ndarray], dict[int, np.ndarray]]
    ):
        """Converts grasp poses to align with URDF frame orientation
        which has approaching +z, baseline +y

        Gripper pose orientation is shown below
                  * gripper link origin
                  |            y <---*
              *-------*              |
              |       |              v
              *       *              z
            left    right

        All params and returns can be {seg_id: np.ndarray} or np.ndarray.

        :param pred_grasp_poses: CGN predicted gripper pose of panda_hand
                                 approaching +z, baseline +x
                                 [n_grasps, 4, 4] or [4, 4] np.floating np.ndarray
        :param pred_gripper_openings: CGN predicted gripper opening widths
                                      [n_grasps,] np.floating np.ndarray or float
                                      If None, use self.gripper_width.
        :return grasp_poses: converted gripper pose to align with URDF frame orientation
                             approaching +z, baseline +y
                             [n_grasps, 4, 4] np.floating np.ndarray
        :return q_vals: gripper joint value, [n_grasps,] np.floating np.ndarray
        """
        if is_grasp_poses_array := isinstance(pred_grasp_poses, np.ndarray):
            assert isinstance(pred_gripper_openings, (float, np.ndarray, type(None)))

            pred_grasp_poses = {1: pred_grasp_poses}
            if pred_gripper_openings is not None:
                pred_gripper_openings = {1: pred_gripper_openings}

        grasp_poses = {}
        q_vals = {}
        for seg_id, grasp_pose in pred_grasp_poses.items():
            grasp_pose = np.asarray([grasp_pose]).reshape(-1, 4, 4)
            num_grasps = len(grasp_pose)

            if pred_gripper_openings is None:
                gripper_opening = np.asarray([self.gripper_width] * num_grasps)
            else:
                gripper_opening = np.asarray([pred_gripper_openings[seg_id]]).reshape(  # type: ignore
                    -1
                )
            assert (
                len(gripper_opening) == num_grasps
            ), f"{gripper_opening.shape = } {grasp_pose.shape = }"

            # Convert gripper width to joint value
            q_vals[seg_id] = gripper_opening / 2

            # Convert gripper pose to align with URDF frame orientation
            T = np.eye(4)
            T[:3, :3] = euler2mat(0, 0, np.pi / 2)
            T = np.tile(T, (num_grasps, 1, 1))
            grasp_poses[seg_id] = grasp_pose @ T

        if is_grasp_poses_array:
            return grasp_poses[1], q_vals[1]

        return grasp_poses, q_vals

    def get_mesh(
        self, q: float = 0.0, pose: np.ndarray = np.eye(4)
    ) -> rendering.TriangleMeshModel:
        """Return gripper mesh at given q value and pose

        :param q: gripper finger joint value
        :param pose: gripper pose from world frame to panda_hand
        :return: gripper mesh, rendering.TriangleMeshModel
        """
        T = np.eye(4)
        T[:3, -1] = self.finger_l_joint_axis * q
        finger_l = transform_geometry(self.finger_l_o3d, T)

        T = np.eye(4)
        T[:3, -1] = self.finger_r_joint_axis * q
        finger_r = transform_geometry(self.finger_r_o3d, T)

        gripper = merge_geometries([self.hand_o3d, finger_l, finger_r])
        return transform_geometry(gripper, pose)

    def get_control_points(
        self,
        q: float | np.ndarray,
        pose: np.ndarray = np.eye(4),
        *,
        use_tf=False,
        symmetric=False,
    ) -> np.ndarray | tf.Tensor:
        """
        Return the 5 control points of gripper representation at given q value and pose.
        Control point order indices are shown below (symmetric=True is in parentheses)
                  * 0 (0)
                  |            y <---*
        1 (2) *-------* 2 (1)        |
              |       |              v
        3 (4) *       * 4 (3)         z
            left    right

        :param q: gripper finger joint value
        :param pose: gripper pose from world frame to panda_hand
        :return: [batch_size, 5, 3] np.floating np.ndarray
        """
        q = np.asarray([q]).reshape(-1)
        pose = np.asarray([pose]).reshape(-1, 4, 4)
        assert len(q) == len(pose), f"{q.shape = } {pose.shape = }"

        T = np.tile(np.eye(4), (len(q), 1, 1))
        T[..., :3, -1] = self.finger_l_joint_axis * q[:, None]
        finger_l_pts = transform_points_batch(
            self.finger_l_pts.reshape(2, 2, 3).mean(1), T
        )  # [B, 2, 3]

        T = np.tile(np.eye(4), (len(q), 1, 1))
        T[..., :3, -1] = self.finger_r_joint_axis * q[:, None]
        finger_r_pts = transform_points_batch(
            self.finger_r_pts.reshape(2, 2, 3).mean(1), T
        )  # [B, 2, 3]

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


class XArmGripper(Gripper):
    """An object representing a UFactory XArm gripper."""

    def __init__(self, root_folder=Path(__file__).resolve().parent):
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
        # See xarm_description/meshes/gripper/xarm/left_finger.STL
        self.finger_l_pts = np.array([
            [0.01475, -0.026003, 0.022253],
            [-0.01475, -0.026003, 0.022253],
            [-0.01475, -0.026003, 0.059753],
            [0.01475, -0.026003, 0.059753],
        ])
        self.finger_r_pts = self.finger_l_pts * [-1, -1, 1]  # invert xy coords
        T_gripper_finger_l = np.eye(4)
        T_gripper_finger_l[:3, -1] = [0, 0.02682323, 0.11348719]  # left_finger_joint
        self.finger_l_pts = transform_points(self.finger_l_pts, T_gripper_finger_l)
        self.finger_l = transform_geometry(self.finger_l, T_gripper_finger_l)
        self.finger_l_joint_axis = np.array([0, 0.96221329, -0.27229686])
        T_gripper_finger_r = np.eye(4)
        T_gripper_finger_r[:3, -1] = [0, -0.02682323, 0.11348719]  # right_finger_joint
        self.finger_r_pts = transform_points(self.finger_r_pts, T_gripper_finger_r)
        self.finger_r = transform_geometry(self.finger_r, T_gripper_finger_r)
        self.finger_r_joint_axis = np.array([0, -0.96221329, -0.27229686])

        self.gripper_opening_to_q_val = 2 * np.cos(
            np.arctan2(self.finger_l_joint_axis[2], self.finger_l_joint_axis[1])
        )
        self.gripper_width = 0.086

    def convert_grasp_poses(
        self,
        pred_grasp_poses: np.ndarray | dict[int, np.ndarray],
        pred_gripper_openings: Optional[
            float | np.ndarray | dict[int, float | np.ndarray]
        ] = None,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[dict[int, np.ndarray], dict[int, np.ndarray]]
    ):
        """Converts grasp poses to align with URDF frame orientation
        which has approaching +z, baseline +y.
        Also, converts grasp pose from panda gripper to xarm gripper.

        Gripper pose orientation is shown below
                  * gripper link origin
                  |                  *---> y
              *-------*              |
              |       |              v
              *       *              z
            right    left

        All params and returns can be {seg_id: np.ndarray} or np.ndarray.

        :param pred_grasp_poses: CGN predicted gripper pose of panda_hand
                                 approaching +z, baseline +x
                                 [n_grasps, 4, 4] or [4, 4] np.floating np.ndarray
        :param pred_gripper_openings: CGN predicted gripper opening widths
                                      [n_grasps,] np.floating np.ndarray or float
                                      If None, use self.gripper_width.
        :return grasp_poses: converted gripper pose to align with URDF frame orientation
                             approaching +z, baseline +y
                             [n_grasps, 4, 4] np.floating np.ndarray
        :return q_vals: gripper joint value, [n_grasps,] np.floating np.ndarray
        """
        # TODO: Switch to revolute gripper joints
        # FIXME: gripper finger_l/r_pts might be wrong (swapped left / right finger)
        if is_grasp_poses_array := isinstance(pred_grasp_poses, np.ndarray):
            assert isinstance(pred_gripper_openings, (np.ndarray, type(None)))

            pred_grasp_poses = {1: pred_grasp_poses}
            if pred_gripper_openings is not None:
                pred_gripper_openings = {1: pred_gripper_openings}

        xarm_obj_grasp_poses = {}
        xarm_obj_q_vals = {}
        for seg_id, grasp_poses in pred_grasp_poses.items():
            grasp_poses = np.asarray([grasp_poses]).reshape(-1, 4, 4)
            num_grasps = len(grasp_poses)

            if pred_gripper_openings is None:
                gripper_openings = np.asarray([self.gripper_width] * num_grasps)
            else:
                gripper_openings = np.asarray([pred_gripper_openings[seg_id]]).reshape(  # type: ignore
                    -1
                )
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

        if is_grasp_poses_array:
            return xarm_obj_grasp_poses[1], xarm_obj_q_vals[1]

        return xarm_obj_grasp_poses, xarm_obj_q_vals

    def get_mesh(
        self, q: float = 0.0, pose: np.ndarray = np.eye(4)
    ) -> rendering.TriangleMeshModel:
        """Return gripper mesh at given q value and pose

        :param q: gripper finger joint value
        :param pose: gripper pose from world frame to xarm_gripper_base_link
        :return: gripper mesh, rendering.TriangleMeshModel
        """
        # TODO: Switch to revolute gripper joints
        T = np.eye(4)
        T[:3, -1] = self.finger_l_joint_axis * q
        finger_l = transform_geometry(self.finger_l, T)

        T = np.eye(4)
        T[:3, -1] = self.finger_r_joint_axis * q
        finger_r = transform_geometry(self.finger_r, T)

        gripper = merge_geometries([self.camera, self.hand, finger_l, finger_r])
        return transform_geometry(gripper, pose)

    def get_control_points(
        self,
        q: float | np.ndarray,
        pose: np.ndarray = np.eye(4),
        *,
        use_tf=False,
        symmetric=False,
    ) -> np.ndarray | tf.Tensor:
        """
        Return the 5 control points of gripper representation at given q value and pose.
        Control point order indices are shown below (symmetric=True is in parentheses)
                  * 0 (0)
                  |                  *---> y
        2 (1) *-------* 1 (2)        |
              |       |              v
        4 (3) *       * 3 (4)         z
            right    left

        :param q: gripper finger joint value
        :param pose: gripper pose from world frame to xarm_gripper_base_link
        :return: [batch_size, 5, 3] np.floating np.ndarray
        """
        # TODO: Switch to revolute gripper joints
        # FIXME: gripper finger_l/r_pts might be wrong (swapped left / right finger)
        q = np.asarray([q]).reshape(-1)
        pose = np.asarray([pose]).reshape(-1, 4, 4)
        assert len(q) == len(pose), f"{q.shape = } {pose.shape = }"

        T = np.tile(np.eye(4), (len(q), 1, 1))
        T[..., :3, -1] = self.finger_l_joint_axis * q[:, None]
        finger_l_pts = transform_points_batch(
            self.finger_l_pts.reshape(2, 2, 3).mean(1), T
        )  # [B, 2, 3]

        T = np.tile(np.eye(4), (len(q), 1, 1))
        T[..., :3, -1] = self.finger_r_joint_axis * q[:, None]
        finger_r_pts = transform_points_batch(
            self.finger_r_pts.reshape(2, 2, 3).mean(1), T
        )  # [B, 2, 3]

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


def create_gripper(name, root_folder=Path(__file__).resolve().parent):
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
        return PandaGripper(root_folder=root_folder)
    elif name.lower() == "xarm":
        return XArmGripper(root_folder=root_folder)
    else:
        raise ValueError(f"Unknown gripper: {name}")
