"""Helper classes and functions to sample grasps for a given object mesh."""

from __future__ import annotations

from copy import deepcopy
from typing import Union

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering  # type: ignore
import trimesh
from tqdm import tqdm


class Object:
    """Represents a graspable object."""

    def __init__(self, filename):
        """Constructor.

        :param filename: Mesh to load
        :param scale: Scaling factor
        """
        self.mesh: trimesh.Trimesh = trimesh.load(filename)  # type: ignore
        self.scale = 1.0

        # print(filename)
        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)  # type: ignore

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
    from .gripper import create_gripper

    manager = trimesh.collision.CollisionManager()
    manager.add_object("object", object_mesh)
    gripper_meshes = [create_gripper(gripper_name).hand]
    min_distance = []
    for tf in tqdm(gripper_transforms, disable=silent):
        min_distance.append(  # noqa: PERF401
            np.min([  # pyright: ignore
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
    from .gripper import create_gripper

    res = []
    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:  # type: ignore
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(  # type: ignore
            object_mesh, scale_to_box=True
        )
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)  # type: ignore
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
                    # dot_prods = (
                    #     contact_dict["contact_face_normals"]
                    #     * contact_dict["contact_directions"]
                    # ).sum(axis=1)
                    # contact_dict["contact_cosine_angles"] = np.cos(dot_prods)
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
ANY_O3D_GEOMETRY = Union[O3D_GEOMETRIES]  # type: ignore


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


O3D_GEOMETRY_LIST = Union[tuple(list[t] for t in O3D_GEOMETRIES)]  # type: ignore


def merge_geometries(geometries: O3D_GEOMETRY_LIST) -> ANY_O3D_GEOMETRY:
    """Merge a list of o3d geometries, must be of same type"""
    geometry_types = {type(geometry) for geometry in geometries}
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
