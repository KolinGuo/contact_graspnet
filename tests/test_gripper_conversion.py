import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from real_robot.utils.logger import get_logger
from real_robot.utils.visualization import Visualizer, visualizer

from contact_graspnet import CGNGraspEstimator
from contact_graspnet.data import depth2xyz
from contact_graspnet.gripper import create_gripper

# ruff: noqa: F841


def download_test_data(output_dir: Path) -> None:
    # Use gdown to download data
    import gdown

    logger.info("Begin downloading test data...")
    ret = gdown.download_folder(
        id="1v0_QMTUIEOcu09Int5V6N2Nuq7UCtuAA", output=str(output_dir)
    )
    if ret is None:
        raise RuntimeError("Failed to download test data")
    logger.info("Finish downloading test data...")


def predict_grasps(
    self,
    depth: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    pcd: Optional[np.ndarray] = None,
    seg: Optional[np.ndarray] = None,
    *,
    local_regions: bool = True,
    filter_grasps: bool = True,
    seg_id: Optional[int] = None,
    z_clip_range: tuple[float, float] = (0.2, 1.8),
    skip_border_objects: bool = False,
    margin_px: int = 5,
) -> tuple[np.ndarray, ...] | tuple[dict[int, np.ndarray], ...]:
    """Predict 6-DoF grasps given depth or pcd input

    :param depth: [H, W] or [H, W, 1] np.floating/np.uint16 np.ndarray
    :param K: [3, 3] camera intrinsics matrix or [fx, fy, cx, cy]
    :param pcd: [N, 3] np.floating np.ndarray
    :param seg: bool/np.uint8 np.ndarray [H, W] for depth input [N,] for pcd input
    :param local_regions: Crop 3D local regions around given segments.
    :param filter_grasps: Filter and assign grasp contacts according to seg.
    :param seg_id: only return grasps from the specified seg_id.
    :param z_clip_range: z range to crop pcd outlier points. Default: (0.2, 1.8)m
    :param skip_border_objects: Whether to skip object near image border
                                to avoid artificial edge.
    :param margin_px: Pixel margin of skip_border_objects (default: {5})

    Returns:
        All returns are {seg_id: np.ndarray} or np.ndarray when given seg_id
    :return pred_grasps_cam: predicted grasp poses in camera frame.
                                [n_grasps, 4, 4] np.floating np.ndarray
    :return scores: predicted grasp pose confidence scores.
                    [n_grasps,] np.floating np.ndarray
    :return contact_pts: predicted grasp pose contact points in camera frame.
                            [n_grasps, 3] np.floating np.ndarray
    :return gripper_openings: predicted gripper opening widths.
                              [n_grasps,] np.floating np.ndarray
    """
    assert (depth is None) ^ (pcd is None), "Need one of depth/pcd input"
    if seg is None and (local_regions or filter_grasps or skip_border_objects):
        raise ValueError(
            "Need segmentation map to extract local regions, filter grasps "
            "or skip border objects"
        )
    if local_regions and not filter_grasps:
        self.logger.warning(
            "When cropping local_regions, it's better to also filter_grasps "
            "based on segmentation"
        )

    self.obs_dict = {}  # Empty input observation cache for visualization
    if depth is not None:  # depth input
        assert K is not None, "Need camera intrinsics for depth input"
        xyz_image = depth2xyz(depth, K)
        pcd = xyz_image.reshape(-1, 3)

        # Filter out border objects (set mask to background id=0)
        if skip_border_objects:
            for i in np.unique(seg[seg > 0]):  # type: ignore
                obj_mask = seg == i
                obj_y, obj_x = np.where(obj_mask)
                if (
                    np.any(obj_x < margin_px)
                    or np.any(obj_x > seg.shape[1] - margin_px)  # type: ignore
                    or np.any(obj_y < margin_px)
                    or np.any(obj_y > seg.shape[0] - margin_px)  # type: ignore
                ):
                    self.logger.info(
                        "Skipping object seg_id=%d: not entirely in image bounds", i
                    )
                    seg[obj_mask] = 0  # set as background id # type: ignore
        self.obs_dict["depth_image"] = depth
        self.obs_dict["K"] = K
        self.obs_dict["xyz_image"] = xyz_image
        if seg is not None:
            self.obs_dict["seg_mask"] = seg
    else:
        self.obs_dict["input_points"] = pcd

    # Filter z range for outlier points
    z_range_mask = (pcd[:, 2] >= z_clip_range[0]) & (pcd[:, 2] <= z_clip_range[1])  # type: ignore
    pcd = pcd[z_range_mask]  # type: ignore
    self.logger.info("After z_range filtering, %d points remain", pcd.shape[0])

    # Extract instance point clouds from segmap and depth map
    pc_segments = {}
    if seg is not None:
        self.logger.info("Predict on segmented pointcloud")

        seg = seg.reshape(-1)[z_range_mask]
        if seg.dtype == bool:  # when given binary mask, return grasps for True
            seg_id = True
        obj_instances = [seg_id] if seg_id is not None else np.unique(seg[seg > 0])
        for i in obj_instances:
            pc_segments[i] = pcd[seg == i]
            self.logger.info("Segment id=%d: %d points", i, pc_segments[i].shape[0])
    else:
        self.logger.info("Predict on full pointcloud")
        self.logger.info(
            "This will oversample/downsample to %d points",
            self.grasp_estimator._contact_grasp_cfg["DATA"]["raw_num_points"],
        )

    pred_grasps_cam, scores, contact_pts, gripper_openings = (
        self.grasp_estimator.predict_scene_grasps(
            self.tf_sess,
            pcd,
            pc_segments=pc_segments,
            local_regions=local_regions,
            filter_grasps=filter_grasps,
            forward_passes=self.forward_passes,
            logger=self.logger,
        )
    )

    if seg_id is not None:
        pred_grasps_cam = pred_grasps_cam[seg_id]
        scores = scores[seg_id]
        contact_pts = contact_pts[seg_id]
        gripper_openings = gripper_openings[seg_id]

    return pred_grasps_cam, scores, contact_pts, gripper_openings  # type: ignore


def main(data, args) -> None:
    grippers = {
        gripper_type: create_gripper(gripper_type)
        for gripper_type in ["panda", "xarm", "mycobot"]
    }

    grasp_estimator = CGNGraspEstimator(
        "hor_sigma_0025", forward_passes=args.forward_passes
    )
    pred_grasps_cam, scores, contact_pts, gripper_openings = predict_grasps(
        grasp_estimator,
        depth=data["depth"],
        seg=data["seg"],
        K=data["K"],
        local_regions=args.no_local_regions,
        filter_grasps=args.no_filter_grasps,
        seg_id=args.seg_id,
        z_clip_range=args.z_range,
        skip_border_objects=args.skip_border_objects,
        margin_px=args.margin_px,
    )

    vis = Visualizer()

    for gripper_type, gripper in grippers.items():
        logger.info("Converting q_vals for gripper type: %s", gripper_type)
        pred_gripper_grasps_cam, q_vals = gripper.convert_grasp_poses(
            pred_grasps_cam,
            gripper_openings,  # type: ignore
        )

        grasp_estimator.gripper = gripper
        grasp_estimator.obs_dict["K"] = data["K"]  # K will be poped in visualize_grasps
        vis: Visualizer = grasp_estimator.visualize_grasps(
            pred_gripper_grasps_cam,
            scores,
            q_vals,
            cam_pose=np.eye(4),
            rgb_image=data["rgb"][..., ::-1],  # BGR -> RGB
            group_prefix=f"CGN/{gripper_type}",
            vis=vis,
            pause_render=False,
        )

    visualizer.pause_render = True  # Pause the Visualizer
    vis.render()  # render once

    vis.close()


if __name__ == "__main__":
    logger = get_logger("CGN")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--np_path",
        default="test_data/7.npy",
        help=(
            'Input data: npz/npy file with keys either "depth" & camera matrix "K" or '
            'just point cloud "pc" in meters. Optionally, a 2D "segmap"'
        ),
    )
    parser.add_argument(
        "--forward_passes",
        type=int,
        default=1,
        help=(
            "Run multiple parallel forward passes to mesh_utils more potential "
            "contact points."
        ),
    )
    parser.add_argument(
        "--no_local_regions",
        action="store_false",
        help="Do not crop 3D local regions around given segments.",
    )
    parser.add_argument(
        "--no_filter_grasps",
        action="store_false",
        help="Do not filter grasp contacts according to segmap.",
    )
    parser.add_argument(
        "--seg_id",
        type=int,
        default=None,
        help="Only return grasps of the given object id",
    )
    parser.add_argument(
        "--z_range",
        default=[0.2, 1.8],
        help="Z value threshold to crop the input point cloud",
    )
    parser.add_argument(
        "--skip_border_objects",
        action="store_true",
        default=False,
        help="When extracting local_regions, ignore segments at depth map boundary.",
    )
    parser.add_argument(
        "--margin_px",
        type=int,
        default=5,
        help="Pixel margin of skip_border_objects (default: {5})",
    )
    args = parser.parse_args()

    test_data_dir = Path(__file__).resolve().parents[1] / "test_data"
    if not test_data_dir.is_dir():
        logger.info("No test_data found locally.")
        download_test_data(test_data_dir)

    np_path = Path(args.np_path)
    logger.info("Loading from %s", np_path)
    assert np_path.is_file(), f"{np_path} does not exist!"

    data = np.load(np_path, allow_pickle=True).item()

    main(data, args)
