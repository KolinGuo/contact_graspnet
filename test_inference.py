import argparse
from pathlib import Path

import numpy as np
from real_robot.utils.logger import get_logger
from real_robot.utils.visualization import Visualizer

from contact_graspnet import CGNGraspEstimator

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


def main(data, args) -> None:
    grasp_estimator = CGNGraspEstimator(
        "hor_sigma_0025", forward_passes=args.forward_passes, gripper_type="panda"
    )
    pred_grasps_cam, scores, contact_pts, q_vals = grasp_estimator.predict_grasps(
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
    vis: Visualizer = grasp_estimator.visualize_grasps(
        pred_grasps_cam,
        scores,
        q_vals,
        cam_pose=np.eye(4),
        rgb_image=data["rgb"][..., ::-1],  # BGR -> RGB
    )


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

    test_data_dir = Path(__file__).resolve().parent / "test_data"
    if not test_data_dir.is_dir():
        logger.info("No test_data found locally.")
        download_test_data(test_data_dir)

    np_path = Path(args.np_path)
    logger.info("Loading from %s", np_path)
    assert np_path.is_file(), f"{np_path} does not exist!"

    data = np.load(np_path, allow_pickle=True).item()

    main(data, args)
