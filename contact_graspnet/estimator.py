from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow.compat.v1 as tf
from real_robot.utils.logger import get_logger
from real_robot.utils.visualization import O3DGUIVisualizer

from contact_graspnet import config_utils
from contact_graspnet.contact_grasp_estimator import GraspEstimator
from contact_graspnet.data import depth2xyz
from contact_graspnet.mesh_utils import XArmGripper
from contact_graspnet.utils import timer

tf.disable_eager_execution()
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


class CGNGraspEstimator:
    """Contact-GraspNet Grasp Estimator wrapper"""

    CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"

    CGN_CKPT_DIRS = {
        "train_test": CKPT_DIR / "contact_graspnet_train_and_test",
        "rad2_32": CKPT_DIR / "scene_2048_bs3_rad2_32",
        "hor_sigma_001": CKPT_DIR / "scene_test_2048_bs3_hor_sigma_001",
        "hor_sigma_0025": CKPT_DIR / "scene_test_2048_bs3_hor_sigma_0025",
    }

    CGN_CKPT_GDOWN_IDS = {
        "train_test": "1aHlRwSq4WJ7bzutASKOFUAgVlpKuO9MH",
        "rad2_32": "1YpWo-xr1jGMmE2dVNM8bn1iRsjNS1dV1",
        "hor_sigma_001": "1vcHNeKgMpHpeWKggkRdWmMr4-ymEPQx8",
        "hor_sigma_0025": "1r5leQmnbJP2kLOB-I4wX_jdZuW5L_MTZ",
    }

    logger = get_logger("CGN")

    def __init__(
        self,
        cgn_model_variant: str = "hor_sigma_0025",
        forward_passes: int = 1,
        arg_configs: Optional[list[str]] = None,
        save_dir: str = "results",
        gripper_type: str = "xarm",
        device: str = "cuda",
    ):
        """
        :param forward_passes: Number of forward passes to run on each point cloud.
                               Same as batch_size
        """
        self.logger.info('Using CGN model variant: "%s"', cgn_model_variant)

        self.ckpt_dir = self.CGN_CKPT_DIRS[cgn_model_variant]
        self.forward_passes = forward_passes
        if arg_configs is None:
            arg_configs = []
        self.save_dir = Path(save_dir)
        self.device = device
        assert device == "cuda", "Using GPU other than cuda:0 is not implemented yet"

        if not self.ckpt_dir.is_dir():
            self.logger.info("No checkpoint found locally.")
            self.download_model_checkpoint(cgn_model_variant)

        self.config = config_utils.load_config(
            self.ckpt_dir, batch_size=self.forward_passes, arg_configs=arg_configs
        )

        # Load CGN model
        self.load_cgn_model()

        self.gripper_type = gripper_type
        if self.gripper_type == "xarm":
            self.gripper = XArmGripper(
                Path(__file__).resolve().parent / "contact_graspnet"
            )

    def download_model_checkpoint(self, model_variant: str) -> None:
        import gdown

        self.logger.info("Begin downloading model checkpoint...")
        ret = gdown.download_folder(
            id=self.CGN_CKPT_GDOWN_IDS[model_variant], output=str(self.ckpt_dir)
        )
        if ret is None:
            raise RuntimeError(f'Failed to download model "{model_variant}"')
        self.logger.info("Finish downloading model checkpoint...")

    @timer
    def load_cgn_model(self) -> None:
        self.grasp_estimator = GraspEstimator(self.config)
        self.grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        self.tf_saver = tf.train.Saver(save_relative_paths=True)

        # Create a session (limit total memory to 6 GiB)
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        total_memory_gib = int(pynvml.nvmlDeviceGetMemoryInfo(h).total) / 1024**3

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 6.0 / total_memory_gib
        config.allow_soft_placement = True
        self.tf_sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(
            self.tf_sess, self.tf_saver, self.ckpt_dir, mode="test"
        )

    @timer
    def predict_grasps(
        self,
        depth: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
        pcd: Optional[np.ndarray] = None,
        seg: Optional[np.ndarray] = None,
        local_regions: bool = True,
        filter_grasps: bool = True,
        seg_id: Optional[int] = None,
        z_clip_range: tuple[float, float] = (0.2, 1.8),
        skip_border_objects: bool = False,
        margin_px: int = 5,
        save: bool = False,
    ) -> tuple[np.ndarray, ...] | tuple[dict[int, np.ndarray], ...]:
        """Predict 6-DoF grasps given depth or pcd input

        :param depth: [H, W] or [H, W, 1] np.floating np.ndarray
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
        :param save: Whether to save predicted results as npz

        Returns:
            All returns are {seg_id: np.ndarray} or np.ndarray when given seg_id
        :return pred_grasps_cam: predicted grasp poses in camera frame.
                                 [n_grasps, 4, 4] np.floating np.ndarray
        :return scores: predicted grasp pose confidence scores.
                        [n_grasps,] np.floating np.ndarray
        :return contact_pts: predicted grasp pose contact points in camera frame.
                             [n_grasps, 3] np.floating np.ndarray
        :return q_vals: predicted gripper joint values
                        [n_grasps,] np.floating np.ndarray
        """
        assert (depth is None) ^ (pcd is None), "Need one of depth/pcd input"
        if seg is None and (local_regions or filter_grasps):
            raise ValueError(
                "Need segmentation map to extract local regions or filter grasps"
            )

        if depth is not None:  # depth input
            assert K is not None, "Need camera intrinsics for depth input"
            xyz_image = depth2xyz(depth, K, depth_scale=1.0)
            pcd = xyz_image.reshape(-1, 3)

            # Filter out border objects (set mask to background id=0)
            if seg is not None and skip_border_objects:
                for i in np.unique(seg[seg > 0]):
                    obj_mask = seg == i
                    obj_y, obj_x = np.where(obj_mask)
                    if (
                        np.any(obj_x < margin_px)
                        or np.any(obj_x > seg.shape[1] - margin_px)
                        or np.any(obj_y < margin_px)
                        or np.any(obj_y > seg.shape[0] - margin_px)
                    ):
                        print(f"object {i} not entirely in image bounds, skipping")
                        seg[obj_mask] = 0

        # Threshold distance
        z_range_mask = (pcd[:, 2] < z_clip_range[1]) & (pcd[:, 2] > z_clip_range[0])  # type: ignore
        pcd = pcd[z_range_mask]  # type: ignore

        # Extract instance point clouds from segmap and depth map
        pc_segments = {}
        if seg is not None:
            seg = seg.reshape(-1)[z_range_mask]
            if seg.dtype == bool:  # when given binary mask
                seg_id = True
            obj_instances = [seg_id] if seg_id is not None else np.unique(seg[seg > 0])
            for i in obj_instances:
                pc_segments[i] = pcd[seg == i]

        pred_grasps_cam, scores, contact_pts, gripper_openings = (
            self.grasp_estimator.predict_scene_grasps(
                self.tf_sess,
                pcd,
                pc_segments=pc_segments,
                local_regions=local_regions,
                filter_grasps=filter_grasps,
                forward_passes=self.forward_passes,
            )
        )

        if self.gripper_type == "xarm":
            pred_grasps_cam, q_vals = self.gripper.convert_grasp_poses_from_panda(
                pred_grasps_cam, gripper_openings
            )
        else:
            q_vals = {
                seg_id: openings / 2 for seg_id, openings in gripper_openings.items()
            }

        # Save results
        if save:
            self.save_dir.mkdir(parents=True, exist_ok=True)

            np.savez(
                self.save_dir / "predictions.npz",
                pred_grasps_cam=pred_grasps_cam,  # type: ignore
                scores=scores,  # type: ignore
                contact_pts=contact_pts,  # type: ignore
                q_vals=q_vals,  # type: ignore
            )

        if seg_id is not None:
            pred_grasps_cam = pred_grasps_cam[seg_id]
            scores = scores[seg_id]
            contact_pts = contact_pts[seg_id]
            q_vals = q_vals[seg_id]

        return pred_grasps_cam, scores, contact_pts, q_vals

    def visualize_grasps(
        self,
        vis: O3DGUIVisualizer,
        pred_grasps_cam: np.ndarray | dict[int, np.ndarray],
        scores: np.ndarray | dict[int, np.ndarray],
        q_vals: np.ndarray | dict[int, np.ndarray],
        cam_pose: Optional[np.ndarray] = None,
        group_prefix="CGN",
    ) -> None:
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
        if cam_pose is None:
            cam_pose = np.eye(4)

        self.gripper.visualize_grasps(
            vis, pred_grasps_cam, scores, q_vals, cam_pose, group_prefix
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--ckpt_dir",
    #     default="checkpoints/scene_test_2048_bs3_hor_sigma_001",
    #     help="Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]",
    # )
    parser.add_argument(
        "--np_path",
        default="test_data/7.npy",
        help=(
            'Input data: npz/npy file with keys either "depth" & camera matrix "K" or '
            'just point cloud "pc" in meters. Optionally, a 2D "segmap"'
        ),
    )
    parser.add_argument(
        "--png_path", default="", help="Input data: depth map png in meters"
    )
    parser.add_argument(
        "--K",
        default=None,
        help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"',
    )
    parser.add_argument(
        "--z_range",
        default=[0.2, 1.8],
        help="Z value threshold to crop the input point cloud",
    )
    parser.add_argument(
        "--local_regions",
        action="store_true",
        default=False,
        help="Crop 3D local regions around given segments.",
    )
    parser.add_argument(
        "--filter_grasps",
        action="store_true",
        default=False,
        help="Filter grasp contacts according to segmap.",
    )
    parser.add_argument(
        "--skip_border_objects",
        action="store_true",
        default=False,
        help="When extracting local_regions, ignore segments at depth map boundary.",
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
        "--segmap_id",
        type=int,
        default=None,
        help="Only return grasps of the given object id",
    )
    parser.add_argument(
        "--arg_configs",
        nargs="*",
        type=str,
        default=[],
        help="overwrite config parameters",
    )
    FLAGS = parser.parse_args()

    grasp_estimator = CGNGraspEstimator(
        forward_passes=FLAGS.forward_passes,
        arg_configs=FLAGS.arg_configs,
    )

    from contact_graspnet.data import load_available_input_data

    input_paths = FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path
    input_paths = Path(input_paths)

    # Process example test scenes
    for p in [input_paths] if input_paths.is_file() else input_paths.iterdir():
        print(f"Loading {p}")

        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(
            str(p), K=FLAGS.K
        )

        grasp_estimator.predict_grasps(
            depth,
            pc_full,
            segmap,
            cam_K,
            local_regions=FLAGS.local_regions,
            filter_grasps=FLAGS.filter_grasps,
            seg_id=FLAGS.segmap_id,
            z_clip_range=eval(str(FLAGS.z_range)),
            skip_border_objects=FLAGS.skip_border_objects,
            save=True,
        )

    # inference(
    #     global_config,
    #     FLAGS.ckpt_dir,
    #     FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path,
    #     z_range=eval(str(FLAGS.z_range)),
    #     K=FLAGS.K,
    #     local_regions=FLAGS.local_regions,
    #     filter_grasps=FLAGS.filter_grasps,
    #     segmap_id=FLAGS.segmap_id,
    #     forward_passes=FLAGS.forward_passes,
    #     skip_border_objects=FLAGS.skip_border_objects,
    # )
