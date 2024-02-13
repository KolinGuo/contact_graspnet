from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d.visualization.gui as gui  # type: ignore
import tensorflow.compat.v1 as tf  # type: ignore
from real_robot.utils.logger import get_logger
from real_robot.utils.visualization import Visualizer, visualizer
from real_robot.utils.visualization.utils import _palette

from contact_graspnet import config_utils
from contact_graspnet.contact_grasp_estimator import GraspEstimator
from contact_graspnet.data import depth2xyz
from contact_graspnet.gripper import create_gripper
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

    SUPPORTED_GRIPPER_TYPE = ("panda", "xarm")

    logger = get_logger("CGN")

    def __init__(
        self,
        cgn_model_variant: str = "hor_sigma_0025",
        forward_passes: int = 1,
        arg_configs: Optional[list[str]] = None,
        save_dir: str = "results",
        gripper_type: str = "panda",
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

        # Create gripper for pose conversion and visualization
        self.gripper_type = gripper_type
        if gripper_type not in self.SUPPORTED_GRIPPER_TYPE:
            self.logger.critical(
                "Gripper type should be: %s", self.SUPPORTED_GRIPPER_TYPE
            )
            raise ValueError(f"Unknown gripper type: {gripper_type}")
        self.gripper = create_gripper(gripper_type)

        # Cache observation input for visualization
        self.obs_dict = {}

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
        # FIXME: lots of tf warnings
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
            self.tf_sess,
            self.tf_saver,
            self.ckpt_dir,
            mode="test",
            logger=self.logger,
        )

    @timer
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
        save: bool = False,
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

        self.logger.info("Converting q_vals for gripper type: %s", self.gripper_type)
        pred_grasps_cam, q_vals = self.gripper.convert_grasp_poses(
            pred_grasps_cam, gripper_openings
        )

        # Save results
        if save:
            self.save_dir.mkdir(parents=True, exist_ok=True)

            save_path = self.save_dir / "predictions.npz"
            self.logger.info("Saving results to: %s", save_path)

            np.savez(
                save_path,
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

        return pred_grasps_cam, scores, contact_pts, q_vals  # type: ignore

    def visualize_grasps(
        self,
        pred_grasps_cam: np.ndarray | dict[int, np.ndarray],
        scores: np.ndarray | dict[int, np.ndarray],
        q_vals: np.ndarray | dict[int, np.ndarray],
        *,
        cam_pose: np.ndarray = np.eye(4),
        rgb_image: Optional[np.ndarray] = None,
        group_prefix="CGN",
        vis: Optional[Visualizer] = None,
    ) -> Visualizer:
        """Visualize grasps using Visualizer

        :param pred_grasps_cam: predicted grasp poses in camera frame.
                                [n_grasps, 4, 4] np.floating np.ndarray
        :param scores: predicted grasp pose confidence scores.
                       [n_grasps,] np.floating np.ndarray
        :param q_vals: predicted gripper joint values
                       [n_grasps,] np.floating np.ndarray
        :param cam_pose: camera pose in world frame, [4, 4] np.floating np.ndarray
        :param rgb_image: RGB image for visualization, [H, W, 3] np.uint8 np.ndarray
        :param group_prefix: prefix for visualization geometry name
        :param vis: Visualizer instance
        :return: Visualizer instance
        """
        if vis is None:
            vis = Visualizer()
        o3d_vis = vis.o3dvis

        if rgb_image is not None:
            self.obs_dict["color_image"] = rgb_image
        # Draw camera lineset and frame
        if (depth_image := self.obs_dict.get("depth_image")) is not None:
            o3d_vis.add_camera(
                f"{group_prefix}_camera",
                *depth_image.shape[1::-1],
                self.obs_dict.pop("K"),  # type: ignore
                cam_pose,  # type: ignore
                fmt="CV",
            )
        # Show depth_image / color_image / segmentation masks / point cloud
        vis.show_obs({
            f"{group_prefix}/{k}": v for k, v in sorted(self.obs_dict.items())
        })

        if isinstance(pred_grasps_cam, np.ndarray):
            assert isinstance(scores, np.ndarray)
            assert isinstance(q_vals, np.ndarray)
            pred_grasps_cam = {1: pred_grasps_cam}
            scores = {1: scores}
            q_vals = {1: q_vals}

        # Show predicted grasp pose using gripper mesh and lineset
        for seg_id, _pred_grasps_cam in pred_grasps_cam.items():
            if len(_pred_grasps_cam) == 0:
                self.logger.warning("No predicted grasps for seg_id=%d", seg_id)
                continue

            _q_vals = q_vals[seg_id]
            max_score_idx = scores[seg_id].argmax()

            pred_grasps_world = cam_pose @ _pred_grasps_cam

            seg_id = int(seg_id)
            o3d_vis.add_geometry(
                f"{group_prefix}/obj_{seg_id}/mesh",
                self.gripper.get_mesh(
                    _q_vals[max_score_idx], pred_grasps_world[max_score_idx]
                ),
            )

            control_points = self.gripper.get_control_points(_q_vals, pred_grasps_world)
            control_points_geometry_name = f"{group_prefix}/obj_{seg_id}/pts"
            o3d_vis.add_geometry(
                control_points_geometry_name,
                self.gripper.get_control_points_lineset(control_points),
                show=False,
            )

            # Apply the same color to control points lineset as the mask segmentation
            node = o3d_vis.geometries[control_points_geometry_name]
            node.mat_color = gui.Color(
                *np.asarray(_palette[seg_id * 3 : (seg_id + 1) * 3]) / 255.0, 1.0
            )
            current_selected_item = o3d_vis._geometry_tree.selected_item
            o3d_vis._on_geometry_tree(node.id)
            # Update geometry material using o3d_vis.settings.material
            o3d_vis._scene.scene.modify_geometry_material(
                node.name, o3d_vis.settings.material
            )
            o3d_vis._on_geometry_tree(current_selected_item)

        # Set o3d_vis focused camera
        if "depth_image" in self.obs_dict:
            o3d_vis.set_focused_camera(f"{group_prefix}_camera")

        self.logger.info(
            "The Visualizer will be paused. Press the 'Single Step' button "
            "on the top right corner of the Open3D window to continue."
        )
        visualizer.pause_render = True  # Pause the Visualizer
        vis.render()  # render once
        return vis


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
