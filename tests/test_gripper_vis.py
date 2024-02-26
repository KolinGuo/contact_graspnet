import numpy as np
from real_robot.utils.visualization import Visualizer, visualizer

from contact_graspnet.gripper import create_gripper

if __name__ == "__main__":
    vis = Visualizer()
    o3d_vis = vis.o3dvis

    gripper = create_gripper("mycobot")

    for q in [
        gripper.joint_limits[0],
        np.mean(gripper.joint_limits),
        gripper.joint_limits[1],
    ]:
        o3d_vis.add_geometry(f"CGN/{q}/mesh", gripper.get_mesh(q, np.eye(4)))
        o3d_vis.add_geometry(
            f"CGN/{q}/pts",
            gripper.get_control_points_lineset(
                gripper.get_control_points(q, np.eye(4))
            ),
        )

    visualizer.pause_render = True  # Pause the Visualizer
    vis.render()  # render once

    vis.close()
