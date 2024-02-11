from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="contact_graspnet",
        packages=find_packages(include="contact_graspnet*"),
        package_dir={"contact_graspnet": "contact_graspnet"},
        package_data={
            "contact_graspnet.pointnet2.tf_ops": ["**/*.so"],
            "contact_graspnet": ["config.yaml"],
            "contact_graspnet.gripper_control_points": [
                "panda.npy",
                "panda_gripper_coords.pickle",
                "panda_gripper_coords.yml",
            ],
            "contact_graspnet.gripper_models": ["**/*"],
        },
    )
