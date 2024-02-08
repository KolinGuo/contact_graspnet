from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="contact_graspnet",
        packages=find_packages(include="contact_graspnet*"),
        package_data={
            "contact_graspnet.pointnet2.tf_ops": ["**/*.so"],
            "contact_graspnet": ["config.yaml"],
        },
    )
