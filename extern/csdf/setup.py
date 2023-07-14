import os
from setuptools import setup, find_packages
from glob import glob

package_name = "csdf"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "models"),
            glob("resource/models/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Vasileios Vasilopoulos",
    maintainer_email="vasileios.v@samsung.com",
    description="C-SDF computation for Panda robot",
    license="TODO: License declaration",
    tests_require=["pytest"],
)
