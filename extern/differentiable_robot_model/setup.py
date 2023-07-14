# Copyright (c) Facebook, Inc. and its affiliates.
######################################################################
# \file setup.py
# \author Franziska Meier
#######################################################################
"""Installation for the differentiable-robot-model project."""

import pathlib
import os
import subprocess
from setuptools import setup, find_packages
from glob import glob

package_name = "differentiable_robot_model"

# dependencies
install_requires = [
    "pyquaternion >= 0.9.9",
    "hydra-core >= 1.0.3",
    "urdf_parser_py >= 0.0.3",
    "Sphinx >= 3.5.4",
    "recommonmark >= 0.7.1",
]

# run setup
setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=install_requires,
    zip_safe=True,
    maintainer="Vasileios Vasilopoulos",
    maintainer_email="vasileios.v@samsung.com",
    description="Differentiable robot model",
    author="Franziska Meier",
    author_email="fmeier@fb.com",
    url="https://github.com/facebookresearch/differentiable-robot-model",
    keywords="robotics, differentiable, optimization",
    license="MIT",
)
