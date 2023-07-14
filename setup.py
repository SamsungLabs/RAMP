import os
import sys
from setuptools import setup, find_packages

root_dir = os.path.dirname(os.path.realpath(__file__))

requirements_default = set(
    ["numpy",
    f"isaacgym @ file://localhost{root_dir}/extern/isaacgym/python", 
    f"csdf @ file://localhost{root_dir}/extern/csdf",
    f"differentiable_robot_model @ file://localhost{root_dir}/extern/differentiable_robot_model",     
    "KNN-CUDA @ https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl",
    ]
)

# requirements_isaacgym = set(
#     [
#     f"isaacgym @ file://localhost{root_dir}/extern/isaacgym/python",  # Isaac Gym Simulation
#     ]
# )
setup(
    name='simenv',
    version='0.1',
    description='RAMP simulation environment',
    author='Vasileios Vasilopoulos, Jinwook Huh',
    author_email='vasileios.v@samsung.com',
    packages=find_packages(),
    install_requires=list(requirements_default),
    extras_require={
    # "issacgym": list(requirements_isaacgym),
},
    entry_points={
        # add your console scripts and other executables here
    },
)