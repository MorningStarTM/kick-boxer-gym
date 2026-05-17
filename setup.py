from setuptools import setup, find_packages

setup(
    name="kick-boxing-rl",
    version="0.1.0",
    description="RL agent for kick-boxing with SMPL-X humanoid in PyBullet",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "pybullet>=3.2.5",
        "stable-baselines3>=2.1.0",
        "smplx>=0.1.28",
        "trimesh>=4.0.0",
        "tensorboard>=2.14.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
)
