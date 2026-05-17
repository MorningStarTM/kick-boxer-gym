"""
Quick visualization of the SMPL-X humanoid URDF in PyBullet GUI.

Usage:
    python scripts/visualize_urdf.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pybullet as p
import pybullet_data


def main():
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=client)

    p.loadURDF("plane.urdf", physicsClientId=client)

    urdf_path = str(Path(__file__).parent.parent / "models" / "urdf" / "humanoid_smplx.urdf")
    humanoid = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0.95],
        useFixedBase=False,
        physicsClientId=client,
    )

    n_joints = p.getNumJoints(humanoid, physicsClientId=client)
    print(f"Loaded humanoid with {n_joints} joints:")
    for i in range(n_joints):
        info = p.getJointInfo(humanoid, i, physicsClientId=client)
        name = info[1].decode("utf-8")
        joint_type = ["revolute", "prismatic", "spherical", "planar", "fixed"][info[2]]
        link_name = info[12].decode("utf-8")
        print(f"  Joint {i}: {name} ({joint_type}) -> link: {link_name}")

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0.8],
    )

    print("\nSimulating... Press Ctrl+C to exit.")
    try:
        while True:
            p.stepSimulation(physicsClientId=client)
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()
