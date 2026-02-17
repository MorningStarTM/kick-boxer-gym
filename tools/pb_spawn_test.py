import argparse
import time
from pathlib import Path

import pybullet as p
import pybullet_data


def print_joint_info(body_id: int):
    n = p.getNumJoints(body_id)
    print(f"\nHumanoid joint count: {n}\n" + "-" * 60)
    for i in range(n):
        info = p.getJointInfo(body_id, i)
        joint_name = info[1].decode("utf-8")
        parent_link = info[12].decode("utf-8")
        child_link = info[0]
        joint_type = info[2]
        lower = info[8]
        upper = info[9]
        axis = info[13]
        print(
            f"[{i:02d}] {joint_name:28s} "
            f"type={joint_type} limits=({lower:.2f},{upper:.2f}) axis={axis} parent={parent_link}"
        )
    print("-" * 60 + "\n")


def disable_default_motors(body_id: int):
    """Important if you later do torque control."""
    for j in range(p.getNumJoints(body_id)):
        p.setJointMotorControl2(
            bodyUniqueId=body_id,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            force=0.0,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    ap.add_argument("--steps", type=int, default=2400, help="Sim steps to run")
    ap.add_argument("--dt", type=float, default=1.0 / 240.0, help="Time step")
    args = ap.parse_args()

    client = p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(args.dt)

    # Plane
    p.loadURDF("plane.urdf")

    # URDF path (repo-relative)
    repo_root = Path(__file__).resolve().parents[1]
    urdf_path = repo_root / "src" / "kick_boxer_gym" / "assets" / "urdf" / "simple_human.urdf"

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Spawn slightly above ground
    start_pos = [0.0, 0.0, 1.3]
    start_orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

    humanoid_id = p.loadURDF(
        str(urdf_path),
        basePosition=start_pos,
        baseOrientation=start_orn,
        useFixedBase=False,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    # Print joints
    print_joint_info(humanoid_id)

    # Optional: disable motors (safe baseline)
    disable_default_motors(humanoid_id)

    # Run sim
    for step in range(args.steps):
        p.stepSimulation()

        if args.gui:
            time.sleep(args.dt)

        # simple stability check: if NaNs / huge positions, break
        pos, orn = p.getBasePositionAndOrientation(humanoid_id)
        if abs(pos[0]) > 50 or abs(pos[1]) > 50 or pos[2] < -5:
            print(f"⚠️ Unstable: base position looks wrong at step {step}: {pos}")
            break

    if args.gui:
        print("Press Ctrl+C to exit GUI.")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass

    p.disconnect(client)


if __name__ == "__main__":
    main()
