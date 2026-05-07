import argparse
import math
import random
import time
from pathlib import Path

import pybullet as p
import pybullet_data

URDF_DIR = Path(__file__).resolve().parents[1] / "src" / "kick_boxer_gym" / "assets" / "urdf"


def print_joint_info(body_id: int) -> None:
    n = p.getNumJoints(body_id)
    print(f"\nJoint count: {n}\n" + "-" * 66)
    for i in range(n):
        info = p.getJointInfo(body_id, i)
        name = info[1].decode()
        parent = info[12].decode()
        jtype = info[2]
        lo, hi = info[8], info[9]
        axis = info[13]
        print(
            f"[{i:02d}] {name:34s} type={jtype} "
            f"lim=({lo:.2f},{hi:.2f}) axis={axis} parent={parent}"
        )
    print("-" * 66 + "\n")


def disable_default_motors(body_id: int) -> None:
    """Remove passive joint friction so torque control is clean."""
    for j in range(p.getNumJoints(body_id)):
        p.setJointMotorControl2(
            bodyUniqueId=body_id,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            force=0.0,
        )


def build_joint_map(body_id: int) -> dict:
    """Return {joint_name: index} for all revolute joints."""
    return {
        p.getJointInfo(body_id, i)[1].decode(): i
        for i in range(p.getNumJoints(body_id))
        if p.getJointInfo(body_id, i)[2] == p.JOINT_REVOLUTE
    }


class RandomGaitController:
    """
    Sinusoidal CPG walk controller for advanced_human.urdf.

    Drives hip flexion, knee, ankle, and arm joints with antiphase
    sinusoids. Amplitude and frequency are randomly sampled so each
    run produces a different gait.
    """

    # (joint_name, phase_offset_radians)
    _SCHEDULE = [
        # --- Legs ---
        ("hip_abduct_L_to_thigh_L",  0.0),           # hip flex L
        ("hip_abduct_R_to_thigh_R",  math.pi),        # hip flex R (antiphase)
        ("thigh_L_to_shin_L",         0.0 + 0.5),     # knee L (delayed into swing)
        ("thigh_R_to_shin_R",         math.pi + 0.5), # knee R
        ("shin_L_to_foot_L",          math.pi),       # ankle L (push-off)
        ("shin_R_to_foot_R",          0.0),            # ankle R
        # --- Arms (counter-swing) ---
        ("clavicle_L_to_upperarm_L",  math.pi),
        ("clavicle_R_to_upperarm_R",  0.0),
    ]

    # knees bend only during forward swing (positive half of sinusoid)
    _POSITIVE_ONLY = {"thigh_L_to_shin_L", "thigh_R_to_shin_R"}

    def __init__(self, body_id: int, jmap: dict, seed=None):
        self.body_id = body_id
        self.jmap = jmap

        rng = random.Random(seed)
        self.freq      = rng.uniform(0.8,  1.4)
        self.hip_amp   = rng.uniform(0.35, 0.55)
        self.knee_amp  = rng.uniform(0.40, 0.65)
        self.ankle_amp = rng.uniform(0.15, 0.28)
        self.arm_amp   = rng.uniform(0.25, 0.45)

        self._amp = {
            "hip_abduct_L_to_thigh_L":  self.hip_amp,
            "hip_abduct_R_to_thigh_R":  self.hip_amp,
            "thigh_L_to_shin_L":        self.knee_amp,
            "thigh_R_to_shin_R":        self.knee_amp,
            "shin_L_to_foot_L":         self.ankle_amp,
            "shin_R_to_foot_R":         self.ankle_amp,
            "clavicle_L_to_upperarm_L": self.arm_amp,
            "clavicle_R_to_upperarm_R": self.arm_amp,
        }

        missing = [n for n, _ in self._SCHEDULE if n not in jmap]
        if missing:
            print(f"[Gait] WARNING — joints not found in URDF: {missing}")

        print(
            f"[Gait] freq={self.freq:.2f} Hz  "
            f"hip={self.hip_amp:.2f} rad  knee={self.knee_amp:.2f} rad  "
            f"ankle={self.ankle_amp:.2f} rad  arm={self.arm_amp:.2f} rad"
        )

    def step(self, t: float) -> None:
        w = 2.0 * math.pi * self.freq * t
        for joint_name, phase in self._SCHEDULE:
            if joint_name not in self.jmap:
                continue
            idx = self.jmap[joint_name]
            amp = self._amp[joint_name]
            raw = amp * math.sin(w + phase)
            if joint_name in self._POSITIVE_ONLY:
                raw = max(0.0, raw)
            info = p.getJointInfo(self.body_id, idx)
            lo, hi = info[8], info[9]
            target = max(lo, min(hi, raw))
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id,
                jointIndex=idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=350.0,
                positionGain=0.6,
                velocityGain=0.05,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="PyBullet humanoid spawn / walk test")
    ap.add_argument("--gui",   action="store_true", help="Open PyBullet GUI")
    ap.add_argument("--walk",  action="store_true", help="Enable random sinusoidal gait")
    ap.add_argument("--steps", type=int,   default=4800,         help="Simulation steps")
    ap.add_argument("--dt",    type=float, default=1.0 / 240.0,  help="Time step (s)")
    ap.add_argument("--seed",  type=int,   default=None,         help="RNG seed for gait params")
    ap.add_argument(
        "--urdf", type=str, default="advanced_human.urdf",
        help="URDF filename inside assets/urdf/",
    )
    args = ap.parse_args()

    client = p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(args.dt)

    p.loadURDF("plane.urdf")

    urdf_path = URDF_DIR / args.urdf
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Place pelvis just high enough for legs to clear the ground when straight
    start_z = 0.92 if args.walk else 1.3
    humanoid_id = p.loadURDF(
        str(urdf_path),
        basePosition=[0.0, 0.0, start_z],
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        useFixedBase=False,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    print_joint_info(humanoid_id)
    disable_default_motors(humanoid_id)

    gait = None
    if args.walk:
        jmap = build_joint_map(humanoid_id)
        gait = RandomGaitController(humanoid_id, jmap, seed=args.seed)

    sim_time = 0.0
    for step in range(args.steps):
        if gait is not None:
            gait.step(sim_time)

        p.stepSimulation()
        sim_time += args.dt

        if args.gui:
            time.sleep(args.dt)

        pos, _ = p.getBasePositionAndOrientation(humanoid_id)
        if abs(pos[0]) > 50 or abs(pos[1]) > 50 or pos[2] < -5:
            print(f"Unstable at step {step}: pos={pos}")
            break

    if args.gui:
        print("Simulation done. Press Ctrl+C to close.")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass

    p.disconnect(client)


if __name__ == "__main__":
    main()