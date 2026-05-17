"""
Simulate the SMPL-X humanoid walking for a configurable duration using a
CPG (Central Pattern Generator) gait controller.

Usage:
    python scripts/simulate_walk.py                     # walk 10 seconds (default)
    python scripts/simulate_walk.py --duration 20       # walk 20 seconds
    python scripts/simulate_walk.py --speed 1.5         # faster gait
    python scripts/simulate_walk.py --record walk.mp4   # save video (requires ffmpeg)
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pybullet as p
import pybullet_data


# ---------------------------------------------------------------------------
# Gait parameters -- tuned for the humanoid_smplx.urdf proportions
# ---------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 240.0

GAIT_DEFAULTS = dict(
    cycle_freq=1.2,          # steps per second (full left-right cycle)
    hip_swing_amp=0.45,      # rad – hip flexion/extension amplitude
    hip_swing_offset=-0.1,   # slight forward lean
    knee_bend_amp=0.7,       # rad – peak knee bend during swing
    knee_stance_bend=0.15,   # slight bend in stance leg for natural look
    ankle_amp=0.25,          # rad – ankle dorsi/plantar flexion
    arm_swing_amp=0.35,      # rad – shoulder swing (opposite to legs)
    elbow_bend=0.8,          # rad – resting elbow flexion while walking
    spine_twist_amp=0.06,    # slight counter-rotation in spine
    lateral_sway_amp=0.03,   # lateral pelvis tilt for weight shift
)


def axis_angle_to_quat(axis, angle):
    """Axis (unit vector) + angle (rad) -> [x, y, z, w] quaternion (PyBullet order)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = angle / 2.0
    s = math.sin(half)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)]


def compose_quats(q1, q2):
    """Hamilton product q1 * q2, both in [x,y,z,w] layout."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Joint bookkeeping
# ---------------------------------------------------------------------------

def build_joint_map(body_id, client):
    """Return {name: (index, type)} for every non-fixed joint."""
    jmap = {}
    n = p.getNumJoints(body_id, physicsClientId=client)
    for i in range(n):
        info = p.getJointInfo(body_id, i, physicsClientId=client)
        name = info[1].decode()
        jtype = info[2]
        if jtype != p.JOINT_FIXED:
            jmap[name] = (i, jtype)
    return jmap


def set_spherical_target(body_id, joint_idx, target_quat, max_force, client):
    p.setJointMotorControlMultiDof(
        body_id,
        joint_idx,
        controlMode=p.POSITION_CONTROL,
        targetPosition=target_quat,
        force=[max_force, max_force, max_force],
        physicsClientId=client,
    )


def set_revolute_target(body_id, joint_idx, target_angle, max_force, client):
    p.setJointMotorControl2(
        body_id,
        joint_idx,
        controlMode=p.POSITION_CONTROL,
        targetPosition=target_angle,
        force=max_force,
        physicsClientId=client,
    )


# ---------------------------------------------------------------------------
# CPG walk controller
# ---------------------------------------------------------------------------

class WalkController:
    def __init__(self, body_id, joint_map, client, gait: dict = None):
        self.body = body_id
        self.jmap = joint_map
        self.client = client
        self.g = {**GAIT_DEFAULTS, **(gait or {})}

    def _phase(self, t):
        """Return phase in [0, 2*pi) for the gait cycle."""
        return (2.0 * math.pi * self.g["cycle_freq"] * t) % (2.0 * math.pi)

    def update(self, t: float):
        g = self.g
        phi = self._phase(t)

        # -- Leg phase signals (left leads, right is offset by pi) --
        left_swing = math.sin(phi)
        right_swing = math.sin(phi + math.pi)

        # Knee bends more during swing (positive sin half-cycle)
        left_knee_phase = max(0, math.sin(phi)) * g["knee_bend_amp"] + g["knee_stance_bend"]
        right_knee_phase = max(0, math.sin(phi + math.pi)) * g["knee_bend_amp"] + g["knee_stance_bend"]

        # Ankle pushes off at end of stance
        left_ankle_phase = -g["ankle_amp"] * math.sin(phi + 0.5)
        right_ankle_phase = -g["ankle_amp"] * math.sin(phi + math.pi + 0.5)

        # -- Hips (spherical): flex/extend around X axis --
        left_hip_angle = g["hip_swing_amp"] * left_swing + g["hip_swing_offset"]
        right_hip_angle = g["hip_swing_amp"] * right_swing + g["hip_swing_offset"]

        # Add small lateral sway for weight shift (Z axis rotation on hips)
        lateral = g["lateral_sway_amp"] * math.sin(phi)

        left_hip_q = compose_quats(
            axis_angle_to_quat([1, 0, 0], left_hip_angle),
            axis_angle_to_quat([0, 0, 1], lateral),
        )
        right_hip_q = compose_quats(
            axis_angle_to_quat([1, 0, 0], right_hip_angle),
            axis_angle_to_quat([0, 0, 1], -lateral),
        )

        self._set_spherical("left_hip", left_hip_q, 150)
        self._set_spherical("right_hip", right_hip_q, 150)

        # -- Knees (revolute): negative angle = flexion in our URDF --
        self._set_revolute("left_knee", -left_knee_phase, 120)
        self._set_revolute("right_knee", -right_knee_phase, 120)

        # -- Ankles (spherical): dorsi/plantar flexion around X --
        self._set_spherical("left_ankle", axis_angle_to_quat([1, 0, 0], left_ankle_phase), 80)
        self._set_spherical("right_ankle", axis_angle_to_quat([1, 0, 0], right_ankle_phase), 80)

        # -- Arm swing (opposite to ipsilateral leg) --
        left_arm_angle = -g["arm_swing_amp"] * left_swing
        right_arm_angle = -g["arm_swing_amp"] * right_swing

        self._set_spherical(
            "left_shoulder",
            axis_angle_to_quat([1, 0, 0], left_arm_angle),
            60,
        )
        self._set_spherical(
            "right_shoulder",
            axis_angle_to_quat([1, 0, 0], right_arm_angle),
            60,
        )

        # Elbows stay bent (revolute)
        self._set_revolute("left_elbow", g["elbow_bend"], 40)
        self._set_revolute("right_elbow", -g["elbow_bend"], 40)

        # -- Spine: counter-twist for natural torso rotation --
        spine_twist = g["spine_twist_amp"] * math.sin(phi)
        for spine_name in ["spine1_joint", "spine2_joint", "spine3_joint"]:
            self._set_spherical(
                spine_name,
                axis_angle_to_quat([0, 0, 1], spine_twist / 3.0),
                80,
            )

        # -- Head + neck: keep upright --
        self._set_spherical("neck_joint", IDENTITY_QUAT, 30)
        self._set_spherical("head_joint", IDENTITY_QUAT, 20)

        # -- Wrists: neutral --
        self._set_spherical("left_wrist", IDENTITY_QUAT, 15)
        self._set_spherical("right_wrist", IDENTITY_QUAT, 15)

    # helpers --
    def _set_spherical(self, name, quat, force):
        if name in self.jmap:
            idx, jtype = self.jmap[name]
            if jtype == p.JOINT_SPHERICAL:
                set_spherical_target(self.body, idx, quat, force, self.client)

    def _set_revolute(self, name, angle, force):
        if name in self.jmap:
            idx, jtype = self.jmap[name]
            if jtype == p.JOINT_REVOLUTE:
                set_revolute_target(self.body, idx, angle, force, self.client)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simulate SMPL-X humanoid walking")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to simulate")
    parser.add_argument("--speed", type=float, default=1.0, help="Gait speed multiplier")
    parser.add_argument("--record", type=str, default=None, help="Save video to file (requires ffmpeg)")
    parser.add_argument("--no-realtime", action="store_true", help="Run as fast as possible")
    args = parser.parse_args()

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(PHYSICS_DT, physicsClientId=client)

    p.loadURDF("plane.urdf", physicsClientId=client)

    urdf_path = str(Path(__file__).parent.parent / "models" / "urdf" / "humanoid_smplx.urdf")
    humanoid = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0.95],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False,
        physicsClientId=client,
    )

    joint_map = build_joint_map(humanoid, client)
    print(f"Active joints ({len(joint_map)}):")
    for name, (idx, jtype) in sorted(joint_map.items(), key=lambda x: x[1][0]):
        tname = {p.JOINT_REVOLUTE: "rev", p.JOINT_SPHERICAL: "sph"}.get(jtype, "?")
        print(f"  [{idx:2d}] {name} ({tname})")

    # Disable default velocity motors on all joints so position control works cleanly
    n_joints = p.getNumJoints(humanoid, physicsClientId=client)
    for i in range(n_joints):
        info = p.getJointInfo(humanoid, i, physicsClientId=client)
        jtype = info[2]
        if jtype == p.JOINT_SPHERICAL:
            p.setJointMotorControlMultiDof(
                humanoid, i, p.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                force=[0, 0, 0],
                physicsClientId=client,
            )
        elif jtype == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(
                humanoid, i, p.VELOCITY_CONTROL,
                targetVelocity=0, force=0,
                physicsClientId=client,
            )

    gait_overrides = {"cycle_freq": GAIT_DEFAULTS["cycle_freq"] * args.speed}
    walker = WalkController(humanoid, joint_map, client, gait=gait_overrides)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    if args.record:
        log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, args.record,
            physicsClientId=client,
        )

    # Let the character settle for a moment before walking
    for _ in range(int(0.5 / PHYSICS_DT)):
        walker.update(0)
        p.stepSimulation(physicsClientId=client)

    total_steps = int(args.duration / PHYSICS_DT)
    sim_time = 0.0
    wall_start = time.perf_counter()

    print(f"\nWalking for {args.duration:.1f}s (speed x{args.speed:.1f}) ...")

    for step in range(total_steps):
        sim_time = step * PHYSICS_DT
        walker.update(sim_time)
        p.stepSimulation(physicsClientId=client)

        # Camera follows the character
        if step % 24 == 0:
            pos, _ = p.getBasePositionAndOrientation(humanoid, physicsClientId=client)
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5,
                cameraYaw=60,
                cameraPitch=-15,
                cameraTargetPosition=[pos[0], pos[1], 0.8],
            )

        if not args.no_realtime:
            wall_elapsed = time.perf_counter() - wall_start
            sim_elapsed = (step + 1) * PHYSICS_DT
            sleep_time = sim_elapsed - wall_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Print final position
    final_pos, _ = p.getBasePositionAndOrientation(humanoid, physicsClientId=client)
    distance = math.sqrt(final_pos[0] ** 2 + final_pos[1] ** 2)
    print(f"Done. Final position: x={final_pos[0]:.2f} y={final_pos[1]:.2f} z={final_pos[2]:.2f}")
    print(f"Distance traveled: {distance:.2f}m in {args.duration:.1f}s ({distance / args.duration:.2f} m/s)")

    if args.record:
        p.stopStateLogging(log_id, physicsClientId=client)
        print(f"Video saved to {args.record}")

    # Hold the window open briefly so user can see final pose
    print("Press Ctrl+C to exit.")
    try:
        while True:
            p.stepSimulation(physicsClientId=client)
            time.sleep(PHYSICS_DT)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()
