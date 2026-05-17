import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from pathlib import Path
from typing import Optional

from utils.config import load_config
from utils.rewards import KickBoxingReward


class KickBoxingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    CONTROLLABLE_JOINTS = [
        "spine1_joint", "spine2_joint", "spine3_joint",
        "neck_joint", "head_joint",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
    ]

    STRIKE_LINKS = ["left_hand", "right_hand", "left_foot", "right_foot"]

    def __init__(self, config: Optional[dict] = None, render_mode: Optional[str] = None):
        super().__init__()

        self.config = config or load_config()
        self.render_mode = render_mode
        env_cfg = self.config.get("environment", {})
        humanoid_cfg = self.config.get("humanoid", {})

        self.physics_dt = env_cfg.get("physics_timestep", 0.002)
        self.control_dt = env_cfg.get("control_timestep", 0.02)
        self.sim_steps_per_control = int(self.control_dt / self.physics_dt)
        self.max_episode_steps = env_cfg.get("max_episode_steps", 1000)
        self.max_torque = humanoid_cfg.get("max_torque", 200.0)

        self._urdf_path = str(
            Path(__file__).parent.parent / "models" / "urdf" / "humanoid_smplx.urdf"
        )

        self._setup_physics()

        self.num_joints = 0
        self.joint_indices = []
        self.joint_names = []
        self.link_name_to_index = {}
        self._load_humanoids()

        obs_dim = self._get_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        act_dim = len(self.joint_indices)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        self.reward_fn = KickBoxingReward(self.config)
        self.step_count = 0

    def _setup_physics(self):
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        gravity = self.config.get("environment", {}).get("gravity", [0, 0, -9.81])
        p.setGravity(*gravity, physicsClientId=self.physics_client)
        p.setTimeStep(self.physics_dt, physicsClientId=self.physics_client)

        self.ground_id = p.loadURDF(
            "plane.urdf", physicsClientId=self.physics_client
        )
        friction = self.config.get("environment", {}).get("ground_friction", 1.0)
        p.changeDynamics(
            self.ground_id, -1, lateralFriction=friction,
            physicsClientId=self.physics_client,
        )

    def _load_humanoids(self):
        init_height = self.config.get("humanoid", {}).get("initial_height", 0.95)

        self.agent_id = p.loadURDF(
            self._urdf_path,
            basePosition=[0, 0, init_height],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.physics_client,
        )

        self.opponent_id = p.loadURDF(
            self._urdf_path,
            basePosition=[1.5, 0, init_height],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.physics_client,
        )

        self._build_joint_map(self.agent_id)
        self._build_link_map(self.agent_id)

        damping = self.config.get("humanoid", {}).get("joint_damping", 0.5)
        for body_id in [self.agent_id, self.opponent_id]:
            for i in range(p.getNumJoints(body_id, physicsClientId=self.physics_client)):
                p.changeDynamics(
                    body_id, i, jointDamping=damping,
                    physicsClientId=self.physics_client,
                )

    def _build_joint_map(self, body_id: int):
        self.joint_indices = []
        self.joint_names = []
        n = p.getNumJoints(body_id, physicsClientId=self.physics_client)
        for i in range(n):
            info = p.getJointInfo(body_id, i, physicsClientId=self.physics_client)
            name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_type != p.JOINT_FIXED:
                self.joint_indices.append(i)
                self.joint_names.append(name)
        self.num_joints = len(self.joint_indices)

    def _build_link_map(self, body_id: int):
        self.link_name_to_index = {}
        n = p.getNumJoints(body_id, physicsClientId=self.physics_client)
        for i in range(n):
            info = p.getJointInfo(body_id, i, physicsClientId=self.physics_client)
            link_name = info[12].decode("utf-8")
            self.link_name_to_index[link_name] = i

    def _get_obs_dim(self) -> int:
        # root pos(3) + root orn(4) + root vel(3) + root ang_vel(3)
        # + joint_pos(num_joints) + joint_vel(num_joints)
        # + opponent root pos(3) + opponent root orn(4)
        # + opponent joint_pos(num_joints)
        # + contact flags(4 strike links)
        return 13 + 2 * self.num_joints + 7 + self.num_joints + 4

    def _get_obs(self) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(
            self.agent_id, physicsClientId=self.physics_client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.agent_id, physicsClientId=self.physics_client
        )

        joint_states = p.getJointStates(
            self.agent_id, self.joint_indices, physicsClientId=self.physics_client
        )
        joint_pos = [s[0] for s in joint_states]
        joint_vel = [s[1] for s in joint_states]

        opp_pos, opp_orn = p.getBasePositionAndOrientation(
            self.opponent_id, physicsClientId=self.physics_client
        )
        opp_joint_states = p.getJointStates(
            self.opponent_id, self.joint_indices, physicsClientId=self.physics_client
        )
        opp_joint_pos = [s[0] for s in opp_joint_states]

        contacts = self._get_contact_flags()

        obs = np.concatenate([
            np.array(pos),
            np.array(orn),
            np.array(vel),
            np.array(ang_vel),
            np.array(joint_pos),
            np.array(joint_vel),
            np.array(opp_pos),
            np.array(opp_orn),
            np.array(opp_joint_pos),
            np.array(contacts),
        ]).astype(np.float32)

        return obs

    def _get_contact_flags(self) -> list[float]:
        flags = [0.0] * len(self.STRIKE_LINKS)
        for i, link_name in enumerate(self.STRIKE_LINKS):
            if link_name in self.link_name_to_index:
                link_idx = self.link_name_to_index[link_name]
                pts = p.getContactPoints(
                    bodyA=self.agent_id,
                    bodyB=self.opponent_id,
                    linkIndexA=link_idx,
                    physicsClientId=self.physics_client,
                )
                if pts:
                    flags[i] = 1.0
        return flags

    def _get_contacts(self) -> list[dict]:
        contacts = []
        pts = p.getContactPoints(
            bodyA=self.agent_id,
            bodyB=self.opponent_id,
            physicsClientId=self.physics_client,
        )
        for pt in pts:
            link_a_idx = pt[3]
            link_b_idx = pt[4]

            link_a_name = self._index_to_link_name(self.agent_id, link_a_idx)
            link_b_name = self._index_to_link_name(self.opponent_id, link_b_idx)

            contacts.append({
                "link_a": link_a_name,
                "link_b": link_b_name,
                "normal_force": pt[9],
                "position": pt[5],
            })
        return contacts

    def _index_to_link_name(self, body_id: int, link_idx: int) -> str:
        if link_idx == -1:
            return "pelvis"
        info = p.getJointInfo(body_id, link_idx, physicsClientId=self.physics_client)
        return info[12].decode("utf-8")

    def _is_fallen(self) -> bool:
        pos, orn = p.getBasePositionAndOrientation(
            self.agent_id, physicsClientId=self.physics_client
        )
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        up_vec = rot_matrix[:, 2]
        return pos[2] < 0.4 or up_vec[2] < 0.3

    def _get_com_height(self) -> float:
        pos, _ = p.getBasePositionAndOrientation(
            self.agent_id, physicsClientId=self.physics_client
        )
        return pos[2]

    def _get_upright_proj(self) -> float:
        _, orn = p.getBasePositionAndOrientation(
            self.agent_id, physicsClientId=self.physics_client
        )
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return rot_matrix[2, 2]

    def _apply_opponent_policy(self):
        for idx in self.joint_indices:
            p.setJointMotorControl2(
                self.opponent_id,
                idx,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=50.0,
                physicsClientId=self.physics_client,
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        init_height = self.config.get("humanoid", {}).get("initial_height", 0.95)

        p.resetBasePositionAndOrientation(
            self.agent_id, [0, 0, init_height],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(
            self.agent_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.physics_client,
        )

        p.resetBasePositionAndOrientation(
            self.opponent_id, [1.5, 0, init_height],
            p.getQuaternionFromEuler([0, 0, np.pi]),
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(
            self.opponent_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.physics_client,
        )

        for body_id in [self.agent_id, self.opponent_id]:
            for idx in self.joint_indices:
                p.resetJointState(
                    body_id, idx, 0.0, 0.0,
                    physicsClientId=self.physics_client,
                )

        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physics_client)

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        torques = action * self.max_torque

        for i, idx in enumerate(self.joint_indices):
            if i < len(torques):
                p.setJointMotorControl2(
                    self.agent_id,
                    idx,
                    p.TORQUE_CONTROL,
                    force=torques[i],
                    physicsClientId=self.physics_client,
                )

        self._apply_opponent_policy()

        for _ in range(self.sim_steps_per_control):
            p.stepSimulation(physicsClientId=self.physics_client)

        self.step_count += 1
        obs = self._get_obs()

        agent_pos = np.array(
            p.getBasePositionAndOrientation(
                self.agent_id, physicsClientId=self.physics_client
            )[0]
        )
        agent_vel = np.array(
            p.getBaseVelocity(self.agent_id, physicsClientId=self.physics_client)[0]
        )
        opp_pos = np.array(
            p.getBasePositionAndOrientation(
                self.opponent_id, physicsClientId=self.physics_client
            )[0]
        )

        contacts = self._get_contacts()
        is_fallen = self._is_fallen()
        com_height = self._get_com_height()
        upright_proj = self._get_upright_proj()

        reward, reward_info = self.reward_fn.compute(
            agent_pos=agent_pos,
            agent_vel=agent_vel,
            opponent_pos=opp_pos,
            contacts=contacts,
            joint_torques=torques,
            com_height=com_height,
            upright_proj=upright_proj,
            is_fallen=is_fallen,
        )

        terminated = is_fallen
        truncated = self.step_count >= self.max_episode_steps

        info = {"reward_info": reward_info, "step": self.step_count}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            width, height = 640, 480
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.75, 0, 0.8],
                distance=3.0,
                yaw=45,
                pitch=-20,
                roll=0,
                upAxisIndex=2,
            )
            proj = p.computeProjectionMatrixFOV(
                fov=60, aspect=width / height, nearVal=0.1, farVal=100.0
            )
            _, _, img, _, _ = p.getCameraImage(
                width, height, view, proj,
                physicsClientId=self.physics_client,
            )
            return np.array(img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        return None

    def close(self):
        if hasattr(self, "physics_client"):
            p.disconnect(self.physics_client)
