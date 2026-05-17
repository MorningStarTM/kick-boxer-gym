import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

SMPLX_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

KICKBOXING_JOINTS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 21,
]


class SMPLXMocapDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_length: int = 64,
        subset: str = "train",
        fps: int = 30,
    ):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.fps = fps
        self.sequences = []
        self._load_data(subset)

    def _load_data(self, subset: str):
        data_path = self.data_dir / subset
        if not data_path.exists():
            data_path = self.data_dir

        npz_files = sorted(data_path.glob("*.npz"))
        pkl_files = sorted(data_path.glob("*.pkl"))
        files = npz_files + pkl_files

        if len(files) == 0:
            print(f"[Warning] No mocap files found in {data_path}. "
                  "Generating synthetic placeholder data for testing.")
            self._generate_synthetic_data()
            return

        for f in files:
            try:
                data = np.load(f, allow_pickle=True)
                poses = self._extract_poses(data)
                if poses is not None and len(poses) >= self.seq_length:
                    self.sequences.append(poses)
            except Exception as e:
                print(f"[Warning] Failed to load {f}: {e}")

        if len(self.sequences) == 0:
            print("[Warning] No valid sequences loaded. Using synthetic data.")
            self._generate_synthetic_data()

    def _extract_poses(self, data) -> Optional[np.ndarray]:
        for key in ["poses", "body_pose", "smplx_body_pose", "pose_body"]:
            if key in data:
                poses = np.array(data[key])
                if poses.ndim == 2:
                    return poses.astype(np.float32)
        return None

    def _generate_synthetic_data(self, n_sequences: int = 50):
        joint_dim = len(KICKBOXING_JOINTS) * 3
        for _ in range(n_sequences):
            length = np.random.randint(self.seq_length * 2, self.seq_length * 5)
            t = np.linspace(0, 2 * np.pi * np.random.uniform(1, 3), length)
            poses = np.zeros((length, joint_dim), dtype=np.float32)
            for j in range(joint_dim):
                freq = np.random.uniform(0.5, 2.0)
                amp = np.random.uniform(0.1, 0.5)
                phase = np.random.uniform(0, 2 * np.pi)
                poses[:, j] = amp * np.sin(freq * t + phase)
            self.sequences.append(poses)

    def __len__(self) -> int:
        total = 0
        for seq in self.sequences:
            if len(seq) >= self.seq_length:
                total += len(seq) - self.seq_length + 1
        return total

    def __getitem__(self, idx: int) -> torch.Tensor:
        for seq in self.sequences:
            n_windows = len(seq) - self.seq_length + 1
            if n_windows <= 0:
                continue
            if idx < n_windows:
                window = seq[idx : idx + self.seq_length]
                return torch.tensor(window, dtype=torch.float32)
            idx -= n_windows
        raise IndexError("Index out of range")

    @property
    def pose_dim(self) -> int:
        if self.sequences:
            return self.sequences[0].shape[1]
        return len(KICKBOXING_JOINTS) * 3
