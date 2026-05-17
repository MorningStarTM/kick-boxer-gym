import numpy as np


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis_angle / angle
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def normalize_angle(angle: float) -> float:
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def compute_com(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return np.sum(positions * masses[:, None], axis=0) / np.sum(masses)


def rotation_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)
