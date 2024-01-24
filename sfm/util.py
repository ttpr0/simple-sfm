import numpy as np


def normalize_homogeneous(x: np.ndarray) -> np.ndarray:
    """normalizes homogeneous coordinates

    Args:
        x: [n x d+1] points in homogeneous coordinates as matrix
    
    Returns:
        [n x d+1] normalized points
    """
    n = x.shape[0]
    d = x.shape[1]
    return x / np.broadcast_to(x[:, d-1].reshape((n, 1)), (n, d))

def normalize_homogeneous_inplace(x: np.ndarray):
    """normalizes homogeneous coordinates inplace

    Args:
        x: [n x d+1] points in homogeneous coordinates as matrix
    """
    n = x.shape[0]
    d = x.shape[1]
    x /= np.broadcast_to(x[:, d-1].reshape((n, 1)), (n, d))

def make_skew_symetric(X: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -X[2], X[1]],
        [X[2], 0, -X[0]],
        [-X[1], X[0], 0],
    ])

def build_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    R_z = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1],
    ])
    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)],
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)],
    ])
    return R_x @ R_y @ R_z
