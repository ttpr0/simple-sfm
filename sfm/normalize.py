import numpy as np
from scipy.linalg import inv, norm

from .util import normalize_homogeneous, normalize_homogeneous_inplace

def normalize_points(x: np.ndarray) -> np.ndarray:
    N = get_normalization_matrix(x)
    return transform_points(x, N)

def get_normalization_matrix(x: np.ndarray) -> np.ndarray:
    """computes normalization matrix for homogeneous coordinates

    Args:
        x: [n x 3] points in image space (homogeneous coordinates)
    
    Returns:
        [3 x 3] normalization matrix
    """
    x = normalize_homogeneous(x)
    x_mean = np.mean(x, axis=0)
    d = norm(x - x_mean, axis=1)
    s = np.mean(d, axis=0)
    return np.array([
        [1, 0, -x_mean[0]],
        [0, 1, -x_mean[1]],	
        [0, 0, s]
    ])

def transform_points(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """transforms points

    Args:
        x: [n x 3] points in image space (homogeneous coordinates)
        T: [3 x 3] normalization matrix

    Returns:
        [n x 3] transformed points
    """
    x_t = np.zeros(x.shape)
    n = x.shape[0]
    for i in range(n):
        x_t[i, :] = T @ x[i, :]
    normalize_homogeneous_inplace(x_t)
    return x_t
