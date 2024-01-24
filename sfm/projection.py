import numpy as np
from scipy.linalg import svd, qr, inv, norm

from .util import normalize_homogeneous_inplace

def project(X: np.ndarray, K: np.ndarray, R: np.ndarray, X0: np.ndarray) -> np.ndarray:
    """transform point from object space to image space

    Args:
        X: [n x 4] points in object space (homogeneous coordinates)
        K: [3 x 3] calibration matrix
        R: [3 x 3] rotation matrix
        X0: [3 x 1] coordinate of projection center
    
    Returns:
        [n x 3] point in image space (homogeneous coordinates)
    """
    n = X.shape[0]
    if X.shape[1] < 4:
        raise ValueError("invalid input")

    P = K @ R @ np.hstack((np.eye(3), -X0.reshape(3, 1)))
    
    x = np.zeros((n, 3))
    for i in range(n):
        x[i, :] = (P @ X[i, :].reshape((4, 1))).reshape((3,))

    normalize_homogeneous_inplace(x)
    return x

def reproject(x: np.ndarray, K: np.ndarray, R: np.ndarray) -> np.ndarray:
    """transform point from image space to object space

    Args:
        x: [3 x 1] point in image space in homogeneous coordinates
        K: [3 x 3] calibration matrix
        R: [3 x 3] rotation matrix
    
    Returns:
        [3 x 1] projection vector from projection center towards object point (normalized to length 1)
    """
    X = inv(R) @ inv(K) @ x
    X = X / norm(X)
    return X
