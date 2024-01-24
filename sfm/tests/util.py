import numpy as np

from ..projection import reproject
from ..triangulate import compute_intersection


def make_random_points(count: int, bounds: list[tuple[float, float]]) -> np.ndarray:
    dims = len(bounds)
    
    points = np.zeros((count, dims+1))
    for i in range(count):
        for j in range(dims):
            d = bounds[j][1] - bounds[j][0]
            m = bounds[j][0]
            points[i, j] = np.random.rand() * d + m
        points[i, dims] = 1
    return points

def add_random_noise(points: np.ndarray, scale: float = 4.0) -> np.ndarray:
    n = points.shape[0]
    dims = points.shape[1] - 1
    new_points = points.copy()
    
    for i in range(n):
        for j in range(dims):
            new_points[i, j] += np.random.rand() * scale - 0.5*scale
    return new_points
    
def is_in_front(x_1: np.ndarray, x_2: np.ndarray, R: np.ndarray, X0: np.ndarray, K: np.ndarray) -> bool:
    X0_ = np.array([0, 0, 0])
    R_ = np.eye(3)
    
    # project image points to object space rays
    r_1 = reproject(x_1, K, R_)
    r_2 = reproject(x_2, K, R)

    # compute intersection point in camera 1 coordinate system
    P = compute_intersection(X0_.T, r_1.T, X0.T, r_2.T)
    if P[2] < 0:
        return False

    # convert point to camera 2 coordinate system
    P = R @ (P - X0)
    if P[2] < 0:
        return False

    return True
