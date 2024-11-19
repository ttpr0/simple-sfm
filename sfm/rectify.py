import numpy as np
from scipy.optimize import minimize
import cv2

def stereo_rectify(K1: np.ndarray, K2: np.ndarray, R12: np.ndarray, T12: np.ndarray, img_size: tuple[float, float]):
    """Computes the stereo rectification of two images.

    Args:
        K1: [3 x 3] Calibration of the first image
        K2: [3 x 3] Calibration of the second image
        R12: [3 x 3] Rotation from the first to the second camera coordinate system
        T12: [3 x 1] Translation of the second camera in the first camera coordinate system
        img_size: image size as tuple (H, W)

    Returns:
        Tuple of R1 [3 x 3], R2 [3 x 3], P1 [3 x 3], P2 [3 x 3] and Q [4 x 4].
        To create the homography matrices use H = P @ R @ inv(K)
        Q is a matrix to project disparities from the rectified first image back into 3D (coordinate system of the rectified first camera).
    """
    # compute rotations
    v1 = np.array([0, 0, 1])
    v2 = R12.T @ v1
    v = (v1 + v2) / 2
    t = T12 / np.linalg.norm(T12)
    u = np.cross(v, t)
    u = u / np.linalg.norm(u)
    v = np.cross(t, u)
    v = v / np.linalg.norm(v)
    R1 = np.array([t, u, v])
    R2 = (R12 @ R1.T).T
    # project corners into new image
    corners = np.array([
        [0, 0, 1],
        [img_size[1], 0, 1],
        [img_size[1], img_size[0], 1],
        [0, img_size[0], 1]
    ])
    P1 = R1 @ np.linalg.inv(K1)
    projected1 = (P1 @ corners.T).T
    projected1 /= projected1[:, 2].reshape(-1, 1)
    P2 = R2 @ np.linalg.inv(K2)
    projected2 = (P2 @ corners.T).T
    projected2 /= projected2[:, 2].reshape(-1, 1)
    # compute new projection centers
    f, cx1, cx2, cy = _optimize_intrinsics(projected1, projected2, img_size)
    # compute new projection matrices
    P1 = np.array([
        [f, 0, cx1],
        [0, f, cy],
        [0, 0, 1]
    ])
    P2 = np.array([
        [f, 0, cx2],
        [0, f, cy],
        [0, 0, 1]
    ])
    # compute Q matrix
    T_ = R1 @ T12
    Tx = T_[0]
    Q = np.array([
        [1, 0, 0, -cx1],
        [0, 1, 0, -cy],
        [0, 0, 0, f],
        [0, 0, -1/Tx, (cx1-cx2)/Tx]
    ])
    return R1, R2, P1, P2, Q

def _optimize_intrinsics(points1, points2, img_size):
    def penalty(val, min_val, max_val):
        if val < min_val:
            return (min_val - val)**2
        elif val > max_val:
            return (val - max_val)**2
        return 0
    def objective(params, points1, points2, u_range, v_range):
        f, cx1, cx2, cy = params
        u_min, u_max = u_range
        v_min, v_max = v_range
        total_penalty = 0
        for i in range(points1.shape[0]):
            X = points1[i, 0]
            Y = points1[i, 1]
            u = X * f + cx1
            v = Y * f + cy
            total_penalty += penalty(u, u_min, u_max)
            total_penalty += penalty(v, v_min, v_max)
        for i in range(points2.shape[0]):
            X = points2[i, 0]
            Y = points2[i, 1]
            u = X * f + cx2
            v = Y * f + cy
            total_penalty += penalty(u, u_min, u_max)
            total_penalty += penalty(v, v_min, v_max)
        return total_penalty
    u_range = (0, img_size[1])
    v_range = (0, img_size[0])
    # Initial guess for parameters [f, cx1, cx2, cy]
    initial_guess = [1430, img_size[1]/2, img_size[1]/2, img_size[0]/2]
    bounds = [(1420, 1440), # f bounds
            (0, img_size[1]), # cx1 bounds
            (0, img_size[1]), # cx2 bounds
            (0, img_size[0])] # cy bounds
    # Optimize
    result = minimize(objective, initial_guess, args=(points1, points2, u_range, v_range), method='L-BFGS-B', bounds=bounds)
    # Get the optimized parameters
    f_opt, cx1_opt, cx2_opt, cy_opt = result.x
    return f_opt, cx1_opt, cx2_opt, cy_opt
