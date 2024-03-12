import numpy as np
from scipy.linalg import inv, svd, det

from .least_squares import least_squares_svd
from .projection import reproject
from .triangulate import compute_intersection

def fundamental_matrix(x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
    """computes fundamental-matrix from relative oriented image-points

    Args:
        x_1: [n x 3] points in image 1 as homogeneous coordinates (min 8 points)
        x_2: [n x 3] corresponding points in image 2 as homogeneous coordinates
    
    Returns:
        [3 x 3] fundamental-matrix
    """
    n = x_1.shape[0]
    if n < 8 or x_2.shape[0] != n or x_1.shape[1] != 3 or x_2.shape[1] != 3:
        raise ValueError("invalid input")

    A = np.zeros((n, 9))
    for i in range(0,n):
        A[i,:] = (x_1[i,:].reshape((3,1)) @ x_2[i,:].reshape((1,3))).reshape((9,))

    p, _ = least_squares_svd(A)

    F = p.reshape((3,3))

    U, s, Vh = svd(F)
    S = np.zeros((3,3))
    S[0,0] = s[0]
    S[1,1] = s[1]
    F = U @ S @ Vh

    return F

def essential_matrix(x_1: np.ndarray, x_2: np.ndarray, K: np.ndarray) -> np.ndarray:
    """computes essential-matrix from relative oriented image-points

    Args:
        x_1: [n x 3] points in image 1 as homogeneous coordinates (min 8 points)
        x_2: [n x 3] corresponding points in image 2 as homogeneous coordinates
        K: [3 x 3] kalibration matrix

    Returns:
        [3 x 3] essential-matrix
    """
    n = x_1.shape[0]
    if n < 8 or x_2.shape[0] != n or x_1.shape[1] != 3 or x_2.shape[1] != 3:
        raise ValueError("invalid input")

    K_inv = inv(K)

    A = np.zeros((n, 9))
    for i in range(0,n):
        x_1_k = K_inv @ x_1[i,:].reshape((3, 1))
        x_2_k = K_inv @ x_2[i,:].reshape((3, 1))
        A[i,:] = (x_1_k.reshape((3,1)) @ x_2_k.reshape((1,3))).reshape((9,))

    p, _ = least_squares_svd(A)

    E = p.reshape((3,3))

    U, s, Vh = svd(E)
    S = np.zeros((3,3))
    S[0,0] = s[0]
    S[1,1] = s[1]
    E = U @ S @ Vh

    return E

def fundamental_from_essential(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """computes essential-matrix from relative oriented image-points

    Args:
        E: [3 x 3] essential matrix
        K: [3 x 3] kalibration matrix

    Returns:
        [3 x 3] fundamental-matrix
    """
    K_inv = inv(K)
    F = K_inv.T @ E @ K_inv
    # F = F / F[2,2]
    return F

def comute_epipolar(x_1: np.ndarray, x_2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """computes epipolar-geometry using fundamental-matrix

    Args:
        x_1: [n x 3] points in image 1 as homogeneous coordinates (min 8 points)
        x_2: [n x 3] corresponding points in image 2 as homogeneous coordinates
        F: [3 x 3] fundamental-matrix
    
    Returns:
        [n x 1] result values (x_1*F*x_2)
    """
    n = x_1.shape[0]
    if x_2.shape[0] != n or x_1.shape[1] != 3 or x_2.shape[1] != 3:
        raise ValueError("invalid input")

    results = np.zeros((n,))
    for i in range(0,n):
        results[i] = x_1[i,:].reshape((1,3)) @ F @ x_2[i,:].reshape((3,1))

    return results

def comute_epipolar_distance(x_1: np.ndarray, x_2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """computes distances from epipolar-lines using fundamental-matrix

    Args:
        x_1: [n x 3] points in image 1 as homogeneous coordinates (min 8 points)
        x_2: [n x 3] corresponding points in image 2 as homogeneous coordinates
        F: [3 x 3] fundamental-matrix
    
    Returns:
        [n x 1] result values (x_1*F*x_2)
    """
    n = x_1.shape[0]
    if x_2.shape[0] != n or x_1.shape[1] != 3 or x_2.shape[1] != 3:
        raise ValueError("invalid input")

    results = np.zeros((n,))
    for i in range(0,n):
        line = F @ x_1[i,:]
        a = line[0]
        b = line[1]
        c = line[2]
        x = x_2[i,0] / x_2[i,2]
        y = x_2[i,1] / x_2[i,2]
        dx = -(b/a)*y - c/a - x
        dy = -(a/b)*x - c/b - y
        results[i] = dx*dy / np.sqrt(dx*dx + dy*dy)

    return results

def solutions_from_essential(E: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """computes rotations and baseline-vectors from epipolar-geometry
    
    four possibile solutions are: (R1, T1), (R2, T1), (R1, T2) and (R2, T2)

    Args:
        F: [3 x 3] essential-matrix
    
    Returns:
        R1: [3 x 3] first rotation matrix
        R2: [3 x 3] second rotation matrix
        T1: [3 x 1] first baseline vector
        T2: [3 x 1] second baseline vector
    """
    U, _, V_T = svd(E)

    # extract base-line vectors
    T1 = U[:,2]
    T2 = -T1

    # make helper matrix
    W = np.zeros((3,3))
    W[0,1] = -1
    W[1,0] = 1
    W[2,2] = 1

    # compute rotations
    R1 = inv(U @ W @ V_T)
    R2 = inv(U @ W.T @ V_T)
    if det(R1) < 0:
        R1 = -R1
    if det(R2) < 0:
        R2 = -R2

    return R1, R2, T1, T2

def pick_valid_solution(x_1: np.ndarray, x_2: np.ndarray, K: np.ndarray, R1: np.ndarray, R2: np.ndarray, T1: np.ndarray, T2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """picks the valid solution (point in front of both cameras)
    
    four possibile solutions are: (R1, T1), (R2, T1), (R1, T2) and (R2, T2)

    Args:
        x_1: [3 x 1] point in first image
        x_2: [3 x 1] point in second image
        K: [3 x 3] calibration matrix
        R1: [3 x 3] first rotation matrix
        R2: [3 x 3] second rotation matrix
        T1: [3 x 1] first baseline vector
        T2: [3 x 1] second baseline vector
    
    Returns:
        R: [3 x 3] rotation matrix
        T: [3 x 1] baseline vector
    """
    solutions = [
        (R1, T1), (R1, T2), (R2, T1), (R2, T2)
    ]
    X0 = np.array([0, 0, 0])
    R0 = np.eye(3)
    for R, T in solutions:
        # project image points to object space rays
        r_1 = reproject(x_1, K, R0)
        r_2 = reproject(x_2, K, R)

        # compute intersection point in camera 1 coordinate system
        try:
            P = compute_intersection(X0.T, r_1.T, T.T, r_2.T)
        except:
            continue
        if P[2] < 0:
            continue

        # convert point to camera 2 coordinate system
        P = R @ (P - T)
        if P[2] < 0:
            continue

        return R, T

    raise ValueError("no valid solution found")
