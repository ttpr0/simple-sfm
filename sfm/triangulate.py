import numpy as np
from scipy.linalg import inv, norm, solve


def triangulate_point(x_1: np.ndarray, x_2: np.ndarray, K: np.ndarray, R_2: np.ndarray, X0_2: np.ndarray, R_1: np.ndarray | None = None, X0_1: np.ndarray | None = None) -> np.ndarray:
    """Triangulate a batch of rays

    Args:
        x_1: [3 x 1] point in first image
        x_2: [3 x 1] point in second image
        K: [3 x 3] calibration matrix
        R_2: [3 x 3] rotation of camera 2
        X0_2: [3 x 1] projection center of camera 2
        R_1: [3 x 3] rotation of camera 1 (or None if camera 1 is object-space)
        X0_1: [3 x 1] projection center of camera 1 (or None if camera 1 is object-space)

    Output:
        X: [3 x 1] 3D points in object-space
    """
    # compute direction in corresponding camera frame
    k_x_1 = np.dot(inv(K), x_1.T)
    k_x_2 = np.dot(inv(K), x_2.T)

    # rotate rays of 2nd cam s.t. they are represented in the coord system of 1st cam
    if R_1 is not None:
        k_x_1 = np.dot(R_1, k_x_1)
    k_x_2 = np.dot(R_2, k_x_2)

    # intersect point
    if X0_1 is None:
        X0_1 = np.array([0,0,0])
    X = compute_intersection(X0_1, k_x_1, X0_2, k_x_2)

    return X.T

def triangulate_points(x_1: np.ndarray, x_2: np.ndarray, K: np.ndarray, R_2: np.ndarray, X0_2: np.ndarray, R_1: np.ndarray | None = None, X0_1: np.ndarray | None = None) -> np.ndarray:
    """Triangulate a batch of rays

    Args:
        x_1: [N x 3] points in first image
        x_2: [N x 3] points in second image
        K: [3 x 3] calibration matrix
        R_2: [3 x 3] rotation of camera 2
        X0_2: [3 x 1] projection center of camera 2
        R_1: [3 x 3] rotation of camera 1 (or None if camera 1 is object-space)
        X0_1: [3 x 1] projection center of camera 1 (or None if camera 1 is object-space)

    Output:
        X: [N x 3] 3D points in object-space
    """
    # compute direction in corresponding camera frame
    k_x_1 = np.dot(inv(K), x_1.T)
    k_x_2 = np.dot(inv(K), x_2.T)

    # rotate rays of 2nd cam s.t. they are represented in the coord system of 1st cam,
    if R_1 is not None:
        k_x_1 = np.dot(R_1, k_x_1)
    k_x_2 = np.dot(R_2, k_x_2)

    # intersect points,
    X = np.zeros(k_x_1.shape)
    if X0_1 is None:
        X0_1 = np.array([0,0,0])
    for i in range(k_x_2.shape[1]):
        X[:, i] = compute_intersection(X0_1, k_x_1[:, i], X0_2, k_x_2[:, i])

    return X.T

def compute_intersection(X0_1: np.ndarray, r_1: np.ndarray, X0_2: np.ndarray, r_2: np.ndarray) -> np.ndarray:
    """ Compute the intersection of rays of two points

    Input:
        X0_1: [3 x 1] position of 1st camera
        r_1: [3 x 1] direction vector of 1st point
        X0_2: [3 x 1] position of 2nd camera
        r_2: [3 x 1] direction vector of 2nd point
    Output:
        X: [3 x 1] point of intersection
    """
    r_1 = r_1 / norm(r_1)
    r_2 = r_2 / norm(r_2)

    # derived from geometrical constraints",
    A = np.array([[np.dot(r_1.T,r_1), np.dot(-r_2.T,r_1)],
                  [np.dot(r_1.T,r_2), np.dot(-r_2.T,r_2)]])

    b = np.array([[np.dot((X0_2-X0_1).T, r_1)],
                  [np.dot((X0_2-X0_1).T, r_2)]])
    
    x = solve(A, b) # length of rays

    # X0 - point in r where the lines should intersect
    # X1 - point in s where the lines should intersect
    F = X0_1 + x[0] * r_1
    H = X0_2 + x[1] * r_2
    
    # we need a mean point if two lines do not intersect in 3D
    X = (F + H) / 2

    return X
