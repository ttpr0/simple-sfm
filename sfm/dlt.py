import numpy as np
from scipy.linalg import svd, qr, inv, norm

from .least_squares import least_squares_svd


def dlt(x: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """direkt linear transform

    Args:
        x: [n x 2] image coordinates as matrix, at least n=6 rows
        X: [n x 3] objekt coordinates as matrix

    Returns:
        K: [3 x 3] calibration matrix
        R: [3 x 3] rotation matrix
        X0: [3 x 1] vector to projection center
    """
    n = x.shape[0]
    if n < 6 or X.shape[0] != n or X.shape[1] < 3 or x.shape[1] < 2:
        raise ValueError("invalid input")

    A = np.zeros((n*2, 12))
    for i in range(0,n):
        A[2*i,:] = [-X[i,0], -X[i,1], -X[i,2], 1, 0, 0, 0, 0, X[i,0]*x[i,0], X[i,1]*x[i,0], X[i,2]*x[i,0], x[i,0]]
        A[2*i+1,:] = [0, 0, 0, 0, -X[i,0], -X[i,1], -X[i,2], 1, X[i,0]*x[i,1], X[i,1]*x[i,1], X[i,2]*x[i,1], x[i,1]]

    p, _ = least_squares_svd(A)
    p = np.reshape(p, (3,4))

    H = p[:,0:3]
    h = p[:,3]

    X0: np.ndarray = -inv(H) @ h
    X0 = X0.reshape((3,1))

    Q, R = qr(inv(H))
    K = inv(R)
    K = K / K[2,2]
    R = Q.T

    return K, R, X0
