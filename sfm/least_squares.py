import numpy as np
from scipy.linalg import svd, inv, lstsq

def least_squares_svd(A: np.ndarray) -> tuple[np.ndarray, float]:
    n = A.shape[0]
    m = A.shape[1]
    U, S, V_T = svd(A)
    s = 0
    if A.shape[0] >= A.shape[1]:
        s = S[A.shape[1]-1]
    return V_T[m-1, :].copy(), s
