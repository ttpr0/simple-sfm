import numpy as np
from scipy.linalg import inv, det

from .relative import fundamental_matrix, comute_epipolar, essential_matrix, fundamental_from_essential

def filter_correspondance_points(x_1: np.ndarray, x_2: np.ndarray, max_iter: int = 100, threshold: float = 0.01, consensus_count: int = 12) -> tuple[np.ndarray, bool]:
    n = x_1.shape[0]
    if x_2.shape[0] != n or x_1.shape[1] != 3 or x_2.shape[1] != 3:
        raise ValueError("invalid input")

    SUB_SIZE = 8

    sub_1 = np.zeros((SUB_SIZE, 3))
    sub_2 = np.zeros((SUB_SIZE, 3))
    subset = np.zeros((n,), dtype='bool')

    for i in range(max_iter):
        cho = np.random.choice(n, SUB_SIZE, replace=False)
        for j in range(SUB_SIZE):
            sub_1[j] = x_1[cho[j], :]
            sub_2[j] = x_2[cho[j], :]

        F = fundamental_matrix(sub_1, sub_2)
        res = comute_epipolar(x_1, x_2, F)
        res = np.abs(res)

        con = 0
        for j in range(n):
            if res[j] < threshold:
                con += 1
        
        if con >= consensus_count:
            for j in range(n):
                if res[j] < threshold:
                    subset[j] = True
                else:
                    subset[j] = False
            return subset, True

    return subset, False

def filter_correspondance_points_2(x_1: np.ndarray, x_2: np.ndarray, K: np.ndarray, max_iter: int = 100, threshold: float = 0.01, consensus_count: int = 12) -> tuple[np.ndarray, bool]:
    n = x_1.shape[0]
    if x_2.shape[0] != n or x_1.shape[1] != 3 or x_2.shape[1] != 3:
        raise ValueError("invalid input")

    SUB_SIZE = 8

    sub_1 = np.zeros((SUB_SIZE, 3))
    sub_2 = np.zeros((SUB_SIZE, 3))
    subset = np.zeros((n,), dtype='bool')

    for i in range(max_iter):
        cho = np.random.choice(n, SUB_SIZE, replace=False)
        cho = np.random.choice(n, SUB_SIZE, replace=False)
        for j in range(SUB_SIZE):
            sub_1[j] = x_1[cho[j], :]
            sub_2[j] = x_2[cho[j], :]

        E = essential_matrix(sub_1, sub_2, K)
        F = fundamental_from_essential(E, K)
        F_det = det(F)
        res = comute_epipolar(x_1, x_2, F)
        res = np.abs(res)

        con = 0
        for j in range(n):
            if res[j] < threshold:
                con += 1
        
        if con >= consensus_count:
            for j in range(n):
                if res[j] < threshold:
                    subset[j] = True
                else:
                    subset[j] = False
            return subset, True

    return subset, False

def filter_correspondance_points_3(x_1: np.ndarray, x_2: np.ndarray, K: np.ndarray, max_iter: int = 100, threshold: float = 0.01, consensus_count: int = 12) -> tuple[np.ndarray, bool]:
    n = x_1.shape[0]
    if x_2.shape[0] != n or x_1.shape[1] != 3 or x_2.shape[1] != 3:
        raise ValueError("invalid input")

    SUB_SIZE = 8

    sub_1 = np.zeros((SUB_SIZE, 3))
    sub_2 = np.zeros((SUB_SIZE, 3))
    subset = np.zeros((n,), dtype='bool')
    found = False
    max_con = 0

    for i in range(max_iter):
        cho = np.random.choice(n, SUB_SIZE, replace=False)
        cho = np.random.choice(n, SUB_SIZE, replace=False)
        for j in range(SUB_SIZE):
            sub_1[j] = x_1[cho[j], :]
            sub_2[j] = x_2[cho[j], :]

        E = essential_matrix(sub_1, sub_2, K)
        F = fundamental_from_essential(E, K)
        F_det = det(F)
        res = comute_epipolar(x_1, x_2, F)
        res = np.abs(res)

        con = 0
        for j in range(n):
            if res[j] < threshold:
                con += 1
        
        if con >= consensus_count and con > max_con:
            for j in range(n):
                if res[j] < threshold:
                    subset[j] = True
                else:
                    subset[j] = False
            max_con = con
            found = True

    return subset, found
