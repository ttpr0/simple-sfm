import numpy as np
from scipy.linalg import inv, det, norm

from .relative import fundamental_matrix, comute_epipolar, essential_matrix, fundamental_from_essential, comute_epipolar_distance
from .relative import solutions_from_essential, pick_valid_solution
from .util import make_skew_symetric
from .normalize import normalize_points

def filter_correspondance_points(x_1: np.ndarray, x_2: np.ndarray, max_iter: int = 100, threshold: float = 0.01, consensus_count: int = 12, normalize: bool = True, take_first: bool = True) -> tuple[np.ndarray, bool]:
    n = x_1.shape[0]
    if x_2.shape[0] != n or x_1.shape[1] != 3 or x_2.shape[1] != 3:
        raise ValueError("invalid input")

    if normalize:
        x_1 = normalize_points(x_1)
        x_2 = normalize_points(x_2)

    SUB_SIZE = 8

    sub_1 = np.zeros((SUB_SIZE, 3))
    sub_2 = np.zeros((SUB_SIZE, 3))
    subset = np.zeros((n,), dtype='bool')
    found = False
    max_con = 0

    for i in range(max_iter):
        cho = np.random.choice(n, SUB_SIZE, replace=False)
        for j in range(SUB_SIZE):
            sub_1[j] = x_1[cho[j], :]
            sub_2[j] = x_2[cho[j], :]

        F = fundamental_matrix(sub_1, sub_2)
        # res = comute_epipolar_distance(x_1, x_2, F)
        res = comute_epipolar(x_1, x_2, F)
        res = np.abs(res)

        con = 0
        for j in range(n):
            if res[j] < threshold:
                con += 1
        
        if con >= consensus_count and con > max_con:
            found = True
            for j in range(n):
                if res[j] < threshold:
                    subset[j] = True
                else:
                    subset[j] = False
            if take_first:
                return subset, found
            else:
                max_con = con

    return subset, found

def filter_correspondance_points_2(x_1: np.ndarray, x_2: np.ndarray, K: np.ndarray, max_iter: int = 100, threshold: float = 0.01, consensus_count: int = 12, take_first: bool = True) -> tuple[np.ndarray, bool]:
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
        if np.random.rand() > 0.5:
            cho = np.random.choice(int(n/2), SUB_SIZE, replace=False)
        for j in range(SUB_SIZE):
            sub_1[j] = x_1[cho[j], :]
            sub_2[j] = x_2[cho[j], :]

        E = essential_matrix(sub_1, sub_2, K)
        try:
            _R1, _R2, _T1, _T2 = solutions_from_essential(E)
            R, T = pick_valid_solution(sub_1[0,:], sub_2[0,:], K, _R1, _R2, _T1, _T2)
            E = make_skew_symetric(T) @ inv(R)
        except:
            continue
        F = fundamental_from_essential(E, K)
        F_det = det(F)
        res = comute_epipolar(x_1, x_2, F)
        res = np.abs(res)

        con = 0
        for j in range(n):
            if res[j] < threshold:
                con += 1
        
        if con >= consensus_count and con > max_con:
            found = True
            for j in range(n):
                if res[j] < threshold:
                    subset[j] = True
                else:
                    subset[j] = False
            if take_first:
                return subset, found
            else:
                max_con = con

    return subset, found
