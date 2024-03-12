import numpy as np
from scipy.linalg import inv, det, norm

from .util import make_random_points, add_random_noise, is_in_front
from ..ransac import filter_correspondance_points, filter_correspondance_points_2
from ..relative import fundamental_matrix, essential_matrix, comute_epipolar, fundamental_from_essential, solutions_from_essential, pick_valid_solution
from ..projection import project
from ..util import make_skew_symetric, build_rotation_matrix
from ..normalize import normalize_points

def test_fundamental():
    # init parameters
    K = np.array([
        [100, 0, 300],
        [0, 100, 400],
        [0, 0, 1],
    ])
    R_1 = build_rotation_matrix(0, 0, 0)
    R_2 = build_rotation_matrix(-np.pi, np.pi/2, np.pi/4)
    X0_1 = np.array([0, 0, 0])
    X0_2 = np.array([15, 200, 50])

    # test computation of fundamental and essential matrix with 8-point-algorithm
    POINT_COUNT = 20
    X = make_random_points(POINT_COUNT, [(10, 90), (10, 90), (10, 90)])
    x_1 = project(X, K, R_1, X0_1)
    x_2 = project(X, K, R_2, X0_2)
    x_1[:,:] = add_random_noise(x_1[:,:], scale=1)
    x_2[:,:] = add_random_noise(x_2[:,:], scale=1)

    F_ = fundamental_matrix(x_1, x_2)
    E_ = essential_matrix(x_1, x_2, K)

    S_b = make_skew_symetric(X0_2)
    F = inv(K).T @ S_b @ inv(R_2) @ inv(K)
    E = S_b @ inv(R_2)

    # _R1, _R2, _T1, _T2 = solutions_from_essential(E_)
    # _R, _T = pick_valid_solution(x_1[0,:], x_2[0,:], K, _R1, _R2, _T1, _T2)
    # _X = np.array([1, 2, 3])
    # print("_R", _R @ _X)
    # print("R", R_2 @ _X)
    # _T = _T * (norm(X0_2) / norm(_T))
    # print("_T", _T)
    # print("T", X0_2)
    # x_3 = project(X, K, _R, _T)
    # print(x_3 - x_2)

    # test ransac filtering
    POINT_COUNT = 25
    NOISE_COUNT = 25
    X = make_random_points(POINT_COUNT, [(30, 60), (30, 60), (30, 60)])
    x_1 = np.zeros((POINT_COUNT+NOISE_COUNT, 3))
    x_2 = np.zeros((POINT_COUNT+NOISE_COUNT, 3))
    x_1[:POINT_COUNT, :] = project(X, K, R_1, X0_1)
    x_2[:POINT_COUNT, :] = project(X, K, R_2, X0_2)
    x_1[:POINT_COUNT, :] = add_random_noise(x_1[:POINT_COUNT, :], scale=1)
    x_2[:POINT_COUNT, :] = add_random_noise(x_2[:POINT_COUNT, :], scale=1)
    x_1[POINT_COUNT:, :] = make_random_points(NOISE_COUNT, [(-1000, 1000), (-1000, 1000)])
    x_2[POINT_COUNT:, :] = make_random_points(NOISE_COUNT, [(-1000, 1000), (-1000, 1000)])

    x_1_t = normalize_points(x_1)
    x_2_t = normalize_points(x_2)
    subset, ok = filter_correspondance_points(x_1_t, x_2_t, max_iter=10000, threshold=0.0005, consensus_count=20)

    # subset, ok = filter_correspondance_points_2(x_1, x_2, K, max_iter=10000, threshold=0.1, consensus_count=20)
    print(ok, subset)

    x_1_subset = x_1[subset]
    x_2_subset = x_2[subset]

    _E = essential_matrix(x_1_subset, x_2_subset, K)
    _R1, _R2, _T1, _T2 = solutions_from_essential(_E)
    _R, _T = pick_valid_solution(x_1_subset[0,:], x_2_subset[0,:], K, _R1, _R2, _T1, _T2)
    _X = np.array([1, 2, 3])
    print("_R", _R @ _X)
    print("R", R_2 @ _X)
    _T = _T * (norm(X0_2) / norm(_T))
    print("_T", _T)
    print("T", X0_2)

    subset, ok = filter_correspondance_points(x_1, x_2, threshold=0.01, consensus_count=25)
    # print(ok, subset)
