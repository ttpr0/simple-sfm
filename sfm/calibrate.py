import cv2 as cv
import numpy as np
from scipy.linalg import inv, cholesky
from scipy.optimize import least_squares, approx_fprime
import matplotlib.pyplot as plt

from .least_squares import least_squares_svd

def calibrate_from_checkerboards(images: list[np.ndarray], checkerboard: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []

    # defining object-space coordinates for checkerboard
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), dtype=np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    for img in images:
        # Find corners
        ret, corners = cv.findChessboardCorners(img, checkerboard, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        # if corners are detected
        if ret == True:
            # add points in object-space
            objpoints.append(objp)

            # refining pixel coordinates
            corners2 = cv.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)

            # add points in image-space
            corners2 = corners2.reshape(-1, 2)
            imgpoints.append(corners2)

        # plt.imshow(img)
        # plt.show()

    # compute calibration
    _, K, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    dist = dist.reshape((-1,))[:5]

    return K, dist

def calibrate_from_charuco(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
    board = cv.aruco.CharucoBoard((5,7), 1, 0.6, dictionary)
    detector = cv.aruco.CharucoDetector(board)

    obj_points = []
    img_points = []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for image in images:
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
        # img_corners = cv.cornerSubPix(image, charuco_corners, (11,11), (-1,-1), criteria)
        img_corners = charuco_corners
        obj_corners = board.getChessboardCorners()
        obj_points.append(obj_corners[charuco_ids].reshape(-1, 3))
        img_points.append(img_corners.reshape(-1, 2))

    # compute calibration
    _, K, dist, _, _ = cv.calibrateCamera(obj_points, img_points, images[0].shape[::-1], None, None)
    dist = dist.reshape((-1,))[:5]

    return K, dist

def calibrate_from_charuco_2(images: list[np.ndarray]) -> np.ndarray:
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
    board = cv.aruco.CharucoBoard((5,7), 1, 0.6, dictionary)
    detector = cv.aruco.CharucoDetector(board)

    obj_points = []
    img_points = []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for image in images:
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
        img_corners = cv.cornerSubPix(image, charuco_corners, (11,11), (-1,-1), criteria)
        img_corners = charuco_corners
        obj_corners = board.getChessboardCorners()
        obj_points.append(obj_corners[charuco_ids].reshape(-1, 3))
        img_points.append(img_corners.reshape(-1, 2))
    # compute homography
    Hs = []
    for img_pts, obj_pts in zip(img_points, obj_points):
        A = np.zeros((img_pts.shape[0]*2, 9))
        for i in range(img_pts.shape[0]):
            xi = img_pts[i,0]
            yi = img_pts[i,1]
            Xi = obj_pts[i,0]
            Yi = obj_pts[i,1]
            A[2*i, :] = np.array([-Xi, -Yi, -1, 0, 0, 0, xi*Xi, xi*Yi, xi])
            A[2*i+1, :] = np.array([0, 0, 0, -Xi, -Yi, -1, yi*Xi, yi*Yi, yi])
        h, _ = least_squares_svd(A)
        H = h.reshape((3, 3))
        opt = np.array([obj_pts[0,0], obj_pts[0,1], 1])
        ipt = H @ opt
        H /= ipt[2]
        Hs.append(H)
    # compute B
    A = np.zeros((len(Hs)*2, 6))
    def v_ij(H, i, j):
        return np.array([H[0,i]*H[0,j], H[0,i]*H[1,j]+H[1,i]*H[0,j], H[2,i]*H[0,j]+H[0,i]*H[2,j], H[1,i]*H[1,j], H[2,i]*H[1,j]+H[1,i]*H[2,j], H[2,i]*H[2,j]])
    for i, H in enumerate(Hs):
        A[2*i, :] = v_ij(H, 0, 1)
        A[2*i+1, :] = v_ij(H, 0, 0) - v_ij(H, 1, 1)
    b, _ = least_squares_svd(A)
    B = np.array([
        [b[0], b[1], b[2]],
        [b[1], b[3], b[4]],
        [b[2], b[4], b[5]]
    ])
    # extract K
    # v0 = (B[0,1]*B[0,2] - B[0,0]*B[1,2]) / (B[0,0]*B[1,1] - B[0,1]**2)
    # lam = B[2,2] - (B[0,2]**2 + v0*(B[0,1]*B[0,2] - B[0,0]*B[1,2])) / B[0,0]
    # alpha = np.sqrt(lam/B[0,0])
    # beta = np.sqrt(lam*B[0,0]/(B[0,0]*B[1,1] - B[0,1]**2))
    # gamma = -B[0,1]*(alpha**2)*beta/lam
    # u0 = gamma*v0/beta - B[0,2]*alpha**2/lam
    # K = np.array([
    #     [alpha, gamma, u0],
    #     [0, beta, v0],
    #     [0, 0, 1]
    # ])
    L = cholesky(B, lower=True)
    K0 = inv(L.T)
    K0 /= K0[2,2]
    Ps = []
    for H in Hs:
        K_inv = inv(K0)
        # lam = 1 / np.linalg.norm(K_inv@H[:,0])
        # P = np.zeros((3,4))
        # P[:,0] = lam * K_inv @ H[:,0]
        # P[:,1] = lam * K_inv @ H[:,1]
        # P[:,2] = np.cross(P[:,0], P[:,1])
        # P[:,3] = lam * K_inv @ H[:,2]
        T = K_inv @ H
        lam = (np.linalg.norm(T[:,0]) + np.linalg.norm(T[:,1])) / 2
        T /= lam
        P = np.zeros((3,4))
        P[:,0] = T[:,0]
        P[:,1] = T[:,1]
        P[:,2] = np.cross(P[:,0], P[:,1])
        P[:,3] = T[:,2]
        Ps.append(P)
    # least squares
    def f(xk):
        K = np.array([
            [xk[0], xk[1], xk[2]],
            [0, xk[3], xk[4]],
            [0, 0, 1]
        ])
        l = np.array([])
        for i, (img_pts, obj_pts) in enumerate(zip(img_points, obj_points)):
            P = xk[5+i*12:5+(i+1)*12].reshape((3,4))
            x = (K @ P @ np.hstack([obj_pts, np.ones((obj_pts.shape[0],1))]).T).T
            x /= x[:,2].reshape((-1,1))
            l = np.append(l, x[:, :2].flatten())
        return l
    X0 = np.array([K0[0,0], K0[0,1], K0[0,2], K0[1,1], K0[1,2]])
    for P in Ps:
        X0 = np.append(X0, P.flatten())
    A = approx_fprime(X0, f, epsilon=1e-2)
    L = np.array([])
    for img_pts in img_points:
        L = np.append(L, img_pts.flatten())
    L0 = f(X0)
    l = L - L0
    x_dach = inv(A.T@A) @ A.T @ l
    X_dach = X0 + x_dach
    K = np.array([
        [X_dach[0], X_dach[1], X_dach[2]],
        [0, X_dach[3], X_dach[4]],
        [0, 0, 1]
    ])
    return K
