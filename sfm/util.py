import numpy as np
import matplotlib.pyplot as plt

def normalize_homogeneous(x: np.ndarray) -> np.ndarray:
    """normalizes homogeneous coordinates

    Args:
        x: [n x d+1] points in homogeneous coordinates as matrix
    
    Returns:
        [n x d+1] normalized points
    """
    n = x.shape[0]
    d = x.shape[1]
    return x / np.broadcast_to(x[:, d-1].reshape((n, 1)), (n, d))

def normalize_homogeneous_inplace(x: np.ndarray):
    """normalizes homogeneous coordinates inplace

    Args:
        x: [n x d+1] points in homogeneous coordinates as matrix
    """
    n = x.shape[0]
    d = x.shape[1]
    x /= np.broadcast_to(x[:, d-1].reshape((n, 1)), (n, d))

def make_skew_symetric(X: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -X[2], X[1]],
        [X[2], 0, -X[0]],
        [-X[1], X[0], 0],
    ])

def build_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    R_z = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1],
    ])
    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)],
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)],
    ])
    return R_x @ R_y @ R_z

def plot_keypoints(ax: plt.Axes, img: np.ndarray, kp: np.ndarray):
    ax.imshow(img, cmap='gray')
    ax.scatter(kp[:, 0], kp[:, 1], s=1, marker='o')

def plot_matches(ax: plt.Axes, img_1: np.ndarray, img_2: np.ndarray, kp_1: np.ndarray, kp_2: np.ndarray):
    img_height = max(img_1.shape[0], img_2.shape[0])
    img_width = img_1.shape[1] + img_2.shape[1]
    img = np.zeros((img_height, img_width))
    img[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
    img[0:img_2.shape[0], img_1.shape[1]:img_width] = img_2
    ax.imshow(img, cmap='gray')
    n = kp_1.shape[0]
    for i in range(0, n):
        ax.plot((kp_1[i, 0], kp_2[i, 0] + img_1.shape[1]), (kp_1[i, 1], kp_2[i, 1]), 'o-', linewidth=0.5, markersize=1)

def plot_epipolar_geometry(ax: plt.Axes, img_1: np.ndarray, kp_1: np.ndarray, kp_2: np.ndarray, F: np.ndarray):
    ax.imshow(img_1, cmap='gray')
    n = kp_1.shape[0]
    for i in range(0, n):
        line = F @ kp_2[i, :]
        a = line[0]
        b = line[1]
        c = line[2]
        h = img_1.shape[0]
        w = img_1.shape[1]
        p_1 = (0, -c/b)
        if p_1[1] < 0:
            p_1 = (-c/a, 0)
        elif p_1[1] > h:
            p_1 = ((-b/a)*h-c/a, h)
        p_2 = (w, (-a/b)*w-c/b)
        if p_2[1] < 0:
            p_2 = (-c/a, 0)
        elif p_2[1] > h:
            p_2 = ((-b/a)*h-c/a, h)
        ax.plot((p_1[0], p_2[0]), (p_1[1], p_2[1]), '-', linewidth=0.5, color='blue')
    ax.scatter(kp_1[:, 0], kp_1[:, 1], s=5, marker='o')

