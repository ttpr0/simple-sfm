import cv2 as cv
import numpy as np

SIFT = cv.SIFT_create()
BF = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

def detect_sift_features(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global SIFT
    kps, des = SIFT.detectAndCompute(img, None)

    N = len(kps)

    x = np.zeros((N, 3))
    for i in range(N):
        kp = kps[i]
        pt = kp.pt
        x[i, :] = (pt[0], -pt[1], 1)

    return x, des

def detect_sift_features_2(img: np.ndarray) -> tuple[np.ndarray, list, np.ndarray]:
    global SIFT
    kps, des = SIFT.detectAndCompute(img, None)

    N = len(kps)

    x = np.zeros((N, 3))
    for i in range(N):
        kp = kps[i]
        pt = kp.pt
        x[i, :] = (pt[0], -pt[1], 1)

    return x, kps, des

def match_features(des_1: np.ndarray, des_2: np.ndarray, count: int) -> tuple[np.ndarray | None, bool]:
    global BF
    if des_1.shape[0] < count or des_2.shape[0] < count:
        return None, False

    matches = BF.match(des_1, des_2)
    matches = sorted(matches, key = lambda x: x.distance)
    if len(matches) < count:
        return None, False

    mp = np.zeros((count, 2), dtype=np.int32)
    for i in range(count):
        m = matches[i]
        mp[i, 0] = m.queryIdx
        mp[i, 1] = m.trainIdx
    return mp, True

def match_features_2(des_1: np.ndarray, des_2: np.ndarray, count: int) -> tuple[np.ndarray, list]:
    global BF

    matches = BF.match(des_1, des_2)
    matches = sorted(matches, key = lambda x: x.distance)

    mp = np.zeros((count, 2), dtype=np.int32)
    for i in range(count):
        m = matches[i]
        mp[i, 0] = m.queryIdx
        mp[i, 1] = m.trainIdx

    return mp, matches[:count]
