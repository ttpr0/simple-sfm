import cv2 as cv
import numpy as np

SIFT = cv.SIFT_create()
BF = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

def detect_sift_features(img: np.ndarray, max_count: int) -> tuple[np.ndarray, np.ndarray]:
    global SIFT
    kps, des = SIFT.detectAndCompute(img, None)
    ids = [i for i in range(len(kps))]
    ids = sorted(ids, key = lambda x: kps[x].response, reverse=True)

    N = len(kps)

    x = np.zeros((N, 3))
    for i in range(N):
        kp = kps[i]
        pt = kp.pt
        x[i, :] = (pt[0], pt[1], 1)

    if N <= max_count:
        return x, des
    else:
        filt = np.zeros((N,), dtype='bool')
        for id in ids[:max_count]:
            filt[id] = True
        return x[filt], des[filt]

def match_features(des_1: np.ndarray, des_2: np.ndarray, max_count: int) -> np.ndarray:
    global BF

    matches = BF.match(des_1, des_2)
    matches = sorted(matches, key = lambda x: x.distance)
    
    count = min(max_count, len(matches))
    mp = np.zeros((count, 2), dtype=np.int32)
    for i in range(count):
        m = matches[i]
        mp[i, 0] = m.queryIdx
        mp[i, 1] = m.trainIdx
    return mp
