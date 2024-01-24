import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
