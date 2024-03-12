import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

from ..features import detect_sift_features, match_features
from ..ransac import filter_correspondance_points_2, filter_correspondance_points
from ..relative import fundamental_matrix
from ..calibrate import calibrate_from_checkerboards
from ..normalize import transform_points
from ..util import plot_keypoints, plot_matches, plot_epipolar_geometry


def test_matching():
    IMG_SIZE = (800, 600)
    IMG_SIZE_2 = (1600, 1200)
    CHECKERBOARD = (6,9)
    KP_COUNT = 5000
    MATCH_COUNT = 1000

    # calib_names = [f"./images/box_2/calib_{i}.jpg" for i in range(1, 6)]
    # images = []
    # for name in calib_names:
    #     img = cv.imread(name, cv.IMREAD_GRAYSCALE)
    #     img = cv.resize(img, IMG_SIZE)
    #     images.append(img)
    # K, _ = calibrate_from_checkerboards(images, CHECKERBOARD)
    # print(K)

    # img1 = cv.imread('./images/box_2/box_1.jpg', cv.IMREAD_GRAYSCALE)
    # img2 = cv.imread('./images/box_2/box_2.jpg', cv.IMREAD_GRAYSCALE)
    img1 = cv.imread('./images/data/mount_rushmore/1.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('./images/data/mount_rushmore/2.jpg', cv.IMREAD_GRAYSCALE)
    # img1 = cv.imread('./images/data/pic_a.jpg', cv.IMREAD_GRAYSCALE)
    # img2 = cv.imread('./images/data/pic_b.jpg', cv.IMREAD_GRAYSCALE)

    # img1 = cv.resize(img1, IMG_SIZE)
    # img2 = cv.resize(img2, IMG_SIZE)

    x_1, des_1 = detect_sift_features(img1, KP_COUNT)
    x_2, des_2 = detect_sift_features(img2, KP_COUNT)

    mp = match_features(des_1, des_2, MATCH_COUNT)
    print(mp.shape[0])

    x_1_match = x_1[mp[:,0]]
    x_2_match = x_2[mp[:,1]]

    subset, ok = filter_correspondance_points(x_1_match, x_2_match, max_iter=10000, threshold=0.001, consensus_count=30)
    # subset, ok = filter_correspondance_points_2(x_1_match, x_2_match, K, max_iter=5000, threshold=0.001, consensus_count=30)

    x_1_subset = x_1_match[subset]
    x_2_subset = x_2_match[subset]

    F = fundamental_matrix(x_1_subset, x_2_subset)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_epipolar_geometry(ax, img1, x_1_subset, x_2_subset, F)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_matches(ax, img1, img2, x_1_subset, x_2_subset)
    plt.show()
