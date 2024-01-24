import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sfm.features import detect_sift_features, match_features, detect_sift_features_2, match_features_2
from sfm.ransac import filter_correspondance_points_2
from sfm.relative import essential_matrix_2

img1 = cv.imread('./images/box_1.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('./images/box_2.jpg', cv.IMREAD_GRAYSCALE)

img1 = cv.resize(img1, (600, 800))
img2 = cv.resize(img2, (600, 800))

x_1, kp_1, des_1 = detect_sift_features_2(img1)
x_2, kp_2, des_2 = detect_sift_features_2(img2)

K = np.array([
    [538.46233217, 0, 311.12815071],
    [0, 539.74600652, 393.34709933],
    [0, 0, 1],
])
N = 100

mp, matches = match_features_2(des_1, des_2, N)

x_1_match = x_1[mp[:,0]]
x_2_match = x_2[mp[:,1]]

subset, ok = filter_correspondance_points_2(x_1_match, x_2_match, K, max_iter=30000, threshold=0.00001, consensus_count=30)

x_1_subset = x_1_match[subset]
x_2_subset = x_2_match[subset]
E, s = essential_matrix_2(x_1_subset, x_2_subset, K)
print("s:", s)

matches_subset = []
for i in range(N):
    if subset[i]:
        matches_subset.append(matches[i])

img3 = cv.drawMatches(img1,kp_1,img2,kp_2,matches_subset,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)

plt.show()
