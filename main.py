import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sfm.ransac import filter_correspondance_points, filter_correspondance_points_2
from sfm.relative import essential_matrix_2

img1 = cv.imread('./images/box_1.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('./images/box_2.jpg', cv.IMREAD_GRAYSCALE)

img1 = cv.resize(img1, (600, 800))
img2 = cv.resize(img2, (600, 800))

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

# img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3)
# plt.show()

K = np.array([
    [538.46233217, 0, 311.12815071],
    [0, 539.74600652, 393.34709933],
    [0, 0, 1],
])
N = 100

x_1 = np.zeros((N, 3))
x_2 = np.zeros((N, 3))
for i in range(N):
    m = matches[i]
    kp_1 = kp1[m.queryIdx]
    pt_1 = kp_1.pt
    x_1[i, :] = (pt_1[0], -pt_1[1], 1)
    kp_2 = kp2[m.trainIdx]
    pt_2 = kp_2.pt
    x_2[i, :] = (pt_2[0], -pt_2[1], 1)

subset, ok = filter_correspondance_points_2(x_1, x_2, K, max_iter=30000, threshold=0.00001, consensus_count=30)

x_1_subset = x_1[subset]
x_2_subset = x_2[subset]
E, s = essential_matrix_2(x_1_subset, x_2_subset, K)
print("s:", s)

matches_subset = []
for i in range(N):
    if subset[i]:
        matches_subset.append(matches[i])

img3 = cv.drawMatches(img1,kp1,img2,kp2,matches_subset,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)

plt.show()
