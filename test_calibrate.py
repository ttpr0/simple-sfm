import cv2 as cv

from sfm.calibrate import calibrate_from_checkerboards

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)

img_names = ["./images/calib_1.jpg", "./images/calib_4.jpg", "./images/calib_5.jpg"]
images = []
for name in img_names:
    img = cv.imread(name)
    img = cv.resize(img, (600, 800))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    images.append(gray)

K, dist = calibrate_from_checkerboards(images, CHECKERBOARD)

print("Camera matrix : \n")
print(K)
print("dist : \n")
print(dist)
