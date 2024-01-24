import cv2 as cv
import numpy as np

from sfm.sfm import compute_sfm


# image_names = ["./images/box_1.jpg", "./images/box_2.jpg"]
image_names = [f"./images/box/{i}.jpg" for i in range(1, 5)]
calib_names = ["./images/calib_1.jpg", "./images/calib_4.jpg", "./images/calib_5.jpg"]

K, keypoints, orientation, points = compute_sfm(image_names, calib_names)

print(orientation)
print(points)

for name in image_names:
    kp, des, gid = keypoints[name]
    print(gid.shape)
    print(gid[100:110])

IMG_SIZE = (600, 800)
OUT_PATH = "./out"

# write output
with open(f"{OUT_PATH}/boundler.out", "w") as file:
    file.write(f"{len(image_names)} 0\n")
    num = 1
    for name in image_names:
        img = cv.imread(name, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, IMG_SIZE)
        cv.imwrite(OUT_PATH + "/{:05d}.jpg".format(num), img)
        num += 1
        T, R = orientation[name]
        t = -(R @ T).reshape((3,))
        file.write(f"{K[0,0]} 0 0\n")
        file.write(f"{R[0,0]} {R[0,1]} {R[0,2]}\n")
        file.write(f"{R[1,0]} {R[1,1]} {R[1,2]}\n")
        file.write(f"{R[2,0]} {R[2,1]} {R[2,2]}\n")
        file.write(f"{t[0]} {t[1]} {t[2]}\n")
