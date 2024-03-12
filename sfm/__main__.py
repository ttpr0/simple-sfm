import argparse
import os
import cv2 as cv
import numpy as np

from .sfm import compute_sfm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the relative orientation of images usign SfM.")
    parser.add_argument(
        'images',
        action='store',
        type=str,
        help="specify directory containing images",
    )
    parser.add_argument(
        '-c'
        '--calib',
        action='store',
        type=str,
        help="specify directory containing calibration images",
    )
    parser.add_argument(
        '-o'
        '--out',
        action='store',
        type=str,
        help="specify output directory",
    )
    args = parser.parse_args()

    # load images
    def load_images(path):
        if not os.path.isdir(path):
            raise ValueError(f"Images directory {path} does not exist.")
        for name in os.listdir(path):
            if not os.path.isfile(os.path.join(path, name)):
                continue
            if name.endswith(".jpg"):
                yield os.path.join(path, name)
    image_names = list(load_images(args.images))
    calib_names = list(load_images(args.calib))
    K, keypoints, orientation, points = compute_sfm(image_names, calib_names)

    IMG_SIZE = (600, 800)
    OUT_PATH = args.out

    # write output
    with open(f"{OUT_PATH}/boundler.out", "w") as file:
        file.write(f"{len(image_names)} 0\n")
        num = 1
        for name in image_names:
            img = cv.imread(name, cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, IMG_SIZE)
            cv.imwrite(OUT_PATH + "/{:05d}.jpg".format(num), img)
            num += 1
            if name in orientation:
                T, R = orientation[name]
                t = (R @ T).reshape((3,)) * -1
                file.write(f"{K[0,0]} 0.0 0.0\n")
                file.write(f"{R[0,0]} {R[0,1]} {R[0,2]}\n")
                file.write(f"{R[1,0]} {R[1,1]} {R[1,2]}\n")
                file.write(f"{R[2,0]} {R[2,1]} {R[2,2]}\n")
                file.write(f"{t[0]} {t[1]} {t[2]}\n")
            else:
                file.write(f"{K[0,0]} 0.0 0.0\n")
                file.write(f"{1.0} {0.0} {0.0}\n")
                file.write(f"{0.0} {1.0} {0.0}\n")
                file.write(f"{0.0} {0.0} {1.0}\n")
                file.write(f"{0.0} {0.0} {0.0}\n")
