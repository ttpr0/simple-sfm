import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from .calibrate import calibrate_from_checkerboards
from .features import detect_sift_features, match_features
from .ransac import filter_correspondance_points_2
from .relative import essential_matrix, solutions_from_essential, pick_valid_solution
from .util import build_rotation_matrix
from .triangulate import triangulate_point
from .dlt import dlt

def compute_sfm(image_names: list[str], calib_names: list[str]) -> tuple[np.ndarray, dict, dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    IMG_SIZE = (600, 800)
    CHECKERBOARD = (6,9)
    KP_COUNT = 5000
    MATCH_COUNT = 100

    #****************************************
    # compute camera calibration
    #****************************************
    images = []
    for name in calib_names:
        img = cv.imread(name, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, IMG_SIZE)
        images.append(img)
    K, _ = calibrate_from_checkerboards(images, CHECKERBOARD)

    #****************************************
    # detect keypoints for every image
    #****************************************
    keypoints = {}
    for name in image_names:
        img = cv.imread(name, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, IMG_SIZE)
        x, des = detect_sift_features(img, KP_COUNT)
        global_ids = np.full((x.shape[0],), -1, dtype=np.int32)
        keypoints[name] = (x, des, global_ids)

    #****************************************
    # match images
    #****************************************
    matches = {} # stores matches between image pairs with key 'name_1;name_2'
    processed = [image_names[0]]
    is_equal = {} # stores equality of object-space point ids (higher id -> lower id)
    id_count = 0 # number of object-space points created
    for name_1 in image_names[1:]:
        for name_2 in processed:
            # get keypoints of images
            x_1, des_1, gid_1 = keypoints[name_1]
            x_2, des_2, gid_2 = keypoints[name_2]

            # match features and compute filtered matches
            mp, ok = match_features(des_1, des_2, MATCH_COUNT)
            if not ok:
                continue
            x_1_match = x_1[mp[:,0]]
            x_2_match = x_2[mp[:,1]]
            subset, ok = filter_correspondance_points_2(x_1_match, x_2_match, K, max_iter=10000, threshold=0.0001, consensus_count=30)
            if not ok:
                continue
            mp_filtered = mp[subset]

            # add matches to dict
            matches[name_1 + ';' + name_2] = mp_filtered

            # set ids of object space points
            for i in range(mp_filtered.shape[0]):
                i_1 = mp_filtered[i,0]
                i_2 = mp_filtered[i,1]

                if gid_1[i_1] == -1:
                    # if point wasnt already numbered in image
                    if gid_2[i_2] == -1:
                        gid_1[i_1] = id_count
                        gid_2[i_2] = id_count
                        id_count += 1
                    else:
                        gid_1[i_1] = gid_2[i_2]
                else:
                    # if point already has id
                    if gid_2[i_2] == -1:
                        gid_2[i_2] = gid_1[i_1]
                    else:
                        # set both ids equal using is_equal dictionary
                        g1 = gid_1[i_1]
                        g2 = gid_2[i_2]
                        if g1 < g2:
                            is_equal[g2] = g1
                        elif g2 < g1:
                            is_equal[g1] = g2
        processed.append(name_1)
    # remap ids using is_equal mappings
    gid_mapping = np.zeros((id_count,), dtype=np.int32)
    for i in range(gid_mapping.shape[0]):
        id = i
        while id in is_equal:
            id = is_equal[id]
        gid_mapping[i] = id
    for name in image_names:
        _, _, gid = keypoints[name]
        for i in range(gid.shape[0]):
            if gid[i] == -1:
                continue
            gid[i] = gid_mapping[gid[i]]

    #****************************************
    # compute approximate values
    #****************************************
    orientation = {}
    points = np.ones((id_count, 4), dtype=np.float32)
    is_point = np.zeros((id_count,), dtype='bool')
    processed = []

    # select initial model
    while True:
        max_matches = 0
        name_1 = ""
        name_2 = ""
        for name, mp in matches.items():
            c = mp.shape[0]
            if c > max_matches:
                max_matches = c
                name_1, name_2 = name.split(";")
        # compute initial model
        mp = matches[name_1 + ';' + name_2]
        x_1, des_1, gid_1 = keypoints[name_1]
        x_2, des_2, gid_2 = keypoints[name_2]
        x_1_subset = x_1[mp[:,0]]
        x_2_subset = x_2[mp[:,1]]
        try:
            E = essential_matrix(x_1_subset, x_2_subset, K)
            _R1, _R2, _T1, _T2 = solutions_from_essential(E)
            R_2, T_2 = pick_valid_solution(x_1_subset[0,:], x_2_subset[0,:], K, _R1, _R2, _T1, _T2)
        except:
            del matches[name_1 + ';' + name_2]
            continue
        R_1 = build_rotation_matrix(0, 0, 0)
        T_1 = np.array([0, 0, 0])
        orientation[name_1] = (T_1, R_1)
        orientation[name_2] = (T_2, R_2)
        for i in range(mp.shape[0]):
            i_1 = mp[i, 0]
            i_2 = mp[i, 1]
            try:
                X = triangulate_point(x_1[i_1,:], x_2[i_2,:], K, R_2, T_2, R_1, T_1)
            except:
                continue
            if gid_1[i_1] != gid_2[i_2]:
                raise ValueError("TODO: fix id mapping")
            gid = gid_1[i_1]
            points[gid, :3] = X
            is_point[gid] = True
        processed.append(name_1)
        processed.append(name_2)
        break

    while True:
        # TODO: select next image
        name = ""
        max_subset_count = 0
        for n in image_names:
            if n in processed:
                continue
            subset_count = 0
            x, des, gid = keypoints[n]
            for i in range(x.shape[0]):
                if gid[i] == -1:
                    continue
                if not is_point[gid[i]]:
                    continue
                subset_count += 1
            if subset_count > max_subset_count:
                max_subset_count = subset_count
                name = n
        if name == "":
            break
        
        # compute orientation using dlt
        x, des, gid = keypoints[name]
        x_subset = np.ones((30,3), dtype=np.float32)
        X_subset = np.ones((30,4), dtype=np.float32)
        subset_count = 0
        for i in range(x.shape[0]):
            if gid[i] == -1:
                continue
            if not is_point[gid[i]]:
                continue
            x_subset[subset_count, :] = x[i]
            X_subset[subset_count, :] = points[gid[i]]
            subset_count += 1
            if subset_count == 30:
                break
        if subset_count < 10:
            processed.append(name)
            continue
        _, R, T = dlt(x_subset[:subset_count,:], X_subset[:subset_count, :])
        orientation[name] = (T, R)

        # compute new object points
        for processed_name in processed:
            # select points and orientations if matches exist
            if name + ";" + processed_name in matches:
                mp = matches[name + ";" + processed_name]
                x_1, des_1, gid_1 = keypoints[name]
                T_1, R_1 = orientation[name]
                x_2, des_2, gid_2 = keypoints[processed_name]
                T_2, R_2 = orientation[processed_name]
            elif processed_name + ";" + name in matches:
                mp = matches[processed_name + ";" + name]
                x_1, des_1, gid_1 = keypoints[processed_name]
                T_1, R_1 = orientation[processed_name]
                x_2, des_2, gid_2 = keypoints[name]
                T_2, R_2 = orientation[name]
            else:
                continue 
            # compute new point for every mapping if point does not already exist
            for i in range(mp.shape[0]):
                i_1 = mp[i, 0]
                i_2 = mp[i, 1]
                if gid_1[i_1] != gid_2[i_2]:
                    raise ValueError("TODO: fix id mapping")
                if is_point[gid_1[i_1]]:
                    continue
                try:
                    X = triangulate_point(x_1[i_1,:], x_2[i_2,:], K, R_2, T_2, R_1, T_1)
                except:
                    continue
                gid = gid_1[i_1]
                points[gid, :3] = X
                is_point[gid] = True
        processed.append(name)

    return K, keypoints, orientation, points
