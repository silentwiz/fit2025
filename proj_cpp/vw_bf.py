import os
import json
import cv2
import numpy as np
import time

def load_vw_data(vw_data_path):
    """Load VW clusters: return dicts of descs and 3D points per vw_id."""
    with open(vw_data_path, 'r') as f:
        vw_data = json.load(f)
    cluster_descs = {}
    cluster_pts   = {}
    for vw_id, items in vw_data.items():
        vid = int(vw_id)
        descs = np.vstack([np.array(it['desc'],dtype=np.float32) for it in items])
        pts   = np.vstack([np.array(it['3dp'],dtype=np.float32) for it in items])
        cluster_descs[vid] = descs
        cluster_pts[vid]   = pts
    return cluster_descs, cluster_pts


def build_codebook_matcher(codebook):
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )
    flann.add([codebook])
    flann.train()
    return flann


def vw_matchers(cluster_descs):
    cluster_matchers = {}
    for vid, descs in cluster_descs.items():
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        bf.add([descs])
        cluster_matchers[vid] = bf
    return cluster_matchers


def extract_sift_query(path, resize_ratio):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")
    img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(img, None)
    return kps, desc


def assign_vw_ids(desc, codebook_matcher):
    """Assign each query descriptor to its nearest VW id."""
    if desc is None or len(desc)==0:
        return np.array([], dtype=np.int32)
    matches = codebook_matcher.knnMatch(desc, k=1)
    vw_ids = np.array([m[0].trainIdx for m in matches], dtype=np.int32)
    return vw_ids

def match_query_clusters(vw_ids, query_desc, cluster_pts, cluster_matchers, ratio=0.8):
    object_pts = []
    image_idx  = []
    for vid in np.unique(vw_ids):

        bf = cluster_matchers.get(int(vid))

        if bf is None:
            #print(f"VW : {int(vid)} Matcher not defined")
            continue

        qidx = np.where(vw_ids == vid)[0]
        if qidx.size == 0:
            continue

        qdesc = query_desc[qidx]
        matches = bf.knnMatch(qdesc, k=2)

        for pair in matches:
            if len(pair) < 2: # Must check pair
                continue
            m, n = pair 
            if m.distance < ratio * n.distance:
                object_pts.append(cluster_pts[vid][m.trainIdx])
                image_idx.append(qidx[m.queryIdx])

    if not object_pts:
        return np.zeros((0,3), np.float32), np.zeros((0,), np.int32)
    return np.vstack(object_pts), np.array(image_idx, dtype=np.int32)

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def main():

    startTotal = time.perf_counter()
    root_dir = os.getcwd()
    # query settings
    query_img_list = [
        'query/IMG_3341-3342.JPG',
        'query/IMG_3373-3374.JPG',
        'query/IMG_3446-3447.JPG',
        'query/IMG_3566-3567.JPG'
    ]
    query_center_list = {
        "IMG_3341-3342.JPG": [2.44078875924163, -0.7737608908904852, 2.7345119514144469],
        "IMG_3373-3374.JPG": [2.302423849966294, 1.7738270420756984, -3.907427444640278],
        "IMG_3446-3447.JPG": [-8.99687790703836, 10.72583146599812, -28.378997768627884],
        "IMG_3566-3567.JPG": [-41.32418791968729, 4.713223991127274, -25.31351628191326]
    }

    # camera intrinsics
    resize_ratio     = 0.35
    ori_resize_ratio = 0.4762
    fx, fy, cx, cy = 1450.0, 1450.0, 960.0, 720.0
    fx *= resize_ratio
    fy *= resize_ratio
    cx *= resize_ratio
    cy *= resize_ratio
    camera_matrix = np.array([[fx,0,cx],
                              [0,fy,cy],
                              [0,0,1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1), dtype=np.float32)

    # load VW clusters and codebook
    cluster_descs, cluster_pts = load_vw_data(os.path.join(root_dir, 'vw_data.json'))
    codebook = np.array(json.load(open('codebook.json')), dtype=np.float32)

    # build matchers
    codebook_matcher = build_codebook_matcher(codebook)
    cluster_matchers = vw_matchers(cluster_descs)

    # process each query image
    for img_rel in query_img_list:
        img_name   = os.path.basename(img_rel)
        gt_center  = query_center_list[img_name]
        img_path   = os.path.join(root_dir, img_rel)
        print(f"==== image[{img_name}] ====")
        
        start = time.perf_counter()  #<=== Fot Test
        # cluster_matchers = vw_matchers(cluster_descs)
        # SIFT extraction
        kps, desc = extract_sift_query(img_path, resize_ratio*ori_resize_ratio)
        # VW assignment
        vw_ids = assign_vw_ids(desc, codebook_matcher)
        # cluster-based matching
        obj_pts, img_idxs = match_query_clusters(vw_ids, desc, cluster_pts, cluster_matchers)

        if obj_pts.shape[0] < 4:
            print("VW 기반 PnP 실행에 충분한 매칭이 없습니다.")
            continue

        # prepare for PnP
        img_pts = np.array([kps[i].pt for i in img_idxs], dtype=np.float32)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=2, confidence=0.99, iterationsCount=1000
        )

        if not success:
            print("PnP fail 이유? 몰루???.\n")
            continue

        R = cv2.Rodrigues(rvec)[0]
        cam_pos = (-R.T @ tvec).ravel()
        error   = euclidean_distance(cam_pos, gt_center)
        end = time.perf_counter() 
        #endTotal = time.perf_counter()

        #print(f"Inliers: {len(inliers)}")
        print(f"distance err : {error*2:.2f}m \texe time : {end-start:.4f}sec\n total time : {end-startTotal:.4f}\n")

if __name__ == '__main__':
    main()