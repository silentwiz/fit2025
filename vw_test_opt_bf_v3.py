import os
import json
import cv2
import numpy as np
import time

def load_query_centers(sfm_data_json_path, query_img_list):
    """
    sfm_data.json에서
      1) views → (basename → view_id) 맵
      2) extrinsics → (view_id → center) 맵
    을 만들어, query_img_list 기반으로 자동으로 center 딕셔너리를 반환.
    """
    with open(sfm_data_json_path, 'r') as f:
        sfm = json.load(f)

    # 1) filename → view_id
    fname_to_vid = {}
    for view in sfm.get('views', []):
        data = view['value']['ptr_wrapper']['data']
        fn = os.path.basename(data.get('filename', ''))
        vid = view['key']
        fname_to_vid[fn] = vid

    # 2) view_id → center
    vid_to_center = {}
    # OpenMVG 2.x.x: 'extrinsics' 배열에 pose 정보가 들어있습니다.
    for pose in sfm.get('extrinsics', []):
        vid = pose['key']
        center = pose['value'].get('center')
        if center is not None:
            vid_to_center[vid] = center

    # 3) query_img_list 기반으로 최종 매핑 생성
    query_center_list = {}
    for rel in query_img_list:
        bn = os.path.basename(rel)
        if bn not in fname_to_vid:
            raise KeyError(f"sfm_data.json의 views에서 '{bn}' 을 찾을 수 없습니다.")
        vid = fname_to_vid[bn]
        if vid not in vid_to_center:
            raise KeyError(f"sfm_data.json의 extrinsics에서 view_id={vid} 의 center를 찾을 수 없습니다.")
        query_center_list[bn] = vid_to_center[vid]

    return query_center_list

def write_ply(filename, points, colors):
    assert points.shape[0] == colors.shape[0]
    n = points.shape[0]
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

def visualize_camera_pose(filename, camera_position, R, points_3d=None):
    forward_vector = R.T @ np.array([0, 0, 1], dtype=np.float32).reshape(3,1)
    direction_end = camera_position + (forward_vector.reshape(3,) * 2.0)
    pts = [camera_position, direction_end]
    colors = [[255,0,0], [0,0,255]]
    if points_3d is not None:
        pts.extend(points_3d)
        colors.extend([[200,200,200]] * len(points_3d))
    pts = np.array(pts, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("element edge 1\nproperty int vertex1\nproperty int vertex2\nend_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
        f.write("0 1\n")

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
            if len(pair) < 2:           # 반드시 2개 이상인지 확인
                continue
            m, n = pair               # 이제 안전하게 언패킹
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
    sfm_json = os.path.join(root_dir, 'dataset', 'output','reconstruction_sequential', 'sfm_data.json')

    # query settings
    query_img_list = [
        'query/IMG_3342.JPG',
        'query/IMG_3374.JPG',
        'query/IMG_3447.JPG',
        'query/IMG_3567.JPG'
    ]
   
    query_center_list = load_query_centers(sfm_json, query_img_list)
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
        visualize_camera_pose("pnp.ply", cam_pos, R, obj_pts)

        #print(f"Inliers: {len(inliers)}")
        print(f"distance err : {error*2:.2f}m \texe time : {end-start:.4f}sec\n total time : {end-startTotal:.4f}\n")

if __name__ == '__main__':
    main()