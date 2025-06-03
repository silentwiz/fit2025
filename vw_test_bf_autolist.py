import os
import json
import time
import csv
import cv2
import numpy as np
from tqdm import tqdm

from vw_test_opt_bf_v3 import (  # 실제로는 이 함수들을 정의한 파일을 import
    load_query_centers,
    extract_sift_query,
    assign_vw_ids,
    match_query_clusters,
    build_codebook_matcher,
    vw_matchers,
    euclidean_distance,
)

def run_pnp_for_query(img_path, gt_center, codebook_matcher, cluster_pts, cluster_matchers, camera_matrix, dist_coeffs, resize_ratio):
    """ 한 개 query 이미지에 대해 VW→PnP 실행, (error, pnp_time) 반환 """
    start = time.perf_counter()
    kps, desc = extract_sift_query(img_path, resize_ratio)
    vw_ids = assign_vw_ids(desc, codebook_matcher)
    obj_pts, img_idxs = match_query_clusters(vw_ids, desc, cluster_pts, cluster_matchers)
    if obj_pts.shape[0] < 4:
        return None, None

    img_pts = np.array([kps[i].pt for i in img_idxs], dtype=np.float32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=2, confidence=0.99, iterationsCount=1000
    )
    if not success:
        return None, None

    R = cv2.Rodrigues(rvec)[0]
    cam_pos = (-R.T @ tvec).ravel()
    error = euclidean_distance(cam_pos, gt_center)
    return error, (time.perf_counter() - start)

def main():
    root = os.getcwd()
    sfm_json = os.path.join(root, 'dataset', 'output', 'reconstruction_sequential', 'sfm_data.json')
    
    query_imgs = [
        'query/IMG_3342.JPG',
        'query/IMG_3374.JPG',
        'query/IMG_3447.JPG',
        'query/IMG_3567.JPG'
    ]
    # 1) GT center 로드
    query_centers = load_query_centers(sfm_json, query_imgs)

    # img res For SfM 
    # ::: example ::: original-image-resulution(4032x 3024) -> x0.24 -> for SfM-image-resulution(968x726)
    #ori_resize_ratio = 0.4762
    #ori_resize_ratio = 0.24
    ori_resize_ratio = 1

    # For camera intrinsics
    # Lower will be fast, but lower accurate
    # Almost environment, 480p is recommended
    #resize_ratio = 0.35  
    resize_ratio = 0.24
    #resize_ratio = 1

    #just fit for iPhone 13 Pro with ARkit
    fx, fy, cx, cy = 1450.0, 1450.0, 960.0, 720.0
    fx *= resize_ratio
    fy *= resize_ratio
    cx *= resize_ratio
    cy *= resize_ratio
    camera_matrix = np.array([[fx,0,cx],
                              [0,fy,cy],
                              [0,0,1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1), dtype=np.float32)

    # 3) 결과 저장용 리스트
    results = []
    # CSV 헤더
    csv_path = os.path.join(root, 'results.csv')
    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['K','query_image','error_m','pnp_time_s'])

        # 4) k_<K> 폴더 순회
        for folder in tqdm(sorted(os.listdir(root))):
            if not folder.startswith('k_'):
                continue
            K = int(folder.split('_',1)[1])
            folder_path = os.path.join(root, folder)

            # load codebook, vw_data
            codebook = np.array(json.load(open(os.path.join(folder_path,'codebook.json'))), dtype=np.float32)
            vw_data = json.load(open(os.path.join(folder_path,'vw_data.json')))
            # cluster_descs, cluster_pts 생성
            cluster_descs = {int(vid): np.vstack([np.array(it['desc'],np.float32) for it in items])
                             for vid, items in vw_data.items()}
            cluster_pts   = {int(vid): np.vstack([np.array(it['3dp'],np.float32) for it in items])
                             for vid, items in vw_data.items()}

            # matcher 준비
            codebook_matcher = build_codebook_matcher(codebook)
            cluster_matchers = vw_matchers(cluster_descs)

            # 각 query 이미지에 대해 PnP
            for rel in query_imgs:
                img_name = os.path.basename(rel)
                gt_center = query_centers[img_name]
                img_path = os.path.join(root, rel)

                error, t_pnp = run_pnp_for_query(
                    img_path, gt_center,
                    codebook_matcher, cluster_pts, cluster_matchers,
                    camera_matrix, dist_coeffs,
                    resize_ratio
                )
                # 결측치(실패)는 빈칸으로 기록
                row = [K, img_name,
                       f"{error:.4f}" if error is not None else '0',
                       f"{t_pnp:.4f}" if t_pnp is not None else '0']
                writer.writerow(row)
                results.append({
                    'K': K,
                    'image_name': img_name,
                    'gt_err': error,
                    'ext_time': t_pnp
                })

    print(f" CSV: {csv_path}")

if __name__ == '__main__':
    main()