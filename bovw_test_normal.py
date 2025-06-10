import os
import json
import cv2
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_result(img_name, top_k, exe_time):
    plt.plot(top_k,exe_time,label = img_name)
    plt.legend()
    plt.show()
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

def load_json_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def extract_sift_features(image_path, resize_ratio=1.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clr_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    if resize_ratio != 1.0:
        img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
        clr_img = cv2.resize(clr_img, (0, 0), fx=resize_ratio, fy=resize_ratio)
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(img, None)
    return kps, desc, clr_img

def compute_bovw_histogram(descriptors, codebook):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros((codebook.shape[0],), dtype=np.float32)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(descriptors, codebook)
    
    vw_ids = [m.trainIdx for m in matches]
    hist = np.bincount(vw_ids, minlength=codebook.shape[0]).astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist

def find_top_images(query_hist, db_histograms, top_k=5):
    distances = {}
    for img_id, hist in db_histograms.items():
        hist = np.array(hist, dtype=np.float32)
        dist = np.linalg.norm(query_hist - hist)
        distances[img_id] = dist
    sorted_img_ids = sorted(distances, key=distances.get)
    return sorted_img_ids[:top_k]

def aggregate_candidate_data(candidate_ids, desc3d_data):
    candidate_descriptors = []
    candidate_3d_points = []
    for img_id in candidate_ids:
        key = str(img_id)
        if key in desc3d_data:
            for entry in desc3d_data[key]:
                candidate_descriptors.append(entry["desc"])
                candidate_3d_points.append(entry["3dp"])
    if len(candidate_descriptors) == 0:
        return None, None
    candidate_descriptors = np.array(candidate_descriptors, dtype=np.float32)
    candidate_3d_points = np.array(candidate_3d_points, dtype=np.float32)
    return candidate_descriptors, candidate_3d_points

def visualize_camera_pose(filename, camera_position, R, points_3d=None):
    forward_vector = R.T @ np.array([0, 0, 1]).reshape(3, 1)
    direction_length = 2.0
    direction_end = camera_position + (forward_vector.reshape(3,) * direction_length)
    
    all_points = [camera_position, direction_end]
    all_colors = [[255, 0, 0], [0, 0, 255]]
    
    if points_3d is not None:
        all_points.extend(points_3d)
        all_colors.extend([[200, 200, 200]] * len(points_3d))
    
    all_points = np.array(all_points)
    all_colors = np.array(all_colors)
    
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(all_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("element edge 1\nproperty int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for point, color in zip(all_points, all_colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        f.write("0 1\n")


def euclidean_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)


def main():
    query_img_list = [
        'query_1/IMG_2679.jpeg',
        'query_1/IMG_2701.jpeg',
        'query_1/IMG_2810.jpeg',
        'query_1/IMG_2820.jpeg',
        'query_1/IMG_2879.jpeg',
        'query_1/IMG_2913.jpeg',
        'query_1/IMG_2951.jpeg',
        'query_1/IMG_2962.jpeg',
        'query_1/IMG_2977.jpeg',
        'query_1/IMG_2996.jpeg',
        'query_1/IMG_3043.jpeg',
        'query_1/IMG_3099.jpeg'
    ]
    
    # k_num for directories, num : number for clustering
    k_num = 'k_11000'

    root_dir = os.getcwd()
    sfm_json = os.path.join(root_dir, 'dataset_1', 'output','reconstruction_sequential', 'sfm_data.json')


    codebook_path = os.path.join(root_dir,k_num, "codebook.json")
    bovw_data_path = os.path.join(root_dir,k_num, "bovw_data.json")
    desc3d_data_path = os.path.join(root_dir,k_num, "desc3d_data.json")
    query_center_list = load_query_centers(sfm_json, query_img_list)
    codebook = np.array(load_json_data(codebook_path), dtype=np.float32)
    db_histograms = load_json_data(bovw_data_path)
    desc3d_data = load_json_data(desc3d_data_path)

    # image resolution for 1440p(1920x1440) for ARkit currentFrame(camera intrinsics)
    # EXAMPLE 4032x3024 -> x0.4762 -> 1920.0384x1440.0288 (=1920x1440)
    #ori_resize_ratio = 0.2118
    #ori_resize_ratio = 0.24
    ori_resize_ratio = 0.4762
    #ori_resize_ratio = 0.66116
    #ori_resize_ratio = 1

    # For camera intrinsics
    # orizinal camera_matrix  x (image resolution For PnP)
    # example 1920x1440 -> x0.35 ->   672x504

    #resize_ratio = 0.2118
    #resize_ratio = 0.24
    resize_ratio = 0.35
    #resize_ratio = 1

    # This data is just fit for iPhone 13 Pro with ARkit
    fx, fy, cx, cy = 1450.0, 1450.0, 960.0, 720.0
    fx *= resize_ratio
    fy *= resize_ratio
    cx *= resize_ratio
    cy *= resize_ratio
    camera_matrix = np.array([[fx,0,cx],
                              [0,fy,cy],
                              [0,0,1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1), dtype=np.float32)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    result = {} 
    for query_img_path in query_img_list:
        img_name = os.path.basename(query_img_path)
        gt_center = query_center_list[img_name]
        query_image_path = os.path.join(root_dir, query_img_path)
        print(f'\ns====== [{img_name}] ========')
        result[img_name] = {}
        top_k_list =[]
        exe_time_list = []

        for top_k in range(1, 11):  # top_k=1~5
            #print(f"=== top_k = {top_k} ===")
            startTime = time.perf_counter()

            kps, query_desc, query_img = extract_sift_features(query_image_path, resize_ratio * ori_resize_ratio)
            query_hist = compute_bovw_histogram(query_desc, codebook)
            top_candidate_ids = find_top_images(query_hist, db_histograms, top_k=top_k)

            candidate_descriptors, candidate_3d_points = aggregate_candidate_data(top_candidate_ids, desc3d_data)
            if candidate_descriptors is None:
                print(f"[{img_name}] No candidate descriptors found.")
                continue
            
            #startTime = time.perf_counter() # For matching debug timer
            matches = bf.knnMatch(query_desc, candidate_descriptors, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                print(f"top_k = {top_k} : PnP fail reason -> good matches < 4")
                continue

            image_points = np.array([kps[m.queryIdx].pt for m in good_matches], dtype=np.float32)
            object_points = np.array([candidate_3d_points[m.trainIdx] for m in good_matches], dtype=np.float32)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=2,
                confidence=0.99,
                iterationsCount=1000
            )

            if not success:
                print(f"top_k = {top_k} : PnP solve fail")
                continue

            R, _ = cv2.Rodrigues(rvec)
            camera_position = (-R.T @ tvec).ravel()

            error = euclidean_distance(camera_position, gt_center)
            endTime = time.perf_counter()

            exe_time = endTime - startTime
            top_k_list.append(top_k)
            exe_time_list.append(exe_time)
 
            print(f"[top_k : {top_k}] : distance err : {error*3.671:.2f}m \texe time : {endTime - startTime:.4f}sec")
        plot_name = img_name + "'s matching time"
        #plot_result(plot_name, top_k_list, exe_time_list)

if __name__ == "__main__":
    main()