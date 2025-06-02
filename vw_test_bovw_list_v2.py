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
        'query/IMG_3341-3342.JPG',
        'query/IMG_3373-3374.JPG',
        'query/IMG_3446-3447.JPG',
        'query/IMG_3566-3567.JPG'
    ]
    query_key_list = {
        "IMG_3341-3342.JPG": 73,
        "IMG_3376-3377.JPG": 102,
        "IMG_3446-3447.JPG": 172,
        "IMG_3566-3567.JPG": 275
    }
    query_center_list = {
        "IMG_3341-3342.JPG": [2.44078875924163, -0.7737608908904852, 2.7345119514144469],
        "IMG_3373-3374.JPG": [2.302423849966294, 1.7738270420756984, -3.907427444640278],
        "IMG_3446-3447.JPG": [-8.99687790703836, 10.72583146599812, -28.378997768627884],
        "IMG_3566-3567.JPG": [-41.32418791968729, 4.713223991127274, -25.31351628191326]
    }

    root_dir = os.getcwd()

    codebook_path = os.path.join(root_dir, "codebook.json")
    bovw_data_path = os.path.join(root_dir, "bovw_data.json")
    desc3d_data_path = os.path.join(root_dir, "desc3d_data.json")

    codebook = np.array(load_json_data(codebook_path), dtype=np.float32)
    db_histograms = load_json_data(bovw_data_path)
    desc3d_data = load_json_data(desc3d_data_path)

    resize_ratio = 0.35
    ori_resize_ratio = 0.4762
    fx, fy, cx, cy = 1450.0, 1450.0, 960.0, 720.0
    fx *= resize_ratio
    fy *= resize_ratio
    cx *= resize_ratio
    cy *= resize_ratio
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    result = {}  # 결과 저장
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
 
            print(f"[top_k : {top_k}] : distance err : {error*2:.2f}m \texe time : {endTime - startTime:.4f}sec")
        plot_name = img_name + "'s matching time"
        #plot_result(plot_name, top_k_list, exe_time_list)

if __name__ == "__main__":
    main()