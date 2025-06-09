import os
import sys
import json
import shutil
import subprocess
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from cuml.cluster import KMeans as cumlKMeans

def say(msg="Finish", voice=""):
    os.system(f'say -v {voice} {msg}')

def process_for_k(k, config, sfm_data, db_features, image_id_to_filename, idfeat_to_dbidx_map):
    """k 값을 받아 해당 디렉토리에 모든 JSON 결과물을 저장."""
    k_folder = os.path.join(config['root_dir'], f"k_{k}")
    os.makedirs(k_folder, exist_ok=True)

    # 1) codebook
    codebook_json = os.path.join(k_folder, "codebook.json")
    kmeans = create_kmeans_and_assign(
        db_features,
        codebook_json=codebook_json,
        n_clusters=k,
        random_state=24
    )

    # 2) vw_data
    vw_data_json = os.path.join(k_folder, "vw_data.json")
    vw_data = indexing_vw_3d_and_desc_optimized(
        sfm_data,
        db_features,
        idfeat_to_dbidx_map,
        vw_data_json=vw_data_json
    )

    # 3) bovw_data
    bovw_data_json = os.path.join(k_folder, "bovw_data.json")
    create_bovw_data(
        db_features,
        kmeans,
        out_json=bovw_data_json
    )

    # 4) desc3d_data
    desc3d_data_json = os.path.join(k_folder, "desc3d_data.json")
    create_desc3d_data(
        sfm_data,
        db_features,
        idfeat_to_dbidx_map,
        out_json=desc3d_data_json
    )

    print(f"[k={k}] 저장 완료 → {k_folder}")



'''
brute-force
slow but most accurate
'''
def create_fp_mapping(matches_dir, image_id_to_filename, db_features, mapping_json="fp_openmvg_to_opencv.json"):
    if os.path.exists(mapping_json):
        print(f"{mapping_json} already exists, loading mapping.")
        return load_fp_mapping_json(mapping_json)

    print("Creating feature mapping between OpenMVG and OpenCV features...")
    idfeat_to_dbidx_map = {}
    for image_id, img_path in tqdm(image_id_to_filename.items()):
        feat_name = get_feat_filename_from_image(img_path)
        feat_path = os.path.join(matches_dir, feat_name)
        keypoints_info = load_feat_file(feat_path)
        db_kps = db_features[image_id]['keypoints']
        local_map = {}
        for id_feat, (x, y, sc, ori) in enumerate(keypoints_info):
            min_dist = float('inf')
            best_idx = None
            for db_idx, kp in enumerate(db_kps):
                dist = (kp.pt[0] - x)**2 + (kp.pt[1] - y)**2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = db_idx
            if best_idx is not None:
                local_map[id_feat] = best_idx
        idfeat_to_dbidx_map[image_id] = local_map
    # 매핑 정보를 JSON으로 저장
    with open(mapping_json, 'w') as f:
        json.dump({str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in idfeat_to_dbidx_map.items()}, f, indent=2)
    print(f"Mapping saved to {mapping_json}")
    return idfeat_to_dbidx_map    


'''
KD-tree
faster but accurate is lower then brute-force
'''
def create_fp_mapping_fast(matches_dir, image_id_to_filename, db_features, mapping_json="fp_openmvg_to_opencv.json"):
    
    if os.path.exists(mapping_json):
        print(f"{mapping_json} already exists, loading mapping.")
        with open(mapping_json, 'r') as f:
            return json.load(f)

    print("Creating feature mapping OpenMVG <-> OpenCV features (fast KD-Tree)")
    idfeat_to_dbidx_map = {}

    for image_id, img_path in tqdm(image_id_to_filename.items()):
        feat_name = os.path.splitext(os.path.basename(img_path))[0] + ".feat"
        feat_path = os.path.join(matches_dir, feat_name)
        keypoints = load_feat_file(feat_path)
        if not keypoints:
            idfeat_to_dbidx_map[image_id] = {}
            continue

        db_kps = db_features[image_id]['keypoints']
        pts_db = np.array([kp.pt for kp in db_kps], dtype=np.float32)
        tree = KDTree(pts_db)

        pts_q = np.array([[x, y] for x, y, _, _ in keypoints], dtype=np.float32)
        # 5) 한 개의 가장 가까운 이웃 색인(idx) 찾기
        dist, idx = tree.query(pts_q, k=1)  # dist.shape=(N,1), idx.shape=(N,1)

        # 6) 매핑 딕셔너리 생성
        local_map = {str(i): int(idx[i,0]) for i in range(len(keypoints))}
        idfeat_to_dbidx_map[image_id] = local_map

    # 7) JSON으로 저장
    # JSON으로 저장 (키는 str)
    with open(mapping_json, 'w') as f:
        json.dump({str(k): v for k, v in idfeat_to_dbidx_map.items()}, f, indent=2)
    print(f"Mapping saved to {mapping_json}")

    # 다시 불러와서 키를 int로 변환
    raw = json.load(open(mapping_json, 'r'))
    int_map = {
        int(img_id): {        # 문자열 키 → 정수 키
            int(feat_id): db_idx
            for feat_id, db_idx in featmap.items()
        }
        for img_id, featmap in raw.items()
    }
    return int_map



'''
???
'''
def create_fp_mapping_hybrid_nn(matches_dir, image_id_to_filename, db_features, mapping_json="fp_openmvg_to_opencv.json"):
    if os.path.exists(mapping_json):
        print(f"{mapping_json} already exists, loading mapping.")
        return load_fp_mapping_json(mapping_json)

    print("Creating feature mapping (brute force)...")
    idfeat_to_dbidx_map = {}

    for image_id, img_path in tqdm(image_id_to_filename.items()):
        feat_name = get_feat_filename_from_image(img_path)
        feat_path = os.path.join(matches_dir, feat_name)
        keypoints_info = load_feat_file(feat_path)
        db_kps = db_features[image_id]['keypoints']
        pts_db = np.array([kp.pt for kp in db_kps], dtype=np.float32)

        if len(pts_db) == 0 or len(keypoints_info) == 0:
            idfeat_to_dbidx_map[image_id] = {}
            continue

        nn = NearestNeighbors(n_neighbors=2,algorithm='brute').fit(pts_db)
        query_pts = np.array([[x, y] for x, y, _, _ in keypoints_info], dtype=np.float32)
        #dists, indices = tree.query(query_pts, k=10)  # 상위 10개 후보 고려
        dists, indices = nn.kneighbors(query_pts,n_neighbors=1)
        #print(f"dists : {dists}")
        local_map = { id_feat: int(indices[id_feat][0]) 
                      for id_feat in range(len(keypoints_info)) }

        idfeat_to_dbidx_map[image_id] = local_map

    with open(mapping_json, 'w') as f:
        json.dump({str(k): {str(k2): int(v2) for k2, v2 in v.items()} 
                   for k, v in idfeat_to_dbidx_map.items()}, f, indent=2)

    print(f"Mapping saved to {mapping_json}")
    return idfeat_to_dbidx_map


def setup_folders():
    current_dir = os.getcwd()
    root_dir = current_dir
    
    dataset_dir = os.path.join(root_dir, 'dataset')
    output_desc_dir = os.path.join(dataset_dir, 'desc_output')
    
    if not os.path.exists(dataset_dir):
        print(f"Error: {dataset_dir} not found.")
        sys.exit(1)
    if not os.path.exists(output_desc_dir):
        os.mkdir(output_desc_dir)
    
    # OpenMVG 실행 파일 및 카메라 센서 폭 파일 경로
    OPENMVG_SFM_BIN = "/home/silentwiz/openMVG_Build/Linux-x86_64-RELEASE"
    CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/silentwiz/develop/fit2025/sensor_width_camera_database.txt"
        
    output_dir = os.path.join(dataset_dir, "output")
    matches_dir = os.path.join(output_dir, "matches")
    reconstruction_dir = os.path.join(output_dir, "reconstruction_sequential")
    
    return {
        'root_dir': root_dir,
        'dataset_dir': dataset_dir,
        'output_desc_dir': output_desc_dir,
        'output_dir': output_dir,
        'matches_dir': matches_dir,
        'reconstruction_dir': reconstruction_dir,
        'OPENMVG_SFM_BIN': OPENMVG_SFM_BIN,
        'CAMERA_SENSOR_WIDTH_DIRECTORY': CAMERA_SENSOR_WIDTH_DIRECTORY
    }

def run_sfm(config):
    sfm_data_bin = os.path.join(config['reconstruction_dir'], "sfm_data.bin")
    if os.path.exists(sfm_data_bin):
        print("sfm_data.bin exists, SfM skip.")
        return

    if not os.path.exists(config['output_dir']):
        os.mkdir(config['output_dir'])
    if not os.path.exists(config['matches_dir']):
        os.mkdir(config['matches_dir'])
    if not os.path.exists(config['reconstruction_dir']):
        os.mkdir(config['reconstruction_dir'])
    OPENMVG_SFM_BIN = config['OPENMVG_SFM_BIN']
    camera_file_params = config['CAMERA_SENSOR_WIDTH_DIRECTORY']
    input_dir = config['dataset_dir']
    matches_dir = config['matches_dir']
    reconstruction_dir = config['reconstruction_dir']

    print("1. Intrinsics analysis")
    pIntrisics = subprocess.Popen([
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),
        "-i", input_dir,
        "-o", matches_dir,
        "-d", camera_file_params,
        "-c", "3"
    ])
    pIntrisics.wait()

    print("2. Compute features")
    pFeatures = subprocess.Popen([
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),
        "-i", os.path.join(matches_dir, "sfm_data.json"),
        "-o", matches_dir,
        "-m", "SIFT"
    ])
    pFeatures.wait()

    print("2. Compute matches")
    pMatches = subprocess.Popen([
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),
        "-i", os.path.join(matches_dir, "sfm_data.json"),
        "-o", os.path.join(matches_dir, "matches.putative.bin"),
        "-f", "1"
    ])
    pMatches.wait()

    print("2. Filter matches")
    pFiltering = subprocess.Popen([
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"),
        "-i", os.path.join(matches_dir, "sfm_data.json"),
        "-m", os.path.join(matches_dir, "matches.putative.bin"),
        "-g", "f",
        "-o", os.path.join(matches_dir, "matches.f.bin")
    ])
    pFiltering.wait()

    print("3. Do Incremental/Sequential reconstruction")
    pRecons = subprocess.Popen([
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"),
        "--sfm_engine", "INCREMENTAL",
        "--input_file", os.path.join(matches_dir, "sfm_data.json"),
        "--match_dir", matches_dir,
        "--output_dir", reconstruction_dir
    ])
    pRecons.wait()

    print("5. Colorize Structure")
    pRecons = subprocess.Popen([
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),
        "-i", os.path.join(reconstruction_dir, "sfm_data.bin"),
        "-o", os.path.join(reconstruction_dir, "colorized.ply")
    ])
    pRecons.wait()

    print("4. Structure from Known Poses (robust triangulation)")
    pRecons = subprocess.Popen([
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),
        "-i", os.path.join(reconstruction_dir, "sfm_data.bin"),
        "-m", matches_dir,
        "-o", os.path.join(reconstruction_dir, "robust.ply")
    ])
    pRecons.wait()
    print(f"{input_dir} : processing done\n\n\n\n")

def convert_sfm_data_bin_to_json(config):
    sfm_data_bin = os.path.join(config['reconstruction_dir'], "sfm_data.bin")
    sfm_data_json = os.path.join(config['reconstruction_dir'], "sfm_data.json")
    if os.path.exists(sfm_data_json):
        print("sfm_data.json exists, skip.")
        return
    print("convert sfm_data.bin ---> sfm_data.json")
    subprocess.run([
        os.path.join(config['OPENMVG_SFM_BIN'], "openMVG_main_ConvertSfM_DataFormat"),
        "-i", sfm_data_bin,
        "-o", sfm_data_json
    ], check=True)
    print("convert done.")

def load_sfm_data_json(config):
    sfm_data_json = os.path.join(config['reconstruction_dir'], "sfm_data.json")
    if not os.path.exists(sfm_data_json):
        print(f"{sfm_data_json} not found.")
        sys.exit(1)
    with open(sfm_data_json, 'r') as f:
        sfm_data = json.load(f)
    return sfm_data

def load_feat_file(feat_path):
    keypoints_info = []
    with open(feat_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 4:
                x, y, scale, orientation = map(float, parts)
                keypoints_info.append((x, y, scale, orientation))
    return keypoints_info

def get_feat_filename_from_image(image_name):
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    return base_name + ".feat"

def extract_opencv_sift_features(sfm_data, dataset_dir):
    image_id_to_filename = {}
    for view in sfm_data['views']:
        image_id = view['key']
        rel_path = view['value']['ptr_wrapper']['data']['filename']
        abs_path = os.path.join(dataset_dir, rel_path)
        if os.path.exists(abs_path):
            image_id_to_filename[image_id] = abs_path
        else:
            print(f"abs_path {abs_path} not found.")
    
    sift = cv2.SIFT_create()
    db_features = {}
    for image_id, filename in tqdm(image_id_to_filename.items()):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: cannot read {filename}")
            continue
        kps, desc = sift.detectAndCompute(img, None)
        db_features[image_id] = {
            'keypoints': kps,
            'descriptors': desc
        }
    print(f"Extracted features for {len(db_features)} images.")
    return db_features, image_id_to_filename

def load_fp_mapping_json(mapping_json):
    with open(mapping_json, 'r') as f:
        data = json.load(f)
    recovered_map = {
        int(image_id): {
            int(id_feat): int(db_idx) 
            for id_feat, db_idx in feat_map.items()
        }
        for image_id, feat_map in data.items()
    }
    return recovered_map


def create_kmeans_and_assign(db_features, codebook_json="codebook.json", n_clusters=1000, random_state=24):
    all_desc = []
    for image_id, feats in db_features.items():
        desc = feats['descriptors']
        if desc is not None and desc.shape[0] > 0:
            all_desc.append(desc)
    if len(all_desc) == 0:
        print("No descriptors found in DB. Skipping kmeans.")
        return None
    all_desc = np.vstack(all_desc)
    print(f"Total descriptor shape: {all_desc.shape}")
    kmeans = cumlKMeans(n_clusters=n_clusters, init="random",  random_state=random_state)
    kmeans.fit(all_desc)
    codebook = kmeans.cluster_centers_.astype(np.float32)
    with open(codebook_json, 'w') as f:
        json.dump(codebook.tolist(), f)
    print(f"Codebook saved to {codebook_json}")
    print("Assigning visual words to each image's descriptors...")
    for image_id, feats in tqdm(db_features.items()):
        desc = feats['descriptors']
        if desc is None or desc.shape[0] == 0:
            feats['visual_words'] = None
            continue
        vw_ids = kmeans.predict(desc)
        feats['visual_words'] = vw_ids
    return kmeans

def indexing_vw_3d_and_desc_optimized(sfm_data, db_features, idfeat_to_dbidx_map, vw_data_json="vw_data.json"):
    if os.path.exists(vw_data_json):
        print(f"{vw_data_json} already exists. Loading VW data.")
        with open(vw_data_json, 'r') as f:
            vw_data = json.load(f)
        return vw_data
    structure = sfm_data.get('structure', [])
    print(f"Number of 3D points from SfM: {len(structure)}")
    points3D_dict = {point['key']: point['value']['X'] for point in structure}
    vw_data = {}
    print("Indexing visual words to (3D point, descriptor) mapping...")
    for point in tqdm(structure):
        pid = point['key']
        xyz = points3D_dict[pid]
        observations = point['value']['observations']
        for obs in observations:
            image_id = obs['key']
            id_feat = obs['value']['id_feat']
            if image_id not in idfeat_to_dbidx_map:
                continue
            db_idx = idfeat_to_dbidx_map[image_id].get(id_feat, None)
            if db_idx is None:
                continue
            feats = db_features.get(image_id, None)
            if feats is None or feats.get('visual_words', None) is None:
                continue
            if db_idx >= len(feats['visual_words']):
                continue
            vw = feats['visual_words'][db_idx]
            descriptor = feats['descriptors'][db_idx].tolist()
            vw_str = str(vw)
            if vw_str not in vw_data:
                vw_data[vw_str] = []
            vw_data[vw_str].append({
                "3dp": xyz,
                "desc": descriptor
            })
    with open(vw_data_json, 'w') as f:
        json.dump(vw_data, f, indent=2)
    print(f"Visual words data saved to {vw_data_json}")
    return vw_data

# === 추가 부분: BoVW 히스토그램 및 2D–3D 대응 데이터 생성 ===

def create_bovw_data(db_features, kmeans, out_json="bovw_data.json"):
    """
    각 이미지에 대해 BoVW 히스토그램을 생성합니다.
    - 각 이미지의 디스크립터를 kmeans.predict로 클러스터 할당 후,
    - 클러스터 별 빈도수를 카운트하여 히스토그램 생성 (L2 정규화 수행).
    결과: { image_id: [histogram vector] }
    """
    if os.path.exists(out_json):
        print(f"{out_json} already exists, loading.")
        with open(out_json, 'r') as f:
            return json.load(f)
    n_clusters = kmeans.n_clusters
    bovw_data = {}
    for image_id, feats in tqdm(db_features.items(), desc="Creating BoVW histograms"):
        desc = feats['descriptors']
        if desc is None or desc.shape[0] == 0:
            hist = np.zeros(n_clusters, dtype=np.float32)
        else:
            cluster_ids = kmeans.predict(desc)
            hist = np.bincount(cluster_ids, minlength=n_clusters).astype(np.float32)
        norm_val = np.linalg.norm(hist)
        if norm_val > 1e-12:
            hist = hist / norm_val
        bovw_data[str(image_id)] = hist.tolist()
    with open(out_json, 'w') as f:
        json.dump(bovw_data, f, indent=2)
    print(f"BoVW histogram data saved to {out_json}")
    return bovw_data

def create_desc3d_data(sfm_data, db_features, idfeat_to_dbidx_map, out_json="desc3d_data.json"):
    """
    각 이미지에 대해 디스크립터와 대응하는 3D 포인트 데이터를 생성합니다.
    결과 형식: { image_id: [{ "desc": [128-dim], "3dp": [x,y,z] }, ...] }
    """
    if os.path.exists(out_json):
        print(f"{out_json} already exists, loading.")
        with open(out_json, 'r') as f:
            return json.load(f)
    structure = sfm_data.get('structure', [])
    print(f"Total number of 3D points from SfM: {len(structure)}")
    points3D_dict = {p['key']: p['value']['X'] for p in structure}
    desc3d_data = {}
    for point in tqdm(structure, desc="Creating desc3d_data"):
        pid = point['key']
        xyz = points3D_dict[pid]
        observations = point['value']['observations']
        for obs in observations:
            image_id = obs['key']
            feat_id = obs['value']['id_feat']
            if image_id not in idfeat_to_dbidx_map:
                continue
            db_idx = idfeat_to_dbidx_map[image_id].get(feat_id, None)
            if db_idx is None:
                continue
            feats = db_features.get(image_id, None)
            if feats is None:
                continue
            descriptors = feats['descriptors']
            if descriptors is None or db_idx >= len(descriptors):
                continue
            one_desc = descriptors[db_idx].tolist()
            if str(image_id) not in desc3d_data:
                desc3d_data[str(image_id)] = []
            desc3d_data[str(image_id)].append({
                "desc": one_desc,
                "3dp": xyz
            })
    with open(out_json, 'w') as f:
        json.dump(desc3d_data, f, indent=2)
    print(f"desc3d data saved to {out_json}")
    return desc3d_data

# === 최종 main 함수: 전체 DB 생성 및 BoVW 관련 데이터 생성 ===

def main():
    # 1. 환경 설정 및 폴더 생성
    config = setup_folders()
    print(f"Configuration: {config}")
    k_list = [50, 100, 300, 500, 700, 1000, 1500, 2000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000, 23000, 24000, 26000, 28000, 30000, 32000, 34000, 36000,
              38000, 40000, 42000, 44000, 46000, 48000, 50000]

    
    # 2. OpenMVG SfM 수행
    run_sfm(config)
    
    # 3. sfm_data.bin -> sfm_data.json 변환
    convert_sfm_data_bin_to_json(config)
    
    # 4. sfm_data.json 로드
    sfm_data = load_sfm_data_json(config)
    structure = sfm_data.get('structure', [])
    print(f"Number of 3D points from SfM: {len(structure)}")
    say(msg="OpenMVG SfM is done")
    
    # 5. OpenCV SIFT를 사용하여 데이터베이스 이미지의 특징 추출
    db_features, image_id_to_filename = extract_opencv_sift_features(sfm_data, config['dataset_dir'])
    
    # 6. OpenMVG와 OpenCV 간의 특징 매핑 생성
    fp_mapping_json = "fp_openmvg_to_opencv.json"
    idfeat_to_dbidx_map = create_fp_mapping(config['matches_dir'], image_id_to_filename, db_features, mapping_json=fp_mapping_json)
    print(f"Feature mapping created for {len(idfeat_to_dbidx_map)} images.")

    for k in k_list:
        process_for_k(
            k,
            config,
            sfm_data,
            db_features,
            image_id_to_filename,
            idfeat_to_dbidx_map
        )


if __name__ == "__main__":
    main()
