// main.cpp
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>
#include <vector>
#include <string>
#include <set>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// --- load_query_centers 相当 ---
std::map<std::string, cv::Vec3f>
loadQueryCenters(const std::string& sfmPath,
                 const std::vector<std::string>& queryImgs)
{
    std::ifstream ifs(sfmPath);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open " + sfmPath);
    json sfm; ifs >> sfm;

    // filename → view_id
    std::map<std::string,int> fname2vid;
    for (auto& view : sfm["views"]) {
        std::string fn = view["value"]["ptr_wrapper"]["data"]["filename"]
                           .get<std::string>();
        fn = fn.substr(fn.find_last_of("\\/")+1);
        int vid = view["key"].get<int>();
        fname2vid[fn] = vid;
    }

    // view_id → center
    std::map<int,cv::Vec3f> vid2center;
    for (auto& pose : sfm["extrinsics"]) {
        int vid = pose["key"].get<int>();
        if (pose["value"].contains("center")) {
            auto c = pose["value"]["center"];
            vid2center[vid] = cv::Vec3f(
                c[0].get<float>(),
                c[1].get<float>(),
                c[2].get<float>()
            );
        }
    }

    // 最終マップ作成
    std::map<std::string,cv::Vec3f> result;
    for (auto& rel : queryImgs) {
        std::string bn = rel.substr(rel.find_last_of("\\/")+1);
        if (!fname2vid.count(bn))
            throw std::runtime_error("No view for " + bn);
        int vid = fname2vid[bn];
        if (!vid2center.count(vid))
            throw std::runtime_error("No center for view_id=" + std::to_string(vid));
        result[bn] = vid2center[vid];
    }
    return result;
}

// --- write_ply 相当 ---
void writePLY(const std::string& fn,
              const std::vector<cv::Vec3f>& pts,
              const std::vector<cv::Vec3b>& cols)
{
    std::ofstream f(fn);
    int nVert = pts.size();
    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << nVert << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f << "end_header\n";
    for (int i = 0; i < nVert; ++i) {
        auto&p = pts[i]; auto&c = cols[i];
        f << p[0]<<" "<<p[1]<<" "<<p[2]<<" "
          << (int)c[0]<<" "<<(int)c[1]<<" "<<(int)c[2]<<"\n";
    }
}

// --- visualize_camera_pose 相当 ---
void visualizeCameraPose(const std::string& fn,
                         const cv::Vec3f& camPos,
                         const cv::Mat& R,
                         const std::vector<cv::Vec3f>& pts3d = {})
{
    // 前方ベクトル (R^T * [0,0,1])
    cv::Mat forward = R.t() * (cv::Mat_<float>(3,1)<<0,0,1);
    cv::Vec3f dir(
        forward.at<float>(0),
        forward.at<float>(1),
        forward.at<float>(2)
    );
    cv::Vec3f endPt = camPos + dir * 2.0f;

    // 頂点・色リスト
    std::vector<cv::Vec3f> verts = { camPos, endPt };
    std::vector<cv::Vec3b> cols  = { {255,0,0}, {0,0,255} };
    for (auto& p : pts3d) {
        verts.push_back(p);
        cols.push_back({200,200,200});
    }

    // PLY 書き込み（頂点＋1本のエッジ）
    std::ofstream f(fn);
    int N = verts.size();
    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << N << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f << "element edge 1\nproperty int vertex1\nproperty int vertex2\n";
    f << "end_header\n";
    for (int i = 0; i < N; ++i) {
        auto&p = verts[i]; auto&c = cols[i];
        f << p[0]<<" "<<p[1]<<" "<<p[2]<<" "
          << (int)c[0]<<" "<<(int)c[1]<<" "<<(int)c[2]<<"\n";
    }
    // エッジ：0→1
    f << "0 1\n";
}

// --- 以下は前回の VW-PnP コードと同じ ---
void loadVWData(const std::string& path,
                std::map<int,cv::Mat>& descs,
                std::map<int,cv::Mat>& pts)
{
    std::ifstream ifs(path);
    json vw; ifs >> vw;
    for (auto& [key, arr] : vw.items()) {
        int vid = std::stoi(key);
        int M = arr.size();
        int D = arr[0]["desc"].size();
        cv::Mat d(M, D, CV_32F), p(M,3,CV_32F);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < D; ++j)
                d.at<float>(i,j) = arr[i]["desc"][j].get<float>();
            for (int k = 0; k < 3; ++k)
                p.at<float>(i,k) = arr[i]["3dp"][k].get<float>();
        }
        descs[vid] = d; pts[vid] = p;
    }
}

cv::Ptr<cv::FlannBasedMatcher>
buildCodebookMatcher(const cv::Mat& codebook)
{
    auto m = cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::KDTreeIndexParams>(5),
        cv::makePtr<cv::flann::SearchParams>(50)
    );
    m->add(std::vector<cv::Mat>{codebook});
    m->train();
    return m;
}

std::map<int,cv::Ptr<cv::BFMatcher>>
buildVWMatchers(const std::map<int,cv::Mat>& descs)
{
    std::map<int,cv::Ptr<cv::BFMatcher>> mp;
    for (auto& [vid,d] : descs) {
        auto bf = cv::makePtr<cv::BFMatcher>(cv::NORM_L2,false);
        bf->add(std::vector<cv::Mat>{d});
        mp[vid] = bf;
    }
    return mp;
}

void extractSIFT(const std::string& path, double rz,
                 std::vector<cv::KeyPoint>& kps, cv::Mat& desc)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("Cannot load " + path);
    cv::resize(img, img, {}, rz, rz);
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(img, cv::noArray(), kps, desc);
}

std::vector<int> assignVWIds(const cv::Mat& desc,
                             cv::Ptr<cv::FlannBasedMatcher>& m)
{
    if (desc.empty()) return {};
    std::vector<std::vector<cv::DMatch>> ms;
    m->knnMatch(desc, ms, 1);
    std::vector<int> ids(ms.size());
    for (size_t i = 0; i < ms.size(); ++i)
        ids[i] = ms[i][0].trainIdx;
    return ids;
}

void matchClusters(const std::vector<int>& vw_ids,
                   const cv::Mat& queryDesc,
                   const std::map<int,cv::Mat>& pts,
                   const std::map<int,cv::Ptr<cv::BFMatcher>>& mps,
                   std::vector<cv::Vec3f>& outObj,
                   std::vector<int>& outImg,
                   float ratio=0.8f)
{
    std::set<int> uni(vw_ids.begin(), vw_ids.end());
    for (int vid : uni) {
        auto it = mps.find(vid);
        if (it == mps.end()) continue;
        // 対応インデックス集め
        std::vector<int> idx;
        for (int i = 0; i < (int)vw_ids.size(); ++i)
            if (vw_ids[i]==vid) idx.push_back(i);
        if (idx.empty()) continue;
        // 部分行列生成
        cv::Mat sub(idx.size(), queryDesc.cols, CV_32F);
        for (int i = 0; i < (int)idx.size(); ++i)
            queryDesc.row(idx[i]).copyTo(sub.row(i));
        // knnMatch 2
        std::vector<std::vector<cv::DMatch>> ms;
        it->second->knnMatch(sub, ms, 2);
        for (auto& pr : ms) {
            if (pr.size()<2) continue;
            auto m0 = pr[0], m1 = pr[1];
            if (m0.distance < ratio*m1.distance) {
                outObj.push_back( pts.at(vid).at<cv::Vec3f>(m0.trainIdx) );
                outImg.push_back(idx[m0.queryIdx]);
            }
        }
    }
}

double euclid(const cv::Vec3d& a, const cv::Vec3d& b) {
    return cv::norm(a-b);
}

int main(){
    using Clock = std::chrono::high_resolution_clock;
    auto tAll0 = Clock::now();

    // paths
    std::string root = std::filesystem::current_path().string();
    std::string sfmJson = root+"/dataset/output/reconstruction_sequential/sfm_data.json";

    std::vector<std::string> queryList = {
        "query/IMG_3342.JPG",
        "query/IMG_3374.JPG",
        "query/IMG_3447.JPG",
        "query/IMG_3567.JPG"
    };
    // 1) query_center_list
    std::cout << "Load Query gt";
    auto queryCenters = loadQueryCenters(sfmJson, queryList);
    auto tLoadgt = Clock::now();
    double tLd = std::chrono::duration<double>(tLoadgt - tAll0).count();
    std::cout << "Loaded time : " << tLd <<"\n";
    // intrinsics
    double rz = 0.35*0.4762;
    double fx=1450*0.35, fy=1450*0.35, cx=960*0.35, cy=720*0.35;
    cv::Mat K = (cv::Mat_<double>(3,3)<<fx,0,cx,0,fy,cy,0,0,1);
    cv::Mat dist = cv::Mat::zeros(4,1,CV_64F);

    // VW & codebook
    std::map<int,cv::Mat> clusterDescs, clusterPts;
    loadVWData(root+"/vw_data.json", clusterDescs, clusterPts);

    std::ifstream ifsCB("codebook.json");
    json jcb; ifsCB>>jcb;
    cv::Mat codebook((int)jcb.size(), (int)jcb[0].size(), CV_32F);
    for (int i=0;i<codebook.rows;i++)
      for(int j=0;j<codebook.cols;j++)
        codebook.at<float>(i,j)= jcb[i][j].get<float>();

    auto cbMatcher = buildCodebookMatcher(codebook);
    auto vwMatchers = buildVWMatchers(clusterDescs);

    // loop
    for (auto& rel : queryList) {
        std::string name = rel.substr(rel.find_last_of("\\/")+1);
        //auto gt = queryCenters[name];
        cv::Vec3f gt_f = queryCenters[name]; //   名前を gt_f に変える
        std::cout<<"==== image["<<name<<"] ====\n";
        auto t0 = Clock::now();

        std::vector<cv::KeyPoint> kps; cv::Mat desc;
        extractSIFT(rel, rz, kps, desc);
        auto vw_ids = assignVWIds(desc, cbMatcher);
        std::vector<cv::Vec3f> objPts; std::vector<int> imgIdx;
        matchClusters(vw_ids, desc, clusterPts, vwMatchers, objPts, imgIdx);

        if (objPts.size()<4) {
            std::cout<<"매칭 부족\n\n"; continue;
        }

        // PnP
        std::vector<cv::Point3f> ob3; std::vector<cv::Point2f> im2;
        for (auto&p: objPts) ob3.emplace_back(p);
            for (int idx: imgIdx) im2.emplace_back(kps[idx].pt);

                cv::Mat rvec,tvec,inliers;
                bool ok = cv::solvePnPRansac(
                    ob3, im2, K, dist,
                    rvec, tvec, false, 1000, 2.0, 0.99, inliers
                );
                if(!ok){ std::cout<<"PnP 실패\n\n"; continue; }

                // PnP 成功後...
                cv::Mat R;
                cv::Rodrigues(rvec, R);

                // cam の計算
                cv::Mat camMat = -R.t() * tvec;
                cv::Vec3d cam(
                    camMat.at<double>(0,0),
                    camMat.at<double>(1,0),
                    camMat.at<double>(2,0)
                );

                // ground truth center を Vec3f で取得済み(gt_f)
                //cv::Vec3d gt(
                //    static_cast<double>(gt_f[0]),
                //    static_cast<double>(gt_f[1]),
                //    static_cast<double>(gt_f[2])
                //);
                //double err = euclid(cam, gt);
                
                cv::Vec3d gt_d(
                    static_cast<double>(gt_f[0]),
                    static_cast<double>(gt_f[1]),
                    static_cast<double>(gt_f[2])
                );
                double err = euclid(cam, gt_d);

                auto t1 = Clock::now();
                double tRun = std::chrono::duration<double>(t1-t0).count();
                double tAll = std::chrono::duration<double>(t1-tAll0).count();

                // PLY으로 시각화
                //visualizeCameraPose("pnp.ply", cv::Vec3f(cam), R, objPts);

                std::cout
                <<"distance err : "<<err*2<<" m"
                <<"\t\texe time : "<<tRun<<" s"
                <<"\n total time : "<<tAll<<" s\n\n";
    }
    return 0;
}