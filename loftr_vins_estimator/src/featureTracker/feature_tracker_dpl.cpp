#include "feature_tracker_dpl.h"
#include <chrono>
#include <algorithm>
#include <numeric>

FeatureTrackerLoFTR::FeatureTrackerLoFTR() : FeatureTracker()
{
    loftr_initialized_ = true;
    total_frames_ = 0;
    frames_since_loftr_temporal_ = 0;
    frames_since_loftr_stereo_ = 0;
    
    // 創建 LoFTR interface
    loftr_interface_ = std::make_unique<LoFTR_Interface>();
    
    // 初始化 CSV 日誌 - 根據參數決定是否啟用
    csv_logging_enabled_ = LOFTR_SHOW_PERFORMANCE_STATS;
    if (csv_logging_enabled_) {
        initializeCSVLogging();
    }
    
    // 初始化性能指標
    resetCurrentMetrics();
    
    ROS_INFO("[LoFTR] Simplified feature tracker initialized with pure strategy selection");
}

FeatureTrackerLoFTR::~FeatureTrackerLoFTR()
{
    if (csv_logging_enabled_) {
        finalizeCSVLogging();
    }
}

map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> 
FeatureTrackerLoFTR::trackImage_loftr(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_total;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;

    cur_pts.clear();
    total_frames_++;
    
    // 重置當前幀的性能指標
    resetCurrentMetrics();

    if (LOFTR_ENABLE_DEBUG_LOG) {
        ROS_INFO("[LoFTR] Processing Frame %zu", total_frames_);
    }

    // 第一幀，使用傳統方法初始化
    if (prev_pts.empty()) {
        processFirstFrame(_img);
    } 
    else {
        // 後續幀，先光流追蹤，再補充特徵
        processSubsequentFrames(_img);
    }

    // 計算去畸變座標和速度
    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    // 立體匹配 
    TicToc t_stereo;
    if (!rightImg.empty() && stereo_cam) {
        processStereoMatching(_img, rightImg);
    }
    current_metrics_.stereo_time_ms = t_stereo.toc();

    // 可視化
    if (SHOW_TRACK) {
        drawTrack(_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
    }

    // 記錄總時間
    current_metrics_.total_time_ms = t_total.toc();
    current_metrics_.features_tracked = cur_pts.size();
    
    // 寫入 CSV 日誌
    if (csv_logging_enabled_) {
        logPerformanceMetrics();
    }

    // 更新歷史數據
    prev_img = _img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for (size_t i = 0; i < cur_pts.size(); i++) {
        prevLeftPtsMap[ids[i]] = cur_pts[i];
    }
    
    // 構建特徵幀
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    
    // 左目觀測
    for (size_t i = 0; i < ids.size(); i++) {
        int feature_id = ids[i];
        double x = cur_un_pts[i].x;
        double y = cur_un_pts[i].y;
        double z = 1;
        double p_u = cur_pts[i].x;
        double p_v = cur_pts[i].y;
        double velocity_x = pts_velocity[i].x;
        double velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(0, xyz_uv_velocity);
    }
    
    // 右目觀測
    if (!rightImg.empty() && stereo_cam) {
        for (size_t i = 0; i < ids_right.size(); i++) {
            int feature_id = ids_right[i];
            double x = cur_un_right_pts[i].x;
            double y = cur_un_right_pts[i].y;
            double z = 1;
            double p_u = cur_right_pts[i].x;
            double p_v = cur_right_pts[i].y;
            double velocity_x = right_pts_velocity[i].x;
            double velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(1, xyz_uv_velocity);
        }
    }


    
    return featureFrame;
}

void FeatureTrackerLoFTR::processFirstFrame(const cv::Mat &img)
{
    ROS_INFO("[LoFTR] First frame: Using traditional initialization");
    
    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
    
    // 使用傳統方法初始化
    vector<cv::Point2f> detected_features;
    cv::goodFeaturesToTrack(img, detected_features, MAX_CNT, 0.01, MIN_DIST);
    
    for (const auto& pt : detected_features) {
        if (inBorder(pt)) {
            cur_pts.push_back(pt);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
    }
    
    ROS_INFO("[LoFTR] First frame initialized: %zu features", cur_pts.size());
}

void FeatureTrackerLoFTR::processSubsequentFrames(const cv::Mat &img)
{
    // 步驟1: 永遠先跑光流追蹤
    trackWithOpticalFlow(img);
    
    // 步驟2: 使用LoFTR校正和補充
    if (shouldUseLoFTRCorrection()) {
        correctAndSupplementWithLoFTR(img);
        frames_since_loftr_temporal_ = 0;
    } else {
        frames_since_loftr_temporal_++;
    }
    
    // 步驟3: 如果還需要更多特徵，用傳統方法補充
    if (cur_pts.size() < MAX_CNT) {
        supplementWithTraditional(img);
    }
}

void FeatureTrackerLoFTR::trackWithOpticalFlow(const cv::Mat &img)
{

    // [新增] 1. 記錄上一幀的特徵數量 (作為時序匹配率的分母)
    current_metrics_.prev_features_count = prev_pts.size();

    if (prev_pts.empty()) {
        current_metrics_.flow_tracked_count = 0;
        return;
    }

    if (prev_pts.empty()) return;
    
    TicToc t_optical_flow;
    vector<cv::Point2f> tracked_pts;
    vector<uchar> status;
    vector<float> err;
    
    // 使用預測值（如果有）
    if (hasPrediction) {
        tracked_pts = predict_pts;
        cv::calcOpticalFlowPyrLK(prev_img, img, prev_pts, tracked_pts, status, err, 
                                cv::Size(21, 21), 1,
                                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), 
                                cv::OPTFLOW_USE_INITIAL_FLOW);
        
        int success_count = std::count(status.begin(), status.end(), 1);
        if (success_count < 10) {
            cv::calcOpticalFlowPyrLK(prev_img, img, prev_pts, tracked_pts, status, err, 
                                    cv::Size(21, 21), 3);
        }
    } else {
        tracked_pts = prev_pts;
        cv::calcOpticalFlowPyrLK(prev_img, img, prev_pts, tracked_pts, status, err, 
                                cv::Size(21, 21), 3);
    }
    
    // 反向檢查
    if (FLOW_BACK) {
        vector<uchar> reverse_status;
        vector<cv::Point2f> reverse_pts = prev_pts;
        cv::calcOpticalFlowPyrLK(img, prev_img, tracked_pts, reverse_pts, reverse_status, err, 
                                cv::Size(21, 21), 1,
                                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), 
                                cv::OPTFLOW_USE_INITIAL_FLOW);
        
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] && reverse_status[i] && inBorder(tracked_pts[i]) && 
                distance(prev_pts[i], reverse_pts[i]) <= 0.5) {
                status[i] = 1;
            } else {
                status[i] = 0;
            }
        }
    }
    
    // 邊界檢查
    for (size_t i = 0; i < tracked_pts.size(); i++) {
        if (status[i] && !inBorder(tracked_pts[i])) {
            status[i] = 0;
        }
    }
    
    // 移除追蹤結果
    reduceVector(prev_pts, status);
    reduceVector(tracked_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    
    // 增加追蹤計數
    for (auto &n : track_cnt) n++;
    
    cur_pts = tracked_pts;
    
    current_metrics_.flow_tracked_count = cur_pts.size();
    // 記錄光流追蹤時間
    current_metrics_.optical_flow_time_ms = t_optical_flow.toc();
    current_metrics_.used_loftr_temporal = false;
    
    if (LOFTR_ENABLE_DEBUG_LOG) {
        ROS_INFO("[LoFTR] Optical flow tracking: %zu features remaining", cur_pts.size());
    }
}


void FeatureTrackerLoFTR::correctAndSupplementWithLoFTR(const cv::Mat &img)
{
    if (prev_img.empty() || !USE_LOFTR || !loftr_initialized_) return;
    
    TicToc t_loftr;
    
    // 執行LoFTR匹配
    auto match_result = loftr_interface_->match_images(prev_img, img);
    
    if (match_result.num_matches == 0) {
        ROS_WARN("[LoFTR] No LoFTR matches found");
        current_metrics_.loftr_time_ms = t_loftr.toc();
        return;
    }
    
    ROS_INFO("[LoFTR] Got %d LoFTR matches for correction and supplement", match_result.num_matches);
    
    int corrected_features = 0;
    int new_features = 0;
    
    // 設置mask避免新特徵過密
    setMask();
    
    for (int i = 0; i < match_result.num_matches; i++) {
        cv::Point2f prev_pt = match_result.keypoints0[i];
        cv::Point2f cur_pt = match_result.keypoints1[i];
        
        if (!inBorder(cur_pt)) continue;
        
        // 信心度檢查
        if (!match_result.confidence.empty() && 
            match_result.confidence[i] < LOFTR_MATCH_THRESHOLD) continue;
        
        // 步驟1：檢查是否可以校正現有的光流點
        bool corrected = false;
        double correction_threshold = 5.0; // 校正閾值，可配置
        
        for (size_t j = 0; j < prev_pts.size() && !corrected; j++) {
            double prev_dist = cv::norm(prev_pt - prev_pts[j]);
            
            if (prev_dist < 8.0) { // 在前一幀找到對應點
                // 找到當前幀中對應的光流點
                for (size_t k = 0; k < cur_pts.size(); k++) {
                    if (k < ids.size() && j < ids.size() && ids[k] == ids[j]) {
                        // 比較光流結果和LoFTR結果的距離
                        double flow_loftr_dist = cv::norm(cur_pts[k] - cur_pt);
                        
                        if (flow_loftr_dist > correction_threshold) {
                            // 距離較大，需要校正
                            // 但要謹慎，避免錯誤校正
                            double prev_flow_dist = cv::norm(prev_pts[j] - cur_pts[k]);
                            double prev_loftr_dist = cv::norm(prev_pts[j] - cur_pt);
                            
                            // 只有當LoFTR的一致性明顯更好時才校正
                            if (prev_loftr_dist < prev_flow_dist * 0.8) {
                                ROS_DEBUG("[LoFTR] Correcting feature ID %d: flow_dist=%.2f, using LoFTR", 
                                         ids[k], flow_loftr_dist);
                                cur_pts[k] = cur_pt;
                                corrected_features++;
                                corrected = true;
                            }
                        } else {
                            // 光流結果很好，不需要校正
                            corrected = true; // 標記為已處理，避免重複
                        }
                        break;
                    }
                }
                break;
            }
        }
        
        // 步驟2：如果沒有校正任何點，檢查是否可以添加新特徵
        if (!corrected && cur_pts.size() < MAX_CNT) {
            // 檢查是否與現有特徵點太近
            bool too_close = false;
            for (const auto& existing_pt : cur_pts) {
                if (cv::norm(cur_pt - existing_pt) < MIN_DIST) {
                    too_close = true;
                    break;
                }
            }
            
            // 檢查mask
            if (!too_close && mask.at<uchar>(cur_pt) == 255) {
                // 添加新特徵
                cur_pts.push_back(cur_pt);
                ids.push_back(n_id++);
                track_cnt.push_back(1);
                new_features++;
                
                ROS_DEBUG("[LoFTR] Added new feature with ID %d", n_id - 1);
            }
        }
    }
    
    // 記錄統計
    current_metrics_.loftr_time_ms = t_loftr.toc();
    current_metrics_.used_loftr_temporal = true;
    current_metrics_.features_added_loftr = new_features;
    
    ROS_INFO("[LoFTR] LoFTR correction: %d corrected, %d new features added", 
             corrected_features, new_features);
}

void FeatureTrackerLoFTR::supplementWithTraditional(const cv::Mat &img)
{
    setMask();
    
    int needed_features = MAX_CNT - static_cast<int>(cur_pts.size());
    if (needed_features <= 0) return;
    
    vector<cv::Point2f> new_features;
    addFeaturesUsingTraditional(img, new_features, needed_features);
    
    current_metrics_.features_added_traditional = new_features.size();
    current_metrics_.features_added_loftr = 0;
    
    // 添加新特徵到系統
    for (const auto& pt : new_features) {
        cur_pts.push_back(pt);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
    
    ROS_INFO("[LoFTR] Traditional supplement: %zu features added", new_features.size());
}

void FeatureTrackerLoFTR::supplementWithLoFTR(const cv::Mat &img)
{
    // 這個函數現在主要用於純新特徵檢測，不再混合校正邏輯
    setMask();
    
    int needed_features = MAX_CNT - static_cast<int>(cur_pts.size());
    if (needed_features <= 0) return;
    
    vector<cv::Point2f> new_features;
    TicToc t_loftr_supplement;
    
    if (USE_LOFTR && loftr_initialized_) {
        addFeaturesUsingLoFTR(img, new_features, needed_features);
    }
    
    // 如果LoFTR不足，用傳統方法補充
    int still_needed = needed_features - static_cast<int>(new_features.size());
    if (still_needed > 0) {
        size_t before_traditional = new_features.size();
        addFeaturesUsingTraditional(img, new_features, still_needed);
        current_metrics_.features_added_traditional = new_features.size() - before_traditional;
    } else {
        current_metrics_.features_added_traditional = 0;
    }
    
    // 注意：這裡的features_added_loftr只計算純新特徵，不包括校正
    current_metrics_.features_added_loftr = needed_features - still_needed;
    
    // 添加新特徵到系統
    for (const auto& pt : new_features) {
        cur_pts.push_back(pt);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
    
    ROS_INFO("[LoFTR] Supplement: %zu features added (%d LoFTR + %d traditional)", 
             new_features.size(), current_metrics_.features_added_loftr, current_metrics_.features_added_traditional);
}

void FeatureTrackerLoFTR::processStereoMatching(const cv::Mat &leftImg, const cv::Mat &rightImg)
{
    ids_right.clear();
    cur_right_pts.clear();
    cur_un_right_pts.clear();
    right_pts_velocity.clear();
    cur_un_right_pts_map.clear();
    
    if (cur_pts.empty()) return;
    
    // 步驟1：永遠先執行光流立體匹配
    stereoMatchingWithOpticalFlow(leftImg, rightImg);
    
    // 步驟2：使用 LoFTR 補充未匹配的特徵點
    if (shouldUseLoFTRStereo() && USE_LOFTR && loftr_initialized_) {
        supplementStereoWithLoFTR(leftImg, rightImg);
        frames_since_loftr_stereo_ = 0;
    } else {
        frames_since_loftr_stereo_++;
    }
    
    // 通用後處理
    if (!cur_right_pts.empty()) {
        cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
        right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts,
                                       cur_un_right_pts_map, prev_un_right_pts_map);
    }
    
    prev_un_right_pts_map = cur_un_right_pts_map;
    
    // 更新性能指標
    current_metrics_.stereo_matches = cur_right_pts.size();
    current_metrics_.used_loftr_stereo = (frames_since_loftr_stereo_ == 0);
    
    ROS_INFO("[LoFTR] Stereo processing: %zu optical flow + LoFTR supplement", cur_right_pts.size());
}


void FeatureTrackerLoFTR::stereoMatchingWithLoFTR(const cv::Mat &leftImg, const cv::Mat &rightImg)
{
    if (!USE_LOFTR || !loftr_initialized_) {
        stereoMatchingWithOpticalFlow(leftImg, rightImg);
        return;
    }
    
    ROS_INFO("[LoFTR] Using LoFTR stereo matching");
    current_metrics_.used_loftr_stereo = true;
    
    matchStereoLoFTR(leftImg, rightImg, cur_pts, ids, cur_right_pts, ids_right);
    current_metrics_.stereo_matches = cur_right_pts.size();
}

void FeatureTrackerLoFTR::stereoMatchingWithOpticalFlow(const cv::Mat &leftImg, const cv::Mat &rightImg)
{
    TicToc t_flow;
    vector<uchar> status;
    vector<float> err;
    vector<cv::Point2f> temp_right_pts;
    
    cv::calcOpticalFlowPyrLK(leftImg, rightImg, cur_pts, temp_right_pts, status, err, 
                            cv::Size(21, 21), 3);
    
    // 反向檢查
    if (FLOW_BACK) {
        vector<uchar> statusRightLeft;
        vector<cv::Point2f> reverseLeftPts;
        cv::calcOpticalFlowPyrLK(rightImg, leftImg, temp_right_pts, reverseLeftPts, 
                                statusRightLeft, err, cv::Size(21, 21), 3);
        
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] && statusRightLeft[i] && inBorder(temp_right_pts[i]) && 
                distance(cur_pts[i], reverseLeftPts[i]) <= 0.5) {
                status[i] = 1;
            } else {
                status[i] = 0;
            }
        }
    }
    
    // 構建結果 - 保持與左目特徵的對應關係
    cur_right_pts.clear();
    ids_right.clear();
    
    for (size_t i = 0; i < status.size() && i < cur_pts.size() && i < ids.size(); i++) {
        if (status[i] && inBorder(temp_right_pts[i])) {
            cur_right_pts.push_back(temp_right_pts[i]);
            ids_right.push_back(ids[i]);
        }
    }

    // [新增] 記錄立體光流的時間與數量
    current_metrics_.stereo_flow_time_ms = t_flow.toc();
    current_metrics_.stereo_flow_matches = cur_right_pts.size(); // 記錄光流貢獻數
    
    ROS_INFO("[LoFTR] Optical flow stereo: %zu matches from %zu features", 
             cur_right_pts.size(), cur_pts.size());
}
// 新增函數：LoFTR 立體補充
void FeatureTrackerLoFTR::supplementStereoWithLoFTR(const cv::Mat &leftImg, const cv::Mat &rightImg)
{
    if (!loftr_initialized_) return;
    
    TicToc t_loftr; // [新增] 開始計時
    
    // 執行 LoFTR 立體匹配
    auto match_result = loftr_interface_->match_images(leftImg, rightImg);
    
    if (match_result.num_matches == 0) {
        current_metrics_.stereo_loftr_time_ms = t_loftr.toc(); // 記錄時間
        current_metrics_.stereo_added_by_loftr = 0;
        return;
    }
    
    // 找出未被光流匹配的左目特徵點
    std::set<int> matched_left_ids(ids_right.begin(), ids_right.end());
    std::vector<int> unmatched_left_indices;
    std::vector<cv::Point2f> unmatched_left_pts;
    
    for (size_t i = 0; i < cur_pts.size() && i < ids.size(); i++) {
        if (matched_left_ids.find(ids[i]) == matched_left_ids.end()) {
            unmatched_left_indices.push_back(i);
            unmatched_left_pts.push_back(cur_pts[i]);
        }
    }
    
    if (unmatched_left_pts.empty()) {
        ROS_DEBUG("[LoFTR] All left features already matched by optical flow");
        return;
    }
    
    int loftr_supplements = 0;
    
    // 為未匹配的左目特徵尋找 LoFTR 對應點
    for (size_t i = 0; i < match_result.keypoints0.size(); i++) {
        cv::Point2f loftr_left = match_result.keypoints0[i];
        cv::Point2f loftr_right = match_result.keypoints1[i];
        
        if (!inBorder(loftr_left) || !inBorder(loftr_right)) continue;
        
        // 信心度檢查
        if (!match_result.confidence.empty() && 
            match_result.confidence[i] < LOFTR_MATCH_THRESHOLD) continue;
        
        // 找到最接近的未匹配左目特徵點
        int best_left_idx = -1;
        double min_dist = LOFTR_STEREO_SEARCH_RADIUS;
        
        for (size_t j = 0; j < unmatched_left_pts.size(); j++) {
            double dist = cv::norm(loftr_left - unmatched_left_pts[j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_left_idx = unmatched_left_indices[j];
            }
        }

        // [新增] 記錄 LoFTR 立體補充的時間與數量
        current_metrics_.stereo_loftr_time_ms = t_loftr.toc();
        current_metrics_.stereo_added_by_loftr = loftr_supplements;
        
        ROS_INFO("[LoFTR] Stereo LoFTR supplement: %d matches (%.2f ms)", 
                loftr_supplements, current_metrics_.stereo_loftr_time_ms);
        
        // 添加新的立體匹配
        if (best_left_idx != -1 && best_left_idx < static_cast<int>(ids.size())) {
            // 檢查是否與現有右目點太接近
            bool too_close = false;
            for (const auto& existing_right_pt : cur_right_pts) {
                if (cv::norm(loftr_right - existing_right_pt) < MIN_DIST / 2) {
                    too_close = true;
                    break;
                }
            }
            
            if (!too_close) {
                cur_right_pts.push_back(loftr_right);
                ids_right.push_back(ids[best_left_idx]);
                loftr_supplements++;
                
                // 從未匹配列表中移除
                auto it = std::find(unmatched_left_indices.begin(), unmatched_left_indices.end(), best_left_idx);
                if (it != unmatched_left_indices.end()) {
                    size_t idx = std::distance(unmatched_left_indices.begin(), it);
                    unmatched_left_indices.erase(it);
                    unmatched_left_pts.erase(unmatched_left_pts.begin() + idx);
                }
            }
        }
    }
    
    double loftr_stereo_time = t_loftr.toc();
    
    ROS_INFO("[LoFTR] Stereo LoFTR supplement: %d new matches added (%.2f ms)", 
             loftr_supplements, loftr_stereo_time);
}

// 策略判斷函數
bool FeatureTrackerLoFTR::shouldUseLoFTRCorrection() const
{
    return USE_LOFTR && 
           loftr_initialized_ && 
           (total_frames_ >= 5) &&  // 至少處理5幀後才開始
           (frames_since_loftr_temporal_ >= LOFTR_FRAME_INTERVAL) &&
           !prev_img.empty();
}

bool FeatureTrackerLoFTR::shouldUseLoFTRNewFeatures() const
{
    return USE_LOFTR && 
           loftr_initialized_ && 
           (total_frames_ % (LOFTR_FRAME_INTERVAL * 2) == 0);  // 新特徵間隔可以更長
}

bool FeatureTrackerLoFTR::shouldUseLoFTRStereo() const
{
    return USE_LOFTR && 
           loftr_initialized_ && 
           (frames_since_loftr_stereo_ >= LOFTR_STEREO_INTERVAL);
}

void FeatureTrackerLoFTR::addFeaturesUsingLoFTR(const cv::Mat &img,
                                               vector<cv::Point2f> &new_features, 
                                               int max_features)
{
    if (prev_img.empty()) {
        ROS_WARN("[LoFTR] No previous image for LoFTR matching");
        return;
    }
    
    // 執行LoFTR匹配
    auto match_result = loftr_interface_->match_images(prev_img, img);
    
    if (match_result.num_matches == 0) {
        ROS_WARN("[LoFTR] No LoFTR matches found for new features");
        return;
    }
    
    ROS_INFO("[LoFTR] Got %d LoFTR matches for new features", match_result.num_matches);
    
    // 只添加全新的特徵點（不與現有特徵對應）
    int added_new_features = 0;
    for (int i = 0; i < match_result.num_matches && added_new_features < max_features; i++) {
        cv::Point2f prev_pt = match_result.keypoints0[i];
        cv::Point2f cur_pt = match_result.keypoints1[i];
        
        if (!inBorder(cur_pt)) continue;
        
        // 信心度檢查
        if (!match_result.confidence.empty() && 
            match_result.confidence[i] < LOFTR_MATCH_THRESHOLD) continue;
        
        // 檢查是否與前一幀的任何特徵點接近
        bool has_prev_correspondence = false;
        for (size_t j = 0; j < prev_pts.size(); j++) {
            if (cv::norm(prev_pt - prev_pts[j]) < 8.0) {
                has_prev_correspondence = true;
                break;
            }
        }
        
        if (has_prev_correspondence) continue; // 已有對應關係，跳過
        
        // 檢查是否與當前已有特徵重疊
        bool too_close_to_existing = false;
        for (const auto& existing_pt : cur_pts) {
            if (cv::norm(cur_pt - existing_pt) < MIN_DIST) {
                too_close_to_existing = true;
                break;
            }
        }
        
        if (too_close_to_existing) continue;
        
        // Mask檢查（避免過密）
        if (mask.at<uchar>(cur_pt) != 255) continue;
        
        // 這是一個全新的特徵點
        new_features.push_back(cur_pt);
        added_new_features++;
    }
    
    ROS_INFO("[LoFTR] Added %d completely new features", added_new_features);
}

void FeatureTrackerLoFTR::addFeaturesUsingTraditional(const cv::Mat &img, 
                                                     vector<cv::Point2f> &new_features,
                                                     int max_features)
{
    vector<cv::Point2f> detected_features;
    cv::goodFeaturesToTrack(img, detected_features, max_features, 0.01, MIN_DIST, mask);
    
    for (const auto& pt : detected_features) {
        if (inBorder(pt)) {
            new_features.push_back(pt);
        }
    }
}

void FeatureTrackerLoFTR::matchStereoLoFTR(const cv::Mat &leftImg, const cv::Mat &rightImg,
                                          const vector<cv::Point2f> &cur_pts, const vector<int> &ids,
                                          vector<cv::Point2f> &cur_right_pts, vector<int> &ids_right)
{
    if (!loftr_initialized_) {
        ROS_ERROR("[LoFTR] Interface not initialized for stereo matching!");
        return;
    }
    
    // 執行LoFTR立體匹配
    auto match_result = loftr_interface_->match_images(leftImg, rightImg);
    
    if (match_result.num_matches == 0) {
        ROS_WARN("[LoFTR] No stereo matches found");
        return;
    }
    
    cur_right_pts.clear();
    ids_right.clear();
    
    // ID對應，將LoFTR左目結果對應到現有的cur_pts
    for (size_t i = 0; i < match_result.keypoints0.size(); i++) {
        cv::Point2f left_pt = match_result.keypoints0[i];
        cv::Point2f right_pt = match_result.keypoints1[i];
        
        if (!inBorder(left_pt) || !inBorder(right_pt)) {
            continue;
        }
        
        // 信心度檢查 - 使用配置文件參數
        if (!match_result.confidence.empty() && 
            match_result.confidence[i] < LOFTR_MATCH_THRESHOLD) {
            continue;
        }
        
        // 找到最接近的左目特徵點 - 使用配置文件的搜索半徑
        int best_id = -1;
        double min_dist = LOFTR_STEREO_SEARCH_RADIUS;
        
        for (size_t j = 0; j < cur_pts.size(); j++) {
            double dist = cv::norm(left_pt - cur_pts[j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_id = ids[j];
            }
        }
        
        if (best_id != -1) {
            cur_right_pts.push_back(right_pt);
            ids_right.push_back(best_id);
        }
    }
    
    if (LOFTR_ENABLE_DEBUG_LOG) {
        ROS_INFO("[LoFTR] LoFTR stereo matched %zu features", cur_right_pts.size());
    }
}

bool FeatureTrackerLoFTR::initializeLoFTR()
{
    if (!USE_LOFTR) {
        ROS_INFO("[LoFTR] LoFTR disabled in configuration");
        return false;
    }
    
    ROS_INFO("[LoFTR] Initializing LoFTR...");
    
    loftr_config_.model_path = LOFTR_MODEL_PATH;
    loftr_config_.engine_path = LOFTR_ENGINE_PATH;
    loftr_config_.input_width = LOFTR_INPUT_WIDTH;
    loftr_config_.input_height = LOFTR_INPUT_HEIGHT;
    loftr_config_.match_threshold = LOFTR_MATCH_THRESHOLD;
    loftr_config_.max_matches = LOFTR_MAX_FEATURES;
    
    switch(LOFTR_BACKEND) {
        case 0: loftr_config_.backend = LoFTR_Interface::BackendType::AUTO; break;
        case 1: loftr_config_.backend = LoFTR_Interface::BackendType::ONNX_RUNTIME; break;
        case 2: loftr_config_.backend = LoFTR_Interface::BackendType::TENSORRT; break;
        default: loftr_config_.backend = LoFTR_Interface::BackendType::AUTO; break;
    }
    
    loftr_initialized_ = loftr_interface_->initialize(loftr_config_);
    
    if (loftr_initialized_) {
        ROS_INFO("[LoFTR] Initialization successful");
        ROS_INFO("[LoFTR] Backend: %s", loftr_interface_->getBackendInfo().c_str());
    } else {
        ROS_ERROR("[LoFTR] Initialization failed");
    }
    
    return loftr_initialized_;
}

void FeatureTrackerLoFTR::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts,
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap)
{
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2BGR);

    // draw left feature point (red ----> blue)
    for (size_t j = 0; j < curLeftPts.size(); j++) {
        // longer--blue
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    
    // draw right feature point (green) 
    if (!imRight.empty() && stereo_cam) {
        for (size_t i = 0; i < curRightPts.size(); i++) {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
        }
        
        // 創建右側特徵點ID到索引的映射
        std::unordered_map<int, size_t> rightIdToIndex;
        for (size_t i = 0; i < ids_right.size(); i++) {
            rightIdToIndex[ids_right[i]] = i;
        }
        
        // 創建臨時圖像用於繪製半透明線條
        cv::Mat overlay = imTrack.clone();
        
        // 只為確實匹配成功的特徵點對繪製深紅色連接線
        for (size_t i = 0; i < curLeftIds.size(); i++) {
            int leftId = curLeftIds[i];
            auto rightIt = rightIdToIndex.find(leftId);
            
            if (rightIt != rightIdToIndex.end()) {
                // 找到匹配的右側特徵點
                size_t rightIndex = rightIt->second;
                if (rightIndex < curRightPts.size()) {
                    cv::Point2f leftPt = curLeftPts[i];
                    cv::Point2f rightPt = curRightPts[rightIndex];
                    rightPt.x += cols; // 偏移到右圖位置
                    
                    // 繪製深紅色連接線到overlay圖像
                    cv::line(overlay, leftPt, rightPt, cv::Scalar(0, 0, 139), 1); // 深紅色 (BGR: 0, 0, 139)
                }
            }
        }
        
        // 將overlay以70%透明度混合到原圖像
        cv::addWeighted(imTrack, 0.3, overlay, 0.7, 0, imTrack);
    }
    
    // draw motion trace (green arrows from previous to current left points)
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++) {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if (mapIt != prevLeftPtsMap.end()) {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}

void FeatureTrackerLoFTR::loadConfiguration()
{
    // 兼容性函數，加載 LoFTR 配置參數
    ROS_INFO("[LoFTR] Configuration loaded:");
    ROS_INFO("[LoFTR]   Temporal interval: %d", LOFTR_FRAME_INTERVAL);
    ROS_INFO("[LoFTR]   Stereo interval: %d", LOFTR_STEREO_INTERVAL);
    ROS_INFO("[LoFTR]   Use LoFTR: %s", USE_LOFTR ? "true" : "false");
    
    if (USE_LOFTR) {
        ROS_INFO("[LoFTR]   Model path: %s", LOFTR_MODEL_PATH.c_str());
        ROS_INFO("[LoFTR]   Match threshold: %.3f", LOFTR_MATCH_THRESHOLD);
        ROS_INFO("[LoFTR]   Max features: %d", LOFTR_MAX_FEATURES);
        ROS_INFO("[LoFTR]   Stereo search radius: %.1f", LOFTR_STEREO_SEARCH_RADIUS);
    }
}

// ==========================================
// CSV 日誌相關函數實現
// ==========================================

void FeatureTrackerLoFTR::initializeCSVLogging()
{
    // 創建輸出目錄
    csv_output_dir_ = "/home/lin/loftrvins_ws/output_data/loftr_debug_logs/";
    std::string mkdir_cmd = "mkdir -p " + csv_output_dir_;
    int ret = system(mkdir_cmd.c_str());
    (void)ret; // 消除編譯警告
    
    // 生成帶時間戳的文件名
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();
    
    // 初始化單一整合日誌文件
    std::string combined_filename = csv_output_dir_ + "track_analysis_" + timestamp + ".csv";
    csv_performance_log_.open(combined_filename);
    
    if (csv_performance_log_.is_open()) {
        csv_performance_log_ << "frame,timestamp,"
                            // --- 時序數據 (Temporal) ---
                            << "prev_features,"         // 上一幀特徵數
                            << "tracked_features,"      // 追蹤後特徵數
                            << "temp_match_rate,"       // [新增] 時序匹配率 (Tracked / Prev)
                            << "loftr_added_temp,"      // LoFTR時序補充數
                            // --- 立體數據 (Stereo) ---
                            << "current_features,"      // 當前總特徵數 (分母)
                            << "stereo_matches,"        // 立體匹配數 (分子)
                            << "stereo_match_rate,"     // [新增] 立體匹配率 (Stereo / Current)
                            << "stereo_flow_matches,"   // 光流貢獻
                            << "stereo_loftr_added,"    // LoFTR貢獻
                            // --- 耗時 ---
                            << "time_flow_temp,"
                            << "time_loftr_temp,"
                            << "time_flow_stereo,"
                            << "time_loftr_stereo,"
                            << "total_time_ms\n";
                            
        ROS_INFO("[LoFTR] Combined analysis CSV log initialized");
    }
}

void FeatureTrackerLoFTR::logPerformanceMetrics()
{
    if (!csv_performance_log_.is_open()) return;
    
    // 1. 計算時序匹配率 (Temporal Match Rate)
    // 公式：成功追蹤的點 / 上一幀總點數
    double temp_match_rate = 0.0;
    if (current_metrics_.prev_features_count > 0) {
        temp_match_rate = (double)current_metrics_.flow_tracked_count / current_metrics_.prev_features_count;
    }

    // 2. 計算立體匹配率 (Stereo Match Rate)
    // 公式：右眼匹配到的點 / 左眼當前總點數
    double stereo_match_rate = 0.0;
    if (!cur_pts.empty()) {
        stereo_match_rate = (double)current_metrics_.stereo_matches / cur_pts.size();
    }

    // 數據一致性檢查
    if (current_metrics_.stereo_flow_matches == 0 && current_metrics_.stereo_matches > 0 && current_metrics_.stereo_added_by_loftr == 0) {
        current_metrics_.stereo_flow_matches = current_metrics_.stereo_matches;
    }

    // 寫入 CSV
    csv_performance_log_ << total_frames_ << ","
                        << std::fixed << std::setprecision(6) << cur_time << ","
                        // 時序
                        << current_metrics_.prev_features_count << ","
                        << current_metrics_.flow_tracked_count << ","
                        << std::setprecision(4) << temp_match_rate << ","   // [輸出] 時序匹配率
                        << current_metrics_.features_added_loftr << ","
                        // 立體
                        << cur_pts.size() << ","
                        << current_metrics_.stereo_matches << ","
                        << std::setprecision(4) << stereo_match_rate << "," // [輸出] 立體匹配率
                        << current_metrics_.stereo_flow_matches << ","
                        << current_metrics_.stereo_added_by_loftr << ","
                        // 耗時
                        << std::setprecision(3) << current_metrics_.optical_flow_time_ms << ","
                        << current_metrics_.loftr_time_ms << ","
                        << current_metrics_.stereo_flow_time_ms << ","
                        << current_metrics_.stereo_loftr_time_ms << ","
                        << current_metrics_.total_time_ms << "\n";
                        
    csv_performance_log_.flush();
}



void FeatureTrackerLoFTR::finalizeCSVLogging()
{
    if (csv_performance_log_.is_open()) {
        csv_performance_log_.close();
        ROS_INFO("[LoFTR] Performance CSV log finalized");
    }
    
    if (csv_feature_log_.is_open()) {
        csv_feature_log_.close();
        ROS_INFO("[LoFTR] Feature CSV log finalized");
    }
    
    if (csv_stereo_log_.is_open()) {
        csv_stereo_log_.close();
        ROS_INFO("[LoFTR] Stereo CSV log finalized");
    }
    
    ROS_INFO("[LoFTR] All CSV logs saved to: %s", csv_output_dir_.c_str());
}

void FeatureTrackerLoFTR::resetCurrentMetrics()
{
    // ... 原有變數重置 ...
    current_metrics_.optical_flow_time_ms = 0.0;
    current_metrics_.loftr_time_ms = 0.0;
    current_metrics_.stereo_time_ms = 0.0;
    current_metrics_.total_time_ms = 0.0;
    
    // [新增] 時序計數初始化
    current_metrics_.prev_features_count = 0;
    current_metrics_.flow_tracked_count = 0;
    
    current_metrics_.features_tracked = 0;
    current_metrics_.features_added_traditional = 0;
    current_metrics_.features_added_loftr = 0;
    current_metrics_.stereo_matches = 0;
    
    current_metrics_.stereo_flow_time_ms = 0.0;
    current_metrics_.stereo_loftr_time_ms = 0.0;
    current_metrics_.stereo_flow_matches = 0;
    current_metrics_.stereo_added_by_loftr = 0;
}