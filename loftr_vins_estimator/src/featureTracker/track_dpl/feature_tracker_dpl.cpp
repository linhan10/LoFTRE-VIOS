/*******************************************************
 *  LoFTR Feature Tracker - Complete Implementation
 *******************************************************/

#include "feature_tracker_dpl.h"
#include <chrono>
#include <algorithm>
#include <numeric>

FeatureTrackerLoFTR::FeatureTrackerLoFTR()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
    loftr_initialized_ = false;
    total_frames_ = 0;
    
    // ID管理 - 使用分離的範圍
    next_loftr_id_ = FeatureIDRange::LOFTR_START;
    next_traditional_id_ = FeatureIDRange::TRADITIONAL_START;
    
    tracked_features_.clear();
    feature_track_count_.clear();
    
    // 從全局參數加載配置
    loadConfiguration();
    
    // 初始化LoFTR接口
    loftr_interface_ = std::make_unique<LoFTR_Interface>();
    
    ROS_INFO("[LoFTR] Feature tracker initialized");
    ROS_INFO("[LoFTR]   Frame interval: %d", loftr_frame_interval_);
    ROS_INFO("[LoFTR]   Min feature threshold: %d", min_feature_threshold_);
    ROS_INFO("[LoFTR]   Adaptive interval: %s", adaptive_interval_ ? "Yes" : "No");
    ROS_INFO("[LoFTR]   ID ranges - Traditional: %d-%d, LoFTR: %d-%d", 
             FeatureIDRange::TRADITIONAL_START, FeatureIDRange::TRADITIONAL_END,
             FeatureIDRange::LOFTR_START, FeatureIDRange::LOFTR_END);
}

FeatureTrackerLoFTR::~FeatureTrackerLoFTR()
{
    if (loftr_initialized_) {
        printPerformanceStats();
    }
}

void FeatureTrackerLoFTR::loadConfiguration()
{
    // 從全局參數讀取所有配置
    max_features_ = LOFTR_MAX_FEATURES;
    match_threshold_ = LOFTR_MATCH_THRESHOLD;
    use_traditional_tracker_ = USE_TRADITIONAL_TRACKER;
    
    // 間隔控制參數
    loftr_frame_interval_ = LOFTR_FRAME_INTERVAL;
    min_feature_threshold_ = LOFTR_MIN_FEATURE_THRESHOLD;
    adaptive_interval_ = LOFTR_ADAPTIVE_INTERVAL;
    performance_mode_ = LOFTR_PERFORMANCE_MODE;
    
    // Stereo參數
    stereo_loftr_interval_ = LOFTR_STEREO_INTERVAL;
    stereo_failure_threshold_ = LOFTR_STEREO_FAILURE_THRESHOLD;
    
    // 初始化計數器
    frames_since_loftr_ = 0;
    frames_since_stereo_loftr_ = 0;
    use_loftr_this_frame_ = false;
    
    ROS_INFO("[LoFTR] Configuration loaded from global parameters");
}

bool FeatureTrackerLoFTR::initializeLoFTR()
{
    if (!USE_LOFTR) {
        ROS_INFO("[LoFTR] LoFTR disabled in configuration");
        return false;
    }
    
    ROS_INFO("[LoFTR] Initializing with configuration parameters...");
    
    // 檢查模型文件
    std::ifstream model_file(LOFTR_MODEL_PATH);
    if (!model_file.good()) {
        ROS_ERROR("[LoFTR] Model file not found: %s", LOFTR_MODEL_PATH.c_str());
        return false;
    }
    
    // 配置LoFTR
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

int FeatureTrackerLoFTR::allocateFeatureID(bool is_loftr_feature)
{
    int new_id;
    
    if (is_loftr_feature) {
        // 分配LoFTR ID範圍
        new_id = next_loftr_id_++;
        if (next_loftr_id_ > FeatureIDRange::LOFTR_END) {
            next_loftr_id_ = FeatureIDRange::LOFTR_START;
        }
    } else {
        // 分配傳統ID範圍
        new_id = next_traditional_id_++;
        if (next_traditional_id_ > FeatureIDRange::TRADITIONAL_END) {
            next_traditional_id_ = FeatureIDRange::TRADITIONAL_START;
        }
    }
    
    // 確保ID未被使用
    while (tracked_features_.find(new_id) != tracked_features_.end()) {
        if (is_loftr_feature) {
            new_id = next_loftr_id_++;
            if (next_loftr_id_ > FeatureIDRange::LOFTR_END) {
                next_loftr_id_ = FeatureIDRange::LOFTR_START;
            }
        } else {
            new_id = next_traditional_id_++;
            if (next_traditional_id_ > FeatureIDRange::TRADITIONAL_END) {
                next_traditional_id_ = FeatureIDRange::TRADITIONAL_START;
            }
        }
    }
    
    return new_id;
}

bool FeatureTrackerLoFTR::shouldUseLoFTRThisFrame()
{
    // 第一幀使用傳統方法初始化
    if (total_frames_ == 0) {
        ROS_INFO("[LoFTR] First frame - using traditional feature detection");
        return false;
    }
    
    // 檢查LoFTR是否初始化
    if (!loftr_initialized_ && USE_LOFTR) {
        if (!initializeLoFTR()) {
            ROS_WARN("[LoFTR] LoFTR initialization failed, using traditional tracker");
            return false;
        }
    }
    
    if (!USE_LOFTR || !loftr_initialized_) {
        return false;
    }
    
    // 檢查間隔
    bool interval_reached = (frames_since_loftr_ >= loftr_frame_interval_);
    
    // 檢查特徵數量
    bool low_features = (cur_pts.size() < static_cast<size_t>(min_feature_threshold_));
    
    // 自適應間隔
    if (adaptive_interval_) {
        if (performance_mode_ == 1) {
            // 精度優先
            if (cur_pts.size() < static_cast<size_t>(min_feature_threshold_ * 1.5)) {
                interval_reached = true;
            }
        } else if (performance_mode_ == 2) {
            // 速度優先
            if (cur_pts.size() > static_cast<size_t>(min_feature_threshold_ * 2)) {
                interval_reached = false;
            }
        }
    }
    
    bool use_loftr = interval_reached || low_features;
    
    if (LOFTR_DEBUG_INTERVAL) {
        ROS_INFO("[LoFTR] Frame %zu: interval=%d/%d, features=%zu/%d, use_loftr=%s",
                 total_frames_, frames_since_loftr_, loftr_frame_interval_,
                 cur_pts.size(), min_feature_threshold_,
                 use_loftr ? "YES" : "NO");
    }
    
    return use_loftr;
}

void FeatureTrackerLoFTR::initializeFirstFrame(const cv::Mat &img)
{
    ROS_INFO("[LoFTR] Initializing first frame with traditional features");
    
    // 設置遮罩
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
    
    // 使用傳統方法檢測角點
    vector<cv::Point2f> new_features;
    cv::goodFeaturesToTrack(img, new_features, MAX_CNT, 0.01, MIN_DIST, mask);
    
    ROS_INFO("[LoFTR] Detected %zu initial features", new_features.size());
    
    // 分配ID
    for (const auto& pt : new_features) {
        int new_id = allocateFeatureID(false);  // 使用傳統ID範圍
        
        cur_pts.push_back(pt);
        ids.push_back(new_id);
        track_cnt.push_back(1);
        
        tracked_features_[new_id] = pt;
        feature_track_count_[new_id] = 1;
    }
    
    ROS_INFO("[LoFTR] First frame initialized with %zu features", cur_pts.size());
}

void FeatureTrackerLoFTR::trackFeaturesOpticalFlow()
{
    if (prev_pts.empty()) {
        return;
    }
    
    vector<uchar> status;
    vector<float> err;
    vector<cv::Point2f> tracked_pts;
    
    // 前向光流
    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, tracked_pts, 
                            status, err, cv::Size(21, 21), 3);
    
    // 反向檢查
    if (FLOW_BACK) {
        vector<uchar> reverse_status;
        vector<cv::Point2f> reverse_pts;
        cv::calcOpticalFlowPyrLK(cur_img, prev_img, tracked_pts, reverse_pts, 
                                reverse_status, err, cv::Size(21, 21), 3);
        
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i] && reverse_status[i] && 
                distance(prev_pts[i], reverse_pts[i]) > 0.5f) {
                status[i] = 0;
            }
        }
    }
    
    // 更新跟蹤特徵
    int optical_flow_success = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] && inBorder(tracked_pts[i])) {
            cur_pts.push_back(tracked_pts[i]);
            ids.push_back(this->ids[i]);
            track_cnt.push_back(this->track_cnt[i] + 1);
            
            tracked_features_[this->ids[i]] = tracked_pts[i];
            feature_track_count_[this->ids[i]]++;
            optical_flow_success++;
        }
    }
    
    ROS_DEBUG("[LoFTR] Optical flow: %d features tracked", optical_flow_success);
}

void FeatureTrackerLoFTR::matchFeaturesLoFTR()
{
    if (!loftr_initialized_) {
        ROS_ERROR("[LoFTR] LoFTR not initialized!");
        trackFeaturesOpticalFlow();
        return;
    }
    
    auto match_result = loftr_interface_->match_images(prev_img, cur_img);
    
    if (match_result.num_matches == 0) {
        ROS_WARN("[LoFTR] No LoFTR matches found, using optical flow");
        trackFeaturesOpticalFlow();
        return;
    }
    
    // 處理LoFTR匹配
    std::set<int> used_ids;
    int loftr_new = 0, loftr_continued = 0;
    
    for (size_t i = 0; i < match_result.keypoints0.size() && 
                    i < static_cast<size_t>(max_features_); ++i) {
        cv::Point2f prev_pt = match_result.keypoints0[i];
        cv::Point2f cur_pt = match_result.keypoints1[i];
        
        if (!inBorder(cur_pt)) {
            continue;
        }
        
        if (!match_result.confidence.empty() && 
            match_result.confidence[i] < match_threshold_) {
            continue;
        }
        
        // 尋找匹配的已跟蹤特徵
        int matched_id = -1;
        float min_dist = 10.0f;
        
        for (const auto& [id, pt] : tracked_features_) {
            if (used_ids.find(id) != used_ids.end()) continue;
            
            float dist = cv::norm(prev_pt - pt);
            if (dist < min_dist) {
                min_dist = dist;
                matched_id = id;
            }
        }
        
        if (matched_id != -1) {
            // 更新現有特徵
            cur_pts.push_back(cur_pt);
            ids.push_back(matched_id);
            track_cnt.push_back(feature_track_count_[matched_id] + 1);
            
            tracked_features_[matched_id] = cur_pt;
            feature_track_count_[matched_id]++;
            used_ids.insert(matched_id);
            loftr_continued++;
        } else {
            // 新增LoFTR特徵
            int new_id = allocateFeatureID(true);
            
            cur_pts.push_back(cur_pt);
            ids.push_back(new_id);
            track_cnt.push_back(1);
            
            tracked_features_[new_id] = cur_pt;
            feature_track_count_[new_id] = 1;
            loftr_new++;
        }
    }
    
    ROS_DEBUG("[LoFTR] LoFTR matching: %d new, %d continued", loftr_new, loftr_continued);
}

void FeatureTrackerLoFTR::combineWithOpticalFlow()
{
    if (!use_traditional_tracker_ || prev_pts.empty()) {
        return;
    }
    
    vector<uchar> status;
    vector<float> err;
    vector<cv::Point2f> tracked_pts;
    
    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, tracked_pts, 
                            status, err, cv::Size(21, 21), 3);
    
    int optical_flow_added = 0;
    
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] && inBorder(tracked_pts[i])) {
            // 檢查是否與LoFTR特徵重複
            bool is_duplicate = false;
            for (const auto &pt : cur_pts) {
                if (distance(tracked_pts[i], pt) < 5.0f) {
                    is_duplicate = true;
                    break;
                }
            }
            
            if (!is_duplicate && i < this->ids.size()) {
                cur_pts.push_back(tracked_pts[i]);
                ids.push_back(this->ids[i]);
                track_cnt.push_back(this->track_cnt[i] + 1);
                optical_flow_added++;
            }
        }
    }
    
    ROS_DEBUG("[LoFTR] Optical flow added %d features", optical_flow_added);
}

void FeatureTrackerLoFTR::addNewFeatures()
{
    int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
    if (n_max_cnt <= 0) {
        return;
    }
    
    // 設置遮罩避免現有特徵
    setMask();
    
    // 檢測新特徵
    vector<cv::Point2f> new_features;
    cv::goodFeaturesToTrack(cur_img, new_features, n_max_cnt, 0.01, MIN_DIST, mask);
    
    // 添加新特徵（使用傳統ID）
    for (const auto& pt : new_features) {
        int new_id = allocateFeatureID(false);
        
        cur_pts.push_back(pt);
        ids.push_back(new_id);
        track_cnt.push_back(1);
        
        tracked_features_[new_id] = pt;
        feature_track_count_[new_id] = 1;
    }
    
    ROS_DEBUG("[LoFTR] Added %zu new features", new_features.size());
}

void FeatureTrackerLoFTR::removeDuplicates()
{
    std::map<int, std::pair<cv::Point2f, int>> unique_features;
    
    for (size_t i = 0; i < ids.size(); ++i) {
        int feature_id = ids[i];
        auto it = unique_features.find(feature_id);
        
        if (it != unique_features.end()) {
            ROS_WARN("[LoFTR] Duplicate ID %d found and removed", feature_id);
            // 保留跟蹤更長的
            if (i < track_cnt.size() && track_cnt[i] > it->second.second) {
                unique_features[feature_id] = {cur_pts[i], track_cnt[i]};
            }
        } else {
            unique_features[feature_id] = {cur_pts[i], 
                                          i < track_cnt.size() ? track_cnt[i] : 1};
        }
    }
    
    // 重建向量
    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
    
    for (const auto& [id, pt_cnt] : unique_features) {
        cur_pts.push_back(pt_cnt.first);
        ids.push_back(id);
        track_cnt.push_back(pt_cnt.second);
    }
}

void FeatureTrackerLoFTR::cleanupOldFeatures()
{
    if (tracked_features_.size() <= static_cast<size_t>(SystemLimits::MAX_FEATURE_COUNT)) {
        return;
    }
    
    std::vector<std::pair<int, int>> track_pairs;
    for (const auto& [id, count] : feature_track_count_) {
        track_pairs.emplace_back(count, id);
    }
    
    std::sort(track_pairs.begin(), track_pairs.end(), 
              std::greater<std::pair<int, int>>());
    
    std::map<int, cv::Point2f> new_tracked_features;
    std::map<int, int> new_track_counts;
    
    size_t keep_count = std::min(track_pairs.size(), 
                                 static_cast<size_t>(SystemLimits::MAX_FEATURE_COUNT));
    
    for (size_t i = 0; i < keep_count; ++i) {
        int feature_id = track_pairs[i].second;
        new_tracked_features[feature_id] = tracked_features_[feature_id];
        new_track_counts[feature_id] = feature_track_count_[feature_id];
    }
    
    size_t removed = tracked_features_.size() - new_tracked_features.size();
    tracked_features_ = std::move(new_tracked_features);
    feature_track_count_ = std::move(new_track_counts);
    
    if (removed > 0) {
        ROS_INFO("[LoFTR] Cleaned up %zu old features", removed);
    }
}

void FeatureTrackerLoFTR::processStereo(const cv::Mat &cur_img, const cv::Mat &rightImg)
{
    ids_right.clear();
    cur_right_pts.clear();
    cur_un_right_pts.clear();
    right_pts_velocity.clear();
    cur_un_right_pts_map.clear();
    
    if (cur_pts.empty()) {
        return;
    }
    
    // 嘗試光流stereo
    vector<uchar> status;
    vector<float> err;
    vector<cv::Point2f> temp_right_pts;
    
    cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, temp_right_pts, 
                            status, err, cv::Size(21, 21), 3);
    
    // 計算失敗率
    int failed_count = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i] || !inBorder(temp_right_pts[i])) {
            failed_count++;
        }
    }
    float failure_rate = static_cast<float>(failed_count) / status.size();
    
    // 決定是否使用LoFTR stereo
    bool use_stereo_loftr = shouldUseStereoLoFTR(failure_rate);
    
    if (use_stereo_loftr && loftr_initialized_) {
        matchStereoLoFTR(cur_img, rightImg);
        frames_since_stereo_loftr_ = 0;
    } else {
        // 使用光流結果
        if (FLOW_BACK) {
            vector<uchar> statusRightLeft;
            vector<cv::Point2f> reverseLeftPts;
            cv::calcOpticalFlowPyrLK(rightImg, cur_img, temp_right_pts, reverseLeftPts, 
                                    statusRightLeft, err, cv::Size(21, 21), 3);
            
            for (size_t i = 0; i < status.size(); ++i) {
                if (status[i] && statusRightLeft[i] && inBorder(temp_right_pts[i]) && 
                   distance(cur_pts[i], reverseLeftPts[i]) <= 0.5) {
                    status[i] = 1;
                } else {
                    status[i] = 0;
                }
            }
        }
        
        // 處理有效的光流結果
        std::set<int> right_feature_ids;
        for (size_t i = 0; i < status.size() && i < cur_pts.size(); ++i) {
            if (status[i] && inBorder(temp_right_pts[i])) {
                int feature_id = ids[i];
                if (right_feature_ids.find(feature_id) == right_feature_ids.end()) {
                    right_feature_ids.insert(feature_id);
                    cur_right_pts.push_back(temp_right_pts[i]);
                    ids_right.push_back(feature_id);
                }
            }
        }
        frames_since_stereo_loftr_++;
    }
    
    // 處理右目特徵
    if (!cur_right_pts.empty()) {
        cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
        right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts,
                                       cur_un_right_pts_map, prev_un_right_pts_map);
    }
    
    ROS_DEBUG("[LoFTR] Stereo: %s, failure_rate: %.2f, right_features: %zu", 
             use_stereo_loftr ? "LoFTR" : "OpticalFlow", 
             failure_rate, cur_right_pts.size());
}

void FeatureTrackerLoFTR::matchStereoLoFTR(const cv::Mat &cur_img, const cv::Mat &rightImg)
{
    if (!loftr_initialized_) {
        return;
    }
    
    auto match_result = loftr_interface_->match_images(cur_img, rightImg);
    
    if (match_result.num_matches == 0) {
        ROS_WARN("[LoFTR] No stereo matches found");
        return;
    }
    
    // 對齊到現有的左目特徵
    for (size_t i = 0; i < match_result.keypoints0.size(); i++) {
        cv::Point2f left_pt = match_result.keypoints0[i];
        cv::Point2f right_pt = match_result.keypoints1[i];
        
        if (!inBorder(left_pt) || !inBorder(right_pt)) {
            continue;
        }
        
        if (!match_result.confidence.empty() && 
            match_result.confidence[i] < match_threshold_) {
            continue;
        }
        
        // 找到最接近的左目特徵
        int best_id = -1;
        double min_dist = 2.0;
        
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
    
    ROS_DEBUG("[LoFTR] Stereo LoFTR matched %zu features", cur_right_pts.size());
}

bool FeatureTrackerLoFTR::shouldUseStereoLoFTR(float optical_flow_failure_rate)
{
    // 到達間隔時間
    if (frames_since_stereo_loftr_ >= stereo_loftr_interval_) {
        return true;
    }
    
    // 光流失敗率過高
    if (optical_flow_failure_rate > stereo_failure_threshold_) {
        ROS_INFO("[LoFTR] Force stereo LoFTR due to high failure rate: %.2f", 
                 optical_flow_failure_rate);
        return true;
    }
    
    return false;
}

void FeatureTrackerLoFTR::updateHistory()
{
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;
    
    prevLeftPtsMap.clear();
    for (size_t i = 0; i < cur_pts.size(); i++) {
        prevLeftPtsMap[ids[i]] = cur_pts[i];
    }
    
    prev_un_right_pts_map = cur_un_right_pts_map;
}

void FeatureTrackerLoFTR::buildFeatureFrame(
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    // 檢查數據一致性
    if (cur_pts.size() != ids.size() || 
        cur_pts.size() != cur_un_pts.size() || 
        cur_pts.size() != pts_velocity.size()) {
        ROS_ERROR("[LoFTR] Data size mismatch");
        return;
    }
    
    // 添加左相機觀測
    std::set<int> processed_left_features;
    for (size_t i = 0; i < ids.size(); i++) {
        int feature_id = ids[i];
        
        if (processed_left_features.find(feature_id) != processed_left_features.end()) {
            ROS_ERROR("[LoFTR] Duplicate left feature ID %d", feature_id);
            continue;
        }
        processed_left_features.insert(feature_id);
        
        double x = cur_un_pts[i].x;
        double y = cur_un_pts[i].y;
        double z = 1;
        double p_u = cur_pts[i].x;
        double p_v = cur_pts[i].y;
        double velocity_x = pts_velocity[i].x;
        double velocity_y = pts_velocity[i].y;
        
        Eigen::Matrix<double, 7, 1> left_obs;
        left_obs << x, y, z, p_u, p_v, velocity_x, velocity_y;
        
        featureFrame[feature_id].emplace_back(0, left_obs);
    }
    
    // 添加右相機觀測
    if (stereo_cam && !ids_right.empty()) {
        std::set<int> processed_right_features;
        
        for (size_t i = 0; i < ids_right.size(); i++) {
            int feature_id = ids_right[i];
            
            if (featureFrame.find(feature_id) == featureFrame.end()) {
                continue;
            }
            
            if (processed_right_features.find(feature_id) != processed_right_features.end()) {
                ROS_ERROR("[LoFTR] Duplicate right feature ID %d", feature_id);
                continue;
            }
            processed_right_features.insert(feature_id);
            
            if (i >= cur_un_right_pts.size() || 
                i >= cur_right_pts.size() || 
                i >= right_pts_velocity.size()) {
                continue;
            }
            
            double x_r = cur_un_right_pts[i].x;
            double y_r = cur_un_right_pts[i].y;
            double z_r = 1;
            double p_u_r = cur_right_pts[i].x;
            double p_v_r = cur_right_pts[i].y;
            double velocity_x_r = right_pts_velocity[i].x;
            double velocity_y_r = right_pts_velocity[i].y;
            
            Eigen::Matrix<double, 7, 1> right_obs;
            right_obs << x_r, y_r, z_r, p_u_r, p_v_r, velocity_x_r, velocity_y_r;
            
            featureFrame[feature_id].emplace_back(1, right_obs);
        }
    }
    
    // 驗證結果
    int mono_count = 0, stereo_count = 0;
    for (const auto& [id, obs] : featureFrame) {
        if (obs.size() == 1) mono_count++;
        else if (obs.size() == 2) stereo_count++;
    }
    
    ROS_DEBUG("[LoFTR] Feature frame: %d mono, %d stereo", mono_count, stereo_count);
}

// 主跟蹤函數
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> 
FeatureTrackerLoFTR::trackImage_loftr(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;
    
    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
    
    ROS_DEBUG("[LoFTR] ========== FRAME %zu START ==========", total_frames_);
    
    // 第一幀初始化
    if (total_frames_ == 0 || prev_img.empty()) {
        initializeFirstFrame(cur_img);
        use_loftr_this_frame_ = false;
        frames_since_loftr_ = 0;
    } else {
        // 決定跟蹤方法
        use_loftr_this_frame_ = shouldUseLoFTRThisFrame();
        
        TicToc t_match;
        
        if (use_loftr_this_frame_) {
            // 使用LoFTR
            ROS_DEBUG("[LoFTR] Using LoFTR matching");
            matchFeaturesLoFTR();
            
            if (use_traditional_tracker_) {
                combineWithOpticalFlow();
            }
            
            frames_since_loftr_ = 0;
        } else {
            // 僅使用光流
            ROS_DEBUG("[LoFTR] Using optical flow only");
            trackFeaturesOpticalFlow();
        }
        
        // 補充新特徵
        if (cur_pts.size() < static_cast<size_t>(MAX_CNT * 0.5)) {
            addNewFeatures();
        }
        
        ROS_DEBUG("[LoFTR] Tracking time: %.3f ms", t_match.toc());
    }
    
    frames_since_loftr_++;
    total_frames_++;
    
    // 移除重複
    removeDuplicates();
    
    // 清理過舊特徵
    cleanupOldFeatures();
    
    ROS_INFO("[LoFTR] Frame %zu: %zu features, method=%s", 
             total_frames_, cur_pts.size(), 
             use_loftr_this_frame_ ? "LoFTR" : "OpticalFlow");
    
    // 計算歸一化座標和速度
    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);
    
    // 立體處理
    if (!_img1.empty() && stereo_cam) {
        processStereo(cur_img, rightImg);
    }
    
    // 可視化
    if (SHOW_TRACK) {
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
    }
    
    // 更新歷史
    updateHistory();
    
    // 建立特徵幀
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    buildFeatureFrame(featureFrame);
    
    ROS_DEBUG("[LoFTR] ========== FRAME %zu END ==========", total_frames_);
    ROS_DEBUG("[LoFTR] Total frame processing time: %.3f ms", t_r.toc());
    
    return featureFrame;
}

// 其他輔助函數實現
void FeatureTrackerLoFTR::resetTracking()
{
    tracked_features_.clear();
    feature_track_count_.clear();
    next_loftr_id_ = FeatureIDRange::LOFTR_START;
    next_traditional_id_ = FeatureIDRange::TRADITIONAL_START;
    frames_since_loftr_ = 0;
    frames_since_stereo_loftr_ = 0;
    use_loftr_this_frame_ = false;
    total_frames_ = 0;
    ROS_INFO("[LoFTR] Feature tracking reset");
}

bool FeatureTrackerLoFTR::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && 
           BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double FeatureTrackerLoFTR::distance(const cv::Point2f &pt1, const cv::Point2f &pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void FeatureTrackerLoFTR::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
    
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));
    
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), 
         [](const pair<int, pair<cv::Point2f, int>> &a, 
            const pair<int, pair<cv::Point2f, int>> &b)
         { return a.first > b.first; });
    
    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
    
    for (auto &it : cnt_pts_id) {
        if (mask.at<uchar>(it.second.first) == 255) {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTrackerLoFTR::reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void FeatureTrackerLoFTR::reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void FeatureTrackerLoFTR::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++) {
        ROS_INFO("[LoFTR] Reading camera parameters: %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = 
            CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

vector<cv::Point2f> FeatureTrackerLoFTR::undistortedPts(vector<cv::Point2f> &pts, 
                                                        camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++) {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTrackerLoFTR::ptsVelocity(vector<int> &ids, 
                                                     vector<cv::Point2f> &pts,
                                                     map<int, cv::Point2f> &cur_id_pts, 
                                                     map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    
    for (unsigned int i = 0; i < ids.size(); i++) {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }
    
    if (!prev_id_pts.empty()) {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++) {
            auto it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end()) {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            } else {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    } else {
        for (unsigned int i = 0; i < cur_pts.size(); i++) {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
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

    //draw left feature point (red ----> blue)
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
    }
    // draw motivation trace
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++) {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if (mapIt != prevLeftPtsMap.end()) {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}
void FeatureTrackerLoFTR::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    
    for (size_t i = 0; i < ids.size(); i++) {
        int id = ids[i];
        auto itPredict = predictPts.find(id);
        if (itPredict != predictPts.end()) {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        } else {
            predict_pts.push_back(prev_pts[i]);
        }
    }
}

void FeatureTrackerLoFTR::removeOutliers(set<int> &removePtsIds)
{
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++) {
        if (removePtsIds.find(ids[i]) != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }
    
    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

cv::Mat FeatureTrackerLoFTR::getTrackImage()
{
    return imTrack;
}

void FeatureTrackerLoFTR::updatePerformanceStats(double match_time)
{
    match_times_.push_back(match_time);
    
    if (match_times_.size() > 100) {
        match_times_.erase(match_times_.begin());
    }
}

void FeatureTrackerLoFTR::printPerformanceStats() const
{
    if (match_times_.empty()) {
        return;
    }
    
    double avg_time = std::accumulate(match_times_.begin(), match_times_.end(), 0.0) 
                     / match_times_.size();
    double min_time = *std::min_element(match_times_.begin(), match_times_.end());
    double max_time = *std::max_element(match_times_.begin(), match_times_.end());
    
    ROS_INFO("[LoFTR] === Performance Statistics ===");
    ROS_INFO("[LoFTR] Total frames: %zu", total_frames_);
    ROS_INFO("[LoFTR] Average matching time: %.3f ms", avg_time);
    ROS_INFO("[LoFTR] Min/Max time: %.3f / %.3f ms", min_time, max_time);
    ROS_INFO("[LoFTR] Average FPS: %.2f", 1000.0 / avg_time);
    
    if (loftr_initialized_) {
        ROS_INFO("[LoFTR] Backend: %s", loftr_interface_->getBackendInfo().c_str());
        
        double loftr_usage_rate = (total_frames_ > 0) ? 
            (double)(total_frames_ / loftr_frame_interval_) * 100.0 / total_frames_ : 0.0;
        ROS_INFO("[LoFTR] LoFTR usage rate: %.1f%% (interval: %d frames)", 
                loftr_usage_rate, loftr_frame_interval_);
    }
    ROS_INFO("[LoFTR] ================================");
}