/*******************************************************
 * Modern LoFTR Feature Tracker for VINS Integration
 * 使用高性能C++接口替代Python接口
 * 支援間隔執行LoFTR以提升性能
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <set>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

// 使用新的LoFTR接口
#include "loftr_interface.h"
#include "loftr_utils.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

class FeatureTrackerLoFTR
{
public:
    FeatureTrackerLoFTR();
    ~FeatureTrackerLoFTR();
    
    // 主要跟蹤函數
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage_loftr(
        double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    
    // 初始化函數
    bool initializeLoFTR();
    void initializeFirstFrame(const cv::Mat &img);
    
    // 基礎功能函數
    void setMask();
    void readIntrinsicParameter(const vector<string> &calib_file);
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                   vector<int> &curLeftIds,
                   vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(const cv::Point2f &pt1, const cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);
    void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
    void reduceVector(vector<int> &v, vector<uchar> status);

    // LoFTR 特定函數
    void matchFeaturesLoFTR();
    void trackFeaturesOpticalFlow();
    void addNewFeatures();
    void combineWithOpticalFlow();
    void removeDuplicates();
    void processStereo(const cv::Mat &cur_img, const cv::Mat &rightImg);
    void updateHistory();
    void buildFeatureFrame(map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    
    // ID管理
    int allocateFeatureID(bool is_loftr_feature);
    void cleanupOldFeatures();
    
    // 間隔控制
    bool shouldUseLoFTRThisFrame();
    void resetTracking();
    
    // Stereo matching
    bool shouldUseStereoLoFTR(float optical_flow_failure_rate);
    void matchStereoLoFTR(const cv::Mat &cur_img, const cv::Mat &rightImg);
    
    // 配置函數
    void loadConfiguration();
    
    // 獲取狀態信息
    bool isLoFTRInitialized() const { return loftr_initialized_; }
    size_t getTrackedFeatureCount() const { return tracked_features_.size(); }
    size_t getTotalFrames() const { return total_frames_; }
    
    // 性能統計
    void updatePerformanceStats(double match_time);
    void printPerformanceStats() const;

public:
    // 公開成員變量（保持與原始接口兼容）
    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;

private:
    // LoFTR 相關成員
    unique_ptr<LoFTR_Interface> loftr_interface_;
    bool loftr_initialized_;
    LoFTR_Interface::Config loftr_config_;
    
    // 特徵ID管理
    int next_loftr_id_;
    int next_traditional_id_;
    std::map<int, cv::Point2f> tracked_features_;
    std::map<int, int> feature_track_count_;
    
    // 間隔控制（從全局參數讀取）
    int loftr_frame_interval_;
    int frames_since_loftr_;
    bool use_loftr_this_frame_;
    int min_feature_threshold_;
    bool adaptive_interval_;
    int performance_mode_;
    
    // Stereo控制（從全局參數讀取）
    int frames_since_stereo_loftr_;
    int stereo_loftr_interval_;
    float stereo_failure_threshold_;
    
    // 配置參數（從全局參數讀取）
    int max_features_;
    float match_threshold_;
    bool use_traditional_tracker_;
    
    // 性能統計
    mutable std::vector<double> match_times_;
    mutable size_t total_frames_;
};
