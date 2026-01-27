#pragma once

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <set>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

#include "loftr_interface.h"
#include "loftr_utils.h"
#include "feature_tracker.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

class FeatureTrackerLoFTR : public FeatureTracker
{
private:
    bool loftr_initialized_;
    size_t total_frames_;
    int frames_since_loftr_temporal_;
    int frames_since_loftr_stereo_;
    
    std::unique_ptr<LoFTR_Interface> loftr_interface_;
    LoFTR_Interface::Config loftr_config_;

    bool csv_logging_enabled_;
    std::string csv_output_dir_;
    std::ofstream csv_performance_log_;
    std::ofstream csv_feature_log_;
    std::ofstream csv_stereo_log_;

    // 性能指標結構
    struct PerformanceMetrics {
        // --- 時序 (Temporal) ---
        double optical_flow_time_ms;
        double loftr_time_ms;
        
        int prev_features_count;        // [新增] 上一幀特徵數 (時序分母)
        int flow_tracked_count;         // [新增] 光流成功追蹤數 (時序分子)
        
        int features_tracked;           // 當前幀最終特徵數
        int features_added_loftr;       // LoFTR 時序補充數
        int features_added_traditional;
        bool used_loftr_temporal;

        // --- 立體 (Stereo) ---
        double stereo_time_ms;          
        int stereo_matches;             // 立體匹配總數 (立體分子)
        bool used_loftr_stereo;

        // 立體細分數據
        double stereo_flow_time_ms = 0.0;   
        double stereo_loftr_time_ms = 0.0;  
        int stereo_flow_matches = 0;        
        int stereo_added_by_loftr = 0;      
        
        double total_time_ms;
    } current_metrics_;

private:
    // ... (保留原有的函數聲明) ...
    void processFirstFrame(const cv::Mat &img);
    void processSubsequentFrames(const cv::Mat &img);
    void trackWithOpticalFlow(const cv::Mat &img);
    void correctAndSupplementWithLoFTR(const cv::Mat &img);
    void supplementWithTraditional(const cv::Mat &img);
    void supplementWithLoFTR(const cv::Mat &img);
    void processStereoMatching(const cv::Mat &leftImg, const cv::Mat &rightImg);
    void stereoMatchingWithLoFTR(const cv::Mat &leftImg, const cv::Mat &rightImg);
    
    // 這裡我們只修改這些函數的內部實現，介面不變
    void stereoMatchingWithOpticalFlow(const cv::Mat &leftImg, const cv::Mat &rightImg);
    void supplementStereoWithLoFTR(const cv::Mat &leftImg, const cv::Mat &rightImg);
    
    bool shouldUseLoFTRCorrection() const;
    bool shouldUseLoFTRNewFeatures() const;
    bool shouldUseLoFTRStereo() const;
    
    void addFeaturesUsingLoFTR(const cv::Mat &img, vector<cv::Point2f> &new_features, int max_features);
    void addFeaturesUsingTraditional(const cv::Mat &img, vector<cv::Point2f> &new_features, int max_features);
    void matchStereoLoFTR(const cv::Mat &leftImg, const cv::Mat &rightImg,
                         const vector<cv::Point2f> &cur_pts, const vector<int> &ids,
                         vector<cv::Point2f> &cur_right_pts, vector<int> &ids_right);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                   vector<int> &curLeftIds,
                   vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   map<int, cv::Point2f> &prevLeftPtsMap);

    // CSV 相關
    void initializeCSVLogging();
    void logPerformanceMetrics();
    void finalizeCSVLogging();
    void resetCurrentMetrics();
    
public:
    FeatureTrackerLoFTR();
    ~FeatureTrackerLoFTR();
    
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> 
    trackImage_loftr(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1);
    
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> 
    trackImage_loftr(double _cur_time, const cv::Mat &_img) {
        cv::Mat empty_right;
        return trackImage_loftr(_cur_time, _img, empty_right);
    }
    
    bool initializeLoFTR();
    void loadConfiguration();
};
