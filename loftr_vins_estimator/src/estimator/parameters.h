/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;

extern bool LOFTR_USE_TRADITIONAL_TRACKER;   // 是否結合傳統跟蹤器
extern float LOFTR_TRADITIONAL_DISTANCE_THRESHOLD; // 傳統跟蹤器距離閾值
extern int LOFTR_PERFORMANCE_STATS_WINDOW;   // 性能統計窗口大小
extern bool LOFTR_ENABLE_DEBUG_LOG;          // 是否啟用調試日誌
extern bool LOFTR_SHOW_PERFORMANCE_STATS;    // 是否顯示性能統計
extern int LOFTR_START_FRAME;              // LoFTR 開始執行的幀數


void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

// 傳統深度學習特徵相關參數 (保留兼容性)
extern string extractor_weight_global_path; 
extern string matcher_weight_global_path; 
extern string extractor_weight_relative_path; 
extern string matcher_weight_relative_path; 
extern double ransacReprojThreshold;
extern float MATCHER_THRESHOLD;

// ============= LoFTR 相關全局變量聲明 =============
extern bool USE_LOFTR;                      // 是否使用 LoFTR
extern std::string LOFTR_MODEL_PATH;        // LoFTR 模型文件路徑
extern std::string LOFTR_ENGINE_PATH;       // TensorRT 引擎路徑（可選）
extern int LOFTR_INPUT_WIDTH;               // 網絡輸入寬度
extern int LOFTR_INPUT_HEIGHT;              // 網絡輸入高度
extern float LOFTR_MATCH_THRESHOLD;         // 匹配置信度閾值
extern int LOFTR_MAX_FEATURES;              // 最大特徵點數量
extern int LOFTR_BACKEND;                   // 後端類型 (0: 自動, 1: ONNX, 2: TensorRT)
extern bool USE_TRADITIONAL_TRACKER;       // 是否結合傳統光流跟蹤

// ============= LoFTR Frame Interval Configuration =============
extern int LOFTR_FRAME_INTERVAL;            // LoFTR執行間隔 (每N幀執行一次)
extern int LOFTR_MIN_FEATURE_THRESHOLD;     // 最小特徵點數量閾值 (低於此值強制使用LoFTR)
extern bool LOFTR_ADAPTIVE_INTERVAL;        // 是否使用自適應間隔
extern int LOFTR_PERFORMANCE_MODE;          // 性能模式 (0: 平衡, 1: 精度優先, 2: 速度優先)
extern bool LOFTR_DEBUG_INTERVAL;          // 是否開啟詳細間隔調試日誌
extern bool LOFTR_PRINT_STATS;             // 是否定期打印性能統計

// ============= NEW: 增加更多LoFTR可配置參數 =============
extern int LOFTR_MAX_TRACK_LENGTH;          // 最大跟蹤長度
extern int LOFTR_MAX_TOTAL_FEATURES;        // 最大總特徵數量
extern float LOFTR_SEARCH_RADIUS;           // LoFTR特徵搜索半徑
extern float LOFTR_OPTICAL_FLOW_MERGE_RADIUS; // 光流合併半徑
extern int LOFTR_FORCE_INTERVAL_MAX;        // 強制LoFTR最大間隔
extern float LOFTR_STEREO_FAILURE_THRESHOLD; // 立體匹配失敗率閾值
extern int LOFTR_STEREO_INTERVAL;           // 立體LoFTR間隔
extern float LOFTR_STEREO_SEARCH_RADIUS;    // 立體匹配搜索半徑

enum ExtractorType //deep-learning based feature extractor type
{
    SUPERPOINT = 0,
    DISK = 1
};

enum DescriptorSize //deep-learning based feature matcher type
{
    SUPERPOINT_SIZE = 256,
    DISK_SIZE = 128
};

// LoFTR 後端類型枚舉
enum LoFTRBackendType
{
    LOFTR_AUTO = 0,          // 自動選擇最優後端
    LOFTR_ONNX_RUNTIME = 1,  // 使用 ONNX Runtime
    LOFTR_TENSORRT = 2       // 使用 TensorRT
};

// LoFTR 性能模式枚舉
enum LoFTRPerformanceMode
{
    LOFTR_BALANCED = 0,      // 平衡模式：使用配置的間隔
    LOFTR_ACCURACY = 1,      // 精度優先：跟蹤質量差時減少間隔
    LOFTR_SPEED = 2          // 速度優先：跟蹤質量好時增加間隔
};

// ============= LoFTR 常量定義（非可配置） =============
namespace LoFTRConstants {
    // 間隔控制常量
    constexpr int MIN_FRAME_INTERVAL = 1;               // 最小間隔
    constexpr int MAX_FRAME_INTERVAL = 10;              // 最大間隔
    constexpr int MIN_FEATURE_THRESHOLD = 5;            // 最小特徵點閾值下限
    constexpr int MAX_FEATURE_THRESHOLD = 500;          // 最小特徵點閾值上限
    
     // 新增的範圍檢查常數
    constexpr int MIN_START_FRAME = 1;               // 最小開始幀數
    constexpr int MAX_START_FRAME = 10;              // 最大開始幀數
    constexpr int DEFAULT_START_FRAME = 1;           // 默認開始幀數
    
    // 算法常量
    constexpr int BORDER_SIZE = 1;                      // 邊界大小
    constexpr float REVERSE_FLOW_THRESHOLD = 0.5f;     // 反向光流閾值
    constexpr int OPTICAL_FLOW_WINDOW_SIZE = 21;        // 光流窗口大小
    constexpr int OPTICAL_FLOW_MAX_LEVEL = 3;           // 光流金字塔層數
    
    // 默認值
    constexpr int DEFAULT_FRAME_INTERVAL = 2;           // 默認間隔
    constexpr int DEFAULT_MIN_FEATURE_THRESHOLD = 20;   // 默認最小特徵點數量
    constexpr int DEFAULT_MAX_TRACK_LENGTH = 30;        // 默認最大跟蹤長度
    constexpr int DEFAULT_MAX_TOTAL_FEATURES = 200;     // 默認最大總特徵數量
    constexpr float DEFAULT_SEARCH_RADIUS = 15.0f;      // 默認搜索半徑
    constexpr float DEFAULT_OPTICAL_FLOW_MERGE_RADIUS = 5.0f; // 默認光流合併半徑
    constexpr int DEFAULT_FORCE_INTERVAL_MAX = 5;       // 默認強制間隔最大值
    constexpr float DEFAULT_STEREO_FAILURE_THRESHOLD = 0.3f; // 默認立體失敗率閾值
    constexpr int DEFAULT_STEREO_INTERVAL = 2;          // 默認立體間隔
    constexpr float DEFAULT_STEREO_SEARCH_RADIUS = 2.0f; // 默認立體搜索半徑
}

// ============= 參數驗證函數聲明 =============
bool validateLoFTRParameters();
void printLoFTRConfiguration();
