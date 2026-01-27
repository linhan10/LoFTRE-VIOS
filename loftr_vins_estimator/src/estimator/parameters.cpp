/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

// 傳統深度學習特徵相關參數 (保留兼容性)
string extractor_weight_global_path;
string matcher_weight_global_path;
string extractor_weight_relative_path;
string matcher_weight_relative_path;
float MATCHER_THRESHOLD;
double ransacReprojThreshold;

// ============= LoFTR 相關變量定義 =============
bool USE_LOFTR = false;
std::string LOFTR_MODEL_PATH = "";
std::string LOFTR_ENGINE_PATH = "";
int LOFTR_INPUT_WIDTH = 640;
int LOFTR_INPUT_HEIGHT = 352;
float LOFTR_MATCH_THRESHOLD = 0.3f;
int LOFTR_MAX_FEATURES = 150;
int LOFTR_BACKEND = 0;  // 默認自動選擇
bool USE_TRADITIONAL_TRACKER = true;
int LOFTR_START_FRAME = 1;  

// ============= LoFTR Frame Interval Variables =============
int LOFTR_FRAME_INTERVAL = LoFTRConstants::DEFAULT_FRAME_INTERVAL;
int LOFTR_MIN_FEATURE_THRESHOLD = LoFTRConstants::DEFAULT_MIN_FEATURE_THRESHOLD;
bool LOFTR_ADAPTIVE_INTERVAL = false;
int LOFTR_PERFORMANCE_MODE = 0;  // 默認平衡模式
bool LOFTR_DEBUG_INTERVAL = true;
bool LOFTR_PRINT_STATS = true;

// ============= 新增的LoFTR可配置參數 =============
int LOFTR_MAX_TRACK_LENGTH = LoFTRConstants::DEFAULT_MAX_TRACK_LENGTH;
int LOFTR_MAX_TOTAL_FEATURES = LoFTRConstants::DEFAULT_MAX_TOTAL_FEATURES;
float LOFTR_SEARCH_RADIUS = LoFTRConstants::DEFAULT_SEARCH_RADIUS;
float LOFTR_OPTICAL_FLOW_MERGE_RADIUS = LoFTRConstants::DEFAULT_OPTICAL_FLOW_MERGE_RADIUS;
int LOFTR_FORCE_INTERVAL_MAX = LoFTRConstants::DEFAULT_FORCE_INTERVAL_MAX;
float LOFTR_STEREO_FAILURE_THRESHOLD = LoFTRConstants::DEFAULT_STEREO_FAILURE_THRESHOLD;
int LOFTR_STEREO_INTERVAL = LoFTRConstants::DEFAULT_STEREO_INTERVAL;
float LOFTR_STEREO_SEARCH_RADIUS = LoFTRConstants::DEFAULT_STEREO_SEARCH_RADIUS;

bool LOFTR_USE_TRADITIONAL_TRACKER = true;   // 預設啟用傳統跟蹤器
float LOFTR_TRADITIONAL_DISTANCE_THRESHOLD = 5.0f; // 預設距離閾值
int LOFTR_PERFORMANCE_STATS_WINDOW = 100;    // 預設性能統計窗口
bool LOFTR_ENABLE_DEBUG_LOG = false;         // 預設關閉調試日誌
bool LOFTR_SHOW_PERFORMANCE_STATS = true;    // 預設顯示性能統計


template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

// ============= 參數驗證函數 =============
bool validateLoFTRParameters()
{
    bool valid = true;
    
    // 驗證間隔參數
    if (LOFTR_FRAME_INTERVAL < LoFTRConstants::MIN_FRAME_INTERVAL ||
        LOFTR_FRAME_INTERVAL > LoFTRConstants::MAX_FRAME_INTERVAL) {
        ROS_ERROR("Invalid LOFTR_FRAME_INTERVAL: %d (valid range: %d-%d)", 
                 LOFTR_FRAME_INTERVAL, LoFTRConstants::MIN_FRAME_INTERVAL, LoFTRConstants::MAX_FRAME_INTERVAL);
        valid = false;
    }
    
    // 驗證特徵閾值
    if (LOFTR_MIN_FEATURE_THRESHOLD < LoFTRConstants::MIN_FEATURE_THRESHOLD ||
        LOFTR_MIN_FEATURE_THRESHOLD > LoFTRConstants::MAX_FEATURE_THRESHOLD) {
        ROS_ERROR("Invalid LOFTR_MIN_FEATURE_THRESHOLD: %d (valid range: %d-%d)", 
                 LOFTR_MIN_FEATURE_THRESHOLD, LoFTRConstants::MIN_FEATURE_THRESHOLD, LoFTRConstants::MAX_FEATURE_THRESHOLD);
        valid = false;
    }
    
    // 驗證性能模式
    if (LOFTR_PERFORMANCE_MODE < 0 || LOFTR_PERFORMANCE_MODE > 2) {
        ROS_ERROR("Invalid LOFTR_PERFORMANCE_MODE: %d (valid range: 0-2)", LOFTR_PERFORMANCE_MODE);
        valid = false;
    }
    
    // 驗證後端類型
    if (LOFTR_BACKEND < 0 || LOFTR_BACKEND > 2) {
        ROS_ERROR("Invalid LOFTR_BACKEND: %d (valid range: 0-2)", LOFTR_BACKEND);
        valid = false;
    }
    
    // 驗證閾值參數
    if (LOFTR_MATCH_THRESHOLD < 0.0f || LOFTR_MATCH_THRESHOLD > 1.0f) {
        ROS_ERROR("Invalid LOFTR_MATCH_THRESHOLD: %.3f (valid range: 0.0-1.0)", LOFTR_MATCH_THRESHOLD);
        valid = false;
    }
    
    // 驗證立體參數
    if (LOFTR_STEREO_FAILURE_THRESHOLD < 0.0f || LOFTR_STEREO_FAILURE_THRESHOLD > 1.0f) {
        ROS_ERROR("Invalid LOFTR_STEREO_FAILURE_THRESHOLD: %.3f (valid range: 0.0-1.0)", LOFTR_STEREO_FAILURE_THRESHOLD);
        valid = false;
    }
    
    // 驗證搜索半徑
    if (LOFTR_SEARCH_RADIUS <= 0.0f || LOFTR_SEARCH_RADIUS > 50.0f) {
        ROS_ERROR("Invalid LOFTR_SEARCH_RADIUS: %.2f (valid range: 0.1-50.0)", LOFTR_SEARCH_RADIUS);
        valid = false;
    }
    
    return valid;
}

void printLoFTRConfiguration()
{
    if (!USE_LOFTR) {
        ROS_INFO("LoFTR is disabled");
        return;
    }
    
    ROS_INFO("=== LoFTR Configuration Summary ===");
    ROS_INFO("Model Path: %s", LOFTR_MODEL_PATH.c_str());
    ROS_INFO("Engine Path: %s", LOFTR_ENGINE_PATH.empty() ? "Not specified" : LOFTR_ENGINE_PATH.c_str());
    ROS_INFO("Input Size: %dx%d", LOFTR_INPUT_WIDTH, LOFTR_INPUT_HEIGHT);
    ROS_INFO("Match Threshold: %.3f", LOFTR_MATCH_THRESHOLD);
    ROS_INFO("Max Features: %d", LOFTR_MAX_FEATURES);
    ROS_INFO("Backend: %d", LOFTR_BACKEND);
    ROS_INFO("Use Traditional Tracker: %s", USE_TRADITIONAL_TRACKER ? "Yes" : "No");
    
    ROS_INFO("=== LoFTR Interval Configuration ===");
    ROS_INFO("Frame Interval: %d", LOFTR_FRAME_INTERVAL);
    ROS_INFO("Min Feature Threshold: %d", LOFTR_MIN_FEATURE_THRESHOLD);
    ROS_INFO("Adaptive Interval: %s", LOFTR_ADAPTIVE_INTERVAL ? "Enabled" : "Disabled");
    ROS_INFO("Performance Mode: %d", LOFTR_PERFORMANCE_MODE);
    ROS_INFO("Max Track Length: %d", LOFTR_MAX_TRACK_LENGTH);
    ROS_INFO("Max Total Features: %d", LOFTR_MAX_TOTAL_FEATURES);
    ROS_INFO("Search Radius: %.2f", LOFTR_SEARCH_RADIUS);
    ROS_INFO("Stereo Interval: %d", LOFTR_STEREO_INTERVAL);
    ROS_INFO("Stereo Failure Threshold: %.3f", LOFTR_STEREO_FAILURE_THRESHOLD);
    ROS_INFO("=========================================");
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(), "r");
    if (fh == NULL)
    {
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    // 讀取 LoFTR 配置
    if (fsSettings["use_loftr"].isNone()) {
        ROS_INFO("LoFTR configuration not found, using default values");
        USE_LOFTR = false;
    } else {
        fsSettings["use_loftr"] >> USE_LOFTR;
        ROS_INFO("USE_LOFTR: %s", USE_LOFTR ? "true" : "false");
        
        if (USE_LOFTR) {
            // 讀取 LoFTR 模型路徑
            fsSettings["loftr_model_path"] >> LOFTR_MODEL_PATH;
            if (LOFTR_MODEL_PATH.empty()) {
                ROS_ERROR("LoFTR model path is empty! Please set loftr_model_path in config file");
                USE_LOFTR = false;
            } else {
                ROS_INFO("LOFTR_MODEL_PATH: %s", LOFTR_MODEL_PATH.c_str());
            }
            
            // 讀取可選的 TensorRT 引擎路徑
            if (!fsSettings["loftr_engine_path"].isNone()) {
                fsSettings["loftr_engine_path"] >> LOFTR_ENGINE_PATH;
                ROS_INFO("LOFTR_ENGINE_PATH: %s", LOFTR_ENGINE_PATH.c_str());
            }
            
            // 讀取網絡輸入尺寸
            if (!fsSettings["loftr_input_width"].isNone()) {
                fsSettings["loftr_input_width"] >> LOFTR_INPUT_WIDTH;
            }
            if (!fsSettings["loftr_input_height"].isNone()) {
                fsSettings["loftr_input_height"] >> LOFTR_INPUT_HEIGHT;
            }
            ROS_INFO("LoFTR input size: %dx%d", LOFTR_INPUT_WIDTH, LOFTR_INPUT_HEIGHT);
            
            // 讀取匹配閾值
            if (!fsSettings["loftr_match_threshold"].isNone()) {
                fsSettings["loftr_match_threshold"] >> LOFTR_MATCH_THRESHOLD;
            }
            
            // 讀取最大特徵點數量
            if (!fsSettings["loftr_max_features"].isNone()) {
                fsSettings["loftr_max_features"] >> LOFTR_MAX_FEATURES;
            }
            
            // 讀取後端類型
            if (!fsSettings["loftr_backend"].isNone()) {
                fsSettings["loftr_backend"] >> LOFTR_BACKEND;
            }
            
            // 讀取是否使用傳統跟蹤器
            if (!fsSettings["use_traditional_tracker"].isNone()) {
                fsSettings["use_traditional_tracker"] >> USE_TRADITIONAL_TRACKER;
            }
            
            // ============= 讀取LoFTR間隔配置 =============
            ROS_INFO("=== Reading LoFTR Frame Interval Configuration ===");
            
            // 讀取 LoFTR 執行間隔
            if (!fsSettings["loftr_frame_interval"].isNone()) {
                fsSettings["loftr_frame_interval"] >> LOFTR_FRAME_INTERVAL;
            }
            
            // 讀取最小特徵點閾值
            if (!fsSettings["loftr_min_feature_threshold"].isNone()) {
                fsSettings["loftr_min_feature_threshold"] >> LOFTR_MIN_FEATURE_THRESHOLD;
            }
            
            // 讀取自適應間隔設置
            if (!fsSettings["loftr_adaptive_interval"].isNone()) {
                fsSettings["loftr_adaptive_interval"] >> LOFTR_ADAPTIVE_INTERVAL;
            }
            
            // 讀取性能模式
            if (!fsSettings["loftr_performance_mode"].isNone()) {
                fsSettings["loftr_performance_mode"] >> LOFTR_PERFORMANCE_MODE;
            }
            
            // 讀取調試選項
            if (!fsSettings["loftr_debug_interval"].isNone()) {
                fsSettings["loftr_debug_interval"] >> LOFTR_DEBUG_INTERVAL;
            }
            
            if (!fsSettings["loftr_print_stats"].isNone()) {
                fsSettings["loftr_print_stats"] >> LOFTR_PRINT_STATS;
            }
            
            // ============= 讀取新增的LoFTR參數 =============
            if (!fsSettings["loftr_max_track_length"].isNone()) {
                fsSettings["loftr_max_track_length"] >> LOFTR_MAX_TRACK_LENGTH;
            }

            // 讀取開始幀數
            if (!fsSettings["loftr_start_frame"].isNone()) {
                fsSettings["loftr_start_frame"] >> LOFTR_START_FRAME;
                ROS_INFO("LOFTR_START_FRAME: %d", LOFTR_START_FRAME);
            }
            
            if (!fsSettings["loftr_max_total_features"].isNone()) {
                fsSettings["loftr_max_total_features"] >> LOFTR_MAX_TOTAL_FEATURES;
            }
            
            if (!fsSettings["loftr_search_radius"].isNone()) {
                fsSettings["loftr_search_radius"] >> LOFTR_SEARCH_RADIUS;
            }
            
            if (!fsSettings["loftr_optical_flow_merge_radius"].isNone()) {
                fsSettings["loftr_optical_flow_merge_radius"] >> LOFTR_OPTICAL_FLOW_MERGE_RADIUS;
            }
            
            if (!fsSettings["loftr_force_interval_max"].isNone()) {
                fsSettings["loftr_force_interval_max"] >> LOFTR_FORCE_INTERVAL_MAX;
            }
            
            if (!fsSettings["loftr_stereo_failure_threshold"].isNone()) {
                fsSettings["loftr_stereo_failure_threshold"] >> LOFTR_STEREO_FAILURE_THRESHOLD;
            }
            
            if (!fsSettings["loftr_stereo_interval"].isNone()) {
                fsSettings["loftr_stereo_interval"] >> LOFTR_STEREO_INTERVAL;
            }
            
            if (!fsSettings["loftr_stereo_search_radius"].isNone()) {
                fsSettings["loftr_stereo_search_radius"] >> LOFTR_STEREO_SEARCH_RADIUS;
            }
            
            // 在現有的 LoFTR 參數讀取代碼後面添加：
            if (!fsSettings["loftr_use_traditional_tracker"].isNone()) {
                int use_traditional;
                fsSettings["loftr_use_traditional_tracker"] >> use_traditional;
                LOFTR_USE_TRADITIONAL_TRACKER = (use_traditional != 0);
                ROS_INFO("LOFTR_USE_TRADITIONAL_TRACKER: %s", LOFTR_USE_TRADITIONAL_TRACKER ? "true" : "false");
            }

            if (!fsSettings["loftr_traditional_distance_threshold"].isNone()) {
                fsSettings["loftr_traditional_distance_threshold"] >> LOFTR_TRADITIONAL_DISTANCE_THRESHOLD;
                ROS_INFO("LOFTR_TRADITIONAL_DISTANCE_THRESHOLD: %.1f", LOFTR_TRADITIONAL_DISTANCE_THRESHOLD);
            }

            if (!fsSettings["loftr_performance_stats_window"].isNone()) {
                fsSettings["loftr_performance_stats_window"] >> LOFTR_PERFORMANCE_STATS_WINDOW;
                LOFTR_PERFORMANCE_STATS_WINDOW = std::max(10, LOFTR_PERFORMANCE_STATS_WINDOW);
                ROS_INFO("LOFTR_PERFORMANCE_STATS_WINDOW: %d", LOFTR_PERFORMANCE_STATS_WINDOW);
            }

            if (!fsSettings["loftr_enable_debug_log"].isNone()) {
                int enable_debug;
                fsSettings["loftr_enable_debug_log"] >> enable_debug;
                LOFTR_ENABLE_DEBUG_LOG = (enable_debug != 0);
                ROS_INFO("LOFTR_ENABLE_DEBUG_LOG: %s", LOFTR_ENABLE_DEBUG_LOG ? "true" : "false");
            }

            if (!fsSettings["loftr_show_performance_stats"].isNone()) {
                int show_stats;
                fsSettings["loftr_show_performance_stats"] >> show_stats;
                LOFTR_SHOW_PERFORMANCE_STATS = (show_stats != 0);
                ROS_INFO("LOFTR_SHOW_PERFORMANCE_STATS: %s", LOFTR_SHOW_PERFORMANCE_STATS ? "true" : "false");
            }   

            // 讀取傳統跟蹤器距離閾值
            if (!fsSettings["loftr_traditional_distance_threshold"].isNone()) {
                fsSettings["loftr_traditional_distance_threshold"] >> LOFTR_TRADITIONAL_DISTANCE_THRESHOLD;
                ROS_INFO("LOFTR_TRADITIONAL_DISTANCE_THRESHOLD: %.1f", LOFTR_TRADITIONAL_DISTANCE_THRESHOLD);
            }

            // 讀取性能統計窗口大小
            if (!fsSettings["loftr_performance_stats_window"].isNone()) {
                fsSettings["loftr_performance_stats_window"] >> LOFTR_PERFORMANCE_STATS_WINDOW;
                LOFTR_PERFORMANCE_STATS_WINDOW = std::max(10, LOFTR_PERFORMANCE_STATS_WINDOW);
                ROS_INFO("LOFTR_PERFORMANCE_STATS_WINDOW: %d", LOFTR_PERFORMANCE_STATS_WINDOW);
            }
            ROS_INFO("=== LoFTR Configuration Complete ===");
        }
    }

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if (USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else
    {
        if (ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if (NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib;
        CAM_NAMES.push_back(cam1Path);

        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if (!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    // 讀取傳統深度學習參數（保留兼容性）
    if (!fsSettings["extractor_weight_path"].isNone()) {
        fsSettings["extractor_weight_path"] >> extractor_weight_relative_path;
    }
    if (!fsSettings["matcher_weight_path"].isNone()) {
        fsSettings["matcher_weight_path"] >> matcher_weight_relative_path;
    }
    if (!fsSettings["matche_score_threshold"].isNone()) {
        MATCHER_THRESHOLD = fsSettings["matche_score_threshold"];
    }
    if (!fsSettings["ransacReprojThreshold"].isNone()) {
        ransacReprojThreshold = fsSettings["ransacReprojThreshold"];
    }

    fsSettings.release();
    
    // 驗證LoFTR參數
    if (USE_LOFTR) {
        if (!validateLoFTRParameters()) {
            ROS_ERROR("LoFTR parameter validation failed! Please check your configuration.");
            USE_LOFTR = false;
        } else {
            printLoFTRConfiguration();
        }
    }
    
    // 打印最終配置驗證
    ROS_INFO("=== Final Configuration Validation ===");
    if (USE_LOFTR && LOFTR_FRAME_INTERVAL > 1) {
        ROS_INFO("LoFTR interval mode enabled: Using LoFTR every %d frames", LOFTR_FRAME_INTERVAL);
        if (USE_TRADITIONAL_TRACKER) {
            ROS_INFO("Hybrid tracking: LoFTR + Optical Flow");
        } else {
            ROS_INFO("LoFTR-only tracking with intervals (may lose features in non-LoFTR frames)");
            ROS_WARN("Consider enabling use_traditional_tracker for better feature continuity");
        }
    } else if (USE_LOFTR) {
        ROS_INFO("LoFTR continuous mode: Using LoFTR every frame");
    } else {
        ROS_INFO("Traditional feature tracking enabled");
    }
    ROS_INFO("=======================================");
}