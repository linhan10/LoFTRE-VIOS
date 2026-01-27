#include "loftr_interface.h"
#include "loftr_onnx.h"
#include "loftr_tensorrt.h"
#include <iostream>
#include <chrono> // for time consumption
#include <algorithm>
#include <fstream> // For file I/O

LoFTR_Interface::LoFTR_Interface() 
    : initialized_(false), active_backend_(BackendType::AUTO), 
      total_matches_(0), scale_factor_(1.0f) {
}

// Interallgence pinter will automatacally release source
LoFTR_Interface::~LoFTR_Interface() {
}

// 1. Initial function
/*
1. Input:config
2. Output:bool(sueecss or not)
*/
bool LoFTR_Interface::initialize(const Config& config) {
    config_ = config;
    network_input_size_ = cv::Size(config_.input_width, config_.input_height);
    
    std::cout << "[LoFTR] Start Initialize..." << std::endl;
    std::cout << "[LoFTR] Input Size: " << config_.input_width << "x" << config_.input_height << std::endl;
    std::cout << "[LoFTR] Matching Threshold: " << config_.match_threshold << std::endl;
    
    // 选择后端
    if (config_.backend == BackendType::AUTO) {
        active_backend_ = selectBestBackend();
    } else {
        active_backend_ = config_.backend;
    }
    
    bool success = false;
    
    // initialized according to the chosened backend
    switch (active_backend_) {
        case BackendType::TENSORRT:
            std::cout << "[LoFTR] start TensorRT backend..." << std::endl;
            try {
                tensorrt_backend_ = std::make_unique<LoFTR_TensorRT>();
                success = tensorrt_backend_->initialize(config_.model_path, config_.engine_path,
                                                       config_.input_width, config_.input_height);
                if (success) {
                    std::cout << "[LoFTR] TensorRT successed" << std::endl;
                } else {
                    std::cout << "[LoFTR] TensorRT failed，try ONNX Runtime..." << std::endl;
                    tensorrt_backend_.reset();
                    active_backend_ = BackendType::ONNX_RUNTIME;
                }
            } catch (const std::exception& e) {
                std::cout << "[LoFTR] TensorRT error: " << e.what() << std::endl;
                tensorrt_backend_.reset();
                active_backend_ = BackendType::ONNX_RUNTIME;
            }
            break;
            
        case BackendType::ONNX_RUNTIME:
            // 如果 TensorRT 失败，尝试 ONNX Runtime
            success = false;
            break;
            
        case BackendType::AUTO:
            // 🔧 修复：处理AUTO枚举值
            // 这种情况不应该发生，因为上面已经转换了
            std::cout << "[LoFTR] Auto backend already selected" << std::endl;
            success = false;
            break;
    }
    
    // 如果 TensorRT 失败，或者直接选择了 ONNX Runtime
    if (!success && active_backend_ == BackendType::ONNX_RUNTIME) {
        std::cout << "[LoFTR] initialized ONNX Runtime backend..." << std::endl;
        try {
            onnx_backend_ = std::make_unique<LoFTR_ONNX>();
            success = onnx_backend_->initialize(config_.model_path, config_.input_width, config_.input_height);
            if (success) {
                std::cout << "[LoFTR] ONNX Runtime initialized success" << std::endl;
            } else {
                std::cout << "[LoFTR] ONNX Runtime initilized failed" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "[LoFTR] ONNX Runtime error: " << e.what() << std::endl;
            onnx_backend_.reset();
        }
    }
    
    initialized_ = success;
    
    if (initialized_) {
        std::cout << "[LoFTR] initialized success " << getBackendInfo() << std::endl;
        resetStats();
    } else {
        std::cout << "[LoFTR] initialized failed" << std::endl;
    }
    
    return initialized_;
}


// 2. Start matching function
/*
1. Input: img0 img1
2. Output:Match result(keypoints0, keypoints1, confidence, num_matches, inference_time_ms)
*/
LoFTR_Interface::MatchResult LoFTR_Interface::match_images(const cv::Mat& img0, const cv::Mat& img1) {
    MatchResult result;
    
    if (!initialized_) {
        std::cerr << "[LoFTR] error:Not initilized" << std::endl;
        return result;
    }
    
    if (img0.empty() || img1.empty()) {
        std::cerr << "[LoFTR] error:input image is empty" << std::endl;
        return result;
    }
    // 1.pre_process, 2.inference, 3.post_process time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Pre_process Image
        cv::Mat processed_img0 = preprocessImage(img0);
        cv::Mat processed_img1 = preprocessImage(img1);
        
        // Inference according to the backend(trt or onnx)
        bool inference_success = false;
        std::vector<float> raw_output;
        
        if (active_backend_ == BackendType::TENSORRT && tensorrt_backend_) {
            inference_success = tensorrt_backend_->infer(processed_img0, processed_img1, raw_output);
        } else if (active_backend_ == BackendType::ONNX_RUNTIME && onnx_backend_) {
            inference_success = onnx_backend_->infer(processed_img0, processed_img1, raw_output);
        }
        
        // Post_process result
        if (inference_success) {
            
            result = postprocessMatches(raw_output, img0.size(), img1.size());
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.inference_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            // 更新统计信息
            inference_times_.push_back(result.inference_time_ms);
            total_matches_++;
            
            std::cout << "[LoFTR] Matching Finished: " << result.num_matches << " pair matching, "
                      << "time: " << result.inference_time_ms << " ms" << std::endl;
        } else {
            std::cerr << "[LoFTR] Inference failed" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[LoFTR] matching error: " << e.what() << std::endl;
    }
    
    return result;
}

// 3. Select the backend automatically
LoFTR_Interface::BackendType LoFTR_Interface::selectBestBackend() {
    std::cout << "[LoFTR] auto select the best backend..." << std::endl;
    
    // Firse is TensorRT
    try {
        auto test_tensorrt = std::make_unique<LoFTR_TensorRT>();
        if (test_tensorrt->isAvailable()) {
            std::cout << "[LoFTR]  TensorRT supported" << std::endl;
            return BackendType::TENSORRT;
        }
    } catch (...) {
        // TensorRT 不可用
    }
    
    // 回退到 ONNX Runtime
    std::cout << "[LoFTR] Back to ONNX Runtime" << std::endl;
    return BackendType::ONNX_RUNTIME;
}

// 4. Preprocess Image
cv::Mat LoFTR_Interface::preprocessImage(const cv::Mat& img) {
    cv::Mat processed;
    
    // Convert to grayscale image
    if (img.channels() == 3) {
        cv::cvtColor(img, processed, cv::COLOR_BGR2GRAY);
    } else {
        processed = img.clone();
    }
    
    // Adjust the size
    if (processed.size() != network_input_size_) {
        cv::resize(processed, processed, network_input_size_);
        
        // Caculate the scale factor (for the cooderinate transfer)
        scale_factor_ = std::max(
            static_cast<float>(img.cols) / network_input_size_.width,
            static_cast<float>(img.rows) / network_input_size_.height
        );
    }
    
    // normalized 
    processed.convertTo(processed, CV_32F, 1.0/255.0);
    
    return processed;
}

// 🔧 修复后的 postprocessMatches 函数
LoFTR_Interface::MatchResult LoFTR_Interface::postprocessMatches(
    const std::vector<float>& raw_result, cv::Size img0_size, cv::Size img1_size) {
    
    MatchResult result;
    
    if (raw_result.empty()) {
        std::cout << "[LoFTR] 警告: 原始结果为空" << std::endl;
        return result;
    }
    
    try {
        // 🔧 修正：使用正确的分辨率
        float threshold = config_.match_threshold;  // 使用配置的阈值
        int feature_resolution = 8;  // LoFTR标准分辨率是16，不是8
        int input_height = config_.input_height;
        int input_width = config_.input_width;
        
        // 计算特征图尺寸
        int hw0_h = input_height / feature_resolution;  // 352/16 = 22
        int hw0_w = input_width / feature_resolution;   // 640/16 = 40
        int feature_num = hw0_h * hw0_w;  // 22*40 = 880

        int expected_size = feature_num * feature_num;
        
        std::cout << "[LoFTR] 输入图像尺寸: " << input_width << "x" << input_height << std::endl;
        std::cout << "[LoFTR] 特征图尺寸: " << hw0_w << "x" << hw0_h 
                  << " (total: " << feature_num << ")" << std::endl;
        std::cout << "[LoFTR] 期望矩阵大小: " << expected_size 
                  << ", 实际大小: " << raw_result.size() << std::endl;
        
        if (raw_result.size() != expected_size) {
            std::cerr << "[LoFTR] 输出大小不匹配！" << std::endl;
            return result;
        }
        
        // 🔧 搜索 confidence matrix 中的高值匹配
        std::vector<std::tuple<int, int, float>> matches;
        
        for (int i = 0; i < feature_num; ++i) {
            for (int j = 0; j < feature_num; ++j) {
                int idx = i * feature_num + j;
                float confidence = raw_result[idx];
                
                if (confidence > threshold) {
                    matches.emplace_back(i, j, confidence);
                }
            }
        }
        
        std::cout << "[LoFTR] 找到 " << matches.size() 
                  << " 个候选匹配 (阈值: " << threshold << ")" << std::endl;
        
        if (matches.empty()) {
            std::cout << "[LoFTR] 没有找到满足阈值的匹配点" << std::endl;
            return result;
        }
        
        // 按置信度排序
        std::sort(matches.begin(), matches.end(), 
                 [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });
        
        // 限制匹配数量
        int max_matches = std::min(static_cast<int>(matches.size()), config_.max_matches);
        
        // 🔧 关键修复：正确的坐标转换逻辑
        for (int m = 0; m < max_matches; ++m) {
            int i = std::get<0>(matches[m]);  // img0的特征点索引
            int j = std::get<1>(matches[m]);  // img1的特征点索引
            float confidence = std::get<2>(matches[m]);
            
            // *** 关键修复：分别处理两张图片的坐标 ***
            
            // img0的特征点坐标（i索引对应）
            int grid_x0 = i % hw0_w;
            int grid_y0 = i / hw0_w;
            float network_x0 = grid_x0 * feature_resolution + feature_resolution / 2.0f;
            float network_y0 = grid_y0 * feature_resolution + feature_resolution / 2.0f;
            
            // img1的特征点坐标（j索引对应）
            int grid_x1 = j % hw0_w;
            int grid_y1 = j / hw0_w;
            float network_x1 = grid_x1 * feature_resolution + feature_resolution / 2.0f;
            float network_y1 = grid_y1 * feature_resolution + feature_resolution / 2.0f;
            
            // 转换为原始图像坐标
            float final_x0 = network_x0 * img0_size.width / input_width;
            float final_y0 = network_y0 * img0_size.height / input_height;
            float final_x1 = network_x1 * img1_size.width / input_width;
            float final_y1 = network_y1 * img1_size.height / input_height;
            
            // 调试前几个匹配点的坐标转换
            if (m < 3) {
                std::cout << "[LoFTR] Match " << m << ": "
                          << "grid0(" << grid_x0 << "," << grid_y0 << ") -> img0(" << final_x0 << "," << final_y0 << "), "
                          << "grid1(" << grid_x1 << "," << grid_y1 << ") -> img1(" << final_x1 << "," << final_y1 << "), "
                          << "conf=" << confidence << std::endl;
            }
            
            // 检查坐标有效性
            if (final_x0 >= 0 && final_x0 < img0_size.width &&
                final_y0 >= 0 && final_y0 < img0_size.height &&
                final_x1 >= 0 && final_x1 < img1_size.width &&
                final_y1 >= 0 && final_y1 < img1_size.height) {
                
                result.keypoints0.emplace_back(final_x0, final_y0);
                result.keypoints1.emplace_back(final_x1, final_y1);
                result.confidence.push_back(confidence);
            }
        }
        
        result.num_matches = result.keypoints0.size();
        
        std::cout << "[LoFTR] 最终有效匹配: " << result.num_matches << std::endl;
        
        if (result.num_matches > 0) {
            // 显示坐标范围以验证修复效果
            auto [min_x0, max_x0] = std::minmax_element(result.keypoints0.begin(), result.keypoints0.end(),
                [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
            auto [min_y0, max_y0] = std::minmax_element(result.keypoints0.begin(), result.keypoints0.end(),
                [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
                
            auto [min_x1, max_x1] = std::minmax_element(result.keypoints1.begin(), result.keypoints1.end(),
                [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
            auto [min_y1, max_y1] = std::minmax_element(result.keypoints1.begin(), result.keypoints1.end(),
                [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
            
            std::cout << "[LoFTR] img0坐标范围: x=[" << min_x0->x << ", " << max_x0->x 
                      << "], y=[" << min_y0->y << ", " << max_y0->y << "]" << std::endl;
            std::cout << "[LoFTR] img1坐标范围: x=[" << min_x1->x << ", " << max_x1->x 
                      << "], y=[" << min_y1->y << ", " << max_y1->y << "]" << std::endl;
            
            if (!result.confidence.empty()) {
                float min_conf = *std::min_element(result.confidence.begin(), result.confidence.end());
                float max_conf = *std::max_element(result.confidence.begin(), result.confidence.end());
                std::cout << "[LoFTR] 置信度范围: " << min_conf << " - " << max_conf << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[LoFTR] 后处理异常: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<cv::Point2f> LoFTR_Interface::rescalePoints(
    const std::vector<cv::Point2f>& points, cv::Size network_size, cv::Size original_size) {
    
    std::vector<cv::Point2f> rescaled_points;
    rescaled_points.reserve(points.size());
    
    float scale_x = static_cast<float>(original_size.width) / network_size.width;
    float scale_y = static_cast<float>(original_size.height) / network_size.height;
    
    for (const auto& pt : points) {
        rescaled_points.emplace_back(pt.x * scale_x, pt.y * scale_y);
    }
    
    return rescaled_points;
}

std::string LoFTR_Interface::getBackendInfo() const {
    switch (active_backend_) {
        case BackendType::TENSORRT:
            return "TensorRT";
        case BackendType::ONNX_RUNTIME:
            return "ONNX Runtime";
        case BackendType::AUTO:
            return "AUTO (not selected)";
        default:
            return "Unknown";
    }
}

double LoFTR_Interface::getAverageInferenceTime() const {
    if (inference_times_.empty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (double time : inference_times_) {
        sum += time;
    }
    return sum / inference_times_.size();
}

void LoFTR_Interface::resetStats() {
    inference_times_.clear();
    total_matches_ = 0;
}

// LoFTR_Utils 实现
namespace LoFTR_Utils {

// 6. Visualized function
cv::Mat visualizeMatches(const cv::Mat& img0, const cv::Mat& img1,
                        const LoFTR_Interface::MatchResult& result,
                        bool show_lines) {
    // 创建并排显示的图像
    cv::Mat vis_img;
    cv::hconcat(img0, img1, vis_img);
    
    // 转换为彩色图像（如果需要）
    if (vis_img.channels() == 1) {
        cv::cvtColor(vis_img, vis_img, cv::COLOR_GRAY2BGR);
    }
    
    int img0_width = img0.cols;
    
    // 绘制关键点和匹配线
    for (size_t i = 0; i < result.keypoints0.size(); ++i) {
        cv::Point2f pt0 = result.keypoints0[i];
        cv::Point2f pt1 = result.keypoints1[i];
        pt1.x += img0_width; // 调整右图的 x 坐标
        
        // 根据置信度确定颜色
        float conf = result.confidence.empty() ? 1.0f : result.confidence[i];
        cv::Scalar color(0, 255 * conf, 255 * (1 - conf)); // 绿色到红色
        
        // 绘制关键点
        cv::circle(vis_img, pt0, 3, color, -1);
        cv::circle(vis_img, pt1, 3, color, -1);
        
        // 绘制连接线
        if (show_lines) {
            cv::line(vis_img, pt0, pt1, color, 1);
        }
    }
    
    // 添加信息文本
    std::string info = "Matches: " + std::to_string(result.num_matches) + 
                      " | Time: " + std::to_string(static_cast<int>(result.inference_time_ms)) + "ms";
    cv::putText(vis_img, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    return vis_img;
}

// 7. File I/O
bool saveMatchesToFile(const LoFTR_Interface::MatchResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "# LoFTR Match Results\n";
    file << "# Format: x0 y0 x1 y1 confidence\n";
    file << "# Matches: " << result.num_matches << "\n";
    file << "# Inference time: " << result.inference_time_ms << " ms\n";
    
    for (size_t i = 0; i < result.keypoints0.size(); ++i) {
        file << result.keypoints0[i].x << " " << result.keypoints0[i].y << " "
             << result.keypoints1[i].x << " " << result.keypoints1[i].y << " ";
        
        if (!result.confidence.empty()) {
            file << result.confidence[i];
        } else {
            file << "1.0";
        }
        file << "\n";
    }
    
    return true;
}

LoFTR_Interface::MatchResult loadMatchesFromFile(const std::string& filename) {
    LoFTR_Interface::MatchResult result;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        return result;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        float x0, y0, x1, y1, conf;
        if (iss >> x0 >> y0 >> x1 >> y1 >> conf) {
            result.keypoints0.emplace_back(x0, y0);
            result.keypoints1.emplace_back(x1, y1);
            result.confidence.push_back(conf);
        }
    }
    
    result.num_matches = result.keypoints0.size();
    return result;
}

double computeMatchQuality(const LoFTR_Interface::MatchResult& result) {
    if (result.confidence.empty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (float conf : result.confidence) {
        sum += conf;
    }
    return sum / result.confidence.size();
}

} // namespace LoFTR_Utils