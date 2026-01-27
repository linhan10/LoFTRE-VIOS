#ifndef LOFTR_INTERFACE_H
#define LOFTR_INTERFACE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <utility>

// 前向声明
class LoFTR_ONNX;
class LoFTR_TensorRT;

/**
 * @brief LoFTR 特征匹配接口类
 * 
 * 这是 LoFTR 功能的统一接口，支持：
 * - ONNX Runtime 推理
 * - TensorRT 推理
 * - 自动后端选择
 */



class LoFTR_Interface {
public:
    enum BackendType {
        ONNX_RUNTIME = 0,
        TENSORRT = 1,
        AUTO = 2  // 自动选择最优后端
    };
    struct Config {
        int input_width = 640;          // input image width
        int input_height = 352;         // input image height
        float match_threshold = 0.2f;   // 匹配置信度阈值
        int max_matches = 2000;         // 最大匹配点数
        bool use_outdoor_model = true;  // 使用室外模型
        BackendType backend = AUTO;     // 推理后端
        std::string model_path;         // 模型文件路径
        std::string engine_path;        // TensorRT 引擎路径（可选）
        
        Config() = default;
    };
    struct MatchResult {
        std::vector<cv::Point2f> keypoints0;  // 图像0的关键点
        std::vector<cv::Point2f> keypoints1;  // 图像1的关键点
        std::vector<float> confidence;        // 匹配置信度
        int num_matches;                      // 匹配点数量
        double inference_time_ms;             // 推理时间（毫秒）
        
        MatchResult() : num_matches(0), inference_time_ms(0.0) {}
    };

public:
    LoFTR_Interface();
    ~LoFTR_Interface();
    bool initialize(const Config& config);
    MatchResult match_images(const cv::Mat& img0, const cv::Mat& img1);
    const Config& getConfig() const { return config_; }
    bool isInitialized() const { return initialized_; }
    std::string getBackendInfo() const;
    double getAverageInferenceTime() const;
    void resetStats();

private:

    BackendType selectBestBackend();
    cv::Mat preprocessImage(const cv::Mat& img);
    MatchResult postprocessMatches(const std::vector<float>& raw_result, 
                                  cv::Size img0_size, cv::Size img1_size);

    std::vector<cv::Point2f> rescalePoints(const std::vector<cv::Point2f>& points,
                                          cv::Size network_size, cv::Size original_size);

private:
    Config config_;                                    // 配置参数
    bool initialized_;                                 // 是否已初始化
    BackendType active_backend_;                       // 当前使用的后端
    
    // 推理后端实例
    std::unique_ptr<LoFTR_ONNX> onnx_backend_;
    std::unique_ptr<LoFTR_TensorRT> tensorrt_backend_;
    
    // 性能统计
    mutable std::vector<double> inference_times_;      // 推理时间记录
    mutable size_t total_matches_;                     // 总匹配次数
    
    // 图像预处理参数
    cv::Size network_input_size_;                      // 网络输入尺寸
    float scale_factor_;                               // 缩放因子
};


namespace LoFTR_Utils {
    /**
     * @brief 可视化匹配结果
     * @param img0 第一张图像
     * @param img1 第二张图像
     * @param result 匹配结果
     * @param show_lines 是否显示连接线
     * @return 可视化图像
     */
    cv::Mat visualizeMatches(const cv::Mat& img0, const cv::Mat& img1,
                            const LoFTR_Interface::MatchResult& result,
                            bool show_lines = true);

    /**
     * @brief 保存匹配结果到文件
     * @param result 匹配结果
     * @param filename 文件路径
     * @return 是否保存成功
     */
    bool saveMatchesToFile(const LoFTR_Interface::MatchResult& result,
                          const std::string& filename);

    /**
     * @brief 从文件加载匹配结果
     * @param filename 文件路径
     * @return 匹配结果
     */
    LoFTR_Interface::MatchResult loadMatchesFromFile(const std::string& filename);

    /**
     * @brief 计算匹配质量指标
     * @param result 匹配结果
     * @return 质量指标（平均置信度）
     */
    double computeMatchQuality(const LoFTR_Interface::MatchResult& result);
}

#endif // LOFTR_INTERFACE_H
