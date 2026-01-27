#ifndef LOFTR_ONNX_H
#define LOFTR_ONNX_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

// ONNX Runtime 头文件
#include "ort_include/onnxruntime_cxx_api.h"

/**
 * @brief LoFTR ONNX Runtime 后端实现
 */
class LoFTR_ONNX {
public:
    LoFTR_ONNX();
    ~LoFTR_ONNX();
    
    /**
     * @brief 初始化 ONNX Runtime 模型
     * @param model_path ONNX 模型文件路径
     * @param input_width 输入图像宽度
     * @param input_height 输入图像高度
     * @return 初始化是否成功
     */
    bool initialize(const std::string& model_path, int input_width = 640, int input_height = 480);
    
    /**
     * @brief 执行推理
     * @param img0 第一张图像（已预处理）
     * @param img1 第二张图像（已预处理）
     * @param output 输出结果（confidence matrix）
     * @return 推理是否成功
     */
    bool infer(const cv::Mat& img0, const cv::Mat& img1, std::vector<float>& output);
    
    /**
     * @brief 检查是否已初始化
     * @return 是否已初始化
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief 获取模型输入尺寸
     * @return 输入尺寸 (width, height)
     */
    cv::Size getInputSize() const { return cv::Size(input_width_, input_height_); }

private:
    /**
     * @brief 准备输入张量
     * @param img0 第一张图像
     * @param img1 第二张图像
     */
    void prepareInputTensors(const cv::Mat& img0, const cv::Mat& img1);
    
    /**
     * @brief 将 cv::Mat 转换为输入张量数据
     * @param img 输入图像
     * @param data 输出数据指针
     */
    void imageToTensor(const cv::Mat& img, float* data);

private:
    bool initialized_;
    
    // ONNX Runtime 组件
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    
    // 模型信息
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    // 输入参数
    int input_width_;
    int input_height_;
    int batch_size_;
    int channels_;
    
    // 输入数据缓冲区
    std::vector<float> input_data0_;
    std::vector<float> input_data1_;
    
    // 输入张量
    std::vector<Ort::Value> input_tensors_;
    
    // 内存信息
    Ort::MemoryInfo memory_info_;
};

#endif // LOFTR_ONNX_H
