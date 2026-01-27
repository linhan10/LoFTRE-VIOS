#ifndef LOFTR_TENSORRT_H
#define LOFTR_TENSORRT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

// 🔧 使用 CMakeLists.txt 中定义的宏
#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <NvInferRuntime.h>
#endif

/**
 * @brief LoFTR TensorRT 后端实现
 * 提供高性能的 GPU 推理支持
 * 
 * 🔧 修复说明：
 * - 基于 Python 版本分析，TensorRT 输出两个 1200x1200 矩阵
 * - 使用第二个输出作为 confidence matrix（与 Python 版本一致）
 * - 改进了输出处理和坐标转换逻辑
 */
class LoFTR_TensorRT {
public:
    LoFTR_TensorRT();
    ~LoFTR_TensorRT();
    
    /**
     * @brief 检查 TensorRT 是否可用
     * @return TensorRT 是否可用
     */
    static bool isAvailable();
    
    /**
     * @brief 初始化 TensorRT 引擎
     * @param model_path ONNX 模型文件路径
     * @param engine_path TensorRT 引擎保存路径（可选）
     * @param input_width 输入图像宽度
     * @param input_height 输入图像高度
     * @return 初始化是否成功
     */
    bool initialize(const std::string& model_path, 
                   const std::string& engine_path = "",
                   int input_width = 640, 
                   int input_height = 480);
    
    /**
     * @brief 执行推理
     * @param img0 第一张图像（已预处理）
     * @param img1 第二张图像（已预处理）
     * @param output 输出结果（confidence matrix，1200x1200）
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
    
    /**
     * @brief 获取引擎信息
     * @return 引擎信息字符串
     */
    std::string getEngineInfo() const;

private:
    // 基础成员变量
    bool initialized_;
    int input_width_;
    int input_height_;
    int batch_size_;
    int channels_;

    // 🔧 修复：Logger类定义移到条件编译外，确保虚函数表完整
    class Logger : public 
#ifdef TENSORRT_AVAILABLE
        nvinfer1::ILogger
#else
        // 当 TensorRT 不可用时提供一个空的基类
        struct { public: enum Severity { kINTERNAL_ERROR, kERROR, kWARNING, kINFO, kVERBOSE }; }
#endif
    {
    public:
#ifdef TENSORRT_AVAILABLE
        void log(Severity severity, const char* msg) noexcept override;
#else
        void log(int severity, const char* msg) {} // 空实现
#endif
    };

#ifdef TENSORRT_AVAILABLE
    /**
     * @brief TensorRT 专用方法
     */
    bool buildEngineFromONNX(const std::string& onnx_path, const std::string& engine_path);
    bool loadEngineFromFile(const std::string& engine_path);
    bool saveEngineToFile(const std::string& engine_path);
    bool prepareBuffers();
    void copyInputToDevice(const cv::Mat& img0, const cv::Mat& img1);
    
    /**
     * @brief 从设备复制输出数据
     * 🔧 修复：基于 Python 版本，使用第二个输出矩阵
     * @param output 输出向量，包含 1200x1200 的 confidence matrix
     */
    void copyOutputFromDevice(std::vector<float>& output);
    
    // TensorRT 成员变量
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    // CUDA 相关
    cudaStream_t stream_;
    void* input_buffer0_;
    void* input_buffer1_;
    void* output_buffer_;
    
    // 缓冲区大小
    size_t input_size_;
    size_t output_size_;
    
    // 绑定信息
    std::vector<void*> bindings_;
    std::vector<int> input_indices_;
    std::vector<int> output_indices_;
    
#else
    // 🔧 非 TensorRT 环境下的占位符成员
    Logger logger_;  // 仍然需要这个成员，但会使用空实现
#endif // TENSORRT_AVAILABLE
};

#endif // LOFTR_TENSORRT_H
