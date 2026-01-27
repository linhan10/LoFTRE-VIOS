#include "loftr_onnx.h"
#include <iostream>
#include <algorithm>
#include <numeric>

LoFTR_ONNX::LoFTR_ONNX() 
    : initialized_(false), 
      input_width_(640), 
      input_height_(352),
      batch_size_(1),
      channels_(1),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

LoFTR_ONNX::~LoFTR_ONNX() {
    // 智能指针会自动释放资源
}

bool LoFTR_ONNX::initialize(const std::string& model_path, int input_width, int input_height) {
    try {
        std::cout << "[LoFTR_ONNX] 初始化 ONNX Runtime..." << std::endl;
        
        input_width_ = input_width;
        input_height_ = input_height;
        
        // 创建 ONNX Runtime 环境
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LoFTR_ONNX");
        
        // 创建会话选项
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(4); // 设置线程数
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // 检查是否有 CUDA 提供者
        auto available_providers = Ort::GetAvailableProviders();
        bool cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") != available_providers.end();
        
        if (cuda_available) {
            std::cout << "[LoFTR_ONNX] 启用 CUDA 执行提供者" << std::endl;
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024; // 2GB
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            session_options_->AppendExecutionProvider_CUDA(cuda_options);
        } else {
            std::cout << "[LoFTR_ONNX] 使用 CPU 执行提供者" << std::endl;
        }
        
        // 创建会话
        std::cout << "[LoFTR_ONNX] 加载模型: " << model_path << std::endl;
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(), *session_options_);
        
        // 获取模型输入输出信息
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 获取输入信息
        size_t num_input_nodes = ort_session_->GetInputCount();
        std::cout << "[LoFTR_ONNX] 输入节点数量: " << num_input_nodes << std::endl;
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            // 获取输入名称
            auto input_name = ort_session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(std::string(input_name.get()));
            
            // 获取输入形状
            Ort::TypeInfo input_type_info = ort_session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();
            
            // 处理动态维度
            for (auto& dim : input_dims) {
                if (dim == -1) {
                    dim = batch_size_; // 假设是 batch 维度
                }
            }
            
            input_shapes_.push_back(input_dims);
            
            std::cout << "[LoFTR_ONNX] 输入 " << i << ": " << input_names_[i] << ", 形状: [";
            for (size_t j = 0; j < input_dims.size(); ++j) {
                std::cout << input_dims[j];
                if (j < input_dims.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // 获取输出信息
        size_t num_output_nodes = ort_session_->GetOutputCount();
        std::cout << "[LoFTR_ONNX] 输出节点数量: " << num_output_nodes << std::endl;
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            // 获取输出名称
            auto output_name = ort_session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(std::string(output_name.get()));
            
            // 获取输出形状
            Ort::TypeInfo output_type_info = ort_session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto output_dims = output_tensor_info.GetShape();
            
            // 处理动态维度
            for (auto& dim : output_dims) {
                if (dim == -1) {
                    dim = batch_size_; // 假设是 batch 维度
                }
            }
            
            output_shapes_.push_back(output_dims);
            
            std::cout << "[LoFTR_ONNX] 输出 " << i << ": " << output_names_[i] << ", 形状: [";
            for (size_t j = 0; j < output_dims.size(); ++j) {
                std::cout << output_dims[j];
                if (j < output_dims.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // 验证输入形状是否符合预期
        if (input_shapes_.size() < 2) {
            std::cerr << "[LoFTR_ONNX] 错误: 模型输入数量不足，期望至少2个输入" << std::endl;
            return false;
        }
        
        // 准备输入数据缓冲区
        size_t input_size = batch_size_ * channels_ * input_height_ * input_width_;
        input_data0_.resize(input_size);
        input_data1_.resize(input_size);
        
        std::cout << "[LoFTR_ONNX] 输入数据缓冲区大小: " << input_size << std::endl;
        
        initialized_ = true;
        std::cout << "[LoFTR_ONNX] ONNX Runtime 初始化成功" << std::endl;
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[LoFTR_ONNX] ONNX Runtime 异常: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[LoFTR_ONNX] 标准异常: " << e.what() << std::endl;
        return false;
    }
}

bool LoFTR_ONNX::infer(const cv::Mat& img0, const cv::Mat& img1, std::vector<float>& output) {
    if (!initialized_) {
        std::cerr << "[LoFTR_ONNX] 错误: 未初始化" << std::endl;
        return false;
    }
    
    if (img0.empty() || img1.empty()) {
        std::cerr << "[LoFTR_ONNX] 错误: 输入图像为空" << std::endl;
        return false;
    }
    
    if (img0.size() != cv::Size(input_width_, input_height_) || 
        img1.size() != cv::Size(input_width_, input_height_)) {
        std::cerr << "[LoFTR_ONNX] 错误: 输入图像尺寸不匹配" << std::endl;
        return false;
    }
    
    try {
        // 准备输入张量
        prepareInputTensors(img0, img1);
        
        // 转换字符串名称为 const char* 数组（修复API调用问题）
        std::vector<const char*> input_names_cstr;
        std::vector<const char*> output_names_cstr;
        
        for (const auto& name : input_names_) {
            input_names_cstr.push_back(name.c_str());
        }
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }
        
        // 执行推理 - 修复后的API调用
        auto output_tensors = ort_session_->Run(Ort::RunOptions{nullptr}, 
                                               input_names_cstr.data(), 
                                               input_tensors_.data(), 
                                               input_tensors_.size(),
                                               output_names_cstr.data(), 
                                               output_names_cstr.size());
        
        if (output_tensors.empty()) {
            std::cerr << "[LoFTR_ONNX] 错误: 推理返回空结果" << std::endl;
            return false;
        }
        
        // 获取第一个输出（通常是 confidence matrix）
        auto& result_tensor = output_tensors[0];
        
        // 获取输出数据 - 修复模板调用
        float* output_data = result_tensor.template GetTensorMutableData<float>();
        auto output_shape = result_tensor.GetTensorTypeAndShapeInfo().GetShape();
        
        // 计算输出数据大小
        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= static_cast<size_t>(dim);
        }
        
        // 复制输出数据
        output.resize(output_size);
        std::copy(output_data, output_data + output_size, output.begin());
        
        std::cout << "[LoFTR_ONNX] 推理成功，输出大小: " << output_size << std::endl;
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[LoFTR_ONNX] 推理异常: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[LoFTR_ONNX] 推理错误: " << e.what() << std::endl;
        return false;
    }
}

void LoFTR_ONNX::prepareInputTensors(const cv::Mat& img0, const cv::Mat& img1) {
    // 清空之前的张量
    input_tensors_.clear();
    
    // 将图像转换为张量数据
    imageToTensor(img0, input_data0_.data());
    imageToTensor(img1, input_data1_.data());
    
    // 创建输入张量
    std::vector<int64_t> input_shape = {batch_size_, channels_, input_height_, input_width_};
    
    // 创建第一个输入张量
    input_tensors_.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info_, 
        input_data0_.data(), 
        input_data0_.size(),
        input_shape.data(), 
        input_shape.size()));
    
    // 创建第二个输入张量
    input_tensors_.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info_, 
        input_data1_.data(), 
        input_data1_.size(),
        input_shape.data(), 
        input_shape.size()));
}

void LoFTR_ONNX::imageToTensor(const cv::Mat& img, float* data) {
    // 确保图像是单通道浮点型
    cv::Mat processed_img;
    if (img.type() != CV_32F) {
        img.convertTo(processed_img, CV_32F);
    } else {
        processed_img = img;
    }
    
    // 检查图像尺寸
    if (processed_img.size() != cv::Size(input_width_, input_height_)) {
        cv::resize(processed_img, processed_img, cv::Size(input_width_, input_height_));
    }
    
    // 转换为 CHW 格式（Channel, Height, Width）
    // 对于灰度图像，只有一个通道
    if (processed_img.channels() == 1) {
        // 直接复制数据
        std::memcpy(data, processed_img.ptr<float>(), input_width_ * input_height_ * sizeof(float));
    } else {
        // 如果是多通道，转换为灰度
        cv::Mat gray;
        cv::cvtColor(processed_img, gray, cv::COLOR_BGR2GRAY);
        std::memcpy(data, gray.ptr<float>(), input_width_ * input_height_ * sizeof(float));
    }
}