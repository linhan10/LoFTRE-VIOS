#include "loftr_tensorrt.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

// Logger实现
void LoFTR_TensorRT::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

LoFTR_TensorRT::LoFTR_TensorRT() 
    : initialized_(false), 
      input_width_(640), 
      input_height_(480),
      batch_size_(1),
      channels_(1) {
#ifdef TENSORRT_AVAILABLE
    stream_ = nullptr;
    input_buffer0_ = nullptr;
    input_buffer1_ = nullptr;
    output_buffer_ = nullptr;
    input_size_ = 0;
    output_size_ = 0;
#endif
}

LoFTR_TensorRT::~LoFTR_TensorRT() {
#ifdef TENSORRT_AVAILABLE
    if (initialized_) {
        // 释放 CUDA 内存
        if (input_buffer0_) {
            cudaFree(input_buffer0_);
            input_buffer0_ = nullptr;
        }
        if (input_buffer1_) {
            cudaFree(input_buffer1_);
            input_buffer1_ = nullptr;
        }
        if (output_buffer_) {
            cudaFree(output_buffer_);
            output_buffer_ = nullptr;
        }
        
        // 销毁 CUDA 流
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }
#endif
}

bool LoFTR_TensorRT::isAvailable() {
#ifdef TENSORRT_AVAILABLE
    try {
        // 检查 CUDA 设备
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            std::cout << "[TensorRT] 未检测到 CUDA 设备: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        // 检查设备属性
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, 0);
        if (error != cudaSuccess) {
            std::cout << "[TensorRT] 无法获取设备属性: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        std::cout << "[TensorRT] 检测到设备: " << prop.name << std::endl;
        std::cout << "[TensorRT] 计算能力: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "[TensorRT] TensorRT 版本: " << NV_TENSORRT_MAJOR << "." 
                  << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "[TensorRT] 检测异常: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "[TensorRT] TensorRT 支持未编译" << std::endl;
    return false;
#endif
}

bool LoFTR_TensorRT::initialize(const std::string& model_path, 
                               const std::string& engine_path,
                               int input_width, 
                               int input_height) {
#ifdef TENSORRT_AVAILABLE
    try {
        input_width_ = input_width;
        input_height_ = input_height;
        
        std::cout << "[TensorRT] 初始化 TensorRT 后端..." << std::endl;
        std::cout << "[TensorRT] 输入尺寸: " << input_width_ << "x" << input_height_ << std::endl;
        
        // 创建 TensorRT 运行时
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            std::cerr << "[TensorRT] 创建运行时失败" << std::endl;
            return false;
        }
        
        bool engine_loaded = false;
        
        // 尝试加载已有的引擎文件
        if (!engine_path.empty() && std::ifstream(engine_path).good()) {
            std::cout << "[TensorRT] 尝试加载引擎文件: " << engine_path << std::endl;
            engine_loaded = loadEngineFromFile(engine_path);
            if (!engine_loaded) {
                std::cout << "[TensorRT] 引擎文件加载失败，尝试重新构建..." << std::endl;
            }
        }
        
        // 如果没有引擎文件或加载失败，从 ONNX 构建
        if (!engine_loaded) {
            std::cout << "[TensorRT] 从 ONNX 构建引擎..." << std::endl;
            if (!buildEngineFromONNX(model_path, engine_path)) {
                std::cerr << "[TensorRT] 构建引擎失败" << std::endl;
                return false;
            }
        }
        
        // 验证引擎
        if (!engine_) {
            std::cerr << "[TensorRT] 引擎为空" << std::endl;
            return false;
        }
        
        // 创建执行上下文
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            std::cerr << "[TensorRT] 创建执行上下文失败" << std::endl;
            return false;
        }
        
        // 创建 CUDA 流 - 使用高优先级流
        int priority_high, priority_low;
        cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
        cudaError_t error = cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority_high);
        if (error != cudaSuccess) {
            std::cerr << "[TensorRT] 创建 CUDA 流失败: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        // 准备缓冲区
        if (!prepareBuffers()) {
            std::cerr << "[TensorRT] 准备缓冲区失败" << std::endl;
            return false;
        }
        // 修复预热问题
        initialized_ = true;  // 必须在预热前设置为true！
        // 预热GPU
        std::cout << "[TensorRT] GPU预热中..." << std::endl;
        cv::Mat dummy0 = cv::Mat::zeros(input_height_, input_width_, CV_32F);
        cv::Mat dummy1 = cv::Mat::zeros(input_height_, input_width_, CV_32F);
        std::vector<float> dummy_output;
        for(int i = 0; i < 3; i++) {
            infer(dummy0, dummy1, dummy_output);
        }
        
        initialized_ = true;
        std::cout << "[TensorRT] 初始化成功" << std::endl;
        std::cout << "[TensorRT] " << getEngineInfo() << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[TensorRT] 初始化异常: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "[TensorRT] TensorRT 支持未编译" << std::endl;
    return false;
#endif
}

bool LoFTR_TensorRT::infer(const cv::Mat& img0, const cv::Mat& img1, std::vector<float>& output) {
#ifdef TENSORRT_AVAILABLE
    if (!initialized_) {
        std::cerr << "[TensorRT] 错误: 未初始化" << std::endl;
        return false;
    }
    
    if (img0.empty() || img1.empty()) {
        std::cerr << "[TensorRT] 错误: 输入图像为空" << std::endl;
        return false;
    }
    
    if (img0.size() != cv::Size(input_width_, input_height_) || 
        img1.size() != cv::Size(input_width_, input_height_)) {
        std::cerr << "[TensorRT] 错误: 图像尺寸不匹配" << std::endl;
        return false;
    }
    
    try {
        // 将输入数据复制到 GPU
        copyInputToDevice(img0, img1);
        
        // 执行推理
        bool success = context_->enqueueV2(bindings_.data(), stream_, nullptr);
        if (!success) {
            std::cerr << "[TensorRT] 推理执行失败" << std::endl;
            return false;
        }
        
        // 等待推理完成
        cudaError_t sync_error = cudaStreamSynchronize(stream_);
        if (sync_error != cudaSuccess) {
            std::cerr << "[TensorRT] CUDA 同步失败: " << cudaGetErrorString(sync_error) << std::endl;
            return false;
        }
        
        // 复制输出数据
        copyOutputFromDevice(output);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[TensorRT] 推理异常: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "[TensorRT] TensorRT 支持未编译" << std::endl;
    return false;
#endif
}

std::string LoFTR_TensorRT::getEngineInfo() const {
#ifdef TENSORRT_AVAILABLE
    if (!initialized_ || !engine_) {
        return "TensorRT 引擎未初始化";
    }
    
    std::string info = "TensorRT 引擎信息:\n";
    info += "- 绑定数量: " + std::to_string(engine_->getNbBindings()) + "\n";
    info += "- 输入尺寸: " + std::to_string(input_width_) + "x" + std::to_string(input_height_) + "\n";
    info += "- 最大批次大小: " + std::to_string(engine_->getMaxBatchSize()) + "\n";
    
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        auto dims = engine_->getBindingDimensions(i);
        std::string binding_info = "- ";
        binding_info += engine_->bindingIsInput(i) ? "输入" : "输出";
        binding_info += std::to_string(i) + ": [";
        for (int j = 0; j < dims.nbDims; ++j) {
            binding_info += std::to_string(dims.d[j]);
            if (j < dims.nbDims - 1) binding_info += ", ";
        }
        binding_info += "]\n";
        info += binding_info;
    }
    
    return info;
#else
    return "TensorRT 支持未编译";
#endif
}

#ifdef TENSORRT_AVAILABLE

bool LoFTR_TensorRT::buildEngineFromONNX(const std::string& onnx_path, const std::string& engine_path) {
    try {
        // 创建构建器
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
        if (!builder) {
            std::cerr << "[TensorRT] 创建构建器失败" << std::endl;
            return false;
        }
        
        // 创建网络定义
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network) {
            std::cerr << "[TensorRT] 创建网络定义失败" << std::endl;
            return false;
        }
        
        // 创建 ONNX 解析器
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
        if (!parser) {
            std::cerr << "[TensorRT] 创建 ONNX 解析器失败" << std::endl;
            return false;
        }
        
        // 解析 ONNX 模型
        std::cout << "[TensorRT] 解析 ONNX 模型: " << onnx_path << std::endl;
        if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "[TensorRT] 解析 ONNX 模型失败" << std::endl;
            for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
                std::cerr << "[TensorRT] 解析错误: " << parser->getError(i)->desc() << std::endl;
            }
            return false;
        }
        
        // 创建构建配置 - Orin AGX 优化
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            std::cerr << "[TensorRT] 创建构建配置失败" << std::endl;
            return false;
        }
        
        // ========== Orin AGX 专门优化 ==========
        
        // 1. 设置更大的工作空间（Orin有充足内存）
        config->setMaxWorkspaceSize(8ULL << 30);  // 8GB 工作空间
        std::cout << "[TensorRT] 设置工作空间: 8GB" << std::endl;
        
        // 2. 启用 FP16 精度（Orin AGX 有强大的FP16支持）
        if (builder->platformHasFastFp16()) {
            std::cout << "[TensorRT] 启用 FP16 优化（Orin AGX优化）" << std::endl;
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        
        // 3. 启用 TF32（Ampere架构特性）
        config->setFlag(nvinfer1::BuilderFlag::kTF32);
        std::cout << "[TensorRT] 启用 TF32 优化（Ampere架构）" << std::endl;
        
        // 4. 禁用某些可能导致性能问题的功能
        config->setFlag(nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE);
        
        // 5. 设置优化器选项
        if (builder->getNbDLACores() > 0) {
            std::cout << "[TensorRT] 检测到 " << builder->getNbDLACores() << " 个DLA核心（不使用，GPU更快）" << std::endl;
        }
        
        // 6. 设置优化配置文件
        auto profile = builder->createOptimizationProfile();
        if (profile) {
            // 为输入设置动态形状范围
            nvinfer1::Dims4 input_dims(batch_size_, channels_, input_height_, input_width_);
            
            for (int i = 0; i < network->getNbInputs(); ++i) {
                auto input = network->getInput(i);
                const char* input_name = input->getName();
                
                profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, input_dims);
                profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, input_dims);
                profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, input_dims);
                
                std::cout << "[TensorRT] 配置输入 " << input_name << ": " 
                          << input_dims.d[0] << "x" << input_dims.d[1] << "x" 
                          << input_dims.d[2] << "x" << input_dims.d[3] << std::endl;
            }
            
            config->addOptimizationProfile(profile);
        }
        
        // 7. 设置策略源（使用CUDNN和CUBLAS）
        config->setTacticSources(
            1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS) |
            1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT) |
            1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUDNN)
        );
        
        // 构建引擎
        std::cout << "[TensorRT] 开始构建引擎（Orin AGX优化版本）..." << std::endl;
        std::cout << "[TensorRT] 这可能需要5-10分钟，请耐心等待..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        engine_.reset(builder->buildEngineWithConfig(*network, *config));
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (!engine_) {
            std::cerr << "[TensorRT] 构建引擎失败" << std::endl;
            return false;
        }
        
        double build_time = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "[TensorRT] 引擎构建成功，耗时: " << build_time << " 秒" << std::endl;
        
        // 保存引擎到文件
        if (!engine_path.empty()) {
            if (saveEngineToFile(engine_path)) {
                std::cout << "[TensorRT] 引擎已保存: " << engine_path << std::endl;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[TensorRT] 构建引擎异常: " << e.what() << std::endl;
        return false;
    }
}

bool LoFTR_TensorRT::loadEngineFromFile(const std::string& engine_path) {
    try {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "[TensorRT] 引擎文件不存在: " << engine_path << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        if (size == 0) {
            std::cerr << "[TensorRT] 引擎文件为空: " << engine_path << std::endl;
            return false;
        }
        
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();
        
        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
        if (!engine_) {
            std::cerr << "[TensorRT] 反序列化引擎失败" << std::endl;
            return false;
        }
        
        std::cout << "[TensorRT] 成功加载引擎文件: " << engine_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[TensorRT] 加载引擎异常: " << e.what() << std::endl;
        return false;
    }
}

bool LoFTR_TensorRT::saveEngineToFile(const std::string& engine_path) {
    try {
        if (!engine_) {
            std::cerr << "[TensorRT] 引擎为空，无法保存" << std::endl;
            return false;
        }
        
        auto engine_data = std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
        if (!engine_data) {
            std::cerr << "[TensorRT] 序列化引擎失败" << std::endl;
            return false;
        }
        
        std::ofstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "[TensorRT] 无法创建引擎文件: " << engine_path << std::endl;
            return false;
        }
        
        file.write(static_cast<char*>(engine_data->data()), engine_data->size());
        file.close();
        
        std::cout << "[TensorRT] 引擎保存成功: " << engine_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[TensorRT] 保存引擎异常: " << e.what() << std::endl;
        return false;
    }
}

bool LoFTR_TensorRT::prepareBuffers() {
    try {
        if (!engine_) {
            std::cerr << "[TensorRT] 引擎未初始化" << std::endl;
            return false;
        }
        
        // 计算输入输出大小
        input_size_ = batch_size_ * channels_ * input_height_ * input_width_ * sizeof(float);
        
        // 动态计算输出大小
        size_t total_output_size = 0;
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            if (!engine_->bindingIsInput(i)) {
                auto dims = engine_->getBindingDimensions(i);
                size_t binding_size = sizeof(float);
                for (int j = 0; j < dims.nbDims; ++j) {
                    binding_size *= dims.d[j];
                }
                total_output_size += binding_size;
                std::cout << "[TensorRT] 输出绑定 " << i << " 尺寸: ";
                for (int j = 0; j < dims.nbDims; ++j) {
                    std::cout << dims.d[j];
                    if (j < dims.nbDims - 1) std::cout << "x";
                }
                std::cout << " (" << binding_size << " bytes)" << std::endl;
            }
        }
        output_size_ = total_output_size;
        
        std::cout << "[TensorRT] 输入缓冲区大小: " << input_size_ << " bytes" << std::endl;
        std::cout << "[TensorRT] 输出缓冲区大小: " << output_size_ << " bytes" << std::endl;
        
        // 分配 GPU 内存 - 使用固定内存提高传输速度
        cudaError_t error;
        
        error = cudaMalloc(&input_buffer0_, input_size_);
        if (error != cudaSuccess) {
            std::cerr << "[TensorRT] 分配输入缓冲区0失败: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        error = cudaMalloc(&input_buffer1_, input_size_);
        if (error != cudaSuccess) {
            std::cerr << "[TensorRT] 分配输入缓冲区1失败: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        error = cudaMalloc(&output_buffer_, output_size_);
        if (error != cudaSuccess) {
            std::cerr << "[TensorRT] 分配输出缓冲区失败: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        // 设置绑定
        bindings_.resize(engine_->getNbBindings());
        input_indices_.clear();
        output_indices_.clear();
        
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            if (engine_->bindingIsInput(i)) {
                if (input_indices_.size() == 0) {
                    bindings_[i] = input_buffer0_;
                    input_indices_.push_back(i);
                    std::cout << "[TensorRT] 绑定输入0到索引 " << i << std::endl;
                } else if (input_indices_.size() == 1) {
                    bindings_[i] = input_buffer1_;
                    input_indices_.push_back(i);
                    std::cout << "[TensorRT] 绑定输入1到索引 " << i << std::endl;
                }
            } else {
                bindings_[i] = output_buffer_;
                output_indices_.push_back(i);
                std::cout << "[TensorRT] 绑定输出到索引 " << i << std::endl;
            }
        }
        
        std::cout << "[TensorRT] 缓冲区准备完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[TensorRT] 准备缓冲区异常: " << e.what() << std::endl;
        return false;
    }
}

void LoFTR_TensorRT::copyInputToDevice(const cv::Mat& img0, const cv::Mat& img1) {
    cv::Mat processed_img0, processed_img1;
    
    // 确保图像是float32类型
    if (img0.type() != CV_32F) {
        img0.convertTo(processed_img0, CV_32F);
    } else {
        processed_img0 = img0;
    }
    
    if (img1.type() != CV_32F) {
        img1.convertTo(processed_img1, CV_32F);
    } else {
        processed_img1 = img1;
    }
    
    // 确保数据连续
    if (!processed_img0.isContinuous()) {
        processed_img0 = processed_img0.clone();
    }
    if (!processed_img1.isContinuous()) {
        processed_img1 = processed_img1.clone();
    }
    
    // 异步复制到 GPU
    cudaMemcpyAsync(input_buffer0_, processed_img0.ptr<float>(), input_size_, 
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(input_buffer1_, processed_img1.ptr<float>(), input_size_, 
                    cudaMemcpyHostToDevice, stream_);
}

void LoFTR_TensorRT::copyOutputFromDevice(std::vector<float>& output) {
    // 使用第一个输出
    if (output_indices_.size() >= 2) {
        size_t single_output_size = 1200 * 1200;
        std::vector<float> all_outputs(output_size_ / sizeof(float));
        
        cudaMemcpyAsync(all_outputs.data(), output_buffer_, output_size_, 
                       cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        
        if (all_outputs.size() >= single_output_size) {
            output.assign(all_outputs.begin(), all_outputs.begin() + single_output_size);
        } else {
            output = all_outputs;
        }
    } else {
        size_t output_count = output_size_ / sizeof(float);
        output.resize(output_count);
        
        cudaMemcpyAsync(output.data(), output_buffer_, output_size_, 
                       cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }
}

#endif // TENSORRT_AVAILABLE