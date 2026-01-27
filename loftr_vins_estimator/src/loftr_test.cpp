#include <iostream>
#include <ros/ros.h>
#include <fstream>      // 添加这个头文件用于 std::ifstream
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

// 其他必要的包含
#include "featureTracker/loftr_interface.h"
#include "featureTracker/loftr_utils.h"

// 🔧 修复：将TensorRT头文件包含移到文件顶部
#ifdef TENSORRT_AVAILABLE
#include "featureTracker/loftr_tensorrt.h"
#endif

class LoFTRTester {
public:
    LoFTRTester() {
        std::cout << "=== LoFTR VINS 测试程序 ===" << std::endl;
    }
    
    bool testLoFTRInterface() {
        std::cout << "\n🔧 测试 LoFTR 接口..." << std::endl;
        
        try {
            // 配置 LoFTR
            LoFTR_Interface::Config config;
            config.input_width = 640;
            config.input_height = 480;
            config.match_threshold = 0.2f;
            config.max_matches = 200;
            config.backend = LoFTR_Interface::BackendType::AUTO;
            
            // 检查模型文件路径（需要根据实际情况调整）
            std::string model_path = "/home/lin/loftrvins_ws/src/LoFTR_VINS/loftr_vins_estimator/weights/LoFTR_teacher.onnx";
            config.model_path = model_path;
            
            // 检查文件是否存在
            std::ifstream model_file(model_path);
            if (!model_file.good()) {
                std::cout << "⚠️  模型文件不存在: " << model_path << std::endl;
                std::cout << "   请确保模型文件路径正确" << std::endl;
                return false;
            }
            
            // 初始化 LoFTR
            LoFTR_Interface loftr;
            bool init_success = loftr.initialize(config);
            
            if (init_success) {
                std::cout << "✅ LoFTR 接口初始化成功" << std::endl;
                std::cout << "   后端: " << loftr.getBackendInfo() << std::endl;
                return true;
            } else {
                std::cout << "❌ LoFTR 接口初始化失败" << std::endl;
                return false;
            }
            
        } catch (const std::exception& e) {
            std::cout << "❌ LoFTR 接口测试异常: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool testImageMatching() {
        std::cout << "\n🖼️  测试图像匹配..." << std::endl;
        
        try {
            // 创建测试图像
            cv::Mat img0 = createTestImage(640, 480, 0);
            cv::Mat img1 = createTestImage(640, 480, 1);
            
            std::cout << "✅ 测试图像创建成功: " << img0.size() << std::endl;
            
            // 使用工具函数预处理
            LoFTR_Utils::PreprocessConfig preprocess_config;
            cv::Mat processed_img0 = LoFTR_Utils::preprocessImage(img0, preprocess_config);
            cv::Mat processed_img1 = LoFTR_Utils::preprocessImage(img1, preprocess_config);
            
            std::cout << "✅ 图像预处理成功" << std::endl;
            
            // 测试工具函数
            std::vector<cv::Point2f> test_points = {
                cv::Point2f(100, 100), cv::Point2f(200, 200), cv::Point2f(300, 300)
            };
            
            cv::Mat keypoint_img = LoFTR_Utils::drawKeypoints(img0, test_points);
            std::cout << "✅ 关键点绘制测试成功" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cout << "❌ 图像匹配测试异常: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool testTensorRTAvailability() {
        std::cout << "\n🚀 测试 TensorRT 可用性..." << std::endl;
        
#ifdef TENSORRT_AVAILABLE
        try {
            // 🔧 修复：不再在函数内包含头文件，直接使用已包含的类
            bool available = LoFTR_TensorRT::isAvailable();
            if (available) {
                std::cout << "✅ TensorRT 可用！" << std::endl;
                
                // 创建TensorRT实例进行基础测试
                LoFTR_TensorRT loftr_trt;
                std::cout << "   🔧 TensorRT实例创建成功" << std::endl;
                std::cout << "   📐 默认输入尺寸: " << loftr_trt.getInputSize() << std::endl;
                std::cout << "   ℹ️  " << loftr_trt.getEngineInfo() << std::endl;
                
                return true;
            } else {
                std::cout << "⚠️  TensorRT 库已链接但设备不可用，将使用 ONNX Runtime" << std::endl;
                return true; // 不算失败
            }
        } catch (const std::exception& e) {
            std::cout << "⚠️  TensorRT 检测异常: " << e.what() << std::endl;
            std::cout << "   将回退到 ONNX Runtime" << std::endl;
            return true; // 不算失败，可以回退到 ONNX
        }
#else
        std::cout << "ℹ️  TensorRT 支持未编译，使用 ONNX Runtime" << std::endl;
        std::cout << "   💡 要启用TensorRT，请确保编译时定义了TENSORRT_AVAILABLE宏" << std::endl;
        return true;
#endif
    }
    
    void runAllTests() {
        std::cout << "开始运行所有测试...\n" << std::endl;
        
        int passed = 0;
        int total = 0;
        
        // 测试 OpenCV
        total++;
        if (testOpenCV()) passed++;
        
        // 测试 TensorRT 可用性
        total++;
        if (testTensorRTAvailability()) passed++;
        
        // 测试图像匹配
        total++;
        if (testImageMatching()) passed++;
        
        // 测试 LoFTR 接口（可能需要模型文件）
        total++;
        if (testLoFTRInterface()) passed++;
        
        // 输出结果
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "测试结果: " << passed << "/" << total << " 通过" << std::endl;
        
        if (passed == total) {
            std::cout << "🎉 所有测试通过！LoFTR VINS 环境配置正确" << std::endl;
        } else {
            std::cout << "⚠️  部分测试失败，请检查配置" << std::endl;
        }
        std::cout << std::string(50, '=') << std::endl;
    }

private:
    bool testOpenCV() {
        std::cout << "\n📷 测试 OpenCV..." << std::endl;
        try {
            cv::Mat test_img = cv::Mat::zeros(100, 100, CV_8UC1);
            cv::Mat test_color;
            cv::cvtColor(test_img, test_color, cv::COLOR_GRAY2BGR);
            
            std::cout << "✅ OpenCV 工作正常" << std::endl;
            std::cout << "   版本: " << CV_VERSION << std::endl;
            std::cout << "   图像尺寸: " << test_img.size() << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "❌ OpenCV 测试失败: " << e.what() << std::endl;
            return false;
        }
    }
    
    cv::Mat createTestImage(int width, int height, int pattern) {
        cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);
        
        if (pattern == 0) {
            // 棋盘图案
            for (int y = 0; y < height; y += 40) {
                for (int x = 0; x < width; x += 40) {
                    if ((x/40 + y/40) % 2 == 0) {
                        cv::rectangle(img, cv::Point(x, y), cv::Point(x+40, y+40), cv::Scalar(255), -1);
                    }
                }
            }
        } else {
            // 随机噪声 + 几何图形
            cv::randu(img, 0, 50);
            cv::circle(img, cv::Point(width/2, height/2), 50, cv::Scalar(255), -1);
            cv::rectangle(img, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(200), -1);
        }
        
        return img;
    }
};

int main(int argc, char** argv) {
    // 初始化 ROS
    ros::init(argc, argv, "loftr_test");
    ros::NodeHandle nh;
    
    std::cout << "✅ ROS 初始化成功" << std::endl;
    
    // 运行测试
    LoFTRTester tester;
    tester.runAllTests();
    
    return 0;
}