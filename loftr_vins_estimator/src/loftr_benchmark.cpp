#include <iostream>
#include <ros/ros.h>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>  // 用于创建目录

// 其他必要的包含
#include "featureTracker/loftr_interface.h"
#include "featureTracker/loftr_utils.h"

// for run the rosbag
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/foreach.hpp>

class LoFTRBenchmark {
public:
    LoFTRBenchmark(bool use_rosbag = false, 
                const std::string& rosbag_path = "", 
                const std::string& image_topic = "/cam0/image_raw",
                int max_frames = 1000,  // 🔧 新增：最大测试帧数
                bool save_all_results = false)  // 🔧 新增：是否保存所有结果
        : use_rosbag_(use_rosbag), rosbag_path_(rosbag_path), image_topic_(image_topic),
          max_test_frames_(max_frames), save_all_results_(save_all_results) {
        
        std::cout << "=== LoFTR VINS 大规模性能测试 ===" << std::endl;
        std::cout << "🎯 测试帧数: " << max_test_frames_ << std::endl;
        std::cout << "💾 保存所有结果: " << (save_all_results_ ? "是" : "否") << std::endl;
        
        if (use_rosbag_) {
            std::cout << "📦 模式: ROS Bag 测试 (EuRoC 数据集)" << std::endl;
            std::cout << "   Bag 文件: " << rosbag_path_ << std::endl;
            std::cout << "   图像话题: " << image_topic_ << std::endl;
            
            use_stereo_ = (image_topic_ == "stereo");
            if (use_stereo_) {
                left_topic_ = "/cam0/image_raw";
                right_topic_ = "/cam1/image_raw";
                std::cout << "   双目模式: " << left_topic_ << " + " << right_topic_ << std::endl;
            }
        } else {
            std::cout << "🖼️  模式: 静态图像测试" << std::endl;
            dataset_path_ = "/home/lin/loftrvins_ws/dataset/";
            img1_path_ = dataset_path_ + "match1.png";
            img2_path_ = dataset_path_ + "match2.png";
            use_stereo_ = false;
        }
        
        // 设置结果保存目录
        result_path_ = "/home/lin/loftrvins_ws/result/";
        
        // 🔧 新增：为大量结果创建子目录
        if (save_all_results_) {
            // 创建时间戳目录
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            char timestamp[100];
            std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&time_t));
            
            batch_result_dir_ = result_path_ + "loftr_batch_" + std::string(timestamp) + "/";
            match_results_dir_ = batch_result_dir_ + "matches/";
            
            createDirectory(batch_result_dir_);
            createDirectory(match_results_dir_);
            
            std::cout << "📁 批量结果目录: " << batch_result_dir_ << std::endl;
        }
        
        // 设置模型路径
        model_base_path_ = "/home/lin/loftrvins_ws/src/LoFTR_VINS/loftr_vins_estimator/weights/";
        onnx_model_path_ = model_base_path_ + "LoFTR_teacher.onnx";
        trt_model_path_ = model_base_path_ + "LoFTR_teacher.trt";
    }
        
    void runBenchmark() {
        std::cout << "\n🚀 开始大规模性能基准测试..." << std::endl;
        
        // 加载测试图像
        if (!loadTestImages()) {
            std::cout << "❌ 图像加载失败，无法继续测试" << std::endl;
            return;
        }

        // 检查测试图像是否存在
        if (!use_rosbag_ && !checkTestImages()) {
            std::cout << "❌ 测试图像不存在，无法继续测试" << std::endl;
            return;
        }
        
        // 固定分辨率测试
        std::vector<std::pair<int, int>> test_sizes = {
            {640, 480}
        };
        
        std::vector<LoFTR_Interface::BackendType> backends = {
            LoFTR_Interface::BackendType::ONNX_RUNTIME,
#ifdef TENSORRT_AVAILABLE
            LoFTR_Interface::BackendType::TENSORRT
#endif
        };
        
        // 性能统计
        std::vector<BenchmarkResult> results;
        
        for (auto backend : backends) {
            for (auto [width, height] : test_sizes) {
                BenchmarkResult result = runBatchBenchmark(backend, width, height);
                results.push_back(result);
            }
        }
        
        // 输出结果
        printBenchmarkResults(results);
        saveBenchmarkResults(results);
        
        // 生成统计报告
        generateDetailedReport(results);
    }

private:
    struct FrameResult {
        int frame_id;
        double inference_time;
        int num_matches;
        double match_quality;
        std::string image_filename;
        bool success;
    };

    struct BenchmarkResult {
        std::string backend_name;
        int width, height;
        bool success;
        double avg_inference_time;
        double min_inference_time;
        double max_inference_time;
        double fps;
        int num_matches;
        double match_quality;
        std::string error_message;
        
        // 🔧 新增：详细的帧级统计
        std::vector<FrameResult> frame_results;
        int total_frames_processed;
        int successful_frames;
        double success_rate;
        
        BenchmarkResult() : width(0), height(0), success(false), 
                           avg_inference_time(0), min_inference_time(0), 
                           max_inference_time(0), fps(0), num_matches(0), 
                           match_quality(0), total_frames_processed(0),
                           successful_frames(0), success_rate(0.0) {}
    };

    // 配置参数
    bool use_rosbag_;
    std::string rosbag_path_;
    std::string image_topic_;
    int max_test_frames_;  // 🔧 新增：最大测试帧数
    bool save_all_results_;  // 🔧 新增：是否保存所有结果

    // 目录管理
    std::string batch_result_dir_;  // 🔧 新增：批量结果目录
    std::string match_results_dir_;  // 🔧 新增：匹配结果目录

    // rosbag 相关
    bool use_stereo_;
    std::string left_topic_;
    std::string right_topic_;
    std::vector<cv::Mat> left_images_;
    std::vector<cv::Mat> right_images_;

    // 图像数据 - 🔧 修改：支持大量图像
    std::vector<std::pair<cv::Mat, cv::Mat>> test_image_pairs_;  // 图像对
    cv::Mat original_img1_;
    cv::Mat original_img2_;

    // 路径配置
    std::string dataset_path_;
    std::string result_path_;
    std::string img1_path_;
    std::string img2_path_;
    std::string model_base_path_;
    std::string onnx_model_path_;
    std::string trt_model_path_;

    // 🔧 新增：创建目录的辅助函数
    bool createDirectory(const std::string& path) {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            // 目录不存在，尝试创建
            #ifdef _WIN32
                return _mkdir(path.c_str()) == 0;
            #else
                return mkdir(path.c_str(), 0755) == 0;
            #endif
        } else if (info.st_mode & S_IFDIR) {
            // 目录已存在
            return true;
        }
        return false;
    }

    bool loadTestImages() {
        std::cout << "🔄 开始加载大量测试图像..." << std::endl;
        
        if (use_rosbag_) {
            return loadImagesFromRosbag();
        } else {
            return generateImagePairs();  // 🔧 生成多个测试图像对
        }
    }

    bool loadImagesFromRosbag() {
        std::cout << "📦 从 ROS bag 加载 " << max_test_frames_ << " 帧图像..." << std::endl;
        
        try {
            rosbag::Bag bag;
            bag.open(rosbag_path_, rosbag::bagmode::Read);
            
            if (use_stereo_) {
                // 双目模式：加载左右相机图像对
                std::vector<std::string> topics = {left_topic_, right_topic_};
                rosbag::View view(bag, rosbag::TopicQuery(topics));
                
                std::map<ros::Time, cv::Mat> left_map, right_map;
                
                BOOST_FOREACH(rosbag::MessageInstance const m, view) {
                    sensor_msgs::Image::ConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
                    if (img_msg != nullptr) {
                        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
                        
                        if (m.getTopic() == left_topic_) {
                            left_map[img_msg->header.stamp] = cv_ptr->image.clone();
                        } else if (m.getTopic() == right_topic_) {
                            right_map[img_msg->header.stamp] = cv_ptr->image.clone();
                        }
                        
                        if (left_map.size() >= max_test_frames_ && right_map.size() >= max_test_frames_) {
                            break;
                        }
                    }
                }
                
                // 时间戳对齐并创建图像对
                for (auto& left_pair : left_map) {
                    if (test_image_pairs_.size() >= max_test_frames_) break;
                    
                    auto right_it = right_map.find(left_pair.first);
                    if (right_it != right_map.end()) {
                        test_image_pairs_.push_back(std::make_pair(left_pair.second, right_it->second));
                    }
                }
                
            } else {
                // 单目模式：创建连续帧对
                rosbag::View view(bag, rosbag::TopicQuery(std::vector<std::string>{image_topic_}));
                
                std::vector<cv::Mat> frames;
                BOOST_FOREACH(rosbag::MessageInstance const m, view) {
                    if (frames.size() >= max_test_frames_ * 2) break;  // 需要足够的帧
                    
                    sensor_msgs::Image::ConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
                    if (img_msg != nullptr) {
                        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
                        frames.push_back(cv_ptr->image.clone());
                    }
                }
                
                // 创建连续帧对 (frame_i, frame_i+gap)
                int gap = std::max(1, static_cast<int>(frames.size()) / max_test_frames_);
                for (int i = 0; i < static_cast<int>(frames.size()) - gap && test_image_pairs_.size() < max_test_frames_; i += gap) {
                    test_image_pairs_.push_back(std::make_pair(frames[i], frames[i + gap]));
                }
            }
            
            bag.close();
            
        } catch (const std::exception& e) {
            std::cout << "❌ 加载 ROS bag 失败: " << e.what() << std::endl;
            return false;
        }
        
        if (test_image_pairs_.empty()) {
            std::cout << "❌ 未能从 bag 文件中加载图像对" << std::endl;
            return false;
        }
        
        // 设置参考图像
        original_img1_ = test_image_pairs_[0].first;
        original_img2_ = test_image_pairs_[0].second;
        
        std::cout << "✅ 成功加载 " << test_image_pairs_.size() << " 个图像对" << std::endl;
        return true;
    }

    bool generateImagePairs() {
        std::cout << "🎨 生成 " << max_test_frames_ << " 个测试图像对..." << std::endl;
        
        // 尝试加载原始图像
        cv::Mat base_img1 = cv::imread(img1_path_, cv::IMREAD_GRAYSCALE);
        cv::Mat base_img2 = cv::imread(img2_path_, cv::IMREAD_GRAYSCALE);
        
        // 如果无法加载，创建基础图像
        if (base_img1.empty()) {
            base_img1 = createRichTestImage(640, 480, 0);
        }
        if (base_img2.empty()) {
            base_img2 = createRichTestImage(640, 480, 1);
        }
        
        // 生成变化的图像对
        for (int i = 0; i < max_test_frames_; ++i) {
            cv::Mat img1_variant, img2_variant;
            
            // 应用不同的变换
            float rotation = (i % 10 - 5) * 2.0f;  // -10 到 +10 度旋转
            float scale = 1.0f + (i % 20 - 10) * 0.01f;  // 0.9 到 1.1 缩放
            int noise_level = (i % 5) * 10;  // 不同噪声级别
            
            // 对第一张图像应用变换
            cv::Point2f center(base_img1.cols/2.0f, base_img1.rows/2.0f);
            cv::Mat M = cv::getRotationMatrix2D(center, rotation, scale);
            cv::warpAffine(base_img1, img1_variant, M, base_img1.size());
            
            // 对第二张图像应用不同变换
            cv::Point2f center2(base_img2.cols/2.0f, base_img2.rows/2.0f);
            cv::Mat M2 = cv::getRotationMatrix2D(center2, rotation + 1.0f, scale * 1.01f);
            cv::warpAffine(base_img2, img2_variant, M2, base_img2.size());
            
            // 添加噪声
            if (noise_level > 0) {
                cv::Mat noise1, noise2;
                cv::randn(noise1, 0, noise_level);
                cv::randn(noise2, 0, noise_level);
                noise1.convertTo(noise1, CV_8U);
                noise2.convertTo(noise2, CV_8U);
                cv::addWeighted(img1_variant, 0.9, noise1, 0.1, 0, img1_variant);
                cv::addWeighted(img2_variant, 0.9, noise2, 0.1, 0, img2_variant);
            }
            
            test_image_pairs_.push_back(std::make_pair(img1_variant, img2_variant));
            
            if ((i + 1) % 100 == 0) {
                std::cout << "   生成进度: " << (i + 1) << "/" << max_test_frames_ << std::endl;
            }
        }
        
        original_img1_ = test_image_pairs_[0].first;
        original_img2_ = test_image_pairs_[0].second;
        
        std::cout << "✅ 成功生成 " << test_image_pairs_.size() << " 个测试图像对" << std::endl;
        return true;
    }

    BenchmarkResult runBatchBenchmark(LoFTR_Interface::BackendType backend, int width, int height) {
        BenchmarkResult result;
        result.backend_name = getBackendName(backend);
        result.width = width;
        result.height = height;
        result.success = false;
        
        std::cout << "\n📊 大规模测试: " << result.backend_name 
                  << " @ " << width << "x" << height 
                  << " (" << test_image_pairs_.size() << " 帧对)" << std::endl;
        
        try {
            // 配置 LoFTR
            LoFTR_Interface::Config config;
            config.backend = backend;
            config.input_width = width;
            config.input_height = height;
            config.match_threshold = 0.2f;
            config.max_matches = 500;
            
            if (backend == LoFTR_Interface::BackendType::TENSORRT) {
                config.model_path = onnx_model_path_;
                config.engine_path = trt_model_path_;
            } else {
                config.model_path = onnx_model_path_;
            }
            
            // 检查模型文件
            std::ifstream model_file(config.model_path);
            if (!model_file.good()) {
                result.error_message = "模型文件不存在: " + config.model_path;
                return result;
            }
            
            // 初始化 LoFTR
            LoFTR_Interface loftr;
            if (!loftr.initialize(config)) {
                result.error_message = "LoFTR 初始化失败";
                return result;
            }
            
            std::cout << "✅ LoFTR 初始化成功" << std::endl;
            
            // 预热
            std::cout << "🔥 预热中..." << std::endl;
            for (int i = 0; i < 3; ++i) {
                if (i < test_image_pairs_.size()) {
                    cv::Mat img1, img2;
                    cv::resize(test_image_pairs_[i].first, img1, cv::Size(width, height));
                    cv::resize(test_image_pairs_[i].second, img2, cv::Size(width, height));
                    loftr.match_images(img1, img2);
                }
            }
            
            // 🔧 批量处理所有图像对
            std::cout << "🚀 开始批量处理 " << test_image_pairs_.size() << " 个图像对..." << std::endl;
            
            std::vector<double> inference_times;
            std::vector<int> match_counts;
            std::vector<double> qualities;
            result.frame_results.clear();
            
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < static_cast<int>(test_image_pairs_.size()); ++i) {
                FrameResult frame_result;
                frame_result.frame_id = i;
                frame_result.success = false;
                
                // 准备图像
                cv::Mat img1, img2;
                cv::resize(test_image_pairs_[i].first, img1, cv::Size(width, height));
                cv::resize(test_image_pairs_[i].second, img2, cv::Size(width, height));
                
                // 执行匹配
                auto start = std::chrono::high_resolution_clock::now();
                LoFTR_Interface::MatchResult match_result = loftr.match_images(img1, img2);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration<double, std::milli>(end - start);
                
                frame_result.inference_time = duration.count();
                frame_result.num_matches = match_result.num_matches;
                
                if (match_result.num_matches > 0) {
                    inference_times.push_back(duration.count());
                    match_counts.push_back(match_result.num_matches);
                    
                    // 计算匹配质量
                    if (!match_result.confidence.empty()) {
                        double avg_confidence = std::accumulate(match_result.confidence.begin(), 
                                                              match_result.confidence.end(), 0.0) / match_result.confidence.size();
                        qualities.push_back(avg_confidence);
                        frame_result.match_quality = avg_confidence;
                    }
                    
                    frame_result.success = true;
                    result.successful_frames++;
                    
                    // 🔧 保存匹配结果图像（如果启用）
                    if (save_all_results_) {
                        std::string filename = saveFrameMatchVisualization(img1, img2, match_result, 
                                                                         backend, width, height, i);
                        frame_result.image_filename = filename;
                    }
                }
                
                result.frame_results.push_back(frame_result);
                result.total_frames_processed++;
                
                // 显示进度
                if ((i + 1) % 50 == 0 || i == test_image_pairs_.size() - 1) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration<double>(current_time - batch_start).count();
                    double progress = static_cast<double>(i + 1) / test_image_pairs_.size();
                    double eta = elapsed / progress - elapsed;
                    
                    std::cout << "   进度: " << (i + 1) << "/" << test_image_pairs_.size() 
                              << " (" << std::fixed << std::setprecision(1) << progress * 100 << "%)"
                              << " | 当前: " << std::setprecision(1) << duration.count() << "ms"
                              << " | 匹配: " << match_result.num_matches
                              << " | ETA: " << std::setprecision(0) << eta << "s" << std::endl;
                }
            }
            
            // 计算总体统计信息
            if (!inference_times.empty()) {
                result.avg_inference_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / inference_times.size();
                result.min_inference_time = *std::min_element(inference_times.begin(), inference_times.end());
                result.max_inference_time = *std::max_element(inference_times.begin(), inference_times.end());
                result.fps = 1000.0 / result.avg_inference_time;
                result.success_rate = static_cast<double>(result.successful_frames) / result.total_frames_processed;
                
                if (!match_counts.empty()) {
                    result.num_matches = std::accumulate(match_counts.begin(), match_counts.end(), 0) / match_counts.size();
                }
                
                if (!qualities.empty()) {
                    result.match_quality = std::accumulate(qualities.begin(), qualities.end(), 0.0) / qualities.size();
                }
                
                result.success = true;
                
                auto batch_end = std::chrono::high_resolution_clock::now();
                auto total_time = std::chrono::duration<double>(batch_end - batch_start).count();
                
                std::cout << "\n✅ 批量测试完成:" << std::endl;
                std::cout << "   总处理时间: " << std::fixed << std::setprecision(1) << total_time << "s" << std::endl;
                std::cout << "   成功率: " << std::setprecision(1) << result.success_rate * 100 << "%" << std::endl;
                std::cout << "   平均推理时间: " << std::setprecision(2) << result.avg_inference_time << "ms" << std::endl;
                std::cout << "   平均FPS: " << std::setprecision(1) << result.fps << std::endl;
                std::cout << "   平均匹配数: " << result.num_matches << std::endl;
            } else {
                result.error_message = "所有推理都失败了";
            }
            
        } catch (const std::exception& e) {
            result.error_message = std::string("异常: ") + e.what();
        }
        
        return result;
    }

    std::string saveFrameMatchVisualization(const cv::Mat& img1, const cv::Mat& img2, 
                                          const LoFTR_Interface::MatchResult& match_result,
                                          LoFTR_Interface::BackendType backend, 
                                          int width, int height, int frame_id) {
        try {
            // 创建匹配可视化图像
            cv::Mat match_img;
            cv::Mat img1_color, img2_color;
            cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
            cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
            cv::hconcat(img1_color, img2_color, match_img);
            
            // 绘制匹配点和连线
            for (int i = 0; i < match_result.num_matches && i < static_cast<int>(match_result.keypoints0.size()) && i < static_cast<int>(match_result.keypoints1.size()); ++i) {
                cv::Point2f pt1 = match_result.keypoints0[i];
                cv::Point2f pt2 = match_result.keypoints1[i];
                pt2.x += width;
                
                // 根据置信度选择颜色
                cv::Scalar color;
                if (i < static_cast<int>(match_result.confidence.size())) {
                    float conf = match_result.confidence[i];
                    if (conf > 0.8) color = cv::Scalar(0, 255, 0);      // 绿色
                    else if (conf > 0.5) color = cv::Scalar(0, 255, 255); // 黄色
                    else color = cv::Scalar(0, 0, 255);                   // 红色
                } else {
                    color = cv::Scalar(255, 0, 0); // 蓝色
                }
                
                cv::circle(match_img, pt1, 2, color, -1);
                cv::circle(match_img, pt2, 2, color, -1);
                cv::line(match_img, pt1, pt2, color, 1);
            }
            
            // 添加帧信息文本
            std::string info_text = "Frame " + std::to_string(frame_id) + " | " + 
                                   getBackendName(backend) + " | Matches: " + std::to_string(match_result.num_matches);
            cv::putText(match_img, info_text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            // 保存图像
            char filename[256];
            snprintf(filename, sizeof(filename), "frame_%04d_%s_%dx%d_matches_%d.jpg", 
                    frame_id, getBackendName(backend).c_str(), width, height, match_result.num_matches);
            
            std::string full_path = match_results_dir_ + std::string(filename);
            cv::imwrite(full_path, match_img);
            
            return std::string(filename);
            
        } catch (const std::exception& e) {
            std::cout << "   ⚠️ 保存帧 " << frame_id << " 可视化失败: " << e.what() << std::endl;
            return "";
        }
    }

    bool checkTestImages() {
        // 对于批量测试，如果是 rosbag 模式就跳过文件检查
        if (use_rosbag_) return true;
        
        std::cout << "🔍 检查测试图像..." << std::endl;
        std::ifstream file1(img1_path_);
        std::ifstream file2(img2_path_);
        
        if (!file1.good() || !file2.good()) {
            std::cout << "⚠️ 静态图像文件不存在，将生成测试图像" << std::endl;
        }
        
        return true;  // 总是返回 true，因为我们可以生成图像
    }
    
    void printBenchmarkResults(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n" << std::string(100, '=') << std::endl;
        std::cout << "                    LoFTR 大规模性能基准测试结果" << std::endl;
        std::cout << std::string(100, '=') << std::endl;
        
        // 表头
        std::cout << std::left << std::setw(12) << "Backend"
                  << std::setw(12) << "Resolution"  
                  << std::setw(10) << "Frames"
                  << std::setw(12) << "Success(%)"
                  << std::setw(12) << "Avg Time(ms)"
                  << std::setw(8) << "FPS"
                  << std::setw(10) << "Matches"
                  << std::setw(10) << "Quality"
                  << "Status" << std::endl;
        std::cout << std::string(100, '-') << std::endl;
        
        // 数据行
        for (const auto& result : results) {
            std::cout << std::left << std::setw(12) << result.backend_name
                      << std::setw(12) << (std::to_string(result.width) + "x" + std::to_string(result.height))
                      << std::setw(10) << result.total_frames_processed
                      << std::setw(12) << std::fixed << std::setprecision(1) << (result.success_rate * 100)
                      << std::setw(12) << std::fixed << std::setprecision(1) << result.avg_inference_time
                      << std::setw(8) << std::fixed << std::setprecision(1) << result.fps
                      << std::setw(10) << result.num_matches
                      << std::setw(10) << std::fixed << std::setprecision(3) << result.match_quality
                      << (result.success ? "✅" : "❌");
            
            if (!result.success && !result.error_message.empty()) {
                std::cout << " " << result.error_message;
            }
            std::cout << std::endl;
        }
        
        std::cout << std::string(100, '=') << std::endl;
        
        // 性能分析
        std::cout << "\n📈 性能分析:" << std::endl;
        
        for (const auto& result : results) {
            if (result.success) {
                std::cout << "\n🔍 " << result.backend_name << " 详细分析:" << std::endl;
                std::cout << "   总处理帧数: " << result.total_frames_processed << std::endl;
                std::cout << "   成功处理: " << result.successful_frames << " (" 
                          << std::fixed << std::setprecision(1) << result.success_rate * 100 << "%)" << std::endl;
                std::cout << "   时间范围: " << std::setprecision(1) << result.min_inference_time 
                          << " - " << result.max_inference_time << " ms" << std::endl;
                std::cout << "   吞吐量: " << std::setprecision(1) << result.fps << " FPS" << std::endl;
                
                // 计算性能分布统计
                if (!result.frame_results.empty()) {
                    std::vector<double> times;
                    std::vector<int> matches;
                    for (const auto& frame : result.frame_results) {
                        if (frame.success) {
                            times.push_back(frame.inference_time);
                            matches.push_back(frame.num_matches);
                        }
                    }
                    
                    if (!times.empty()) {
                        std::sort(times.begin(), times.end());
                        std::sort(matches.begin(), matches.end());
                        
                        double p50_time = times[times.size() * 0.5];
                        double p95_time = times[times.size() * 0.95];
                        int p50_matches = matches[matches.size() * 0.5];
                        int p95_matches = matches[matches.size() * 0.95];
                        
                        std::cout << "   时间中位数: " << std::setprecision(1) << p50_time << " ms" << std::endl;
                        std::cout << "   时间95%分位: " << std::setprecision(1) << p95_time << " ms" << std::endl;
                        std::cout << "   匹配数中位数: " << p50_matches << std::endl;
                        std::cout << "   匹配数95%分位: " << p95_matches << std::endl;
                    }
                }
            }
        }
        
        // 对比分析
        if (results.size() > 1) {
            std::cout << "\n⚖️  后端对比:" << std::endl;
            auto fastest = std::min_element(results.begin(), results.end(), 
                [](const BenchmarkResult& a, const BenchmarkResult& b) {
                    return a.success && (!b.success || a.avg_inference_time < b.avg_inference_time);
                });
            
            if (fastest != results.end() && fastest->success) {
                std::cout << "   🏆 最快后端: " << fastest->backend_name 
                          << " (" << std::fixed << std::setprecision(1) << fastest->avg_inference_time << "ms)" << std::endl;
                
                // 计算加速比
                for (const auto& result : results) {
                    if (result.success && result.backend_name != fastest->backend_name) {
                        double speedup = result.avg_inference_time / fastest->avg_inference_time;
                        std::cout << "   📊 相比 " << result.backend_name << ": " 
                                  << std::setprecision(2) << speedup << "x 加速" << std::endl;
                    }
                }
            }
        }
    }
    
    void saveBenchmarkResults(const std::vector<BenchmarkResult>& results) {
        std::string csv_path = (save_all_results_ ? batch_result_dir_ : result_path_) + "loftr_batch_results.csv";
        std::ofstream file(csv_path);
        
        if (!file.is_open()) {
            std::cout << "⚠️ 无法保存结果到文件: " << csv_path << std::endl;
            return;
        }
        
        // CSV 头
        file << "Backend,Width,Height,TotalFrames,SuccessfulFrames,SuccessRate,"
             << "AvgTime_ms,MinTime_ms,MaxTime_ms,P50Time_ms,P95Time_ms,"
             << "FPS,AvgMatches,P50Matches,P95Matches,AvgQuality\n";
        
        // 数据
        for (const auto& result : results) {
            if (result.success && !result.frame_results.empty()) {
                // 计算详细统计
                std::vector<double> times;
                std::vector<int> matches;
                std::vector<double> qualities;
                
                for (const auto& frame : result.frame_results) {
                    if (frame.success) {
                        times.push_back(frame.inference_time);
                        matches.push_back(frame.num_matches);
                        if (frame.match_quality > 0) {
                            qualities.push_back(frame.match_quality);
                        }
                    }
                }
                
                std::sort(times.begin(), times.end());
                std::sort(matches.begin(), matches.end());
                
                double p50_time = !times.empty() ? times[times.size() * 0.5] : 0;
                double p95_time = !times.empty() ? times[times.size() * 0.95] : 0;
                int p50_matches = !matches.empty() ? matches[matches.size() * 0.5] : 0;
                int p95_matches = !matches.empty() ? matches[matches.size() * 0.95] : 0;
                double avg_quality = !qualities.empty() ? 
                    std::accumulate(qualities.begin(), qualities.end(), 0.0) / qualities.size() : 0;
                
                file << result.backend_name << ","
                     << result.width << "," << result.height << ","
                     << result.total_frames_processed << ","
                     << result.successful_frames << ","
                     << result.success_rate << ","
                     << result.avg_inference_time << ","
                     << result.min_inference_time << ","
                     << result.max_inference_time << ","
                     << p50_time << "," << p95_time << ","
                     << result.fps << ","
                     << result.num_matches << ","
                     << p50_matches << "," << p95_matches << ","
                     << avg_quality << "\n";
            }
        }
        
        file.close();
        std::cout << "📊 批量结果已保存到: " << csv_path << std::endl;
        
        // 🔧 保存详细的帧级结果
        if (save_all_results_) {
            saveDetailedFrameResults(results);
        }
    }
    
    void saveDetailedFrameResults(const std::vector<BenchmarkResult>& results) {
        for (const auto& result : results) {
            if (result.success && !result.frame_results.empty()) {
                std::string frame_csv = batch_result_dir_ + "frames_" + result.backend_name + 
                                       "_" + std::to_string(result.width) + "x" + std::to_string(result.height) + ".csv";
                
                std::ofstream file(frame_csv);
                if (file.is_open()) {
                    // CSV 头
                    file << "FrameID,Success,InferenceTime_ms,NumMatches,MatchQuality,ImageFilename\n";
                    
                    // 帧数据
                    for (const auto& frame : result.frame_results) {
                        file << frame.frame_id << ","
                             << (frame.success ? "true" : "false") << ","
                             << frame.inference_time << ","
                             << frame.num_matches << ","
                             << frame.match_quality << ","
                             << "\"" << frame.image_filename << "\"\n";
                    }
                    
                    file.close();
                    std::cout << "📋 帧级详细结果已保存: " << frame_csv << std::endl;
                }
            }
        }
    }
    
    void generateDetailedReport(const std::vector<BenchmarkResult>& results) {
        if (!save_all_results_) return;
        
        std::string report_path = batch_result_dir_ + "detailed_report.md";
        std::ofstream report(report_path);
        
        if (!report.is_open()) {
            std::cout << "⚠️ 无法创建详细报告: " << report_path << std::endl;
            return;
        }
        
        // 生成 Markdown 报告
        report << "# LoFTR 大规模性能测试报告\n\n";
        
        // 测试配置
        report << "## 测试配置\n\n";
        report << "- **测试帧数**: " << max_test_frames_ << "\n";
        report << "- **数据源**: " << (use_rosbag_ ? ("ROS Bag: " + rosbag_path_) : "生成的测试图像") << "\n";
        report << "- **图像话题**: " << image_topic_ << "\n";
        report << "- **测试模式**: " << (use_stereo_ ? "双目立体匹配" : "时序匹配") << "\n\n";
        
        // 性能总结
        report << "## 性能总结\n\n";
        report << "| 后端 | 分辨率 | 总帧数 | 成功率(%) | 平均时间(ms) | FPS | 平均匹配数 |\n";
        report << "|------|--------|--------|-----------|-------------|-----|----------|\n";
        
        for (const auto& result : results) {
            if (result.success) {
                report << "| " << result.backend_name 
                       << " | " << result.width << "x" << result.height
                       << " | " << result.total_frames_processed
                       << " | " << std::fixed << std::setprecision(1) << (result.success_rate * 100)
                       << " | " << std::setprecision(1) << result.avg_inference_time
                       << " | " << std::setprecision(1) << result.fps
                       << " | " << result.num_matches << " |\n";
            }
        }
        report << "\n";
        
        // 详细分析
        for (const auto& result : results) {
            if (result.success) {
                report << "### " << result.backend_name << " 详细分析\n\n";
                report << "- **处理成功率**: " << std::fixed << std::setprecision(2) 
                       << (result.success_rate * 100) << "%\n";
                report << "- **时间统计**: " << std::setprecision(1) << result.min_inference_time 
                       << "ms (最小) / " << result.avg_inference_time << "ms (平均) / " 
                       << result.max_inference_time << "ms (最大)\n";
                report << "- **实时性能**: " << std::setprecision(1) << result.fps << " FPS\n";
                report << "- **匹配质量**: 平均 " << result.num_matches << " 个匹配点，置信度 " 
                       << std::setprecision(3) << result.match_quality << "\n\n";
                
                // 性能分布直方图数据
                if (!result.frame_results.empty()) {
                    report << "#### 时间分布统计\n\n";
                    std::vector<double> times;
                    for (const auto& frame : result.frame_results) {
                        if (frame.success) times.push_back(frame.inference_time);
                    }
                    
                    if (!times.empty()) {
                        std::sort(times.begin(), times.end());
                        report << "- **P50**: " << std::setprecision(1) << times[times.size() * 0.5] << "ms\n";
                        report << "- **P90**: " << std::setprecision(1) << times[times.size() * 0.9] << "ms\n";
                        report << "- **P95**: " << std::setprecision(1) << times[times.size() * 0.95] << "ms\n";
                        report << "- **P99**: " << std::setprecision(1) << times[times.size() * 0.99] << "ms\n\n";
                    }
                }
            }
        }
        
        // 文件说明
        report << "## 输出文件说明\n\n";
        report << "- `loftr_batch_results.csv`: 汇总性能数据\n";
        report << "- `frames_*.csv`: 每帧详细结果\n";
        report << "- `matches/`: 所有帧的匹配可视化图像\n";
        report << "- `detailed_report.md`: 本报告文件\n\n";
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        report << "---\n*报告生成时间: " << std::ctime(&time_t) << "*\n";
        
        report.close();
        std::cout << "📄 详细报告已生成: " << report_path << std::endl;
    }
    
    std::string getBackendName(LoFTR_Interface::BackendType backend) {
        switch (backend) {
            case LoFTR_Interface::BackendType::ONNX_RUNTIME:
                return "ONNX";
            case LoFTR_Interface::BackendType::TENSORRT:
                return "TensorRT";
            case LoFTR_Interface::BackendType::AUTO:
                return "AUTO";
            default:
                return "Unknown";
        }
    }
    
    cv::Mat createRichTestImage(int width, int height, int variant) {
        cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);
        
        // 添加结构化特征
        int block_size = 32;
        for (int y = 0; y < height; y += block_size) {
            for (int x = 0; x < width; x += block_size) {
                if ((x/block_size + y/block_size) % 2 == variant) {
                    cv::rectangle(img, cv::Point(x, y), cv::Point(x+block_size, y+block_size), 
                                 cv::Scalar(120 + variant * 30), -1);
                }
            }
        }
        
        // 添加圆形特征
        for (int i = 0; i < 15; ++i) {
            cv::Point center(50 + i * 40, 50 + (i % 3) * 100);
            if (center.x < width && center.y < height) {
                cv::circle(img, center, 20 + variant * 5, cv::Scalar(200 + variant * 20), 2);
            }
        }
        
        // 添加线条特征
        for (int i = 0; i < 10; ++i) {
            cv::Point pt1(i * 60, 0);
            cv::Point pt2(i * 60, height);
            cv::line(img, pt1, pt2, cv::Scalar(80 + variant * 10), 1);
        }
        
        // 添加噪声纹理
        cv::Mat noise = cv::Mat::zeros(height, width, CV_8UC1);
        cv::randu(noise, 0, 30);
        cv::addWeighted(img, 0.9, noise, 0.1, 0, img);
        
        // 为第二张图像添加变换
        if (variant == 1) {
            cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(width/2, height/2), 2.0, 1.0);
            cv::warpAffine(img, img, M, cv::Size(width, height));
        }
        
        return img;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "loftr_batch_benchmark");
    ros::NodeHandle nh;
    
    // 🔧 增强的参数解析
    bool use_rosbag = false;
    std::string rosbag_path = "";
    std::string image_topic = "/cam0/image_raw";
    int max_frames = 1000;
    bool save_all_results = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rosbag" && i + 1 < argc) {
            use_rosbag = true;
            rosbag_path = argv[++i];
        } else if (arg == "--topic" && i + 1 < argc) {
            image_topic = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            max_frames = std::atoi(argv[++i]);
        } else if (arg == "--save-all") {
            save_all_results = true;
        } else if (arg == "--stereo") {
            image_topic = "stereo";
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "LoFTR VINS 大规模性能测试工具\n\n";
            std::cout << "用法:\n";
            std::cout << "  " << argv[0] << " [选项]\n\n";
            std::cout << "选项:\n";
            std::cout << "  --rosbag <path>     使用 ROS bag 文件作为数据源\n";
            std::cout << "  --topic <name>      指定图像话题 (默认: /cam0/image_raw)\n";
            std::cout << "  --stereo            使用双目立体匹配模式\n";
            std::cout << "  --frames <number>   测试帧数 (默认: 1000)\n";
            std::cout << "  --save-all          保存所有帧的匹配结果图像\n";
            std::cout << "  --help, -h          显示此帮助信息\n\n";
            std::cout << "示例:\n";
            std::cout << "  " << argv[0] << " --frames 500\n";
            std::cout << "  " << argv[0] << " --rosbag dataset/MH_01_easy.bag --frames 1000 --save-all\n";
            std::cout << "  " << argv[0] << " --rosbag dataset/V1_01_easy.bag --stereo --frames 2000\n\n";
            std::cout << "输出:\n";
            std::cout << "  结果将保存到 ~/loftrvins_ws/result/loftr_batch_[timestamp]/ 目录\n";
            return 0;
        }
    }
    
    // 参数验证
    if (max_frames <= 0 || max_frames > 10000) {
        std::cout << "❌ 无效的帧数: " << max_frames << " (范围: 1-10000)" << std::endl;
        return 1;
    }
    
    std::cout << "✅ ROS 初始化成功" << std::endl;
    srand(static_cast<unsigned int>(time(nullptr)));
    
    // 创建并运行基准测试
    LoFTRBenchmark benchmark(use_rosbag, rosbag_path, image_topic, max_frames, save_all_results);
    benchmark.runBenchmark();
    
    std::cout << "\n=== 大规模性能测试完成 ===" << std::endl;
    if (save_all_results) {
        std::cout << "📁 所有结果文件和图像已保存到批量结果目录" << std::endl;
        std::cout << "   - 汇总报告: loftr_batch_results.csv" << std::endl;
        std::cout << "   - 帧级详情: frames_*.csv" << std::endl;
        std::cout << "   - 匹配图像: matches/ 目录" << std::endl;
        std::cout << "   - 详细报告: detailed_report.md" << std::endl;
    } else {
        std::cout << "📊 性能统计已保存到 ~/loftrvins_ws/result/" << std::endl;
    }
    
    return 0;
}