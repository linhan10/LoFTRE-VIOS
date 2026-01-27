#include "loftr_utils.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>
#include <future>
#include <sstream>
#include <iomanip>

namespace LoFTR_Utils {

// ==================== 2. 结构体方法实现 ====================

std::string PerformanceReport::toString() const {
    std::stringstream ss;
    ss << "=== LoFTR Performance Report ===" << std::endl;
    ss << "Total frames: " << total_frames << std::endl;
    ss << "Average time: " << std::fixed << std::setprecision(3) << average_time_ms << " ms" << std::endl;
    ss << "Min time: " << min_time_ms << " ms" << std::endl;
    ss << "Max time: " << max_time_ms << " ms" << std::endl;
    ss << "Average FPS: " << std::setprecision(1) << average_fps << std::endl;
    if (backend_inference_time_ms > 0) {
        ss << "Backend inference: " << std::setprecision(3) << backend_inference_time_ms << " ms" << std::endl;
    }
    if (total_matches > 0) {
        ss << "Total matches: " << total_matches << std::endl;
        ss << "Average confidence: " << std::setprecision(3) << average_confidence << std::endl;
    }
    return ss.str();
}

void PerformanceReport::reset() {
    total_frames = 0;
    average_time_ms = 0.0;
    min_time_ms = 0.0;
    max_time_ms = 0.0;
    average_fps = 0.0;
    backend_inference_time_ms = 0.0;
    total_matches = 0;
    average_confidence = 0.0;
}

// ==================== 3. 图像预处理函数实现 ====================

cv::Mat preprocessImage(const cv::Mat& img, const PreprocessConfig& config) {
    cv::Mat processed = img.clone();
    
    // 3.1 转换为灰度图（如果需要）
    if (processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2GRAY);
    }
    
    // 3.2 调整尺寸
    if (config.maintain_aspect_ratio) {
        auto [resized, scale] = resizeKeepAspectRatio(processed, 
            cv::Size(config.target_width, config.target_height));
        processed = resized;
        
        // 如果尺寸不匹配，进行中心裁剪
        if (processed.size() != cv::Size(config.target_width, config.target_height)) {
            processed = centerCrop(processed, cv::Size(config.target_width, config.target_height));
        }
    } else {
        cv::resize(processed, processed, cv::Size(config.target_width, config.target_height));
    }
    
    // 3.3 转换为浮点型
    if (processed.type() != CV_32F) {
        processed.convertTo(processed, CV_32F);
    }
    
    // 3.4 归一化
    if (config.normalize) {
        processed /= 255.0f;
    }
    
    return processed;
}

std::pair<cv::Mat, float> resizeKeepAspectRatio(const cv::Mat& img, cv::Size target_size) {
    float scale_x = static_cast<float>(target_size.width) / img.cols;
    float scale_y = static_cast<float>(target_size.height) / img.rows;
    float scale = std::min(scale_x, scale_y);
    
    int new_width = static_cast<int>(img.cols * scale);
    int new_height = static_cast<int>(img.rows * scale);
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_width, new_height));
    
    return {resized, scale};
}

cv::Mat centerCrop(const cv::Mat& img, cv::Size target_size) {
    int x = (img.cols - target_size.width) / 2;
    int y = (img.rows - target_size.height) / 2;
    
    x = std::max(0, x);
    y = std::max(0, y);
    
    int crop_width = std::min(target_size.width, img.cols - x);
    int crop_height = std::min(target_size.height, img.rows - y);
    
    cv::Rect crop_rect(x, y, crop_width, crop_height);
    cv::Mat cropped = img(crop_rect);
    
    // 如果裁剪后尺寸仍不匹配，进行填充
    if (cropped.size() != target_size) {
        cv::Mat padded = cv::Mat::zeros(target_size, cropped.type());
        int pad_x = (target_size.width - cropped.cols) / 2;
        int pad_y = (target_size.height - cropped.rows) / 2;
        
        cv::Rect roi(pad_x, pad_y, cropped.cols, cropped.rows);
        cropped.copyTo(padded(roi));
        return padded;
    }
    
    return cropped;
}

cv::Mat normalizeImage(const cv::Mat& img, cv::Scalar mean, cv::Scalar std) {
    cv::Mat normalized;
    img.convertTo(normalized, CV_32F);
    
    if (img.channels() == 3) {
        // 多通道归一化
        std::vector<cv::Mat> channels;
        cv::split(normalized, channels);
        
        for (int i = 0; i < 3; ++i) {
            channels[i] = (channels[i] - mean[i]) / std[i];
        }
        
        cv::merge(channels, normalized);
    } else {
        // 单通道归一化
        normalized = (normalized - mean[0]) / std[0];
    }
    
    return normalized;
}

// ==================== 4. 匹配结果处理函数实现 ====================

MatchResult extractMatches(const std::vector<float>& conf_matrix,
                          cv::Size img0_size, cv::Size img1_size,
                          cv::Size network_size,
                          float match_threshold,
                          int max_matches) {
    MatchResult result;

    if (conf_matrix.empty()) {
        std::cerr << "[LoFTR_Utils] 置信度矩阵为空" << std::endl;
        return result;
    }

    int resolution = 8;  // 或16，根据你的模型
    int feature_h = network_size.height / resolution;
    int feature_w = network_size.width / resolution;
    int feature_num = feature_h * feature_w;

    if (static_cast<int>(conf_matrix.size()) != feature_num * feature_num) {
        std::cerr << "[LoFTR_Utils] 置信度矩阵尺寸不匹配" << std::endl;
        return result;
    }

    // 🔧 关键修改：使用不同的匹配策略
    
    // 1. 不进行全局归一化，使用原始置信度
    // 2. 对每一行找最佳匹配（而不是全局搜索）
    std::vector<std::tuple<int, int, float>> matches;
    
    for (int i = 0; i < feature_num; ++i) {
        // 为第i个img0特征点找最佳img1匹配
        float max_conf_in_row = 0.0f;
        int best_j = -1;
        
        for (int j = 0; j < feature_num; ++j) {
            int idx = i * feature_num + j;
            float confidence = conf_matrix[idx];
            
            if (confidence > max_conf_in_row) {
                max_conf_in_row = confidence;
                best_j = j;
            }
        }
        
        // 🔧 关键：排除自匹配，确保有真正的对应关系
        if (best_j != -1 && max_conf_in_row > match_threshold && best_j != i) {
            matches.emplace_back(i, best_j, max_conf_in_row);
        }
    }

    std::cout << "[LoFTR_Utils] 找到 " << matches.size() << " 个非对角线匹配" << std::endl;

    if (matches.empty()) {
        // 🔧 如果没有非对角线匹配，降低阈值重试
        std::cout << "[LoFTR_Utils] 没有非对角线匹配，降低阈值重试..." << std::endl;
        
        float lower_threshold = match_threshold * 0.1f; // 降低到1/10
        for (int i = 0; i < feature_num; ++i) {
            for (int j = 0; j < feature_num; ++j) {
                if (i == j) continue; // 跳过对角线
                
                int idx = i * feature_num + j;
                float confidence = conf_matrix[idx];
                
                if (confidence > lower_threshold) {
                    matches.emplace_back(i, j, confidence);
                }
            }
        }
        
        std::cout << "[LoFTR_Utils] 降低阈值后找到 " << matches.size() << " 个匹配" << std::endl;
    }

    // 按置信度排序
    std::sort(matches.begin(), matches.end(),
             [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });

    // 应用NMS和坐标转换（你的原始逻辑是正确的）
    std::vector<bool> used_i(feature_num, false);
    std::vector<bool> used_j(feature_num, false);
    int valid_matches = 0;

    float scale_x0 = static_cast<float>(img0_size.width) / network_size.width;
    float scale_y0 = static_cast<float>(img0_size.height) / network_size.height;
    float scale_x1 = static_cast<float>(img1_size.width) / network_size.width;
    float scale_y1 = static_cast<float>(img1_size.height) / network_size.height;

    for (const auto& match : matches) {
        if (valid_matches >= max_matches) break;

        int i = std::get<0>(match);
        int j = std::get<1>(match);
        float confidence = std::get<2>(match);

        if (used_i[i] || used_j[j]) continue;

        // 🔧 添加调试：检查i和j是否不同
        if (i == j) {
            std::cout << "[LoFTR_Utils] 警告：仍然发现对角线匹配 i=" << i << std::endl;
            continue;
        }

        // 坐标转换（你的逻辑是正确的）
        int grid_x0 = i % feature_w;
        int grid_y0 = i / feature_w;
        float network_x0 = grid_x0 * resolution + resolution / 2.0f;
        float network_y0 = grid_y0 * resolution + resolution / 2.0f;

        int grid_x1 = j % feature_w;
        int grid_y1 = j / feature_w;
        float network_x1 = grid_x1 * resolution + resolution / 2.0f;
        float network_y1 = grid_y1 * resolution + resolution / 2.0f;

        float orig_x0 = network_x0 * scale_x0;
        float orig_y0 = network_y0 * scale_y0;
        float orig_x1 = network_x1 * scale_x1;
        float orig_y1 = network_y1 * scale_y1;

        // 🔧 添加调试输出
        if (valid_matches < 5) {
            std::cout << "[LoFTR_Utils] Valid Match " << valid_matches 
                      << ": i=" << i << "(grid:" << grid_x0 << "," << grid_y0 << ")" 
                      << " -> j=" << j << "(grid:" << grid_x1 << "," << grid_y1 << ")"
                      << ", img0(" << orig_x0 << "," << orig_y0 << ")"
                      << " -> img1(" << orig_x1 << "," << orig_y1 << ")"
                      << ", conf=" << confidence << std::endl;
        }

        if (orig_x0 >= 0 && orig_x0 < img0_size.width &&
            orig_y0 >= 0 && orig_y0 < img0_size.height &&
            orig_x1 >= 0 && orig_x1 < img1_size.width &&
            orig_y1 >= 0 && orig_y1 < img1_size.height) {

            result.keypoints0.emplace_back(orig_x0, orig_y0);
            result.keypoints1.emplace_back(orig_x1, orig_y1);
            result.confidence.push_back(confidence);

            used_i[i] = true;
            used_j[j] = true;
            valid_matches++;
        }
    }

    result.num_matches = result.keypoints0.size();
    return result;
}


MatchResult filterMatches(const MatchResult& matches,
                         float min_confidence,
                         float max_distance) {
    MatchResult filtered;
    
    for (size_t i = 0; i < matches.keypoints0.size(); ++i) {
        // 检查置信度
        if (!matches.confidence.empty() && matches.confidence[i] < min_confidence) {
            continue;
        }
        
        // 检查距离
        cv::Point2f p0 = matches.keypoints0[i];
        cv::Point2f p1 = matches.keypoints1[i];
        float distance = cv::norm(p1 - p0);
        
        if (distance > max_distance) {
            continue;
        }
        
        // 添加到过滤结果
        filtered.keypoints0.push_back(p0);
        filtered.keypoints1.push_back(p1);
        if (!matches.confidence.empty()) {
            filtered.confidence.push_back(matches.confidence[i]);
        }
    }
    
    filtered.num_matches = filtered.keypoints0.size();
    filtered.inference_time_ms = matches.inference_time_ms;
    
    return filtered;
}

MatchResult ransacFilter(const MatchResult& matches,
                        double threshold,
                        double confidence) {
    MatchResult filtered = matches; // 复制原始结果
    
    if (matches.num_matches < 4) {
        return filtered; // RANSAC 需要至少4个点
    }
    
    try {
        // 使用 OpenCV 的 findHomography 进行 RANSAC
        cv::Mat mask;
        cv::Mat H = cv::findHomography(matches.keypoints0, matches.keypoints1,
                                      cv::RANSAC, threshold, mask, 2000, confidence);
        
        if (H.empty()) {
            return filtered;
        }
        
        // 根据 mask 过滤结果
        MatchResult ransac_filtered;
        for (int i = 0; i < mask.rows; ++i) {
            if (mask.at<uchar>(i) > 0) {
                ransac_filtered.keypoints0.push_back(matches.keypoints0[i]);
                ransac_filtered.keypoints1.push_back(matches.keypoints1[i]);
                if (!matches.confidence.empty()) {
                    ransac_filtered.confidence.push_back(matches.confidence[i]);
                }
            }
        }
        
        ransac_filtered.num_matches = ransac_filtered.keypoints0.size();
        ransac_filtered.inference_time_ms = matches.inference_time_ms;
        
        return ransac_filtered;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[LoFTR_Utils] RANSAC 过滤失败: " << e.what() << std::endl;
        return filtered;
    }
}

MatchResult convertFromVINSFormat(const std::vector<cv::Point2f>& prev_pts,
                                 const std::vector<cv::Point2f>& cur_pts,
                                 const std::vector<float>& confidence) {
    MatchResult result;
    
    result.keypoints0 = prev_pts;
    result.keypoints1 = cur_pts;
    result.confidence = confidence;
    result.num_matches = std::min(prev_pts.size(), cur_pts.size());
    
    return result;
}

// ==================== 坐标转换函数实现 ====================

std::vector<cv::Point2f> rescalePoints(const std::vector<cv::Point2f>& points,
                                      cv::Size network_size,
                                      cv::Size original_size) {
    std::vector<cv::Point2f> rescaled_points;
    rescaled_points.reserve(points.size());
    
    float scale_x = static_cast<float>(original_size.width) / network_size.width;
    float scale_y = static_cast<float>(original_size.height) / network_size.height;
    
    for (const auto& pt : points) {
        rescaled_points.emplace_back(pt.x * scale_x, pt.y * scale_y);
    }
    
    return rescaled_points;
}

cv::Point2f featureIndexToImageCoord(int feature_idx, int feature_width, int resolution) {
    int x = feature_idx % feature_width;
    int y = feature_idx / feature_width;
    return cv::Point2f(x * resolution, y * resolution);
}

// ==================== 可视化函数实现 ====================

cv::Mat visualizeMatches(const cv::Mat& img0, const cv::Mat& img1,
                        const MatchResult& matches,
                        const VisualizationConfig& config) {
    // 创建并排显示的图像
    cv::Mat vis_img;
    if (!img1.empty()) {
        cv::hconcat(img0, img1, vis_img);
    } else {
        vis_img = img0.clone();
    }
    
    // 转换为彩色图像（如果需要）
    if (vis_img.channels() == 1) {
        cv::cvtColor(vis_img, vis_img, cv::COLOR_GRAY2BGR);
    }
    
    int img0_width = img0.cols;
    
    // 绘制关键点和匹配线
    for (size_t i = 0; i < matches.keypoints0.size(); ++i) {
        cv::Point2f pt0 = matches.keypoints0[i];
        cv::Point2f pt1 = matches.keypoints1[i];
        
        if (!img1.empty()) {
            pt1.x += img0_width; // 调整右图的 x 坐标
        }
        
        // 根据置信度确定颜色
        cv::Scalar color;
        if (!matches.confidence.empty()) {
            float conf = matches.confidence[i];
            if (conf >= config.confidence_threshold) {
                color = config.good_match_color;
            } else {
                color = config.bad_match_color;
            }
            
            // 根据置信度调整颜色强度
            color *= conf;
        } else {
            color = config.good_match_color;
        }
        
        // 绘制关键点
        if (config.show_keypoints) {
            cv::circle(vis_img, pt0, config.keypoint_radius, color, -1);
            if (!img1.empty()) {
                cv::circle(vis_img, pt1, config.keypoint_radius, color, -1);
            }
        }
        
        // 绘制连接线
        if (config.show_matches && !img1.empty()) {
            cv::line(vis_img, pt0, pt1, color, config.line_thickness);
        }
        
        // 显示置信度
        if (config.show_confidence && !matches.confidence.empty()) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << matches.confidence[i];
            cv::putText(vis_img, ss.str(), pt0, cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }
    }
    
    // 添加信息文本
    std::string info = "Matches: " + std::to_string(matches.num_matches);
    if (matches.inference_time_ms > 0) {
        info += " | Time: " + std::to_string(static_cast<int>(matches.inference_time_ms)) + "ms";
    }
    cv::putText(vis_img, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
               cv::Scalar(255, 255, 255), 2);
    
    return vis_img;
}

cv::Mat visualizeMatchesWithTrajectory(const cv::Mat& img0, const cv::Mat& img1,
                                      const MatchResult& matches,
                                      const std::vector<cv::Point2f>& prev_points,
                                      const std::vector<int>& track_counts,
                                      const VisualizationConfig& config) {
    cv::Mat vis_img = visualizeMatches(img0, img1, matches, config);
    
    // 添加轨迹信息（VINS特有功能）
    for (size_t i = 0; i < matches.keypoints0.size() && i < track_counts.size(); ++i) {
        cv::Point2f current_pt = matches.keypoints0[i];
        
        // 根据跟踪长度调整颜色
        double track_ratio = std::min(1.0, static_cast<double>(track_counts[i]) / 20.0);
        cv::Scalar track_color(255 * (1 - track_ratio), 0, 255 * track_ratio);
        
        // 绘制增强的关键点
        cv::circle(vis_img, current_pt, config.keypoint_radius + 1, track_color, 2);
        
        // 如果有前一帧的点，绘制轨迹
        if (i < prev_points.size() && config.show_trajectory) {
            cv::arrowedLine(vis_img, current_pt, prev_points[i], 
                           config.trajectory_color, 1, 8, 0, 0.2);
        }
    }
    
    return vis_img;
}

cv::Mat drawKeypoints(const cv::Mat& img,
                     const std::vector<cv::Point2f>& keypoints,
                     cv::Scalar color,
                     int radius) {
    cv::Mat result = img.clone();
    
    // 转换为彩色图像（如果需要）
    if (result.channels() == 1) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    
    for (const auto& pt : keypoints) {
        cv::circle(result, pt, radius, color, -1);
    }
    
    return result;
}

cv::Mat drawMatches(const cv::Mat& img_pair,
                   const std::vector<cv::Point2f>& keypoints0,
                   const std::vector<cv::Point2f>& keypoints1,
                   const std::vector<float>& confidence,
                   const VisualizationConfig& config) {
    cv::Mat result = img_pair.clone();
    
    // 转换为彩色图像（如果需要）
    if (result.channels() == 1) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    
    int img0_width = result.cols / 2; // 假设是并排图像
    
    for (size_t i = 0; i < keypoints0.size() && i < keypoints1.size(); ++i) {
        cv::Point2f pt0 = keypoints0[i];
        cv::Point2f pt1 = keypoints1[i];
        pt1.x += img0_width;
        
        cv::Scalar color = config.good_match_color;
        if (!confidence.empty() && i < confidence.size()) {
            if (confidence[i] < config.confidence_threshold) {
                color = config.bad_match_color;
            }
        }
        
        cv::line(result, pt0, pt1, color, config.line_thickness);
        cv::circle(result, pt0, config.keypoint_radius, color, -1);
        cv::circle(result, pt1, config.keypoint_radius, color, -1);
    }
    
    return result;
}

// ==================== 质量评估函数实现 ====================

double computeMatchQuality(const MatchResult& matches) {
    if (matches.confidence.empty()) {
        return 0.0;
    }
    
    double sum = std::accumulate(matches.confidence.begin(), matches.confidence.end(), 0.0);
    return sum / matches.confidence.size();
}

ReprojectionError computeReprojectionError(const MatchResult& matches, const cv::Mat& H) {
    ReprojectionError error_stats;
    
    if (H.empty() || matches.num_matches == 0) {
        return error_stats;
    }
    
    std::vector<double> errors;
    errors.reserve(matches.num_matches);
    
    for (size_t i = 0; i < matches.keypoints0.size(); ++i) {
        // 使用单应性矩阵变换点
        cv::Point2f pt0 = matches.keypoints0[i];
        std::vector<cv::Point2f> src_pts = {pt0};
        std::vector<cv::Point2f> dst_pts;
        
        cv::perspectiveTransform(src_pts, dst_pts, H);
        
        // 计算重投影误差
        cv::Point2f predicted = dst_pts[0];
        cv::Point2f actual = matches.keypoints1[i];
        double error = cv::norm(predicted - actual);
        
        errors.push_back(error);
    }
    
    if (!errors.empty()) {
        error_stats.mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
        error_stats.max_error = *std::max_element(errors.begin(), errors.end());
        
        // 计算标准差
        double variance = 0.0;
        for (double error : errors) {
            variance += (error - error_stats.mean_error) * (error - error_stats.mean_error);
        }
        error_stats.std_error = std::sqrt(variance / errors.size());
        error_stats.errors = errors;
    }
    
    return error_stats;
}

PerformanceReport generatePerformanceReport(const std::map<std::string, double>& stats) {
    PerformanceReport report;
    
    auto find_stat = [&](const std::string& key) -> double {
        auto it = stats.find(key);
        return (it != stats.end()) ? it->second : 0.0;
    };
    
    report.total_frames = static_cast<size_t>(find_stat("total_frames"));
    report.average_time_ms = find_stat("average_time_ms");
    report.min_time_ms = find_stat("min_time_ms");
    report.max_time_ms = find_stat("max_time_ms");
    report.average_fps = find_stat("average_fps");
    report.backend_inference_time_ms = find_stat("backend_inference_time_ms");
    report.total_matches = static_cast<size_t>(find_stat("total_matches"));
    report.average_confidence = find_stat("average_confidence");
    
    return report;
}

// ==================== 文件 I/O 函数实现 ====================

bool saveMatches(const MatchResult& matches,
                const std::string& filename,
                const std::string& format) {
    if (format == "txt") {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[LoFTR_Utils] 无法打开文件: " << filename << std::endl;
            return false;
        }
        
        file << "# LoFTR Match Results\n";
        file << "# Format: x0 y0 x1 y1 confidence\n";
        file << "# Matches: " << matches.num_matches << "\n";
        file << "# Inference time: " << matches.inference_time_ms << " ms\n";
        
        for (size_t i = 0; i < matches.keypoints0.size(); ++i) {
            file << matches.keypoints0[i].x << " " << matches.keypoints0[i].y << " "
                 << matches.keypoints1[i].x << " " << matches.keypoints1[i].y << " ";
            
            if (!matches.confidence.empty()) {
                file << matches.confidence[i];
            } else {
                file << "1.0";
            }
            file << "\n";
        }
        
        return true;
    }
    
    std::cerr << "[LoFTR_Utils] 不支持的文件格式: " << format << std::endl;
    return false;
}

MatchResult loadMatches(const std::string& filename) {
    MatchResult result;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "[LoFTR_Utils] 无法打开文件: " << filename << std::endl;
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

bool savePerformanceStats(const std::map<std::string, double>& stats,
                         const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[LoFTR_Utils] 无法保存性能统计: " << filename << std::endl;
        return false;
    }
    
    file << "# LoFTR Performance Statistics\n";
    for (const auto& [key, value] : stats) {
        file << key << ": " << value << "\n";
    }
    
    return true;
}

bool savePerformanceReport(const PerformanceReport& report,
                          const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << report.toString();
    return true;
}

// ==================== 调试和诊断函数实现 ====================

void printMatchStats(const MatchResult& matches) {
    std::cout << "=== LoFTR Match Statistics ===" << std::endl;
    std::cout << "Total matches: " << matches.num_matches << std::endl;
    std::cout << "Inference time: " << matches.inference_time_ms << " ms" << std::endl;
    
    if (!matches.confidence.empty()) {
        double avg_conf = computeMatchQuality(matches);
        double min_conf = *std::min_element(matches.confidence.begin(), matches.confidence.end());
        double max_conf = *std::max_element(matches.confidence.begin(), matches.confidence.end());
        
        std::cout << "Average confidence: " << avg_conf << std::endl;
        std::cout << "Min confidence: " << min_conf << std::endl;
        std::cout << "Max confidence: " << max_conf << std::endl;
    }
}

bool validateMatches(const MatchResult& matches,
                    cv::Size img0_size,
                    cv::Size img1_size) {
    // 检查基本一致性
    if (matches.keypoints0.size() != matches.keypoints1.size()) {
        std::cerr << "[LoFTR_Utils] 关键点数量不匹配" << std::endl;
        return false;
    }
    
    if (!matches.confidence.empty() && 
        matches.confidence.size() != matches.keypoints0.size()) {
        std::cerr << "[LoFTR_Utils] 置信度数量不匹配" << std::endl;
        return false;
    }
    
    // 检查坐标范围
    for (const auto& pt : matches.keypoints0) {
        if (pt.x < 0 || pt.x >= img0_size.width || 
            pt.y < 0 || pt.y >= img0_size.height) {
            std::cerr << "[LoFTR_Utils] 图像0坐标超出范围" << std::endl;
            return false;
        }
    }
    
    for (const auto& pt : matches.keypoints1) {
        if (pt.x < 0 || pt.x >= img1_size.width || 
            pt.y < 0 || pt.y >= img1_size.height) {
            std::cerr << "[LoFTR_Utils] 图像1坐标超出范围" << std::endl;
            return false;
        }
    }
    
    return true;
}

std::string generateMatchReport(const MatchResult& matches,
                               const std::string& img0_path,
                               const std::string& img1_path) {
    std::stringstream report;
    
    report << "=== LoFTR Match Report ===\n";
    report << "Image 0: " << img0_path << "\n";
    report << "Image 1: " << img1_path << "\n";
    report << "Total matches: " << matches.num_matches << "\n";
    report << "Inference time: " << matches.inference_time_ms << " ms\n";
    
    if (!matches.confidence.empty()) {
        double avg_conf = computeMatchQuality(matches);
        report << "Average confidence: " << std::fixed << std::setprecision(3) << avg_conf << "\n";
    }
    
    // 计算性能指标
    if (matches.inference_time_ms > 0) {
        double fps = 1000.0 / matches.inference_time_ms;
        report << "Estimated FPS: " << std::fixed << std::setprecision(1) << fps << "\n";
    }
    
    return report.str();
}

// ==================== 性能优化函数实现 ====================

std::vector<cv::Mat> batchPreprocess(const std::vector<cv::Mat>& images,
                                    const PreprocessConfig& config) {
    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());
    
    for (const auto& img : images) {
        processed_images.push_back(preprocessImage(img, config));
    }
    
    return processed_images;
}

std::vector<MatchResult> parallelMatch(const std::vector<std::pair<cv::Mat, cv::Mat>>& image_pairs,
                                      std::function<MatchResult(const cv::Mat&, const cv::Mat&)> process_func,
                                      int num_threads) {
    std::vector<MatchResult> results(image_pairs.size());
    std::vector<std::future<void>> futures;
    
    // 限制线程数
    num_threads = std::min(num_threads, static_cast<int>(std::thread::hardware_concurrency()));
    
    auto worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            results[i] = process_func(image_pairs[i].first, image_pairs[i].second);
        }
    };
    
    // 分配任务
    int chunk_size = (image_pairs.size() + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, static_cast<int>(image_pairs.size()));
        
        if (start < end) {
            futures.emplace_back(std::async(std::launch::async, worker, start, end));
        }
    }
    
    // 等待所有任务完成
    for (auto& future : futures) {
        future.wait();
    }
    
    return results;
}

// ==================== 工具和辅助函数实现 ====================

VisualizationConfig createDefaultVisConfig(bool for_vins) {
    VisualizationConfig config;
    
    if (for_vins) {
        config.show_trajectory = true;
        config.show_confidence = false;
        config.trajectory_color = cv::Scalar(0, 255, 255); // 黄色轨迹
        config.keypoint_radius = 2;
        config.line_thickness = 1;
    } else {
        config.show_trajectory = false;
        config.show_confidence = true;
        config.keypoint_radius = 3;
        config.line_thickness = 2;
    }
    
    return config;
}

} // namespace LoFTR_Utils