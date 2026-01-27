#ifndef LOFTR_UTILS_H
#define LOFTR_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <functional>

/**
 * @brief LoFTR 实用工具类
 * 提供图像预处理、匹配结果处理、可视化等功能
 */
namespace LoFTR_Utils {

    /**
     * @brief 匹配结果结构体
     */
    struct MatchResult {
        std::vector<cv::Point2f> keypoints0;  // 图像0的关键点
        std::vector<cv::Point2f> keypoints1;  // 图像1的关键点
        std::vector<float> confidence;        // 匹配置信度
        int num_matches;                      // 匹配点数量
        double inference_time_ms;             // 推理时间（毫秒）
        
        MatchResult() : num_matches(0), inference_time_ms(0.0) {}
        
        // 便利方法
        bool empty() const { return num_matches == 0; }
        void clear() {
            keypoints0.clear();
            keypoints1.clear();
            confidence.clear();
            num_matches = 0;
            inference_time_ms = 0.0;
        }
    };

    /**
     * @brief 图像预处理配置
     */
    struct PreprocessConfig {
        int target_width = 640;               // 目标宽度
        int target_height = 352;              // 目标高度
        bool maintain_aspect_ratio = true;    // 保持宽高比
        bool normalize = true;                // 是否归一化到 [0,1]
        cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);  // ImageNet 均值
        cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225);   // ImageNet 标准差
        
        PreprocessConfig() = default;
    };

    /**
     * @brief 可视化配置
     */
    struct VisualizationConfig {
        bool show_keypoints = true;           // 显示关键点
        bool show_matches = true;             // 显示匹配线
        bool show_confidence = false;         // 显示置信度文本
        bool show_trajectory = false;         // 显示轨迹（VINS专用）
        int keypoint_radius = 3;              // 关键点半径
        int line_thickness = 1;               // 连接线粗细
        cv::Scalar good_match_color = cv::Scalar(0, 255, 0);    // 好匹配颜色（绿色）
        cv::Scalar bad_match_color = cv::Scalar(0, 0, 255);     // 坏匹配颜色（红色）
        cv::Scalar trajectory_color = cv::Scalar(255, 255, 0);  // 轨迹颜色（青色）
        float confidence_threshold = 0.5f;    // 置信度阈值
        
        VisualizationConfig() = default;
    };

    /**
     * @brief 性能报告结构体
     */
    struct PerformanceReport {
        size_t total_frames = 0;
        double average_time_ms = 0.0;
        double min_time_ms = 0.0;
        double max_time_ms = 0.0;
        double average_fps = 0.0;
        double backend_inference_time_ms = 0.0;
        size_t total_matches = 0;
        double average_confidence = 0.0;
        
        std::string toString() const;
        void reset();
    };

    /**
     * @brief 重投影误差统计
     */
    struct ReprojectionError {
        double mean_error = 0.0;
        double max_error = 0.0;
        double std_error = 0.0;
        std::vector<double> errors;
        
        bool empty() const { return errors.empty(); }
        void clear() {
            mean_error = max_error = std_error = 0.0;
            errors.clear();
        }
    };

    // ==================== 图像预处理函数 ====================
    
    /**
     * @brief 预处理图像用于 LoFTR 推理
     * @param img 输入图像
     * @param config 预处理配置
     * @return 预处理后的图像
     */
    cv::Mat preprocessImage(const cv::Mat& img, const PreprocessConfig& config = PreprocessConfig());
    
    /**
     * @brief 保持宽高比的图像缩放
     * @param img 输入图像
     * @param target_size 目标尺寸
     * @return 缩放后的图像和缩放因子
     */
    std::pair<cv::Mat, float> resizeKeepAspectRatio(const cv::Mat& img, cv::Size target_size);
    
    /**
     * @brief 中心裁剪图像
     * @param img 输入图像
     * @param target_size 目标尺寸
     * @return 裁剪后的图像
     */
    cv::Mat centerCrop(const cv::Mat& img, cv::Size target_size);
    
    /**
     * @brief 图像归一化
     * @param img 输入图像
     * @param mean 均值
     * @param std 标准差
     * @return 归一化后的图像
     */
    cv::Mat normalizeImage(const cv::Mat& img, cv::Scalar mean, cv::Scalar std);

    // ==================== 匹配结果处理函数 ====================
    
    /**
     * @brief 从 confidence matrix 提取匹配
     * @param conf_matrix 置信度矩阵
     * @param img0_size 图像0尺寸
     * @param img1_size 图像1尺寸
     * @param network_size 网络输入尺寸
     * @param threshold 置信度阈值
     * @param max_matches 最大匹配数
     * @return 匹配结果
     */
    MatchResult extractMatches(const std::vector<float>& conf_matrix,
                              cv::Size img0_size, cv::Size img1_size,
                              cv::Size network_size,
                              float threshold = 0.2f,
                              int max_matches = 2000);
    
    /**
     * @brief 过滤匹配结果
     * @param matches 输入匹配结果
     * @param min_confidence 最小置信度
     * @param max_distance 最大距离（像素）
     * @return 过滤后的匹配结果
     */
    MatchResult filterMatches(const MatchResult& matches,
                             float min_confidence = 0.2f,
                             float max_distance = 1000.0f);
    
    /**
     * @brief 使用 RANSAC 过滤匹配
     * @param matches 输入匹配结果
     * @param threshold RANSAC 阈值
     * @param confidence RANSAC 置信度
     * @return 过滤后的匹配结果
     */
    MatchResult ransacFilter(const MatchResult& matches,
                            double threshold = 3.0,
                            double confidence = 0.99);

    /**
     * @brief 从 VINS 格式转换为标准匹配结果
     * @param prev_pts 前一帧特征点
     * @param cur_pts 当前帧特征点
     * @param confidence 置信度（可选）
     * @return 标准匹配结果
     */
    MatchResult convertFromVINSFormat(const std::vector<cv::Point2f>& prev_pts,
                                     const std::vector<cv::Point2f>& cur_pts,
                                     const std::vector<float>& confidence = {});

    // ==================== 坐标转换函数 ====================
    
    /**
     * @brief 将网络输出坐标转换为原图坐标
     * @param points 网络输出坐标
     * @param network_size 网络输入尺寸
     * @param original_size 原图尺寸
     * @return 转换后的坐标
     */
    std::vector<cv::Point2f> rescalePoints(const std::vector<cv::Point2f>& points,
                                          cv::Size network_size,
                                          cv::Size original_size);
    
    /**
     * @brief 特征索引转换为图像坐标
     * @param feature_idx 特征索引
     * @param feature_width 特征图宽度
     * @param resolution 分辨率因子
     * @return 图像坐标
     */
    cv::Point2f featureIndexToImageCoord(int feature_idx, int feature_width, int resolution);

    // ==================== 可视化函数 ====================
    
    /**
     * @brief 可视化匹配结果
     * @param img0 第一张图像
     * @param img1 第二张图像
     * @param matches 匹配结果
     * @param config 可视化配置
     * @return 可视化图像
     */
    cv::Mat visualizeMatches(const cv::Mat& img0, const cv::Mat& img1,
                            const MatchResult& matches,
                            const VisualizationConfig& config = VisualizationConfig());
    
    /**
     * @brief 带轨迹的可视化（VINS专用）
     * @param img0 第一张图像
     * @param img1 第二张图像
     * @param matches 匹配结果
     * @param prev_points 前一帧特征点
     * @param track_counts 跟踪计数
     * @param config 可视化配置
     * @return 可视化图像
     */
    cv::Mat visualizeMatchesWithTrajectory(const cv::Mat& img0, const cv::Mat& img1,
                                          const MatchResult& matches,
                                          const std::vector<cv::Point2f>& prev_points,
                                          const std::vector<int>& track_counts,
                                          const VisualizationConfig& config = VisualizationConfig());
    
    /**
     * @brief 在图像上绘制关键点
     * @param img 输入图像
     * @param keypoints 关键点
     * @param color 颜色
     * @param radius 半径
     * @return 绘制后的图像
     */
    cv::Mat drawKeypoints(const cv::Mat& img,
                         const std::vector<cv::Point2f>& keypoints,
                         cv::Scalar color = cv::Scalar(0, 255, 0),
                         int radius = 3);
    
    /**
     * @brief 绘制匹配线
     * @param img_pair 拼接的图像对
     * @param keypoints0 图像0的关键点
     * @param keypoints1 图像1的关键点
     * @param confidence 置信度（可选）
     * @param config 可视化配置
     * @return 绘制后的图像
     */
    cv::Mat drawMatches(const cv::Mat& img_pair,
                       const std::vector<cv::Point2f>& keypoints0,
                       const std::vector<cv::Point2f>& keypoints1,
                       const std::vector<float>& confidence = {},
                       const VisualizationConfig& config = VisualizationConfig());

    // ==================== 质量评估函数 ====================
    
    /**
     * @brief 计算匹配质量指标
     * @param matches 匹配结果
     * @return 质量指标（平均置信度）
     */
    double computeMatchQuality(const MatchResult& matches);
    
    /**
     * @brief 计算重投影误差
     * @param matches 匹配结果
     * @param H 单应性矩阵
     * @return 重投影误差统计
     */
    ReprojectionError computeReprojectionError(const MatchResult& matches, const cv::Mat& H);

    /**
     * @brief 生成性能报告
     * @param stats 性能统计数据
     * @return 性能报告
     */
    PerformanceReport generatePerformanceReport(const std::map<std::string, double>& stats);

    // ==================== 文件 I/O 函数 ====================
    
    /**
     * @brief 保存匹配结果到文件
     * @param matches 匹配结果
     * @param filename 文件路径
     * @param format 文件格式（"txt", "yaml", "json"）
     * @return 是否保存成功
     */
    bool saveMatches(const MatchResult& matches,
                    const std::string& filename,
                    const std::string& format = "txt");
    
    /**
     * @brief 从文件加载匹配结果
     * @param filename 文件路径
     * @return 匹配结果
     */
    MatchResult loadMatches(const std::string& filename);
    
    /**
     * @brief 保存性能统计
     * @param stats 性能统计数据
     * @param filename 文件路径
     * @return 是否保存成功
     */
    bool savePerformanceStats(const std::map<std::string, double>& stats,
                             const std::string& filename);

    /**
     * @brief 保存性能报告
     * @param report 性能报告
     * @param filename 文件路径
     * @return 是否保存成功
     */
    bool savePerformanceReport(const PerformanceReport& report,
                              const std::string& filename);

    // ==================== 调试和诊断函数 ====================
    
    /**
     * @brief 打印匹配结果统计信息
     * @param matches 匹配结果
     */
    void printMatchStats(const MatchResult& matches);
    
    /**
     * @brief 验证匹配结果的有效性
     * @param matches 匹配结果
     * @param img0_size 图像0尺寸
     * @param img1_size 图像1尺寸
     * @return 是否有效
     */
    bool validateMatches(const MatchResult& matches,
                        cv::Size img0_size,
                        cv::Size img1_size);
    
    /**
     * @brief 生成匹配报告
     * @param matches 匹配结果
     * @param img0_path 图像0路径
     * @param img1_path 图像1路径
     * @return 报告字符串
     */
    std::string generateMatchReport(const MatchResult& matches,
                                   const std::string& img0_path,
                                   const std::string& img1_path);

    // ==================== 性能优化函数 ====================
    
    /**
     * @brief 批处理图像预处理
     * @param images 输入图像列表
     * @param config 预处理配置
     * @return 预处理后的图像列表
     */
    std::vector<cv::Mat> batchPreprocess(const std::vector<cv::Mat>& images,
                                        const PreprocessConfig& config = PreprocessConfig());
    
    /**
     * @brief 并行匹配处理
     * @param image_pairs 图像对列表
     * @param process_func 处理函数
     * @param num_threads 线程数
     * @return 匹配结果列表
     */
    std::vector<MatchResult> parallelMatch(const std::vector<std::pair<cv::Mat, cv::Mat>>& image_pairs,
                                          std::function<MatchResult(const cv::Mat&, const cv::Mat&)> process_func,
                                          int num_threads = 4);

    // ==================== 工具和辅助函数 ====================
    
    /**
     * @brief 计算两点之间的距离
     * @param pt1 第一个点
     * @param pt2 第二个点
     * @return 距离
     */
    inline double computeDistance(const cv::Point2f& pt1, const cv::Point2f& pt2) {
        return cv::norm(pt1 - pt2);
    }
    
    /**
     * @brief 检查点是否在图像边界内
     * @param pt 点坐标
     * @param img_size 图像尺寸
     * @param border 边界大小
     * @return 是否在边界内
     */
    inline bool isInBounds(const cv::Point2f& pt, cv::Size img_size, int border = 1) {
        return pt.x >= border && pt.x < img_size.width - border &&
               pt.y >= border && pt.y < img_size.height - border;
    }
    
    /**
     * @brief 创建默认的可视化配置
     * @param for_vins 是否为 VINS 使用
     * @return 可视化配置
     */
    VisualizationConfig createDefaultVisConfig(bool for_vins = false);

} // namespace LoFTR_Utils

#endif // LOFTR_UTILS_H
