cmake_minimum_required(VERSION 3.10)
project(loftr_vins)

set(CMAKE_BUILD_TYPE "Release")

# 🔧 MODIFIED: 升级到C++17 (TensorRT和filesystem需要)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++17")

# 添加优化编译选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -march=native -mtune=native")

# 添加CUDA支持
find_package(CUDA REQUIRED)
enable_language(CUDA)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rosbag
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# ONNX Runtime配置
set(ONNXRUNTIME_ROOTDIR "/home/lin/Documents/onnxruntime-linux-aarch64-gpu-1.16.0")

# 🔧 修复的TensorRT 8.5.2配置
set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")

# 查找TensorRT库文件 (使用正确的库名)
find_library(TENSORRT_LIBRARIES nvinfer
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH)

find_library(TENSORRT_PLUGIN_LIBRARIES nvinfer_plugin
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH)

find_library(TENSORRT_ONNX_LIBRARIES nvonnxparser
    PATHS ${TENSORRT_LIB_DIR}
    NO_DEFAULT_PATH)

# 验证TensorRT完整性
if(TENSORRT_LIBRARIES AND EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
    # 🔧 使用正确的宏定义
    add_definitions(-DTENSORRT_AVAILABLE)
    include_directories(${TENSORRT_INCLUDE_DIR})
    
    message(STATUS "✅ TensorRT 8.5.2 FOUND AND CONFIGURED!")
    message(STATUS "   - Include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "   - Core Library: ${TENSORRT_LIBRARIES}")
    if(TENSORRT_PLUGIN_LIBRARIES)
        message(STATUS "   - Plugin Library: ${TENSORRT_PLUGIN_LIBRARIES}")
    endif()
    if(TENSORRT_ONNX_LIBRARIES)
        message(STATUS "   - ONNX Library: ${TENSORRT_ONNX_LIBRARIES}")
    endif()
    
    set(TENSORRT_FOUND TRUE)
else()
    message(WARNING "⚠️ TensorRT not found or incomplete")
    message(STATUS "   - TensorRT Libraries: ${TENSORRT_LIBRARIES}")
    message(STATUS "   - Include path: ${TENSORRT_INCLUDE_DIR}")
    set(TENSORRT_FOUND FALSE)
endif()

# 添加库目录
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
if(TENSORRT_FOUND)
    link_directories(${TENSORRT_LIB_DIR})
endif()

# 包含目录
include_directories(${PROJECT_SOURCE_DIR}/src/featureTracker/ort_include/)
include_directories(${PROJECT_SOURCE_DIR}/src/estimator/)
include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

# 添加CUDA头文件
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

# 🔧 修复：只保留必要的编译器定义
add_definitions(-DWITH_LOFTR)

# LoFTR VINS 主库
add_library(loftr_vins_lib
    # 核心估计器文件
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    
    # 优化因子
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    
    # 工具类
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    
    # 初始化
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    
    # 特征跟踪 - 基础文件
    src/featureTracker/feature_tracker.cpp
    src/featureTracker/feature_tracker_dpl.cpp
    src/featureTracker/transform_dpl.cpp
    
    # LoFTR 相关文件
    src/featureTracker/loftr_interface.cpp
    src/featureTracker/loftr_tensorrt.cpp
    src/featureTracker/loftr_onnx.cpp
    src/featureTracker/loftr_utils.cpp
)

# 🔧 修复：基础库链接
target_link_libraries(loftr_vins_lib 
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    onnxruntime
    ${CUDA_LIBRARIES}
    cudart
    cublas
    curand
    cudnn)

# 🔧 修复：TensorRT库链接（统一放在这里）
if(TENSORRT_FOUND)
    target_link_libraries(loftr_vins_lib ${TENSORRT_LIBRARIES})
    
    if(TENSORRT_PLUGIN_LIBRARIES)
        target_link_libraries(loftr_vins_lib ${TENSORRT_PLUGIN_LIBRARIES})
    endif()
    
    if(TENSORRT_ONNX_LIBRARIES)
        target_link_libraries(loftr_vins_lib ${TENSORRT_ONNX_LIBRARIES})
    endif()
endif()

# 设置CUDA架构 (Jetson AGX Orin是Ampere架构)
set_property(TARGET loftr_vins_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET loftr_vins_lib PROPERTY CUDA_ARCHITECTURES "87")  # Orin是sm_87

# 主可执行文件
get_filename_component(PROJECT_SOURCE_DIR_ABS "${CMAKE_SOURCE_DIR}" ABSOLUTE)
add_executable(loftr_vins_node src/loftr_vins_main.cpp)
target_compile_definitions(loftr_vins_node PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR_ABS}")
target_link_libraries(loftr_vins_node loftr_vins_lib)

# LoFTR 测试程序 (可选)
add_executable(loftr_test src/loftr_test.cpp)
target_link_libraries(loftr_test loftr_vins_lib)

# LoFTR 性能测试 (可选)
add_executable(loftr_benchmark src/loftr_benchmark.cpp)
target_link_libraries(loftr_benchmark loftr_vins_lib)

# 调试信息
message(STATUS "============ LoFTR VINS Build Configuration ============")
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
if(TENSORRT_FOUND)
    message(STATUS "TensorRT: ENABLED")
    message(STATUS "TensorRT_LIBRARIES: ${TENSORRT_LIBRARIES}")
else()
    message(STATUS "TensorRT: DISABLED")
endif()
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "LoFTR Support: ENABLED")
message(STATUS "=======================================================")
