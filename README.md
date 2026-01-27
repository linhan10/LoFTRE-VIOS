# LoFTRE-VIOS
## LoFTRE-VIOS: Local Feature Transformer-Enhanced Visual-Inertial Odometry System for Feature Tracking and Real-Time Localization of UAV in Indistinctive Environments

<img width="2880" height="1472" alt="Gemini_Generated_Image_hjqd9rhjqd9rhjqd" src="https://github.com/user-attachments/assets/998da437-2d8d-4046-a242-cd8f9d2ec0ff" />

# 1. Introduction
This code is run on Nvidia Orin AGX, so the download file is for arm64 system. X86 system need to modify by yourselves.
# 2. Project
## 2.1 Ubuntu
Ubuntu 20.04. ROS Noetic. JetPack 5.1.2
## 2.2 Ceres Solver
Ceres Solver 2.1.0
## 2.3 OpenCV
OpenCV 4.2.0
## 2.4 ONNXRUNTIME
onnxruntime-aarch64-gpu-1.16.0
# 3. Run the project
## 3.1 Build LoFTRE-VIOS in ROS
```
cd ~/loftrvins_ws/src
git clone https://github.com/linhan10/LoFTRE-VIOS.git
cd ../
catkin_make
source ~/loftrvins_ws/devel/setup.bash
```
## 3.2 Excute the program
```
cd ~/loftrvins_ws
roslaunch loftr_vins loftr_vins_rviz.launch
rosbag play ~/loftrvins_ws/dataset/MH_01_easy.bag
rosrun loftr_vins loftr_vins_node ~/loftrvins_ws/src/LoFTR_VINS/config/euroc/euroc_stereo_imu_config.yaml
```
## 3.3 Test for your own device
```
cd ~/loftrvins_ws
roslaunch loftr_vins loftr_vins_rviz.launch
roslaunch zed_wrapper zed2.launch
rosrun loftr_vins loftr_vins_node ~/loftrvins_ws/src/LoFTR_VINS/config/zed2/zed2_stereo_imu_config.yaml
```
