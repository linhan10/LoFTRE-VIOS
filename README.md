# LoFTRE-VIOS
## LoFTRE-VIOS: Local Feature Transformer-Enhanced Visual-Inertial Odometry System for Feature Tracking and Real-Time Localization of UAV in Indistinctive Environments

<img width="1387" height="732" alt="image" src="https://github.com/user-attachments/assets/2e9ca04f-a7dd-4ee4-a347-3893e14d6406" />



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
# 4. Dataset 
EuRoC MAV Dataset
https://projects.asl.ethz.ch/datasets/euroc-mav/
# 5. Result
## 5.1 EuRoC Dataset result
<img width="472" height="430" alt="image" src="https://github.com/user-attachments/assets/4914c045-3bb6-429f-9c18-7d06a31d3ec6" />

Trajectory compare with LoFTRE_VIOS, SuperVINS, and VINS-Fusion in EuRoC dataset MH05

## 5.2 Real-World experiment
<img width="415" height="396" alt="image" src="https://github.com/user-attachments/assets/2ad7b673-931e-454a-8414-7e0eb579280a" />
<img width="413" height="447" alt="image" src="https://github.com/user-attachments/assets/a74efdbb-00f2-4d4b-90b8-cdc55de0fccd" />

## 5.3 Real-World result
<img width="472" height="134" alt="image" src="https://github.com/user-attachments/assets/e9e52ef7-0d01-409b-b87a-be9d9b7ac248" />

<img width="472" height="134" alt="image" src="https://github.com/user-attachments/assets/149b9ba4-1795-4e8c-ab47-6ed037103e84" />


# Thanks
https://github.com/luohongk/SuperVINS https://github.com/HKUST-Aerial-Robotics/VINS-Mono https://github.com/zju3dv/LoFTR

