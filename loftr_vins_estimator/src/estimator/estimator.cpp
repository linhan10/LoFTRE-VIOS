/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator() : f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while (!accBuf.empty())
        accBuf.pop();
    while (!gyrBuf.empty())
        gyrBuf.pop();
    while (!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl
             << ric[i] << endl
             << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    
    // è¯»å–ç›¸æœºå†…å‚
    featureTracker.readIntrinsicParameter(CAM_NAMES);
    
    // ä½¿ç”¨ LoFTR åˆå§‹åŒ–æ–¹æ³•æ›¿ä»£åŽŸæ¥çš„æ–¹æ³•
    featureTracker.loadConfiguration();  // åŠ è½½é…ç½®
    bool loftr_success = featureTracker.initializeLoFTR();    // åˆå§‹åŒ– LoFTR
    
    if (!loftr_success) {
        ROS_ERROR("[Estimator] LoFTR initialization failed!");
    }
    
    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if (!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if (USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if (USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }

        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if (restart)
    {
        clearState();
        setParameter();
    }
}

// è¾“å…¥
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;
    
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;
    
    // ä½¿ç”¨ LoFTR è¿›è¡Œç‰¹å¾è·Ÿè¸ª
    if (_img1.empty() || !STEREO) {
        // å•ç›®æ¨¡å¼ï¼šä½¿ç”¨ LoFTR è·Ÿè¸ªæ–¹æ³•
        featureFrame = featureTracker.trackImage_loftr(t, _img); 
        ROS_DEBUG("Monocular LoFTR tracking: %zu features", featureFrame.size());
    } else {
        // åŒç›®æ¨¡å¼ï¼šä½¿ç”¨ LoFTR åŒç›®è·Ÿè¸ªæ–¹æ³•
        featureFrame = featureTracker.trackImage_loftr(t, _img, _img1); 
        ROS_DEBUG("Stereo LoFTR tracking: %zu features", featureFrame.size());
    }

    // éªŒè¯ç‰¹å¾æ•°æ®
    int valid_features = 0;
    int stereo_features = 0;
    for (const auto &feature : featureFrame) {
        if (!feature.second.empty()) {
            valid_features++;
            if (feature.second.size() == 2) {
                stereo_features++;
            }
        }
    }
    ROS_DEBUG("Valid features: %d, Stereo features: %d", valid_features, stereo_features);

    if (SHOW_TRACK) {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }

    // æŽ¨å…¥ç‰¹å¾ç¼“å†²åŒº
    if (MULTIPLE_THREAD) {
        if (inputImageCnt % 2 == 0) {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    } else {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame)); 
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
    }
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    // printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if (!MULTIPLE_THREAD)
        processMeasurements();
}

/// @brief èŽ·å–t0å’Œt1ä¹‹é—´çš„IMUæ•°æ®
/// @param t0 //ä¸Šä¸€å¸§æ—¶é—´æˆ³
/// @param t1 //å½“å‰å¸§æ—¶é—´æˆ³
/// @param accVector
/// @param gyrVector
/// @return
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                               vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if (accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    // printf("get imu from %f %f\n", t0, t1);
    // printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if (t1 <= accBuf.back().first) // å¦‚æžœå½“å‰å¸§æ—¶é—´æˆ³t1å°äºŽç­‰äºŽæœ€æ–°çš„IMUæ•°æ®æ—¶é—´æˆ³
    {
        while (accBuf.front().first <= t0) // æœ€è€çš„IMUæ•°æ®å°äºŽä¸Šä¸€å¸§æ—¶é—´æˆ³æ—¶ï¼Œå°±å¼¹å‡ºè¿™æ•°æ®ï¼Œå› ä¸ºè¿™æ•°æ®æ²¡æ³•ç”¨äºŽå¸§é—´é¢„ç§¯åˆ†
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        // ä¸Šä¸€ä¸ªwhileç»“æŸä»¥åŽï¼Œbufferé‡Œçš„æ•°æ®å°±éƒ½æ˜¯å‰ä¸€å¸§t0ä¹‹åŽçš„äº†
        while (accBuf.front().first < t1) // å½“æœ€è€çš„IMUæ•°æ®å°äºŽå½“å‰å¸§æ—¶é—´æˆ³ï¼Œå°±æŠŠæ•°æ®æŽ¨å…¥accVectorã€gyrVector
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front()); // å¤šæŽ¨äº†ä¸€ä¸ªæ•°æ®è¿›æ¥ï¼Œç­‰äºŽæˆ–ç•¥å¾®å¤§äºŽt1
        gyrVector.push_back(gyrBuf.front());
    }
    else // è¦æ˜¯æœ€æ–°çš„æ•°æ®éƒ½å°äºŽå½“å‰å¸§æ—¶é—´æˆ³ï¼Œè¯´æ˜Žå¯èƒ½è¿˜æœ‰å¯èƒ½æœ‰ç”¨çš„æ•°æ®æ²¡è¾“å…¥
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

/// @brief åˆ¤æ–­æ—¶é—´æˆ³tä¹‹åŽæ˜¯å¦æœ‰å¯ç”¨IMUæ•°æ®
/// @param t
/// @return
bool Estimator::IMUAvailable(double t)
{
    if (!accBuf.empty() && t <= accBuf.back().first) // åŠ é€Ÿåº¦è®¡bufferä¸ä¸ºç©ºï¼Œæœ€æ–°çš„imuæ•°æ®æ—¶é—´æˆ³å¤§äºŽtï¼ˆé˜Ÿåˆ—æ˜¯å…ˆè¿›å…ˆå‡ºçš„ï¼Œ.frontæ˜¯æœ€è€çš„ï¼Œ.backæ˜¯æ–°çš„ï¼‰ï¼Œåˆ™æœ‰å¯ç”¨imuæ•°æ®
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    while (1)
    {
        // std_msgs::Header des_header;
        // des_header.frame_id = "world";
        // // å‘å¸ƒæ¯ä¸€å¸§ç‰¹å¾æè¿°ç¬¦çš„æ•°æ®
        // pubSuperPointDescriptors(*this, des_header);

        // printf("process measurments\n");
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature; // ç‰¹å¾ï¼Œdoubleæ˜¯æ—¶é—´æˆ³ï¼ŒåŽé¢é‚£ä¸ªmapå’Œfeature_tracker.track_imageè¿”å›žçš„featureframeä¸€æ ·ï¼Œæ„æ€æ˜¯è¿™ä¸ªæ—¶é—´æˆ³ä¸‹çš„æ‰€æœ‰ç‰¹å¾
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;                     // doubleæ˜¯æ—¶é—´æˆ³
        if (!featureBuf.empty())
        {
            feature = featureBuf.front(); // å› ä¸ºæ¯è¿½è¸ªå®Œä¸€å¸§å›¾åƒå°±æŠŠfeatureframeæŽ¨å…¥å…ˆè¿›å…ˆå‡ºçš„bufferï¼Œæ‰€ä»¥è¦å–æœ€è€çš„featureå‡ºæ¥
            curTime = feature.first + td; // å½“å‰æ—¶åˆ»ç”±ç‰¹å¾ï¼ˆå…¶å®žå°±æ˜¯å›¾åƒæ—¶é—´æˆ³ï¼‰åŠ ä¸Šä¸€ä¸ªå›ºå®šå»¶è¿Ÿ
            while (1)
            {
                if ((!USE_IMU || IMUAvailable(feature.first + td))) // æ£€æŸ¥æ˜¯å¦ä½¿ç”¨IMUï¼Œå¹¶æ£€æŸ¥æ˜¯å¦æœ‰å¯ç”¨IMUæ•°æ®
                    break;
                else // å¦‚æžœæ²¡æœ‰å°±ç­‰å¾…IMUæ•°æ®
                {
                    printf("wait for imu ... \n");
                    if (!MULTIPLE_THREAD) // éžå¤šçº¿ç¨‹æ—¶ç›´æŽ¥è¿”å›žï¼Œç­‰å¾…IMUæ•°æ®åˆ°æ¥ä»¥åŽå†æŽ¥ä¸‹åŽ»è¿è¡Œï¼›å¦‚æžœå¤šçº¿ç¨‹ï¼Œåˆ™æŽ¥ç€å¾€ä¸‹è¿è¡Œ
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock();
            if (USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector); // èŽ·å–ä¸Šä¸€å¸§å’Œå½“å‰å¸§ä¹‹é—´çš„IMUæ•°æ®

            featureBuf.pop(); // å¼¹å‡ºæœ€è€çš„ç‰¹å¾ï¼Œå› ä¸ºå·²ç»èµ‹äºˆfeatureè¿™ä¸ªå˜é‡äº†
            mBuf.unlock();

            if (USE_IMU)
            {
                if (!initFirstPoseFlag) // è¦æ˜¯æ²¡æœ‰åˆå§‹åŒ–ä½å§¿ï¼Œé¦–å…ˆè¦è¿›è¡ŒIMUä½å§¿åˆå§‹åŒ–ï¼Œå°†åˆå§‹ä½å§¿ä¸Žé‡åŠ›æ–¹å‘å¯¹é½
                    initFirstIMUPose(accVector);
                for (size_t i = 0; i < accVector.size(); i++)
                {
                    /*                                                                                                   --------dt----------
                                                                                                                        |                    |
                    IMUæ•°æ®ä¹‹é—´çš„æ—¶é—´é—´éš”ï¼Œå®ƒçš„ç»“æž„æ˜¯ï¼šprevTime---dt--->imu[0]---dt--->imu[1]---dt--->imu[2]---...---imu[i-1]|---dt--->curTime-----imu[i]*/
                    double dt;
                    if (i == 0)
                        dt = accVector[i].first - prevTime; // å› ä¸ºaccVector[i]è‚¯å®šæ¯”prevTimeå¤§ï¼Œæ‰€ä»¥ç¬¬ä¸€ä¸ªæ•°æ®åŽ†å…ƒçš„æ—¶é—´é—´éš”æ˜¯ç¬¬ä¸€ä¸ªåˆ°ä¸Šä¸€å¸§ä¹‹é—´çš„æ—¶é—´å·®å¼‚
                    else if (i == accVector.size() - 1)     // å¦‚æžœaccVector[i]ä¸­é™¤æœ€åŽä¸€ä¸ªåŽ†å…ƒä»¥å¤–éƒ½å°äºŽcurTimeï¼Œæ‰€ä»¥æœ€å€’æ•°ç¬¬äºŒä¸ªåŽ†å…ƒçš„æ—¶é—´é—´éš”æ˜¯å½“å‰å¸§æ—¶é—´æˆ³å‡åŽ»å€’æ•°ç¬¬äºŒä¸ªåŽ†å…ƒæ—¶é—´æˆ³
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second); // æœºæ¢°ç¼–æŽ’+é¢„ç§¯åˆ†
                }
            }
            mProcess.lock();                             // ä¸Šé”ï¼Œé˜²æ­¢å…¶å®ƒçº¿ç¨‹æ”¹å˜å†…å‚ã€çŠ¶æ€é‡ç­‰ç­‰
            
            // å¼€å§‹è®¡æ—¶
            auto start = std::chrono::high_resolution_clock::now();

            processImage(feature.second, feature.first); // å…¥å£å‡½æ•°ï¼ŒçŠ¶æ€ä¼°è®¡éƒ½åœ¨è¿™å‡½æ•°é‡Œé¢å¼€å§‹è¿›è¡Œäº†

            // ç»“æŸè®¡æ—¶
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start; // è®¡ç®—æŒç»­æ—¶é—´

            // è¾“å‡ºæ—¶é—´åˆ°æŽ§åˆ¶å°
            // std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

            // æ‰“å¼€æ–‡ä»¶è¿›è¡Œä¿å­˜ï¼ˆä»¥è¿½åŠ æ¨¡å¼æ‰“å¼€ï¼‰
            std::ofstream outFile("time_consumption/backend_optimization.txt", std::ios::app);
            if (outFile.is_open()) {
                outFile << duration.count() <<std::endl; // å†™å…¥æ‰§è¡Œæ—¶é—´
                outFile.close(); // å…³é—­æ–‡ä»¶
            } else {
                std::cerr << "Unable to open file" << std::endl; // é”™è¯¯å¤„ç†
            }

            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);

            pubSuperPointDescriptors(*this, header);
            mProcess.unlock();
        }

        if (!MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

/// @brief åˆå§‹åŒ–ç¬¬ä¸€ä¸ªIMUä½å§¿
/// @param accVector
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    // return;
    Eigen::Vector3d averAcc(0, 0, 0);
    // è®¡ç®—å¹³å‡åŠ é€Ÿåº¦
    int n = (int)accVector.size();
    for (size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    // è®¡ç®—å¹³å‡åŠ é€Ÿåº¦æ–¹å‘å’Œ[0,0,1]åž‚ç›´æ–¹å‘çš„æ—‹è½¬å…³ç³»
    Matrix3d R0 = Utility::g2R(averAcc);
    // ä»ŽR0ä¸­è§£å‡ºåèˆªè§’
    double yaw = Utility::R2ypr(R0).x();
    // ä»…ä¿®æ­£åèˆªè§’æ–¹å‘çš„æ—‹è½¬ï¼Œå°†åˆå§‹å§¿æ€ä¸Žé‡åŠ›æ–¹å‘å¯¹é½
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl
         << Rs[0] << endl;
    // Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

/// @brief è¿›è¡Œé¢„ç§¯åˆ†ï¼Œå¹¶ç”¨æœºæ¢°ç¼–æŽ’é¢„æµ‹çŠ¶æ€é‡
/// @param t imuæ•°æ®çš„æ—¶é—´æˆ³
/// @param dt imuæ•°æ®æ—¶é—´æˆ³å’Œä¸Šä¸€ä¸ªåŽ†å…ƒæ—¶é—´æˆ³ä¹‹é—´çš„é—´éš”
/// @param linear_acceleration tæ—¶åˆ»imuæ•°æ®çš„åŠ è®¡æ•°æ®
/// @param angular_velocity tæ—¶åˆ»imuæ•°æ®çš„è§’é€Ÿåº¦æ•°æ®
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // å¦‚æžœåˆå§‹IMUæ•°æ®è¿˜æ²¡èµ‹å€¼ï¼Œåˆ™èµ‹äºˆä¸Šä¸€ä¸ªIMUæ•°æ®å˜é‡å½“å‰è¿™ä¸ªIMUæ•°æ®
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // å¦‚æžœç¬¬frame_countä¸ªé¢„ç§¯åˆ†æ˜¯ç©ºçš„ï¼Œåˆ™è¦æ–°å»ºä¸€ä¸ªé¢„ç§¯åˆ†é‡
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // å¦‚æžœå½“å‰å¸§ä¸æ˜¯åˆå§‹ç¬¬ä¸€å¸§ï¼Œåˆ™è¦è¿›è¡Œé¢„ç§¯åˆ†
    if (frame_count != 0)
    {
        // ç¬¬frame_countä¸ªé¢„ç§¯åˆ†é‡è®¡ç®—
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        // if(solver_flag != NON_LINEAR)
        // ä¸´æ—¶é¢„ç§¯åˆ†é‡è®¡ç®—
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
        // ç¬¬frame_countå¸§å’Œä¸Šä¸€å¸§ä¹‹é—´çš„IMUæ•°æ®å­˜å‚¨
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        // æœºæ¢°ç¼–æŽ’é¢„æµ‹ç¬¬frame_countå¸§çš„ä½å§¿å’Œé€Ÿåº¦
        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    // å°†ä¸Šä¸€ä¸ªIMUæ•°æ®å˜é‡èµ‹äºˆå½“å‰IMUæ•°æ®
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/// @brief å¤„ç†å›¾åƒå¸§ï¼ŒçŠ¶æ€ä¼°è®¡
/// @param image
/// @param header
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td)) // æ£€æŸ¥ç‰¹å¾çš„è§†å·®æ¥åˆ¤æ–­è¾¹ç¼˜åŒ–æƒ…å†µï¼ˆåˆ¤æ–­æ˜¯å¦ä¸ºåŠ å…¥æ»‘çª—çš„å…³é”®å¸§ï¼‰ï¼ŒåŒæ—¶è¿™ä¸ªå‡½æ•°ä¹Ÿä¼šå°†ç‰¹å¾æŽ¨å…¥feature_manager
    {
        marginalization_flag = MARGIN_OLD; // è¦æ˜¯å½“å‰å¸§ç‰¹å¾è´¨é‡å¥½ï¼Œå°±æŠŠæœ€è€çš„å…³é”®å¸§è¾¹ç¼˜åŒ–æŽ‰
        // printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW; // è¦æ˜¯å½“å‰å¸§ç‰¹å¾è´¨é‡ä¸æ€Žä¹ˆåœ°ï¼Œå°±è¾¹ç¼˜åŒ–æ–°çš„å¸§
        // printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);                                                        // æž„å»ºå½“å‰å›¾åƒå¸§
    imageframe.pre_integration = tmp_pre_integration;                                            // èµ‹äºˆå½“å‰å›¾åƒå¸§çš„é¢„ç§¯åˆ†é‡ä¸´æ—¶é¢„ç§¯åˆ†é‡
    all_image_frame.insert(make_pair(header, imageframe));                                       // æ’å…¥åˆ°æ‰€æœ‰å›¾åƒå¸§åº“é‡Œ
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; // ä¸Šä¸€ä¸ªä¸´æ—¶é¢„ç§¯åˆ†å·²ç»å®Œæˆäº†å®ƒçš„ä½¿å‘½ï¼Œé©¬ä¸Šè¦åŽ»ç”¨äºŽå½“å‰å¸§çŠ¶æ€ä¼°è®¡äº†ï¼Œæ‰€ä»¥çŽ°åœ¨è¦æ–°å»ºä¸€ä¸ªä¸ºä¸‹ä¸€å¸§æœåŠ¡

    if (ESTIMATE_EXTRINSIC == 2) // å¦‚æžœè¦åœ¨çº¿æ ‡å®šå¤–å‚ï¼Œå°±è¦æ‰¾å½“å‰å¸§å’Œä¸Šä¸€å¸§ä¹‹é—´çš„ç‰¹å¾å…³è”ï¼Œç»“åˆé¢„ç§¯åˆ†é‡è¿›è¡Œå¤–å‚ä¼˜åŒ–
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                               << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // è¦æ˜¯ç³»ç»Ÿæ²¡åˆå§‹åŒ–ï¼Œéœ€è¦å…ˆå°†ç³»ç»Ÿåˆå§‹åŒ–
    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        // å•ç›®è§†è§‰æƒ¯æ€§åˆå§‹åŒ–
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE) // å‡‘å¤Ÿä¸€ä¸ªæ»‘çª—çš„å›¾åƒå¸§æ‰èƒ½è¿›è¡Œåˆå§‹åŒ–
            {
                bool result = false;
                // å¦‚æžœä¸éœ€è¦åœ¨çº¿æ ‡å®šå¤–å‚ï¼Œè€Œä¸”å½“å‰å¸§æ—¶é—´æˆ³ä¸Žåˆå§‹æ—¶é—´æˆ³å¤§äºŽ0.1sï¼Œåˆ™å¯ä»¥è¿›è¡Œsfm
                if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {

                    result = initialStructure();
                    initial_timestamp = header;
                }
                if (result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        // åŒç›®è§†è§‰æƒ¯æ€§åˆå§‹åŒ–
        if (STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            
            if (frame_count == WINDOW_SIZE)
            {
                // æ·»åŠ åˆå§‹åŒ–è´¨é‡æ£€æŸ¥
                int valid_features = f_manager.getFeatureCount();
                ROS_INFO("Features for initialization: %d", valid_features);
                
                if (valid_features < 50) {
                    ROS_WARN("Not enough features for initialization: %d", valid_features);
                    slideWindow();
                    return;
                }
                
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                
                // é™€èžºä»ªåç½®æ ‡å®š
                solveGyroscopeBias(all_image_frame, Bgs);
                ROS_INFO("Gyroscope bias calibrated");
                
                // é‡æ–°é¢„ç§¯åˆ†
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                
                // è§†è§‰-æƒ¯æ€§å¯¹é½
                TicToc t_align;
                if (!visualInitialAlign()) {
                    ROS_ERROR("Visual-Inertial alignment failed!");
                    slideWindow();
                    return;
                }
                ROS_INFO("Visual-Inertial alignment success, time: %.3f ms", t_align.toc());
                
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if (STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if (frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if (frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }
    }
    else
    {
        TicToc t_solve;
        if (!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        optimization();
        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (!MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }

        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }
}

/// @brief åˆå§‹åŒ–ç»“æž„ï¼Œsfmå‡ºä¸€äº›åˆå§‹å§¿æ€ã€ä½ç½®å’Œç‚¹çš„ä½ç½®
/// @return
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // check imu observibility
    // å…ˆæ£€æŸ¥IMUå¯è§‚æ€§ï¼Œå®žé™…ä¸Šå°±æ˜¯æ£€æŸ¥IMUæ˜¯å¦æœ‰å……åˆ†è¿åŠ¨ï¼ˆæ¿€åŠ±ï¼‰ä½†è¿™æ®µä»£ç ç®—äº†åŠå¤©æœ€åŽä¹Ÿæ²¡èµ·ä½œç”¨ï¼Œè¦ä¸å’±ç›´æŽ¥çœ‹ä¸‹é¢sfmçš„ä»£ç å§
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // éåŽ†çŽ°æœ‰çš„æ‰€æœ‰å¸§
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;            // å–å‡ºæ¯å¸§é¢„ç§¯åˆ†çš„æ—¶é—´é—´éš”
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; // ç”¨é€Ÿåº¦å˜åŒ–é‡é™¤ä»¥æ—¶é—´é—´éš”å¾—åˆ°æ¯å¸§çš„å¹³å‡é€Ÿåº¦å˜åŒ–é‡
            sum_g += tmp_g;                                                  // å°†æ‰€æœ‰å¸§çš„å¹³å‡é€Ÿåº¦å˜åŒ–å€¼ç´¯åŠ 
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); // ç®—æ‰€æœ‰å¸§çš„å¹³å‡é€Ÿåº¦å˜åŒ–é‡
        double var = 0;                                           // æ–°å»ºä¸€ä¸ªå˜é‡ç”¨æ¥ä¿å­˜æ–¹å·®
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1)); // è®¡ç®—æ‰€æœ‰å¸§é€Ÿåº¦å˜åŒ–é‡çš„æ–¹å·®
        // ROS_WARN("IMU variation %f!", var);
        if (var < 0.25)
        {
            ROS_INFO("IMU excitation not enough!");
            // return false;
        }
    }
    // global sfm

    Quaterniond Q[frame_count + 1];           // å‡†å¤‡frame_count+1ä¸ªä½ç½®æ¥å­˜æ”¾æ—‹è½¬Q
    Vector3d T[frame_count + 1];              // å‡†å¤‡frame_count+1ä¸ªä½ç½®æ¥å­˜æ”¾å¹³ç§»T
    map<int, Vector3d> sfm_tracked_points;    // å‡†å¤‡ä¸ªmapå­˜æ”¾[feature_idï¼Œç‰¹å¾ç‚¹]
    vector<SFMFeature> sfm_f;                 // å‡†å¤‡ä¸ªvectorå­˜æ”¾sfmå‡ºæ¥çš„ç‰¹å¾
    for (auto &it_per_id : f_manager.feature) // éåŽ†æ»‘çª—å†…æ‰€æœ‰ç‰¹å¾
    {
        int imu_j = it_per_id.start_frame - 1; // å¼€ä¸€ä¸ªè®¡æ•°å™¨
        SFMFeature tmp_feature;                // å»ºç«‹ä¸€ä¸ªä¸´æ—¶sfmfeature
        tmp_feature.state = false;             // sfmfeatureçŠ¶æ€å…ˆç½®ä¸ºå¦ï¼Œè¿™ä¸ªçŠ¶æ€ä»£è¡¨æ˜¯å¦ä¸‰è§’åŒ–
        tmp_feature.id = it_per_id.feature_id; // sfmfeatureçš„idç½®ä¸ºç‰¹å¾id
        // éåŽ†è¯¥ç‰¹å¾çš„åŽ†å²è§‚æµ‹ï¼ˆå½’ä¸€åŒ–å¹³é¢çš„ç‚¹åæ ‡ï¼‰ï¼Œå°†å­˜åˆ°tmp_featureå½“ä¸­
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            // SFMFeatureçš„observationä¸­ï¼Œå­˜å‚¨äº†ä¸€ä¸ªç©ºé—´ç‚¹åº”çš„æ‰€æœ‰çš„è§‚æµ‹å¸§idå’Œå½’ä¸€åŒ–å¹³é¢ç‚¹
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    // åœ¨æ»‘çª—å†…æ‰¾ä¸€å¸§è´¨é‡å¥½çš„ä½œä¸ºå‚è€ƒå¸§ï¼Œè®¡ç®—å…¶ä¸Žæ»‘çª—å†…æœ€åŽä¸€å¸§çš„ç›¸å¯¹ä½å§¿
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    // åˆ›å»ºä¸€ä¸ªsfmå™¨ï¼Œåˆ©ç”¨åˆšæ‰æ‰¾åˆ°çš„ç¬¬lå¸§å’Œæ»‘çª—æœ€åŽä¸€å¸§çš„ç›¸å¯¹ä½å§¿è¿›è¡Œå…¨å±€sfm
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        // å…¨å±€sfmå¤±è´¥äº†ï¼Œå°†æœ€è€çš„å¸§ä¸¢å¼ƒ
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    // éåŽ†æ¯ä¸ªå›¾åƒå¸§
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i]) // çª—å£å†…çš„å¸§ä½å§¿æ€éƒ½å·²ç»sfmå‡ºæ¥äº†ï¼Œç›´æŽ¥èµ‹å€¼ï¼Œè®¾ä¸ºå…³é”®å¸§
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix(); // å¦‚æžœä¸æ˜¯çª—å£å†…çš„å¸§ï¼Œè€Œæ˜¯æ—¶é—´æˆ³å¤§äºŽçª—å£å†…ç¬¬iå¸§æ—¶é—´æˆ³ï¼Œä»¥çª—å£å†…ç¬¬iå¸§ä½å§¿ä¸ºåˆå€¼
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points) // éåŽ†ä¸€ä¸‹å›¾åƒå¸§çš„ç‰¹å¾ç‚¹ï¼Œæ‰¾åˆ°sfmå‡ºæ¥çš„å¯¹åº”ç©ºé—´ç‚¹ï¼Œä¸ºPnPåšå‡†å¤‡
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        // æ±‚è§£PnPï¼Œè¿™é‡Œå¯ä»¥çœ‹å‡ºï¼Œæœ‰ä¸€å¸§è¦æ˜¯æ²¡æ±‚è§£å‡ºæ¥ï¼Œæ•´ä¸ªinitialStructureå°±ç®—å¤±è´¥äº†
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // åœ¨èŽ·å–äº†sfmåˆå§‹åŒ–ç»“æžœä»¥åŽï¼Œå°±è¦åšæƒ¯æ€§ç›¸å…³çš„åˆå§‹åŒ–äº†
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;

    ROS_INFO("Starting visual-inertial alignment...");
    ROS_INFO("Window frames: %d", (int)all_image_frame.size());
    // solve scale
    // æƒ¯æ€§åˆå§‹åŒ–ï¼Œæ±‚è§£biaså’Œé‡åŠ›æ–¹å‘ï¼Œå®Œæˆè§†è§‰-æƒ¯æ€§å¯¹é½
    // é€è¿›è¿™ä¸ªå‡½æ•°æ˜¯estimatoræ»‘çª—ä¸­çš„é™€èžºé›¶åã€é‡åŠ›å¼•ç”¨ï¼Œä»¥åŠç”¨äºŽå­˜æ”¾å°ºåº¦å› å­çš„x
    // è¿™é‡Œé¢éƒ½æ˜¯æž„é€ æ®‹å·®é¡¹ï¼Œé€šè¿‡ä»¤error=0æž„é€ Hx=bçº¿æ€§æ–¹ç¨‹ï¼Œåˆ©ç”¨ä¹”é‡Œæ–¯åŸºåˆ†è§£ç›´æŽ¥æ±‚è§£çº¿æ€§æ–¹ç¨‹ï¼Œä¸æ˜¯é€šè¿‡éžçº¿æ€§ä¼˜åŒ–æ±‚è§£çš„
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    ROS_INFO("Visual-inertial alignment solved, scale: %.6f", (x.tail<1>())(0));

    // change state
    // æ›´æ–°çŠ¶æ€
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }
    double s = (x.tail<1>())(0);
    // æ ¹æ®æ–°çš„biasé‡æ–°ç§¯åˆ†
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // æ ¹æ®æ–°çš„å°ºåº¦å› å­é‡æ–°æ”¾ç¼©å¹³ç§»
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    // æ›´æ–°é€Ÿåº¦
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    ROS_INFO("Gravity direction: [%.6f, %.6f, %.6f]", g.x(), g.y(), g.z());
    ROS_INFO("Gravity norm: %.6f", g.norm());

    return true;
}

/// @brief æ‰¾æ»‘çª—å†…åŒ¹é…è¶³å¤Ÿå¤šçš„ä¸€ä¸ªåŽ†å²å¸§ï¼Œå¹¶è®¡ç®—å…¶ä¸Žæœ€æ–°ä¸€å¸§çš„ç›¸å¯¹ä½å§¿
/// @param relative_R
/// @param relative_T
/// @param l
/// @return
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE); // èŽ·å–çª—å£é‡Œç¬¬iå¸§å’Œæœ€åŽä¸€å¸§ä¹‹é—´çš„ç‰¹å¾å…³è”
        if (corres.size() > 20)                              // è¦æ˜¯å¤§äºŽ20ä¸ªåŒ¹é…ï¼Œå°±æŽ¥ç€è¿›è¡Œè§†å·®å¤§å°çš„åˆ¤æ–­
        {
            double sum_parallax = 0;
            double average_parallax;
            // ç®—ä¸€ä¸‹ç‰¹å¾çš„å¹³å‡è§†å·®å¤§å°
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // å¹³å‡è§†å·®å¤§äºŽä¸€å®šæ•°å€¼ï¼Œä½¿ç”¨motion_estimatoræ±‚è§£ç¬¬iå¸§å’Œç¬¬WINDOW_SIZEå¸§(å°±æ˜¯æ»‘çª—æœ€åŽä¸€å¸§)ä¹‹é—´çš„ä½å§¿å˜æ¢
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i; // è®¡ç®—æˆåŠŸäº†ï¼
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if (USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if (USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                         para_Pose[0][3],
                                                         para_Pose[0][4],
                                                         para_Pose[0][5])
                                                 .toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        // TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5])
                                   .toRotationMatrix()
                                   .transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                        para_Pose[i][1] - para_Pose[0][1],
                                        para_Pose[i][2] - para_Pose[0][2]) +
                    origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                              para_SpeedBias[i][4],
                              para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                              para_SpeedBias[i][7],
                              para_SpeedBias[i][8]);
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if (USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5])
                         .normalized()
                         .toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if (USE_IMU)
        td = para_Td[0][0];
}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        // return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        // ROS_INFO(" big translation");
        // return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        // ROS_INFO(" big z translation");
        // return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        // return true;
    }
    return false;
}

/// @brief ä¼˜åŒ–çŠ¶æ€é‡
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    // loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    // loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    // ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if (USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if (!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            // ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            // ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if (USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if (STEREO && it_per_frame.is_stereo)
            {
                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    // printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    // printf("solver costs: %f \n", t_solver.toc());

    double2vector();
    // printf("frame_count: %d \n", frame_count);

    if (frame_count < WINDOW_SIZE)
        return;

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if (USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if (STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if (imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if (USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    // printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    // printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if (USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if (USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if (USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}
/// @brief é¢„æµ‹å½“å‰å¸§ç‰¹å¾ç‚¹åœ¨ä¸‹ä¸€å¸§çš„ä½ç½®
void Estimator::predictPtsInNextFrame()
{
    // printf("predict pts in next frame\n");
    if (frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);                   // å°†CurTèµ‹äºˆå½“å‰ä½å§¿ï¼Œå³ç¬¬[frame_count]ä¸ªä½å§¿
    getPoseInWorldFrame(frame_count - 1, prevT); // å°†prevTèµ‹äºˆä¸Šä¸€å¸§ä½å§¿ï¼Œå³ç¬¬frame_count-1ä¸ªä½å§¿
    nextT = curT * (prevT.inverse() * curT);     // åˆ©ç”¨ä¸Šä¸¤å¸§ä¹‹é—´çš„ç›¸å¯¹ä½å§¿é¢„æµ‹ä¸‹ä¸€å¸§çš„ä½å§¿
    map<int, Eigen::Vector3d> predictPts;        // å­˜å‚¨é¢„æµ‹çš„ç‰¹å¾ç‚¹

    for (auto &it_per_id : f_manager.feature) // éåŽ†ç‰¹å¾ç®¡ç†ä¸­çš„æ¯ä¸ªç‰¹å¾ç‚¹
    {
        if (it_per_id.estimated_depth > 0) // ç‰¹å¾ç‚¹æ·±åº¦å¤§äºŽ0æ‰èƒ½ç”¨
        {
            int firstIndex = it_per_id.start_frame;                                         // å‘çŽ°è¯¥ç‰¹å¾ç‚¹çš„ç¬¬ä¸€å¸§çš„ç´¢å¼•
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1; // å‘çŽ°è¯¥ç‰¹å¾ç‚¹çš„æœ€åŽä¸€å¸§çš„ç´¢å¼•
            // printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if ((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count) // åªæœ‰å½“è¯¥ç‰¹å¾è¢«è§‚æµ‹åˆ°è¶…è¿‡ä¸¤æ¬¡ï¼Œä¸”æœ€åŽä¸€æ¬¡å°±æ˜¯åœ¨å½“å‰å¸§è¢«è§‚æµ‹åˆ°
            {
                double depth = it_per_id.estimated_depth;                                                     // ä¼°è®¡çš„æ·±åº¦æ˜¯åœ¨çœ‹åˆ°çš„ç¬¬ä¸€å¸§ç›¸æœºç³»ä¸‹çš„
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];            // å°†ç›¸æœºç³»ä¸‹çš„è·¯æ ‡ç‚¹è½¬æ¢åˆ°è½½ä½“ç³»ä¸‹ï¼ˆç¬¬ä¸€å¸§æ—¶çš„ï¼‰
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];                                     // åˆ©ç”¨çœ‹åˆ°è¿™ä¸ªç‚¹çš„ç¬¬ä¸€å¸§æ—¶çš„è½½ä½“ç³»ä½å§¿ï¼Œå°†è¯¥ç‚¹ä»Žè½½ä½“ç³»è½¬æ¢è‡³ä¸–ç•Œç³»ä¸‹
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3)); // åˆ©ç”¨é¢„æµ‹çš„ä¸‹ä¸€å¸§ä½å§¿ï¼Œå°†è·¯æ ‡ç‚¹è½¬æ¢è‡³é¢„æµ‹çš„ä¸‹ä¸€å¸§è½½ä½“ç³»ä¸‹
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);                                 // å°†ä¸‹ä¸€å¸§è½½ä½“ç³»ä¸‹çš„è·¯æ ‡ç‚¹è½¬æ¢åˆ°ç›¸æœºç³»ä¸‹
                int ptsIndex = it_per_id.feature_id;                                                          // èŽ·å–ç‰¹å¾ç‚¹çš„ID
                predictPts[ptsIndex] = pts_cam;                                                               // å°†é¢„æµ‹çš„è·¯æ ‡ç‚¹å­˜å…¥predicPtsï¼ŒID-ç›¸æœºç³»ä¸‹ç‚¹åæ ‡
            }
        }
    }
    featureTracker.setPrediction(predictPts); // å°†é¢„æµ‹ç‰¹å¾ç‚¹ä¼ ç»™featureTracker
    // printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                    Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    // return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                     Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                     depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if (STEREO && it_per_frame.is_stereo)
            {

                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }
        double ave_err = err / errCnt;
        if (ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);
    }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while (!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}