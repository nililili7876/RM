//
// Created by luojunhui on 1/28/20.
//

#include <csignal>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <memory>
#include "MyThread.hpp"

using namespace std;
using namespace cv;

namespace rm
{
    bool ImgProdCons::quitFlag = false;


    /**
     * @brief initializer of the frame buffer
     * @param size_ the length of the frame queue
     * @return none
     * @details none
     */
    FrameBuffer::FrameBuffer(int32_t size_) :
            frames(size_),
            mutexs(size_),
            tailIdx(0),
            headIdx(0),
            lastGetTimeStamp(0.0),
            size(size_)
            {
            }


    /**
     * @brief push frame into frame buffer
     * @param frame
     * @return if the frame successfully pushed into the buffer, return true
     * @details none
     */
    bool FrameBuffer::push(const Frame &frame)
    {
        int32_t newHeadIdx = (headIdx + 1) % size;

        //try for 2ms to lock
        unique_lock<timed_mutex> lock(mutexs[newHeadIdx], chrono::milliseconds(2));
        if (!lock.owns_lock())
        {
            return false;
        }

        frames[newHeadIdx] = frame;
        if (newHeadIdx == tailIdx)
        {
            tailIdx = (tailIdx + 1) % size;
        }
        headIdx = newHeadIdx;
        return true;
    }


    bool FrameBuffer::pop(Frame &frame)
    {
        //volatile const size_t _headIdx = headIdx;

        //try for 2ms to lock
        unique_lock<timed_mutex> lock(mutexs[headIdx], chrono::milliseconds(2));
        if (!lock.owns_lock() ||
            frames[headIdx].img.empty() ||
            frames[headIdx].timeStamp == lastGetTimeStamp)
        {
            perror("frame pop error!");
            return false;
        }

        frame = frames[headIdx];
        lastGetTimeStamp = frames[headIdx].timeStamp;
        return true;
    }


    States::States()
    {
        curArmorState = BIG_ARMOR;
        lastArmorState = BIG_ARMOR;

        curControlState = AUTO_SHOOT_STATE;
        lastControlState = AUTO_SHOOT_STATE;

        curFindState = FIND_ARMOR_NO;
        lastFindState = FIND_ARMOR_NO;

        curAttackMode = SEARCH_MODE;
        lastAttackMode = SEARCH_MODE;
    }


    void States::UpdateStates(int8_t armor_state, int8_t find_state, int8_t control_state, int8_t attack_mode)
    {
        //last states update
        lastArmorState = curArmorState;
        lastFindState = curFindState;
        lastControlState = curControlState;
        lastAttackMode = curAttackMode;

        //current states update
        curArmorState = armor_state;
        curFindState = find_state;
        curControlState = control_state;
        curAttackMode = attack_mode;
    }


    void ImgProdCons::SignalHandler(int)
    {
        perror("sigaction error:");
        ImgProdCons::quitFlag = true;
    }


    void ImgProdCons::InitSignals()
    {
        ImgProdCons::quitFlag = false;
        struct sigaction sigact{};
        sigact.sa_handler = SignalHandler;
        sigemptyset(&sigact.sa_mask);
        sigact.sa_flags = 0;
        /*if interrupt occurs,set _quit_flag as true*/
        sigaction(SIGINT, &sigact, (struct sigaction *)nullptr);
    }


    ImgProdCons::ImgProdCons() :
            videoCapturePtr(make_unique<VideoCapture>()),
            buffer(6), /*frame size is 6*/
            serialPtr(make_unique<Serial>()),
            solverPtr(make_unique<SolveAngle>()),
            armorDetectorPtr(make_unique<ArmorDetector>()),
            states(make_unique<States>())
    {
    }

    void ImgProdCons::Init()
    {
        /*initialize signal*/
        InitSignals();

        /*initialize camera*/
//        videoCapturePtr->RMopen(0, 2); //camera id 0 buffersize 2
//        videoCapturePtr->setVideoFormat(640, 480, 2);//1 is mjpg,2 is jepg,3 is yuyv
//        videoCapturePtr->setExposureTime(100);
//        videoCapturePtr->setFPS(200);
//        videoCapturePtr->startStream();
//        videoCapturePtr->info();
        videoCapturePtr->open(0);
    }
    void ImgProdCons::Produce()
    {
        auto startTime = chrono::high_resolution_clock::now();
        static uint32_t seq = 1;
        for (;;)
        {
            if (ImgProdCons::quitFlag)
                return;
//            if (!videoCapturePtr->grab())
//                continue;
            Mat newImg;
            double timeStamp = (static_cast<chrono::duration<double, std::milli>>(chrono::high_resolution_clock::now() -startTime)).count();
            /*write somthing, try to send serial some information,to make sure serial is working well,
            because when interruption occurs we need to restart it from the beginning*/
            videoCapturePtr->read(newImg);
            if (newImg.empty())
            {
                continue;
            }
            cout<<"push Image!"<<endl;
            buffer.push(Frame{newImg, seq, timeStamp});
            seq++;
        }
    }


    void ImgProdCons::Consume()
    {
        Frame frame;
        Mat image0;
        armorDetectorPtr->Init();
        if (!runWithCamera)
        {
            if(blueTarget)
                videoCapturePtr->open("/home/ljh/??????/Videos/LINUX_Video_0.avi");
            else
                videoCapturePtr->open("/home/ljh/??????/Videos/LINUX_Video_0.avi");
        }

        videoCapturePtr->read(image0);
        while(image0.empty())
        {
            videoCapturePtr->read(image0);
        }

        FRAMEWIDTH = image0.cols;
        FRAMEHEIGHT = image0.rows;

        do
        {
            auto startTime = chrono::high_resolution_clock::now();
            videoCapturePtr->read(image0);
//            double time = (static_cast<chrono::duration<double, std::milli>>(chrono::high_resolution_clock::now() -startTime)).count();
//            cout<<"GRAB  FREQUENCY: "<< 1000.0/time<<endl;

            if (image0.empty())
                continue;
            if (states->curControlState == BIG_ENERGY_STATE)
            {

            }
            else if (states->curControlState == SMALL_ENERGY_STATE)
            {

            }
            else if (states->curControlState == AUTO_SHOOT_STATE)
            {
                switch (states->curAttackMode)
                {
                    case SEARCH_MODE:
                    {
                        cout<<"Begin Search Mode!"<<endl;
                        if (armorDetectorPtr->ArmorDetectTask(image0))
                        {
//                            bool is_small_ = armorDetectorPtr->isSmall();
                            int8_t armorType_ = (armorDetectorPtr->IsSmall()) ? (SMALL_ARMOR) : (BIG_ARMOR);
                            solverPtr->GetPoseV(armorDetectorPtr->targetArmor.pts, 15,armorDetectorPtr->IsSmall());
                            serialPtr->pack(solverPtr->yaw, solverPtr->pitch, solverPtr->dist,solverPtr->shoot,
                                            FIND_ARMOR_YES,AUTO_SHOOT_STATE);
//                            cout<<"IF SHOOT: "<< solverPtr->shoot<<endl;
                            serialPtr->WriteData();
                            states->UpdateStates(armorType_, FIND_ARMOR_YES, AUTO_SHOOT_STATE, TRACKING_MODE);

                            armorDetectorPtr->tracker = TrackerKCF::create();
                            armorDetectorPtr->tracker->init(image0, armorDetectorPtr->targetArmor.rect);
                        }
                        else
                        {
                            serialPtr->pack(0,0,solverPtr->dist, solverPtr->shoot, FIND_ARMOR_NO,AUTO_SHOOT_STATE);
                            serialPtr->WriteData();
                            states->UpdateStates(SMALL_ARMOR, FIND_ARMOR_NO, AUTO_SHOOT_STATE, SEARCH_MODE);
                        }
                        //????????????????????????roi????????????tracking??????,???????????????tracking??????
                    }
                    break;
                    case TRACKING_MODE:
                    {
                        cout<<"Begin tracking Mode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
                        //?????????????????????trakcking??????????????????????????????????????????????????????searching,??????tracking???????????????????????????????????????
                        if (armorDetectorPtr->trackingTarget(image0, armorDetectorPtr->targetArmor.rect))
                        {
                            bool is_small_ = armorDetectorPtr->IsSmall();
                            int8_t armor_type_ = (is_small_) ? (SMALL_ARMOR) : (BIG_ARMOR);
                            solverPtr->GetPoseV(armorDetectorPtr->targetArmor.pts, 15,armorDetectorPtr->IsSmall());
                            serialPtr->pack(solverPtr->yaw, solverPtr->pitch, solverPtr->dist,solverPtr->shoot,
                                            FIND_ARMOR_YES,AUTO_SHOOT_STATE);
                            cout<<"IF SHOOT: "<< solverPtr->shoot<<endl;
                            serialPtr->WriteData();
                            states->UpdateStates(armor_type_, FIND_ARMOR_YES, AUTO_SHOOT_STATE, TRACKING_MODE);
                        }
                        else
                        {
                            serialPtr->pack(0,0,solverPtr->dist, solverPtr->shoot, FIND_ARMOR_NO,AUTO_SHOOT_STATE);
                            serialPtr->WriteData();
                            states->UpdateStates(SMALL_ARMOR, FIND_ARMOR_NO, AUTO_SHOOT_STATE, SEARCH_MODE);
                        }
                    }
                    break;
                }
            }
            if(showOrigin)
            {
                imshow("src",image0);
                waitKey(30);
            }

            auto time = (static_cast<chrono::duration<double, std::milli>>(chrono::high_resolution_clock::now() -startTime)).count();
            cout<<"FREQUENCY: "<< 1000.0/time<<endl;
        } while (!ImgProdCons::quitFlag);
    }
} // namespace rm
