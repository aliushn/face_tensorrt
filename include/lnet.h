//
// Created by dylee on 2020-03-22.
//
#pragma once

#ifndef FACE_TENSORRT_LNET_H
#define FACE_TENSORRT_LNET_H

#endif //FACE_TENSORRT_LNET_H

#include "util.h"
#include "ocrCommon.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"

//#include "parserOnnxConfig.h"

#include "NvCaffeParser.h"

#include "NvInfer.h"
#include <iostream>
#include <string>
#include <fstream>

#include <memory>
#include "logger.h"




#define PI                              3.141592
#define MAX_WARP_CORNER                 100
#define MAX_CHEKER_STATE                8
#define LANDMARK_POINT                  4
#define CARD_WIDTH                      720
#define CARD_HEIGHT                     454
#define CARD_CHECKER_PARAM              1
#define SAVE_IMAGE                      0
#define PRINTgLogInfo_LOG                       1

using namespace std;

class lnet {

public:
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
    lnet(const samplesCommon::FaceDetecNetParams& params) : mParams(params), mEngine(nullptr)
    {
    }
    ~lnet();

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    bool infer();

//	ncnn::Net m_LandmarkNet;
//	ncnn::Net m_ConditionCheckNet;
//	ncnn::Net m_CardCheckNet;
//    cv::CascadeClassifier m_FaceDetector;

//    int m_ImageFormat;          // UNKNOWN(0), YUV_420_888(1), NV21(2), YV12(3), YUV_422_888(4)
    // YUV_444_888(5), NV16(6), YUY2(7)
//    int m_ImageWidth;
//    int m_ImageHeight;

//    float m_StableRatio1;
//    int m_StableCount;
//    float m_StableRatio3;
//    float m_GuidPosition;
//    int m_LandmarkInputWidth, m_LandmarkInputHeight;
//    int m_CardWidth, m_CardHeight;
//    int m_PrevLocationY;
//    float m_CheckerResultState[8];
//    int m_CardCheckerFlag;
//    int m_CardUncheckFlag;
//    int m_FaceRocation;
//    int m_CurrentLandmark[8];
//
//    cv::Mat m_CropImage;
//    cv::Mat m_OriginImage;
//    cv::Mat m_CropGrayImage;
//
//    std::vector<cv::Point2f> *m_WarpCorners;
//    std::vector<cv::Point2f> *m_StabilizedCorner;
//    cv::Size m_WarpSize;
//    cv::Size m_OriginSize;

//    int m_CardType;

    int Init(int imageFormat, int imageWidth, int imageHeight, int rotateDegree, float imageMargin,
             float stableRatio1, int stableCount, std::string modelsPath);
    int Clear();

    // MakeFacfeatureofBlacklist
//    void makeBlackListFeatures(string path);

    // Image Capture Functions
//    int ImageCapture(cv::Mat input, int imageWidth, int imageHeight, int rotateDegree, float guidPoisition);
//    int DeepCardCheckerFirst(cv::Mat input);

    //int IsCardCheckerSecond(cv::Mat input);
    void OCRReset();
    bool exists_test(const std::string& name);

private:
    samplesCommon::FaceDetecNetParams mParams; //!< The parameters for the sample.
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutput1Dims; //!< The dimensions of the output to the network.
    nvinfer1::Dims mOutput2Dims; //!< The dimensions of the output to the network.
//    nvinfer1::Dims mOutput3Dims; //!< The dimensions of the output to the network.
//    nvinfer1::Dims mOutput4Dims; //!< The dimensions of the output to the network.

    int NbOutputsOnet; //!No of Pnet outputs
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network


    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                          SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                          SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};