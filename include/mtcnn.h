//
// Created by dylee on 2020-03-21.
//
#pragma once
#ifndef FACE_TENSORRT_MTCNN_H
#define FACE_TENSORRT_MTCNN_H

#endif //FACE_TENSORRT_MTCNN_H

#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include "base.h"



class MtcnnDetector{
public:
    MtcnnDetector();
    ~MtcnnDetector();
    std::vector<FaceInfo> Detect(cv::Mat img);
    void Init(std::string model_folder);

private:
    const float minsize=20;
    float threshold[3] = {0.6f,0.7f,0.8f};
    float factor = 0.709f;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.0078125f, 0.0078125f, 0.0078125f};

    std::vector<FaceInfo> Pnet_Detect(cv::Mat img);
    std::vector<FaceInfo> Rnet_Detect(cv::Mat img);
    std::vector<FaceInfo> Onet_Detect(cv::Mat img);

    void Lnet_Detect(cv::Mat img, std::vector<FaceInfo> &bboxs);
    vector<FaceInfo> generateBbox();
};
