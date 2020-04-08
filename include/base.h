//
// Created by dylee on 2020-03-21.
//
#pragma once
#ifndef FACE_TENSORRT_BASE_H
#define FACE_TENSORRT_BASE_H

#endif //FACE_TENSORRT_BASE_H

#include <cmath>
#include <cstring>
#include <opencv2/opencv.hpp>

typedef struct FaceInfo {
    float score;
    int x[2];
    int y[2];
    float area;
    float regreCoord[4];
    int landmark[10];
} FaceInfo;

void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M);
void warpAffineMatrix(cv::Mat src, cv::Mat dst, float* M, int dst_W, int dst_h);