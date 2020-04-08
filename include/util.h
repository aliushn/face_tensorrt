//
// Created by dylee on 2020-03-19.
//
#pragma once
#ifndef FACE_TENSORRT_UTIL_H
#define FACE_TENSORRT_UTIL_H

#endif //FACE_TENSORRT_UTIL_H

#include <iostream>
#include <experimental/filesystem>
#include <vector>
//#include <io.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>
#include <istream>
#include <opencv2/opencv.hpp>


//std::vector<std::string>;

//(std::string folder);
std::vector<std::string> getImageFileListFromFolder(std::string folder);
void ChangeExt(char *path, char *newext);
std::vector<float> read_csv(int row, const char *filename);
std::vector<std::string> csv_read_row(std::istream &file, char delimiter);







