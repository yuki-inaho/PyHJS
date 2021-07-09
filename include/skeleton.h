#ifndef PYHJS_INCLUDE_SKELETON_H_
#define PYHJS_INCLUDE_SKELETON_H_

#include <cmath>
#include <opencv2/opencv.hpp>

void flux(const cv::Mat &Dx, const cv::Mat &Dy, cv::Mat &F);
std::vector<cv::Point> getContourPoints(const cv::Mat &mask_image);
cv::Mat getContourMask(const cv::Mat &mask_image);

#endif