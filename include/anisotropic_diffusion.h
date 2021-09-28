#pragma once

#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

void opFirstDerivative(const cv::Mat &image, int x, int y, int width, int height, float &dx, float &dy);
void firstDerivativeOmp(const cv::Mat &image, cv::Mat &image_dx, cv::Mat &image_dy);
void opSecondDerivative(const cv::Mat &image, int x, int y, int width, int height, float &dxx, float &dxy, float &dyy);
void secondDerivativeOmp(const cv::Mat &image, cv::Mat &image_dxx, cv::Mat &image_dxy, cv::Mat &image_dyy);
cv::Mat derivative_d2I_d2xi(const cv::Mat &image, const cv::Mat &image_x, const cv::Mat &image_y, const cv::Mat &image_xx, const cv::Mat &image_xy, const cv::Mat &image_yy, float epsilon = 10e-8);
cv::Mat derivative_d2I_d2eta(const cv::Mat &image, const cv::Mat &image_x, const cv::Mat &image_y, const cv::Mat &image_xx, const cv::Mat &image_xy, const cv::Mat &image_yy, float epsilon = 10e-8);
cv::Mat update(cv::Mat &image_ad, const cv::Mat &image_d2xi, const cv::Mat &image_d2eta, float delta_t, float c);
cv::Mat anisotropicDiffusionOMP(const cv::Mat &binary_mask, const float &delta_t = 0.05, const float &c = 0.5, const int &n_iter = 1000);
std::vector<cv::Mat> gradient(const cv::Mat &binary_mask);
std::vector<cv::Mat> gradientSecond(const cv::Mat &binary_mask);