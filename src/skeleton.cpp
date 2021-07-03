#include <opencv2/opencv.hpp>
#include "skeleton.h"

/*
Compute the average outward flux
*/
void flux(const cv::Mat &Dx, const cv::Mat &Dy, cv::Mat &F)
{
  CV_Assert(F.type() == CV_32F);
  int32_t width = F.cols;
  int32_t height = F.rows;
  for (int32_t y = 1; y < height - 1; y++)
  {
    for (int32_t x = 1; x < width - 1; x++)
    {
      float flux_var = 0;
      for (int32_t ky = -1; ky <= 1; ky++)
      {
        for (int32_t kx = -1; kx <= 1; kx++)
        {
          if (kx == 0 && ky == 0)
            continue;
          float normal_norm = std::sqrt(kx * kx + ky * ky);
          float nx = float(kx) / normal_norm;
          float ny = float(ky) / normal_norm;
          flux_var += Dx.at<float>(y + ky, x + kx) * nx + Dy.at<float>(y + ky, x + kx) * ny;
        }
      }
      F.at<float>(y, x) = flux_var/8.0;
    }
  }
}

std::vector<cv::Point> getContourPoints(const cv::Mat &mask_image)
{
    std::vector<std::vector<cv::Point>> contours_list;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask_image, contours_list, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    std::vector<cv::Point> contour_points_concatenated;
    for (std::vector<cv::Point> contours : contours_list)
    {
        std::copy(contours.begin(), contours.end(), std::back_inserter(contour_points_concatenated));
    }
    return contour_points_concatenated;
}
