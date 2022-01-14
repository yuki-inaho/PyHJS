#include "anisotropic_diffusion.h"

void opFirstDerivative(const cv::Mat &image, int x, int y, int width, int height, float &dx, float &dy)
{
    // dI/dx

    if (x > 0)
    {
        dx = (x < width - 1) ? (image.at<float>(y, x + 1) - image.at<float>(y, x - 1)) / 2.0 : 0.0;
    }
    else
    {
        dx = 0.0;
    }

    // dI/dy
    if (y > 0)
    {
        dy = (y < height - 1) ? (image.at<float>(y + 1, x) - image.at<float>(y - 1, x)) / 2.0 : 0.0;
    }
    else
    {
        dy = 0.0;
    }
}

void firstDerivativeOmp(const cv::Mat &image, cv::Mat &image_dx, cv::Mat &image_dy)
{
    int width = image.cols;
    int height = image.rows;

    image_dx = cv::Mat::zeros(image.size(), CV_32FC1);
    image_dy = cv::Mat::zeros(image.size(), CV_32FC1);
    parallel_for_omp(cv::Range(0, width * height), [&](const cv::Range &range)
                      {
                          for (int r = range.start; r < range.end; r++)
                          {
                              int y = r / width;
                              int x = r % width;
                              float dx, dy;
                              opFirstDerivative(image, x, y, width, height, dx, dy);
                              image_dx.at<float>(y, x) = dx;
                              image_dy.at<float>(y, x) = dy;
                          }
                      });
}

void opSecondDerivative(const cv::Mat &image, int x, int y, int width, int height, float &dxx, float &dxy, float &dyy)
{
    // d^2I/dxdx
    if (x > 0)
    {
        dxx = (x < width - 1) ? (image.at<float>(y, x + 1) - 2 * image.at<float>(y, x) + image.at<float>(y, x - 1)) : 0.0;
    }
    else
    {
        dxx = 0.0;
    }

    // d^2I/dxdy
    if ((y > 0) & (x > 0))
    {
        dxy = ((y < height - 1) & (x < width - 1)) ? (image.at<float>(y + 1, x + 1) - image.at<float>(y + 1, x - 1) - image.at<float>(y - 1, x + 1) + image.at<float>(y - 1, x - 1)) / 4.0 : 0;
    }
    else
    {
        dxy = 0.0;
    }

    // d^2I/dydy
    if (y > 0)
    {
        dyy = (y < height - 1) ? (image.at<float>(y + 1, x) - 2 * image.at<float>(y, x) + image.at<float>(y - 1, x)) : 0.0;
    }
    else
    {
        dyy = 0.0;
    }
}

void secondDerivativeOmp(const cv::Mat &image, cv::Mat &image_dxx, cv::Mat &image_dxy, cv::Mat &image_dyy)
{
    int width = image.cols;
    int height = image.rows;

    image_dxx = cv::Mat::zeros(image.size(), CV_32FC1);
    image_dxy = cv::Mat::zeros(image.size(), CV_32FC1);
    image_dyy = cv::Mat::zeros(image.size(), CV_32FC1);
    parallel_for_omp(cv::Range(0, width * height), [&](const cv::Range &range)
                      {
                          for (int r = range.start; r < range.end; r++)
                          {
                              int y = r / width;
                              int x = r % width;
                              float dxx, dxy, dyy;
                              /*
                              if (image.at<float>(y, x) == 0)
                                  continue;
                            */
                              opSecondDerivative(image, x, y, width, height, dxx, dxy, dyy);
                              image_dxx.at<float>(y, x) = dxx;
                              image_dxy.at<float>(y, x) = dxy;
                              image_dyy.at<float>(y, x) = dyy;
                          }
                      });
}

cv::Mat derivative_d2I_d2xi(const cv::Mat &image, const cv::Mat &image_x, const cv::Mat &image_y, const cv::Mat &image_xx, const cv::Mat &image_xy, const cv::Mat &image_yy, float epsilon)
{
    cv::Mat image_dxixi = cv::Mat::zeros(image_x.size(), CV_32FC1);
    int width = image_x.cols;
    int height = image_x.rows;
    parallel_for_omp(cv::Range(0, width * height), [&](const cv::Range &range)
                      {
                          for (int r = range.start; r < range.end; r++)
                          {
                              int y = r / width;
                              int x = r % width;
                              /*
                              if (image.at<float>(y, x) == 0)
                                  continue;
                                */

                              float Ix = image_x.at<float>(y, x);
                              float Iy = image_y.at<float>(y, x);
                              float Ixx = image_xx.at<float>(y, x);
                              float Ixy = image_xy.at<float>(y, x);
                              float Iyy = image_yy.at<float>(y, x);

                              float Ix_pow_2 = Ix * Ix;
                              float Iy_pow_2 = Iy * Iy;

                              float denominator = Ix_pow_2 + Iy_pow_2 + epsilon;
                              float numerator = Ixx * Iy_pow_2 - 2 * Ix * Iy * Ixy + Iyy * Ix_pow_2;

                              image_dxixi.at<float>(y, x) = numerator / denominator;
                          }
                      });
    return image_dxixi;
}

cv::Mat derivative_d2I_d2eta(const cv::Mat &image, const cv::Mat &image_x, const cv::Mat &image_y, const cv::Mat &image_xx, const cv::Mat &image_xy, const cv::Mat &image_yy, float epsilon)
{
    cv::Mat image_detaeta = cv::Mat::zeros(image_x.size(), CV_32FC1);
    int width = image_x.cols;
    int height = image_x.rows;

    parallel_for_omp(cv::Range(0, width * height), [&](const cv::Range &range)
                      {
                          for (int r = range.start; r < range.end; r++)
                          {
                              int y = r / width;
                              int x = r % width;
                              /*
                              if (image.at<float>(y, x) == 0)
                                  continue;
                                */
                              float Ix = image_x.at<float>(y, x);
                              float Iy = image_y.at<float>(y, x);
                              float Ixx = image_xx.at<float>(y, x);
                              float Ixy = image_xy.at<float>(y, x);
                              float Iyy = image_yy.at<float>(y, x);

                              float Ix_pow_2 = Ix * Ix;
                              float Iy_pow_2 = Iy * Iy;

                              float denominator = Ix_pow_2 + Iy_pow_2 + epsilon;
                              float numerator = (Ixx * Iy_pow_2 + 2 * Ix * Iy * Ixy + Iyy * Ix_pow_2);

                              image_detaeta.at<float>(y, x) = numerator / denominator;
                          }
                      });

    return image_detaeta;
}

cv::Mat update(cv::Mat &image_ad, const cv::Mat &image_d2xi, const cv::Mat &image_d2eta, float delta_t, float c)
{
    int width = image_ad.cols;
    int height = image_ad.rows;
    cv::Mat image_ad_new = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);

    parallel_for_omp(cv::Range(0, width * height), [&](const cv::Range &range)
                      {
                          for (int r = range.start; r < range.end; r++)
                          {
                              int y = r / width;
                              int x = r % width;
                              if (image_ad.at<float>(y, x) == 0)
                                  continue;

                              image_ad_new.at<float>(y, x) = image_ad.at<float>(y, x) + delta_t * (image_d2xi.at<float>(y, x) + c * image_d2eta.at<float>(y, x));
                          }
                      });

    return image_ad_new;
}

float getMax(const cv::Mat &mat)
{
    float max_var = 0;
    int width = mat.cols;
    int height = mat.rows;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float cur_var = mat.at<float>(y, x);
            if (cur_var > max_var)
            {
                max_var = cur_var;
            }
        }
    }
    return max_var;
}

float maxDiff(const cv::Mat &mat1, const cv::Mat &mat2)
{
    float max_diff = 0;
    int width = mat1.cols;
    int height = mat2.rows;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float diff = mat1.at<float>(y, x) - mat2.at<float>(y, x);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}

cv::Mat anisotropicDiffusionOMP(const cv::Mat &input_image, const float &delta_t, const float &c, const int &n_iter)
{
    /* initialize image_ad */
    cv::Mat image_ad;
    input_image.copyTo(image_ad);
    for (int i = 0; i < n_iter; i++)
    {
        cv::Mat image_dx, image_dy, image_dxx, image_dxy, image_dyy, image_d2xi, image_d2eta;
        firstDerivativeOmp(image_ad, image_dx, image_dy);
        secondDerivativeOmp(image_ad, image_dxx, image_dxy, image_dyy);
        image_d2xi = derivative_d2I_d2xi(image_ad, image_dx, image_dy, image_dxx, image_dxy, image_dyy);
        image_d2eta = derivative_d2I_d2eta(image_ad, image_dx, image_dy, image_dxx, image_dxy, image_dyy);
        cv::Mat image_ad_updated = update(image_ad, image_d2xi, image_d2eta, delta_t, c);
        image_ad_updated.copyTo(image_ad);
    }
    return image_ad;
}

std::vector<cv::Mat> gradient(const cv::Mat &input_image)
{
    std::vector<cv::Mat> mat_list;
    cv::Mat image_ad = cv::Mat::zeros(input_image.size(), CV_32FC1);

    /* initialize image_ad */
    //input_image.convertTo(image_ad, CV_32FC1, 1.0);
    input_image.copyTo(image_ad);

    cv::Mat image_dx, image_dy;
    firstDerivativeOmp(image_ad, image_dx, image_dy);
    mat_list.push_back(image_dx);
    mat_list.push_back(image_dy);
    return mat_list;
}

std::vector<cv::Mat> gradientSecond(const cv::Mat &input_image)
{
    std::vector<cv::Mat> mat_list;
    cv::Mat image_ad = cv::Mat::zeros(input_image.size(), CV_32FC1);

    /* initialize image_ad */
    input_image.copyTo(image_ad);
    cv::Mat image_dxx, image_dxy, image_dyy;
    secondDerivativeOmp(image_ad, image_dxx, image_dxy, image_dyy);
    mat_list.push_back(image_dxx);
    mat_list.push_back(image_dxy);
    mat_list.push_back(image_dyy);
    return mat_list;
}