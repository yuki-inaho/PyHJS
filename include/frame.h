#ifndef PYHJS_INCLUDE_FRAME_H_
#define PYHJS_INCLUDE_FRAME_H_

#include <iostream>
#include <opencv2/core/core.hpp>

class BinaryFrame
{
public:
    BinaryFrame(const cv::Mat &binary_image)
    {
        SetBinaryImage(binary_image);
    }

    cv::Mat cvmat, cvmat_f;
    std::vector<float> data;
    size_t image_width, image_height;
    unsigned char max_value, min_value;

private:
    void SetBinaryImage(const cv::Mat &binary_image)
    {
        cvmat = binary_image.clone();
        cvmat.convertTo(cvmat_f, CV_32F);

        TypeValidationBinaryImage(binary_image);
        binary_image_ = binary_image;
        image_width = binary_image.cols;
        image_height = binary_image.rows;

        double max_value_d, min_value_d;
        cv::minMaxLoc(binary_image, &min_value_d, &max_value_d);
        min_value = static_cast<unsigned char>(min_value_d);
        max_value = static_cast<unsigned char>(max_value_d);

        std::vector<float> data_(image_width * image_height, 0);
        for (size_t v = 0; v < image_height; v++)
        {
            for (size_t u = 0; u < image_width; u++)
            {
                size_t k = v * image_width + u;
                data_[k] = static_cast<float>(binary_image.at<unsigned char>(v, u));
            }
        }
        data = data_;
    }

    void TypeValidationBinaryImage(const cv::Mat binary_image)
    {
        if (binary_image.type() != CV_8UC1)
        {
            std::cerr << "binary_image.channels() != CV_8UC1" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    cv::Mat binary_image_;
};

#endif