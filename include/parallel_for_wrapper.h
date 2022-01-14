#pragma once
#include <omp.h>
#include <thread>
#include <opencv2/opencv.hpp>

/*
    On OpenCV 3.2.0, cv::parallel_for_ combined with lambda function is not supported (occurring below error).
    If version of the library is greater than 3.2.0.
    "parallel_for_omp" can be replaced by cv::parallel_for_ without loss of lambda function calling.

    /usr/local/include/opencv2/core/utility.hpp:478:75: note: in passing argument 2 of ‘void cv::parallel_for_(const cv::Range&, const cv::ParallelLoopBody&, double)’
    478 | CV_EXPORTS void parallel_for_(const Range& range, const ParallelLoopBody& body, double nstripes=-1.);
*/
template <class BODY>
void parallel_for_omp(const cv::Range& range, BODY body)
{
    #pragma omp parallel for
    for (int i = range.start; i < range.end; ++i)
        body(cv::Range(i, i + 1));
}