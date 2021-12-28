#ifndef PYHJS_INCLUDE_HJS_H_
#define PYHJS_INCLUDE_HJS_H_

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "frame.h"
#include "pruning.h"
#include "skeleton.h"
#include "thinning.h"
#include "anisotropic_diffusion.h"

class HamiltonJacobiSkeleton
{
public:
    HamiltonJacobiSkeleton(float gamma, float epsilon, float threshold_arc_angle_inscribed_circle = 0)
        : gamma_(gamma), epsilon_(epsilon), threshold_arc_angle_inscribed_circle_(threshold_arc_angle_inscribed_circle){};
    ~HamiltonJacobiSkeleton(){};

    void compute(const BinaryFrame &frame, bool enable_anisotropic_diffusion = true)
    {
        /* normalize and copy the images */
        double min_frame_value, max_frame_value;
        cv::Mat L_mat = frame.cvmat_f.clone();
        cv::minMaxLoc(L_mat, &min_frame_value, &max_frame_value);
        cv::normalize(L_mat, L_mat, 1, 0, cv::NORM_MINMAX);
        cv::Mat L_mat_raw = L_mat.clone();
        cv::threshold(L_mat, L_mat, 0.5, 1.0, cv::THRESH_BINARY_INV);

        /* compute the distance function inside the silhouette */
        cv::Mat D_mat;
        cv::distanceTransform(frame.cvmat, D_mat, cv::DIST_L2, 3);
        std::vector<cv::Point> contour_points = getContourPoints(frame.cvmat);

        cv::Mat F_mat;
        if (enable_anisotropic_diffusion)
        {
            /*
            Generate skeleton with anisotropic diffusion
            (the skeleton is less likely to generate sprious skeleton. But it doesn't have completely thinned structure.)
            */
            cv::Mat skeleton_image_ad;
            cv::Mat D_mat_ad = anisotropicDiffusionOMP(D_mat, 0.05, 0.2, 50);
            getSkeletonFromSlopyImage(D_mat_ad, L_mat, frame.image_width, frame.image_height, skeleton_image_ad, F_mat, contour_points);

            /* Get thinned skeleton combined with two skeletons */
            cv::dilate(skeleton_image_ad, skeleton_image_ad, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            getSkeletonFromSlopyImage(D_mat, L_mat, frame.image_width, frame.image_height, skeleton_image_, F_mat, contour_points);
            skeleton_image_ = skeleton_image_ & skeleton_image_ad;
        }
        else
        {
            getSkeletonFromSlopyImage(D_mat, L_mat, frame.image_width, frame.image_height, skeleton_image_, F_mat, contour_points);
        }

        cv::Mat contour_mask = getContourMask(frame.cvmat);
        PruningSkeleton pruning = PruningSkeleton(threshold_arc_angle_inscribed_circle_);
        pruning.setImages(skeleton_image_, D_mat, contour_mask);
        pruning.setInscribedCircles();
        skeleton_image_ = pruning.getPrunedSkeleton();

        distance_transform_image_ = D_mat;
        flux_image_ = F_mat;
    }

    void setParameters(const float &gamma, const float &epsilon, float threshold_arc_angle_inscribed_circle = 0)
    {
        gamma_ = gamma;
        epsilon_ = epsilon;
        if (threshold_arc_angle_inscribed_circle > 0)
        {
            threshold_arc_angle_inscribed_circle_ = threshold_arc_angle_inscribed_circle;
        }
    }

    cv::Mat getSkeletonImage() { return skeleton_image_.clone(); }

    cv::Mat getDistanceTransformImage() { return distance_transform_image_.clone(); }

    cv::Mat getFluxImage() { return flux_image_.clone(); }

private:
    void getSkeletonFromSlopyImage(const cv::Mat &D_mat, const cv::Mat &L_mat, int image_width, int image_height, cv::Mat &skeleton_mat, cv::Mat &F_mat, std::vector<cv::Point> &contour_points)
    {
        /* compute the gradient */
        cv::Mat Dx_mat, Dy_mat;
        cv::Sobel(D_mat, Dx_mat, CV_32F, 1, 0);
        cv::Sobel(D_mat, Dy_mat, CV_32F, 0, 1);

        /* flux computation */
        F_mat = cv::Mat::zeros(cv::Size(image_width, image_height), CV_32F);
        flux(Dx_mat, Dy_mat, F_mat);

        /* homotopy preserved thinning */
        double F_max, F_min;
        cv::minMaxLoc(F_mat, &F_min, &F_max);
        float flux_threshold = F_min / gamma_;

        HomotopyPreservingThinning thinning = HomotopyPreservingThinning(flux_threshold);
        thinning.setImages(L_mat, D_mat, F_mat);
        thinning.setContourPoints(contour_points);
        thinning.compute();

        /* store results */
        skeleton_mat = thinning.getSkeletonImage().clone();
    }

    cv::Mat distance_transform_image_;
    cv::Mat flux_image_;
    cv::Mat skeleton_image_;

    float threshold_arc_angle_inscribed_circle_;
    float gamma_, epsilon_;
};

#endif