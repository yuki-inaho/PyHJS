#ifndef PYHJS_INCLUDE_HOMOTOPY_PRUNING_H_
#define PYHJS_INCLUDE_HOMOTOPY_PRUNING_H_

#include <opencv2/opencv.hpp>


class InscribedCircle {
   public:
    InscribedCircle(const int32_t& center_x, const int32_t& center_y, const float& radius)
        : m_center_x_(center_x), m_center_y_(center_y), m_is_sprious_(true) {
        m_radius_ = std::max(static_cast<int32_t>(radius), 1);
    };

    /*
    Search 2-boundary points touching inscribed circles
    */
    void searchTouchingPoints(const cv::Mat& contour_mask, int32_t margin = 3) {
        int32_t image_width = contour_mask.cols;
        int32_t image_height = contour_mask.rows;

        int32_t search_roi_start_x = std::max(m_center_x_ - m_radius_ - margin, 0);
        int32_t search_roi_start_y = std::max(m_center_y_ - m_radius_ - margin, 0);
        int32_t search_roi_end_x = std::min(m_center_x_ + m_radius_ + margin, image_width - 1);
        int32_t search_roi_end_y = std::min(m_center_y_ + m_radius_ + margin, image_height - 1);

        /// get touching point candidate
        std::vector<cv::Point> touching_point_list;
        for (int32_t y = search_roi_start_y; y < search_roi_end_y; y++) {
            for (int32_t x = search_roi_start_x; x < search_roi_end_x; x++) {
                if (contour_mask.at<float>(y, x) == 0) continue;
                float dist_to_point = std::sqrt((x - m_center_x_) * (x - m_center_x_) + (y - m_center_y_) * (y - m_center_y_));
                if (dist_to_point < m_radius_) continue;
                float radial_diff = std::abs(dist_to_point - float(m_radius_));
                if (radial_diff <= 1.414 * (1 + margin)) touching_point_list.push_back(cv::Point(x, y));
            }
        }

        if (touching_point_list.size() < 2) return;

        float max_arc_angle_point_pair = -1.0;
        std::pair<int32_t, int32_t> argmax_arc_angle_point_pair(-1, -1);
        int32_t n_point_candidate = touching_point_list.size();

        for (int32_t candidate_index = 0; candidate_index < n_point_candidate; candidate_index++) {
            for (int32_t candidate_index_compare = 0; candidate_index_compare < n_point_candidate; candidate_index_compare++) {
                if (candidate_index <= candidate_index_compare) continue;
                cv::Point point_a = touching_point_list[candidate_index];
                cv::Point point_b = touching_point_list[candidate_index_compare];

                std::vector<float> vector_center_to_a{float(point_a.x - m_center_x_), float(point_a.y - m_center_y_)};
                float vector_center_to_a_norm =
                    std::sqrt(vector_center_to_a[0] * vector_center_to_a[0] + vector_center_to_a[1] * vector_center_to_a[1]);
                vector_center_to_a[0] /= vector_center_to_a_norm;
                vector_center_to_a[1] /= vector_center_to_a_norm;
                std::vector<float> vector_center_to_b{float(point_b.x - m_center_x_), float(point_b.y - m_center_y_)};
                float vector_center_to_b_norm =
                    std::sqrt(vector_center_to_b[0] * vector_center_to_b[0] + vector_center_to_b[1] * vector_center_to_b[1]);
                vector_center_to_b[0] /= vector_center_to_b_norm;
                vector_center_to_b[1] /= vector_center_to_b_norm;

                float arc_angle = std::abs(std::acos(vector_center_to_a[0] * vector_center_to_b[0] + vector_center_to_a[1] * vector_center_to_b[1]));

                if (max_arc_angle_point_pair <= arc_angle) {
                    argmax_arc_angle_point_pair = std::pair<int32_t, int32_t>{candidate_index, candidate_index_compare};
                    max_arc_angle_point_pair = arc_angle;
                }
            }
        }

        boundary_point_touch_inscribed_circle_a = touching_point_list[argmax_arc_angle_point_pair.first];
        boundary_point_touch_inscribed_circle_b = touching_point_list[argmax_arc_angle_point_pair.second];
        m_arc_angle_inscribed_circle_ = max_arc_angle_point_pair;

        /*
        std::cout << max_arc_angle_point_pair << " " << m_is_sprious_ << ", (" << m_center_x_ << ", " << m_center_y_ << "), "
                  << "(" << boundary_point_touch_inscribed_circle_a.x << ", " << boundary_point_touch_inscribed_circle_a.y << "), "
                  << "(" << boundary_point_touch_inscribed_circle_b.x << ", " << boundary_point_touch_inscribed_circle_b.y << "), " << std::endl;
        */
    }

    void centers(int32_t& center_x, int32_t& center_y) const {
        center_x = m_center_x_;
        center_y = m_center_y_;
    }

    void set_spriousness(const bool& spriousness) 
    {
        m_is_sprious_ = spriousness;
    }

    bool is_sprious() const { return m_is_sprious_; }

    /*
    float bending_potential_ratio() const {
        float m_center_x_f = static_cast<float>(m_center_x_);
        float m_center_y_f = static_cast<float>(m_center_y_);
        float bax_f = float(boundary_point_touch_inscribed_circle_a.x);
        float bay_f = float(boundary_point_touch_inscribed_circle_a.y);
        float bbx_f = float(boundary_point_touch_inscribed_circle_b.x);
        float bby_f = float(boundary_point_touch_inscribed_circle_b.y);
        float diff_center2a = std::sqrt((bax_f - m_center_x_f) * (bax_f - m_center_x_f) + (bay_f - m_center_x_f) * (bay_f - m_center_x_f));
        float diff_center2b = std::sqrt((bbx_f - m_center_x_f) * (bbx_f - m_center_x_f) + (bby_f - m_center_x_f) * (bby_f - m_center_x_f));
        float diff_a2b = std::sqrt((bax_f - bbx_f) * (bax_f - bbx_f) + (bay_f - bby_f) * (bay_f - bby_f));

        float lab = m_arc_angle_inscribed_circle_ * std::max(diff_center2a, diff_center2a);
        float hg = 0.5 * std::sqrt(lab * lab - diff_a2b * diff_a2b);
        float hp = diff_center2a * diff_center2b * std::abs(std::sin(m_arc_angle_inscribed_circle_)) / diff_a2b;
        // float hp = std::max(diff_center2a, diff_center2b) * std::cos(m_arc_angle_inscribed_circle_ / 2);

        float potential = std::abs(std::tan(m_arc_angle_inscribed_circle_ / 2)) * std::sqrt(std::max(lab * lab / (diff_a2b * diff_a2b) - 1.0, 0.0));
        return potential;
    }
    */

    /*
    the unit of return value is degree
    */
    float arc_angle_inscribed_points() const { return m_arc_angle_inscribed_circle_ / M_PI * 180; }

   private:
    int32_t m_center_x_, m_center_y_;
    int32_t m_radius_;
    float m_arc_angle_inscribed_circle_;
    cv::Point boundary_point_touch_inscribed_circle_a;
    cv::Point boundary_point_touch_inscribed_circle_b;
    bool m_is_sprious_;
};

class PruningSkeleton {
   public:
    PruningSkeleton(const float& threshold_angle_inscribed_arc) : m_threshold_angle_inscribed_arc_(threshold_angle_inscribed_arc){};

    void setImages(const cv::Mat& skeleton_image, const cv::Mat& distance_transform_image, const cv::Mat& contour_mask) {
        m_skeleton_image_ = skeleton_image.clone();
        m_distance_transform_image_ = distance_transform_image.clone();
        m_contour_mask_ = contour_mask.clone();
    }

    void setInscribedCircles() {
        CV_Assert(m_skeleton_image_.type() == CV_32F);
        CV_Assert(m_distance_transform_image_.type() == CV_32F);
        int32_t image_width = m_skeleton_image_.cols;
        int32_t image_height = m_skeleton_image_.rows;

        /// Set inscribed circles on medial axis
        for (int32_t y = 1; y < image_height - 1; y++) {
            for (int32_t x = 1; x < image_width - 1; x++) {
                if (m_skeleton_image_.at<float>(y, x) > 0)
                    m_inscribed_circles_.push_back(InscribedCircle(x, y, m_distance_transform_image_.at<float>(y, x)));
            }
        }

        for (auto it_inscribed_circle = m_inscribed_circles_.begin(); it_inscribed_circle != m_inscribed_circles_.end(); ++it_inscribed_circle) {
            it_inscribed_circle->searchTouchingPoints(m_contour_mask_);
            if(it_inscribed_circle->arc_angle_inscribed_points() >= m_threshold_angle_inscribed_arc_)
                it_inscribed_circle->set_spriousness(false);
        }
    }

    cv::Mat getPrunedSkeleton() {
        cv::Mat skeleton_image_pruned = cv::Mat::zeros(cv::Size(m_skeleton_image_.cols, m_skeleton_image_.rows), CV_32F);
        for (auto it_inscribed_circle = m_inscribed_circles_.begin(); it_inscribed_circle != m_inscribed_circles_.end(); ++it_inscribed_circle) {
            int32_t center_x, center_y;
            it_inscribed_circle->centers(center_x, center_y);
            if (it_inscribed_circle->is_sprious()) {
                skeleton_image_pruned.at<float>(center_y, center_x) = 0;
            } else {
                skeleton_image_pruned.at<float>(center_y, center_x) = 1.0;
            }
        }
        return skeleton_image_pruned;
    }

   private:
    cv::Mat m_skeleton_image_;
    cv::Mat m_distance_transform_image_;
    cv::Mat m_contour_mask_;

    std::vector<InscribedCircle> m_inscribed_circles_;
    float m_threshold_angle_inscribed_arc_;
};

#endif
