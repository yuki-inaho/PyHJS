#ifndef PYHJS_INCLUDE_HOMOTOPY_THINNING_H_
#define PYHJS_INCLUDE_HOMOTOPY_THINNING_H_

#include <opencv2/opencv.hpp>
#include <map>
#include <chrono>
#include <iostream>

enum struct PointStatus;
typedef std::pair<int32_t, int32_t> PointPosition;
typedef std::map<PointPosition, PointStatus> MapPosition2Status;

enum struct PointStatus
{
    kSearching,
    kRemoved,
    kSkeletonCandidate
};

struct SkeletonPoint
{
    int32_t x, y;
    float dt; // distance transform
    SkeletonPoint(const int32_t &x_input, const int32_t &y_input, const float &dt_input) : x(x_input), y(y_input), dt(dt_input){};
};

struct FluxPoint
{
    int32_t x, y;
    float flux;
    FluxPoint();
    FluxPoint(const int32_t &x_input, const int32_t &y_input, const float &flux_input) : x(x_input), y(y_input), flux(flux_input){};
    FluxPoint clone() const
    {
        return FluxPoint(x, y, flux);
    }
    friend bool operator<(const FluxPoint &p1, const FluxPoint &p2) { return p1.flux < p2.flux; };
};

class HomotopyPreservingThinning
{
public:
    HomotopyPreservingThinning(){};
    HomotopyPreservingThinning(float flux_threshold) : m_flux_threshold_(flux_threshold){};
    void setImages(const cv::Mat &skeleton_mat, const cv::Mat &distance_mat, const cv::Mat &flux_mat)
    {
        m_skeleton_mat_ = skeleton_mat.clone();
        m_distance_mat_ = distance_mat.clone();
        m_flux_mat_ = flux_mat.clone();
        m_image_width_ = m_skeleton_mat_.cols;
        m_image_height_ = m_skeleton_mat_.rows;
    };

    void setContourPoints(const std::vector<cv::Point> &contour_points)
    {
        m_contour_points_.clear();
        std::copy(contour_points.begin(), contour_points.end(), std::back_inserter(m_contour_points_));
    };

    void compute()
    {
        std::priority_queue<FluxPoint> priority_queue_flux_points;
        m_map_position2status_.clear();
        for (int32_t y = 0; y < m_image_height_; y++)
        {
            for (int32_t x = 0; x < m_image_width_; x++)
            {
                // TODO: change skeleton condition
                if (m_skeleton_mat_.at<float>(y, x) == 0)
                {
                    PointPosition pos{x, y};
                    m_map_position2status_.insert({pos, PointStatus::kSkeletonCandidate});
                }
                else
                {
                    PointPosition pos{x, y};
                    m_map_position2status_.insert({pos, PointStatus::kRemoved});
                }
            }
        }

        // Insert boundary points to heap for thinning procedure
        for (size_t k = 0; k < m_contour_points_.size(); k++)
        {
            int32_t cp_x = m_contour_points_[k].x;
            int32_t cp_y = m_contour_points_[k].y;
            if (is_image_boundary(cp_x, cp_y))
                continue;
            if (is_simple(cp_x, cp_y))
            {
                m_map_position2status_[{cp_x, cp_y}] = PointStatus::kSkeletonCandidate;
                priority_queue_flux_points.push(FluxPoint(cp_x, cp_y, m_flux_mat_.at<float>(cp_y, cp_x)));
            }
        }

        // Iterative thinning
        while (!priority_queue_flux_points.empty())
        {
            FluxPoint flux_point = priority_queue_flux_points.top().clone();
            priority_queue_flux_points.pop();

            m_map_position2status_[{flux_point.x, flux_point.y}] = PointStatus::kSkeletonCandidate;
            if (!is_simple(flux_point.x, flux_point.y))
                continue;

            if (!is_end_point(flux_point.x, flux_point.y) || flux_point.flux > m_flux_threshold_)
            {
                m_map_position2status_[{flux_point.x, flux_point.y}] = PointStatus::kRemoved;
                for (int32_t ky = -1; ky <= 1; ky++)
                {
                    for (int32_t kx = -1; kx <= 1; kx++)
                    {
                        if (m_map_position2status_[{flux_point.x + kx, flux_point.y + ky}] == PointStatus::kSkeletonCandidate)
                        {
                            if (is_image_boundary(flux_point.x + kx, flux_point.y + ky))
                                continue;
                            if (is_simple(flux_point.x + kx, flux_point.y + ky))
                            {
                                m_map_position2status_[{flux_point.x + kx, flux_point.y + ky}] = PointStatus::kSearching;
                                priority_queue_flux_points.push(
                                    FluxPoint(
                                        flux_point.x + kx,
                                        flux_point.y + ky,
                                        m_flux_mat_.at<float>(flux_point.y + ky, flux_point.x + kx)));
                            }
                        }
                    }
                }
            }
            else
            {
                m_map_position2status_[{flux_point.x, flux_point.y}] = PointStatus::kSkeletonCandidate;
            }
        }

        m_skeleton_point_list_.clear();
        for (auto kv : m_map_position2status_)
        {
            PointPosition pos = kv.first;
            PointStatus status = kv.second;
            if (status == PointStatus::kSkeletonCandidate)
            {
                int32_t p_x = pos.first;
                int32_t p_y = pos.second;
                m_skeleton_point_list_.push_back(SkeletonPoint(p_x, p_y, m_distance_mat_.at<float>(p_y, p_x)));
            }
        }
    };

    cv::Mat getSkeletonImage()
    {
        cv::Mat skeleton_image = cv::Mat::zeros(cv::Size(m_image_width_, m_image_height_), CV_32F);
        for (SkeletonPoint skeleton_point : m_skeleton_point_list_)
        {
            skeleton_image.at<float>(skeleton_point.y, skeleton_point.x) = 1.0;
        }
        return skeleton_image;
    }

private:
    int32_t neighbor_indices_to_hash(int32_t kx, int32_t ky)
    {
        int32_t local_index = (ky + 1) * 3 + (kx + 1);
        switch (local_index)
        {
        case 0: // {kx = -1, ky = -1}
            return 0;
        case 1: // {kx = 0, ky = -1}
            return 1;
        case 2: // {kx = 1, ky = -1}
            return 2;
        case 5: // {kx = 1, ky = 0}
            return 3;
        case 8: // {kx = 1, ky = 1}
            return 4;
        case 7: // {kx = 0, ky = 1}
            return 5;
        case 6: // {kx = -1, ky = 1}
            return 6;
        case 3: // {kx = -1, ky = 0}
            return 7;
        default:
            std::cerr << "Invarid neighbor indices is received" << std::endl;
            exit(EXIT_FAILURE);
        }
        return -1;
    }

    bool is_simple(const int32_t& p_x, const int32_t& p_y)
    {
        if (is_image_boundary(p_x, p_y))
            return true;

        std::set<int32_t> neighbor_vertice_list;
        for (int32_t ky = -1; ky <= 1; ky++)
        {
            for (int32_t kx = -1; kx <= 1; kx++)
            {
                if (kx == 0 && ky == 0)
                    continue;
                if (m_map_position2status_[{p_x + kx, p_y + ky}] != PointStatus::kRemoved)
                    neighbor_vertice_list.insert(neighbor_indices_to_hash(kx, ky));
            }
        }

        int32_t num_vertices = 0;
        int32_t num_edges = 0;
        for (int32_t neighbor_hash : neighbor_vertice_list)
        {
            int32_t prev_vertex = (neighbor_hash - 1 >= 0) ? (neighbor_hash - 1) : (neighbor_hash + 7);
            int32_t post_vertex = (neighbor_hash + 1 <= 7) ? (neighbor_hash + 1) : (neighbor_hash - 7);
            if ((neighbor_vertice_list.find(post_vertex) != neighbor_vertice_list.end()) ||
                (neighbor_hash == 1 && (neighbor_vertice_list.find(3) != neighbor_vertice_list.end())) ||
                (neighbor_hash == 3 && (neighbor_vertice_list.find(5) != neighbor_vertice_list.end())) ||
                (neighbor_hash == 5 && (neighbor_vertice_list.find(7) != neighbor_vertice_list.end())) ||
                (neighbor_hash == 7 && (neighbor_vertice_list.find(1) != neighbor_vertice_list.end())))
            {
                num_edges += 1;
                num_vertices += 1;
            }
            else
            {
                num_vertices += 1;
            }

            if (neighbor_hash == 0 || neighbor_hash == 2 || neighbor_hash == 4 || neighbor_hash == 6)
            {
                if ((neighbor_vertice_list.find(prev_vertex) != neighbor_vertice_list.end()) &&
                    (neighbor_vertice_list.find(post_vertex) != neighbor_vertice_list.end()))
                {
                    num_edges -= 1;
                    num_vertices -= 1;
                }
            }
        }

        // Is local graph a tree?
        return (num_vertices - num_edges == 1);
    }

    bool is_end_point(const int32_t& p_x, const int32_t& p_y)
    {
        std::vector<int32_t> neighbor_vertice_list;
        for (int32_t ky = -1; ky <= 1; ky++)
        {
            for (int32_t kx = -1; kx <= 1; kx++)
            {
                if (kx == 0 && ky == 0)
                    continue;
                if (m_map_position2status_[{p_x + kx, p_y + ky}] != PointStatus::kRemoved)
                
                    neighbor_vertice_list.push_back(neighbor_indices_to_hash(kx, ky));
            }
        }
        if (neighbor_vertice_list.size() > 2)
            return false;

        // In the case of isolated point
        if (neighbor_vertice_list.size() == 1)
            return true;

        if ((abs(neighbor_vertice_list[0] - neighbor_vertice_list[1]) == 1) ||
            (neighbor_vertice_list[0] == 7 && neighbor_vertice_list[1] == 0) ||
            (neighbor_vertice_list[0] == 0 && neighbor_vertice_list[1] == 7))
        {
            return true;
        }
        return false;
    }

    bool is_image_boundary(const int32_t &p_x, const int32_t &p_y)
    {
        return p_x <= 0 || p_x >= m_image_width_ - 1 || p_y <= 0 || p_y >= m_image_height_ - 1;
    }

    float m_flux_threshold_;
    int32_t m_image_width_, m_image_height_;

    cv::Mat m_skeleton_mat_, m_distance_mat_, m_flux_mat_;
    cv::Mat m_label_mat_;
    std::vector<cv::Point2i> m_contour_points_;
    std::vector<SkeletonPoint> m_skeleton_point_list_;
    MapPosition2Status m_map_position2status_;
};

#endif