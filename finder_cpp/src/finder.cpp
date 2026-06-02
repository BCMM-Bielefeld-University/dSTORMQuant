// Port of FINDER (MIT, Copyright (c) 2021 Andreas Nold)
// C++ implementation Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE and https://github.com/NoldAndreas/FINDER

#include "finder.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <chrono>
#include <nanoflann/nanoflann.hpp>
#include <set>

namespace finder {

// KNN adaptor for nanoflann
template<typename Point>
struct KNNAdaptor {
    const std::vector<Point>& points;
    KNNAdaptor(const std::vector<Point>& points) : points(points) { }
    
    inline size_t kdtree_get_point_count() const { return points.size(); }
    
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        if constexpr (std::is_same_v<Point, point2>) {
            return (dim == 0) ? points[idx].x : points[idx].y;
        } else {
            if (dim == 0) return points[idx].x;
            if (dim == 1) return points[idx].y;
            return points[idx].z;
        }
    }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

template<typename Point>
Finder<Point>::Finder(
    int threshold,
    int points_per_dimension,
    const std::string& algo,
    std::vector<int> minmax_threshold,
    const std::string& one_two_d,
    const std::string& similarity_score_computation,
    bool log_thresholds,
    bool log_sigmas,
    bool adaptive_sigma_boundaries,
    float decay,
    int n_dim
) : threshold_(threshold),
    no_points_sigma_(points_per_dimension),
    no_points_thresholds_(points_per_dimension),
    algo_(algo),
    minmax_threshold_(minmax_threshold),
    one_two_d_(one_two_d),
    similarity_score_computation_(similarity_score_computation),
    log_thresholds_(log_thresholds),
    log_sigmas_(log_sigmas),
    adaptive_sigma_boundaries_(adaptive_sigma_boundaries),
    decay_(decay),
    n_dim_(n_dim)
{
    if (one_two_d != "oneD" && one_two_d != "oneD_thresholds" && one_two_d != "twoD") {
        throw std::invalid_argument("one_two_d must be 'oneD', 'oneD_thresholds', or 'twoD'");
    }
    if (similarity_score_computation != "total" && similarity_score_computation != "threshold") {
        throw std::invalid_argument("similarity_score_computation must be 'total' or 'threshold'");
    }
}

template<typename Point>
std::vector<int> Finder<Point>::fit(const std::vector<Point>& data, bool skip_similarity_score) {
    std::cout << "Analysing " << data.size() << " points" << std::endl;
    
    if (n_dim_ == -1) {
        if constexpr (std::is_same_v<Point, point2>) {
            n_dim_ = 2;
        } else {
            n_dim_ = 3;
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Step 1: Get parameters
    ParameterSet params;
    if (one_two_d_ == "oneD") {
        params = get_params_sigmas(data);
    } else if (one_two_d_ == "oneD_thresholds") {
        params = get_params_thresholds(data);
    } else {
        params = get_params_sigmas_thresholds(data);
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // Step 2: Compute phase space
    phase_space_ = compute_phase_space(data, params);
    
    auto t3 = std::chrono::high_resolution_clock::now();
    
    // Step 3: Post-process and get consensus
    phase_space_post_process(data, skip_similarity_score);
    
    if (skip_similarity_score) {
        // Simple selection based on threshold
        size_t best_idx = 0;
        int best_clusters = 0;
        float target_threshold = static_cast<float>(threshold_);
        
        for (size_t i = 0; i < phase_space_.size(); ++i) {
            if (std::abs(phase_space_[i].threshold - target_threshold) < 0.5f) {
                if (phase_space_[i].no_clusters > best_clusters) {
                    best_clusters = phase_space_[i].no_clusters;
                    best_idx = i;
                }
            }
        }
        
        labels_ = phase_space_[best_idx].labels;
        selected_parameters_.sigma = phase_space_[best_idx].sigma;
        selected_parameters_.threshold = phase_space_[best_idx].threshold;
    } else {
        if (one_two_d_ == "twoD") {
            auto result = get_consensus_clustering(data);
            labels_ = result.first;
            selected_parameters_ = result.second;
        } else {
            auto result = get_consensus_clustering_1d(data);
            labels_ = result.first;
            selected_parameters_ = result.second;
        }
    }
    
    auto t4 = std::chrono::high_resolution_clock::now();
    
    // Save computation times
    computation_times_["Step1"] = std::chrono::duration<double>(t2 - t1).count();
    computation_times_["Step2"] = std::chrono::duration<double>(t3 - t2).count();
    computation_times_["Step3"] = std::chrono::duration<double>(t4 - t3).count();
    
    std::cout << "Comp time Step 1 (set boundaries): " << computation_times_["Step1"] << " seconds" << std::endl;
    std::cout << "Comp time Step 2 (clustering): " << computation_times_["Step2"] << " seconds" << std::endl;
    std::cout << "Comp time Step 3 (postprocessing): " << computation_times_["Step3"] << " seconds" << std::endl;
    std::cout << "Selected parameters: sigma=" << selected_parameters_.sigma 
              << ", threshold=" << selected_parameters_.threshold << std::endl;
    
    return labels_;
}

template<typename Point>
std::vector<int> Finder<Point>::compute_clusters(float sigma, int threshold, const std::vector<Point>& data) {
    if (algo_ == "dbscan") {
        auto span_data = std::span<const Point>(data.data(), data.size());
        auto clusters = dbscan(span_data, sigma, threshold);
        
        // Convert clusters to labels
        std::vector<int> labels(data.size(), -1);
        for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
            for (size_t idx : clusters[cluster_id]) {
                labels[idx] = static_cast<int>(cluster_id);
            }
        }
        return labels;
    } else if (algo_ == "DbscanLoop") {
        DbscanLoop dbl(sigma, threshold);
        dbl.fit(data);
        return dbl.get_labels();
    } else {
        throw std::runtime_error("Unknown algorithm: " + algo_);
    }
}

template<typename Point>
std::vector<float> Finder<Point>::get_log_distribution(float min_x, float max_x, int n) {
    float min_log = std::log(min_x);
    float max_log = std::log(max_x);
    
    std::vector<float> log_vec(n);
    for (int i = 0; i < n; ++i) {
        log_vec[i] = min_log + (max_log - min_log) * i / (n - 1);
    }
    
    std::vector<float> vec(n);
    for (int i = 0; i < n; ++i) {
        vec[i] = std::exp(log_vec[i]);
    }
    
    // Get unique values
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
    
    return vec;
}

template<typename Point>
std::vector<float> Finder<Point>::determine_sigma_boundaries(const std::vector<Point>& data) {
    int k = threshold_ + 1;
    
    constexpr int n_cols = std::is_same_v<Point, point2> ? 2 : 3;
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, KNNAdaptor<Point>>,
        KNNAdaptor<Point>,
        n_cols
    >;
    
    KNNAdaptor<Point> adaptor(data);
    KDTree index(n_cols, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();
    
    std::vector<float> kth_distances;
    std::vector<size_t> indices(k);
    std::vector<float> dists_sq(k);
    
    for (size_t i = 0; i < data.size(); ++i) {
        if constexpr (std::is_same_v<Point, point2>) {
            float query[2] = {data[i].x, data[i].y};
            index.knnSearch(query, k, indices.data(), dists_sq.data());
        } else {
            float query[3] = {data[i].x, data[i].y, data[i].z};
            index.knnSearch(query, k, indices.data(), dists_sq.data());
        }
        kth_distances.push_back(std::sqrt(dists_sq[k-1]));
    }
    
    std::sort(kth_distances.begin(), kth_distances.end());
    float sigma_min = kth_distances[static_cast<size_t>(kth_distances.size() * 0.1)];
    float sigma_max = kth_distances[static_cast<size_t>(kth_distances.size() * 0.9)];
    
    std::cout << "Boundaries for sigma: " << sigma_min << " , " << sigma_max << std::endl;
    
    return {sigma_min, sigma_max};
}

template<typename Point>
std::vector<float> Finder<Point>::determine_sigma_boundaries_adaptive(const std::vector<Point>& data) {
    constexpr int n_cols = std::is_same_v<Point, point2> ? 2 : 3;
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, KNNAdaptor<Point>>,
        KNNAdaptor<Point>,
        n_cols
    >;
    
    KNNAdaptor<Point> adaptor(data);
    KDTree index(n_cols, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();
    
    // For min
    int k_min = minmax_threshold_[0] + 1;
    std::vector<float> kth_distances_min;
    std::vector<size_t> indices(std::max(k_min, minmax_threshold_[1] + 1));
    std::vector<float> dists_sq(std::max(k_min, minmax_threshold_[1] + 1));
    
    for (size_t i = 0; i < data.size(); ++i) {
        if constexpr (std::is_same_v<Point, point2>) {
            float query[2] = {data[i].x, data[i].y};
            index.knnSearch(query, k_min, indices.data(), dists_sq.data());
        } else {
            float query[3] = {data[i].x, data[i].y, data[i].z};
            index.knnSearch(query, k_min, indices.data(), dists_sq.data());
        }
        kth_distances_min.push_back(std::sqrt(dists_sq[k_min-1]));
    }
    
    std::sort(kth_distances_min.begin(), kth_distances_min.end());
    float sigma_min = kth_distances_min[static_cast<size_t>(kth_distances_min.size() * 0.1)];
    
    // For max
    int k_max = minmax_threshold_[1] + 1;
    std::vector<float> kth_distances_max;
    
    for (size_t i = 0; i < data.size(); ++i) {
        if constexpr (std::is_same_v<Point, point2>) {
            float query[2] = {data[i].x, data[i].y};
            index.knnSearch(query, k_max, indices.data(), dists_sq.data());
        } else {
            float query[3] = {data[i].x, data[i].y, data[i].z};
            index.knnSearch(query, k_max, indices.data(), dists_sq.data());
        }
        kth_distances_max.push_back(std::sqrt(dists_sq[k_max-1]));
    }
    
    std::sort(kth_distances_max.begin(), kth_distances_max.end());
    float sigma_max = kth_distances_max[static_cast<size_t>(kth_distances_max.size() * 0.9)];
    
    std::cout << "Boundaries for sigma: " << sigma_min << " , " << sigma_max << std::endl;
    
    return {sigma_min, sigma_max};
}

template<typename Point>
typename Finder<Point>::ParameterSet Finder<Point>::get_params_sigmas_thresholds(const std::vector<Point>& data) {
    ParameterSet params;
    
    auto minmax_sigma = adaptive_sigma_boundaries_ ? 
        determine_sigma_boundaries_adaptive(data) : 
        determine_sigma_boundaries(data);
    
    if (log_sigmas_) {
        params.sigmas = get_log_distribution(minmax_sigma[0], minmax_sigma[1], no_points_sigma_);
    } else {
        params.sigmas.resize(no_points_sigma_);
        for (int i = 0; i < no_points_sigma_; ++i) {
            params.sigmas[i] = minmax_sigma[0] + (minmax_sigma[1] - minmax_sigma[0]) * i / (no_points_sigma_ - 1);
        }
    }
    
    if (log_thresholds_) {
        std::vector<float> temp(no_points_sigma_);
        for (int i = 0; i < no_points_sigma_; ++i) {
            temp[i] = minmax_threshold_[0] + 
                     (minmax_threshold_[1] - minmax_threshold_[0]) * static_cast<float>(i) / (no_points_sigma_ - 1);
        }
        std::set<int> unique_thresholds;
        for (float t : temp) {
            unique_thresholds.insert(static_cast<int>(std::round(t)));
        }
        params.thresholds.assign(unique_thresholds.begin(), unique_thresholds.end());
    } else {
        for (int t = minmax_threshold_[0]; t < minmax_threshold_[1]; ++t) {
            params.thresholds.push_back(t);
        }
    }
    
    std::cout << "Sigmas are:" << std::endl;
    for (float s : params.sigmas) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Thresholds are:" << std::endl;
    for (int t : params.thresholds) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
    
    return params;
}

template<typename Point>
typename Finder<Point>::ParameterSet Finder<Point>::get_params_sigmas(const std::vector<Point>& data) {
    ParameterSet params;
    
    auto minmax_sigma = adaptive_sigma_boundaries_ ? 
        determine_sigma_boundaries_adaptive(data) : 
        determine_sigma_boundaries(data);
    
    if (log_sigmas_) {
        params.sigmas = get_log_distribution(minmax_sigma[0], minmax_sigma[1], no_points_sigma_);
    } else {
        params.sigmas.resize(no_points_sigma_);
        for (int i = 0; i < no_points_sigma_; ++i) {
            params.sigmas[i] = minmax_sigma[0] + (minmax_sigma[1] - minmax_sigma[0]) * i / (no_points_sigma_ - 1);
        }
    }
    
    params.thresholds.assign(params.sigmas.size(), threshold_);
    
    return params;
}

template<typename Point>
typename Finder<Point>::ParameterSet Finder<Point>::get_params_thresholds(const std::vector<Point>& data) {
    ParameterSet params;
    
    // Compute sigma using KNN
    int k = 10;
    constexpr int n_cols = std::is_same_v<Point, point2> ? 2 : 3;
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, KNNAdaptor<Point>>,
        KNNAdaptor<Point>,
        n_cols
    >;
    
    KNNAdaptor<Point> adaptor(data);
    KDTree index(n_cols, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();
    
    std::vector<float> kth_distances;
    std::vector<size_t> indices(k);
    std::vector<float> dists_sq(k);
    
    for (size_t i = 0; i < data.size(); ++i) {
        if constexpr (std::is_same_v<Point, point2>) {
            float query[2] = {data[i].x, data[i].y};
            index.knnSearch(query, k, indices.data(), dists_sq.data());
        } else {
            float query[3] = {data[i].x, data[i].y, data[i].z};
            index.knnSearch(query, k, indices.data(), dists_sq.data());
        }
        kth_distances.push_back(std::sqrt(dists_sq[k-1]));
    }
    
    std::sort(kth_distances.begin(), kth_distances.end());
    float sigma = kth_distances[kth_distances.size() / 2];
    
    if (log_thresholds_) {
        std::vector<float> temp(no_points_sigma_);
        for (int i = 0; i < no_points_sigma_; ++i) {
            temp[i] = minmax_threshold_[0] + 
                     (minmax_threshold_[1] - minmax_threshold_[0]) * static_cast<float>(i) / (no_points_sigma_ - 1);
        }
        std::set<int> unique_thresholds;
        for (float t : temp) {
            unique_thresholds.insert(static_cast<int>(std::round(t)));
        }
        params.thresholds.assign(unique_thresholds.begin(), unique_thresholds.end());
    } else {
        for (int t = minmax_threshold_[0]; t < minmax_threshold_[1]; ++t) {
            params.thresholds.push_back(t);
        }
    }
    
    params.sigmas.assign(params.thresholds.size(), sigma);
    
    return params;
}

template<typename Point>
std::vector<PhaseSpaceRow> Finder<Point>::compute_phase_space(
    const std::vector<Point>& data,
    const ParameterSet& params
) {
    std::vector<PhaseSpaceRow> phase_space;
    
    // Generate all combinations
    for (float sigma : params.sigmas) {
        for (int threshold : params.thresholds) {
            PhaseSpaceRow row;
            row.sigma = sigma;
            row.threshold = threshold;
            phase_space.push_back(row);
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    printProgressBar(0, phase_space.size(), "Clustering progress:", "Complete", 1, 50);
    
    for (size_t i = 0; i < phase_space.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        phase_space[i].labels = compute_clusters(phase_space[i].sigma, phase_space[i].threshold, data);
        auto end = std::chrono::high_resolution_clock::now();
        phase_space[i].time = std::chrono::duration<double>(end - start).count();
        
        printProgressBar(i + 1, phase_space.size(), "Progress:", "Complete", 1, 50);
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Computing clusters: " << std::chrono::duration<double>(t2 - t1).count() << " seconds" << std::endl;
    
    return phase_space;
}

template<typename Point>
void Finder<Point>::phase_space_post_process(
    const std::vector<Point>& data,
    bool skip_similarity_score
) {
    std::cout << "Postprocessing.." << std::endl;
    
    // Compute number of clusters for each configuration
    for (auto& ps : phase_space_) {
        int max_label = *std::max_element(ps.labels.begin(), ps.labels.end());
        ps.no_clusters = max_label + 1;
    }
    
    // Compute cluster information
    cluster_info_ = getClusterSizesAll(data, phase_space_, n_dim_);
    
    if (skip_similarity_score) {
        for (auto& ps : phase_space_) {
            ps.similarityScore = std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        std::vector<int> cli_similarityScore;
        std::vector<float> similarityScore;
        
        if (similarity_score_computation_ == "total") {
            auto result = getSimilarityScore(data, phase_space_, cluster_info_, n_dim_);
            cli_similarityScore = result.first;
            similarityScore = result.second;
        } else {
            auto result = getSimilarityScoreByThreshold(data, phase_space_, cluster_info_, n_dim_);
            cli_similarityScore = result.first;
            similarityScore = result.second;
        }
        
        cluster_info_.similarityScore = cli_similarityScore;
        for (size_t i = 0; i < phase_space_.size(); ++i) {
            phase_space_[i].similarityScore = similarityScore[i];
        }
    }
}

template<typename Point>
std::pair<std::vector<int>, SelectedParameters> Finder<Point>::get_consensus_clustering_1d(
    const std::vector<Point>& data
) {
    // Find maximum similarity score
    float max_score = -std::numeric_limits<float>::infinity();
    size_t max_idx = 0;
    
    for (size_t i = 0; i < phase_space_.size(); ++i) {
        if (phase_space_[i].similarityScore > max_score) {
            max_score = phase_space_[i].similarityScore;
            max_idx = i;
        }
    }
    
    SelectedParameters selected;
    selected.sigma = phase_space_[max_idx].sigma;
    selected.threshold = phase_space_[max_idx].threshold;
    
    std::cout << "Selected threshold, sigma: " << selected.threshold << " , " << selected.sigma << std::endl;
    
    return {phase_space_[max_idx].labels, selected};
}

template<typename Point>
std::pair<std::vector<int>, SelectedParameters> Finder<Point>::get_consensus_clustering(
    const std::vector<Point>& data
) {
    // Get thresholds as floats for GetLineOfOptima
    std::vector<float> thresholds;
    std::vector<float> similarities;
    for (const auto& ps : phase_space_) {
        thresholds.push_back(static_cast<float>(ps.threshold));
        similarities.push_back(ps.similarityScore);
    }
    
    auto optima = GetLineOfOptima(thresholds, similarities);
    
    // Normalize
    float min_sim = *std::min_element(optima.y_values.begin(), optima.y_values.end());
    float max_sim = *std::max_element(optima.y_values.begin(), optima.y_values.end());
    
    std::vector<float> opt_normalized(optima.y_values.size());
    for (size_t i = 0; i < optima.y_values.size(); ++i) {
        opt_normalized[i] = (optima.y_values[i] - min_sim) / (max_sim - min_sim);
    }
    
    // Find first index where normalized score < decay
    size_t ind = 0;
    for (size_t i = 0; i < opt_normalized.size(); ++i) {
        if (opt_normalized[i] < decay_) {
            ind = i;
            break;
        }
    }
    
    size_t optimal_idx = optima.indices[ind];
    
    SelectedParameters selected;
    selected.sigma = phase_space_[optimal_idx].sigma;
    selected.threshold = phase_space_[optimal_idx].threshold;
    
    std::cout << "Selected threshold, sigma: " << selected.threshold << " , " << selected.sigma << std::endl;
    
    return {phase_space_[optimal_idx].labels, selected};
}

// Explicit template instantiations
template class Finder<point2>;
template class Finder<point3>;

} // namespace finder
