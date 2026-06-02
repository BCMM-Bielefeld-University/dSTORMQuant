// Port of FINDER (MIT, Copyright (c) 2021 Andreas Nold)
// C++ implementation Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE and https://github.com/NoldAndreas/FINDER

#pragma once

#include "dbscan.hpp"
#include "dbscan_loop.hpp"
#include "auxiliary_functions.hpp"
#include "similarity_score.hpp"
#include <vector>
#include <string>
#include <map>

namespace finder {

// Selected parameters result
struct SelectedParameters {
    float sigma;
    int threshold;
};

/**
 * The FINDER algorithm class
 * Explores the two parameters (threshold and sigma) of DBSCAN and finds the best combination
 */
template<typename Point>
class Finder {
public:
    /**
     * Constructor
     * 
     * @param threshold Default threshold value (minPts)
     * @param points_per_dimension Number of values for each axis of phase space
     * @param algo Algorithm to use ("dbscan" or "DbscanLoop")
     * @param minmax_threshold Min and max values for threshold parameter
     * @param one_two_d Type of optimization ("oneD", "oneD_thresholds", "twoD")
     * @param similarity_score_computation Method for computing similarity ("total" or "threshold")
     * @param log_thresholds Use log scale for threshold values
     * @param log_sigmas Use log scale for sigma values
     * @param adaptive_sigma_boundaries Use adaptive sigma boundaries
     * @param decay Decay value for selecting parameters
     * @param n_dim Dimensionality of data (2 or 3, or -1 for auto-detect)
     */
    Finder(
        int threshold = 10,
        int points_per_dimension = 15,
        const std::string& algo = "DbscanLoop",
        std::vector<int> minmax_threshold = {5, 21},
        const std::string& one_two_d = "twoD",
        const std::string& similarity_score_computation = "threshold",
        bool log_thresholds = false,
        bool log_sigmas = true,
        bool adaptive_sigma_boundaries = false,
        float decay = 0.5f,
        int n_dim = -1
    );
    
    /**
     * Fit the FINDER model to data
     * 
     * @param data Input data points
     * @param skip_similarity_score Skip similarity score computation
     * @return Cluster labels for each point
     */
    std::vector<int> fit(const std::vector<Point>& data, bool skip_similarity_score = false);
    
    /**
     * Compute clusters for specific parameters
     * 
     * @param sigma Radius parameter
     * @param threshold Minimum points parameter
     * @param data Input data points
     * @return Cluster labels
     */
    std::vector<int> compute_clusters(float sigma, int threshold, const std::vector<Point>& data);
    
    /**
     * Get the phase space (all parameter combinations and their results)
     */
    const std::vector<PhaseSpaceRow>& get_phase_space() const { return phase_space_; }
    
    /**
     * Get selected parameters
     */
    const SelectedParameters& get_selected_parameters() const { return selected_parameters_; }
    
    /**
     * Get final labels
     */
    const std::vector<int>& get_labels() const { return labels_; }
    
    /**
     * Get cluster info
     */
    const ClusterInfo& get_cluster_info() const { return cluster_info_; }
    
    /**
     * Get computation times
     */
    const std::map<std::string, double>& get_computation_times() const { return computation_times_; }
    
private:
    // Parameters
    int threshold_;
    int no_points_sigma_;
    int no_points_thresholds_;
    std::string algo_;
    std::vector<int> minmax_threshold_;
    std::string one_two_d_;
    std::string similarity_score_computation_;
    bool log_thresholds_;
    bool log_sigmas_;
    bool adaptive_sigma_boundaries_;
    float decay_;
    int n_dim_;
    
    // Results
    std::vector<PhaseSpaceRow> phase_space_;
    ClusterInfo cluster_info_;
    std::vector<int> labels_;
    SelectedParameters selected_parameters_;
    std::map<std::string, double> computation_times_;
    
    // Helper functions
    struct ParameterSet {
        std::vector<float> sigmas;
        std::vector<int> thresholds;
    };
    
    ParameterSet get_params_sigmas_thresholds(const std::vector<Point>& data);
    ParameterSet get_params_sigmas(const std::vector<Point>& data);
    ParameterSet get_params_thresholds(const std::vector<Point>& data);
    
    std::vector<float> determine_sigma_boundaries(const std::vector<Point>& data);
    std::vector<float> determine_sigma_boundaries_adaptive(const std::vector<Point>& data);
    std::vector<float> get_log_distribution(float min_x, float max_x, int n);
    
    std::vector<PhaseSpaceRow> compute_phase_space(
        const std::vector<Point>& data,
        const ParameterSet& params
    );
    
    void phase_space_post_process(
        const std::vector<Point>& data,
        bool skip_similarity_score
    );
    
    std::pair<std::vector<int>, SelectedParameters> get_consensus_clustering_1d(
        const std::vector<Point>& data
    );
    
    std::pair<std::vector<int>, SelectedParameters> get_consensus_clustering(
        const std::vector<Point>& data
    );
};

} // namespace finder
