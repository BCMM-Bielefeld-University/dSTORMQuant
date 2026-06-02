// Port of FINDER (MIT, Copyright (c) 2021 Andreas Nold)
// C++ implementation Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE and https://github.com/NoldAndreas/FINDER

#pragma once

#include "dbscan.hpp"
#include <vector>
#include <map>

namespace finder {

// Forward declarations
struct PhaseSpaceRow;

// Structure to store cluster information
struct ClusterInfo {
    std::vector<int> labels;           // Cluster label for each entry
    std::vector<int> clusterSize;      // Size of each cluster
    std::vector<int> threshold;        // Threshold parameter for each cluster
    std::vector<float> sigma;          // Sigma parameter for each cluster
    std::vector<size_t> index;         // Phase space index for each cluster
    std::vector<std::vector<float>> centers;  // Center coordinates for each cluster
    std::vector<float> radii;          // Radius for each cluster
    std::vector<int> similarityScore;  // Similarity score for each cluster
};

/**
 * Compute cluster sizes, centers, and radii for all phase space configurations
 * 
 * @param data The input data points (2D or 3D)
 * @param phase_space Vector of phase space configurations
 * @param n_dim Dimensionality of the data (2 or 3)
 * @return ClusterInfo structure with all cluster information
 */
template<typename Point>
ClusterInfo getClusterSizesAll(
    const std::vector<Point>& data,
    const std::vector<PhaseSpaceRow>& phase_space,
    int n_dim
);

/**
 * Compute similarity score between all phase space configurations
 * 
 * @param data The input data points
 * @param phase_space Vector of phase space configurations
 * @param cluster_info Cluster information from getClusterSizesAll
 * @param n_dim Dimensionality of the data
 * @return Pair of (per-cluster similarity scores, per-configuration similarity scores)
 */
template<typename Point>
std::pair<std::vector<int>, std::vector<float>> getSimilarityScore(
    const std::vector<Point>& data,
    const std::vector<PhaseSpaceRow>& phase_space,
    const ClusterInfo& cluster_info,
    int n_dim
);

/**
 * Compute similarity score only between configurations with the same threshold
 * 
 * @param data The input data points
 * @param phase_space Vector of phase space configurations
 * @param cluster_info Cluster information from getClusterSizesAll
 * @param n_dim Dimensionality of the data
 * @return Pair of (per-cluster similarity scores, per-configuration similarity scores)
 */
template<typename Point>
std::pair<std::vector<int>, std::vector<float>> getSimilarityScoreByThreshold(
    const std::vector<Point>& data,
    const std::vector<PhaseSpaceRow>& phase_space,
    const ClusterInfo& cluster_info,
    int n_dim
);

/**
 * Helper function to compute similarity between two phase space configurations
 */
std::pair<std::vector<int>, std::vector<int>> getSimilarityScore_ij(
    size_t i,
    size_t j,
    const std::vector<PhaseSpaceRow>& phase_space,
    const ClusterInfo& cluster_info
);

/**
 * Check if two clusters do not overlap based on distance
 */
bool noOverlapClusters_Distance(
    int i1,
    int i2,
    const std::vector<std::vector<float>>& centers_1,
    const std::vector<std::vector<float>>& centers_2,
    const std::vector<float>& radii_1,
    const std::vector<float>& radii_2
);

/**
 * Check if two clusters overlap based on number of shared points
 */
bool overlapClusters_NumberOfLocs(
    int i1,
    int i2,
    const std::vector<int>& labels_1,
    const std::vector<int>& labels_2
);

/**
 * Compute centers and radii for a single phase space configuration
 */
template<typename Point>
std::pair<std::vector<std::vector<float>>, std::vector<float>> computeCenters_Radii_rowPS(
    const std::vector<Point>& data,
    const PhaseSpaceRow& ps_row,
    int n_dim
);

} // namespace finder
