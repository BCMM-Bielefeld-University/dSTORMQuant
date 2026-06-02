// Port of FINDER (MIT, Copyright (c) 2021 Andreas Nold)
// C++ implementation Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE and https://github.com/NoldAndreas/FINDER

#pragma once

#include "dbscan.hpp"
#include <vector>
#include <span>

namespace finder {

/**
 * The noise-free implementation of DBSCAN, as described in the paper.
 * This iteratively applies DBSCAN, keeping only core samples until convergence.
 */
class DbscanLoop {
public:
    /**
     * Constructor
     * @param eps The radius parameter (epsilon)
     * @param min_samples The minimum number of samples (minPts)
     */
    DbscanLoop(float eps, int min_samples);
    
    /**
     * Fit the model to 2D data
     * @param data Input points (2D)
     * @return Reference to this object for chaining
     */
    DbscanLoop& fit(const std::vector<point2>& data);
    
    /**
     * Fit the model to 3D data
     * @param data Input points (3D)
     * @return Reference to this object for chaining
     */
    DbscanLoop& fit(const std::vector<point3>& data);
    
    /**
     * Get the cluster labels after fitting
     * Labels are -1 for noise points, 0+ for cluster IDs
     * @return Vector of labels for each point
     */
    const std::vector<int>& get_labels() const { return labels_; }
    
private:
    float eps_;
    int min_samples_;
    std::vector<int> labels_;
    
    // Helper to convert cluster indices to labels
    template<typename Point>
    void process_clusters(const std::vector<std::vector<size_t>>& clusters, 
                         size_t original_size);
};

} // namespace finder
