// Port of FINDER (MIT, Copyright (c) 2021 Andreas Nold)
// C++ implementation Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE and https://github.com/NoldAndreas/FINDER

#include "dbscan_loop.hpp"
#include <algorithm>
#include <set>
#include <numeric>

namespace finder {

DbscanLoop::DbscanLoop(float eps, int min_samples)
    : eps_(eps), min_samples_(min_samples) {}

DbscanLoop& DbscanLoop::fit(const std::vector<point2>& data) {
    if (data.empty()) {
        labels_.clear();
        return *this;
    }
    
    // Initialize all labels as noise
    labels_.assign(data.size(), -1);
    
    // Keep track of current data and indices
    std::vector<point2> current_data = data;
    std::vector<size_t> current_indices(data.size());
    std::iota(current_indices.begin(), current_indices.end(), 0);
    
    size_t n_old = 0;
    
    // Iteratively apply DBSCAN on core samples
    while (n_old != current_data.size() && !current_data.empty()) {
        n_old = current_data.size();
        
        // Run DBSCAN
        auto span_data = std::span<const point2>(current_data.data(), current_data.size());
        auto clusters = dbscan(span_data, eps_, min_samples_);
        
        // Extract core sample indices
        std::set<size_t> core_indices;
        std::vector<int> temp_labels(current_data.size(), -1);
        
        for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
            for (size_t idx : clusters[cluster_id]) {
                core_indices.insert(idx);
                temp_labels[idx] = static_cast<int>(cluster_id);
            }
        }
        
        // If converged, assign final labels
        if (core_indices.size() == current_data.size() || core_indices.empty()) {
            // Assign labels to original indices
            for (size_t i = 0; i < current_indices.size(); ++i) {
                labels_[current_indices[i]] = temp_labels[i];
            }
            break;
        }
        
        // Keep only core samples for next iteration
        std::vector<point2> new_data;
        std::vector<size_t> new_indices;
        
        for (size_t idx : core_indices) {
            new_data.push_back(current_data[idx]);
            new_indices.push_back(current_indices[idx]);
        }
        
        current_data = std::move(new_data);
        current_indices = std::move(new_indices);
    }
    
    // Final pass: assign labels from converged core samples
    if (!current_data.empty() && current_data.size() < data.size()) {
        auto span_data = std::span<const point2>(current_data.data(), current_data.size());
        auto clusters = dbscan(span_data, eps_, min_samples_);
        
        for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
            for (size_t idx : clusters[cluster_id]) {
                labels_[current_indices[idx]] = static_cast<int>(cluster_id);
            }
        }
    }
    
    return *this;
}

DbscanLoop& DbscanLoop::fit(const std::vector<point3>& data) {
    if (data.empty()) {
        labels_.clear();
        return *this;
    }
    
    // Initialize all labels as noise
    labels_.assign(data.size(), -1);
    
    // Keep track of current data and indices
    std::vector<point3> current_data = data;
    std::vector<size_t> current_indices(data.size());
    std::iota(current_indices.begin(), current_indices.end(), 0);
    
    size_t n_old = 0;
    
    // Iteratively apply DBSCAN on core samples
    while (n_old != current_data.size() && !current_data.empty()) {
        n_old = current_data.size();
        
        // Run DBSCAN
        auto span_data = std::span<const point3>(current_data.data(), current_data.size());
        auto clusters = dbscan(span_data, eps_, min_samples_);
        
        // Extract core sample indices
        std::set<size_t> core_indices;
        std::vector<int> temp_labels(current_data.size(), -1);
        
        for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
            for (size_t idx : clusters[cluster_id]) {
                core_indices.insert(idx);
                temp_labels[idx] = static_cast<int>(cluster_id);
            }
        }
        
        // If converged, assign final labels
        if (core_indices.size() == current_data.size() || core_indices.empty()) {
            // Assign labels to original indices
            for (size_t i = 0; i < current_indices.size(); ++i) {
                labels_[current_indices[i]] = temp_labels[i];
            }
            break;
        }
        
        // Keep only core samples for next iteration
        std::vector<point3> new_data;
        std::vector<size_t> new_indices;
        
        for (size_t idx : core_indices) {
            new_data.push_back(current_data[idx]);
            new_indices.push_back(current_indices[idx]);
        }
        
        current_data = std::move(new_data);
        current_indices = std::move(new_indices);
    }
    
    // Final pass: assign labels from converged core samples
    if (!current_data.empty() && current_data.size() < data.size()) {
        auto span_data = std::span<const point3>(current_data.data(), current_data.size());
        auto clusters = dbscan(span_data, eps_, min_samples_);
        
        for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
            for (size_t idx : clusters[cluster_id]) {
                labels_[current_indices[idx]] = static_cast<int>(cluster_id);
            }
        }
    }
    
    return *this;
}

} // namespace finder
