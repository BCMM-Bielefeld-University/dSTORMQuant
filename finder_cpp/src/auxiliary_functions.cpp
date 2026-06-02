// Port of FINDER (MIT, Copyright (c) 2021 Andreas Nold)
// C++ implementation Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE and https://github.com/NoldAndreas/FINDER

#include "auxiliary_functions.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include <set>
#include <algorithm>

namespace finder {

OptimaResult GetLineOfOptima(const std::vector<float>& x_values, const std::vector<float>& y_values) {
    if (x_values.size() != y_values.size()) {
        throw std::invalid_argument("x_values and y_values must have the same size");
    }
    
    OptimaResult result;
    
    // Get unique x values
    std::set<float> unique_x_set(x_values.begin(), x_values.end());
    std::vector<float> unique_x(unique_x_set.begin(), unique_x_set.end());
    std::sort(unique_x.begin(), unique_x.end());
    
    // For each unique x, find the index with maximum y
    for (float x_val : unique_x) {
        size_t best_idx = 0;
        float best_y = -std::numeric_limits<float>::infinity();
        
        for (size_t i = 0; i < x_values.size(); ++i) {
            if (x_values[i] == x_val && y_values[i] > best_y) {
                best_y = y_values[i];
                best_idx = i;
            }
        }
        
        result.indices.push_back(best_idx);
        result.x_values.push_back(x_val);
        result.y_values.push_back(best_y);
    }
    
    return result;
}

std::vector<int> GetClusterDistribution(const std::vector<int>& labels) {
    if (labels.empty()) {
        return {};
    }
    
    // Find max label (excluding -1)
    int max_label = -1;
    for (int label : labels) {
        if (label > max_label) {
            max_label = label;
        }
    }
    
    if (max_label < 0) {
        return {}; // All noise
    }
    
    // Count points for each label
    std::vector<int> cluster_sizes;
    for (int c = 0; c <= max_label; ++c) {
        int count = 0;
        for (int label : labels) {
            if (label == c) {
                count++;
            }
        }
        cluster_sizes.push_back(count);
    }
    
    return cluster_sizes;
}

void printProgressBar(
    int iteration,
    int total,
    const std::string& prefix,
    const std::string& suffix,
    int decimals,
    int length
) {
    float percent = 100.0f * iteration / total;
    int filled_length = static_cast<int>(length * iteration / total);
    
    std::string bar(filled_length, '#');
    bar += std::string(length - filled_length, '-');
    
    std::cout << "\r" << prefix << " |" << bar << "| " 
              << std::fixed << std::setprecision(decimals) << percent 
              << "% " << suffix << std::flush;
    
    if (iteration == total) {
        std::cout << std::endl;
    }
}

} // namespace finder
