// FINDER C++ port (auxiliaries). Upstream: NoldAndreas/FINDER (MIT, 2021).
// Port: Suraj Karki, Bielefeld University, 2026. SPDX-License-Identifier: MIT
// See finder_cpp/LICENSE and repository ATTRIBUTION.md.

#pragma once

#include <vector>
#include <map>
#include <string>
#include <cstddef>
#include <algorithm>
#include <numeric>

namespace finder {

// Structure to represent phase space data
struct PhaseSpaceRow {
    float sigma;
    int threshold;
    std::vector<int> labels;
    float time;
    int no_clusters;
    float similarityScore;
};

// Result of GetLineOfOptima
struct OptimaResult {
    std::vector<size_t> indices;
    std::vector<float> x_values;
    std::vector<float> y_values;
};

/**
 * For each unique value in x_selector, find the entry which maximizes y_selector.
 * This gives the "line of optima", i.e., the optimal entry for each unique value of x_selector.
 * 
 * @param x_values The x-axis values (e.g., threshold values)
 * @param y_values The y-axis values (e.g., similarity scores)
 * @return OptimaResult containing indices, x_values, and corresponding y_values
 */
OptimaResult GetLineOfOptima(const std::vector<float>& x_values, const std::vector<float>& y_values);

/**
 * Return a vector containing the number of points with each label.
 * NOTE: Points with label -1 (noise) are not included.
 * 
 * @param labels The cluster labels
 * @return Vector of cluster sizes
 */
std::vector<int> GetClusterDistribution(const std::vector<int>& labels);

/**
 * Display a progress bar in the console.
 * 
 * @param iteration Current iteration
 * @param total Total iterations
 * @param prefix Prefix string
 * @param suffix Suffix string
 * @param decimals Number of decimals in percent
 * @param length Character length of bar
 */
void printProgressBar(
    int iteration,
    int total,
    const std::string& prefix = "",
    const std::string& suffix = "",
    int decimals = 1,
    int length = 50
);

} // namespace finder
