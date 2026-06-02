// Port of FINDER (MIT, Copyright (c) 2021 Andreas Nold)
// C++ implementation Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE and https://github.com/NoldAndreas/FINDER

#include "similarity_score.hpp"
#include "auxiliary_functions.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <iostream>
#include <chrono>

namespace finder {

// Helper function to compute Euclidean distance
template<typename Point>
float euclidean_distance(const Point& p1, const Point& p2);

template<>
float euclidean_distance(const point2& p1, const point2& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx*dx + dy*dy);
}

template<>
float euclidean_distance(const point3& p1, const point3& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Helper to compute median
template<typename Point>
Point compute_median(const std::vector<Point>& points) {
    if (points.empty()) {
        return Point{};
    }
    
    size_t n = points.size();
    if constexpr (std::is_same_v<Point, point2>) {
        std::vector<float> x_vals, y_vals;
        for (const auto& p : points) {
            x_vals.push_back(p.x);
            y_vals.push_back(p.y);
        }
        std::sort(x_vals.begin(), x_vals.end());
        std::sort(y_vals.begin(), y_vals.end());
        return Point{x_vals[n/2], y_vals[n/2]};
    } else {
        std::vector<float> x_vals, y_vals, z_vals;
        for (const auto& p : points) {
            x_vals.push_back(p.x);
            y_vals.push_back(p.y);
            z_vals.push_back(p.z);
        }
        std::sort(x_vals.begin(), x_vals.end());
        std::sort(y_vals.begin(), y_vals.end());
        std::sort(z_vals.begin(), z_vals.end());
        return Point{x_vals[n/2], y_vals[n/2], z_vals[n/2]};
    }
}

template<typename Point>
std::pair<std::vector<std::vector<float>>, std::vector<float>> computeCenters_Radii_rowPS(
    const std::vector<Point>& data,
    const PhaseSpaceRow& ps_row,
    int n_dim
) {
    int max_label = *std::max_element(ps_row.labels.begin(), ps_row.labels.end());
    int no_clusters = max_label + 1;
    
    std::vector<std::vector<float>> centers(no_clusters, std::vector<float>(n_dim, 0.0f));
    std::vector<float> radii(no_clusters, 0.0f);
    
    for (int icl = 0; icl <= max_label; ++icl) {
        // Collect points in this cluster
        std::vector<Point> cluster_points;
        for (size_t i = 0; i < ps_row.labels.size(); ++i) {
            if (ps_row.labels[i] == icl) {
                cluster_points.push_back(data[i]);
            }
        }
        
        if (cluster_points.empty()) continue;
        
        // Compute median as center
        Point center = compute_median(cluster_points);
        
        if constexpr (std::is_same_v<Point, point2>) {
            centers[icl][0] = center.x;
            centers[icl][1] = center.y;
        } else {
            centers[icl][0] = center.x;
            centers[icl][1] = center.y;
            centers[icl][2] = center.z;
        }
        
        // Compute radius as max distance from center
        float max_dist = 0.0f;
        for (const auto& p : cluster_points) {
            float dist = euclidean_distance(p, center);
            max_dist = std::max(max_dist, dist);
        }
        radii[icl] = max_dist;
    }
    
    return {centers, radii};
}

template<typename Point>
ClusterInfo getClusterSizesAll(
    const std::vector<Point>& data,
    const std::vector<PhaseSpaceRow>& phase_space,
    int n_dim
) {
    ClusterInfo cluster_info;
    
    for (size_t idx = 0; idx < phase_space.size(); ++idx) {
        const auto& ps_row = phase_space[idx];
        
        int max_label = *std::max_element(ps_row.labels.begin(), ps_row.labels.end());
        if (max_label < 0) continue;
        
        std::vector<int> l(max_label + 1);
        std::iota(l.begin(), l.end(), 0);
        
        // Add labels
        cluster_info.labels.insert(cluster_info.labels.end(), l.begin(), l.end());
        
        // Compute cluster sizes
        for (int label : l) {
            int count = std::count(ps_row.labels.begin(), ps_row.labels.end(), label);
            cluster_info.clusterSize.push_back(count);
        }
        
        // Add threshold and sigma
        for (size_t i = 0; i < l.size(); ++i) {
            cluster_info.threshold.push_back(ps_row.threshold);
            cluster_info.sigma.push_back(ps_row.sigma);
            cluster_info.index.push_back(idx);
        }
        
        // Compute centers and radii
        auto [centers_new, radii_new] = computeCenters_Radii_rowPS(data, ps_row, n_dim);
        cluster_info.centers.insert(cluster_info.centers.end(), centers_new.begin(), centers_new.end());
        cluster_info.radii.insert(cluster_info.radii.end(), radii_new.begin(), radii_new.end());
    }
    
    // Initialize similarity scores
    cluster_info.similarityScore.assign(cluster_info.labels.size(), 0);
    
    return cluster_info;
}

bool noOverlapClusters_Distance(
    int i1,
    int i2,
    const std::vector<std::vector<float>>& centers_1,
    const std::vector<std::vector<float>>& centers_2,
    const std::vector<float>& radii_1,
    const std::vector<float>& radii_2
) {
    const auto& c1 = centers_1[i1];
    const auto& c2 = centers_2[i2];
    float r1 = radii_1[i1];
    float r2 = radii_2[i2];
    
    // Compute Euclidean distance between centers
    float dist = 0.0f;
    for (size_t i = 0; i < c1.size(); ++i) {
        float diff = c2[i] - c1[i];
        dist += diff * diff;
    }
    dist = std::sqrt(dist);
    
    return dist > (r1 + r2);
}

bool overlapClusters_NumberOfLocs(
    int i1,
    int i2,
    const std::vector<int>& labels_1,
    const std::vector<int>& labels_2
) {
    int no_locs_1 = std::count(labels_1.begin(), labels_1.end(), i1);
    int no_locs_2 = std::count(labels_2.begin(), labels_2.end(), i2);
    
    // Count overlap
    int no_locs_overlap = 0;
    for (size_t i = 0; i < labels_1.size(); ++i) {
        if (labels_1[i] == i1 && labels_2[i] == i2) {
            no_locs_overlap++;
        }
    }
    
    if (no_locs_1 == 0 || no_locs_2 == 0) return false;
    
    return (no_locs_overlap > no_locs_1 / 2) && (no_locs_overlap > no_locs_2 / 2);
}

std::pair<std::vector<int>, std::vector<int>> getSimilarityScore_ij(
    size_t i,
    size_t j,
    const std::vector<PhaseSpaceRow>& phase_space,
    const ClusterInfo& cluster_info
) {
    const auto& labels_1 = phase_space[i].labels;
    const auto& labels_2 = phase_space[j].labels;
    
    // Get cluster info for configurations i and j
    std::vector<std::vector<float>> centers_1, centers_2;
    std::vector<float> radii_1, radii_2;
    
    for (size_t k = 0; k < cluster_info.index.size(); ++k) {
        if (cluster_info.index[k] == i) {
            centers_1.push_back(cluster_info.centers[k]);
            radii_1.push_back(cluster_info.radii[k]);
        }
        if (cluster_info.index[k] == j) {
            centers_2.push_back(cluster_info.centers[k]);
            radii_2.push_back(cluster_info.radii[k]);
        }
    }
    
    int max_label_1 = *std::max_element(labels_1.begin(), labels_1.end());
    int max_label_2 = *std::max_element(labels_2.begin(), labels_2.end());
    
    // Return empty if mostly noise
    if (max_label_1 == -1 || max_label_2 == -1) {
        return {{}, {}};
    }
    if (max_label_1 == 0 && std::count(labels_1.begin(), labels_1.end(), 0) > labels_1.size() / 2) {
        return {{}, {}};
    }
    if (max_label_2 == 0 && std::count(labels_2.begin(), labels_2.end(), 0) > labels_2.size() / 2) {
        return {{}, {}};
    }
    
    // Similarity matrix
    std::vector<std::vector<int>> similarityMatrix(max_label_1 + 1, std::vector<int>(max_label_2 + 1, -1));
    
    for (int i1 = 0; i1 <= max_label_1; ++i1) {
        for (int i2 = 0; i2 <= max_label_2; ++i2) {
            if (similarityMatrix[i1][i2] == 0) continue;
            
            if (noOverlapClusters_Distance(i1, i2, centers_1, centers_2, radii_1, radii_2)) {
                similarityMatrix[i1][i2] = 0;
                continue;
            }
            
            if (overlapClusters_NumberOfLocs(i1, i2, labels_1, labels_2)) {
                for (int k = 0; k <= max_label_2; ++k) {
                    similarityMatrix[i1][k] = 0;
                }
                for (int k = 0; k <= max_label_1; ++k) {
                    similarityMatrix[k][i2] = 0;
                }
                similarityMatrix[i1][i2] = 1;
                break;
            } else {
                similarityMatrix[i1][i2] = 0;
            }
        }
    }
    
    // Replace -1 with 0
    for (auto& row : similarityMatrix) {
        std::replace(row.begin(), row.end(), -1, 0);
    }
    
    // Compute row and column sums
    std::vector<int> s_i(max_label_1 + 1, 0);
    std::vector<int> s_j(max_label_2 + 1, 0);
    
    for (int i1 = 0; i1 <= max_label_1; ++i1) {
        for (int i2 = 0; i2 <= max_label_2; ++i2) {
            s_i[i1] += similarityMatrix[i1][i2];
            s_j[i2] += similarityMatrix[i1][i2];
        }
    }
    
    return {s_i, s_j};
}

template<typename Point>
std::pair<std::vector<int>, std::vector<float>> getSimilarityScore(
    const std::vector<Point>& data,
    const std::vector<PhaseSpaceRow>& phase_space,
    const ClusterInfo& cluster_info,
    int n_dim
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t n = phase_space.size();
    std::vector<int> cli_similarityScore(cluster_info.index.size(), 0);
    std::vector<std::vector<float>> similarityScoreMatrix(n, std::vector<float>(n, 0.0f));
    std::vector<float> similarityScore(n, 0.0f);
    
    int progress_i = 0;
    printProgressBar(progress_i, n, "Postprocessing progress:", "Complete", 1, 50);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            if (i == j) continue;
            
            auto [s_i, s_j] = getSimilarityScore_ij(i, j, phase_space, cluster_info);
            
            if (s_i.empty()) continue;
            
            float score = std::accumulate(s_i.begin(), s_i.end(), 0);
            
            // Update cluster similarity scores
            for (size_t k = 0; k < cluster_info.index.size(); ++k) {
                if (cluster_info.index[k] == i && k < cli_similarityScore.size() && cluster_info.labels[k] < s_i.size()) {
                    cli_similarityScore[k] += s_i[cluster_info.labels[k]];
                }
                if (cluster_info.index[k] == j && k < cli_similarityScore.size() && cluster_info.labels[k] < s_j.size()) {
                    cli_similarityScore[k] += s_j[cluster_info.labels[k]];
                }
            }
            
            similarityScoreMatrix[j][i] = score;
            similarityScoreMatrix[i][j] = score;
        }
        
        progress_i++;
        printProgressBar(progress_i, n, "Progress:", "Complete", 1, 50);
    }
    
    // Sum rows for total similarity
    for (size_t i = 0; i < n; ++i) {
        similarityScore[i] = std::accumulate(similarityScoreMatrix[i].begin(), similarityScoreMatrix[i].end(), 0.0f);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Computing similarity scores: " << elapsed.count() << " seconds" << std::endl;
    
    return {cli_similarityScore, similarityScore};
}

template<typename Point>
std::pair<std::vector<int>, std::vector<float>> getSimilarityScoreByThreshold(
    const std::vector<Point>& data,
    const std::vector<PhaseSpaceRow>& phase_space,
    const ClusterInfo& cluster_info,
    int n_dim
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t n = phase_space.size();
    std::vector<int> cli_similarityScore(cluster_info.index.size(), 0);
    std::vector<std::vector<float>> similarityScoreMatrix(n, std::vector<float>(n, 0.0f));
    std::vector<float> similarityScore(n, 0.0f);
    
    // Get unique thresholds
    std::set<int> threshold_set;
    for (const auto& ps : phase_space) {
        threshold_set.insert(ps.threshold);
    }
    std::vector<int> threshold_list(threshold_set.begin(), threshold_set.end());
    
    int progress_i = 0;
    printProgressBar(progress_i, n, "Postprocessing progress:", "Complete", 1, 50);
    
    // Process by threshold
    for (int th : threshold_list) {
        // Get indices with this threshold
        std::vector<size_t> ps_th_indices;
        for (size_t i = 0; i < phase_space.size(); ++i) {
            if (phase_space[i].threshold == th) {
                ps_th_indices.push_back(i);
            }
        }
        
        for (size_t i : ps_th_indices) {
            for (size_t j : ps_th_indices) {
                if (j > i) continue;
                if (i == j) continue;
                
                auto [s_i, s_j] = getSimilarityScore_ij(i, j, phase_space, cluster_info);
                
                if (s_i.empty()) continue;
                
                float score = std::accumulate(s_i.begin(), s_i.end(), 0);
                
                // Update cluster similarity scores
                for (size_t k = 0; k < cluster_info.index.size(); ++k) {
                    if (cluster_info.index[k] == i && k < cli_similarityScore.size() && cluster_info.labels[k] < s_i.size()) {
                        cli_similarityScore[k] += s_i[cluster_info.labels[k]];
                    }
                    if (cluster_info.index[k] == j && k < cli_similarityScore.size() && cluster_info.labels[k] < s_j.size()) {
                        cli_similarityScore[k] += s_j[cluster_info.labels[k]];
                    }
                }
                
                similarityScoreMatrix[j][i] = score;
                similarityScoreMatrix[i][j] = score;
            }
            
            progress_i++;
            printProgressBar(progress_i, n, "Progress:", "Complete", 1, 50);
        }
    }
    
    // Sum rows for total similarity
    for (size_t i = 0; i < n; ++i) {
        similarityScore[i] = std::accumulate(similarityScoreMatrix[i].begin(), similarityScoreMatrix[i].end(), 0.0f);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Computing similarity scores: " << elapsed.count() << " seconds" << std::endl;
    
    return {cli_similarityScore, similarityScore};
}

// Explicit template instantiations
template ClusterInfo getClusterSizesAll<point2>(const std::vector<point2>&, const std::vector<PhaseSpaceRow>&, int);
template ClusterInfo getClusterSizesAll<point3>(const std::vector<point3>&, const std::vector<PhaseSpaceRow>&, int);

template std::pair<std::vector<int>, std::vector<float>> getSimilarityScore<point2>(
    const std::vector<point2>&, const std::vector<PhaseSpaceRow>&, const ClusterInfo&, int);
template std::pair<std::vector<int>, std::vector<float>> getSimilarityScore<point3>(
    const std::vector<point3>&, const std::vector<PhaseSpaceRow>&, const ClusterInfo&, int);

template std::pair<std::vector<int>, std::vector<float>> getSimilarityScoreByThreshold<point2>(
    const std::vector<point2>&, const std::vector<PhaseSpaceRow>&, const ClusterInfo&, int);
template std::pair<std::vector<int>, std::vector<float>> getSimilarityScoreByThreshold<point3>(
    const std::vector<point3>&, const std::vector<PhaseSpaceRow>&, const ClusterInfo&, int);

} // namespace finder
