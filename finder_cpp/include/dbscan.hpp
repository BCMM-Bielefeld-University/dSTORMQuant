// FINDER C++ port. Upstream: NoldAndreas/FINDER (MIT, Copyright (c) 2021 Andreas Nold).
// Port: Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University.
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE. Provenance: repository ATTRIBUTION.md.

#pragma once

#include <cassert>
#include <cstddef>
#include <span>
#include <vector>
#include <cstdlib>

struct point2
{
    float x, y;
};

struct point3
{
    float x, y, z;
};

// Function declarations
auto dbscan(const std::span<const point2>& data, float eps, int min_pts) -> std::vector<std::vector<size_t>>;
auto dbscan(const std::span<const point3>& data, float eps, int min_pts) -> std::vector<std::vector<size_t>>;

/**
 * sklearn-compatible DBSCAN result: labels for all points plus core sample indices.
 * Core points have >= min_pts neighbors within eps (including self).
 * Border points are labeled with a cluster but are not cores; noise is -1.
 */
struct DbscanLabeledResult {
    std::vector<int> labels;
    std::vector<size_t> core_sample_indices;
};

auto dbscan_labeled(const std::span<const point2>& data, float eps, int min_pts) -> DbscanLabeledResult;
auto dbscan_labeled(const std::span<const point3>& data, float eps, int min_pts) -> DbscanLabeledResult;