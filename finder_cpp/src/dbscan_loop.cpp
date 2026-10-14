// FINDER C++ port. Upstream: NoldAndreas/FINDER (MIT, Copyright (c) 2021 Andreas Nold).
// Port: Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University.
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE. Provenance: repository ATTRIBUTION.md.

#include "dbscan_loop.hpp"
#include <numeric>
#include <span>

namespace finder {

DbscanLoop::DbscanLoop(float eps, int min_samples)
    : eps_(eps), min_samples_(min_samples) {}

template<typename Point>
DbscanLoop& DbscanLoop::fit_impl(const std::vector<Point>& data) {
    labels_.assign(data.size(), -1);
    if (data.empty()) {
        return *this;
    }

    // Mirror finder_smlm DbscanLoop.fit exactly:
    //   DB = DBSCAN(...).fit(XS)
    //   XS = XS[DB.core_sample_indices_]
    //   while shrinking:
    //       idx_core = idx_core[DB.core_sample_indices_]
    //       DB = DBSCAN(...).fit(XS)
    //       XS = XS[DB.core_sample_indices_]
    //   labels_[idx_core] = DB.labels_

    std::vector<Point> xs = data;
    std::vector<size_t> idx_core(data.size());
    std::iota(idx_core.begin(), idx_core.end(), 0);

    size_t n_old = xs.size();
    auto result = dbscan_labeled(std::span<const Point>(xs.data(), xs.size()), eps_, min_samples_);

    // First shrink to cores (like Python before the while)
    {
        std::vector<Point> next_xs;
        next_xs.reserve(result.core_sample_indices.size());
        for (size_t ci : result.core_sample_indices) {
            next_xs.push_back(xs[ci]);
        }
        xs = std::move(next_xs);
    }

    while (n_old != xs.size() && !xs.empty()) {
        n_old = xs.size();

        // Map idx_core through previous core indices, then refit
        std::vector<size_t> next_idx;
        next_idx.reserve(result.core_sample_indices.size());
        for (size_t ci : result.core_sample_indices) {
            next_idx.push_back(idx_core[ci]);
        }
        idx_core = std::move(next_idx);

        result = dbscan_labeled(std::span<const Point>(xs.data(), xs.size()), eps_, min_samples_);

        std::vector<Point> next_xs;
        next_xs.reserve(result.core_sample_indices.size());
        for (size_t ci : result.core_sample_indices) {
            next_xs.push_back(xs[ci]);
        }
        xs = std::move(next_xs);
    }

    // Assign labels from the last fit to the corresponding original indices
    if (idx_core.size() == result.labels.size()) {
        for (size_t i = 0; i < idx_core.size(); ++i) {
            labels_[idx_core[i]] = result.labels[i];
        }
    }

    return *this;
}

DbscanLoop& DbscanLoop::fit(const std::vector<point2>& data) {
    return fit_impl(data);
}

DbscanLoop& DbscanLoop::fit(const std::vector<point3>& data) {
    return fit_impl(data);
}

} // namespace finder
