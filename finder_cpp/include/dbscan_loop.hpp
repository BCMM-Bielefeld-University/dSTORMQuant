// FINDER C++ port (DBSCAN loop helpers). Upstream: NoldAndreas/FINDER (MIT, 2021).
// Port: Suraj Karki, Bielefeld University, 2026. SPDX-License-Identifier: MIT
// See finder_cpp/LICENSE and repository ATTRIBUTION.md.

#pragma once

#include "dbscan.hpp"
#include <vector>

namespace finder {

/**
 * Noise-free DBSCAN (DbscanLoop), matching finder_smlm / verzep/finder_code.
 *
 * Iteratively keeps only sklearn-style core samples until the set stabilizes,
 * then assigns labels from the last DBSCAN fit to those core points. All other
 * points remain noise (-1).
 */
class DbscanLoop {
public:
    DbscanLoop(float eps, int min_samples);

    DbscanLoop& fit(const std::vector<point2>& data);
    DbscanLoop& fit(const std::vector<point3>& data);

    const std::vector<int>& get_labels() const { return labels_; }

private:
    float eps_;
    int min_samples_;
    std::vector<int> labels_;

    template<typename Point>
    DbscanLoop& fit_impl(const std::vector<Point>& data);
};

} // namespace finder
