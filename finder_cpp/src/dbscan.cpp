// FINDER C++ port. Upstream: NoldAndreas/FINDER (MIT, Copyright (c) 2021 Andreas Nold).
// Port: Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University.
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE. Provenance: repository ATTRIBUTION.md.

#include "dbscan.hpp"

#include <cstddef>
#include <nanoflann/nanoflann.hpp>
#include <vector>
#include <algorithm>
#include <iterator>

inline auto get_pt(const point2& p, std::size_t dim)
{
    if (dim == 0) return p.x;
    return p.y;
}

inline auto get_pt(const point3& p, std::size_t dim)
{
    if (dim == 0) return p.x;
    if (dim == 1) return p.y;
    return p.z;
}

template<typename Point>
struct adaptor
{
    const std::span<const Point>& points;
    adaptor(const std::span<const Point>& points) : points(points) {}

    inline std::size_t kdtree_get_point_count() const { return points.size(); }

    inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
    {
        return get_pt(points[idx], dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

    auto const* elem_ptr(const std::size_t idx) const
    {
        return &points[idx].x;
    }
};

auto sort_clusters(std::vector<std::vector<size_t>>& clusters)
{
    for (auto& cluster : clusters) {
        std::sort(cluster.begin(), cluster.end());
    }
}

template<int n_cols, typename Adaptor>
auto dbscan(const Adaptor& adapt, float eps, int min_pts)
{
    eps *= eps;
    using namespace nanoflann;
    using my_kd_tree_t = KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<float, decltype(adapt)>, decltype(adapt), n_cols>;

    auto index = my_kd_tree_t(n_cols, adapt, KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    const auto n_points = adapt.kdtree_get_point_count();
    auto visited = std::vector<bool>(n_points);
    auto clusters = std::vector<std::vector<size_t>>();
    auto matches = std::vector<std::pair<size_t, float>>();
    auto sub_matches = std::vector<std::pair<size_t, float>>();

    for (size_t i = 0; i < n_points; i++) {
        if (visited[i]) continue;

        index.radiusSearch(adapt.elem_ptr(i), eps, matches, SearchParams(32, 0.f, false));
        if (matches.size() < static_cast<size_t>(min_pts)) continue;
        visited[i] = true;

        auto cluster = std::vector({i});

        while (!matches.empty()) {
            auto nb_idx = matches.back().first;
            matches.pop_back();
            if (visited[nb_idx]) continue;
            visited[nb_idx] = true;

            index.radiusSearch(adapt.elem_ptr(nb_idx), eps, sub_matches, SearchParams(32, 0.f, false));

            if (sub_matches.size() >= static_cast<size_t>(min_pts)) {
                std::copy(sub_matches.begin(), sub_matches.end(), std::back_inserter(matches));
            }
            cluster.push_back(nb_idx);
        }
        clusters.emplace_back(std::move(cluster));
    }
    sort_clusters(clusters);
    return clusters;
}

template<int n_cols, typename Adaptor>
auto dbscan_labeled_impl(const Adaptor& adapt, float eps, int min_pts) -> DbscanLabeledResult
{
    eps *= eps;
    using namespace nanoflann;
    using my_kd_tree_t = KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<float, decltype(adapt)>, decltype(adapt), n_cols>;

    auto index = my_kd_tree_t(n_cols, adapt, KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    const auto n_points = adapt.kdtree_get_point_count();
    auto visited = std::vector<bool>(n_points);
    auto is_core = std::vector<bool>(n_points, false);
    auto labels = std::vector<int>(n_points, -1);
    auto matches = std::vector<std::pair<size_t, float>>();
    auto sub_matches = std::vector<std::pair<size_t, float>>();

    int cluster_id = 0;
    for (size_t i = 0; i < n_points; i++) {
        if (visited[i]) continue;

        index.radiusSearch(adapt.elem_ptr(i), eps, matches, SearchParams(32, 0.f, false));
        if (matches.size() < static_cast<size_t>(min_pts)) continue;
        visited[i] = true;
        is_core[i] = true;
        labels[i] = cluster_id;

        while (!matches.empty()) {
            auto nb_idx = matches.back().first;
            matches.pop_back();
            if (visited[nb_idx]) continue;
            visited[nb_idx] = true;

            index.radiusSearch(adapt.elem_ptr(nb_idx), eps, sub_matches, SearchParams(32, 0.f, false));

            if (sub_matches.size() >= static_cast<size_t>(min_pts)) {
                is_core[nb_idx] = true;
                std::copy(sub_matches.begin(), sub_matches.end(), std::back_inserter(matches));
            }
            labels[nb_idx] = cluster_id;
        }
        ++cluster_id;
    }

    DbscanLabeledResult result;
    result.labels = std::move(labels);
    for (size_t i = 0; i < n_points; ++i) {
        if (is_core[i]) {
            result.core_sample_indices.push_back(i);
        }
    }
    return result;
}

auto dbscan(const std::span<const point2>& data, float eps, int min_pts)
    -> std::vector<std::vector<size_t>>
{
    const auto adapt = adaptor<point2>(data);
    return dbscan<2>(adapt, eps, min_pts);
}

auto dbscan(const std::span<const point3>& data, float eps, int min_pts)
    -> std::vector<std::vector<size_t>>
{
    const auto adapt = adaptor<point3>(data);
    return dbscan<3>(adapt, eps, min_pts);
}

auto dbscan_labeled(const std::span<const point2>& data, float eps, int min_pts)
    -> DbscanLabeledResult
{
    const auto adapt = adaptor<point2>(data);
    return dbscan_labeled_impl<2>(adapt, eps, min_pts);
}

auto dbscan_labeled(const std::span<const point3>& data, float eps, int min_pts)
    -> DbscanLabeledResult
{
    const auto adapt = adaptor<point3>(data);
    return dbscan_labeled_impl<3>(adapt, eps, min_pts);
}
