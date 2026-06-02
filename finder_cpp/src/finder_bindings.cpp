// Port of FINDER (MIT, Copyright (c) 2021 Andreas Nold)
// C++ implementation Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE and https://github.com/NoldAndreas/FINDER

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "finder.hpp"
#include <vector>

namespace py = pybind11;

/**
 * Python wrapper for 2D FINDER clustering
 */
py::dict run_finder_2d(
    py::array_t<float, py::array::c_style | py::array::forcecast> coords,
    int threshold = 10,
    int points_per_dimension = 15,
    std::string algorithm = "dbscan",
    int min_threshold = 5,
    int max_threshold = 21,
    float decay = 0.5f
) {
    // Get buffer info
    py::buffer_info buf = coords.request();
    
    // Validate input
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D array");
    }
    if (buf.shape[1] != 2) {
        throw std::runtime_error("Input must have shape (n_points, 2) for 2D clustering");
    }
    
    size_t n_points = buf.shape[0];
    float* ptr = static_cast<float*>(buf.ptr);
    
    // Convert to point2 vector
    std::vector<point2> data(n_points);
    for (size_t i = 0; i < n_points; i++) {
        data[i].x = ptr[i * 2];
        data[i].y = ptr[i * 2 + 1];
    }
    
    // Release GIL for parallel processing
    py::gil_scoped_release release;
    
    // Create FINDER instance
    finder::Finder<point2> finder_obj(
        threshold,
        points_per_dimension,
        algorithm,
        {min_threshold, max_threshold},
        "twoD",
        "total",
        false,  // log_thresholds
        true,   // log_sigmas
        false,  // adaptive_sigma_boundaries
        decay,
        2       // n_dim
    );
    
    // Run clustering
    std::vector<int> labels = finder_obj.fit(data);
    
    // Get selected parameters
    const auto& selected_params = finder_obj.get_selected_parameters();
    
    // Reacquire GIL for Python object creation
    py::gil_scoped_acquire acquire;
    
    // Create labels array
    auto labels_array = py::array_t<int>(n_points);
    py::buffer_info labels_buf = labels_array.request();
    int* labels_ptr = static_cast<int*>(labels_buf.ptr);
    std::copy(labels.begin(), labels.end(), labels_ptr);
    
    // Create result dictionary with labels and selected parameters
    py::dict result;
    result["labels"] = labels_array;
    result["sigma"] = selected_params.sigma;
    result["threshold"] = selected_params.threshold;
    
    return result;
}

/**
 * Python wrapper for 3D FINDER clustering
 */
py::dict run_finder_3d(
    py::array_t<float, py::array::c_style | py::array::forcecast> coords,
    int threshold = 10,
    int points_per_dimension = 15,
    std::string algorithm = "dbscan",
    int min_threshold = 5,
    int max_threshold = 21,
    float decay = 0.5f
) {
    // Get buffer info
    py::buffer_info buf = coords.request();
    
    // Validate input
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D array");
    }
    if (buf.shape[1] != 3) {
        throw std::runtime_error("Input must have shape (n_points, 3) for 3D clustering");
    }
    
    size_t n_points = buf.shape[0];
    float* ptr = static_cast<float*>(buf.ptr);
    
    // Convert to point3 vector
    std::vector<point3> data(n_points);
    for (size_t i = 0; i < n_points; i++) {
        data[i].x = ptr[i * 3];
        data[i].y = ptr[i * 3 + 1];
        data[i].z = ptr[i * 3 + 2];
    }
    
    // Release GIL for parallel processing
    py::gil_scoped_release release;
    
    // Create FINDER instance
    finder::Finder<point3> finder_obj(
        threshold,
        points_per_dimension,
        algorithm,
        {min_threshold, max_threshold},
        "twoD",
        "threshold",
        false,  // log_thresholds
        true,   // log_sigmas
        false,  // adaptive_sigma_boundaries
        decay,
        3       // n_dim
    );
    
    // Run clustering
    std::vector<int> labels = finder_obj.fit(data);
    
    // Get selected parameters
    const auto& selected_params = finder_obj.get_selected_parameters();
    
    // Reacquire GIL for Python object creation
    py::gil_scoped_acquire acquire;
    
    // Create labels array
    auto labels_array = py::array_t<int>(n_points);
    py::buffer_info labels_buf = labels_array.request();
    int* labels_ptr = static_cast<int*>(labels_buf.ptr);
    std::copy(labels.begin(), labels.end(), labels_ptr);
    
    // Create result dictionary with labels and selected parameters
    py::dict result;
    result["labels"] = labels_array;
    result["sigma"] = selected_params.sigma;
    result["threshold"] = selected_params.threshold;
    
    return result;
}

// Python module definition
PYBIND11_MODULE(finder_cpp, m) {
    m.doc() = R"pbdoc(
        FINDER: FINding Density-based clustERs
        
        Python bindings for the FINDER clustering algorithm for 
        Single Molecule Localization Microscopy (SMLM) data.
    )pbdoc";
    
    m.def("run_finder_2d", &run_finder_2d,
          py::arg("coords"),
          py::arg("threshold") = 10,
          py::arg("points_per_dimension") = 15,
          py::arg("algorithm") = "dbscan",
          py::arg("min_threshold") = 5,
          py::arg("max_threshold") = 21,
          py::arg("decay") = 0.5f,
          R"pbdoc(
        Run FINDER clustering on 2D coordinates.
        
        Parameters
        ----------
        coords : np.ndarray
            2D array of shape (n_points, 2) containing x, y coordinates
        threshold : int, optional
            Minimum points for cluster (default: 10)
        points_per_dimension : int, optional
            Points per dimension for grid (default: 15)
        algorithm : str, optional
            Algorithm: "DbscanLoop" or "Dbscan" (default: "dbscan")
        min_threshold : int, optional
            Minimum threshold to search (default: 5)
        max_threshold : int, optional
            Maximum threshold to search (default: 21)
        decay : float, optional
            Decay parameter (default: 0.5)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'labels': np.ndarray of cluster labels (-1 for noise)
            - 'sigma': float, selected sigma parameter
            - 'threshold': int, selected threshold parameter
        
        Examples
        --------
        >>> import numpy as np
        >>> import finder_cpp
        >>> coords = np.random.rand(1000, 2).astype(np.float32)
        >>> result = finder_cpp.run_finder_2d(coords, threshold=10)
        >>> labels = result['labels']
        >>> print(f"Selected sigma: {result['sigma']}, threshold: {result['threshold']}")
    )pbdoc");
    
    m.def("run_finder_3d", &run_finder_3d,
          py::arg("coords"),
          py::arg("threshold") = 10,
          py::arg("points_per_dimension") = 15,
          py::arg("algorithm") = "dbscan",
          py::arg("min_threshold") = 5,
          py::arg("max_threshold") = 21,
          py::arg("decay") = 0.5f,
          R"pbdoc(
        Run FINDER clustering on 3D coordinates.
        
        Parameters
        ----------
        coords : np.ndarray
            2D array of shape (n_points, 3) containing x, y, z coordinates
        threshold : int, optional
            Minimum points for cluster (default: 10)
        points_per_dimension : int, optional
            Points per dimension for grid (default: 15)
        algorithm : str, optional
            Algorithm: "DbscanLoop" or "Dbscan" (default: "dbscan")
        min_threshold : int, optional
            Minimum threshold to search (default: 5)
        max_threshold : int, optional
            Maximum threshold to search (default: 21)
        decay : float, optional
            Decay parameter (default: 0.5)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'labels': np.ndarray of cluster labels (-1 for noise)
            - 'sigma': float, selected sigma parameter
            - 'threshold': int, selected threshold parameter
        
        Examples
        --------
        >>> import numpy as np
        >>> import finder_cpp
        >>> coords = np.random.rand(1000, 3).astype(np.float32)
        >>> result = finder_cpp.run_finder_3d(coords, threshold=10)
        >>> labels = result['labels']
        >>> print(f"Selected sigma: {result['sigma']}, threshold: {result['threshold']}")
    )pbdoc");
    
    m.attr("__version__") = "1.0.0";
}