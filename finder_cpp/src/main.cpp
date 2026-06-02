// Port of FINDER (MIT, Copyright (c) 2021 Andreas Nold)
// C++ implementation Copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University
// SPDX-License-Identifier: MIT — see finder_cpp/LICENSE and https://github.com/NoldAndreas/FINDER

#include "finder.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

/**
 * Structure to hold CSV data with all columns preserved
 */
struct CSVData {
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> rows;
    int x_col = -1;
    int y_col = -1;
    int z_col = -1;
};

/**
 * Load CSV file and preserve all columns
 */
CSVData load_csv_full(const std::string& filename) {
    CSVData csv_data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return csv_data;
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        
        if (first_line) {
            // Store headers and find x, y, z columns
            csv_data.headers = row;
            
            for (size_t i = 0; i < row.size(); ++i) {
                std::string header = row[i];
                // Trim whitespace
                header.erase(std::remove_if(header.begin(), header.end(), ::isspace), header.end());
                
                if (header == "x") csv_data.x_col = static_cast<int>(i);
                else if (header == "y") csv_data.y_col = static_cast<int>(i);
                else if (header == "z") csv_data.z_col = static_cast<int>(i);
            }
            
            first_line = false;
            
            if (csv_data.x_col == -1 || csv_data.y_col == -1) {
                std::cerr << "Error: Could not find x and y columns in CSV" << std::endl;
                csv_data.headers.clear();
                return csv_data;
            }
            
            continue;
        }
        
        // Store all data rows
        if (!row.empty()) {
            csv_data.rows.push_back(row);
        }
    }
    
    file.close();
    return csv_data;
}

/**
 * Extract point data from CSV data
 */
template<typename Point>
std::vector<Point> extract_points(const CSVData& csv_data) {
    std::vector<Point> points;
    
    for (const auto& row : csv_data.rows) {
        if (row.size() <= static_cast<size_t>(std::max({csv_data.x_col, csv_data.y_col, csv_data.z_col}))) {
            continue;
        }
        
        try {
            if constexpr (std::is_same_v<Point, point2>) {
                Point p;
                p.x = std::stof(row[csv_data.x_col]);
                p.y = std::stof(row[csv_data.y_col]);
                points.push_back(p);
            } else {
                if (csv_data.z_col == -1) {
                    std::cerr << "Error: z column not found for 3D data" << std::endl;
                    return {};
                }
                Point p;
                p.x = std::stof(row[csv_data.x_col]);
                p.y = std::stof(row[csv_data.y_col]);
                p.z = std::stof(row[csv_data.z_col]);
                points.push_back(p);
            }
        } catch (...) {
            // Skip malformed rows
            continue;
        }
    }
    
    return points;
}

/**
 * Print cluster statistics
 */
void print_cluster_stats(const std::vector<int>& labels) {
    std::map<int, int> cluster_counts;
    for (int label : labels) {
        cluster_counts[label]++;
    }
    
    std::cout << "\n=== Cluster Statistics ===" << std::endl;
    std::cout << "Total points: " << labels.size() << std::endl;
    std::cout << "Number of clusters: " << (cluster_counts.size() - (cluster_counts.count(-1) > 0 ? 1 : 0)) << std::endl;
    
    if (cluster_counts.count(-1) > 0) {
        std::cout << "Noise points: " << cluster_counts[-1] << std::endl;
    }
    
    std::cout << "\nCluster sizes:" << std::endl;
    for (const auto& [label, count] : cluster_counts) {
        if (label >= 0) {
            std::cout << "  Cluster " << label << ": " << count << " points" << std::endl;
        }
    }
}

/**
 * Save results to CSV with all original columns preserved
 */
bool save_results(const std::string& filename,
                  const CSVData& csv_data,
                  const std::vector<int>& labels) {
    std::filesystem::path out_path(filename);
    if (out_path.has_parent_path() && !out_path.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
        if (ec) {
            std::cerr << "Error: Could not create output directory "
                      << out_path.parent_path() << ": " << ec.message() << std::endl;
            return false;
        }
    }

    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file " << filename << std::endl;
        return false;
    }
    
    // Write header with cluster column added
    for (size_t i = 0; i < csv_data.headers.size(); ++i) {
        file << csv_data.headers[i];
        if (i < csv_data.headers.size() - 1) {
            file << ",";
        }
    }
    file << ",cluster" << std::endl;
    
    // Write data rows with cluster labels
    for (size_t i = 0; i < csv_data.rows.size() && i < labels.size(); ++i) {
        for (size_t j = 0; j < csv_data.rows[i].size(); ++j) {
            file << csv_data.rows[i][j];
            if (j < csv_data.rows[i].size() - 1) {
                file << ",";
            }
        }
        file << "," << labels[i] << std::endl;
    }
    
    file.close();
    std::cout << "Results saved to: " << filename << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "=== FINDER C++ Implementation ===" << std::endl;
    std::cout << "Single Molecule Localization Microscopy Clustering" << std::endl;
    std::cout << "====================================\n" << std::endl;
    
    // Check for help flag
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " <input_file> <output_file> [options]" << std::endl;
        std::cout << "\nRequired Arguments:" << std::endl;
        std::cout << "  input_file              : Path to input CSV file" << std::endl;
        std::cout << "  output_file             : Path to output CSV file" << std::endl;
        std::cout << "\nOptional Arguments:" << std::endl;
        std::cout << "  --dimension <2D|3D>     : Data dimension (default: 2D)" << std::endl;
        std::cout << "  --threshold <int>       : Minimum points for cluster (default: 10)" << std::endl;
        std::cout << "  --ppd <int>             : Points per dimension for grid (default: 15)" << std::endl;
        std::cout << "  --algorithm <name>      : Algorithm: DbscanLoop or Dbscan (default: dbscan)" << std::endl;
        std::cout << "  --min-thresh <int>      : Minimum threshold to search (default: 5)" << std::endl;
        std::cout << "  --max-thresh <int>      : Maximum threshold to search (default: 21)" << std::endl;
        std::cout << "  --decay <float>         : Decay parameter (default: 0.5)" << std::endl;
        std::cout << "\nInput CSV Format:" << std::endl;
        std::cout << "  - 2D data: CSV must have 'x' and 'y' columns" << std::endl;
        std::cout << "  - 3D data: CSV must have 'x', 'y', and 'z' columns" << std::endl;
        std::cout << "  - All other columns are preserved in output" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << " input.csv output.csv" << std::endl;
        std::cout << "  " << argv[0] << " input.csv output.csv --dimension 3D" << std::endl;
        std::cout << "  " << argv[0] << " input.csv output.csv --threshold 15 --ppd 20" << std::endl;
        std::cout << "  " << argv[0] << " input.csv output.csv --min-thresh 3 --max-thresh 25 --decay 0.7" << std::endl;
        return 0;
    }
    
    // Parse command line arguments
    if (argc < 3) {
        std::cerr << "Error: Insufficient arguments!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> [options]" << std::endl;
        std::cerr << "Use -h or --help for more information." << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    // Default parameters
    bool use_3d = false;
    int threshold = 10;
    int points_per_dimension = 15;
    std::string algorithm = "dbscan";
    int min_threshold = 5;
    int max_threshold = 21;
    float decay = 0.5f;
    
    // Parse optional arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--dimension" && i + 1 < argc) {
            std::string dim = argv[++i];
            use_3d = (dim == "3d" || dim == "3D");
        }
        else if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::atoi(argv[++i]);
        }
        else if (arg == "--ppd" && i + 1 < argc) {
            points_per_dimension = std::atoi(argv[++i]);
        }
        else if (arg == "--algorithm" && i + 1 < argc) {
            algorithm = argv[++i];
        }
        else if (arg == "--min-thresh" && i + 1 < argc) {
            min_threshold = std::atoi(argv[++i]);
        }
        else if (arg == "--max-thresh" && i + 1 < argc) {
            max_threshold = std::atoi(argv[++i]);
        }
        else if (arg == "--decay" && i + 1 < argc) {
            decay = std::atof(argv[++i]);
        }
        // Legacy support for positional dimension argument
        else if (i == 3 && (arg == "2D" || arg == "3D" || arg == "2d" || arg == "3d")) {
            use_3d = (arg == "3d" || arg == "3D");
        }
    }
    
    std::cout << "Input file: " << input_file << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "Dimension: " << (use_3d ? "3D" : "2D") << std::endl;
    std::cout << "\nParameters:" << std::endl;
    std::cout << "  Threshold (minPts): " << threshold << std::endl;
    std::cout << "  Points per dimension: " << points_per_dimension << std::endl;
    std::cout << "  Algorithm: " << algorithm << std::endl;
    std::cout << "  Threshold search range: [" << min_threshold << ", " << max_threshold << "]" << std::endl;
    std::cout << "  Decay: " << decay << "\n" << std::endl;
    
    // Load CSV with all columns
    std::cout << "Loading CSV data..." << std::endl;
    auto csv_data = load_csv_full(input_file);
    
    if (csv_data.headers.empty()) {
        std::cerr << "Error: Failed to load CSV file!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << csv_data.headers.size() << " columns in CSV" << std::endl;
    std::cout << "Loaded " << csv_data.rows.size() << " rows\n" << std::endl;
    
    if (use_3d) {
        // Check if z column exists for 3D
        if (csv_data.z_col == -1) {
            std::cerr << "Error: Could not find z column for 3D data" << std::endl;
            return 1;
        }
        
        // 3D clustering
        std::cout << "Extracting 3D coordinates..." << std::endl;
        auto data = extract_points<point3>(csv_data);
        
        if (data.empty()) {
            std::cerr << "Error: No valid data points extracted!" << std::endl;
            return 1;
        }
        
        std::cout << "Extracted " << data.size() << " valid points\n" << std::endl;
        
        // Create and configure FINDER
        finder::Finder<point3> finder(
            threshold,              // threshold (minPts)
            points_per_dimension,   // points_per_dimension
            algorithm,              // algorithm
            {min_threshold, max_threshold},  // minmax_threshold
            "twoD",                 // one_two_d
            "threshold",            // similarity_score_computation
            false,                  // log_thresholds
            true,                   // log_sigmas
            false,                  // adaptive_sigma_boundaries
            decay,                  // decay
            3                       // n_dim
        );
        
        // Fit the model
        std::cout << "Running FINDER clustering algorithm..." << std::endl;
        auto labels = finder.fit(data);
        
        // Print results
        print_cluster_stats(labels);
        
        // Save results with all original columns
        if (!save_results(output_file, csv_data, labels)) {
            return 1;
        }

    } else {
        // 2D clustering
        std::cout << "Extracting 2D coordinates..." << std::endl;
        auto data = extract_points<point2>(csv_data);
        
        if (data.empty()) {
            std::cerr << "Error: No valid data points extracted!" << std::endl;
            return 1;
        }
        
        std::cout << "Extracted " << data.size() << " valid points\n" << std::endl;
        
        // Create and configure FINDER
        finder::Finder<point2> finder(
            threshold,              // threshold (minPts)
            points_per_dimension,   // points_per_dimension
            algorithm,              // algorithm
            {min_threshold, max_threshold},  // minmax_threshold
            "twoD",                 // one_two_d
            "total",            // similarity_score_computation
            false,                  // log_thresholds
            true,                   // log_sigmas
            false,                  // adaptive_sigma_boundaries
            decay,                  // decay
            2                       // n_dim
        );
        
        // Fit the model
        std::cout << "Running FINDER clustering algorithm..." << std::endl;
        auto labels = finder.fit(data);
        
        // Print results
        print_cluster_stats(labels);
        
        // Save results with all original columns
        if (!save_results(output_file, csv_data, labels)) {
            return 1;
        }
    }

    std::cout << "\n=== FINDER Complete ===" << std::endl;

    return 0;
}
