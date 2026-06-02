# FINDER Command-Line Tool Usage Guide

## Overview
This guide explains how to use the FINDER command-line executable for clustering Single Molecule Localization Microscopy (SMLM) data.

## Building the Project

### Prerequisites
- C++20 compatible compiler (GCC 10+, Clang 12+, or MSVC 2019+)
- CMake 3.15 or higher

### Build Instructions

#### Linux/macOS
```bash
chmod +x build.sh
./build.sh
```

#### Windows
```batch
build.bat
```

#### Manual Build
```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

After building, the executable will be located at `build/finder`.

## Command-Line Usage

### Basic Syntax
```bash
./finder <input_file> <output_file> [dimension]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input_file` | ✅ Yes | - | Path to input CSV file with point coordinates |
| `output_file` | ✅ Yes | - | Path where clustered results will be saved |
| `dimension` | ❌ No | `2D` | Use `3D` or `3d` for 3D data, otherwise 2D |

### Examples

#### 2D Data (Default)
```bash
# Process 2D data
./finder input.csv output.csv

# Explicitly specify 2D
./finder input.csv output.csv 2D
```

#### 3D Data
```bash
# Process 3D data
./finder input_3d.csv output_3d.csv 3D
```

#### With Full Paths
```bash
# Using absolute paths
./finder /path/to/data/input.csv /path/to/results/output.csv

# Using relative paths
./finder ../data/points.csv ../results/clusters.csv
```

## Input CSV Format

Your input CSV file must have a header row with coordinate columns.

### For 2D Data
Required columns:
- `x` - X coordinate
- `y` - Y coordinate

Example:
```csv
x,y
10.5,20.3
15.2,22.1
12.8,19.5
```

### For 3D Data
Required columns:
- `x` - X coordinate
- `y` - Y coordinate
- `z` - Z coordinate

Example:
```csv
x,y,z
10.5,20.3,5.2
15.2,22.1,4.8
12.8,19.5,5.5
```

**Note:** Column names are case-sensitive. Additional columns are ignored.

## Output Format

The output CSV file will contain the original coordinates plus a cluster label:

### 2D Output
```csv
x,y,cluster
10.5,20.3,0
15.2,22.1,0
12.8,19.5,1
```

### 3D Output
```csv
x,y,z,cluster
10.5,20.3,5.2,0
15.2,22.1,4.8,0
12.8,19.5,5.5,1
```

**Cluster Labels:**
- `-1` = Noise points (not assigned to any cluster)
- `0, 1, 2, ...` = Cluster IDs

## Help
```bash
./finder --help
# or
./finder -h
```

## Example Workflow

```bash
# 1. Build the project
cd /path/to/finder_cpp
./build.sh

# 2. Navigate to build directory
cd build

# 3. Run clustering on example data
./finder ../examples/AHA_24014_U2OS-WT_AST0126_AST0063_posXY1_channels_t0_posZ0_after_cell_detection.csv results.csv

# 4. Check the output
cat results.csv
```

## Algorithm Parameters

The FINDER algorithm uses these default parameters (defined in `src/main.cpp`):
- **threshold (minPts)**: 10 - Minimum points to form a cluster
- **points_per_dimension**: 15 - Grid resolution for parameter search
- **algorithm**: DbscanLoop - Clustering algorithm variant
- **minmax_threshold**: [5, 21] - Range for threshold search
- **similarity_score_computation**: threshold - Method for computing similarity

To modify these parameters, edit the `src/main.cpp` file and rebuild.

## Troubleshooting

### "Error: Could not open file"
- Check that the input file path is correct
- Ensure you have read permissions for the input file

### "Error: Could not find x and y columns in CSV"
- Verify your CSV has a header row
- Check that columns are named exactly `x` and `y` (lowercase)
- Ensure the CSV is comma-separated

### "Error: No data loaded from file!"
- Check that your CSV file is not empty
- Verify the CSV format is valid
- Ensure there are no parsing errors in your data rows

### "Error: Could not open output file"
- Check that the output directory exists
- Ensure you have write permissions for the output location

## Performance Tips

- For large datasets (>100,000 points), the clustering may take several minutes
- Consider using Release build for better performance: `cmake -DCMAKE_BUILD_TYPE=Release ..`
- Progress messages will be displayed during execution

## Additional Resources

- **Project README**: `../README.md`
