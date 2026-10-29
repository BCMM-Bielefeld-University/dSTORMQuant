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

After building, the executable is at `build/finder` (or `build/Release/finder.exe` on some Windows generators).

## Command-Line Usage

### Basic Syntax

```bash
./finder <input_file> <output_file> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `input_file` | Path to input CSV with coordinate columns |
| `output_file` | Path for clustered results CSV |

### Optional Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dimension` | `2D` | `2D` or `3D` |
| `--threshold` | `10` | Minimum points per cluster (minPts) |
| `--ppd` | `15` | Points per dimension (grid resolution) |
| `--algorithm` | `DbscanLoop` | `DbscanLoop` or `Dbscan` |
| `--min-thresh` | `5` | Minimum threshold in search range |
| `--max-thresh` | `21` | Maximum threshold in search range |
| `--decay` | `0.5` | Decay for similarity scoring |
| `-h` / `--help` | — | Print help |

Legacy positional dimension (`./finder in.csv out.csv 3D`) is still accepted.

### Examples

```bash
# Basic 2D clustering
./finder input.csv output.csv

# Custom parameters
./finder input.csv output.csv --threshold 15 --ppd 20

# 3D data
./finder input.csv output.csv --dimension 3D

./finder --help
```

Sample CSVs for experiments are under `finder_cpp/examples/` (see `scripts/RUNNING_INSTRUCTIONS.txt`).

## Input CSV Format

Your input CSV must have a header row.

### For 2D Data

Required columns: `x`, `y` (other columns are preserved in the output).

### For 3D Data

Required columns: `x`, `y`, `z`.

## Output

The output CSV keeps original columns and adds cluster assignment information produced by FINDER.

## See Also

- [finder_cpp/README.md](../README.md) — install, Python bindings, parameter table
- [scripts/RUNNING_INSTRUCTIONS.txt](../scripts/RUNNING_INSTRUCTIONS.txt) — validation runs
