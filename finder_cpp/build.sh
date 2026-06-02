#!/bin/bash

# Build script for FINDER C++ on Linux/macOS

echo "======================================"
echo "FINDER C++ Build Script"
echo "======================================"
echo ""

# Check if build directory exists
if [ -d "build" ]; then
    echo "Removing old build directory..."
    rm -rf build
fi

echo "Creating build directory..."
mkdir build
cd build

echo ""
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: CMake configuration failed!"
    echo "Make sure you have CMake installed."
    exit 1
fi

echo ""
echo "Building..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "======================================"
echo "Build successful!"
echo "======================================"
echo ""
echo "Executable: build/finder"
echo ""
echo "To run clustering:"
echo "  cd build"
echo "  ./finder <input.csv> <output.csv> [2D|3D]"
echo ""
echo "Example:"
echo "  cd build"
echo "  ./finder ../examples/sample2d.csv results.csv"
echo ""
echo "For help:"
echo "  ./finder --help"
echo ""
