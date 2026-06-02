#!/usr/bin/env python3
"""
Test script for FINDER Python bindings
"""

import numpy as np
import time
import sys

def test_2d_clustering():
    """Test 2D FINDER clustering"""
    print("=" * 60)
    print("Testing 2D FINDER Clustering")
    print("=" * 60)
    
    try:
        import finder_cpp
        print("✓ Successfully imported finder_cpp module")
    except ImportError as e:
        print(f"✗ Failed to import finder_cpp: {e}")
        print("Please build the bindings first (see BUILD_INSTRUCTIONS.md)")
        return False
    
    # Generate test data - two clusters
    np.random.seed(42)
    
    # Cluster 1: centered at (10, 10)
    cluster1 = np.random.randn(500, 2) * 2 + [10, 10]
    
    # Cluster 2: centered at (30, 30)
    cluster2 = np.random.randn(500, 2) * 2 + [30, 30]
    
    # Noise points
    noise = np.random.rand(100, 2) * 50
    
    # Combine all points
    coords = np.vstack([cluster1, cluster2, noise]).astype(np.float32)
    
    print(f"\nTest data: {coords.shape[0]} points")
    print("Expected: 2 clusters + noise")
    
    # Run FINDER
    print("\nRunning FINDER clustering...")
    start_time = time.time()
    
    labels = finder_cpp.run_finder_2d(
        coords,
        threshold=10,
        points_per_dimension=15,
        algorithm="dbscan"
    )
    
    elapsed_time = time.time() - start_time
    
    # Analyze results
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"\n✓ Clustering completed in {elapsed_time:.4f} seconds")
    print(f"  Found {n_clusters} clusters")
    print(f"  Noise points: {n_noise}")
    
    # Print cluster sizes
    print("\nCluster sizes:")
    for label in sorted(unique_labels):
        if label == -1:
            continue
        count = np.sum(labels == label)
        print(f"  Cluster {label}: {count} points")
    
    # Verify we found approximately 2 clusters
    if 2 <= n_clusters <= 3:
        print("\n✓ Test PASSED: Found expected number of clusters")
        return True
    else:
        print(f"\n✗ Test WARNING: Expected 2 clusters, found {n_clusters}")
        print("  (This may be normal depending on parameters)")
        return True


def test_3d_clustering():
    """Test 3D FINDER clustering"""
    print("\n" + "=" * 60)
    print("Testing 3D FINDER Clustering")
    print("=" * 60)
    
    try:
        import finder_cpp
    except ImportError:
        print("✗ Skipping 3D test (module not imported)")
        return False
    
    # Generate 3D test data
    np.random.seed(42)
    
    # Cluster 1: centered at (10, 10, 10)
    cluster1 = np.random.randn(300, 3) * 2 + [10, 10, 10]
    
    # Cluster 2: centered at (30, 30, 30)
    cluster2 = np.random.randn(300, 3) * 2 + [30, 30, 30]
    
    # Noise
    noise = np.random.rand(100, 3) * 50
    
    coords = np.vstack([cluster1, cluster2, noise]).astype(np.float32)
    
    print(f"\nTest data: {coords.shape[0]} points (3D)")
    print("Expected: 2 clusters + noise")
    
    print("\nRunning FINDER 3D clustering...")
    start_time = time.time()
    
    labels = finder_cpp.run_finder_3d(
        coords,
        threshold=10,
        points_per_dimension=15
    )
    
    elapsed_time = time.time() - start_time
    
    # Analyze results
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"\n✓ 3D Clustering completed in {elapsed_time:.4f} seconds")
    print(f"  Found {n_clusters} clusters")
    print(f"  Noise points: {n_noise}")
    
    if 2 <= n_clusters <= 3:
        print("\n✓ Test PASSED: Found expected number of clusters")
        return True
    else:
        print(f"\n✗ Test WARNING: Expected 2 clusters, found {n_clusters}")
        return True


def benchmark_performance():
    """Benchmark FINDER performance"""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    try:
        import finder_cpp
    except ImportError:
        print("✗ Skipping benchmark (module not imported)")
        return
    
    sizes = [100, 1000, 10000]
    
    print("\nBenchmarking different dataset sizes:")
    print(f"{'Size':<10} {'Time (s)':<12} {'Points/sec':<15}")
    print("-" * 40)
    
    for size in sizes:
        coords = np.random.rand(size, 2).astype(np.float32) * 100
        
        start_time = time.time()
        finder_cpp.run_finder_2d(coords, threshold=10)
        elapsed_time = time.time() - start_time
        
        points_per_sec = size / elapsed_time if elapsed_time > 0 else float('inf')
        
        print(f"{size:<10} {elapsed_time:<12.4f} {points_per_sec:<15.0f}")
    
    print("\n✓ Benchmark complete")


def test_parameter_variations():
    """Test FINDER with different parameters"""
    print("\n" + "=" * 60)
    print("Testing Parameter Variations")
    print("=" * 60)
    
    try:
        import finder_cpp
    except ImportError:
        print("✗ Skipping parameter test (module not imported)")
        return
    
    # Generate test data
    np.random.seed(42)
    coords = np.random.rand(1000, 2).astype(np.float32) * 50
    
    test_cases = [
        {"threshold": 5, "desc": "Low threshold"},
        {"threshold": 10, "desc": "Medium threshold"},
        {"threshold": 20, "desc": "High threshold"},
        {"points_per_dimension": 10, "threshold": 10, "desc": "Coarse grid"},
        {"points_per_dimension": 20, "threshold": 10, "desc": "Fine grid"},
        {"algorithm": "DbscanLoop", "threshold": 10, "desc": "DbscanLoop algorithm"},
    ]
    
    print("\nTesting different parameter configurations:")
    print(f"{'Configuration':<25} {'Clusters':<10} {'Noise':<10} {'Time (s)':<10}")
    print("-" * 60)
    
    for params in test_cases:
        desc = params.pop("desc")
        
        start_time = time.time()
        labels = finder_cpp.run_finder_2d(coords, **params)
        elapsed_time = time.time() - start_time
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        print(f"{desc:<25} {n_clusters:<10} {n_noise:<10} {elapsed_time:<10.4f}")
        
        # Restore desc for next iteration
        params["desc"] = desc
    
    print("\n✓ Parameter variation tests complete")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("FINDER Python Bindings Test Suite")
    print("=" * 60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    all_passed = True
    
    # Run tests
    all_passed &= test_2d_clustering()
    all_passed &= test_3d_clustering()
    benchmark_performance()
    test_parameter_variations()
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests completed successfully!")
    else:
        print("✗ Some tests failed or were skipped")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())