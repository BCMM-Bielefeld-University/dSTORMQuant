from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os

# Get the directory containing your FINDER C++ source files
finder_src_dir = "./src"  # Source directory
include_dir = "./include"  # Include directory

# Collect all FINDER source files
finder_sources = [
    os.path.join(finder_src_dir, "finder_bindings.cpp"),
    os.path.join(finder_src_dir, "finder.cpp"),
    os.path.join(finder_src_dir, "dbscan.cpp"),
    os.path.join(finder_src_dir, "dbscan_loop.cpp"),
    os.path.join(finder_src_dir, "auxiliary_functions.cpp"),
    os.path.join(finder_src_dir, "similarity_score.cpp"),
]

ext_modules = [
    Pybind11Extension(
        "finder_cpp",
        finder_sources,
        include_dirs=[
            pybind11.get_include(),
            include_dir,  # Include FINDER headers
            "./third_party",  # Third-party libraries (nanoflann)
        ],
        extra_compile_args=[
            "-O3",              # Optimization
            "-march=native",    # Use CPU-specific optimizations
            "-fopenmp",         # OpenMP support for multi-threading
            "-std=c++20",       # C++20 standard
            "-ffast-math",      # Fast math operations
            "-DNDEBUG",         # Disable debug assertions
        ],
        extra_link_args=[
            "-fopenmp",         # Link OpenMP
        ],
        language="c++",
        cxx_std=20,
    ),
]

setup(
    name="finder_cpp",
    version="1.0.0",
    author="Suraj Karki",
    author_email="suraj.karki500@gmail.com",
    description="FINDER clustering algorithm with Python bindings for SMLM data",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pybind11",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)