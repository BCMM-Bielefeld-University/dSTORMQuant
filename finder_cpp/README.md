# FINDER — C++ implementation

High-performance C++ port of the **FINDER** meta-clustering algorithm for single-molecule localization microscopy (SMLM). Used as an optional clustering backend in the [dSTORMQuant pipeline](../README.md).

[License: MIT](https://opensource.org/licenses/MIT)
[C++20](https://en.cppreference.com/w/cpp/20)
[CMake](https://cmake.org/)

## Overview

FINDER selects global DBSCAN parameters for SMLM point clouds. This implementation provides:

- **Standalone CLI** — cluster CSV files from the command line
- **Python extension** — `pip install -e . --no-build-isolation` for use from the main pipeline (`clustering.ch*.method: finder`)
- **2D and 3D** point clouds
- **Header-only dependency** — [nanoflann](third_party/nanoflann/) for spatial indexing

### Reference

> Verzelli, Pietro, Andreas Nold, Chao Sun, Mike Heilemann, Erin M. Schuman, and Tatjana Tchumatchenko.  
> *Unbiased choice of global clustering parameters for single-molecule localization microscopy.*  
> Scientific Reports **12**, 22561 (2022).

[Paper](https://www.nature.com/articles/s41598-022-27074-1) · [Original Python code](https://github.com/NoldAndreas/FINDER)

---

## System requirements


| Component                   | Version                                   |
| --------------------------- | ----------------------------------------- |
| **C++ compiler**            | GCC 10+, Clang 12+, or MSVC 2019+ (C++20) |
| **CMake**                   | 3.15+                                     |
| **Python** (extension only) | 3.10+ with `pybind11` (see install below) |


**Tested on:** Windows 10/11, Ubuntu 24.04 LTS.

No GPU required. OpenMP is used when building the Python extension on supported platforms.

---

## Installation

### Option A — Python extension (recommended for dSTORMQuant pipeline)

From the repository root after `pip install -e .` (main package):

```bash
cd finder_cpp
pip install -r requirements-build.txt
pip install pybind11
pip install -e . --no-build-isolation
cd ..
```

`--no-build-isolation` uses your active environment’s build tools (e.g. `pybind11` from the main install) instead of an isolated build env.

Typical build time: **1 minutes** on a normal desktop.

Enable FINDER in the main pipeline: set `clustering.ch1.method` / `ch2.method` to `finder` and add a `finder` parameter block per channel in `config/config.yaml`. See the main [README](../README.md) (clustering section).

### Option B — Standalone CLI

**Windows:**

```powershell
.\build.bat
cd build
.\finder.exe input.csv output.csv
```

**Linux / macOS:**

```bash
chmod +x build.sh
./build.sh
cd build
./finder input.csv output.csv
```

**Manual CMake:**

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

---

## Project structure

```
finder_cpp/
├── src/                    # Implementation (finder, dbscan, bindings, CLI)
├── include/                # Public headers
├── examples/               # Sample CSV + run scripts
│   ├── sample2d.csv
│   ├── run_finder_example.sh
│   ├── run_finder_example.bat
│   └── README.md
├── docs/
│   └── CLI_USAGE.md        # Command-line reference
├── third_party/nanoflann/
├── build.sh / build.bat    # Build scripts
├── CMakeLists.txt
├── LICENSE                 # MIT (original FINDER + C++ port)
├── setup.py                # pybind11 extension
└── README.md
```

Build artifacts go under `build/` (gitignored).

---

## Command-line usage

```bash
# Basic 2D clustering
./finder input.csv output.csv

# Custom parameters
./finder input.csv output.csv --threshold 15 --ppd 20

# 3D data
./finder input.csv output.csv --dimension 3D

./finder --help
```


| Parameter      | Default    | Description                            |
| -------------- | ---------- | -------------------------------------- |
| `--threshold`  | 10         | Minimum points per cluster (minPts)    |
| `--ppd`        | 15         | Points per dimension (grid resolution) |
| `--algorithm`  | DbscanLoop | `DbscanLoop` or `Dbscan`               |
| `--min-thresh` | 5          | Minimum threshold in search range      |
| `--max-thresh` | 21         | Maximum threshold in search range      |
| `--decay`      | 0.5        | Decay for similarity scoring           |
| `--dimension`  | 2D         | `2D` or `3D`                           |


### Input / output CSV

**Input:** header row with `x`, `y` (and `z` for 3D); other columns are preserved.

**Output:** same columns plus `cluster` (`-1` = noise, `0+` = cluster id).

Full CLI details: [docs/CLI_USAGE.md](docs/CLI_USAGE.md).

---

## Examples

Small 2D sample and wrapper scripts live in `examples/`:

```bash
# Build first (from finder_cpp/)
./build.sh

cd examples
./run_finder_example.sh    # Linux / macOS
# or
run_finder_example.bat     # Windows
```

See [examples/README.md](examples/README.md).

For the **full dSTORMQuant pipeline** (drift, filtering, colocalization), download demo data per [../README.md](../README.md#3-demo).

---

## C++ library usage

```cpp
#include "finder.hpp"
#include <vector>

int main() {
    std::vector<point2> data;
    // ... load points ...

    finder::Finder<point2> finder(
        10,              // threshold (minPts)
        15,              // points_per_dimension
        "DbscanLoop",    // algorithm
        {5, 21},         // minmax_threshold
        "twoD",
        "threshold",
        false, true, false,
        0.5f,
        2                // n_dim
    );

    auto labels = finder.fit(data);
    auto params = finder.get_selected_parameters();
    return 0;
}
```

---

## Integration with dSTORMQuant


| Document                                                   | Description                            |
| ---------------------------------------------------------- | -------------------------------------- |
| [../README.md](../README.md)                               | Main package — install, config, demo (GPL-3.0) |
| [../THRID-PARTY LICENSE.txt](../THRID-PARTY%20LICENSE.txt) | Third-party and FINDER notices         |


FINDER is one clustering method alongside DBSCAN and HDBSCAN in `src/dSTORMQuant/analysis/clustering/`.

---

## Troubleshooting


| Issue                               | Action                                                                               |
| ----------------------------------- | ------------------------------------------------------------------------------------ |
| CMake / compiler errors             | Install C++20 toolchain and CMake 3.15+                                              |
| `pip install -e .` fails on Windows | Install MSVC build tools; ensure `pybind11` is installed; use `--no-build-isolation` |
| Import error in Python pipeline     | Rebuild: `cd finder_cpp && pip install -e . --no-build-isolation --force-reinstall`  |
| CLI not found                       | Run from `build/` or add `build` to `PATH`                                           |


---

## Citation

```bibtex
@article{verzelli2022unbiased,
  title={Unbiased choice of global clustering parameters for single-molecule localization microscopy},
  author={Verzelli, Pietro and Nold, Andreas and Sun, Chao and Heilemann, Mike and Schuman, Erin M and Tchumatchenko, Tatjana},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={22561},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

---

## License

- `**finder_cpp/` (this module):** [MIT License](LICENSE), matching the [original Python FINDER](https://github.com/NoldAndreas/FINDER) (Copyright © 2021 Andreas Nold). The C++ port is Copyright © 2026 Biochemistry and Molecular Medicine - Medical School OWL - Bielefeld University. Full license text: [LICENSE](LICENSE).
- **dSTORMQuant pipeline (repository root):** [GPL-3.0](../LICENSE). When you use `finder_cpp` as part of that package, the combined distribution is GPL-3.0; the MIT notice for FINDER-derived code must still be retained (see [THIRD-PARTY LICENSE.txt](../THRID-PARTY%20LICENSE.txt) §17).

## Authors

- **C++ implementation:** Suraj Karki (Bielefeld University)
- **Original algorithm:** Pietro Verzelli & Andreas Nold

