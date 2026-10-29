# Attribution & Licenses

This repository reuses and adapts open-source algorithms and libraries for SMLM post-processing and analysis. We acknowledge and thank the original authors.

This file documents **provenance**, **what dSTORMQuant changed**, and **whether those changes are expected to alter numerical results** relative to upstream. Full third-party license texts are in [THRID-PARTY LICENSE.txt](./THRID-PARTY%20LICENSE.txt) (and [finder_cpp/LICENSE](./finder_cpp/LICENSE) for the FINDER port).

---

## AIM drift correction (Picasso)

**Algorithm:** Adaptive Intersection Maximization (AIM) — Ma *et al.*, *Science Advances*, 2024  
**Reference MATLAB:** [YangLiuLab/AIM](https://github.com/YangLiuLab/AIM)  
**Source (Python port):** Picasso `picasso.aim` — [jungmannlab/picasso](https://github.com/jungmannlab/picasso) (Schnitzbauer *et al.*, 2017; Jungmann Lab, MPI of Biochemistry)  
**Paths in this repo:**

| Component                 | Path                                                   |
| ------------------------- | ------------------------------------------------------ |
| Adapted AIM module        | `src/dSTORMQuant/processing/drift_correction/AIM.py`   |
| Per-file AIM driver       | `src/dSTORMQuant/processing/drift_correction/drift.py` |
| Dual-channel registration | `src/dSTORMQuant/core/pipeline.py`                     |

**License:** See Picasso / Jungmann Lab notice in [THRID-PARTY LICENSE.txt](./THRID-PARTY%20LICENSE.txt) §16 and the copyright header in `AIM.py`.

### What we changed

- CSV ↔ HDF5 conversion and logging for this pipeline’s localization schema
- Optional localization sanity filters before AIM
- Optional segment drift validation: replace segments whose relative drift exceeds a nm threshold with the mean of preceding valid segments
- Dual-channel workflow (pipeline): undrift each channel independently, then subtract cumulative channel-1 drift from channel-2 coordinates
- **Configurable AIM implementation options** via `drift_correction.aim_options` in `config.yaml` (defaults retain the current dSTORMQuant behaviour):
  - `fft_peak_scale: search_diameter` — scales FFT peak bins by `(2 * roi_r) / box`. Use `intersect_d` for the upstream Picasso AIM convention.
  - `exclude_self_round2: false` — round 2 retains the full reference. Use `true` for the Picasso AIM leave-one-segment-out procedure.
  - `drift_reference: frame0` — anchors the drift curve at the first frame. Use `mean` for Picasso AIM mean-centring.

### Effect on results

- Under default `aim_options`, undrift remains related to Picasso AIM but is not bit-identical, even with drift validation disabled.
- Selecting `fft_peak_scale: intersect_d`, `exclude_self_round2: true`, and `drift_reference: mean` aligns these three numerical choices with upstream Picasso AIM.
- Enabling segment validation can change the drift trace and the corrected coordinates.
- Dual-channel cumulative-drift transfer changes channel-2 absolute coordinates relative to running AIM on each channel alone. It does not re-run AIM.

### Validation

- Qualitative QC via Napari / drift plots in the pipeline

---

## Coordinate-based colocalization (CBC) via locan

**Algorithm:** Coordinate-Based Colocalization — Malkusch *et al.*, 2012  
**Source:** `[locan](https://github.com/super-resolution/Locan)` / LOCAN (Doose, 2022) — `CoordinateBasedColocalization`  
**Path in this repo:** `src/dSTORMQuant/analysis/colocalization/CBC.py`

| Piece              | Role                                                                     |
| ------------------ | ------------------------------------------------------------------------ |
| `CBC.py`           | Thin wrapper: column rename, bidirectional orchestration, plot packaging |
| `locan` dependency | CBC estimator                                                            |

**License:** BSD 3-Clause License (locan) — full text in [THRID-PARTY LICENSE.txt](./THRID-PARTY%20LICENSE.txt) §1.

### What we changed

- Rename `x`/`y` → `position_x`/`position_y` for locan
- Bidirectional Ch1↔Ch2 orchestration and packaging of results for plotting

### Effect on results

- Wrapper changes must not alter CBC scores for the same points, radius, and `n_steps`

---

## Relative enrichment (Ejdrup)

**Algorithm / code:** Relative enrichment — Ejdrup *et al.*, *Nature Communications*, 2022  
([doi:10.1038/s41467-022-32064-y](https://doi.org/10.1038/s41467-022-32064-y))  
**Author (upstream):** Aske Lykke Ejdrup  
**Paths in this repo:**

| Component                   | Path                                                                            |
| --------------------------- | ------------------------------------------------------------------------------- |
| Upstream RE core            | `src/dSTORMQuant/analysis/colocalization/RE_function.py`                        |
| Pipeline wrapper            | `src/dSTORMQuant/analysis/colocalization/relative_enrichment_colocalization.py` |

**License:** MIT License

```
MIT License

Copyright (c) 2022 Aske Lykke Ejdrup

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Also listed in [THRID-PARTY LICENSE.txt](./THRID-PARTY%20LICENSE.txt) §18.

### What we changed

- Retained the MIT license header on `RE_function.py`
- Pipeline wrapper: dual-channel direction handling, FOV normalization for expected counts, scatter-plot export, DataFrame packaging

### Effect on results

- Core `relative_enrichment` / related helpers in `RE_function.py` are upstream logic
- Expected-count normalization and plotting in the wrapper are pipeline-specific and define how RE values are reported in dSTORMQuant outputs

---

## FINDER (C++ port)

**Algorithm:** FINDER — Verzelli *et al.*, *Scientific Reports*, 2022  
**Source (original Python):** [https://github.com/NoldAndreas/FINDER](https://github.com/NoldAndreas/FINDER) (Andreas Nold and colleagues)  
**Paths in this repo:**

| Component                | Path                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------- |
| C++ port + CLI           | `finder_cpp/`                                                                      |
| Python bindings          | `finder_cpp` / `finder_bindings.cpp`                                               |
| Pipeline entry           | `cluster_with_finder` in `src/dSTORMQuant/analysis/clustering/cluster_analysis.py` |
| Spatial index (vendored) | `finder_cpp/third_party/nanoflann/`                                                |

**License:** MIT License (original FINDER and this C++ port)

```
MIT License

Copyright (c) 2021 Andreas Nold

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

C++ port copyright (c) 2026 Suraj Karki, Biochemistry and Molecular Medicine — Medical School OWL, Bielefeld University. Full text: [finder_cpp/LICENSE](./finder_cpp/LICENSE) and [THRID-PARTY LICENSE.txt](./THRID-PARTY%20LICENSE.txt) §17.

### What we changed

- C++ port for performance (optional pipeline backend)
- pybind11 bindings and YAML-configured parameters in dSTORMQuant
- Uses nanoflann for spatial indexing (see `finder_cpp/third_party/`)

### Effect on results

- Designed as a fast implementation of the published FINDER method.
- Differences in programming languages may lead to numerical or labeling differences from the original Python code. Equivalence must be verified.

### Validation

- CI builds the `finder_cpp` wheel
- Systematic equivalence tests against the original Python FINDER implementation and simulated ground truth were performed.

---

## DBSCAN / HDBSCAN (scikit-learn)

**Algorithms:** DBSCAN (Ester *et al.*, 1996); HDBSCAN (Campello *et al.*, 2015)  
**Source:** [scikit-learn](https://scikit-learn.org/) — `sklearn.cluster.DBSCAN`, `sklearn.cluster.HDBSCAN`  
**Path in this repo:** `src/dSTORMQuant/analysis/clustering/cluster_analysis.py`

| Piece                 | Role                                            |
| --------------------- | ----------------------------------------------- |
| scikit-learn          | Clustering implementation                       |
| `cluster_analysis.py` | Per-channel orchestration, config, stats export |

**License:** BSD 3-Clause License (scikit-learn) — full text in [THRID-PARTY LICENSE.txt](./THRID-PARTY%20LICENSE.txt) §11.

### What we changed

- Per-channel orchestration, config-driven parameters, cluster statistics and export

### Effect on results

- **No reimplementation**

### Validation

- Relies on scikit-learn;

---

## Other third-party dependencies

Runtime and build dependencies (e.g. locan, napari, matplotlib, pandas, scipy, h5py, pybind11, nanoflann) retain their own licenses. Full notices: [THRID-PARTY LICENSE.txt](./THRID-PARTY%20LICENSE.txt).

---

## Original content

The following are original or primarily original work of the dSTORMQuant authors (subject to the project license), including but not limited to:

- Pipeline orchestration (`src/dSTORMQuant/core/pipeline.py`, `main.py`)
- Configuration models and YAML schema (`src/dSTORMQuant/core/config/`)
- Filtering, temporal grouping, cell detection, nearest-neighbor analysis, run summary, and most visualization helpers
- Dual-channel metadata handling and batch export
- Documentation structure (`README.md`, `CONTRIBUTION.md`, `docs/` where present)
- Integration glue around the adapted modules above

Third-party attributions remain in effect for all reused or adapted components regardless of the project license.

---

## Our license

Unless otherwise noted, original content in this repository is licensed under the **GNU General Public License v3.0** — see `[LICENSE](LICENSE)`.

Third-party attributions and license obligations remain in effect regardless of that choice.

---

