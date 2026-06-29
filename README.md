# dSTORMQuant: A Python Package for postprocessing and quantitative analysis of SMLM datasets

## Overview

dSTORMQuant is a comprehensive, modular Python tool for processing, filtering, drift correcting, and analyzing super-resolution microscopy (SMLM) data. It features a C++ accelerated FINDER clustering algorithm, flexible YAML-based configuration, and produces both quantitative results and good visualizations.

### Key Features

- **Dual-channel Support**: Analyze single or dual-channel SMLM data
- **Advanced Clustering (optional)**: DBSCAN, HDBSCAN, and C++ accelerated FINDER algorithm
- **Drift Correction**: AIM (Adaptive Intersection Maximization) algorithm
- **Colocalization Analysis (optional)**: Coordinate-based colocalization (CBC) and relative enrichment (RE)
- **Temporal Grouping**: Spatial–Temporal Linking
- **Cell Detection**: Automated cell boundary detection
- **Nearest Neighbor Analysis**: Intra- and inter-channel spatial relationship analysis
- **Rich Visualizations**: Napari integration, matplotlib plots, and automated reporting

### Documentation

- A complete documentation of the project can be found inside docs directory.

---

## 1. Prerequisites & Installation

### Requirements

- Python 3.10+
- C++ compiler (for FINDER algorithm compilation)
- `pip` (Python package manager)
- **Demo data:** [https://github.com/BCMM-Bielefeld-University/dSTORMQuant/releases/download/v0.0.1/examples.zip](https://github.com/BCMM-Bielefeld-University/dSTORMQuant/releases/download/v0.0.1/examples.zip)

### Setup Steps

1. **Clone the repository and go to the project root**
  ```bash
   git clone https://github.com/BCMM-Bielefeld-University/dSTORMQuant.git
   cd dSTORMQuant
  ```
   If you already have a local copy, navigate there instead:
  ```bash
   cd path/to/dSTORMQuant
  ```
2. **Create and activate a virtual environment (recommended)**
  ```bash
   python -m venv .venv
  ```
   **Windows (PowerShell):**
  ```powershell
   .\.venv\Scripts\Activate.ps1
  ```
   **Linux / macOS:**
  ```bash
   source .venv/bin/activate
  ```
3. **Install the package and dependencies**
  ```bash
   pip install -e .
  ```
   Or install from `requirements.txt` first, then the package in editable mode:
  ```bash
   pip install -r requirements.txt
   pip install -e .
  ```
4. **Build and install the FINDER C++ extension**
  ```bash
   cd finder_cpp
   pip install -r requirements-build.txt
   pip install pybind11
   pip install -e . --no-build-isolation
   cd ..
  ```
5. **Install developer tools (optional — Ruff, pre-commit, pytest)**
  ```bash
   pip install -e ".[dev]"
  ```

### Code Quality (Ruff)

Ruff configuration is defined in `ruff.toml`.

- Run lint checks:
  ```bash
  ruff check .
  ```
- Run only unused-import/unused-variable checks:
  ```bash
  ruff check . --select F401,F841
  ```
- Apply safe auto-fixes:
  ```bash
  ruff check . --fix
  ```

### Git Hooks (pre-commit)

Pre-commit configuration is defined in `.pre-commit-config.yaml`.

- Install the hooks (requires `pip install -e ".[dev]"` first):
  ```bash
  pre-commit install
  ```
- Run all hooks on the full repository:
  ```bash
  pre-commit run --all-files
  ```
- Update hook versions later if needed:
  ```bash
  pre-commit autoupdate
  ```

Next: [Demo](#3-demo) or the [user guide](docs/Supplementary matrial_UserGuide.pdf).

---

## 2. Project Structure

```
dSTORMQuant/                     # project root
├── .gitignore
├── CONTRIBUTION.md
├── LICENSE
├── .pre-commit-config.yaml       # Pre-commit hooks
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # GitHub Actions CI/CD workflow
├── config/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── README.md                # Folder layout; demo via examples.zip (§3)
│   ├── input/                   # Localization CSVs (contents gitignored)
│   ├── metadata/                # Excel metadata (contents gitignored)
│   ├── output/                  # Results (created on run; gitignored)
│   └── temp/                    # Scratch (created on run; gitignored)
├── docs/
│   └── Supplementary matrial_UserGuide.pdf  # User guide and documentation
├── finder_cpp/                  # C++ FINDER clustering implementation
│   ├── src/                     # C++ source and bindings
│   │   ├── main.cpp, finder.cpp, dbscan.cpp, dbscan_loop.cpp
│   │   ├── similarity_score.cpp, auxiliary_functions.cpp
│   │   └── finder_bindings.cpp  # Python bindings
│   ├── include/                 # Header files
│   ├── examples/                # Example scripts and sample data
│   ├── docs/                    # FINDER-specific docs
│   ├── third_party/             # e.g. nanoflann
│   ├── CMakeLists.txt
│   ├── build.sh / build.bat
│   └── pyproject.toml / setup.py
├── src/dSTORMQuant/
│   ├── main.py                  # Main pipeline entry point
│   ├── core/                    # Pipeline and config
│   │   ├── pipeline.py
│   │   └── config/              # Configuration loader and models
│   ├── scripts/                 # CLI scripts
│   │   ├── merge_csv.py
│   │   ├── output_discovery.py
│   │   ├── save_napari_visualizations.py
│   │   └── visualize_steps.py
│   ├── analysis/
│   │   ├── clustering/
│   │   ├── colocalization/
│   │   └── nearest_neighbor/
│   ├── processing/
│   ├── visualization/
│   └── utils/
├── setup.py
├── ruff.toml                     # Ruff configuration
├── pyproject.toml               # Project configuration and entry points
├── requirements.txt
└── README.md                    # This file
```

---

## 3. Demo

### Example data

Download `examples.zip` and place the contents in `data/input/` and `data/metadata/` as described below.

**Download URL:**

[https://github.com/BCMM-Bielefeld-University/dSTORMQuant/releases/download/v0.0.1/examples.zip](https://github.com/BCMM-Bielefeld-University/dSTORMQuant/releases/download/v0.0.1/examples.zip)


| File                                                               | Put in           | Description                                                                                                                                                       |
| ------------------------------------------------------------------ | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AHA_24022_U2OS-WT_AST0126_AST0063-2_posXY1_channels_t0_posZ0.csv` | `data/input/`    | Dual-channel U2OS WT data (NimOS). Acquired by Anne Sophie Hammann, Biochemistry and Molecular Medicine, Medical School OWL, Bielefeld University.                |
| `COS7_Tubulin_AF647_nstorm.csv`                                    | `data/input/`    | Single-channel COS7 tubulin ([dataset 233](https://nano-org.bham.ac.uk/upload/dataset/233/)). ThunderSTORM; Dr. Sandeep Shirgrill, University of Birmingham.      |
| `Slp76_259722AM_Slp76_PLL_5min_Cell-2_detections_dc.csv`           | `data/input/`    | Single-channel Jurkat E6.1 Slp76 ([dataset 276](https://nano-org.bham.ac.uk/upload/dataset/276/)). ThunderSTORM; Dr. Sandeep Shirgrill, University of Birmingham. |
| `dSTORM Data_Input.xlsx`                                           | `data/metadata/` | Metadata for all three experiments (`file_name`, channel indices, frame ranges).                                                                                  |


The metadata sheet lists frame ranges and channel indices for each file. `AHA_24022` is dual-channel; **COS7** and **Slp76** are single-channel rows (second channel columns left empty).

`examples.zip` layout:

```
examples/
  AHA_24022_U2OS-WT_AST0126_AST0063-2_posXY1_channels_t0_posZ0.csv
  COS7_Tubulin_AF647_nstorm.csv
  Slp76_259722AM_Slp76_PLL_5min_Cell-2_detections_dc.csv
  dSTORM Data_Input.xlsx
```

After extracting, move files from the `examples/` folder into the pipeline folders already in the repo (`data/input/` and `data/metadata/`).

### Instructions to run on example data

Complete [installation](#1-prerequisites--installation) first.

#### Step 1 — Download `examples.zip`

Download `**examples.zip**` from the link above (replace the URL when your host is ready). Save it in the project root (`dSTORMQuant/`) or your Downloads folder.

#### Step 2 — Extract and move files into `data/`

The repository already includes `data/input/` and `data/metadata/`. The archive contains an `examples/` folder with the three CSVs and the metadata workbook.

1. Extract `examples.zip` in the project root (`dSTORMQuant/`). You should get `dSTORMQuant/examples/` with four files inside.
2. Move the files into `data/`:
  - All three `.csv` files → `data/input/`
  - `dSTORM Data_Input.xlsx` → `data/metadata/`
3. You can delete the empty `examples/` folder afterward (it is only from the ZIP, not part of the pipeline).

**Windows (File Explorer)**

1. Right-click `examples.zip` in `dSTORMQuant` → **Extract All…** → extract to `dSTORMQuant`.
2. Open the new `examples` folder, select the three `.csv` files, **Cut**, open `data\input`, **Paste**.
3. Cut `dSTORM Data_Input.xlsx` from `examples`, paste into `data\metadata`.

**macOS / Linux (terminal, from project root)**

```bash
unzip -o examples.zip
mv examples/*.csv data/input/
mv examples/"dSTORM Data_Input.xlsx" data/metadata/
rmdir examples
```

**Verify:**

- `data/input/` contains the three CSV files  
- `data/metadata/` contains `dSTORM Data_Input.xlsx`

#### Step 3 — Check configuration

1. Open `config/config.yaml` in a text editor (Notepad, VS Code, etc.).
2. Find the block `data:` → `input:`.
3. Confirm this line is present (only change it if your metadata file has a different name):

```yaml
xlsx_filename: "dSTORM Data_Input.xlsx"
```

1. Save the file if you edited it.

#### Step 4 — Run the pipeline

Open a terminal **in the project folder** `dSTORMQuant`:


| System      | How                                                                                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Windows** | In File Explorer, open `dSTORMQuant`, click the address bar, type `powershell`, press Enter. Or: Shift + right-click empty space → **Open PowerShell window here**. |
| **macOS**   | In Finder, right-click `dSTORMQuant` → **New Terminal at Folder** (or open Terminal and `cd` to the folder).                                                        |
| **Linux**   | Open a terminal and `cd` to `dSTORMQuant`, or use your file manager’s **Open in Terminal** if available.                                                            |


If you use a virtual environment ([§1](#1-prerequisites--installation)), activate it first:

- **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

Type this command and press Enter:

```bash
dSTORMQuant
```

Progress appears in the window; a detailed log is saved under `logs/`.

### Expected output

After a successful run you should see:

**1. One ZIP per metadata row** in `data/output/`:

- `data/output/AHA_24022_U2OS-WT_AST0126_AST0063-2_posXY1_channels_t0_posZ0.zip`
- `data/output/COS7_Tubulin_AF647_nstorm.zip`
- `data/output/Slp76_259722AM_Slp76_PLL_5min_Cell-2_detections_dc.zip`

Extract a ZIP to inspect its contents:

- `test_data/` — CSV files from each processing stage (filtered, clustered, colocalization, etc.)
- `test_images/` — PNG plots (histograms, Napari views, clustering, etc.)

The pipeline archives results into ZIP files and clears scratch data under `data/temp/` during processing.

**2. Summary CSVs** directly in `data/output/` (appended across runs):


| File                           | When created                                  |
| ------------------------------ | --------------------------------------------- |
| `processing_summary_stats.csv` | Every run                                     |
| `nn_summary_stats.csv`         | When nearest-neighbor analysis runs           |
| `clustering_summary_stats.csv` | When clustering is enabled                    |
| `coloc_summary_stats.csv`      | When colocalization is enabled (dual-channel) |


**3. Log file** under `logs/` (e.g. `dstorm_pipeline_YYYYMMDD_HHMMSS.log`) with per-step timings.

**Note:** The pipeline may write temporary channel-split CSVs (`*_ch1.csv`, `*_ch2.csv`) and HDF5 sidecars under `data/input/` while processing dual-channel data. These are working files, not the final deliverable.

---

## 4. Configuring the Pipeline (`config.yaml`)

All pipeline options are set in `config/config.yaml`. Here is a summary of the main sections:

### **data**

- `input.file_format`: Format of your input data (e.g., `csv`). Supports CSV format.
- `input.required_columns`: Headers validated on **load** (default: `x (nm)`, `y (nm)`, `channelIndex`, `frameIndex`). Quality columns such as sigma, photons, `**p-value`** (exact header name), and localization precision are **optional** in the CSV. Match `required_columns` to what your files actually contain.
- `input.xlsx_filename`: Name of the metadata Excel file (placed in `data/metadata/`). The pipeline uses this to discover which input files to process; input CSV files should be in `data/input/`.
- `input.required_metadata_columns`: Dict mapping role to metadata Excel column name. **Required keys:** `file_name`, `first_channel_index`, `first_ch_frame_last`, `second_channel_index`, `second_ch_frame_last`. Headers are matched after normalization (strip, lower case, single space). Single-channel rows: leave second channel index and/or last frame blank or `None`.
- `input.optional_metadata_columns`: Optional dict (default `{}`). Allowed keys: `experiment_number`, `initials`, `tag`, `first_channel_label`, `second_channel_label` — each maps to an Excel header; those columns may be absent from the sheet.
- `input.skip_frames`: Number of frames to skip at the start of each channel's frame range (default 200; set to 0 to use all frames).
- **Examples**: Download `examples.zip` into `data/input/` and `data/metadata/` — see [§3 Demo](#3-demo).

### **channels**

- `Ch1`/`Ch2`: Define color (hex) and label for each channel. Channel indices come from the metadata.

### **visualization**

- `use_napari`: If `true` (default), the pipeline saves Napari-based point previews and cluster overview PNGs. If `false`, those outputs are **skipped** (useful for headless/cloud; histograms from `plot_metrics` still run).

### **filtering**

- Each of `sigma`, `intensity`, `localization_precision`, and `p_value` supports `use: true|false` to enable or disable that step.
- `sigma.min_value` / `sigma.max_value`, `intensity.min_value`, `localization_precision.threshold_value`, `p_value.threshold_value`: thresholds read only from YAML. The input localization CSV supplies measurements (e.g. sigma, photons, precision); it does not carry separate filter-parameter columns.

### **cell_detection**

- `use`: Set to `true` to enable cell detection.
- `approach`: `"grid"` or `"vornoi"`.
- `grid.grid_size`, `grid.high_density_threshold`: Grid-based cell detection parameters.

### **drift_correction**

- `pixel_size`: Pixel size in nanometers (e.g., `117`). Used for converting between pixels and nanometers during drift correction.
- `segmentation`: Number of segments for drift correction (e.g., `100`). The data is divided into this many segments for drift estimation.
- `intersect_d_nm`: Intersection distance in nm for drift overlap calculation.
- `roi_r_nm`: ROI radius in nm for drift estimation.
- **drift_validation**: Detection and replacement of unrealistic drift values.
  - `use`: Set to `true` to enable drift validation.
  - `max_segment_drift_nm`: Maximum allowed drift per segment in nm; values above this are replaced.
  - `n_previous_segments`: Number of previous valid segments to average when replacing outliers.
- **sanity_checks**: Post-drift filters applied to localizations.
  - `remove_invalid_values`: Remove localizations with inf/nan values.
  - `filter_boundary`: Remove localizations with x≤0 or y≤0.
  - `filter_parameters`: Remove localizations with invalid sigma or localization precision (sx≤0, sy≤0, lp≤0).
  - **Note**: The CSV→HDF conversion (`src/dSTORMQuant/processing/drift_correction/AIM.py::csv2hdf`) and AIM pipeline saves now respect the `drift_correction.sanity_checks` settings when the pipeline entry points are used. If code calls `csv2hdf()` or `save_locs()` directly without passing a `sanity_config`, those functions will still use their default behavior.

### **temporal_grouping**

- `use`: Set to `true` to run temporal grouping; `false` to skip (filtered data is passed through).
- `max_frame_gap`, `max_distance_nm`, `min_duration`, `max_duration`: temporal grouping parameters when `use` is true (defaults: `2`, `50`, `1`, `50`).

### **clustering**

- `use`, `use_cluster_knn`: Enable clustering and cluster-center kNN analysis.
- `ch1`/`ch2`: Per-channel settings.
  - `method`: `dbscan`, `hdbscan`, or `finder`.
  - **DBSCAN**: `dbscan.eps`, `dbscan.min_samples` (defaults: `100`, `10`).
  - **HDBSCAN**: `hdbscan.min_cluster_size`, `hdbscan.min_samples` (defaults: `10`, `10`).
  - **FINDER**: `finder.`* parameters (defaults: `threshold=10`, `points_per_dimension=15`, `algorithm=dbscan`, `min_threshold=5`, `max_threshold=21`, `decay=0.5`).

### **colocalization**

- `use`: Enable colocalization analysis.
- `cbc.radius`, `cbc.n_steps`: Coordinate-Based Colocalization parameters (defaults: `100`, `10`). Relative enrichment (RE) runs when colocalization is enabled and has no separate YAML subsection.

### **nearest_neighbor_analysis**

- `use`: Set to `true` to run k-Nearest-Neighbor analysis (intra- and inter-channel distances).
- `radius`, `k`, `algorithm`, `metric`: Radius (nm) for mean-within-radius; kNN parameters (defaults: `100`, `1`, `auto`, `euclidean`).

> **Tip:** Each section in `config.yaml` is documented with comments. Adjust values as needed for your experiment.

> **Important:**
> The pipeline discovers input files from the metadata Excel (`data.input.xlsx_filename` in `data/metadata/`) and writes one ZIP archive per experiment into `data/output/`.

---

## 5. Running the Pipeline

1. Place your input data files (CSV) in `data/input/`. Place the metadata Excel in `data/metadata/` and set `data.input.xlsx_filename` in `config/config.yaml` (metadata is required by the current pipeline entrypoint).
2. Review and adjust all relevant parameters in `config/config.yaml`.
3. From the project root, run the pipeline (after `pip install -e .`):
  ```bash
   dSTORMQuant
  ```
   Or: `python -m dSTORMQuant.main`.
4. The pipeline prints progress and writes a ZIP archive per experiment to `data/output/` (e.g. `data/output/<experiment_name>.zip`).
  Post-analysis scripts process existing output folders and do not auto-extract ZIP archives.

---

## 6. Post-Analysis Scripts

After the pipeline has run, you can use these CLI commands from the project root.


| Command                      | Purpose                                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `merge-csv`                  | Merge CSV outputs from multiple experiments into `data/output/merged_channel_data/` by cell line and channel |
| `save-napari-visualizations` | Batch-save Napari views as images into each experiment’s `test_images/` folder                               |


Install the package first (`pip install -e .`). If a command is not found, use e.g. `python -m dSTORMQuant.scripts.merge_csv`.

Important: ensure the corresponding output folder (for example `data/output/<experiment_name>/test_data`) exists before running post-analysis scripts.

**Note:** Earlier versions exposed `visualization-steps` and `roi-clusters` commands, which are no longer available in the current release.

---

## 7. CI/CD (GitHub Actions)

This repository includes a GitHub Actions workflow at `.github/workflows/ci-cd.yml`.

- **When it runs**:
  - On pull requests
  - On pushes to `main` or `master`
  - On tags like `v0.1.0` (release automation)
- **What it does**:
  - Runs `ruff check .`
  - Runs a lightweight **smoke check** (compiles Python sources and checks core imports)
  - Builds distribution artifacts for the main package (`dist/`) and `finder_cpp`
  - On `v`* tags, creates a GitHub Release and attaches the built artifacts

---

## 8. Pipeline Workflow

The pipeline executes the following steps sequentially:

1. **Data Loading**: Reads localization data (HDF5, CSV, or other supported formats)
2. **Initial Visualization**: Creates baseline visualizations
3. **Drift Correction**: Applies AIM drift correction algorithm
4. **Filtering**: Removes low-quality localizations based on configurable thresholds
5. **Temporal Grouping**: Groups localizations over time
6. **Cell Detection** (optional): Identifies cell boundaries
7. **Clustering** (if enabled): Performs spatial clustering (DBSCAN/HDBSCAN/FINDER)
8. **Nearest Neighbor Analysis** (optional): Calculates spatial relationships
9. **Colocalization Analysis** (if enabled): CBC and RE analysis for multi-channel data
10. **Visualization & Reporting**: Generates plots and summary statistics

---

## 9. Output Structure

The pipeline creates a subdirectory in `data/output/` for each experiment (named after the input file base name). Within each experiment folder:

- **test_data/**: Processed data files (CSV, HDF5) after each major step (e.g., filtered, clustered, colocalized).
- **test_images/**: Generated visualizations (PNG images, Napari screenshots, histograms, plots).

Examples of processing steps reflected in outputs:

- Initial visualizations and metrics
- Drift-corrected localizations and trajectory plots
- Filtered data (by sigma, photons, precision, p-value)
- Temporal grouped (merged) localizations
- Cell detection results (if enabled)
- Clustering assignments, statistics, and visualizations
- Nearest neighbor distance histograms and CSV files (if enabled)
- Colocalization analysis results (CBC and RE for multi-channel data)
- Summary CSV files with all metrics

Each processing step generates CSV files for data, PNG images for visualizations, and aggregated statistics.

---

