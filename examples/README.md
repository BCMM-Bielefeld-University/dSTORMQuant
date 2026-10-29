# Example data

Bundled localization CSVs and one metadata workbook for pipeline demos and header checks.

| File | Description |
| ---- | ----------- |
| `AHA_24022_U2OS-WT_AST0126_AST0063-2_posXY1_channels_t0_posZ0.csv` | Dual-channel U2OS WT data (NimOS). Acquired by Anne Sophie Hammann, Biochemistry and Molecular Medicine, Medical School OWL, Bielefeld University. |
| `COS7_Tubulin_AF647_nstorm.csv` | Single-channel COS7 tubulin ([dataset 233](https://nano-org.bham.ac.uk/upload/dataset/233/)). ThunderSTORM; Dr. Sandeep Shirgrill, University of Birmingham. |
| `Slp76_259722AM_Slp76_PLL_5min_Cell-2_detections_dc.csv` | Single-channel Jurkat E6.1 Slp76 ([dataset 276](https://nano-org.bham.ac.uk/upload/dataset/276/)). ThunderSTORM; Dr. Sandeep Shirgrill, University of Birmingham. |
| `SMLM Data_Input.xlsx` | Metadata for all three experiments (`file_name`, channel indices, frame ranges). |

See the main [README](../README.md#3-demo) for copy-and-run instructions.

**Header note:** Geometry/index headers (`x (nm)`, `y (nm)`, `channelIndex`, `frameIndex`) are required. Quality headers should use NimOS-style names (`sigmaX (nm)`, `sigmaY (nm)`, `intensity (photons)`, `localization precision (nm)`, `p-value`). If a quality column is missing or still under another tool’s name, that filter step is **skipped** (warning + `filtering_report.json`); the run continues. Rename ThunderSTORM demos (COS7/Slp76) to apply those filters.

**3D / z:** The current release rejects files whose `z (nm)` column has numeric values. `AHA_24022…` includes z with values—clear or remove that column for a 2D demo run.
