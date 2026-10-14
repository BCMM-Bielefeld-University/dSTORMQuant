#!/usr/bin/env python3
"""
Run finder_cpp on the example CSV and save cluster labels.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# Same defaults as Andreas Nold and Pietro Verzelli FINDER pip package
THRESHOLD = 10
POINTS_PER_DIMENSION = 15
ALGORITHM = "DbscanLoop"
MIN_THRESHOLD = 5
MAX_THRESHOLD = 21
DECAY = 0.5

EXAMPLE_DIR = Path(__file__).resolve().parent / "example"
INPUT_CSV = EXAMPLE_DIR / "COS7_Tubulin_AF647_nstorm.csv"
OUTPUT_CSV = EXAMPLE_DIR / "COS7_Tubulin_AF647_nstorm_cpp_clusters.csv"


def main() -> int:
    print("=" * 60)
    print("C++ FINDER (finder_cpp)")
    print("=" * 60)

    try:
        import finder_cpp
    except ImportError as e:
        print(f"Failed to import finder_cpp: {e}")
        print("Build first: cd finder_cpp && pip install -e . --no-build-isolation")
        return 1

    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(
        f"Config: threshold={THRESHOLD}, ppd={POINTS_PER_DIMENSION}, "
        f"algo={ALGORITHM}, minmax_threshold=[{MIN_THRESHOLD}, {MAX_THRESHOLD}], "
        f"decay={DECAY}"
    )

    df = pd.read_csv(INPUT_CSV)
    coords = df[["x", "y"]].to_numpy(dtype=np.float32)
    print(f"Points: {len(coords)}")

    t0 = time.time()
    result = finder_cpp.run_finder_2d(
        coords,
        threshold=THRESHOLD,
        points_per_dimension=POINTS_PER_DIMENSION,
        algorithm=ALGORITHM,
        min_threshold=MIN_THRESHOLD,
        max_threshold=MAX_THRESHOLD,
        decay=DECAY,
    )
    elapsed = time.time() - t0

    # Bindings return dict {labels, sigma, threshold}; support older array return
    if isinstance(result, dict):
        labels = np.asarray(result["labels"])
        selected = {"sigma": result["sigma"], "threshold": result["threshold"]}
    else:
        labels = np.asarray(result)
        selected = None

    unique = set(labels.tolist())
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int(np.sum(labels == -1))

    print(f"\nDone in {elapsed:.2f} s")
    if selected is not None:
        print(f"Selected parameters: {selected}")
    print(f"Clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")

    out = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": labels,
        }
    )
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
