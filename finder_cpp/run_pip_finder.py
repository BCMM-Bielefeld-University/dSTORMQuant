#!/usr/bin/env python3
"""
Run pip FINDER (finder_smlm) on the example CSV and save cluster labels.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from finder import Finder

# Same defaults as finder_cpp CLI
THRESHOLD = 10
POINTS_PER_DIMENSION = 15
ALGO = "DbscanLoop"
MINMAX_THRESHOLD = [5, 21]
ONE_TWO_D = "twoD"
SIMILARITY_SCORE_COMPUTATION = "threshold"
LOG_THRESHOLDS = False
LOG_SIGMAS = True
ADAPTIVE_SIGMA_BOUNDARIES = False
DECAY = 0.5

EXAMPLE_DIR = Path(__file__).resolve().parent / "example"
INPUT_CSV = EXAMPLE_DIR / "COS7_Tubulin_AF647_nstorm.csv"
OUTPUT_CSV = EXAMPLE_DIR / "COS7_Tubulin_AF647_nstorm_pip_clusters.csv"


def main() -> int:
    print("=" * 60)
    print("pip FINDER (finder_smlm)")
    print("=" * 60)
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(
        f"Config: threshold={THRESHOLD}, ppd={POINTS_PER_DIMENSION}, "
        f"algo={ALGO}, minmax_threshold={MINMAX_THRESHOLD}, "
        f"similarity={SIMILARITY_SCORE_COMPUTATION}, decay={DECAY}"
    )

    df = pd.read_csv(INPUT_CSV)
    xc = df[["x", "y"]].to_numpy(dtype=np.float64)
    print(f"Points: {len(xc)}")

    finder = Finder(
        threshold=THRESHOLD,
        points_per_dimension=POINTS_PER_DIMENSION,
        algo=ALGO,
        minmax_threshold=MINMAX_THRESHOLD,
        one_two_d=ONE_TWO_D,
        similarity_score_computation=SIMILARITY_SCORE_COMPUTATION,
        log_thresholds=LOG_THRESHOLDS,
        log_sigmas=LOG_SIGMAS,
        adaptive_sigma_boundaries=ADAPTIVE_SIGMA_BOUNDARIES,
        decay=DECAY,
        n_dim=2,
    )

    t0 = time.time()
    labels = np.asarray(finder.fit(xc))
    elapsed = time.time() - t0

    unique = set(labels.tolist())
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int(np.sum(labels == -1))

    print(f"\nDone in {elapsed:.2f} s")
    print(f"Selected parameters: {finder.selected_parameters}")
    print(f"Clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")

    out = pd.DataFrame(
        {
            "x": xc[:, 0],
            "y": xc[:, 1],
            "cluster": labels,
        }
    )
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
