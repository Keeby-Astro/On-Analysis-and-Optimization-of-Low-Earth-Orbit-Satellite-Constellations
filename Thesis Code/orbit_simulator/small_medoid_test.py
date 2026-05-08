#!/usr/bin/env python3
"""Small medoid optimization run — 5 clusters × 5 trials.

Validates in-process execution speedup and calibrates per-trial timing
for full-run configuration. Expected ~25 propagations + JIT warmup.
"""

import sys
import traceback

from optimization_config import (
    FidelityConfig,
    ObjectiveWeights,
    OptimizationStudyConfig,
    PolicyBounds,
)
from run_constellation_optimization import run_study

CONFIG = OptimizationStudyConfig(
    cluster_ids=None,       # auto-select by size
    max_clusters=5,

    local_search_fidelity=FidelityConfig.fidelity_1(),
    verification_fidelity=FidelityConfig.fidelity_1(),   # skip verification

    n_local_trials=5,
    n_candidates_per_cluster=3,
    policy_bounds=PolicyBounds(),
    objective_weights=ObjectiveWeights(),

    n_uncertainty_scenarios=1,   # nominal only
    robust_mode="mean",

    stitching_method="beam",
    stitching_beam_width=10,
    stitching_gamma=1.0,
    stitching_R_max=0.01,
    adjacency_sma_threshold_km=50.0,
    adjacency_raan_threshold_deg=30.0,

    n_boundary_representatives=0,   # medoid_only

    random_seed=42,
    max_parallel_workers=4,
    output_dir="small_medoid_outputs",

    tle_data_folders=None,
    tle_satellite_limit=0,
    tle_earliest_start_epoch="2019-10-01",
    simulation_date_cutoff="2035-01-01",
)


if __name__ == "__main__":
    try:
        summary = run_study(CONFIG)
        print("\n" + "=" * 60)
        print(" SMALL MEDOID TEST PASSED")
        print("=" * 60)

        import json
        print(json.dumps(summary, indent=2, default=str))

    except Exception:
        traceback.print_exc()
        print("\n SMALL MEDOID TEST FAILED", file=sys.stderr)
        sys.exit(1)
