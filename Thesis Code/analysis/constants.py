# Shared physical constants used across the analytics pipeline
from __future__ import annotations
import numpy as np

# Earth and orbital constants (km-based units unless noted otherwise)
MU_EARTH = 398600.4418
RADIUS_EARTH = 6378.14
J2_EARTH = 1.082635854e-3
MEAN_SIDEREAL_DAY = 86164.09054
SIDEREAL_YEAR = 365.25636
GRAVITATIONAL_CONSTANT = 6.67430e-11

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
SECONDS_PER_DAY = 86400.0

# Backward-compatible aliases used by legacy call paths
MU = MU_EARTH
R_EARTH_KM = RADIUS_EARTH
J2 = J2_EARTH

# Constants for ingestion/enrichment/synchronization
LOW_ECCENTRICITY_THRESHOLD = 1e-3
DEFAULT_SYNC_TOLERANCE = "12h"

SYNC_MODE_TARGET_NEAREST = "target_nearest"
SYNC_MODE_NEAREST_INTERSECTION = "nearest_intersection"
SYNC_MODE_EXACT_EPOCH_INTERSECTION = "exact_epoch_intersection"
SYNC_MODE_SCALAR_INTERPOLATION = "scalar_interpolation"
SYNC_MODE_SGP4_COMMON_EPOCH = "sgp4_common_epoch"
SYNC_MODES = (SYNC_MODE_TARGET_NEAREST, SYNC_MODE_NEAREST_INTERSECTION, SYNC_MODE_EXACT_EPOCH_INTERSECTION,
              SYNC_MODE_SCALAR_INTERPOLATION, SYNC_MODE_SGP4_COMMON_EPOCH)

# Pipeline-level enforced phase semantics for historical TLE analysis.
PHASE_VARIABLE_TRUE_ANOMALY = "true_anomaly"
PHASE_SEMANTICS_TRUE_ANOMALY_PROXY = "TLE-derived Kepler proxy from mean anomaly"

# Column-level semantic notes for compatibility and provenance-aware workflows
COMPATIBILITY_ALIAS_MAP = {"ballistic_coefficient": "mean_motion_dot", "drag_term": "bstar"}
COLUMN_SEMANTICS = {"mean_motion_dot": "TLE/GP first derivative of mean motion (catalog product field).",
                    "ballistic_coefficient": ("Compatibility alias for mean_motion_dot only; not a physical ballistic coefficient."),
                    "bstar": "TLE/GP B* drag term (catalog product field).",
                    "drag_term": "Compatibility alias for bstar.",
                    "sma_kepler_proxy_km": ("Keplerized proxy from TLE mean motion assuming two-body dynamics."),
                    "true_anomaly": ("TLE-derived Kepler proxy from mean anomaly; diagnostic phase variable, not conjunction-grade."),
                    "true_anomaly_kepler_proxy_deg": ("Keplerized proxy from mean anomaly via Kepler equation; diagnostic, not conjunction-grade."),
                    "specific_angular_momentum_kepler_proxy_km2_s": ("Keplerized proxy from (a,e) under two-body assumptions; diagnostic.")}