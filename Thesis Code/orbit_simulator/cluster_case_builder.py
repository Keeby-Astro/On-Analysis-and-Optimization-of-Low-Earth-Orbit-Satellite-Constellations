"""Cluster case builder for Chapter 7 optimization.

Converts cluster policy vectors and representative packs into the
per-case arrays (oe_cases, sat_ids, start_timestamps,
ballistic_coefficients, case_schedules) ready for ``run_batch_cases()``.

Also expands cases across uncertainty scenarios using common random
numbers for fair cross-policy comparisons.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimization_config import ClusterPolicy, UncertaintyScenario, FidelityConfig
from cluster_representatives import RepresentativePack


# ======================================================================
# Policy application
# ======================================================================

def apply_cluster_policy_to_oe(
    oe: np.ndarray,
    policy: ClusterPolicy,
) -> np.ndarray:
    """Apply a cluster policy vector to a single set of orbital elements.

    Parameters
    ----------
    oe : ndarray, shape (6,)
        [a(km), e, i(rad), w(rad), OM(rad), Ma(rad)]
    policy : ClusterPolicy
        Shared policy vector for this cluster.

    Returns
    -------
    ndarray, shape (6,)
        Modified orbital elements.
    """
    oe_mod = oe.copy()
    oe_mod[0] += policy.delta_a             # semi-major axis offset (km)
    oe_mod[4] += policy.delta_Omega         # RAAN offset (rad)
    oe_mod[5] += policy.delta_lambda        # mean anomaly / along-track offset (rad)

    # Keep angles in [0, 2*pi)
    oe_mod[4] = oe_mod[4] % (2.0 * np.pi)
    oe_mod[5] = oe_mod[5] % (2.0 * np.pi)

    # Clamp semi-major axis above minimum safe altitude
    earth_Re = 6378.1366  # km
    oe_mod[0] = max(oe_mod[0], earth_Re + 120.0)

    return oe_mod


# ======================================================================
# Core case builder
# ======================================================================

def build_cluster_cases(
    pack: RepresentativePack,
    policy: ClusterPolicy,
    tle_latest_df: pd.DataFrame,
    fidelity: FidelityConfig,
    *,
    nominal_ballistic_coef: float = 0.0334,
    override_ballistic_coeffs: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str], List, np.ndarray]:
    """Build per-case arrays for one cluster at a given fidelity.

    Parameters
    ----------
    pack : RepresentativePack
        The representative set for this cluster.
    policy : ClusterPolicy
        Policy vector to apply.
    tle_latest_df : DataFrame
        Source TLE data (one row per satellite, ``sat_id`` column).
    fidelity : FidelityConfig
        Controls which representatives to use.
    nominal_ballistic_coef : float
        Default Cd*A/m (m^2/kg).
    override_ballistic_coeffs : dict | None
        ``{sat_id: beta}`` overrides per satellite.

    Returns
    -------
    oe_cases : ndarray (N, 6)
    sat_ids : list[str]
    start_timestamps : list[pd.Timestamp]
    ballistic_coefficients : ndarray (N,)
    """
    # Select representative subset based on fidelity
    if fidelity.representative_mode == "medoid_only":
        rep_sids = [pack.medoid_sat_id]
    elif fidelity.representative_mode == "full_members":
        rep_sids = pack.all_member_sat_ids
    else:  # medoid+boundary (default)
        rep_sids = pack.representative_sat_ids

    # Look up TLE row for each representative
    tle_indexed = tle_latest_df.set_index("sat_id", drop=False)
    nsims = len(rep_sids)

    oe_cases = np.zeros((nsims, 6), dtype=np.float64)
    sat_ids: list[str] = []
    start_timestamps: list[pd.Timestamp] = []
    ballistic_coefficients = np.full(nsims, nominal_ballistic_coef, dtype=np.float64)

    for k, sid in enumerate(rep_sids):
        if sid not in tle_indexed.index:
            raise KeyError(f"Satellite {sid} not found in TLE DataFrame")
        row = tle_indexed.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]  # handle duplicate sat_id

        oe_raw = np.array([
            float(row["sma"]),
            np.clip(float(row["ecc"]), 0.0, 0.95),
            np.deg2rad(float(row["inc"])),
            np.deg2rad(float(row["aop"])),
            np.deg2rad(float(row["raan"])),
            np.deg2rad(float(row["mean_anomaly"])),
        ], dtype=np.float64)

        # Enforce minimum safe SMA
        oe_raw[0] = max(oe_raw[0], 6378.1366 + 120.0)

        # Apply policy offsets
        oe_cases[k] = apply_cluster_policy_to_oe(oe_raw, policy)
        sat_ids.append(sid)
        start_timestamps.append(pd.Timestamp(row["timestamp"]))

        # Per-satellite beta override
        if override_ballistic_coeffs and sid in override_ballistic_coeffs:
            ballistic_coefficients[k] = override_ballistic_coeffs[sid]

    return oe_cases, sat_ids, start_timestamps, ballistic_coefficients


# ======================================================================
# Uncertainty expansion
# ======================================================================

def build_cluster_uncertainty_cases(
    oe_cases: np.ndarray,
    sat_ids: List[str],
    start_timestamps: List,
    ballistic_coefficients: np.ndarray,
    scenarios: List[UncertaintyScenario],
    seed: int = 42,
) -> Tuple[np.ndarray, List[str], List, np.ndarray, np.ndarray]:
    """Replicate base cases across uncertainty scenarios.

    Uses common random numbers: each scenario applies deterministic
    multiplicative scaling to ballistic coefficients (and optionally
    shifts epochs), but does NOT resample fresh noise.

    Parameters
    ----------
    oe_cases, sat_ids, start_timestamps, ballistic_coefficients
        Base cases from ``build_cluster_cases``.
    scenarios : list[UncertaintyScenario]
        Fixed scenario set (drawn once per study).
    seed : int
        For any needed randomness (currently unused but reserved).

    Returns
    -------
    expanded arrays (N*S, …) plus scenario_indices (length N*S) mapping
    each expanded row to its scenario index.
    """
    n_base = oe_cases.shape[0]
    n_scen = len(scenarios)
    n_total = n_base * n_scen

    oe_exp = np.zeros((n_total, 6), dtype=np.float64)
    bc_exp = np.zeros(n_total, dtype=np.float64)
    sid_exp: list[str] = []
    ts_exp: list = []
    scen_idx = np.zeros(n_total, dtype=np.int32)

    for s, scen in enumerate(scenarios):
        offset = s * n_base
        oe_exp[offset:offset + n_base] = oe_cases.copy()
        bc_exp[offset:offset + n_base] = ballistic_coefficients * scen.beta_scale

        for k in range(n_base):
            sid_exp.append(sat_ids[k])
            ts_base = pd.Timestamp(start_timestamps[k])
            if scen.epoch_shift_days != 0.0:
                ts_base += pd.Timedelta(days=scen.epoch_shift_days)
            ts_exp.append(ts_base)
            scen_idx[offset + k] = s

    return oe_exp, sid_exp, ts_exp, bc_exp, scen_idx


def group_results_by_scenario(
    results: list[dict],
    scenario_indices: np.ndarray,
    n_scenarios: int,
) -> List[List[dict]]:
    """Split flat result list back into per-scenario groups."""
    groups: list[list[dict]] = [[] for _ in range(n_scenarios)]
    for i, result in enumerate(results):
        groups[int(scenario_indices[i])].append(result)
    return groups
