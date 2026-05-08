"""
Arc Building for Trajectory-Matching Stage A
=============================================

Builds per-satellite, per-phase "trajectory arcs" from TLE observations and
manoeuvre-phase interval labels.  Each arc contains the full time-series of
SMA (and auxiliary orbital elements) observed within a single phase interval,
enabling trajectory-level fitting instead of segment-endpoint fitting.

Data flow
---------
    maneuver_phase_intervals CSV  +  TLE DataFrame
        → build_arcs_from_tles_and_intervals()
            → List[ArcRecord]

    List[ArcRecord]
        → ArcDataset(torch.utils.data.Dataset)
            → batches via collate_arcs()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from reduced_dynamics import wrap_to_2pi, wrap_to_pi

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ArcRecord:
    """One contiguous observation arc for a single satellite in a single phase."""
    arc_id: int
    sat_id: str
    phase: str
    phase_sign: float
    # Initial orbital state (from first TLE in the arc)
    a0_km: float
    e0: float
    inc0_rad: float
    raan0_rad: float
    lam0_rad: float
    # Time offsets from arc start [seconds], length = n_obs
    dt_s: np.ndarray
    # Observed SMA at each TLE epoch [km], length = n_obs
    a_obs_km: np.ndarray
    # Endpoint observations for secondary targets
    raan_final_rad: float
    lam_final_rad: float
    # Number of valid observations
    n_obs: int
    # Full duration of the arc [seconds]
    duration_s: float


@dataclass
class ArcBuildConfig:
    """Configuration for arc extraction."""
    min_obs: int = 5
    min_duration_s: float = 6.0 * 3600.0          # 6 hours
    max_duration_s: float = 120.0 * 86400.0        # 120 days
    timestamp_tolerance_s: float = 3600.0           # 1 hour match tolerance


def phase_sign_from_name(phase_name: str) -> float:
    """Return thrust direction sign for a manoeuvre phase.

    +1  = prograde (orbit raise)
    -1  = retrograde (disposal lowering)
     0  = operational shell (drag make-up)
    """
    p = str(phase_name).strip().lower()
    retro_keywords = ["deorbit", "lower", "disposal", "retro", "drop", "descent"]
    if any(k in p for k in retro_keywords):
        return -1.0
    stationkeep_keywords = ["operational", "shell", "station", "maintain"]
    if any(k in p for k in stationkeep_keywords):
        return 0.0
    return 1.0


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Find the first matching column name (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def _compute_mean_longitude(row_raan: float, row_aop: float, row_ma: float) -> float:
    """Compute mean longitude λ = Ω + ω + M, wrapped to [0, 2π)."""
    return float(wrap_to_2pi(row_raan + row_aop + row_ma))


# ── Core arc building ────────────────────────────────────────────────────────

def build_arcs_from_tles_and_intervals(
    tle_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    cfg: ArcBuildConfig = ArcBuildConfig(),
) -> List[ArcRecord]:
    """Build trajectory arcs from TLE data and phase interval labels.

    Parameters
    ----------
    tle_df : pd.DataFrame
        TLE data with columns: sat_id, timestamp, sma (derived), ecc, inc,
        raan, aop, mean_anomaly.  ``timestamp`` must be datetime-parseable.
    intervals_df : pd.DataFrame
        Phase interval labels with columns: sat_id, phase_state, phase_start,
        phase_end.  See ``maneuver_phase_intervals_gen1_full.csv``.
    cfg : ArcBuildConfig
        Arc extraction configuration.

    Returns
    -------
    List[ArcRecord]
        One arc per (satellite, phase interval) that passes quality filters.
    """
    # ── Resolve column names ─────────────────────────────────────────────
    tle_sat_col = _find_col(tle_df, ["sat_id", "norad_cat_id", "object_id"])
    tle_ts_col = _find_col(tle_df, ["timestamp", "tle_epoch"])
    tle_sma_col = _find_col(tle_df, ["sma", "semi_major_axis"])
    tle_ecc_col = _find_col(tle_df, ["ecc", "eccentricity"])
    tle_inc_col = _find_col(tle_df, ["inc", "inclination"])
    tle_raan_col = _find_col(tle_df, ["raan", "right_ascension"])
    tle_aop_col = _find_col(tle_df, ["aop", "argument_of_perigee", "arg_perigee"])
    tle_ma_col = _find_col(tle_df, ["mean_anomaly"])

    int_sat_col = _find_col(intervals_df, ["sat_id", "norad_cat_id", "object_id"])
    int_phase_col = _find_col(intervals_df, ["phase_state", "phase", "label", "maneuver_label"])
    int_start_col = _find_col(intervals_df, ["phase_start", "start_timestamp", "start_time", "start", "t0"])
    int_end_col = _find_col(intervals_df, ["phase_end", "end_timestamp", "end_time", "end", "t1"])

    # ── Ensure timestamps are datetime ───────────────────────────────────
    tle_work = tle_df.copy()
    tle_work["_ts"] = pd.to_datetime(tle_work[tle_ts_col], utc=True, errors="coerce")
    tle_work = tle_work.dropna(subset=["_ts"])

    # ── Convert SMA from km if present, ensure numeric ───────────────────
    tle_work["_sma"] = pd.to_numeric(tle_work[tle_sma_col], errors="coerce")
    tle_work = tle_work.dropna(subset=["_sma"])

    # ── Ensure angle columns are in radians ──────────────────────────────
    # TLE data from load_all_tle_data has angles in radians already
    for col_name in [tle_inc_col, tle_raan_col, tle_aop_col, tle_ma_col, tle_ecc_col]:
        tle_work[col_name] = pd.to_numeric(tle_work[col_name], errors="coerce")

    # ── Group TLE by satellite ───────────────────────────────────────────
    tle_work["_sat_str"] = tle_work[tle_sat_col].astype(str).str.replace(r"\.txt$", "", regex=True)
    tle_by_sat = {sid: grp.sort_values("_ts").reset_index(drop=True) for sid, grp in tle_work.groupby("_sat_str")}

    # ── Parse intervals ──────────────────────────────────────────────────
    intervals_work = intervals_df.copy()
    intervals_work["_sat_str"] = intervals_work[int_sat_col].astype(str).str.replace(r"\.txt$", "", regex=True)
    intervals_work["_phase_start"] = pd.to_datetime(intervals_work[int_start_col], utc=True, errors="coerce")
    intervals_work["_phase_end"] = pd.to_datetime(intervals_work[int_end_col], utc=True, errors="coerce")
    intervals_work = intervals_work.dropna(subset=["_phase_start", "_phase_end"])

    arcs: List[ArcRecord] = []
    arc_id_counter = 0
    n_skipped_no_tle = 0
    n_skipped_too_few_obs = 0
    n_skipped_duration = 0

    for _, irow in intervals_work.iterrows():
        sat_str = str(irow["_sat_str"])
        if sat_str not in tle_by_sat:
            n_skipped_no_tle += 1
            continue

        sat_tles = tle_by_sat[sat_str]
        phase_start = irow["_phase_start"]
        phase_end = irow["_phase_end"]
        phase_name = str(irow[int_phase_col]).strip()

        # Tolerance window
        tol = pd.Timedelta(seconds=cfg.timestamp_tolerance_s)
        mask = (sat_tles["_ts"] >= phase_start - tol) & (sat_tles["_ts"] <= phase_end + tol)
        arc_tles = sat_tles.loc[mask].copy()

        if len(arc_tles) < cfg.min_obs:
            n_skipped_too_few_obs += 1
            continue

        # Sort by time
        arc_tles = arc_tles.sort_values("_ts").reset_index(drop=True)

        # Compute time offsets from first observation
        t0 = arc_tles["_ts"].iloc[0]
        dt_s_arr = (arc_tles["_ts"] - t0).dt.total_seconds().to_numpy(dtype=np.float64)
        duration_s = float(dt_s_arr[-1])

        if duration_s < cfg.min_duration_s or duration_s > cfg.max_duration_s:
            n_skipped_duration += 1
            continue

        # Extract initial orbital state
        first = arc_tles.iloc[0]
        last = arc_tles.iloc[-1]

        a0_km = float(first["_sma"])
        e0 = float(first[tle_ecc_col])
        inc0_rad = float(first[tle_inc_col])
        raan0_rad = float(first[tle_raan_col])
        lam0_rad = _compute_mean_longitude(
            float(first[tle_raan_col]), float(first[tle_aop_col]), float(first[tle_ma_col])
        )

        # SMA observations along the arc
        a_obs_km = arc_tles["_sma"].to_numpy(dtype=np.float64)

        # Endpoint RAAN and lambda
        raan_final_rad = float(last[tle_raan_col])
        lam_final_rad = _compute_mean_longitude(
            float(last[tle_raan_col]), float(last[tle_aop_col]), float(last[tle_ma_col])
        )

        arcs.append(ArcRecord(
            arc_id=arc_id_counter,
            sat_id=sat_str,
            phase=phase_name,
            phase_sign=phase_sign_from_name(phase_name),
            a0_km=a0_km,
            e0=e0,
            inc0_rad=inc0_rad,
            raan0_rad=raan0_rad,
            lam0_rad=lam0_rad,
            dt_s=dt_s_arr,
            a_obs_km=a_obs_km,
            raan_final_rad=raan_final_rad,
            lam_final_rad=lam_final_rad,
            n_obs=len(arc_tles),
            duration_s=duration_s,
        ))
        arc_id_counter += 1

    logger.info(
        "Arc building: %d arcs built, %d skipped (no TLE), %d skipped (too few obs), %d skipped (duration)",
        len(arcs), n_skipped_no_tle, n_skipped_too_few_obs, n_skipped_duration,
    )
    return arcs


# ── Dataset for trajectory matching ──────────────────────────────────────────

class ArcDataset(Dataset):
    """PyTorch Dataset for trajectory-matching arcs.

    Pads variable-length dt/a_obs arrays to ``max_obs`` and provides a binary
    mask tensor.  Maps sat_id → sat_idx, phase → phase_idx.
    """

    def __init__(self, arcs: List[ArcRecord], max_obs: int = 200):
        self.arcs = arcs
        self.max_obs = max_obs

        sat_ids = sorted(set(a.sat_id for a in arcs))
        phases = sorted(set(a.phase for a in arcs))
        self.sat_to_idx = {s: i for i, s in enumerate(sat_ids)}
        self.phase_to_idx = {p: i for i, p in enumerate(phases)}

        # Pre-build padded tensors for fast __getitem__
        n = len(arcs)
        self.a0_km = torch.zeros(n, dtype=torch.float32)
        self.e0 = torch.zeros(n, dtype=torch.float32)
        self.inc0_rad = torch.zeros(n, dtype=torch.float32)
        self.raan0_rad = torch.zeros(n, dtype=torch.float32)
        self.lam0_rad = torch.zeros(n, dtype=torch.float32)
        self.phase_idx = torch.zeros(n, dtype=torch.long)
        self.sat_idx = torch.zeros(n, dtype=torch.long)
        self.phase_sign = torch.zeros(n, dtype=torch.float32)
        self.n_obs = torch.zeros(n, dtype=torch.long)
        self.duration_s = torch.zeros(n, dtype=torch.float32)
        self.raan_final_rad = torch.zeros(n, dtype=torch.float32)
        self.lam_final_rad = torch.zeros(n, dtype=torch.float32)
        # Padded 2D arrays
        self.dt_s = torch.zeros(n, max_obs, dtype=torch.float32)
        self.a_obs_km = torch.zeros(n, max_obs, dtype=torch.float32)
        self.mask = torch.zeros(n, max_obs, dtype=torch.float32)

        for i, arc in enumerate(arcs):
            nobs = min(arc.n_obs, max_obs)
            self.a0_km[i] = arc.a0_km
            self.e0[i] = arc.e0
            self.inc0_rad[i] = arc.inc0_rad
            self.raan0_rad[i] = arc.raan0_rad
            self.lam0_rad[i] = arc.lam0_rad
            self.phase_idx[i] = self.phase_to_idx[arc.phase]
            self.sat_idx[i] = self.sat_to_idx[arc.sat_id]
            self.phase_sign[i] = arc.phase_sign
            self.n_obs[i] = nobs
            self.duration_s[i] = arc.duration_s
            self.raan_final_rad[i] = arc.raan_final_rad
            self.lam_final_rad[i] = arc.lam_final_rad
            self.dt_s[i, :nobs] = torch.from_numpy(arc.dt_s[:nobs].astype(np.float32))
            self.a_obs_km[i, :nobs] = torch.from_numpy(arc.a_obs_km[:nobs].astype(np.float32))
            self.mask[i, :nobs] = 1.0

    def __len__(self) -> int:
        return len(self.arcs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "a0_km": self.a0_km[idx],
            "e0": self.e0[idx],
            "inc0_rad": self.inc0_rad[idx],
            "raan0_rad": self.raan0_rad[idx],
            "lam0_rad": self.lam0_rad[idx],
            "phase_idx": self.phase_idx[idx],
            "sat_idx": self.sat_idx[idx],
            "phase_sign": self.phase_sign[idx],
            "n_obs": self.n_obs[idx],
            "duration_s": self.duration_s[idx],
            "raan_final_rad": self.raan_final_rad[idx],
            "lam_final_rad": self.lam_final_rad[idx],
            "dt_s": self.dt_s[idx],           # (max_obs,)
            "a_obs_km": self.a_obs_km[idx],   # (max_obs,)
            "mask": self.mask[idx],            # (max_obs,)
        }


def collate_arcs(batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack arc batch dicts.  All tensors already padded in ArcDataset."""
    out = {}
    for k in batch_list[0].keys():
        out[k] = torch.stack([b[k] for b in batch_list], dim=0)
    return out


# ── Persistence ──────────────────────────────────────────────────────────────

def save_arcs_to_parquet(arcs: List[ArcRecord], path: Path) -> None:
    """Save arcs as a parquet file (one row per observation, with arc_id column)."""
    rows = []
    for arc in arcs:
        for j in range(arc.n_obs):
            rows.append({
                "arc_id": arc.arc_id,
                "sat_id": arc.sat_id,
                "phase": arc.phase,
                "phase_sign": arc.phase_sign,
                "a0_km": arc.a0_km,
                "e0": arc.e0,
                "inc0_rad": arc.inc0_rad,
                "raan0_rad": arc.raan0_rad,
                "lam0_rad": arc.lam0_rad,
                "dt_s": arc.dt_s[j],
                "a_obs_km": arc.a_obs_km[j],
                "raan_final_rad": arc.raan_final_rad,
                "lam_final_rad": arc.lam_final_rad,
                "n_obs": arc.n_obs,
                "duration_s": arc.duration_s,
                "obs_index": j,
            })
    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d arc observations (%d arcs) to %s", len(df), len(arcs), str(path))


def load_arcs_from_parquet(path: Path) -> List[ArcRecord]:
    """Load arcs from a parquet file previously written by save_arcs_to_parquet."""
    df = pd.read_parquet(path)
    arcs = []
    for arc_id, grp in df.groupby("arc_id"):
        grp = grp.sort_values("obs_index").reset_index(drop=True)
        first = grp.iloc[0]
        arcs.append(ArcRecord(
            arc_id=int(arc_id),
            sat_id=str(first["sat_id"]),
            phase=str(first["phase"]),
            phase_sign=float(first["phase_sign"]),
            a0_km=float(first["a0_km"]),
            e0=float(first["e0"]),
            inc0_rad=float(first["inc0_rad"]),
            raan0_rad=float(first["raan0_rad"]),
            lam0_rad=float(first["lam0_rad"]),
            dt_s=grp["dt_s"].to_numpy(dtype=np.float64),
            a_obs_km=grp["a_obs_km"].to_numpy(dtype=np.float64),
            raan_final_rad=float(first["raan_final_rad"]),
            lam_final_rad=float(first["lam_final_rad"]),
            n_obs=int(first["n_obs"]),
            duration_s=float(first["duration_s"]),
        ))
    arcs.sort(key=lambda a: a.arc_id)
    logger.info("Loaded %d arcs from %s", len(arcs), str(path))
    return arcs
