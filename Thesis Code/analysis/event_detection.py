"""Consolidated maneuver/event and mission-phase analytics for TLE time series.

Design goals (Maneuver Detection and Mission-Phase Inference):
- Preserve public API while separating conceptual stages:
  conditioning -> feature construction -> event detection -> post-processing ->
  phase inference.
- Keep threshold-based detectors for interpretability while adding optional
  adaptive thresholding and propagation-compare evidence.
- Treat TLE-based detections as indirect inference from noisy public products,
  not direct maneuver observations.

References used for design rationale in this module:
- Kelecy et al. on TLE-based maneuver detection workflows.
- Mukundan and Wang simplified perturbation comparison logic.
- Li et al. (2018) operational-status monitoring from historical TLE data.
- Zhang et al. multivariate outlier/event detection logic.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    njit = None
    _HAS_NUMBA = False

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

from constants import MU_EARTH, RADIUS_EARTH
from epoch_sync import find_common_epoch_records
from orbital_features import add_altitude_features, add_orbital_phase_features
from orbital_mechanics import coe_from_sv
from state_models import propagate_row_to_teme_state


_FALSE_ENV = {"0", "false", "no", "off"}
_USE_NUMBA_KERNELS_DEFAULT = _HAS_NUMBA and str(os.getenv("EVENT_DETECTION_USE_NUMBA", "1")).strip().lower() not in _FALSE_ENV
_USE_TORCH_EVIDENCE_DEFAULT = _HAS_TORCH and str(os.getenv("EVENT_DETECTION_USE_TORCH", "1")).strip().lower() not in _FALSE_ENV


if _HAS_NUMBA:

    @njit(cache=True)
    def _first_derivative_irregular_numba(y, t_seconds):
        out = np.empty(y.size, dtype=np.float64)
        for i in range(y.size):
            out[i] = np.nan
        if y.size < 2:
            return out

        for i in range(1, y.size):
            dt = t_seconds[i] - t_seconds[i - 1]
            if dt != 0.0:
                out[i] = (y[i] - y[i - 1]) / dt
            else:
                out[i] = np.nan

        out[0] = out[1]
        return out


    @njit(cache=True)
    def _symmetric_peak_mask_numba(signal, prominence, min_prominence):
        n = signal.size
        out = np.zeros(n, dtype=np.bool_)
        if n == 0:
            return out

        for i in range(1, n - 1):
            x = signal[i]
            left = signal[i - 1]
            right = signal[i + 1]
            local_max = x >= left and x >= right
            local_min = x <= left and x <= right
            if (local_max or local_min) and prominence[i] >= min_prominence:
                out[i] = True

        return out


    @njit(cache=True)
    def _bridge_internal_gaps_numba(mask, max_gap_points):
        m = mask.copy()
        n = m.size
        if n == 0 or max_gap_points <= 0:
            return m

        i = 0
        while i < n:
            if m[i]:
                i += 1
                continue

            j = i
            while j < n and (not m[j]):
                j += 1

            gap = j - i
            left_true = i > 0 and m[i - 1]
            right_true = j < n and m[j]
            if left_true and right_true and gap <= max_gap_points:
                for k in range(i, j):
                    m[k] = True

            i = j

        return m


    @njit(cache=True)
    def _nanmean_from_ranges_numba(values, starts, ends):
        out = np.empty(starts.size, dtype=np.float64)
        if starts.size == 0:
            return out

        for i in range(starts.size):
            s = starts[i]
            e = ends[i]
            total = 0.0
            count = 0
            for j in range(s, e):
                v = values[j]
                if np.isfinite(v):
                    total += v
                    count += 1
            if count > 0:
                out[i] = total / count
            else:
                out[i] = np.nan

        return out


# ---- Panel utilities ----
def ensure_panel_sorted(df, object_col="sat_id", time_col="timestamp"):
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    return out.sort_values([object_col, time_col], kind="mergesort").reset_index(drop=True)


def group_object_timeseries(df, object_col="sat_id"):
    sorted_df = ensure_panel_sorted(df, object_col=object_col)
    return sorted_df.groupby(object_col, sort=True)


def get_object_history(df, sat_id, object_col="sat_id", time_col="timestamp"):
    out = df[df[object_col].astype(str) == str(sat_id)].copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    return out.sort_values(time_col, kind="mergesort").reset_index(drop=True)


def get_nearest_epoch_record(df, sat_id, target_time, object_col="sat_id", time_col="timestamp"):
    hist = get_object_history(df, sat_id, object_col=object_col, time_col=time_col)
    if hist.empty:
        return None

    target_ts = pd.Timestamp(target_time)
    hist = hist.dropna(subset=[time_col])
    if hist.empty:
        return None

    deltas = (pd.to_datetime(hist[time_col]) - target_ts).abs()
    best_idx = deltas.idxmin()
    return hist.loc[best_idx]


def get_common_epoch_records(
    df,
    sat_ids,
    target_time=None,
    tolerance="12H",
    method="nearest_intersection",
    object_col="sat_id",
    time_col="timestamp",
):
    if method != "nearest_intersection":
        raise ValueError("Only method='nearest_intersection' is currently supported")

    records, common_epoch, max_abs_delta_seconds = find_common_epoch_records(
        df,
        sat_ids=sat_ids,
        target_time=target_time,
        tolerance=tolerance,
        object_col=object_col,
        time_col=time_col,
    )

    return {
        "records": records,
        "common_epoch": common_epoch,
        "max_abs_delta_seconds": max_abs_delta_seconds,
    }


def build_snapshot(df, target_time, tolerance="12H", one_per_object=True, object_col="sat_id", time_col="timestamp"):
    sorted_df = ensure_panel_sorted(df, object_col=object_col, time_col=time_col)

    if not one_per_object:
        t = pd.Timestamp(target_time)
        tol = pd.to_timedelta(tolerance)
        ts = pd.to_datetime(sorted_df[time_col], errors="coerce")
        mask = (ts - t).abs() <= tol
        return sorted_df.loc[mask].reset_index(drop=True)

    rows = []
    t = pd.Timestamp(target_time)
    tol = pd.to_timedelta(tolerance)

    for _, grp in sorted_df.groupby(object_col, sort=True):
        ts = pd.to_datetime(grp[time_col], errors="coerce")
        valid = ts.notna()
        if not valid.any():
            continue
        grp_v = grp.loc[valid]
        ts_v = ts.loc[valid]
        deltas = (ts_v - t).abs()
        best_idx = deltas.idxmin()
        if deltas.loc[best_idx] <= tol:
            rows.append(grp_v.loc[best_idx])

    if not rows:
        return sorted_df.iloc[0:0].copy()

    return pd.DataFrame(rows).reset_index(drop=True)


def add_time_deltas(df, by="sat_id", time_col="timestamp", output_col="time_delta_seconds"):
    out = ensure_panel_sorted(df, object_col=by, time_col=time_col)
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[output_col] = out.groupby(by, sort=False)[time_col].diff().dt.total_seconds()
    return out


def add_first_differences(df, cols, by="sat_id", time_col="timestamp"):
    out = ensure_panel_sorted(df, object_col=by, time_col=time_col)
    for col in cols:
        out[f"d1_{col}"] = out.groupby(by, sort=False)[col].diff()
    return out


def add_second_differences(df, cols, by="sat_id", time_col="timestamp"):
    out = ensure_panel_sorted(df, object_col=by, time_col=time_col)
    for col in cols:
        out[f"d2_{col}"] = out.groupby(by, sort=False)[col].diff().diff()
    return out


# ---- Time-series conditioning ----
@dataclass
class TimeseriesModelConfig:
    min_records: int = 8
    duplicate_policy: str = "last"
    preferred_object_cols: Tuple[str, ...] = ("norad_cat_id", "sat_id")


def select_object_id_column(df: pd.DataFrame, preferred: Iterable[str] = ("norad_cat_id", "sat_id")) -> str:
    for col in preferred:
        if col in df.columns:
            return col
    raise KeyError("No valid object identifier column found in DataFrame")


def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def get_mean_motion_dot(df_sat: pd.DataFrame) -> np.ndarray:
    if "mean_motion_dot" in df_sat.columns:
        return _safe_numeric(df_sat["mean_motion_dot"])
    if "ballistic_coefficient" in df_sat.columns:
        return _safe_numeric(df_sat["ballistic_coefficient"])
    return np.full(len(df_sat), np.nan, dtype=np.float64)


def get_bstar(df_sat: pd.DataFrame) -> np.ndarray:
    if "bstar" in df_sat.columns:
        return _safe_numeric(df_sat["bstar"])
    if "drag_term" in df_sat.columns:
        return _safe_numeric(df_sat["drag_term"])
    return np.full(len(df_sat), np.nan, dtype=np.float64)


def _drop_duplicate_timestamps(df_sat: pd.DataFrame, time_col: str, policy: str) -> pd.DataFrame:
    keep = "last" if policy == "last" else "first"
    return df_sat.drop_duplicates(subset=[time_col], keep=keep)


def _cadence_diagnostics(time_values: pd.Series) -> Dict[str, Any]:
    ts = pd.to_datetime(time_values, errors="coerce").dropna().sort_values(kind="mergesort")
    if ts.size < 2:
        return {
            "n_intervals": 0,
            "median_cadence_seconds": np.nan,
            "cadence_mad_seconds": np.nan,
            "cadence_cv": np.nan,
            "gap_count": 0,
            "max_gap_seconds": np.nan,
            "p95_gap_seconds": np.nan,
            "is_too_short_for_cadence": True,
        }

    dt = ts.diff().dt.total_seconds().dropna().to_numpy(dtype=np.float64)
    med = float(np.nanmedian(dt)) if dt.size else np.nan
    mad = float(np.nanmedian(np.abs(dt - med))) if dt.size else np.nan
    std = float(np.nanstd(dt)) if dt.size else np.nan
    cv = float(std / med) if np.isfinite(med) and med > 0.0 else np.nan
    gap_threshold = 3.0 * med if np.isfinite(med) and med > 0.0 else np.inf
    gaps = dt[dt > gap_threshold] if np.isfinite(gap_threshold) else np.array([], dtype=np.float64)

    return {
        "n_intervals": int(dt.size),
        "median_cadence_seconds": med,
        "cadence_mad_seconds": mad,
        "cadence_cv": cv,
        "gap_count": int(gaps.size),
        "max_gap_seconds": float(np.nanmax(dt)) if dt.size else np.nan,
        "p95_gap_seconds": float(np.nanpercentile(dt, 95)) if dt.size else np.nan,
        "is_too_short_for_cadence": bool(dt.size < 3),
    }


def condition_timeseries_for_detection(
    df_sat: pd.DataFrame,
    config: Optional[TimeseriesModelConfig] = None,
    time_col: str = "timestamp",
    enrich_phase: bool = True,
    enrich_altitude: bool = True,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Condition one-object history and produce explicit audit metadata."""
    cfg = config or TimeseriesModelConfig()

    out = df_sat.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")

    before_rows = int(len(out))
    out = out.dropna(subset=[time_col])
    after_valid_time = int(len(out))

    out = out.sort_values(time_col, kind="mergesort")

    duplicate_mask_all = out.duplicated(subset=[time_col], keep=False)
    duplicate_rows = out.loc[duplicate_mask_all, [time_col]].copy()
    duplicate_count = int(duplicate_rows.shape[0])
    duplicate_unique_epochs = int(duplicate_rows[time_col].nunique()) if duplicate_count > 0 else 0

    out = _drop_duplicate_timestamps(out, time_col=time_col, policy=cfg.duplicate_policy)

    if enrich_phase and "mean_longitude_deg" not in out.columns and all(c in out.columns for c in ["raan", "aop", "mean_anomaly", "true_anomaly"]):
        out = add_orbital_phase_features(out)
    if enrich_altitude and "altitude_km" not in out.columns and "sma" in out.columns:
        out = add_altitude_features(out)

    out["mean_motion_dot_effective"] = get_mean_motion_dot(out)
    out["bstar_effective"] = get_bstar(out)

    if out.empty:
        out["elapsed_seconds"] = np.array([], dtype=np.float64)
        out["elapsed_days"] = np.array([], dtype=np.float64)
    else:
        t0 = out[time_col].iloc[0]
        dt = (out[time_col] - t0).dt.total_seconds().to_numpy(dtype=np.float64)
        out["elapsed_seconds"] = dt
        out["elapsed_days"] = dt / 86400.0

    cadence = _cadence_diagnostics(out[time_col])
    audit = {
        "rows_before": before_rows,
        "rows_after_valid_time": after_valid_time,
        "rows_after_duplicate_policy": int(len(out)),
        "rows_dropped_invalid_time": int(before_rows - after_valid_time),
        "duplicate_context": {
            "duplicate_policy": cfg.duplicate_policy,
            "duplicate_rows_detected": duplicate_count,
            "duplicate_unique_epochs": duplicate_unique_epochs,
        },
        "cadence_context": cadence,
        "history_too_short": bool(len(out) < max(2, cfg.min_records)),
    }
    return out.reset_index(drop=True), audit


def prepare_satellite_timeseries(
    df_sat: pd.DataFrame,
    config: Optional[TimeseriesModelConfig] = None,
    time_col: str = "timestamp",
    enrich_phase: bool = True,
    enrich_altitude: bool = True,
    return_audit: bool = False,
):
    sat, audit = condition_timeseries_for_detection(
        df_sat,
        config=config,
        time_col=time_col,
        enrich_phase=enrich_phase,
        enrich_altitude=enrich_altitude,
    )
    if return_audit:
        return sat, audit
    return sat


# ---- Time-series model utilities ----
def first_derivative_irregular(y: np.ndarray, t_seconds: np.ndarray, use_numba: Optional[bool] = None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t_seconds, dtype=np.float64)

    if use_numba is None:
        use_numba = _USE_NUMBA_KERNELS_DEFAULT

    if use_numba and _HAS_NUMBA and y.ndim == 1 and t.ndim == 1 and y.shape == t.shape:
        return _first_derivative_irregular_numba(y, t)

    out = np.full(y.shape, np.nan, dtype=np.float64)
    if y.size < 2:
        return out

    dy = np.diff(y)
    dt = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(dt != 0.0, dy / dt, np.nan)
    out[1:] = slope
    if out.size > 1:
        out[0] = out[1]
    return out


def second_derivative_irregular(y: np.ndarray, t_seconds: np.ndarray, use_numba: Optional[bool] = None) -> np.ndarray:
    d1 = first_derivative_irregular(y, t_seconds, use_numba=use_numba)
    return first_derivative_irregular(d1, t_seconds, use_numba=use_numba)


def rolling_poly_trend(t_seconds: np.ndarray, y: np.ndarray, window: int = 11, order: int = 1) -> np.ndarray:
    t = np.asarray(t_seconds, dtype=np.float64)
    x = np.asarray(y, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)

    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    p = max(1, min(int(order), 3))

    if p == 1:
        s = pd.Series(x)
        trend_fast = s.rolling(window=w, center=True, min_periods=max(3, w // 3)).mean()
        return trend_fast.interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=np.float64)

    trend = np.full(n, np.nan, dtype=np.float64)
    half = w // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        tw = t[lo:hi]
        yw = x[lo:hi]
        valid = np.isfinite(tw) & np.isfinite(yw)
        tw = tw[valid]
        yw = yw[valid]
        if tw.size < (p + 1):
            if np.isfinite(x[i]):
                trend[i] = x[i]
            continue

        tc = tw - tw.mean()
        fit_order = min(p, tw.size - 1)
        coef = np.polyfit(tc, yw, deg=fit_order)
        trend[i] = np.polyval(coef, t[i] - tw.mean())

    nan_mask = ~np.isfinite(trend)
    if np.any(nan_mask):
        trend[nan_mask] = x[nan_mask]
    return trend


# ---- Maneuver detection ----
EVENT_TYPES = {
    "semi_major_axis_raise",
    "semi_major_axis_lower",
    "inclination_change",
    "combined_raise_and_plane_change",
    "possible_stationkeeping",
    "possible_disposal_start",
    "possible_anomaly",
    "unknown_event",
}

DETECTOR_FAMILIES = {
    "local_residual",
    "kelecy_filtered_difference",
    "simplified_propagate_compare",
    "smoothed_change_segment",
    "anomaly_guard",
}


@dataclass
class ManeuverDetectionConfig:
    model: TimeseriesModelConfig = field(default_factory=TimeseriesModelConfig)
    enable_numba_kernels: bool = _USE_NUMBA_KERNELS_DEFAULT
    enable_torch_evidence_ops: bool = _USE_TORCH_EVIDENCE_DEFAULT
    torch_device: str = "cuda"

    method1_window: int = 13
    method1_poly_order: int = 1
    method1_sigma_threshold: float = 3.0
    method1_min_event_spacing_seconds: float = 12 * 3600.0

    method2_smooth_span: int = 9
    method2_d1_threshold_sma_km_per_s: float = 0.002
    method2_d2_threshold_sma_km_per_s2: float = 2e-6
    method2_d1_threshold_inc_deg_per_s: float = 5e-5
    method2_d2_threshold_inc_deg_per_s2: float = 2e-8
    method2_hysteresis_points: int = 2

    lag_seconds: float = 6 * 3600.0
    min_records: int = 10

    threshold_mode: str = "fixed"  # fixed | adaptive
    adaptive_window_points: int = 11
    adaptive_sigma_scale: float = 3.0
    adaptive_cadence_sensitivity: float = 0.2

    enable_propagate_compare: bool = False
    propagate_compare_sigma_threshold: float = 3.5
    propagate_state_model: str = "sgp4_preferred"
    propagate_compare_max_rows: int = 1500
    propagate_compare_sgp4_max_rows: int = 600

    evidence_fusion_mode: str = "union"  # union | weighted
    evidence_score_threshold: float = 0.8
    detector_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "local_residual": 1.0,
            "kelecy_filtered_difference": 1.15,
            "smoothed_change_segment": 1.0,
            "simplified_propagate_compare": 1.2,
            "anomaly_guard": 0.5,
        }
    )

    anomaly_screening_enabled: bool = True
    anomaly_sigma_threshold: float = 5.0

    detector_family_mode: str = "all"  # all|kelecy_only|propagate_only|segment_only|custom
    custom_detector_families: Tuple[str, ...] = tuple()

    min_segment_duration_seconds: float = 6 * 3600.0
    min_segment_support_points: int = 2
    max_internal_gap_points: int = 1
    segment_persistence_min_points: int = 2

    event_merge_mode: str = "segment_overlap"  # segment_overlap|time_spacing
    object_level_debounce_seconds: float = 24 * 3600.0
    stationkeeping_episode_spacing_seconds: float = 3 * 24 * 3600.0
    event_score_accept_threshold: float = 1.0
    high_confidence_threshold: float = 1.6

    use_peak_prominence: bool = True
    prominence_window_points: int = 11
    prominence_min_value: Optional[float] = None
    prominence_accept_threshold: float = 0.5

    robust_sigma_iterations: int = 3
    robust_sigma_clip: float = 4.5
    threshold_channel_mode: str = "global"  # global|local_rolling
    channel_specific_sigma_scales: Dict[str, float] = field(default_factory=dict)

    kelecy_window_points: int = 9
    kelecy_poly_order: int = 1
    kelecy_midpoint_mode: str = "extrapolated_midpoint"  # midpoint|extrapolated_midpoint
    kelecy_energy_channels: Tuple[str, ...] = (
        "sma",
        "mean_motion",
        "orbital_energy_proxy_km2_s2",
        "ecc",
    )
    kelecy_plane_channels: Tuple[str, ...] = ("inc", "raan")

    detector_family_spacing_seconds: Dict[str, float] = field(
        default_factory=lambda: {
            "kelecy_filtered_difference": 18 * 3600.0,
            "simplified_propagate_compare": 24 * 3600.0,
            "smoothed_change_segment": 18 * 3600.0,
            "local_residual": 12 * 3600.0,
        }
    )
    progress_every_objects: int = 100
    print_progress: bool = True


@dataclass
class DetectorOutput:
    mask: np.ndarray
    evidence_score: np.ndarray
    evidence_components: Dict[str, np.ndarray]
    detector_metadata: Dict[str, Any] = field(default_factory=dict)
    segment_proposals: List[Dict[str, Any]] = field(default_factory=list)


def _nanmean_from_ranges(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if starts.size == 0:
        return np.array([], dtype=np.float64)

    if _USE_NUMBA_KERNELS_DEFAULT and _HAS_NUMBA:
        values_arr = np.asarray(values, dtype=np.float64)
        starts_arr = np.asarray(starts, dtype=np.int64)
        ends_arr = np.asarray(ends, dtype=np.int64)
        if values_arr.ndim == 1 and starts_arr.ndim == 1 and ends_arr.ndim == 1 and starts_arr.shape == ends_arr.shape:
            return _nanmean_from_ranges_numba(values_arr, starts_arr, ends_arr)

    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    csum = np.r_[0.0, np.cumsum(safe)]
    ccnt = np.r_[0, np.cumsum(finite.astype(np.int64))]

    win_sum = csum[ends] - csum[starts]
    win_cnt = ccnt[ends] - ccnt[starts]
    return np.divide(win_sum, win_cnt, out=np.full(starts.size, np.nan, dtype=np.float64), where=win_cnt > 0)


def _robust_sigma(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    sigma = 1.4826 * mad
    return float(sigma if sigma > 1e-12 else np.std(arr))


def _iterative_sigma(
    x: np.ndarray,
    iterations: int = 3,
    clip_sigma: float = 4.5,
) -> tuple[float, Dict[str, Any]]:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0, {"sigma_estimate": 1.0, "n_used": 0, "iterations": 0, "mode": "iterative_mad"}

    used = arr.copy()
    n_iter = max(1, int(iterations))
    clip = max(2.0, float(clip_sigma))

    for _ in range(n_iter):
        med = float(np.median(used))
        mad = float(np.median(np.abs(used - med)))
        sigma = 1.4826 * mad if mad > 1e-12 else float(np.std(used))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = float(np.std(used))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = 1.0

        z = np.abs(used - med) / sigma
        keep = z <= clip
        if keep.all() or keep.sum() < 8:
            break
        used = used[keep]

    med = float(np.median(used))
    mad = float(np.median(np.abs(used - med)))
    sigma = 1.4826 * mad if mad > 1e-12 else float(np.std(used))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = 1.0
    return float(sigma), {
        "sigma_estimate": float(sigma),
        "n_used": int(used.size),
        "iterations": int(n_iter),
        "mode": "iterative_mad",
    }


def _rolling_mad_scale(x: np.ndarray, window: int) -> np.ndarray:
    arr = pd.Series(np.asarray(x, dtype=np.float64))

    def _mad(v):
        vv = np.asarray(v, dtype=np.float64)
        vv = vv[np.isfinite(vv)]
        if vv.size == 0:
            return np.nan
        med = np.median(vv)
        return 1.4826 * np.median(np.abs(vv - med))

    out = arr.rolling(window=max(5, int(window)), center=True, min_periods=3).apply(_mad, raw=True)
    out = out.bfill().ffill().to_numpy(dtype=np.float64)
    floor = np.nanmedian(out[np.isfinite(out)]) if np.isfinite(out).any() else 1.0
    floor = max(1e-6, float(floor))
    out = np.where(np.isfinite(out) & (out > 1e-9), out, floor)
    return out


def _rolling_prominence_proxy(x: np.ndarray, window: int = 11) -> np.ndarray:
    arr = pd.Series(np.asarray(x, dtype=np.float64))
    base = arr.rolling(window=max(5, int(window)), center=True, min_periods=3).median()
    prom = np.abs(arr - base)
    prom = prom.bfill().ffill().fillna(0.0).to_numpy(dtype=np.float64)
    return prom


def _symmetric_peak_mask(signal: np.ndarray, prominence: np.ndarray, min_prominence: float) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    prom = np.asarray(prominence, dtype=np.float64)
    if x.size == 0:
        return np.array([], dtype=bool)

    if _USE_NUMBA_KERNELS_DEFAULT and _HAS_NUMBA and x.ndim == 1 and prom.ndim == 1 and x.shape == prom.shape:
        return _symmetric_peak_mask_numba(x, prom, float(min_prominence))

    left = np.r_[x[0], x[:-1]]
    right = np.r_[x[1:], x[-1]]
    local_max = (x >= left) & (x >= right)
    local_min = (x <= left) & (x <= right)
    extrema = local_max | local_min
    if x.size > 1:
        extrema[0] = False
        extrema[-1] = False
    return extrema & (prom >= float(min_prominence))


def _bridge_internal_gaps(mask: np.ndarray, max_gap_points: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool).copy()
    if m.size == 0 or max_gap_points <= 0:
        return m

    if _USE_NUMBA_KERNELS_DEFAULT and _HAS_NUMBA and m.ndim == 1:
        return _bridge_internal_gaps_numba(m, int(max_gap_points))

    i = 0
    n = m.size
    max_gap = int(max_gap_points)
    while i < n:
        if m[i]:
            i += 1
            continue
        j = i
        while j < n and not m[j]:
            j += 1
        gap = j - i
        left_true = i > 0 and m[i - 1]
        right_true = j < n and m[j]
        if left_true and right_true and gap <= max_gap:
            m[i:j] = True
        i = j
    return m


def _mask_to_segments(mask: np.ndarray, max_gap_points: int = 0) -> List[Tuple[int, int]]:
    m = _bridge_internal_gaps(mask, max_gap_points=max_gap_points)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return []
    split = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, split)
    return [(int(g[0]), int(g[-1])) for g in groups if g.size > 0]


def _segment_duration_seconds(times: np.ndarray, start_idx: int, end_idx: int) -> float:
    if times.size == 0:
        return 0.0
    t0 = pd.Timestamp(times[start_idx])
    t1 = pd.Timestamp(times[end_idx])
    if pd.isna(t0) or pd.isna(t1):
        return 0.0
    return max(0.0, float((t1 - t0).total_seconds()))


def _robust_zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(arr)
    sigma = _robust_sigma(arr)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - med) / sigma


def _build_candidate_rows(mask: np.ndarray, hysteresis_points: int = 1) -> np.ndarray:
    m = np.asarray(mask, dtype=bool).copy()
    if m.size == 0:
        return m
    h = max(0, int(hysteresis_points))
    if h == 0:
        return m
    for _ in range(h):
        left = np.r_[False, m[:-1]]
        right = np.r_[m[1:], False]
        m = m | left | right
    return m


def _cadence_multiplier(cadence_context: Dict[str, Any], sensitivity: float) -> float:
    median_cadence = cadence_context.get("median_cadence_seconds", np.nan)
    cv = cadence_context.get("cadence_cv", np.nan)
    if not np.isfinite(median_cadence) or median_cadence <= 0.0:
        return 1.0
    cv_term = float(np.clip(np.nan_to_num(cv, nan=0.0), 0.0, 3.0))
    return float(1.0 + sensitivity * cv_term)


def _threshold_from_signal(
    signal: np.ndarray,
    cfg: ManeuverDetectionConfig,
    channel_name: str,
    sigma_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        z = np.array([], dtype=np.float64)
        prom = np.array([], dtype=np.float64)
        return np.array([], dtype=bool), z, prom, {}

    center = np.nanmedian(x)
    mode = str(cfg.threshold_channel_mode or "global").strip().lower()
    sigma_scale = float(cfg.channel_specific_sigma_scales.get(channel_name, 1.0))

    if mode == "local_rolling":
        sigma_vec = _rolling_mad_scale(x, window=max(7, cfg.adaptive_window_points))
        z = np.abs(x - center) / np.where(sigma_vec > 1e-12, sigma_vec, 1.0)
        sigma_repr = float(np.nanmedian(sigma_vec[np.isfinite(sigma_vec)])) if np.isfinite(sigma_vec).any() else 1.0
        sigma_meta = {
            "sigma_estimate": sigma_repr,
            "n_used": int(np.isfinite(x).sum()),
            "iterations": 1,
            "mode": "local_rolling_mad",
        }
    else:
        sigma, sigma_meta = _iterative_sigma(
            x,
            iterations=cfg.robust_sigma_iterations,
            clip_sigma=cfg.robust_sigma_clip,
        )
        z = np.abs(x - center) / max(1e-9, sigma)

    threshold = float(sigma_threshold) * max(1e-6, sigma_scale)
    amp_mask = z >= threshold

    prom = _rolling_prominence_proxy(x, window=cfg.prominence_window_points)
    if cfg.prominence_min_value is not None:
        prom_thr = float(cfg.prominence_min_value)
    else:
        prom_sigma, _ = _iterative_sigma(prom, iterations=2, clip_sigma=4.0)
        prom_thr = float(np.nanmedian(prom) + cfg.prominence_accept_threshold * prom_sigma)
        prom_thr = max(0.0, prom_thr)

    if cfg.use_peak_prominence:
        peak_mask = _symmetric_peak_mask(x, prom, prom_thr)
        mask = amp_mask & peak_mask
    else:
        mask = amp_mask

    meta = {
        "threshold_mode": mode,
        "sigma_threshold": float(sigma_threshold),
        "sigma_scale": float(sigma_scale),
        "prominence_threshold": float(prom_thr),
        "channel": str(channel_name),
        "sigma_metadata": sigma_meta,
    }
    return mask, np.nan_to_num(z, nan=0.0), np.nan_to_num(prom, nan=0.0), meta


def _classify_event(delta_sma: float, delta_inc: float, bstar_level: float, d1_sma: float) -> str:
    abs_sma = abs(delta_sma)
    abs_inc = abs(delta_inc)

    if delta_sma > 8.0 and abs_inc > 0.05:
        return "combined_raise_and_plane_change"
    if delta_sma > 8.0:
        return "semi_major_axis_raise"
    if delta_sma < -8.0 and (bstar_level > 3e-4 or d1_sma < -0.003):
        return "possible_disposal_start"
    if delta_sma < -8.0:
        return "semi_major_axis_lower"
    if abs_inc > 0.05:
        return "inclination_change"
    if abs_sma <= 4.0 and abs_inc <= 0.02:
        return "possible_stationkeeping"
    if abs_sma > 20.0 or abs_inc > 0.2:
        return "possible_anomaly"
    return "unknown_event"


def _merge_events(events: List[Dict[str, Any]], min_spacing_seconds: float) -> List[Dict[str, Any]]:
    if not events:
        return []
    events_sorted = sorted(events, key=lambda e: e["detection_time"])
    merged = [events_sorted[0]]

    for ev in events_sorted[1:]:
        prev = merged[-1]
        dt = (pd.Timestamp(ev["detection_time"]) - pd.Timestamp(prev["detection_time"])).total_seconds()
        same_type = ev["event_type"] == prev["event_type"]
        if dt <= min_spacing_seconds and same_type:
            if ev["event_score"] > prev["event_score"]:
                merged[-1] = ev
        else:
            merged.append(ev)
    return merged


def _resolve_object_fields(df_sat: pd.DataFrame) -> tuple[str, str, str]:
    sat_id = str(df_sat["sat_id"].iloc[0]) if "sat_id" in df_sat.columns and len(df_sat) > 0 else ""
    norad = str(df_sat["norad_cat_id"].iloc[0]) if "norad_cat_id" in df_sat.columns and len(df_sat) > 0 else ""
    object_id = norad if norad not in ["", "nan", "None"] else sat_id
    return object_id, sat_id, norad


def build_maneuver_feature_frame(df_sat: pd.DataFrame, use_numba: Optional[bool] = None) -> pd.DataFrame:
    """Create detector feature space from conditioned time history.

    Added optional features include mean motion, eccentricity, RAAN,
    energy/angular momentum direction proxies, and shell-relative residuals.
    """
    work = df_sat.copy()
    n = len(work)
    if n == 0:
        return work

    t = pd.to_numeric(work["elapsed_seconds"], errors="coerce").to_numpy(dtype=np.float64)

    for col in ["sma", "inc", "ecc", "raan", "mean_motion", "altitude_km"]:
        if col in work.columns:
            vec = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=np.float64)
            work[f"d1_{col}"] = first_derivative_irregular(vec, t, use_numba=use_numba)
            work[f"d2_{col}"] = second_derivative_irregular(vec, t, use_numba=use_numba)

    if "sma" in work.columns:
        sma = pd.to_numeric(work["sma"], errors="coerce").to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            work["orbital_energy_proxy_km2_s2"] = -MU_EARTH / (2.0 * sma)

    if "inc" in work.columns and "raan" in work.columns:
        inc_rad = np.deg2rad(pd.to_numeric(work["inc"], errors="coerce").to_numpy(dtype=np.float64))
        raan_rad = np.deg2rad(pd.to_numeric(work["raan"], errors="coerce").to_numpy(dtype=np.float64))
        work["hhat_x"] = np.sin(inc_rad) * np.sin(raan_rad)
        work["hhat_y"] = -np.sin(inc_rad) * np.cos(raan_rad)
        work["hhat_z"] = np.cos(inc_rad)

    shell_col = "candidate_shell_id" if "candidate_shell_id" in work.columns else ("shell_id" if "shell_id" in work.columns else None)
    if "altitude_km" in work.columns and shell_col is not None:
        alt = pd.to_numeric(work["altitude_km"], errors="coerce")
        shell_med = alt.groupby(work[shell_col]).transform("median")
        work["shell_relative_altitude_residual_km"] = alt - shell_med

        if "d1_altitude_km" in work.columns:
            drift = pd.to_numeric(work["d1_altitude_km"], errors="coerce")
            shell_drift_med = drift.groupby(work[shell_col]).transform("median")
            work["shell_relative_secular_drift_residual_km_s"] = drift - shell_drift_med
    else:
        if "sma" in work.columns:
            work["shell_relative_altitude_residual_km"] = pd.to_numeric(work["sma"], errors="coerce") - RADIUS_EARTH
        else:
            work["shell_relative_altitude_residual_km"] = np.nan
        work["shell_relative_secular_drift_residual_km_s"] = np.nan

    # Segment-oriented feature channels for robust event scoring.
    if "sma" in work.columns:
        sma = pd.to_numeric(work["sma"], errors="coerce").to_numpy(dtype=np.float64)
        trend_sma = rolling_poly_trend(t, sma, window=11, order=1)
        work["rolling_poly_residual_sma"] = sma - trend_sma
        work["local_variance_sma"] = pd.Series(sma).rolling(window=11, center=True, min_periods=3).var().to_numpy(dtype=np.float64)
        work["local_mad_sma"] = _rolling_mad_scale(sma, window=11)
        work["rolling_prominence_proxy_sma"] = _rolling_prominence_proxy(sma, window=11)

        d1_sma = pd.to_numeric(work.get("d1_sma", np.nan), errors="coerce").to_numpy(dtype=np.float64)
        sign = np.sign(d1_sma)
        work["sign_consistency_score"] = (
            pd.Series(sign)
            .rolling(window=9, center=True, min_periods=3)
            .apply(lambda v: float(np.abs(np.nanmean(v))), raw=True)
            .to_numpy(dtype=np.float64)
        )

        local_scale = _rolling_mad_scale(d1_sma, window=9)
        rolled_slope = pd.Series(d1_sma).rolling(window=9, center=True, min_periods=3).mean().to_numpy(dtype=np.float64)
        work["sustained_slope_score"] = np.divide(
            np.abs(rolled_slope),
            np.where(local_scale > 1e-9, local_scale, 1.0),
        )

        sma_jump = np.abs(d1_sma)
        jump_floor = np.nanmedian(sma_jump[np.isfinite(sma_jump)]) if np.isfinite(sma_jump).any() else 0.0
        jump_sigma, _ = _iterative_sigma(sma_jump, iterations=2, clip_sigma=4.0)
        jump_thr = float(jump_floor + 2.0 * jump_sigma)
        jump_mask = np.isfinite(sma_jump) & (sma_jump > jump_thr)
        work["event_density_suppression_term"] = (
            pd.Series(jump_mask.astype(np.float64))
            .rolling(window=15, center=True, min_periods=3)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    if "inc" in work.columns:
        inc = pd.to_numeric(work["inc"], errors="coerce").to_numpy(dtype=np.float64)
        trend_inc = rolling_poly_trend(t, inc, window=11, order=1)
        work["rolling_poly_residual_inc"] = inc - trend_inc
        work["local_variance_inc"] = pd.Series(inc).rolling(window=11, center=True, min_periods=3).var().to_numpy(dtype=np.float64)
        work["local_mad_inc"] = _rolling_mad_scale(inc, window=11)
        work["rolling_prominence_proxy_inc"] = _rolling_prominence_proxy(inc, window=11)

    work["propagation_residual_history"] = np.nan
    work["guard_short_history"] = False
    work["guard_irregular_cadence"] = False
    work["guard_duplicate_epochs"] = False
    work["guard_sparse_phase_window"] = False
    work["guard_isolated_bstar_spike"] = False

    return work


def _method1_local_residual(feature_df: pd.DataFrame, cfg: ManeuverDetectionConfig, cadence_context: Dict[str, Any]) -> DetectorOutput:
    t = pd.to_numeric(feature_df["elapsed_seconds"], errors="coerce").to_numpy(dtype=np.float64)
    sma = pd.to_numeric(feature_df["sma"], errors="coerce").to_numpy(dtype=np.float64)
    inc = pd.to_numeric(feature_df["inc"], errors="coerce").to_numpy(dtype=np.float64)

    trend_sma = rolling_poly_trend(t, sma, window=cfg.method1_window, order=cfg.method1_poly_order)
    trend_inc = rolling_poly_trend(t, inc, window=cfg.method1_window, order=cfg.method1_poly_order)

    res_sma = sma - trend_sma
    res_inc = inc - trend_inc

    mask_sma, z_sma, prom_sma, meta_sma = _threshold_from_signal(
        res_sma,
        cfg,
        channel_name="local_residual_sma",
        sigma_threshold=cfg.method1_sigma_threshold,
    )
    mask_inc, z_inc, prom_inc, meta_inc = _threshold_from_signal(
        res_inc,
        cfg,
        channel_name="local_residual_inc",
        sigma_threshold=cfg.method1_sigma_threshold,
    )

    threshold_scale = 1.0
    if str(cfg.threshold_mode).lower() == "adaptive":
        threshold_scale = _cadence_multiplier(cadence_context, cfg.adaptive_cadence_sensitivity)

    raw_mask = mask_sma | mask_inc
    mask = _bridge_internal_gaps(raw_mask, cfg.max_internal_gap_points)
    evidence = np.maximum(z_sma, z_inc) / max(1.0, threshold_scale)
    evidence = np.nan_to_num(evidence, nan=0.0)

    proposals = []
    for start_idx, end_idx in _mask_to_segments(mask, max_gap_points=cfg.max_internal_gap_points):
        proposals.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "support_points": int(end_idx - start_idx + 1),
                "peak_evidence": float(np.nanmax(evidence[start_idx : end_idx + 1])),
                "integrated_evidence": float(np.nanmean(evidence[start_idx : end_idx + 1])),
            }
        )

    return DetectorOutput(
        mask=mask,
        evidence_score=evidence,
        evidence_components={
            "z_sma": z_sma,
            "z_inc": z_inc,
            "prominence_sma": prom_sma,
            "prominence_inc": prom_inc,
        },
        detector_metadata={
            "detector_family": "local_residual",
            "threshold_channel_mode": str(cfg.threshold_channel_mode),
            "channel_calibration": {
                "local_residual_sma": meta_sma,
                "local_residual_inc": meta_inc,
            },
        },
        segment_proposals=proposals,
    )


def _poly_predict_at(tw: np.ndarray, yw: np.ndarray, t_eval: float, poly_order: int) -> float:
    t_local = np.asarray(tw, dtype=np.float64)
    y_local = np.asarray(yw, dtype=np.float64)
    valid = np.isfinite(t_local) & np.isfinite(y_local)
    t_local = t_local[valid]
    y_local = y_local[valid]
    if t_local.size < 3:
        return np.nan
    deg = int(max(1, min(poly_order, t_local.size - 1)))
    t_ref = float(np.nanmean(t_local))
    coef = np.polyfit(t_local - t_ref, y_local, deg=deg)
    return float(np.polyval(coef, t_eval - t_ref))


def _adjacent_filtered_difference_signal_linear_fast(
    t_seconds: np.ndarray,
    values: np.ndarray,
    window_points: int,
    midpoint_mode: str,
) -> np.ndarray:
    t = np.asarray(t_seconds, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    n = y.size
    w = int(max(3, window_points))
    out = np.full(n, np.nan, dtype=np.float64)
    if n < (2 * w + 2):
        return out

    valid = np.isfinite(t) & np.isfinite(y)
    v = valid.astype(np.int64)
    t_safe = np.where(valid, t, 0.0)
    y_safe = np.where(valid, y, 0.0)
    t2_safe = np.where(valid, t * t, 0.0)
    ty_safe = np.where(valid, t * y, 0.0)

    c_n = np.r_[0, np.cumsum(v)]
    c_t = np.r_[0.0, np.cumsum(t_safe)]
    c_y = np.r_[0.0, np.cumsum(y_safe)]
    c_t2 = np.r_[0.0, np.cumsum(t2_safe)]
    c_ty = np.r_[0.0, np.cumsum(ty_safe)]

    mode = str(midpoint_mode or "extrapolated_midpoint").strip().lower()

    def _predict_window(lo: int, hi: int, t_eval: float) -> float:
        n_win = int(c_n[hi] - c_n[lo])
        if n_win < 3:
            return np.nan

        s_t = float(c_t[hi] - c_t[lo])
        s_y = float(c_y[hi] - c_y[lo])
        s_t2 = float(c_t2[hi] - c_t2[lo])
        s_ty = float(c_ty[hi] - c_ty[lo])

        denom = float(n_win) * s_t2 - s_t * s_t
        if not np.isfinite(denom) or abs(denom) <= 1e-12:
            return float(s_y / float(n_win))

        slope = (float(n_win) * s_ty - s_t * s_y) / denom
        intercept = (s_y - slope * s_t) / float(n_win)
        return float(intercept + slope * t_eval)

    for i in range(w, n - w - 1):
        t_left = t[i - 1]
        t_right = t[i + 1]
        if not (np.isfinite(t_left) and np.isfinite(t_right)):
            continue

        if mode == "midpoint":
            t_eval = 0.5 * (t_left + t_right)
        else:
            t_eval = t_right

        lead_pred = _predict_window(i - w, i, t_eval)
        trail_pred = _predict_window(i + 1, i + 1 + w, t_eval)
        if np.isfinite(lead_pred) and np.isfinite(trail_pred):
            out[i] = float(trail_pred - lead_pred)

    return out


def _adjacent_filtered_difference_signal(
    t_seconds: np.ndarray,
    values: np.ndarray,
    window_points: int,
    poly_order: int,
    midpoint_mode: str,
) -> np.ndarray:
    if int(poly_order) == 1:
        return _adjacent_filtered_difference_signal_linear_fast(
            t_seconds=t_seconds,
            values=values,
            window_points=window_points,
            midpoint_mode=midpoint_mode,
        )

    t = np.asarray(t_seconds, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    n = y.size
    w = int(max(3, window_points))
    if n < (2 * w + 2):
        return np.full(n, np.nan, dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    mode = str(midpoint_mode or "extrapolated_midpoint").strip().lower()
    for i in range(w, n - w - 1):
        lead_t = t[i - w : i]
        lead_y = y[i - w : i]
        trail_t = t[i + 1 : i + 1 + w]
        trail_y = y[i + 1 : i + 1 + w]

        if mode == "midpoint":
            t_eval = 0.5 * (lead_t[-1] + trail_t[0])
        else:
            half_step = 0.5 * (trail_t[0] - lead_t[-1])
            t_eval = 0.5 * (lead_t[-1] + trail_t[0]) + half_step

        lead_pred = _poly_predict_at(lead_t, lead_y, t_eval=t_eval, poly_order=poly_order)
        trail_pred = _poly_predict_at(trail_t, trail_y, t_eval=t_eval, poly_order=poly_order)
        if np.isfinite(lead_pred) and np.isfinite(trail_pred):
            out[i] = float(trail_pred - lead_pred)
    return out


def _detector_kelecy_filtered_difference(
    feature_df: pd.DataFrame,
    cfg: ManeuverDetectionConfig,
    cadence_context: Dict[str, Any],
) -> DetectorOutput:
    t = pd.to_numeric(feature_df["elapsed_seconds"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(feature_df)
    combined_mask = np.zeros(n, dtype=bool)
    combined_evidence = np.zeros(n, dtype=np.float64)
    evidence_components: Dict[str, np.ndarray] = {}
    channel_meta: Dict[str, Any] = {}
    energy_signals: List[np.ndarray] = []
    plane_signals: List[np.ndarray] = []

    threshold_scale = 1.0
    if str(cfg.threshold_mode).lower() == "adaptive":
        threshold_scale = _cadence_multiplier(cadence_context, cfg.adaptive_cadence_sensitivity)

    all_channels = list(cfg.kelecy_energy_channels) + list(cfg.kelecy_plane_channels)
    for channel in all_channels:
        if channel not in feature_df.columns:
            continue
        values = pd.to_numeric(feature_df[channel], errors="coerce").to_numpy(dtype=np.float64)
        diff = _adjacent_filtered_difference_signal(
            t_seconds=t,
            values=values,
            window_points=cfg.kelecy_window_points,
            poly_order=cfg.kelecy_poly_order,
            midpoint_mode=cfg.kelecy_midpoint_mode,
        )
        mask, z, prom, meta = _threshold_from_signal(
            diff,
            cfg,
            channel_name=f"kelecy_{channel}",
            sigma_threshold=cfg.method1_sigma_threshold * threshold_scale,
        )

        combined_mask |= mask
        combined_evidence = np.maximum(combined_evidence, np.nan_to_num(z, nan=0.0))
        evidence_components[f"{channel}_filtered_difference"] = diff
        evidence_components[f"{channel}_z"] = z
        evidence_components[f"{channel}_prominence"] = prom
        channel_meta[channel] = meta

        if channel in cfg.kelecy_energy_channels:
            energy_signals.append(diff)
        if channel in cfg.kelecy_plane_channels:
            plane_signals.append(diff)

    if energy_signals:
        energy_stack = np.vstack(energy_signals)
        energy_finite = np.isfinite(energy_stack)
        energy_sum = np.sum(np.where(energy_finite, energy_stack, 0.0), axis=0)
        energy_cnt = np.sum(energy_finite, axis=0)
        energy_mean = np.full(n, np.nan, dtype=np.float64)
        valid = energy_cnt > 0
        energy_mean[valid] = energy_sum[valid] / energy_cnt[valid]
        evidence_components["energy_diff"] = energy_mean
    else:
        evidence_components["energy_diff"] = np.full(n, np.nan, dtype=np.float64)

    if plane_signals:
        plane_stack = np.vstack(plane_signals)
        plane_finite = np.isfinite(plane_stack)
        plane_sum = np.sum(np.where(plane_finite, plane_stack, 0.0), axis=0)
        plane_cnt = np.sum(plane_finite, axis=0)
        plane_mean = np.full(n, np.nan, dtype=np.float64)
        valid = plane_cnt > 0
        plane_mean[valid] = plane_sum[valid] / plane_cnt[valid]
        evidence_components["plane_diff"] = plane_mean
    else:
        evidence_components["plane_diff"] = np.full(n, np.nan, dtype=np.float64)

    evidence_components["prominence_proxy"] = np.nan_to_num(
        _rolling_prominence_proxy(evidence_components.get("energy_diff", np.zeros(n)), window=cfg.prominence_window_points),
        nan=0.0,
    )

    proposals = []
    for start_idx, end_idx in _mask_to_segments(combined_mask, max_gap_points=cfg.max_internal_gap_points):
        proposals.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "support_points": int(end_idx - start_idx + 1),
                "peak_evidence": float(np.nanmax(combined_evidence[start_idx : end_idx + 1])),
                "integrated_evidence": float(np.nanmean(combined_evidence[start_idx : end_idx + 1])),
            }
        )

    return DetectorOutput(
        mask=combined_mask,
        evidence_score=np.nan_to_num(combined_evidence, nan=0.0),
        evidence_components=evidence_components,
        detector_metadata={
            "detector_family": "kelecy_filtered_difference",
            "window_points": int(cfg.kelecy_window_points),
            "poly_order": int(cfg.kelecy_poly_order),
            "midpoint_mode": str(cfg.kelecy_midpoint_mode),
            "threshold_mode": str(cfg.threshold_channel_mode),
            "channel_calibration": channel_meta,
        },
        segment_proposals=proposals,
    )


def _method2_smoothed_change(feature_df: pd.DataFrame, cfg: ManeuverDetectionConfig, cadence_context: Dict[str, Any]) -> DetectorOutput:
    t = pd.to_numeric(feature_df["elapsed_seconds"], errors="coerce").to_numpy(dtype=np.float64)
    sma = pd.to_numeric(feature_df["sma"], errors="coerce").to_numpy(dtype=np.float64)
    inc = pd.to_numeric(feature_df["inc"], errors="coerce").to_numpy(dtype=np.float64)

    sma_sm = pd.Series(sma).ewm(span=max(3, cfg.method2_smooth_span), adjust=False).mean().to_numpy()
    inc_sm = pd.Series(inc).ewm(span=max(3, cfg.method2_smooth_span), adjust=False).mean().to_numpy()

    d1_sma = first_derivative_irregular(sma_sm, t, use_numba=cfg.enable_numba_kernels)
    d2_sma = second_derivative_irregular(sma_sm, t, use_numba=cfg.enable_numba_kernels)
    d1_inc = first_derivative_irregular(inc_sm, t, use_numba=cfg.enable_numba_kernels)
    d2_inc = second_derivative_irregular(inc_sm, t, use_numba=cfg.enable_numba_kernels)

    if str(cfg.threshold_mode).lower() == "adaptive":
        scale = _cadence_multiplier(cadence_context, cfg.adaptive_cadence_sensitivity)
        th_d1_sma = max(
            cfg.method2_d1_threshold_sma_km_per_s,
            np.nanmedian(np.abs(d1_sma)) + cfg.adaptive_sigma_scale * _robust_sigma(d1_sma),
        ) * scale
        th_d2_sma = max(
            cfg.method2_d2_threshold_sma_km_per_s2,
            np.nanmedian(np.abs(d2_sma)) + cfg.adaptive_sigma_scale * _robust_sigma(d2_sma),
        ) * scale
        th_d1_inc = max(
            cfg.method2_d1_threshold_inc_deg_per_s,
            np.nanmedian(np.abs(d1_inc)) + cfg.adaptive_sigma_scale * _robust_sigma(d1_inc),
        ) * scale
        th_d2_inc = max(
            cfg.method2_d2_threshold_inc_deg_per_s2,
            np.nanmedian(np.abs(d2_inc)) + cfg.adaptive_sigma_scale * _robust_sigma(d2_inc),
        ) * scale
    else:
        th_d1_sma = float(cfg.method2_d1_threshold_sma_km_per_s)
        th_d2_sma = float(cfg.method2_d2_threshold_sma_km_per_s2)
        th_d1_inc = float(cfg.method2_d1_threshold_inc_deg_per_s)
        th_d2_inc = float(cfg.method2_d2_threshold_inc_deg_per_s2)

    raw = (
        (np.abs(d1_sma) >= th_d1_sma)
        | (np.abs(d2_sma) >= th_d2_sma)
        | (np.abs(d1_inc) >= th_d1_inc)
        | (np.abs(d2_inc) >= th_d2_inc)
    )
    raw = _build_candidate_rows(raw, hysteresis_points=cfg.method2_hysteresis_points)
    mask = _bridge_internal_gaps(raw, cfg.max_internal_gap_points)

    stack_for_evidence = np.vstack(
        [
            np.abs(d1_sma) / max(th_d1_sma, 1e-12),
            np.abs(d2_sma) / max(th_d2_sma, 1e-12),
            np.abs(d1_inc) / max(th_d1_inc, 1e-12),
            np.abs(d2_inc) / max(th_d2_inc, 1e-12),
        ]
    )

    if cfg.enable_torch_evidence_ops and _HAS_TORCH:
        try:
            requested_device = str(cfg.torch_device or "cpu").strip().lower()
            if requested_device.startswith("cuda") and (not torch.cuda.is_available()):
                requested_device = "cpu"
            t_stack = torch.from_numpy(np.nan_to_num(stack_for_evidence, nan=0.0)).to(requested_device)
            evidence = torch.amax(t_stack, dim=0).detach().cpu().numpy().astype(np.float64)
        except Exception:
            evidence = np.nanmax(stack_for_evidence, axis=0)
    else:
        evidence = np.nanmax(stack_for_evidence, axis=0)

    evidence = np.nan_to_num(evidence, nan=0.0)
    prominence = _rolling_prominence_proxy(np.nan_to_num(d1_sma, nan=0.0), window=cfg.prominence_window_points)

    proposals = []
    ts = pd.to_datetime(feature_df["timestamp"], errors="coerce").to_numpy()
    for start_idx, end_idx in _mask_to_segments(mask, max_gap_points=cfg.max_internal_gap_points):
        support_points = int(end_idx - start_idx + 1)
        duration_seconds = _segment_duration_seconds(ts, start_idx, end_idx)
        if support_points < max(cfg.min_segment_support_points, cfg.segment_persistence_min_points):
            continue
        if duration_seconds < float(cfg.min_segment_duration_seconds):
            continue
        proposals.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "support_points": support_points,
                "duration_seconds": float(duration_seconds),
                "peak_evidence": float(np.nanmax(evidence[start_idx : end_idx + 1])),
                "integrated_evidence": float(np.nanmean(evidence[start_idx : end_idx + 1])),
                "prominence": float(np.nanmax(prominence[start_idx : end_idx + 1])),
            }
        )

    return DetectorOutput(
        mask=mask,
        evidence_score=evidence,
        evidence_components={
            "d1_sma": d1_sma,
            "d2_sma": d2_sma,
            "d1_inc": d1_inc,
            "d2_inc": d2_inc,
            "prominence_proxy": prominence,
        },
        detector_metadata={
            "detector_family": "smoothed_change_segment",
            "thresholds": {
                "d1_sma": float(th_d1_sma),
                "d2_sma": float(th_d2_sma),
                "d1_inc": float(th_d1_inc),
                "d2_inc": float(th_d2_inc),
            },
            "threshold_mode": str(cfg.threshold_mode),
            "min_segment_duration_seconds": float(cfg.min_segment_duration_seconds),
            "min_segment_support_points": int(cfg.min_segment_support_points),
        },
        segment_proposals=proposals,
    )


def _predict_with_sgp4_like(prev_row: Any, current_time: pd.Timestamp, state_model: str) -> tuple[Dict[str, float], str]:
    prediction: Dict[str, float] = {}
    state = propagate_row_to_teme_state(prev_row, current_time, state_model=state_model)
    if not state.get("ok", False):
        return prediction, "unavailable"

    r_km = np.asarray(state.get("r_km"), dtype=np.float64)
    v_kms = np.asarray(state.get("v_kms"), dtype=np.float64)
    if r_km.shape != (3,) or v_kms.shape != (3,):
        return prediction, "unavailable"

    try:
        h, e, raan, incl, _, _ = coe_from_sv(r_km, v_kms, MU_EARTH)
    except Exception:
        return prediction, "unavailable"

    prediction["ecc"] = float(e)
    prediction["inc"] = float(np.rad2deg(incl))
    prediction["raan"] = float(np.rad2deg(raan))
    if np.isfinite(e) and e < 1.0 and np.isfinite(h) and h > 0.0:
        a = h * h / (MU_EARTH * max(1e-12, 1.0 - e * e))
        prediction["sma"] = float(a)
        if np.isfinite(a) and a > 0.0:
            n_rad_s = np.sqrt(MU_EARTH / (a**3))
            prediction["mean_motion"] = float(n_rad_s * 86400.0 / (2.0 * np.pi))
    return prediction, "sgp4"


def _predict_with_simplified_perturbation(
    feature_df: pd.DataFrame,
    index: int,
    dt_seconds: float,
    base_fields: Optional[Dict[str, np.ndarray]] = None,
    slope_fields: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    prediction: Dict[str, float] = {}
    if index < 1 or not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        return prediction

    fields = ["sma", "inc", "raan", "ecc", "mean_motion"]

    if base_fields is not None:
        for field_name in fields:
            base_arr = base_fields.get(field_name)
            if base_arr is None or base_arr.size <= (index - 1):
                continue
            base = float(base_arr[index - 1])
            slope_arr = None if slope_fields is None else slope_fields.get(field_name)
            slope = float(slope_arr[index - 1]) if (slope_arr is not None and slope_arr.size > (index - 1)) else np.nan
            if np.isfinite(base):
                prediction[field_name] = base if not np.isfinite(slope) else float(base + slope * dt_seconds)
        return prediction

    prev = feature_df.iloc[index - 1]
    for field_name in fields:
        if field_name not in feature_df.columns:
            continue
        base = float(pd.to_numeric(prev.get(field_name), errors="coerce"))
        d1_col = f"d1_{field_name}"
        slope = float(pd.to_numeric(prev.get(d1_col), errors="coerce")) if d1_col in feature_df.columns else np.nan
        if np.isfinite(base):
            prediction[field_name] = base if not np.isfinite(slope) else float(base + slope * dt_seconds)
    return prediction


def _method3_propagate_compare(feature_df: pd.DataFrame, cfg: ManeuverDetectionConfig) -> DetectorOutput:
    n = len(feature_df)
    mask = np.zeros(n, dtype=bool)
    evidence = np.zeros(n, dtype=np.float64)
    if n == 0:
        return DetectorOutput(mask=mask, evidence_score=evidence, evidence_components={"residual_score": evidence})

    try:
        max_rows = int(cfg.propagate_compare_max_rows)
    except Exception:
        max_rows = 1500
    if max_rows > 0 and n > max_rows:
        return DetectorOutput(
            mask=mask,
            evidence_score=evidence,
            evidence_components={"residual_score": evidence},
            detector_metadata={
                "detector_family": "simplified_propagate_compare",
                "skipped": True,
                "skip_reason": "row_cap_exceeded",
                "row_count": int(n),
                "max_rows": int(max_rows),
            },
        )

    residual_score = np.zeros(n, dtype=np.float64)
    model_used = np.array(["none"] * n, dtype=object)

    times = pd.to_datetime(feature_df["timestamp"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    times_pd = [pd.Timestamp(t) for t in times]
    elapsed = pd.to_numeric(feature_df["elapsed_seconds"], errors="coerce").to_numpy(dtype=np.float64)

    available_fields = [name for name in ["sma", "mean_motion", "inc", "ecc", "raan"] if name in feature_df.columns]
    if not available_fields:
        return DetectorOutput(mask=mask, evidence_score=evidence, evidence_components={"residual_score": evidence})

    rows_cache = feature_df.to_dict("records")
    field_arrays: Dict[str, np.ndarray] = {
        field_name: pd.to_numeric(feature_df[field_name], errors="coerce").to_numpy(dtype=np.float64)
        for field_name in available_fields
    }
    slope_arrays: Dict[str, np.ndarray] = {}
    for field_name in available_fields:
        d1_col = f"d1_{field_name}"
        if d1_col in feature_df.columns:
            slope_arrays[field_name] = pd.to_numeric(feature_df[d1_col], errors="coerce").to_numpy(dtype=np.float64)

    mm = pd.to_numeric(feature_df.get("mean_motion", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    regime_hint = "leo_meo"
    if np.isfinite(mm).any() and float(np.nanmedian(mm)) < 2.0:
        regime_hint = "geo_long_period_sdp4_like"

    diffs_by_field: Dict[str, List[float]] = {f: [] for f in available_fields}
    raw_residuals: List[Dict[str, float]] = []

    try:
        sgp4_max_rows = int(cfg.propagate_compare_sgp4_max_rows)
    except Exception:
        sgp4_max_rows = 600
    use_sgp4 = bool(sgp4_max_rows <= 0 or n <= sgp4_max_rows)

    for i in range(1, n):
        dt_seconds = elapsed[i] - elapsed[i - 1]
        if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raw_residuals.append({})
            continue

        if use_sgp4:
            pred, used = _predict_with_sgp4_like(rows_cache[i - 1], times_pd[i], cfg.propagate_state_model)
        else:
            pred, used = {}, "capped_to_simplified"

        if used in {"unavailable", "capped_to_simplified"}:
            pred = _predict_with_simplified_perturbation(
                feature_df,
                i,
                dt_seconds,
                base_fields=field_arrays,
                slope_fields=slope_arrays,
            )
            used = "simplified" if use_sgp4 else "simplified_capped"
        model_used[i] = used

        local: Dict[str, float] = {}
        for field_name in available_fields:
            if field_name not in pred:
                continue
            obs_v = float(field_arrays[field_name][i])
            pred_v = float(pred[field_name])
            if np.isfinite(obs_v) and np.isfinite(pred_v):
                diff = obs_v - pred_v
                local[field_name] = diff
                diffs_by_field[field_name].append(diff)
        raw_residuals.append(local)

    scales = {}
    sigma_meta = {}
    for field_name in available_fields:
        sigma, meta = _iterative_sigma(
            np.asarray(diffs_by_field[field_name], dtype=np.float64),
            iterations=cfg.robust_sigma_iterations,
            clip_sigma=cfg.robust_sigma_clip,
        )
        scales[field_name] = sigma if np.isfinite(sigma) and sigma > 1e-12 else 1.0
        sigma_meta[field_name] = meta

    field_z: Dict[str, np.ndarray] = {field_name: np.zeros(n, dtype=np.float64) for field_name in available_fields}
    field_residual: Dict[str, np.ndarray] = {field_name: np.full(n, np.nan, dtype=np.float64) for field_name in available_fields}

    for i in range(1, n):
        local = raw_residuals[i - 1]
        if not local:
            continue
        zvals = []
        for field_name, residual in local.items():
            zval = abs(float(residual)) / scales.get(field_name, 1.0)
            zvals.append(zval)
            field_z[field_name][i] = zval
            field_residual[field_name][i] = float(residual)
        if zvals:
            residual_score[i] = float(np.mean(zvals))

    evidence = np.nan_to_num(residual_score, nan=0.0)
    prom = _rolling_prominence_proxy(evidence, window=cfg.prominence_window_points)

    if cfg.prominence_min_value is not None:
        prom_thr = float(cfg.prominence_min_value)
    else:
        prom_sigma, _ = _iterative_sigma(prom, iterations=2, clip_sigma=4.0)
        prom_thr = float(np.nanmedian(prom) + cfg.prominence_accept_threshold * prom_sigma)

    amp_mask = evidence >= float(cfg.propagate_compare_sigma_threshold)
    if cfg.use_peak_prominence:
        peak_mask = _symmetric_peak_mask(evidence, prom, max(0.0, prom_thr))
        mask = amp_mask & peak_mask
    else:
        mask = amp_mask

    proposals = []
    for start_idx, end_idx in _mask_to_segments(mask, max_gap_points=cfg.max_internal_gap_points):
        proposals.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "support_points": int(end_idx - start_idx + 1),
                "peak_evidence": float(np.nanmax(evidence[start_idx : end_idx + 1])),
                "integrated_evidence": float(np.nanmean(evidence[start_idx : end_idx + 1])),
                "prominence": float(np.nanmax(prom[start_idx : end_idx + 1])),
            }
        )

    comps: Dict[str, np.ndarray] = {
        "residual_score": evidence,
        "model_used_code": (model_used == "sgp4").astype(float),
        "prominence_proxy": prom,
    }
    for field_name in available_fields:
        comps[f"{field_name}_residual"] = field_residual[field_name]
        comps[f"{field_name}_z"] = field_z[field_name]

    return DetectorOutput(
        mask=mask,
        evidence_score=evidence,
        evidence_components=comps,
        detector_metadata={
            "detector_family": "simplified_propagate_compare",
            "model_hint": regime_hint,
            "state_model": str(cfg.propagate_state_model),
            "use_sgp4": bool(use_sgp4),
            "sgp4_row_cap": int(sgp4_max_rows),
            "sigma_by_channel": sigma_meta,
            "prominence_threshold": float(max(0.0, prom_thr)),
            "threshold_sigma": float(cfg.propagate_compare_sigma_threshold),
        },
        segment_proposals=proposals,
    )


def _detector_anomaly_guard(feature_df: pd.DataFrame, cfg: ManeuverDetectionConfig) -> DetectorOutput:
    n = len(feature_df)
    if n == 0:
        return DetectorOutput(mask=np.array([], dtype=bool), evidence_score=np.array([], dtype=np.float64), evidence_components={})

    sma = pd.to_numeric(feature_df.get("sma", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    inc = pd.to_numeric(feature_df.get("inc", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    mm = pd.to_numeric(feature_df.get("mean_motion", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    raan = pd.to_numeric(feature_df.get("raan", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    bstar = pd.to_numeric(feature_df.get("bstar_effective", np.nan), errors="coerce").to_numpy(dtype=np.float64)

    d_sma = np.r_[np.nan, np.diff(sma)]
    d_inc = np.r_[np.nan, np.diff(inc)]
    d_mm = np.r_[np.nan, np.diff(mm)]
    d_raan = np.r_[np.nan, np.diff(raan)]
    d_bstar = np.r_[np.nan, np.diff(bstar)]

    scales = np.array([30.0, 0.25, 0.05, 2.0, 6e-4], dtype=np.float64)
    stack = np.vstack([np.abs(d_sma), np.abs(d_inc), np.abs(d_mm), np.abs(d_raan), np.abs(d_bstar)])
    ratio = np.divide(
        stack,
        scales[:, None],
        out=np.full_like(stack, np.nan, dtype=np.float64),
        where=np.isfinite(stack) & np.isfinite(scales[:, None]) & (scales[:, None] > 0.0),
    )
    ratio_safe = np.where(np.isfinite(ratio), ratio, -np.inf)
    score = np.max(ratio_safe, axis=0)
    score = np.where(np.isfinite(score), score, 0.0)

    high = score > float(cfg.anomaly_sigma_threshold)
    neigh = np.r_[False, high[:-1]] | np.r_[high[1:], False]
    isolation = high & (~neigh)
    mask = isolation

    proposals = []
    for start_idx, end_idx in _mask_to_segments(mask, max_gap_points=0):
        proposals.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "support_points": int(end_idx - start_idx + 1),
                "peak_evidence": float(np.nanmax(score[start_idx : end_idx + 1])),
                "integrated_evidence": float(np.nanmean(score[start_idx : end_idx + 1])),
            }
        )

    return DetectorOutput(
        mask=mask,
        evidence_score=score,
        evidence_components={
            "anomaly_score": score,
            "delta_sma": d_sma,
            "delta_inc": d_inc,
            "delta_mean_motion": d_mm,
            "delta_raan": d_raan,
            "delta_bstar": d_bstar,
        },
        detector_metadata={
            "detector_family": "anomaly_guard",
            "anomaly_sigma_threshold": float(cfg.anomaly_sigma_threshold),
        },
        segment_proposals=proposals,
    )


def _resolve_detector_family_selection(cfg: ManeuverDetectionConfig) -> List[str]:
    mode = str(cfg.detector_family_mode or "all").strip().lower()
    if mode == "kelecy_only":
        families = ["kelecy_filtered_difference", "anomaly_guard"]
    elif mode == "propagate_only":
        families = ["simplified_propagate_compare", "anomaly_guard"]
    elif mode == "segment_only":
        families = ["smoothed_change_segment", "anomaly_guard"]
    elif mode == "custom":
        families = [str(v).strip() for v in cfg.custom_detector_families if str(v).strip()]
        if "anomaly_guard" not in families:
            families.append("anomaly_guard")
    else:
        families = [
            "local_residual",
            "kelecy_filtered_difference",
            "smoothed_change_segment",
            "anomaly_guard",
        ]
        if cfg.enable_propagate_compare:
            families.append("simplified_propagate_compare")

    out = []
    for fam in families:
        if fam in DETECTOR_FAMILIES and fam not in out:
            out.append(fam)
    return out


def run_maneuver_detectors(feature_df: pd.DataFrame, cfg: ManeuverDetectionConfig, cadence_context: Dict[str, Any]) -> Dict[str, DetectorOutput]:
    registry = {
        "local_residual": lambda: _method1_local_residual(feature_df, cfg, cadence_context),
        "kelecy_filtered_difference": lambda: _detector_kelecy_filtered_difference(feature_df, cfg, cadence_context),
        "smoothed_change_segment": lambda: _method2_smoothed_change(feature_df, cfg, cadence_context),
        "simplified_propagate_compare": lambda: _method3_propagate_compare(feature_df, cfg),
        "anomaly_guard": lambda: _detector_anomaly_guard(feature_df, cfg),
    }

    selected = _resolve_detector_family_selection(cfg)
    outputs: Dict[str, DetectorOutput] = {}
    for family in selected:
        if family == "simplified_propagate_compare" and (not cfg.enable_propagate_compare) and str(cfg.detector_family_mode).lower() != "propagate_only":
            continue
        outputs[family] = registry[family]()
    return outputs


def fuse_detector_evidence(detectors: Dict[str, DetectorOutput], cfg: ManeuverDetectionConfig, n_rows: int) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    if not detectors:
        return np.zeros(n_rows, dtype=bool), np.zeros(n_rows, dtype=np.float64), {}

    evidence_components = {name: np.nan_to_num(out.evidence_score, nan=0.0) for name, out in detectors.items()}
    candidate_names = [name for name in detectors.keys() if name != "anomaly_guard"]
    if not candidate_names:
        candidate_names = list(detectors.keys())

    use_torch = bool(cfg.enable_torch_evidence_ops and _HAS_TORCH and len(candidate_names) > 0)
    if use_torch:
        try:
            stack = np.vstack(
                [
                    np.nan_to_num(np.asarray(evidence_components[name], dtype=np.float64), nan=0.0)
                    for name in candidate_names
                ]
            )

            requested_device = str(cfg.torch_device or "cpu").strip().lower()
            if requested_device.startswith("cuda") and (not torch.cuda.is_available()):
                requested_device = "cpu"

            t_stack = torch.from_numpy(stack).to(requested_device)
            if str(cfg.evidence_fusion_mode).lower() == "weighted":
                weights = np.asarray([float(cfg.detector_weights.get(name, 1.0)) for name in candidate_names], dtype=np.float64)
                t_weights = torch.from_numpy(weights).to(requested_device).view(-1, 1)
                denom = torch.clamp(torch.sum(t_weights), min=1e-12)
                t_combined = torch.sum(t_stack * t_weights, dim=0) / denom
                combined_score = t_combined.detach().cpu().numpy().astype(np.float64)
                combined_mask = combined_score >= float(cfg.evidence_score_threshold)
            else:
                t_union = torch.any(t_stack > 0.0, dim=0)
                t_combined = torch.amax(t_stack, dim=0)
                combined_score = t_combined.detach().cpu().numpy().astype(np.float64)
                combined_mask = t_union.detach().cpu().numpy().astype(bool)

            return combined_mask, combined_score, evidence_components
        except Exception:
            # Fall back to numpy path if torch cannot execute on requested device.
            pass

    if str(cfg.evidence_fusion_mode).lower() == "weighted":
        weighted = np.zeros(n_rows, dtype=np.float64)
        total_w = 0.0
        for name in candidate_names:
            vec = evidence_components.get(name, np.zeros(n_rows, dtype=np.float64))
            w = float(cfg.detector_weights.get(name, 1.0))
            weighted += w * vec
            total_w += w
        if total_w > 0.0:
            weighted /= total_w
        combined_score = weighted
        combined_mask = combined_score >= float(cfg.evidence_score_threshold)
    else:
        union = np.zeros(n_rows, dtype=bool)
        max_score = np.zeros(n_rows, dtype=np.float64)
        for name in candidate_names:
            out = detectors[name]
            union |= out.mask
            max_score = np.maximum(max_score, np.nan_to_num(out.evidence_score, nan=0.0))
        combined_mask = union
        combined_score = max_score

    return combined_mask, combined_score, evidence_components


def _multivariate_anomaly_score(jump_vector: np.ndarray) -> float:
    if jump_vector.size < 5:
        return 0.0
    scales = np.array([30.0, 0.25, 0.05, 2.0, 6e-4], dtype=np.float64)
    z = np.abs(np.asarray(jump_vector, dtype=np.float64)) / scales
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0
    return float(np.nanmax(z))


def _segment_pre_post_means(values: np.ndarray, start_idx: int, end_idx: int, window: int = 4) -> tuple[float, float]:
    n = values.size
    pre_lo = max(0, start_idx - int(window))
    pre_hi = start_idx
    post_lo = end_idx + 1
    post_hi = min(n, end_idx + 1 + int(window))
    pre = np.nanmean(values[pre_lo:pre_hi]) if pre_hi > pre_lo else np.nan
    post = np.nanmean(values[post_lo:post_hi]) if post_hi > post_lo else np.nan
    return float(pre), float(post)


def _segment_event_type(
    segment: Dict[str, Any],
    feature_df: pd.DataFrame,
    cfg: ManeuverDetectionConfig,
) -> tuple[str, Dict[str, float]]:
    start_idx = int(segment["start_idx"])
    end_idx = int(segment["end_idx"])

    sma = pd.to_numeric(feature_df.get("sma", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    inc = pd.to_numeric(feature_df.get("inc", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    bstar = pd.to_numeric(feature_df.get("bstar_effective", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    d1_sma = pd.to_numeric(feature_df.get("d1_sma", np.nan), errors="coerce").to_numpy(dtype=np.float64)

    pre_sma, post_sma = _segment_pre_post_means(sma, start_idx, end_idx, window=5)
    pre_inc, post_inc = _segment_pre_post_means(inc, start_idx, end_idx, window=5)
    delta_sma = post_sma - pre_sma if np.isfinite(pre_sma) and np.isfinite(post_sma) else 0.0
    delta_inc = post_inc - pre_inc if np.isfinite(pre_inc) and np.isfinite(post_inc) else 0.0

    bstar_level = float(np.nanmean(np.abs(bstar[start_idx : end_idx + 1]))) if bstar.size else 0.0
    sustained_slope = float(np.nanmean(pd.to_numeric(feature_df.get("sustained_slope_score", np.nan), errors="coerce").to_numpy(dtype=np.float64)[start_idx : end_idx + 1]))
    stationkeeping_density = float(np.nanmean(pd.to_numeric(feature_df.get("event_density_suppression_term", np.nan), errors="coerce").to_numpy(dtype=np.float64)[start_idx : end_idx + 1]))
    seg_slope_km_day = 0.0
    if d1_sma.size > end_idx:
        seg_slope = d1_sma[start_idx : end_idx + 1]
        seg_finite = np.isfinite(seg_slope)
        if np.any(seg_finite):
            seg_slope_km_day = float(np.mean(seg_slope[seg_finite]) * 86400.0)

    energy_signal = float(segment.get("energy_signal", 0.0))
    plane_signal = float(segment.get("plane_signal", 0.0))
    peak = float(segment.get("peak_evidence", 0.0))
    anomaly = float(segment.get("anomaly_score", 0.0))
    consistency = float(segment.get("channel_consistency_score", 0.0))

    event_type = "unknown_event"
    if anomaly >= cfg.anomaly_sigma_threshold and consistency < 0.34:
        event_type = "possible_anomaly"
    elif (delta_sma > 4.0) and (abs(plane_signal) > 0.8 or abs(delta_inc) > 0.04) and peak > 1.0:
        event_type = "combined_raise_and_plane_change"
    elif (delta_sma > 4.0) and (energy_signal > 0.6) and peak > 0.9:
        event_type = "semi_major_axis_raise"
    elif abs(delta_inc) > 0.035 and abs(plane_signal) > 0.7 and abs(delta_sma) < 6.0:
        event_type = "inclination_change"
    elif (delta_sma < -4.0) and (energy_signal < -0.5):
        if bstar_level > 3e-4 and sustained_slope > 0.45 and stationkeeping_density < 0.35:
            event_type = "possible_disposal_start"
        else:
            event_type = "semi_major_axis_lower"
    elif ((delta_sma > 3.0) or (seg_slope_km_day > 0.25)) and peak > 0.5:
        event_type = "semi_major_axis_raise"
    elif ((delta_sma < -3.0) or (seg_slope_km_day < -0.25)) and peak > 0.5:
        event_type = "possible_disposal_start" if bstar_level > 3e-4 else "semi_major_axis_lower"
    elif abs(delta_inc) > 0.05 and peak > 0.5:
        event_type = "inclination_change"
    elif abs(delta_sma) <= 3.5 and abs(delta_inc) <= 0.03 and stationkeeping_density >= 0.4 and peak >= 0.8:
        event_type = "possible_stationkeeping"
    elif anomaly >= cfg.anomaly_sigma_threshold:
        event_type = "possible_anomaly"

    return event_type, {
        "delta_sma": float(delta_sma),
        "delta_inc": float(delta_inc),
        "bstar_level": float(np.nan_to_num(bstar_level, nan=0.0)),
        "pre_sma": float(pre_sma),
        "post_sma": float(post_sma),
        "pre_inc": float(pre_inc),
        "post_inc": float(post_inc),
        "segment_slope_km_day": float(seg_slope_km_day),
    }


def _estimate_event_interval(
    segment: Dict[str, Any],
    feature_df: pd.DataFrame,
    cfg: ManeuverDetectionConfig,
    cadence_context: Dict[str, Any],
    detectors: Dict[str, DetectorOutput],
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]:
    ts = pd.to_datetime(feature_df["timestamp"], errors="coerce").to_numpy()
    start_idx = int(segment["start_idx"])
    end_idx = int(segment["end_idx"])
    start_t = pd.Timestamp(ts[start_idx])
    end_t = pd.Timestamp(ts[end_idx])

    cadence = float(np.nan_to_num(cadence_context.get("median_cadence_seconds"), nan=6 * 3600.0))
    cadence = max(900.0, cadence)
    lag = float(max(0.0, cfg.lag_seconds))

    mismatch_onset = start_t
    prop = detectors.get("simplified_propagate_compare")
    if prop is not None:
        score = np.asarray(prop.evidence_score, dtype=np.float64)
        loc = np.flatnonzero(score[start_idx : end_idx + 1] >= float(cfg.propagate_compare_sigma_threshold))
        if loc.size > 0:
            mismatch_onset = pd.Timestamp(ts[start_idx + int(loc[0])])

    det_mid = start_t + (end_t - start_t) / 2
    estimated_event_time = min(det_mid, mismatch_onset) - pd.to_timedelta(lag, unit="s")
    event_time_lower = start_t - pd.to_timedelta(lag + 0.5 * cadence, unit="s")
    event_time_upper = end_t - pd.to_timedelta(max(0.0, 0.25 * lag), unit="s") + pd.to_timedelta(0.5 * cadence, unit="s")

    if event_time_upper < event_time_lower:
        event_time_lower = estimated_event_time - pd.to_timedelta(cadence, unit="s")
        event_time_upper = estimated_event_time + pd.to_timedelta(cadence, unit="s")

    half_uncertainty = 0.5 * float((event_time_upper - event_time_lower).total_seconds())
    return estimated_event_time, event_time_lower, event_time_upper, max(0.0, half_uncertainty)


def _build_candidate_segments(
    feature_df: pd.DataFrame,
    detectors: Dict[str, DetectorOutput],
    cfg: ManeuverDetectionConfig,
    conditioning_audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    n = len(feature_df)
    if n == 0 or not detectors:
        return []

    non_anomaly = [name for name in detectors.keys() if name != "anomaly_guard"]
    if not non_anomaly:
        non_anomaly = list(detectors.keys())

    support = np.zeros(n, dtype=bool)
    for name in non_anomaly:
        support |= np.asarray(detectors[name].mask, dtype=bool)

    # Include detector-proposed segments even when support masks are sparse.
    for name in non_anomaly:
        for proposal in detectors[name].segment_proposals:
            s = int(proposal.get("start_idx", 0))
            e = int(proposal.get("end_idx", -1))
            if 0 <= s <= e < n:
                support[s : e + 1] = True

    segments_idx = _mask_to_segments(support, max_gap_points=cfg.max_internal_gap_points)

    # Backward-compatible fallback: if strict support masks produce no segments,
    # recover high-evidence windows directly from detector score fields.
    if not segments_idx:
        fallback_support = np.zeros(n, dtype=bool)
        for name in non_anomaly:
            score = np.asarray(detectors[name].evidence_score, dtype=np.float64)
            finite = score[np.isfinite(score)]
            if finite.size == 0:
                continue
            dynamic_thr = max(
                0.6,
                float(np.nanmedian(finite) + 1.2 * _robust_sigma(finite)),
                float(np.nanpercentile(finite, 85.0)),
            )
            fallback_support |= np.where(np.isfinite(score), score >= dynamic_thr, False)

        if fallback_support.any():
            fallback_support = _bridge_internal_gaps(
                fallback_support,
                max_gap_points=max(1, int(cfg.max_internal_gap_points)),
            )
            segments_idx = _mask_to_segments(
                fallback_support,
                max_gap_points=max(1, int(cfg.max_internal_gap_points)),
            )

        if not segments_idx:
            score_stack = [
                np.where(np.isfinite(np.asarray(detectors[name].evidence_score, dtype=np.float64)), np.asarray(detectors[name].evidence_score, dtype=np.float64), -np.inf)
                for name in non_anomaly
            ]
            if score_stack:
                combined = np.nanmax(np.vstack(score_stack), axis=0)
                peak = float(np.nanmax(combined))
                if np.isfinite(peak) and peak >= 0.7:
                    peak_idx = int(np.nanargmax(combined))
                    left = max(0, peak_idx - 1)
                    right = min(n - 1, peak_idx + 1)
                    segments_idx = [(left, right)]

    if not segments_idx:
        return []

    times = pd.to_datetime(feature_df["timestamp"], errors="coerce").to_numpy()
    cadence_cv = float(np.nan_to_num(conditioning_audit.get("cadence_context", {}).get("cadence_cv"), nan=0.0))
    cadence_penalty = 1.0 / (1.0 + max(0.0, cadence_cv))

    out_segments: List[Dict[str, Any]] = []
    for start_idx, end_idx in segments_idx:
        support_points = int(end_idx - start_idx + 1)
        duration_seconds = _segment_duration_seconds(times, start_idx, end_idx)
        if support_points < int(cfg.min_segment_support_points):
            continue

        family_support = []
        detector_evidence: Dict[str, float] = {}
        peak_evidence = 0.0
        integrated_evidence = 0.0
        prominence = 0.0

        for name in non_anomaly:
            out = detectors[name]
            local_mask = np.asarray(out.mask[start_idx : end_idx + 1], dtype=bool)
            if local_mask.any():
                family_support.append(name)

            local_score = np.asarray(out.evidence_score[start_idx : end_idx + 1], dtype=np.float64)
            if local_score.size:
                family_peak = float(np.nanmax(local_score))
                family_int = float(np.nanmean(local_score))
            else:
                family_peak = 0.0
                family_int = 0.0

            detector_evidence[name] = family_peak
            peak_evidence = max(peak_evidence, family_peak)
            integrated_evidence += family_int

            prom_vec = out.evidence_components.get("prominence_proxy")
            if isinstance(prom_vec, np.ndarray) and prom_vec.size > end_idx:
                prominence = max(prominence, float(np.nanmax(prom_vec[start_idx : end_idx + 1])))

        integrated_evidence /= max(1, len(non_anomaly))
        consistency = float(len(family_support) / max(1, len(non_anomaly)))

        # Keep strict duration filtering unless detector evidence is strong.
        if duration_seconds < float(cfg.min_segment_duration_seconds) and peak_evidence < 0.7:
            continue

        kelecy = detectors.get("kelecy_filtered_difference")
        energy_signal = 0.0
        plane_signal = 0.0
        if kelecy is not None:
            evec = kelecy.evidence_components.get("energy_diff")
            pvec = kelecy.evidence_components.get("plane_diff")
            if isinstance(evec, np.ndarray) and evec.size > end_idx:
                e_seg = np.asarray(evec[start_idx : end_idx + 1], dtype=np.float64)
                e_fin = np.isfinite(e_seg)
                energy_signal = float(np.mean(e_seg[e_fin])) if np.any(e_fin) else 0.0
            if isinstance(pvec, np.ndarray) and pvec.size > end_idx:
                p_seg = np.asarray(pvec[start_idx : end_idx + 1], dtype=np.float64)
                p_fin = np.isfinite(p_seg)
                plane_signal = float(np.mean(p_seg[p_fin])) if np.any(p_fin) else 0.0

        anomaly_score = 0.0
        anomaly = detectors.get("anomaly_guard")
        if anomaly is not None and anomaly.evidence_score.size > end_idx:
            anomaly_score = float(np.nanmax(anomaly.evidence_score[start_idx : end_idx + 1]))

        score = (
            0.42 * peak_evidence
            + 0.28 * integrated_evidence
            + 0.20 * prominence
            + 0.10 * consistency
        ) * cadence_penalty

        if detector_evidence.get("simplified_propagate_compare", 0.0) > 0.0:
            score += 0.10 * min(1.0, detector_evidence.get("simplified_propagate_compare", 0.0))

        prominence_ok = True
        if cfg.use_peak_prominence:
            prominence_ok = prominence >= float(cfg.prominence_accept_threshold)

        accepted = (score >= float(cfg.event_score_accept_threshold)) and prominence_ok
        if not accepted and peak_evidence >= 0.8 and support_points >= int(cfg.min_segment_support_points):
            accepted = True

        out_segments.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "support_points": support_points,
                "duration_seconds": float(duration_seconds),
                "peak_evidence": float(peak_evidence),
                "integrated_evidence": float(integrated_evidence),
                "prominence": float(prominence),
                "channel_consistency_score": float(consistency),
                "detector_family_support": family_support,
                "detector_family_support_count": int(len(family_support)),
                "detector_evidence": detector_evidence,
                "energy_signal": float(energy_signal),
                "plane_signal": float(plane_signal),
                "anomaly_score": float(anomaly_score),
                "event_score_raw": float(score),
                "accepted": bool(accepted),
            }
        )

    return out_segments


def _debounce_segment_events(events: List[Dict[str, Any]], cfg: ManeuverDetectionConfig) -> List[Dict[str, Any]]:
    if not events:
        return []
    ordered = sorted(events, key=lambda e: pd.Timestamp(e["estimated_event_time"]))
    merged: List[Dict[str, Any]] = [ordered[0]]

    for ev in ordered[1:]:
        prev = merged[-1]
        spacing = float(cfg.object_level_debounce_seconds)
        if ev.get("event_type") == "possible_stationkeeping" or prev.get("event_type") == "possible_stationkeeping":
            spacing = max(spacing, float(cfg.stationkeeping_episode_spacing_seconds))

        prev_t = pd.Timestamp(prev["estimated_event_time"])
        ev_t = pd.Timestamp(ev["estimated_event_time"])
        dt = float((ev_t - prev_t).total_seconds())

        overlap = (
            pd.Timestamp(ev["segment_start_time"]) <= pd.Timestamp(prev["segment_end_time"]) + pd.to_timedelta(spacing, unit="s")
        )
        spacing_match = dt <= spacing

        merge_now = overlap if str(cfg.event_merge_mode).lower() == "segment_overlap" else spacing_match
        if not merge_now:
            merged.append(ev)
            continue

        keep_new = float(ev.get("event_score", 0.0)) > float(prev.get("event_score", 0.0))
        if keep_new:
            base = ev.copy()
            other = prev
        else:
            base = prev.copy()
            other = ev

        base["segment_start_time"] = min(pd.Timestamp(base["segment_start_time"]), pd.Timestamp(other["segment_start_time"]))
        base["segment_end_time"] = max(pd.Timestamp(base["segment_end_time"]), pd.Timestamp(other["segment_end_time"]))
        base["event_time_lower"] = min(pd.Timestamp(base["event_time_lower"]), pd.Timestamp(other["event_time_lower"]))
        base["event_time_upper"] = max(pd.Timestamp(base["event_time_upper"]), pd.Timestamp(other["event_time_upper"]))
        base["timing_uncertainty_seconds"] = 0.5 * float((pd.Timestamp(base["event_time_upper"]) - pd.Timestamp(base["event_time_lower"])).total_seconds())
        base["detector_family_support"] = sorted(set(list(base.get("detector_family_support", [])) + list(other.get("detector_family_support", []))))
        base["detector_family_support_count"] = int(len(base["detector_family_support"]))
        base["peak_evidence"] = float(max(base.get("peak_evidence", 0.0), other.get("peak_evidence", 0.0)))
        base["integrated_evidence"] = float(max(base.get("integrated_evidence", 0.0), other.get("integrated_evidence", 0.0)))
        base["prominence"] = float(max(base.get("prominence", 0.0), other.get("prominence", 0.0)))
        merged[-1] = base

    return merged


def detect_maneuvers_for_satellite(df_sat: pd.DataFrame, config: Optional[ManeuverDetectionConfig] = None) -> pd.DataFrame:
    cfg = config or ManeuverDetectionConfig()
    sat, audit = prepare_satellite_timeseries(df_sat, config=cfg.model, enrich_phase=False, enrich_altitude=True, return_audit=True)

    cols = [
        "object_id",
        "sat_id",
        "norad_cat_id",
        "detection_time",
        "estimated_event_time",
        "event_time_lower",
        "event_time_upper",
        "timing_uncertainty_seconds",
        "event_type",
        "event_score",
        "event_score_raw",
        "detector_evidence",
        "detector_family_support",
        "detector_family_support_count",
        "segment_start_time",
        "segment_end_time",
        "peak_evidence",
        "integrated_evidence",
        "prominence",
        "channel_consistency_score",
        "quality_flag",
        "cadence_context",
        "duplicate_context",
        "contributing_features",
        "pre_trend_sma",
        "post_trend_sma",
        "pre_trend_inc",
        "post_trend_inc",
        "detector_method",
        "anomaly_score",
        "event_layer",
        "is_high_confidence",
        "detector_calibration",
    ]

    if len(sat) < max(cfg.min_records, cfg.model.min_records):
        return pd.DataFrame(columns=cols)

    features = build_maneuver_feature_frame(sat, use_numba=cfg.enable_numba_kernels)
    features["guard_short_history"] = bool(audit.get("history_too_short", False))
    cadence_ctx = dict(audit.get("cadence_context", {}))
    features["guard_irregular_cadence"] = bool(np.nan_to_num(cadence_ctx.get("cadence_cv"), nan=0.0) > 0.8)
    dup_ctx = dict(audit.get("duplicate_context", {}))
    features["guard_duplicate_epochs"] = bool(int(dup_ctx.get("duplicate_rows_detected", 0)) > 0)

    try:
        detectors = run_maneuver_detectors(features, cfg, cadence_context=cadence_ctx)
    except Exception:
        if cfg.enable_propagate_compare:
            # Fallback keeps the run alive when propagation-compare dependencies fail.
            cfg_fallback = deepcopy(cfg)
            cfg_fallback.enable_propagate_compare = False
            detectors = run_maneuver_detectors(features, cfg_fallback, cadence_context=cadence_ctx)
        else:
            raise
    segments = _build_candidate_segments(features, detectors, cfg, audit)
    if not segments:
        return pd.DataFrame(columns=cols)

    events: List[Dict[str, Any]] = []
    for seg in segments:
        if not bool(seg.get("accepted", False)):
            continue

        event_type, trend_info = _segment_event_type(seg, features, cfg)

        # Anomaly guard suppresses maneuver typing unless corroborated.
        if (
            cfg.anomaly_screening_enabled
            and float(seg.get("anomaly_score", 0.0)) >= float(cfg.anomaly_sigma_threshold)
            and int(seg.get("detector_family_support_count", 0)) <= 1
        ):
            event_type = "possible_anomaly"

        est_t, t_lo, t_hi, half_unc = _estimate_event_interval(seg, features, cfg, cadence_ctx, detectors)
        detection_time = pd.Timestamp(features.iloc[int(seg["start_idx"])]["timestamp"])

        quality_flag = "accepted"
        is_high_conf = bool(float(seg.get("event_score_raw", 0.0)) >= float(cfg.high_confidence_threshold))
        if is_high_conf:
            quality_flag = "accepted_high_confidence"
        if event_type == "possible_anomaly":
            quality_flag = "anomaly_screened"

        events.append(
            {
                "detection_time": detection_time,
                "estimated_event_time": est_t,
                "event_time_lower": t_lo,
                "event_time_upper": t_hi,
                "timing_uncertainty_seconds": float(half_unc),
                "event_type": event_type,
                "event_score": float(min(1.0, max(0.0, float(seg.get("event_score_raw", 0.0)) / 2.0))),
                "event_score_raw": float(seg.get("event_score_raw", 0.0)),
                "detector_evidence": dict(seg.get("detector_evidence", {})),
                "detector_family_support": list(seg.get("detector_family_support", [])),
                "detector_family_support_count": int(seg.get("detector_family_support_count", 0)),
                "segment_start_time": pd.Timestamp(features.iloc[int(seg["start_idx"])]["timestamp"]),
                "segment_end_time": pd.Timestamp(features.iloc[int(seg["end_idx"])]["timestamp"]),
                "peak_evidence": float(seg.get("peak_evidence", 0.0)),
                "integrated_evidence": float(seg.get("integrated_evidence", 0.0)),
                "prominence": float(seg.get("prominence", 0.0)),
                "channel_consistency_score": float(seg.get("channel_consistency_score", 0.0)),
                "quality_flag": quality_flag,
                "cadence_context": cadence_ctx,
                "duplicate_context": dup_ctx,
                "contributing_features": "sma,inc,bstar,mean_motion,ecc,raan,energy_proxy,h_direction,shell_residuals,segment_morphology",
                "pre_trend_sma": trend_info["pre_sma"],
                "post_trend_sma": trend_info["post_sma"],
                "pre_trend_inc": trend_info["pre_inc"],
                "post_trend_inc": trend_info["post_inc"],
                "detector_method": "segment_fusion",
                "anomaly_score": float(seg.get("anomaly_score", 0.0)),
                "event_layer": "accepted",
                "is_high_confidence": bool(is_high_conf),
                "detector_calibration": {k: v.detector_metadata for k, v in detectors.items()},
            }
        )

    merged = _debounce_segment_events(events, cfg)
    if not merged:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(merged)
    object_id, sat_id, norad = _resolve_object_fields(sat)
    out["sat_id"] = sat_id
    out["norad_cat_id"] = norad
    out["object_id"] = object_id

    valid_event = out["event_type"].isin(EVENT_TYPES)
    out.loc[~valid_event, "event_type"] = "unknown_event"
    return out[cols].sort_values("detection_time", kind="mergesort").reset_index(drop=True)


def detect_maneuvers_all(df: pd.DataFrame, object_col: str = "sat_id", config: Optional[ManeuverDetectionConfig] = None) -> pd.DataFrame:
    cfg = config or ManeuverDetectionConfig()
    data = ensure_panel_sorted(df, object_col=object_col, time_col="timestamp")

    try:
        group_col = select_object_id_column(data, preferred=cfg.model.preferred_object_cols)
    except KeyError:
        group_col = object_col

    grouped = data.groupby(group_col, sort=True)

    events = []
    min_required = int(max(cfg.min_records, cfg.model.min_records))
    total_objects = int(grouped.ngroups)
    t_loop = pd.Timestamp.utcnow()

    try:
        progress_every = max(1, int(cfg.progress_every_objects))
    except Exception:
        progress_every = 100
    show_progress = bool(cfg.print_progress and total_objects >= progress_every)

    failed_objects: List[Dict[str, Any]] = []
    for idx, (obj_id, grp) in enumerate(grouped, start=1):
        if len(grp) < min_required:
            continue
        if "timestamp" in grp.columns:
            valid_ts = pd.to_datetime(grp["timestamp"], errors="coerce").notna().sum()
            if int(valid_ts) < min_required:
                continue

        try:
            ev = detect_maneuvers_for_satellite(grp, config=cfg)
        except Exception as exc:
            failed_objects.append(
                {
                    "object_id": str(obj_id),
                    "rows": int(len(grp)),
                    "error": str(exc),
                }
            )
            continue

        if not ev.empty:
            events.append(ev)

        if show_progress and (idx % progress_every == 0 or idx == total_objects):
            elapsed = (pd.Timestamp.utcnow() - t_loop).total_seconds()
            rate = float(idx / max(1e-9, elapsed))
            eta = float((total_objects - idx) / max(1e-9, rate))
            print(
                f"[maneuver_detection] processed {idx}/{total_objects} objects "
                f"({rate:.1f} obj/s, eta {eta:.1f}s)"
            )

    if failed_objects:
        print(
            f"[maneuver_detection] Skipped {len(failed_objects)} object(s) due to per-object errors; "
            "continuing with remaining objects."
        )

    if not events:
        return pd.DataFrame(
            columns=[
                "object_id",
                "sat_id",
                "norad_cat_id",
                "detection_time",
                "estimated_event_time",
                "event_time_lower",
                "event_time_upper",
                "timing_uncertainty_seconds",
                "event_type",
                "event_score",
                "event_score_raw",
                "detector_evidence",
                "detector_family_support",
                "detector_family_support_count",
                "segment_start_time",
                "segment_end_time",
                "peak_evidence",
                "integrated_evidence",
                "prominence",
                "channel_consistency_score",
                "quality_flag",
                "cadence_context",
                "duplicate_context",
                "contributing_features",
                "pre_trend_sma",
                "post_trend_sma",
                "pre_trend_inc",
                "post_trend_inc",
                "detector_method",
                "anomaly_score",
                "event_layer",
                "is_high_confidence",
                "detector_calibration",
            ]
        )

    return pd.concat(events, ignore_index=True).sort_values(["object_id", "detection_time"], kind="mergesort").reset_index(drop=True)


# ---- Mission-phase inference ----
PHASE_STATES = [
    "unknown",
    "insertion_or_orbit_raise",
    "transition",
    "operational_shell",
    "relocation",
    "disposal_lowering",
    "passive_decay",
    "likely_nonoperational",
]


@dataclass
class PhaseClassificationConfig:
    model: TimeseriesModelConfig = field(default_factory=TimeseriesModelConfig)
    min_records: int = 10
    raise_slope_km_per_day: float = 0.2
    raise_override_slope_km_per_day: float = 0.6
    insertion_entry_threshold: float = 0.65
    raise_event_rate_threshold: float = 0.04
    raise_maneuver_evidence_threshold: float = 0.2
    lower_slope_km_per_day: float = -0.2
    operational_var_km: float = 8.0
    high_bstar_threshold: float = 3e-4
    relocation_event_rate_threshold: float = 0.15
    rolling_window_points: int = 15

    use_persistence_smoothing: bool = True
    transition_penalty: float = 0.65
    jump_penalty_scale: float = 1.2

    phase_model_mode: str = "transition_penalized"  # transition_penalized|semi_markov_rule_model
    operational_support_weight: float = 1.2
    relocation_entry_threshold: float = 0.8
    disposal_entry_threshold: float = 0.9
    passive_decay_entry_threshold: float = 0.85
    minimum_phase_dwell_days: float = 21.0
    maintenance_regularity_window_days: float = 45.0
    maintenance_loss_window_days: float = 120.0
    phase_confidence_export: bool = True
    progress_every_objects: int = 100
    print_progress: bool = True


def _effective_bstar(df_sat: pd.DataFrame) -> np.ndarray:
    if "bstar_effective" in df_sat.columns:
        return pd.to_numeric(df_sat["bstar_effective"], errors="coerce").to_numpy(dtype=np.float64)
    if "bstar" in df_sat.columns:
        return pd.to_numeric(df_sat["bstar"], errors="coerce").to_numpy(dtype=np.float64)
    if "drag_term" in df_sat.columns:
        return pd.to_numeric(df_sat["drag_term"], errors="coerce").to_numpy(dtype=np.float64)
    return np.full(len(df_sat), np.nan, dtype=np.float64)


def _build_event_time_lookup(events_df: Optional[pd.DataFrame]) -> Dict[str, np.ndarray]:
    if events_df is None or events_df.empty or "object_id" not in events_df.columns:
        return {}

    ev = events_df[["object_id", "estimated_event_time"]].copy()
    ev["estimated_event_time"] = pd.to_datetime(ev["estimated_event_time"], errors="coerce")
    ev = ev.dropna(subset=["estimated_event_time"])
    if ev.empty:
        return {}

    ev["object_id"] = ev["object_id"].astype(str)
    lookup: Dict[str, np.ndarray] = {}
    for object_id, grp in ev.groupby("object_id", sort=False):
        ns = grp["estimated_event_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        lookup[str(object_id)] = np.sort(ns)
    return lookup


def _event_rate_around_times(
    times: np.ndarray,
    events_df: Optional[pd.DataFrame],
    object_id: str,
    window_days: float = 7.0,
    event_ns_lookup: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    if times.size == 0:
        return np.array([], dtype=np.float64)

    if event_ns_lookup is not None:
        ev_ns = event_ns_lookup.get(str(object_id), np.array([], dtype=np.int64))
        if ev_ns.size == 0:
            return np.zeros(times.shape[0], dtype=np.float64)
    elif events_df is None or events_df.empty:
        return np.zeros_like(times, dtype=np.float64)
    else:
        ev = events_df[events_df["object_id"].astype(str) == str(object_id)]
        if ev.empty:
            return np.zeros_like(times, dtype=np.float64)

        ev_t = pd.to_datetime(ev["estimated_event_time"], errors="coerce").dropna().to_numpy()
        if ev_t.size == 0:
            return np.zeros_like(times, dtype=np.float64)
        ev_ns = np.sort(ev_t.astype("datetime64[ns]").astype("int64"))

    t_ns = times.astype("datetime64[ns]").astype("int64")
    w_ns = int(window_days * 86400.0 * 1e9)
    left = np.searchsorted(ev_ns, t_ns - w_ns, side="left")
    right = np.searchsorted(ev_ns, t_ns + w_ns, side="right")
    counts = (right - left).astype(np.float64)
    return counts / (2.0 * window_days)


def _event_evidence_around_times(times: np.ndarray, events_df: Optional[pd.DataFrame], object_id: str) -> np.ndarray:
    if events_df is None or events_df.empty:
        return np.zeros(times.shape[0], dtype=np.float64)

    ev = events_df[events_df["object_id"].astype(str) == str(object_id)].copy()
    if ev.empty:
        return np.zeros(times.shape[0], dtype=np.float64)

    ev["estimated_event_time"] = pd.to_datetime(ev["estimated_event_time"], errors="coerce")
    ev = ev.dropna(subset=["estimated_event_time"])
    if ev.empty:
        return np.zeros(times.shape[0], dtype=np.float64)

    ev = ev.sort_values("estimated_event_time", kind="mergesort").reset_index(drop=True)
    ev_t = ev["estimated_event_time"].to_numpy(dtype="datetime64[ns]")
    ev_s = pd.to_numeric(ev.get("event_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    t_ns = times.astype("datetime64[ns]").astype("int64")
    ev_ns = ev_t.astype("int64")
    if ev_ns.size == 0:
        return np.zeros(times.shape[0], dtype=np.float64)

    half_window_ns = int(2.0 * 86400.0 * 1e9)
    left = np.searchsorted(ev_ns, t_ns - half_window_ns, side="left")
    right = np.searchsorted(ev_ns, t_ns + half_window_ns, side="right")

    finite = np.isfinite(ev_s)
    safe = np.where(finite, ev_s, 0.0)
    csum = np.r_[0.0, np.cumsum(safe)]
    ccnt = np.r_[0, np.cumsum(finite.astype(np.int64))]

    win_sum = csum[right] - csum[left]
    win_cnt = ccnt[right] - ccnt[left]
    out = np.divide(win_sum, win_cnt, out=np.zeros(times.shape[0], dtype=np.float64), where=win_cnt > 0)
    return out


def _maintenance_features_around_times(
    times: np.ndarray,
    events_df: Optional[pd.DataFrame],
    object_id: str,
    cfg: PhaseClassificationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = times.size
    if n == 0 or events_df is None or events_df.empty:
        return (
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
        )

    ev = events_df[events_df.get("object_id", "").astype(str) == str(object_id)].copy()
    if ev.empty:
        return (
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
        )

    ev["estimated_event_time"] = pd.to_datetime(ev["estimated_event_time"], errors="coerce")
    ev = ev.dropna(subset=["estimated_event_time"]).sort_values("estimated_event_time", kind="mergesort")
    if ev.empty:
        return (
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
        )

    ev_ns = ev["estimated_event_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    t_ns = times.astype("datetime64[ns]").astype("int64")

    reg_days = max(3.0, float(cfg.maintenance_regularity_window_days))
    loss_days = max(reg_days, float(cfg.maintenance_loss_window_days))
    reg_ns = int(reg_days * 86400.0 * 1e9)
    loss_ns = int(loss_days * 86400.0 * 1e9)

    l_recent = np.searchsorted(ev_ns, t_ns - reg_ns, side="left")
    r_recent = np.searchsorted(ev_ns, t_ns, side="right")
    recent_count = np.maximum(0, r_recent - l_recent)
    rate = recent_count.astype(np.float64) / max(1.0, reg_days)

    l_hist = np.searchsorted(ev_ns, t_ns - 2 * loss_ns, side="left")
    r_hist = np.searchsorted(ev_ns, t_ns - loss_ns, side="right")
    l_now = np.searchsorted(ev_ns, t_ns - loss_ns, side="left")
    r_now = np.searchsorted(ev_ns, t_ns, side="right")

    hist_rate = np.maximum(0.0, (r_hist - l_hist).astype(np.float64) / max(1.0, loss_days))
    now_rate = np.maximum(0.0, (r_now - l_now).astype(np.float64) / max(1.0, loss_days))
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = np.where(hist_rate > 0.0, np.clip((hist_rate - now_rate) / hist_rate, 0.0, 1.0), 0.0)

    regularity = np.zeros(n, dtype=np.float64)
    idx_reg = np.flatnonzero(recent_count >= 3)
    for i in idx_reg:
        lo = int(l_recent[i])
        hi = int(r_recent[i])
        intervals_days = np.diff(ev_ns[lo:hi]).astype(np.float64) / (86400.0 * 1e9)
        med = np.nanmedian(intervals_days) if intervals_days.size else np.nan
        std = np.nanstd(intervals_days) if intervals_days.size else np.nan
        cv = float(std / med) if np.isfinite(med) and med > 1e-9 else np.inf
        regularity[i] = float(1.0 / (1.0 + max(0.0, cv))) if np.isfinite(cv) else 0.0

    return rate, regularity, loss


def _phase_emission_scores(
    slope_km_day: np.ndarray,
    alt_var_km: np.ndarray,
    bstar: np.ndarray,
    event_rate: np.ndarray,
    shell_residual_km: np.ndarray,
    shell_drift_residual: np.ndarray,
    maneuver_evidence: np.ndarray,
    maintenance_rate: np.ndarray,
    maintenance_regularity: np.ndarray,
    maintenance_loss: np.ndarray,
    cfg: PhaseClassificationConfig,
) -> np.ndarray:
    n = slope_km_day.size
    m = len(PHASE_STATES)
    emissions = np.full((n, m), 1e-6, dtype=np.float64)

    slope = np.nan_to_num(slope_km_day, nan=0.0)
    alt_var = np.nan_to_num(alt_var_km, nan=np.nanmedian(np.abs(alt_var_km[np.isfinite(alt_var_km)])) if np.isfinite(alt_var_km).any() else 5.0)
    bstar_v = np.nan_to_num(bstar, nan=0.0)
    event_v = np.nan_to_num(event_rate, nan=0.0)
    shell_res = np.nan_to_num(shell_residual_km, nan=0.0)
    shell_drift = np.nan_to_num(shell_drift_residual, nan=0.0)
    maneuver_v = np.nan_to_num(maneuver_evidence, nan=0.0)
    maint_rate = np.nan_to_num(maintenance_rate, nan=0.0)
    maint_reg = np.nan_to_num(maintenance_regularity, nan=0.0)
    maint_loss = np.nan_to_num(maintenance_loss, nan=0.0)

    idx = {name: i for i, name in enumerate(PHASE_STATES)}

    emissions[:, idx["unknown"]] = 0.05
    insertion = np.clip((slope - cfg.raise_slope_km_per_day) / 0.8, 0.0, 1.0) * np.clip(0.5 + maneuver_v, 0.0, 1.8)
    emissions[:, idx["insertion_or_orbit_raise"]] = insertion

    transition = np.clip(0.55 * event_v + 0.35 * maneuver_v + 0.10 * np.abs(shell_drift), 0.0, 1.0)
    emissions[:, idx["transition"]] = transition

    stable_altitude = np.clip(1.0 - np.abs(slope) / 0.16, 0.0, 1.0)
    low_var = np.clip(1.0 - alt_var / max(1e-6, cfg.operational_var_km), 0.0, 1.0)
    maintenance_support = np.clip(0.25 + maint_reg + 0.35 * np.tanh(maint_rate), 0.0, 1.6)
    maintenance_penalty = np.clip(1.0 - 0.8 * maint_loss, 0.2, 1.0)
    emissions[:, idx["operational_shell"]] = np.clip(
        stable_altitude * low_var * maintenance_support * maintenance_penalty * max(0.5, cfg.operational_support_weight),
        0.0,
        2.0,
    )

    descending = np.clip((-slope - 0.08) / 0.6, 0.0, 1.0)
    high_drag = np.clip(bstar_v / max(1e-8, cfg.high_bstar_threshold), 0.0, 2.0)

    relocation = np.clip(
        (np.abs(shell_res) / 25.0)
        + 0.7 * maneuver_v
        + 0.4 * event_v
        - 0.5 * maint_reg
        - 0.8 * descending
        - 0.35 * np.clip(high_drag, 0.0, 1.5),
        0.0,
        2.0,
    )
    relocation = np.where(relocation >= float(cfg.relocation_entry_threshold), relocation, 0.25 * relocation)
    emissions[:, idx["relocation"]] = relocation

    disposal = np.clip(0.7 * descending + 0.45 * high_drag + 0.25 * maneuver_v + 0.25 * maint_loss, 0.0, 2.2)
    disposal = np.where(disposal >= float(cfg.disposal_entry_threshold), disposal, 0.2 * disposal)
    emissions[:, idx["disposal_lowering"]] = disposal

    passive = np.clip(0.8 * descending + 0.5 * maint_loss + 0.25 * high_drag - 0.45 * maneuver_v, 0.0, 2.0)
    passive = np.where(passive >= float(cfg.passive_decay_entry_threshold), passive, 0.25 * passive)
    emissions[:, idx["passive_decay"]] = passive

    nonop = np.clip(0.55 * maint_loss + 0.35 * high_drag + 0.30 * descending - 0.45 * stable_altitude, 0.0, 1.8)
    emissions[:, idx["likely_nonoperational"]] = np.where(nonop > 0.7, nonop, 0.25 * nonop)

    emissions = np.maximum(emissions, 1e-6)
    row_sum = emissions.sum(axis=1, keepdims=True)
    emissions = emissions / np.where(row_sum > 0.0, row_sum, 1.0)
    return emissions


def _transition_penalty_matrix(cfg: PhaseClassificationConfig) -> np.ndarray:
    n = len(PHASE_STATES)
    mat = np.full((n, n), cfg.transition_penalty, dtype=np.float64)
    np.fill_diagonal(mat, 0.0)

    idx = {name: i for i, name in enumerate(PHASE_STATES)}

    def penalize(src: str, dst: str, value: float):
        mat[idx[src], idx[dst]] = value

    penalize("insertion_or_orbit_raise", "operational_shell", 0.12)
    penalize("transition", "operational_shell", 0.16)
    penalize("operational_shell", "relocation", 0.45)
    penalize("relocation", "operational_shell", 0.20)
    penalize("operational_shell", "disposal_lowering", 0.65)
    penalize("disposal_lowering", "passive_decay", 0.1)
    penalize("passive_decay", "likely_nonoperational", 0.1)
    penalize("likely_nonoperational", "operational_shell", 1.1)
    penalize("disposal_lowering", "operational_shell", 0.95)

    hard_jumps = [
        ("disposal_lowering", "insertion_or_orbit_raise"),
        ("passive_decay", "insertion_or_orbit_raise"),
        ("likely_nonoperational", "insertion_or_orbit_raise"),
    ]
    for src, dst in hard_jumps:
        penalize(src, dst, cfg.jump_penalty_scale * 2.0)

    return mat


def _viterbi_smooth_states(emissions: np.ndarray, cfg: PhaseClassificationConfig) -> tuple[np.ndarray, np.ndarray]:
    n, m = emissions.shape
    if n == 0:
        return np.array([], dtype=object), np.array([], dtype=np.float64)

    trans_pen = _transition_penalty_matrix(cfg)
    cost = -np.log(np.clip(emissions, 1e-9, 1.0))

    dp = np.full((n, m), np.inf, dtype=np.float64)
    back = np.zeros((n, m), dtype=np.int64)
    dp[0, :] = cost[0, :]

    for i in range(1, n):
        prev = dp[i - 1, :][:, None] + trans_pen
        best_prev = np.argmin(prev, axis=0)
        dp[i, :] = prev[best_prev, np.arange(m)] + cost[i, :]
        back[i, :] = best_prev

    states_idx = np.zeros(n, dtype=np.int64)
    states_idx[-1] = int(np.argmin(dp[-1, :]))
    for i in range(n - 2, -1, -1):
        states_idx[i] = back[i + 1, states_idx[i + 1]]

    states = np.array(PHASE_STATES, dtype=object)[states_idx]
    scores = emissions[np.arange(n), states_idx]
    return states, scores


def _enforce_minimum_phase_dwell(
    states: np.ndarray,
    scores: np.ndarray,
    timestamps: np.ndarray,
    cfg: PhaseClassificationConfig,
) -> np.ndarray:
    if states.size == 0:
        return states
    out = states.astype(object).copy()
    min_days = max(0.0, float(cfg.minimum_phase_dwell_days))
    if min_days <= 0.0:
        return out

    starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
    ends = np.r_[starts[1:] - 1, out.size - 1]
    t = pd.to_datetime(timestamps, errors="coerce")

    for run_idx, (s_idx, e_idx) in enumerate(zip(starts, ends)):
        state = str(out[s_idx])
        if state in {"unknown", "transition"}:
            continue
        t0 = pd.Timestamp(t[s_idx])
        t1 = pd.Timestamp(t[e_idx])
        if pd.isna(t0) or pd.isna(t1):
            continue
        duration_days = max(0.0, float((t1 - t0).total_seconds()) / 86400.0)
        if duration_days >= min_days:
            continue

        mean_score = float(np.nanmean(scores[s_idx : e_idx + 1]))
        if state == "insertion_or_orbit_raise" and mean_score >= float(cfg.insertion_entry_threshold):
            continue
        if state == "disposal_lowering" and mean_score >= float(cfg.disposal_entry_threshold):
            continue
        if state == "passive_decay" and mean_score >= float(cfg.passive_decay_entry_threshold):
            continue

        prev_state = str(out[starts[run_idx - 1]]) if run_idx > 0 else ""
        next_state = str(out[starts[run_idx + 1]]) if run_idx + 1 < starts.size else ""
        replace_state = "transition"
        if prev_state == next_state and prev_state not in {"", "unknown"}:
            replace_state = prev_state
        out[s_idx : e_idx + 1] = replace_state

    return out


def _classify_points_vectorized(
    slope_km_day: np.ndarray,
    alt_var_km: np.ndarray,
    bstar: np.ndarray,
    event_rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = slope_km_day.size
    states = np.full(n, "unknown", dtype=object)
    scores = np.full(n, 0.4, dtype=np.float64)

    slope = np.nan_to_num(slope_km_day, nan=0.0)
    alt_var = np.nan_to_num(alt_var_km, nan=np.inf)
    bstar_v = np.nan_to_num(bstar, nan=0.0)
    event_v = np.nan_to_num(event_rate, nan=0.0)
    idx = np.arange(n)

    unresolved = np.ones(n, dtype=bool)

    cond = unresolved & (idx < 5) & (slope > 0.4)
    states[cond], scores[cond], unresolved[cond] = "insertion_or_orbit_raise", 0.7, False
    cond = unresolved & (slope > 0.3) & (event_v > 0.05)
    states[cond], scores[cond], unresolved[cond] = "relocation", 0.65, False
    cond = unresolved & (slope > 0.2)
    states[cond], scores[cond], unresolved[cond] = "transition", 0.6, False
    cond = unresolved & (slope < -0.3) & (bstar_v > 3e-4)
    states[cond], scores[cond], unresolved[cond] = "disposal_lowering", 0.75, False
    cond = unresolved & (slope < -0.2) & (alt_var < 10.0)
    states[cond], scores[cond], unresolved[cond] = "passive_decay", 0.65, False
    cond = unresolved & (np.abs(slope) <= 0.05) & (alt_var <= 8.0) & (event_v < 0.08)
    states[cond], scores[cond], unresolved[cond] = "operational_shell", 0.8, False
    cond = unresolved & (event_v > 0.2)
    states[cond], scores[cond], unresolved[cond] = "transition", 0.55, False
    cond = unresolved & (bstar_v > 6e-4) & (slope < -0.1)
    states[cond], scores[cond], unresolved[cond] = "likely_nonoperational", 0.7, False

    return states, scores


def classify_mission_phase_for_satellite(
    df_sat: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    config: Optional[PhaseClassificationConfig] = None,
    _event_ns_lookup: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    cfg = config or PhaseClassificationConfig()
    sat, _audit = prepare_satellite_timeseries(df_sat, config=cfg.model, enrich_phase=False, enrich_altitude=True, return_audit=True)

    cols = [
        "object_id",
        "sat_id",
        "norad_cat_id",
        "timestamp",
        "phase_state",
        "phase_score",
        "phase_confidence",
        "altitude_slope_km_day",
        "event_density",
        "maneuver_evidence",
        "maintenance_event_rate",
        "maintenance_regularity_score",
        "maintenance_loss_score",
        "mean_bstar",
        "transition_rationale",
    ]
    if len(sat) < max(cfg.min_records, cfg.model.min_records):
        empty = sat[["timestamp"]].copy() if "timestamp" in sat.columns else pd.DataFrame({"timestamp": []})
        empty["phase_state"] = "unknown"
        empty["phase_score"] = 0.0
        empty["phase_confidence"] = 0.0
        empty["altitude_slope_km_day"] = np.nan
        empty["event_density"] = 0.0
        empty["maneuver_evidence"] = 0.0
        empty["maintenance_event_rate"] = 0.0
        empty["maintenance_regularity_score"] = 0.0
        empty["maintenance_loss_score"] = 0.0
        empty["mean_bstar"] = np.nan
        empty["transition_rationale"] = "insufficient_history"
        empty["sat_id"] = sat["sat_id"].iloc[0] if "sat_id" in sat.columns and len(sat) > 0 else ""
        empty["norad_cat_id"] = sat["norad_cat_id"].iloc[0] if "norad_cat_id" in sat.columns and len(sat) > 0 else ""
        empty["object_id"] = empty["norad_cat_id"].astype(str)
        m = empty["object_id"].isin(["", "nan", "None"])
        empty.loc[m, "object_id"] = empty.loc[m, "sat_id"].astype(str)
        return empty[cols]

    features = build_maneuver_feature_frame(sat)

    alt = pd.to_numeric(features.get("altitude_km", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    t = pd.to_numeric(features["elapsed_seconds"], errors="coerce").to_numpy(dtype=np.float64)
    ts = pd.to_datetime(features["timestamp"], errors="coerce").to_numpy()
    bstar = _effective_bstar(features)

    trend_alt = rolling_poly_trend(t, alt, window=cfg.rolling_window_points, order=1)
    slope_alt_km_s = first_derivative_irregular(trend_alt, t)
    slope_alt_km_day = slope_alt_km_s * 86400.0

    alt_series = pd.Series(alt)
    roll_var = alt_series.rolling(window=max(5, cfg.rolling_window_points), min_periods=3, center=True).std().to_numpy(dtype=np.float64)
    roll_var = np.where(np.isfinite(roll_var), roll_var, np.nanmedian(np.abs(alt - np.nanmedian(alt))))

    object_id, sat_id, norad = _resolve_object_fields(features)

    event_rate = _event_rate_around_times(ts, events_df, object_id=object_id, window_days=7.0, event_ns_lookup=_event_ns_lookup)
    maneuver_evidence = _event_evidence_around_times(ts, events_df, object_id=object_id)
    maint_rate, maint_regularity, maint_loss = _maintenance_features_around_times(ts, events_df, object_id=object_id, cfg=cfg)
    shell_residual = pd.to_numeric(features.get("shell_relative_altitude_residual_km", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    shell_drift = pd.to_numeric(features.get("shell_relative_secular_drift_residual_km_s", np.nan), errors="coerce").to_numpy(dtype=np.float64)

    model_mode = str(cfg.phase_model_mode or "transition_penalized").strip().lower()
    if cfg.use_persistence_smoothing and model_mode == "transition_penalized":
        emissions = _phase_emission_scores(
            slope_alt_km_day,
            roll_var,
            bstar,
            event_rate,
            shell_residual,
            shell_drift,
            maneuver_evidence,
            maint_rate,
            maint_regularity,
            maint_loss,
            cfg,
        )
        raw_states, raw_scores = _viterbi_smooth_states(emissions, cfg)
        states = _enforce_minimum_phase_dwell(raw_states, raw_scores, ts, cfg)
        raw_raise_mask = np.asarray(raw_states, dtype=object) == "insertion_or_orbit_raise"
        chosen_idx = np.array([PHASE_STATES.index(str(s)) for s in states], dtype=np.int64)
        selected = emissions[np.arange(len(states)), chosen_idx]
        second = np.partition(emissions, -2, axis=1)[:, -2] if emissions.shape[1] > 1 else np.zeros_like(selected)
        confidence = np.clip(selected - second, 0.0, 1.0)
        scores = selected
    else:
        raw_states, raw_scores = _classify_points_vectorized(slope_alt_km_day, roll_var, bstar, event_rate)
        states = _enforce_minimum_phase_dwell(raw_states, raw_scores, ts, cfg)
        raw_raise_mask = np.asarray(raw_states, dtype=object) == "insertion_or_orbit_raise"
        scores = np.asarray(raw_scores, dtype=np.float64)
        confidence = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)

    # Conservative compatibility guard: sustained descent with elevated drag
    # should remain disposal-like even when transition smoothing is dominant.
    descent_mask = np.nan_to_num(slope_alt_km_day, nan=0.0) <= min(float(cfg.lower_slope_km_per_day), -0.25)
    drag_mask = np.nan_to_num(bstar, nan=0.0) >= max(3.0e-4, 0.9 * float(cfg.high_bstar_threshold))
    disposal_override = descent_mask & drag_mask
    if np.any(disposal_override):
        states = np.asarray(states, dtype=object)
        scores = np.asarray(scores, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        states[disposal_override] = "disposal_lowering"
        scores[disposal_override] = np.maximum(scores[disposal_override], 0.85)
        confidence[disposal_override] = np.maximum(confidence[disposal_override], 0.70)

    # Preserve short but sharp ascent episodes that are typical of insertion/
    # orbit-raising burns and can otherwise be smoothed away by dwell rules.
    slope_v = np.nan_to_num(slope_alt_km_day, nan=0.0)
    strong_raise = slope_v >= max(float(cfg.raise_override_slope_km_per_day), float(cfg.raise_slope_km_per_day) + 0.15)
    supported_raise = (
        (slope_v >= float(cfg.raise_slope_km_per_day))
        & (
            raw_raise_mask
            | (np.nan_to_num(event_rate, nan=0.0) >= float(cfg.raise_event_rate_threshold))
            | (np.nan_to_num(maneuver_evidence, nan=0.0) >= float(cfg.raise_maneuver_evidence_threshold))
        )
    )
    raise_override = strong_raise | supported_raise
    if np.any(raise_override):
        states = np.asarray(states, dtype=object)
        scores = np.asarray(scores, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        states[raise_override] = "insertion_or_orbit_raise"
        scores[raise_override] = np.maximum(scores[raise_override], float(cfg.insertion_entry_threshold))
        confidence[raise_override] = np.maximum(confidence[raise_override], 0.55)

    # Treat relocation as an operational/maintenance behavior family instead of
    # a standalone long-lived phase state for reporting and plotting.
    reloc_mask = np.asarray(states, dtype=object) == "relocation"
    if np.any(reloc_mask):
        states = np.asarray(states, dtype=object)
        scores = np.asarray(scores, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)

        states[reloc_mask] = "operational_shell"
        scores[reloc_mask] = np.maximum(scores[reloc_mask], 0.65)
        confidence[reloc_mask] = np.maximum(confidence[reloc_mask], 0.50)

    rationale = np.full(states.shape, "transition_evidence", dtype=object)
    rationale[np.asarray(states) == "operational_shell"] = "stable_altitude_and_regular_maintenance"
    rationale[np.asarray(states) == "relocation"] = "persistent_shell_offset_with_maneuver_support"
    rationale[np.asarray(states) == "disposal_lowering"] = "sustained_descent_and_drag_or_event_support"
    rationale[np.asarray(states) == "passive_decay"] = "prolonged_descent_with_maintenance_loss"
    rationale[np.asarray(states) == "likely_nonoperational"] = "maintenance_loss_and_drag_consistency"
    rationale[np.asarray(states) == "insertion_or_orbit_raise"] = "positive_altitude_slope_with_maneuver_support"

    out = pd.DataFrame(
        {
            "object_id": object_id,
            "sat_id": sat_id,
            "norad_cat_id": norad,
            "timestamp": pd.to_datetime(features["timestamp"], errors="coerce"),
            "phase_state": states,
            "phase_score": np.asarray(scores, dtype=np.float64),
            "phase_confidence": np.asarray(confidence, dtype=np.float64),
            "altitude_slope_km_day": np.asarray(slope_alt_km_day, dtype=np.float64),
            "event_density": np.asarray(event_rate, dtype=np.float64),
            "maneuver_evidence": np.asarray(maneuver_evidence, dtype=np.float64),
            "maintenance_event_rate": np.asarray(maint_rate, dtype=np.float64),
            "maintenance_regularity_score": np.asarray(maint_regularity, dtype=np.float64),
            "maintenance_loss_score": np.asarray(maint_loss, dtype=np.float64),
            "mean_bstar": np.asarray(bstar, dtype=np.float64),
            "transition_rationale": rationale,
        }
    )
    if not bool(cfg.phase_confidence_export):
        out["phase_confidence"] = out["phase_score"]
    return out[cols]


def summarize_phase_intervals(phase_df: pd.DataFrame) -> pd.DataFrame:
    if phase_df.empty:
        return pd.DataFrame(columns=[
            "object_id",
            "sat_id",
            "norad_cat_id",
            "phase_state",
            "phase_start",
            "phase_end",
            "n_records",
            "mean_score",
            "interval_confidence",
            "dominant_supporting_features",
            "event_support_statistics",
            "mean_altitude_slope_km_day",
            "median_altitude_slope_km_day",
            "mean_bstar",
            "event_density",
            "maneuver_evidence_summary",
        ])

    rows = []
    for object_id, grp in phase_df.sort_values("timestamp", kind="mergesort").groupby("object_id", sort=True):
        grp = grp.reset_index(drop=True)
        if grp.empty:
            continue

        phase_state = grp["phase_state"].fillna("unknown").to_numpy(dtype=object)
        starts = np.flatnonzero(np.r_[True, phase_state[1:] != phase_state[:-1]])
        ends = np.r_[starts[1:] - 1, len(grp) - 1]
        counts = (ends - starts + 1).astype(int)

        score_arr = pd.to_numeric(grp["phase_score"], errors="coerce").to_numpy(dtype=np.float64)
        score_sum = np.add.reduceat(np.nan_to_num(score_arr, nan=0.0), starts)
        score_count = np.add.reduceat(np.isfinite(score_arr).astype(np.int64), starts)
        mean_score = np.divide(score_sum, score_count, out=np.zeros_like(score_sum, dtype=np.float64), where=score_count > 0)

        part = pd.DataFrame(
            {
                "object_id": object_id,
                "sat_id": grp["sat_id"].to_numpy(dtype=object)[starts],
                "norad_cat_id": grp["norad_cat_id"].to_numpy(dtype=object)[starts],
                "phase_state": phase_state[starts],
                "phase_start": grp["timestamp"].to_numpy()[starts],
                "phase_end": grp["timestamp"].to_numpy()[ends],
                "n_records": counts,
                "mean_score": mean_score,
            }
        )

        conf_arr = pd.to_numeric(grp.get("phase_confidence", grp.get("phase_score", np.nan)), errors="coerce").to_numpy(dtype=np.float64)
        slope_arr = pd.to_numeric(grp.get("altitude_slope_km_day", np.nan), errors="coerce").to_numpy(dtype=np.float64)
        bstar_arr = pd.to_numeric(grp.get("mean_bstar", np.nan), errors="coerce").to_numpy(dtype=np.float64)
        event_density_arr = pd.to_numeric(grp.get("event_density", np.nan), errors="coerce").to_numpy(dtype=np.float64)
        maneuver_arr = pd.to_numeric(grp.get("maneuver_evidence", np.nan), errors="coerce").to_numpy(dtype=np.float64)
        rationale_arr = grp.get("transition_rationale", pd.Series(["n/a"] * len(grp))).astype(str).to_numpy(dtype=object)

        conf_mean = []
        slope_mean = []
        slope_median = []
        bstar_mean = []
        density_mean = []
        maneuver_mean = []
        rationale_mode = []
        support_stats = []

        for s_idx, e_idx in zip(starts, ends):
            sl = slice(int(s_idx), int(e_idx) + 1)
            conf_mean.append(float(np.nanmean(conf_arr[sl])) if np.isfinite(conf_arr[sl]).any() else np.nan)
            slope_mean.append(float(np.nanmean(slope_arr[sl])) if np.isfinite(slope_arr[sl]).any() else np.nan)
            slope_median.append(float(np.nanmedian(slope_arr[sl])) if np.isfinite(slope_arr[sl]).any() else np.nan)
            bstar_mean.append(float(np.nanmean(bstar_arr[sl])) if np.isfinite(bstar_arr[sl]).any() else np.nan)
            density_mean.append(float(np.nanmean(event_density_arr[sl])) if np.isfinite(event_density_arr[sl]).any() else np.nan)
            maneuver_mean.append(float(np.nanmean(maneuver_arr[sl])) if np.isfinite(maneuver_arr[sl]).any() else np.nan)

            rr = pd.Series(rationale_arr[sl]).dropna().astype(str)
            rationale_mode.append(rr.mode().iloc[0] if not rr.empty else "n/a")
            support_stats.append(
                {
                    "mean_event_density": density_mean[-1],
                    "mean_maneuver_evidence": maneuver_mean[-1],
                    "mean_phase_confidence": conf_mean[-1],
                }
            )

        part["interval_confidence"] = conf_mean
        part["dominant_supporting_features"] = rationale_mode
        part["event_support_statistics"] = support_stats
        part["mean_altitude_slope_km_day"] = slope_mean
        part["median_altitude_slope_km_day"] = slope_median
        part["mean_bstar"] = bstar_mean
        part["event_density"] = density_mean
        part["maneuver_evidence_summary"] = maneuver_mean
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=[
            "object_id",
            "sat_id",
            "norad_cat_id",
            "phase_state",
            "phase_start",
            "phase_end",
            "n_records",
            "mean_score",
            "interval_confidence",
            "dominant_supporting_features",
            "event_support_statistics",
            "mean_altitude_slope_km_day",
            "median_altitude_slope_km_day",
            "mean_bstar",
            "event_density",
            "maneuver_evidence_summary",
        ])

    return pd.concat(rows, ignore_index=True)


def classify_mission_phase_all(
    df: pd.DataFrame,
    object_col: str = "sat_id",
    events_df: Optional[pd.DataFrame] = None,
    config: Optional[PhaseClassificationConfig] = None,
) -> pd.DataFrame:
    cfg = config or PhaseClassificationConfig()
    data = ensure_panel_sorted(df, object_col=object_col, time_col="timestamp")

    try:
        group_col = select_object_id_column(data, preferred=cfg.model.preferred_object_cols)
    except KeyError:
        group_col = object_col

    event_lookup = _build_event_time_lookup(events_df)

    grouped = data.groupby(group_col, sort=True)
    chunks = []
    total_objects = int(grouped.ngroups)
    t_loop = pd.Timestamp.utcnow()
    try:
        progress_every = max(1, int(cfg.progress_every_objects))
    except Exception:
        progress_every = 100
    show_progress = bool(cfg.print_progress and total_objects >= progress_every)

    failed_objects: List[Dict[str, Any]] = []
    for idx, (obj_id, grp) in enumerate(grouped, start=1):
        try:
            p = classify_mission_phase_for_satellite(grp, events_df=events_df, config=cfg, _event_ns_lookup=event_lookup)
        except Exception as exc:
            failed_objects.append(
                {
                    "object_id": str(obj_id),
                    "rows": int(len(grp)),
                    "error": str(exc),
                }
            )
            continue

        if not p.empty:
            chunks.append(p)

        if show_progress and (idx % progress_every == 0 or idx == total_objects):
            elapsed = (pd.Timestamp.utcnow() - t_loop).total_seconds()
            rate = float(idx / max(1e-9, elapsed))
            eta = float((total_objects - idx) / max(1e-9, rate))
            print(
                f"[phase_classification] processed {idx}/{total_objects} objects "
                f"({rate:.1f} obj/s, eta {eta:.1f}s)"
            )

    if failed_objects:
        print(
            f"[phase_classification] Skipped {len(failed_objects)} object(s) due to per-object errors; "
            "continuing with remaining objects."
        )

    if not chunks:
        return pd.DataFrame(columns=[
            "object_id",
            "sat_id",
            "norad_cat_id",
            "timestamp",
            "phase_state",
            "phase_score",
            "phase_confidence",
            "altitude_slope_km_day",
            "event_density",
            "maneuver_evidence",
            "maintenance_event_rate",
            "maintenance_regularity_score",
            "maintenance_loss_score",
            "mean_bstar",
            "transition_rationale",
        ])

    return pd.concat(chunks, ignore_index=True).sort_values(["object_id", "timestamp"], kind="mergesort").reset_index(drop=True)


# ---- Evaluation and synthetic hooks ----
def inject_synthetic_maneuvers(
    df_sat: pd.DataFrame,
    schedule: Optional[List[Dict[str, Any]]] = None,
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject controlled step/slope signatures for detector regression tests."""
    if schedule is None:
        schedule = []

    out = df_sat.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")

    truth_rows: List[Dict[str, Any]] = []
    for item in schedule:
        at_time = pd.Timestamp(item.get("time"))
        mode = str(item.get("mode", "step")).strip().lower()
        delta_sma = float(item.get("delta_sma_km", 0.0))
        delta_inc = float(item.get("delta_inc_deg", 0.0))
        set_bstar = item.get("set_bstar")
        duration_points = max(1, int(item.get("duration_points", 5)))
        noise_sigma_sma = float(item.get("noise_sigma_sma_km", 0.0))
        noise_sigma_inc = float(item.get("noise_sigma_inc_deg", 0.0))

        mask = out[time_col] >= at_time
        idx = np.flatnonzero(mask.to_numpy(dtype=bool))
        if idx.size == 0:
            continue

        if mode == "slope":
            ramp_len = min(duration_points, idx.size)
            ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float64)
            tail = np.ones(max(0, idx.size - ramp_len), dtype=np.float64)
            profile = np.concatenate([ramp, tail])
        else:
            profile = np.ones(idx.size, dtype=np.float64)

        if "sma" in out.columns and delta_sma != 0.0:
            sma_vals = pd.to_numeric(out.loc[mask, "sma"], errors="coerce").to_numpy(dtype=np.float64)
            sma_vals = sma_vals + delta_sma * profile
            if noise_sigma_sma > 0.0:
                sma_vals = sma_vals + np.random.normal(0.0, noise_sigma_sma, size=sma_vals.size)
            out.loc[mask, "sma"] = sma_vals

        if "inc" in out.columns and delta_inc != 0.0:
            inc_vals = pd.to_numeric(out.loc[mask, "inc"], errors="coerce").to_numpy(dtype=np.float64)
            inc_vals = inc_vals + delta_inc * profile
            if noise_sigma_inc > 0.0:
                inc_vals = inc_vals + np.random.normal(0.0, noise_sigma_inc, size=inc_vals.size)
            out.loc[mask, "inc"] = inc_vals

        if set_bstar is not None:
            if "bstar" not in out.columns:
                out["bstar"] = np.nan
            out.loc[mask, "bstar"] = float(set_bstar)

        truth_rows.append(
            {
                "event_time": at_time,
                "truth_time": at_time,
                "event_type": str(item.get("event_type", "unknown_event")),
                "maneuver_type": str(item.get("event_type", "unknown_event")),
                "delta_sma_km": delta_sma,
                "delta_inc_deg": delta_inc,
                "tolerance_seconds": float(item.get("tolerance_seconds", 24 * 3600.0)),
            }
        )

    return out, pd.DataFrame(truth_rows)


def evaluate_maneuver_detections(
    detected_events_df: pd.DataFrame,
    truth_events_df: pd.DataFrame,
    *,
    object_col: str = "object_id",
    detected_time_col: str = "estimated_event_time",
    truth_time_col: str = "truth_time",
    detected_type_col: str = "event_type",
    truth_type_col: str = "maneuver_type",
    tolerance_seconds_col: str = "tolerance_seconds",
    tolerance: str = "24h",
) -> Dict[str, Any]:
    """Evaluate detection quality against known maneuver histories when available."""
    tolerance_norm = str(tolerance).replace("H", "h").replace("D", "d")
    tol_seconds = float(pd.to_timedelta(tolerance_norm).total_seconds())

    detected = detected_events_df.copy() if detected_events_df is not None else pd.DataFrame()
    truth = truth_events_df.copy() if truth_events_df is not None else pd.DataFrame()

    if truth_time_col not in truth.columns and "event_time" in truth.columns:
        truth[truth_time_col] = truth["event_time"]

    if detected.empty and truth.empty:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "lag_seconds_mean": np.nan,
            "lag_seconds_median": np.nan,
            "lag_seconds_p95": np.nan,
            "median_absolute_lag_seconds": np.nan,
            "p95_absolute_lag_seconds": np.nan,
            "false_positives": 0,
            "false_negatives": 0,
            "type_confusion": {},
        }

    if detected.empty:
        fn = int(len(truth))
        return {
            "tp": 0,
            "fp": 0,
            "fn": fn,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "lag_seconds_mean": np.nan,
            "lag_seconds_median": np.nan,
            "lag_seconds_p95": np.nan,
            "median_absolute_lag_seconds": np.nan,
            "p95_absolute_lag_seconds": np.nan,
            "false_positives": 0,
            "false_negatives": fn,
            "type_confusion": {},
        }

    if truth.empty:
        fp = int(len(detected))
        return {
            "tp": 0,
            "fp": fp,
            "fn": 0,
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "lag_seconds_mean": np.nan,
            "lag_seconds_median": np.nan,
            "lag_seconds_p95": np.nan,
            "median_absolute_lag_seconds": np.nan,
            "p95_absolute_lag_seconds": np.nan,
            "false_positives": fp,
            "false_negatives": 0,
            "type_confusion": {},
        }

    if object_col not in detected.columns:
        detected[object_col] = "global"
    if object_col not in truth.columns:
        truth[object_col] = "global"

    detected[detected_time_col] = pd.to_datetime(detected[detected_time_col], errors="coerce")
    truth[truth_time_col] = pd.to_datetime(truth[truth_time_col], errors="coerce")
    detected = detected.dropna(subset=[detected_time_col]).reset_index(drop=True)
    truth = truth.dropna(subset=[truth_time_col]).reset_index(drop=True)

    used_detected = np.zeros(len(detected), dtype=bool)
    tp = 0
    lag_seconds: List[float] = []
    confusion: Dict[str, int] = {}

    for _, truth_row in truth.iterrows():
        object_id = str(truth_row[object_col])
        truth_t = pd.Timestamp(truth_row[truth_time_col])

        candidate_idx = detected.index[(detected[object_col].astype(str) == object_id) & (~used_detected)]
        if len(candidate_idx) == 0:
            continue

        dtime = pd.to_datetime(detected.loc[candidate_idx, detected_time_col], errors="coerce")
        delta_s = (dtime - truth_t).dt.total_seconds().abs()
        if delta_s.empty or delta_s.isna().all():
            continue

        best_local = int(delta_s.idxmin())
        best_delta = float(delta_s.loc[best_local])
        row_tol_seconds = float(truth_row.get(tolerance_seconds_col, tol_seconds)) if tolerance_seconds_col in truth.columns else tol_seconds
        if best_delta <= row_tol_seconds:
            used_detected[best_local] = True
            tp += 1
            signed_lag = float((pd.Timestamp(detected.loc[best_local, detected_time_col]) - truth_t).total_seconds())
            lag_seconds.append(signed_lag)
            if truth_type_col in truth.columns and detected_type_col in detected.columns:
                true_label = str(truth_row.get(truth_type_col, "unknown"))
                pred_label = str(detected.loc[best_local, detected_type_col])
                key = f"{true_label}->{pred_label}"
                confusion[key] = int(confusion.get(key, 0) + 1)

    fp = int((~used_detected).sum())
    fn = int(len(truth) - tp)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    lag_arr = np.asarray(lag_seconds, dtype=np.float64)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lag_seconds_mean": float(np.nanmean(lag_arr)) if lag_arr.size else np.nan,
        "lag_seconds_median": float(np.nanmedian(lag_arr)) if lag_arr.size else np.nan,
        "lag_seconds_p95": float(np.nanpercentile(np.abs(lag_arr), 95)) if lag_arr.size else np.nan,
        "median_absolute_lag_seconds": float(np.nanmedian(np.abs(lag_arr))) if lag_arr.size else np.nan,
        "p95_absolute_lag_seconds": float(np.nanpercentile(np.abs(lag_arr), 95)) if lag_arr.size else np.nan,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "type_confusion": confusion,
    }


def evaluate_detector_family_ablation(
    df: pd.DataFrame,
    truth_events_df: pd.DataFrame,
    base_config: Optional[ManeuverDetectionConfig] = None,
    *,
    object_col: str = "sat_id",
    tolerance: str = "24h",
) -> pd.DataFrame:
    """Compare detector-family ablations with consistent evaluation settings."""
    base = deepcopy(base_config) if base_config is not None else ManeuverDetectionConfig()

    experiments = [
        ("local_residual_only", ("local_residual", "anomaly_guard"), False),
        ("smoothed_change_only", ("smoothed_change_segment", "anomaly_guard"), False),
        ("propagate_compare_only", ("simplified_propagate_compare", "anomaly_guard"), True),
        ("kelecy_filtered_difference_only", ("kelecy_filtered_difference", "anomaly_guard"), False),
        ("fused_detector", (), base.enable_propagate_compare),
    ]

    rows: List[Dict[str, Any]] = []
    for label, custom_families, force_propagate in experiments:
        cfg = deepcopy(base)
        if label == "fused_detector":
            cfg.detector_family_mode = "all"
        else:
            cfg.detector_family_mode = "custom"
            cfg.custom_detector_families = tuple(custom_families)
        cfg.enable_propagate_compare = bool(force_propagate)

        detected = detect_maneuvers_all(df, object_col=object_col, config=cfg)
        metrics = evaluate_maneuver_detections(
            detected,
            truth_events_df,
            tolerance=tolerance,
        )
        metrics["ablation_mode"] = label
        metrics["detected_events"] = int(len(detected))
        rows.append(metrics)

    if not rows:
        return pd.DataFrame(columns=["ablation_mode", "precision", "recall", "f1", "false_positives", "false_negatives"])
    return pd.DataFrame(rows)
