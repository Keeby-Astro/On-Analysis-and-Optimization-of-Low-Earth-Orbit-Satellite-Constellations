"""Shared preprocessing helpers for spectral/time-frequency analysis.

This module centralizes deterministic, archive-safe utilities used by:
- global spectral analysis (FFT, Lomb-Scargle, Welch/STFT)
- local time-frequency analysis (CWT/WWZ)
- pairwise coupling analysis (cross-correlation, CSD, coherence)
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

SECONDS_PER_DAY = 86400.0


def to_datetime64_seconds(timestamps) -> Optional[np.ndarray]:
    if timestamps is None:
        return None
    arr = np.asarray(timestamps)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[s]")
    try:
        return arr.astype("datetime64[s]")
    except Exception:
        return None


def build_elapsed_seconds(
    timestamps,
    n_samples: int,
    *,
    fallback_units: str = "samples",
    warning_prefix: str = "spectral",
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Build elapsed time axis in seconds with deterministic fallback semantics."""
    ts = to_datetime64_seconds(timestamps)
    if ts is None or ts.size != int(n_samples):
        warnings.warn(
            f"{warning_prefix}: timestamps missing or invalid; using sample-index time axis.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.arange(int(n_samples), dtype=np.float64), {
            "time_basis": "sample_index",
            "units": fallback_units,
            "is_physical_time": "false",
        }

    ts_int = ts.astype("int64")
    t0 = float(ts_int[0])
    elapsed = ts_int.astype(np.float64) - t0
    return elapsed, {"time_basis": "timestamp", "units": "seconds", "is_physical_time": "true"}


def unique_in_order(values: Iterable) -> List[str]:
    seen = set()
    out = []
    for v in values:
        s = str(v)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def build_satellite_index_map(sat_ids_str: np.ndarray):
    order = np.argsort(sat_ids_str, kind="mergesort")
    sorted_ids = sat_ids_str[order]
    unique_ids, start_idx = np.unique(sorted_ids, return_index=True)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = order.size
    bounds = {
        sid: (int(s), int(e))
        for sid, s, e in zip(unique_ids.tolist(), start_idx.tolist(), end_idx.tolist())
    }

    def get_indices(sid):
        window = bounds.get(str(sid))
        if window is None:
            return np.empty(0, dtype=np.int64)
        s, e = window
        return order[s:e]

    return get_indices


def is_irregular_time_axis(t_seconds: np.ndarray, rel_tol: float = 0.1) -> bool:
    t = np.asarray(t_seconds, dtype=np.float64)
    if t.size < 4:
        return False
    dt = np.diff(np.sort(t))
    dt = dt[dt > 0]
    if dt.size < 3:
        return False
    median_dt = float(np.median(dt))
    if median_dt <= 0.0 or not np.isfinite(median_dt):
        return False
    max_dev = float(np.max(np.abs(dt - median_dt)))
    return bool((max_dev / median_dt) > float(rel_tol))


def infer_positive_cadence_seconds(t_seconds: np.ndarray) -> float:
    t = np.asarray(t_seconds, dtype=np.float64)
    if t.size < 2:
        return np.nan
    dt = np.diff(np.sort(t))
    dt = dt[dt > 0]
    if dt.size == 0:
        return np.nan
    return float(np.median(dt))


def _sanitize_time_value(t_seconds: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_seconds, dtype=np.float64)
    v = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(v)
    t = t[valid]
    v = v[valid]
    if t.size == 0:
        return np.array([]), np.array([])

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[order]

    # Keep first occurrence for duplicate timestamps to preserve deterministic behavior.
    uniq, idx = np.unique(t, return_index=True)
    t = uniq
    v = v[idx]
    return t, v


def resample_to_uniform_grid(
    t_seconds: np.ndarray,
    y: np.ndarray,
    cadence_seconds: Optional[float] = None,
    *,
    interpolation: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, float]:
    t, v = _sanitize_time_value(t_seconds, y)
    if t.size < 4:
        return np.array([]), np.array([]), np.nan

    inferred = infer_positive_cadence_seconds(t)
    cad = inferred if cadence_seconds is None else float(cadence_seconds)
    if not np.isfinite(cad) or cad <= 0.0:
        return np.array([]), np.array([]), np.nan

    cad = max(float(cad), 1e-9)
    grid = np.arange(t[0], t[-1] + 0.5 * cad, cad, dtype=np.float64)
    if grid.size < 4:
        return np.array([]), np.array([]), np.nan

    if str(interpolation).lower() != "linear":
        warnings.warn(
            f"spectral_preprocessing: interpolation '{interpolation}' unsupported; using linear.",
            RuntimeWarning,
            stacklevel=2,
        )
    values = np.interp(grid, t, v)
    return grid, values, cad


def resample_pair_to_uniform_grid(
    t_seconds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cadence_seconds: Optional[float] = None,
    *,
    max_grid_points: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t = np.asarray(t_seconds, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(t) & np.isfinite(x_arr) & np.isfinite(y_arr)
    t = t[valid]
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if t.size < 4:
        return np.array([]), np.array([]), np.array([]), np.nan

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    uniq, idx = np.unique(t, return_index=True)
    t = uniq
    x_arr = x_arr[idx]
    y_arr = y_arr[idx]

    if t.size < 4:
        return np.array([]), np.array([]), np.array([]), np.nan

    inferred = infer_positive_cadence_seconds(t)
    cad = inferred if cadence_seconds is None else float(cadence_seconds)
    if not np.isfinite(cad) or cad <= 0.0:
        return np.array([]), np.array([]), np.array([]), np.nan
    cad = max(float(cad), 1e-9)

    span = float(t[-1] - t[0])
    if not np.isfinite(span) or span <= 0.0:
        return np.array([]), np.array([]), np.array([]), np.nan

    target_max = max(4, int(max_grid_points))
    est_points = int(np.floor(span / cad)) + 1
    if est_points > target_max:
        cad = span / float(target_max - 1)

    grid = np.arange(t[0], t[-1] + 0.5 * cad, cad, dtype=np.float64)
    if grid.size < 4:
        return np.array([]), np.array([]), np.array([]), np.nan

    x_u = np.interp(grid, t, x_arr)
    y_u = np.interp(grid, t, y_arr)
    return grid, x_u, y_u, cad


def preprocess_series(values: np.ndarray, mode: str = "raw") -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    m = str(mode).lower()
    if m == "raw":
        return v
    if m == "detrended":
        if v.size < 2:
            return v
        idx = np.arange(v.size, dtype=np.float64)
        coef = np.polyfit(idx, v, deg=1)
        return v - np.polyval(coef, idx)
    if m in {"zscored", "z-scored", "zscore"}:
        std = float(np.std(v))
        if std <= 0.0 or not np.isfinite(std):
            return v - np.mean(v)
        return (v - np.mean(v)) / std
    if m == "differenced":
        if v.size < 2:
            return np.array([], dtype=np.float64)
        return np.diff(v)

    warnings.warn(
        f"spectral_preprocessing: preprocessing mode '{mode}' not recognized; using raw.",
        RuntimeWarning,
        stacklevel=2,
    )
    return v


def zscore(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    std = float(np.std(arr))
    if std <= 0.0 or not np.isfinite(std):
        return arr - np.mean(arr)
    return (arr - np.mean(arr)) / std


def unwrap_degrees(angle_deg: np.ndarray) -> np.ndarray:
    arr = np.asarray(angle_deg, dtype=np.float64)
    return np.rad2deg(np.unwrap(np.deg2rad(arr)))


def wrap_degrees(angle_deg: np.ndarray) -> np.ndarray:
    arr = np.asarray(angle_deg, dtype=np.float64)
    return np.mod(arr, 360.0)


def substitute_low_e_phase(
    base_phase: np.ndarray,
    low_e_phase: Optional[np.ndarray],
    eccentricities: Optional[np.ndarray],
    *,
    ecc_threshold: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Substitute low-e-safe phase where eccentricity is below threshold."""
    base = np.asarray(base_phase, dtype=np.float64)
    if low_e_phase is None or eccentricities is None:
        return base, {
            "low_e_substitution_applied": 0.0,
            "low_e_fraction": 0.0,
            "ecc_threshold": float(ecc_threshold),
        }

    phase_alt = np.asarray(low_e_phase, dtype=np.float64)
    ecc = np.asarray(eccentricities, dtype=np.float64)
    if phase_alt.shape != base.shape or ecc.shape != base.shape:
        return base, {
            "low_e_substitution_applied": 0.0,
            "low_e_fraction": 0.0,
            "ecc_threshold": float(ecc_threshold),
            "note": "shape_mismatch",
        }

    low_e = np.isfinite(ecc) & (ecc < float(ecc_threshold))
    out = np.where(low_e, phase_alt, base)
    return out, {
        "low_e_substitution_applied": float(np.any(low_e)),
        "low_e_fraction": float(np.mean(low_e)) if low_e.size else 0.0,
        "ecc_threshold": float(ecc_threshold),
    }


def warn_irregular_resampled(module_name: str, sat_id: str) -> None:
    warnings.warn(
        f"{module_name}: irregular timestamps detected for satellite {sat_id}; resampling to uniform cadence for method compatibility.",
        RuntimeWarning,
        stacklevel=2,
    )


def warn_small_sample(module_name: str, sat_id: str, n_samples: int, min_needed: int) -> None:
    warnings.warn(
        f"{module_name}: satellite {sat_id} has {n_samples} samples; stability is limited (recommended >= {min_needed}).",
        RuntimeWarning,
        stacklevel=2,
    )


def warn_method_amplitude_comparison() -> None:
    warnings.warn(
        "Amplitudes from uniform FFT and Lomb-Scargle are not directly interchangeable without explicit normalization and calibration.",
        RuntimeWarning,
        stacklevel=2,
    )


def build_uniform_resampling_metadata(
    t_original_seconds: np.ndarray,
    uniform_grid_seconds: np.ndarray,
    *,
    used_cadence_seconds: float,
    interpolation_method: str,
    resampled: bool,
    input_irregular: bool,
) -> Dict[str, float | bool | str]:
    """Build required provenance fields for uniform-grid spectral products."""
    t_orig = np.asarray(t_original_seconds, dtype=np.float64)
    grid = np.asarray(uniform_grid_seconds, dtype=np.float64)

    t_valid = t_orig[np.isfinite(t_orig)]
    n_original = int(t_valid.size)
    n_uniform = int(np.isfinite(grid).sum())

    if n_original >= 2:
        time_span_days = float((np.nanmax(t_valid) - np.nanmin(t_valid)) / SECONDS_PER_DAY)
    else:
        time_span_days = np.nan

    return {
        "resampled": bool(resampled),
        "input_irregular": bool(input_irregular),
        "used_cadence_seconds": float(used_cadence_seconds) if np.isfinite(used_cadence_seconds) else np.nan,
        "interpolation_method": str(interpolation_method),
        "n_original": n_original,
        "n_uniform": n_uniform,
        "time_span_days": time_span_days,
    }
