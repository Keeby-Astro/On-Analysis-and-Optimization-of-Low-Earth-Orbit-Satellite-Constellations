import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.widgets import Slider
from matplotlib.dates import date2num
from matplotlib.lines import Line2D
from time import perf_counter
from scipy.ndimage import gaussian_filter
import warnings
import os

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    njit = None
    _HAS_NUMBA = False

_NUMBA_DISABLED_ENV = {"0", "false", "no", "off"}
_USE_NUMBA = _HAS_NUMBA and str(os.getenv("ORBITAL_PLOT_USE_NUMBA", "1")).strip().lower() not in _NUMBA_DISABLED_ENV


def _preserve_open_figures_for_export():
    return str(os.getenv("ORBITAL_PLOT_PRESERVE_FIGURES_FOR_EXPORT", "0")).strip().lower() in {
        '1', 'true', 'yes', 'on'
    }

from circular_plot_utils import (angular_axis_ticks, circular_linear_kde, circular_linear_density,
                                 circular_pad_histogram_2d, duplicate_torus_points_for_display,
                                 stable_category_color_map, torus_kde_von_mises, wrap_degrees_180, wrap_degrees_360)


GEN1_INCLINATION_TARGETS = [53.05, 53.217, 70.0, 97.655]
DEFAULT_INCLINATION_FOCUS_WINDOWS = {'53': {'xlim': (18102.615902811503, 20565.8996289785),
                                            'ylim': (52.79907083519085, 53.39992871827338)},
                                     '70': {'xlim': (18102.615902811503, 20565.8996289785),
                                            'ylim': (69.9244939302158, 70.05002547302162)},
                                     '97': {'xlim': (18102.615902811503, 20565.8996289785),
                                            'ylim': (97.49888097408441, 97.69952065763928)}}

GEN1_INC_SMA_TARGET_PROFILES = {"53": [{"label": "53.2@540", "target_sma_km": 6918.137, "target_inc_deg": 53.2},
                                       {"label": "53.0@550", "target_sma_km": 6928.137, "target_inc_deg": 53.0}],
                                "70": [{"label": "70.0@570", "target_sma_km": 6948.137, "target_inc_deg": 70.0}],
                                "97": [{"label": "97.6@560", "target_sma_km": 6938.137, "target_inc_deg": 97.6}]}

DEFAULT_INC_SMA_FOCUS_WINDOWS = {
    "53": {
        "xlim": (6496.16535477994, 6959.739617062109),
        "ylim": (52.88392186965775, 53.30062601380588),
    },
    "70": {
        "xlim": (6496.16535477994, 6959.739617062109),
        "ylim": (69.935, 70.03992878988588),
    },
    "97": {
        "xlim": (6496.16535477994, 6959.739617062109),
        "ylim": (97.49956127615812, 97.6776),
    },
}


if _HAS_NUMBA:

    @njit(cache=True)
    def _unwrap_degrees_numba(values_deg):
        n = values_deg.size
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        out[0] = values_deg[0]
        offset = 0.0
        prev = values_deg[0]
        for i in range(1, n):
            cur = values_deg[i]
            delta = cur - prev
            if delta > 180.0:
                offset -= 360.0
            elif delta < -180.0:
                offset += 360.0
            out[i] = cur + offset
            prev = cur
        return out

    @njit(cache=True)
    def _negative_slope_point_mask_numba(t_days, y_vals, slope_threshold):
        n = t_days.size
        out = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            dt = t_days[i] - t_days[i - 1]
            if not np.isfinite(dt) or dt <= 0.0:
                continue
            y0 = y_vals[i - 1]
            y1 = y_vals[i]
            if not np.isfinite(y0) or not np.isfinite(y1):
                continue
            slope = (y1 - y0) / dt
            if np.isfinite(slope) and slope <= slope_threshold:
                out[i] = True
        return out

    @njit(cache=True)
    def _local_maxima_mask_numba(work, finite):
        n0, n1 = work.shape
        out = np.zeros((n0, n1), dtype=np.bool_)
        for i in range(n0):
            for j in range(n1):
                if not finite[i, j]:
                    continue
                v = work[i, j]
                is_peak = True
                for di in (-1, 0, 1):
                    ii = i + di
                    if ii < 0 or ii >= n0:
                        continue
                    for dj in (-1, 0, 1):
                        jj = j + dj
                        if di == 0 and dj == 0:
                            continue
                        if jj < 0 or jj >= n1:
                            continue
                        if v < work[ii, jj]:
                            is_peak = False
                            break
                    if not is_peak:
                        break
                out[i, j] = is_peak
        return out

else:
    _unwrap_degrees_numba = None
    _negative_slope_point_mask_numba = None
    _local_maxima_mask_numba = None


def _iter_object_index_groups(object_ids):
    ids = np.asarray(object_ids).astype(str)
    if ids.size == 0:
        return
    order = np.argsort(ids, kind='mergesort')
    sorted_ids = ids[order]
    unique_ids, start_idx = np.unique(sorted_ids, return_index=True)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = order.size
    for obj, s, e in zip(unique_ids.tolist(), start_idx.tolist(), end_idx.tolist()):
        yield str(obj), order[int(s):int(e)]

def _normalize_inc_sma_render_mode(mode):
    value = str(mode or "scatter").strip().lower()
    if value in {"scatter", "hexbin", "hist2d"}:
        return value
    return "scatter"

def _normalize_inc_sma_metric_mode(mode):
    value = str(mode or "euclidean").strip().lower()
    valid = {"euclidean", "standardized_euclidean", "mahalanobis",
             "nondimensional_constellation", "local_density_score"}
    if value in valid:
        return value
    return "euclidean"

def _normalize_inc_sma_target_profiles(target_profiles=None):
    def _parse_target_point(item, fallback_label):
        if not isinstance(item, dict):
            return None
        try:
            target_sma = float(item.get("target_sma_km"))
            target_inc = float(item.get("target_inc_deg"))
        except Exception:
            return None

        if not np.isfinite(target_sma) or not np.isfinite(target_inc):
            return None

        label = str(item.get("label") or fallback_label)
        return {"label": label, "target_sma_km": float(target_sma),
                "target_inc_deg": float(target_inc)}

    profiles = {}
    for family_key, points in GEN1_INC_SMA_TARGET_PROFILES.items():
        parsed_points = []
        for idx, point in enumerate(points):
            parsed = _parse_target_point(point, f"{family_key}_target_{idx + 1}")
            if parsed is not None:
                parsed_points.append(parsed)
        if parsed_points:
            profiles[str(family_key)] = parsed_points

    if target_profiles is None:
        return profiles

    if isinstance(target_profiles, dict):
        for family_key, family_points in target_profiles.items():
            candidate_points = family_points
            if isinstance(family_points, dict):
                if isinstance(family_points.get("targets"), (list, tuple)):
                    candidate_points = family_points.get("targets")
                elif isinstance(family_points.get("points"), (list, tuple)):
                    candidate_points = family_points.get("points")

            if not isinstance(candidate_points, (list, tuple)):
                continue

            parsed_points = []
            for idx, point in enumerate(candidate_points):
                parsed = _parse_target_point(point, f"{family_key}_target_{idx + 1}")
                if parsed is not None:
                    parsed_points.append(parsed)

            if parsed_points:
                profiles[str(family_key)] = parsed_points
        return profiles

    if isinstance(target_profiles, (list, tuple)):
        parsed_points = []
        for idx, point in enumerate(target_profiles):
            parsed = _parse_target_point(point, f"all_target_{idx + 1}")
            if parsed is not None:
                parsed_points.append(parsed)
        if parsed_points:
            profiles["all"] = parsed_points
    return profiles

def compute_inc_sma_reference_stats(x_vals, y_vals, target_points, sma_tolerance_km=25.0,
                                    inclination_tolerance_deg=0.4, assignment_mode="joint_nearest"):
    x_arr = np.asarray(x_vals, dtype=np.float64)
    y_arr = np.asarray(y_vals, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x_vals and y_vals must have the same shape")

    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[finite_mask]
    y_clean = y_arr[finite_mask]

    tol_sma = float(sma_tolerance_km)
    tol_inc = float(inclination_tolerance_deg)
    if not np.isfinite(tol_sma) or tol_sma <= 0.0:
        tol_sma = 25.0
    if not np.isfinite(tol_inc) or tol_inc <= 0.0:
        tol_inc = 0.4

    normalized_targets = []
    for idx, item in enumerate(target_points or []):
        if not isinstance(item, dict):
            continue
        try:
            target_sma = float(item.get("target_sma_km"))
            target_inc = float(item.get("target_inc_deg"))
        except Exception:
            continue
        if not np.isfinite(target_sma) or not np.isfinite(target_inc):
            continue
        normalized_targets.append({"label": str(item.get("label") or f"target_{idx + 1}"),
                                   "target_sma_km": float(target_sma),
                                   "target_inc_deg": float(target_inc)})

    if not normalized_targets:
        return {"assignment_mode_requested": str(assignment_mode), "assignment_mode_effective": "none",
                "sma_tolerance_km": float(tol_sma), "inclination_tolerance_deg": float(tol_inc),
                "joint_distance_threshold": float(np.sqrt(2.0)), "target_points": [], "groups": [],
                "assigned_count": 0, "unassigned_count": int(x_clean.size),
                "total_points_considered": int(x_clean.size), "omitted_targets": []}

    if x_clean.size == 0:
        return {"assignment_mode_requested": str(assignment_mode), "assignment_mode_effective": "joint_nearest",
                "sma_tolerance_km": float(tol_sma), "inclination_tolerance_deg": float(tol_inc),
                "joint_distance_threshold": float(np.sqrt(2.0)), "target_points": normalized_targets, "groups": [],
                "assigned_count": 0, "unassigned_count": 0, "total_points_considered": 0,
                "omitted_targets": [point["label"] for point in normalized_targets]}

    target_sma = np.asarray([point["target_sma_km"] for point in normalized_targets], dtype=np.float64)
    target_inc = np.asarray([point["target_inc_deg"] for point in normalized_targets], dtype=np.float64)

    dx_abs = np.abs(x_clean[:, None] - target_sma[None, :])
    dy_abs = np.abs(y_clean[:, None] - target_inc[None, :])
    dx_norm = dx_abs / float(tol_sma)
    dy_norm = dy_abs / float(tol_inc)
    joint_norm = np.hypot(dx_norm, dy_norm)

    nearest_idx = np.argmin(joint_norm, axis=1)
    row_idx = np.arange(x_clean.size, dtype=np.int64)
    best_dx_abs = dx_abs[row_idx, nearest_idx]
    best_dy_abs = dy_abs[row_idx, nearest_idx]
    best_joint_norm = joint_norm[row_idx, nearest_idx]

    assignment_mode_requested = str(assignment_mode or "joint_nearest").strip().lower()
    assignment_mode_effective = assignment_mode_requested
    if assignment_mode_effective not in {"joint_nearest", "both_tolerances"}:
        assignment_mode_effective = "joint_nearest"

    within_both = (best_dx_abs <= tol_sma) & (best_dy_abs <= tol_inc)
    joint_threshold = float(np.sqrt(2.0))
    within_joint = best_joint_norm <= joint_threshold
    if assignment_mode_effective == "both_tolerances":
        assigned = within_both
    else:
        assigned = within_both | within_joint

    groups = []
    omitted_targets = []
    for target_idx, point in enumerate(normalized_targets):
        in_group = (nearest_idx == target_idx) & assigned
        if not np.any(in_group):
            omitted_targets.append(point["label"])
            continue
        x_group = x_clean[in_group]
        y_group = y_clean[in_group]
        groups.append({"label": point["label"], "target_sma_km": float(point["target_sma_km"]),
                       "target_inc_deg": float(point["target_inc_deg"]), "n": int(x_group.size),
                       "mean_sma_km": float(np.nanmean(x_group)), "median_sma_km": float(np.nanmedian(x_group)),
                       "mean_inc_deg": float(np.nanmean(y_group)), "median_inc_deg": float(np.nanmedian(y_group))})

    assigned_count = int(np.sum(assigned))
    return {"assignment_mode_requested": assignment_mode_requested, "assignment_mode_effective": assignment_mode_effective,
            "sma_tolerance_km": float(tol_sma), "inclination_tolerance_deg": float(tol_inc), 
            "joint_distance_threshold": float(joint_threshold), "target_points": normalized_targets, "groups": groups,
            "assigned_count": assigned_count, "unassigned_count": int(x_clean.size - assigned_count), 
            "total_points_considered": int(x_clean.size), "omitted_targets": omitted_targets}

def _format_inc_sma_reference_annotation(reference_stats):
    groups = reference_stats.get("groups", [])
    if not groups:
        return "No target-assigned points"

    lines = []
    for group in groups:
        lines.append(f"{group['label']}: n={group['n']}")
        lines.append(f"  SMA mean/med: {group['mean_sma_km']:.3f} / {group['median_sma_km']:.3f} km")
        lines.append(f"  Inc mean/med: {group['mean_inc_deg']:.3f} / {group['median_inc_deg']:.3f} deg")
    return "\n".join(lines)

def _compute_inc_sma_metric_values(x_vals, y_vals, metric_mode="euclidean", standardized_robust_scale=False,
                                   nondim_sma_scale_km=25.0, nondim_inc_scale_deg=0.4):
    x_arr = np.asarray(x_vals, dtype=np.float64)
    y_arr = np.asarray(y_vals, dtype=np.float64)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x = x_arr[keep]
    y = y_arr[keep]

    requested_mode = _normalize_inc_sma_metric_mode(metric_mode)
    active_mode = requested_mode
    fallback_reason = None
    numerical_fallback = False

    if x.size == 0:
        return {"distances": np.asarray([], dtype=np.float64), "metric_mode_requested": requested_mode,
                "metric_mode_active": active_mode, "metric_label": "Euclidean distance",
                "numerical_fallback": False, "fallback_reason": None, "scales": {}}

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    dy = y - y_mean

    def _safe_scale(values, use_robust=False):
        arr = np.asarray(values, dtype=np.float64)
        if use_robust:
            med = float(np.nanmedian(arr))
            mad = float(np.nanmedian(np.abs(arr - med)))
            robust_scale = 1.4826 * mad
            if np.isfinite(robust_scale) and robust_scale > 1.0e-12:
                return float(robust_scale), "mad"
        std_scale = float(np.nanstd(arr))
        if np.isfinite(std_scale) and std_scale > 1.0e-12:
            return float(std_scale), "std"
        if not use_robust:
            med = float(np.nanmedian(arr))
            mad = float(np.nanmedian(np.abs(arr - med)))
            robust_scale = 1.4826 * mad
            if np.isfinite(robust_scale) and robust_scale > 1.0e-12:
                return float(robust_scale), "mad"
        return 1.0, "unit_fallback"

    try:
        if requested_mode == "standardized_euclidean":
            sx, sx_mode = _safe_scale(x, use_robust=bool(standardized_robust_scale))
            sy, sy_mode = _safe_scale(y, use_robust=bool(standardized_robust_scale))
            distances = np.hypot(dx / sx, dy / sy)
            metric_label = "Standardized Euclidean distance"
            scales = {"sma_scale_km": float(sx), "inc_scale_deg": float(sy),
                      "sma_scale_mode": sx_mode, "inc_scale_mode": sy_mode}
        elif requested_mode == "mahalanobis":
            stacked = np.column_stack((x, y))
            centered = stacked - np.array([x_mean, y_mean], dtype=np.float64)
            cov = np.cov(stacked, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            q = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
            if np.any(~np.isfinite(q)):
                raise FloatingPointError("non-finite quadratic form in mahalanobis")
            distances = np.sqrt(np.maximum(q, 0.0))
            metric_label = "Mahalanobis distance"
            scales = {"covariance_matrix": cov.tolist()}
        elif requested_mode == "nondimensional_constellation":
            scale_sma = float(nondim_sma_scale_km)
            scale_inc = float(nondim_inc_scale_deg)
            if not np.isfinite(scale_sma) or scale_sma <= 0.0:
                scale_sma = 25.0
            if not np.isfinite(scale_inc) or scale_inc <= 0.0:
                scale_inc = 0.4
            distances = np.hypot(dx / scale_sma, dy / scale_inc)
            metric_label = "Nondimensional constellation distance"
            scales = {"sma_scale_km": float(scale_sma), "inc_scale_deg": float(scale_inc),
                      "definition": "sqrt((delta_sma/sma_scale)^2 + (delta_inc/inc_scale)^2)"}
        elif requested_mode == "local_density_score":
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            y_min = float(np.min(y))
            y_max = float(np.max(y))
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            x_bins = np.linspace(x_min, x_max, 51)
            y_bins = np.linspace(y_min, y_max, 51)
            hist, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
            ix = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, hist.shape[0] - 1)
            iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, hist.shape[1] - 1)
            counts = hist[ix, iy]
            max_count = float(np.max(counts)) if counts.size > 0 else 1.0
            distances = max_count - counts
            metric_label = "Local density inverse score"
            scales = {}
        else:
            distances = np.hypot(dx, dy)
            metric_label = "Euclidean distance"
            scales = {}
    except Exception as exc:
        active_mode = "euclidean"
        metric_label = "Euclidean distance"
        distances = np.hypot(dx, dy)
        numerical_fallback = True
        fallback_reason = str(exc)
        scales = {}

    return {"distances": np.asarray(distances, dtype=np.float64), "metric_mode_requested": requested_mode,
            "metric_mode_active": active_mode, "metric_label": metric_label,
            "numerical_fallback": bool(numerical_fallback), "fallback_reason": fallback_reason, "scales": scales}

def _normalize_time_render_mode(mode):
    value = str(mode or 'scatter').strip().lower()
    if value in {'scatter', 'hexbin_time', 'hist2d_time'}:
        return value
    return 'scatter'

def _remove_artist_safe(artist):
    if artist is None:
        return
    try:
        artist.remove()
    except Exception:
        pass

def _clean_timeseries_xy(x_vals, y_vals):
    x_arr = np.asarray(pd.to_datetime(np.asarray(x_vals), errors='coerce'), dtype='datetime64[ns]')
    y_arr = np.asarray(y_vals, dtype=np.float64)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Time and value arrays must have the same length")
    keep = (~np.isnat(x_arr)) & np.isfinite(y_arr)
    x_clean = pd.to_datetime(x_arr[keep])
    y_clean = y_arr[keep]
    return x_clean, y_clean

def _create_timeseries_artist(ax, x_vals, y_vals,
                              render_mode='scatter',
                              color='r',
                              marker_size=4.0,
                              dense_min_points=120):
    mode = _normalize_time_render_mode(render_mode)
    x_clean, y_clean = _clean_timeseries_xy(x_vals, y_vals)

    effective_mode = mode
    if mode != 'scatter' and y_clean.size < int(max(1, dense_min_points)):
        effective_mode = 'scatter'

    if effective_mode == 'scatter':
        artist = ax.scatter(x_clean, y_clean, c=color, s=marker_size, linewidths=0, rasterized=True)
    elif effective_mode == 'hexbin_time':
        artist = ax.hexbin(date2num(x_clean), y_clean, gridsize=(90, 60), mincnt=1,
                           cmap='viridis', rasterized=True)
    else:
        hist = ax.hist2d(date2num(x_clean), y_clean,  bins=[90, 60], cmin=1,
                         cmap='viridis', rasterized=True)
        artist = hist[-1]

    return {'artist': artist, 'requested_mode': mode, 'effective_mode': effective_mode,
            'sparse_fallback': bool(effective_mode != mode), 'point_count': int(y_clean.size)}

def _normalize_fit_mode(mode):
    value = str(mode or 'ols').strip().lower()
    if value in {'ols', 'huber'}:
        return value
    return 'ols'

def _fit_line_ols_huber(x_vals, y_vals, fit_mode='ols', max_iter=50, huber_c=1.345):
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return None

    mode = _normalize_fit_mode(fit_mode)
    if mode == 'ols':
        coeff = np.polyfit(x, y, deg=1)
        return {
            'slope': float(coeff[0]),
            'intercept': float(coeff[1]),
            'fit_mode': 'ols',
        }

    X = np.column_stack((x, np.ones_like(x)))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(int(max_iter)):
        resid = y - X @ beta
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        scale = 1.4826 * mad
        if (not np.isfinite(scale)) or scale <= 1.0e-12:
            break
        u = resid / scale
        w = np.ones_like(u)
        large = np.abs(u) > float(huber_c)
        w[large] = float(huber_c) / np.maximum(np.abs(u[large]), 1.0e-12)
        Xw = X * w[:, None]
        yw = y * w
        beta_new = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < 1.0e-10:
            beta = beta_new
            break
        beta = beta_new

    return {
        'slope': float(beta[0]),
        'intercept': float(beta[1]),
        'fit_mode': 'huber',
    }


def _compute_rolling_time_summary(x_vals, y_vals, window_days=30.0, lower_q=0.10, upper_q=0.90, min_periods=5):
    x = pd.to_datetime(np.asarray(x_vals), errors='coerce')
    y = np.asarray(y_vals, dtype=np.float64)
    keep = (~pd.isna(x)) & np.isfinite(y)
    if np.sum(keep) < max(3, int(min_periods)):
        return None

    x_keep = pd.to_datetime(x[keep])
    y_keep = y[keep]
    order = np.argsort(x_keep.view('int64'), kind='mergesort')
    x_sorted = x_keep[order]
    y_sorted = y_keep[order]

    try:
        win_days = float(window_days)
    except Exception:
        win_days = 30.0
    if not np.isfinite(win_days) or win_days <= 0.0:
        win_days = 30.0

    series = pd.Series(y_sorted, index=pd.DatetimeIndex(x_sorted))
    rolling = series.rolling(f"{win_days:.6f}D", min_periods=max(2, int(min_periods)))
    median = rolling.median()
    low = rolling.quantile(float(lower_q))
    high = rolling.quantile(float(upper_q))

    valid = np.isfinite(median.values) & np.isfinite(low.values) & np.isfinite(high.values)
    if np.sum(valid) < 2:
        return None

    return {
        'x': median.index[valid],
        'median': median.values[valid],
        'low': low.values[valid],
        'high': high.values[valid],
        'window_days': float(win_days),
    }


def _compute_objectwise_unwrapped_angles(times, object_ids, angles_deg, min_points_for_unwrap=3):
    ts = pd.to_datetime(np.asarray(times), errors='coerce')
    ts_ns = ts.view('int64')
    valid_time = ~pd.isna(ts)
    angles = wrap_degrees_360(np.asarray(angles_deg, dtype=np.float64))
    out = np.full(angles.shape, np.nan, dtype=np.float64)

    unwrap_objects = 0
    short_tracks = 0
    min_points = int(max(2, min_points_for_unwrap))
    for _, obj_idx in _iter_object_index_groups(object_ids):
        if obj_idx.size == 0:
            continue
        keep = valid_time[obj_idx] & np.isfinite(angles[obj_idx])
        if np.sum(keep) == 0:
            continue
        idx_keep = obj_idx[keep]
        sort_order = np.argsort(ts_ns[idx_keep], kind='mergesort')
        idx_sorted = idx_keep[sort_order]
        a_sorted = angles[idx_sorted]

        if idx_sorted.size < min_points:
            out[idx_sorted] = a_sorted
            short_tracks += 1
            continue

        if _USE_NUMBA and _unwrap_degrees_numba is not None and idx_sorted.size >= 4:
            unwrapped = _unwrap_degrees_numba(np.asarray(a_sorted, dtype=np.float64))
        else:
            unwrapped = np.rad2deg(np.unwrap(np.deg2rad(a_sorted)))
        out[idx_sorted] = unwrapped
        unwrap_objects += 1

    return out, {
        'objects_unwrapped': int(unwrap_objects),
        'objects_short_tracks': int(short_tracks),
    }


def _compute_objectwise_residual_angles(times, object_ids, unwrapped_deg, fit_mode='huber', min_points=3):
    ts = pd.to_datetime(np.asarray(times), errors='coerce')
    ts_ns = ts.view('int64')
    valid_time = ~pd.isna(ts)
    y_all = np.asarray(unwrapped_deg, dtype=np.float64)
    residual = np.full(y_all.shape, np.nan, dtype=np.float64)

    fit_count = 0
    min_keep = max(2, int(min_points))
    for _, obj_idx in _iter_object_index_groups(object_ids):
        if obj_idx.size == 0:
            continue
        keep = valid_time[obj_idx] & np.isfinite(y_all[obj_idx])
        if np.sum(keep) < min_keep:
            continue

        idx_keep = obj_idx[keep]
        sort_order = np.argsort(ts_ns[idx_keep], kind='mergesort')
        idx_sorted = idx_keep[sort_order]
        t_ns_sorted = ts_ns[idx_sorted].astype(np.float64)
        y_sorted = y_all[idx_sorted]

        t_days = (t_ns_sorted - float(t_ns_sorted[0])) / (86400.0 * 1.0e9)
        fit = _fit_line_ols_huber(t_days, y_sorted, fit_mode=fit_mode)
        if fit is None:
            continue
        y_fit = fit['slope'] * t_days + fit['intercept']
        residual[idx_sorted] = y_sorted - y_fit
        fit_count += 1

    return residual, {
        'objects_fitted': int(fit_count),
        'fit_mode': _normalize_fit_mode(fit_mode),
    }


def _estimate_objectwise_drift_metadata(times, object_ids, unwrapped_deg, window_days=30.0):
    ts = pd.to_datetime(np.asarray(times), errors='coerce')
    ts_ns = ts.view('int64')
    valid_time = ~pd.isna(ts)
    y_all = np.asarray(unwrapped_deg, dtype=np.float64)

    drift_values = []
    try:
        win_days = float(window_days)
    except Exception:
        win_days = 30.0
    if not np.isfinite(win_days) or win_days <= 0.0:
        win_days = 30.0

    for _, obj_idx in _iter_object_index_groups(object_ids):
        keep = valid_time[obj_idx] & np.isfinite(y_all[obj_idx])
        if np.sum(keep) < 3:
            continue
        idx_keep = obj_idx[keep]
        order = np.argsort(ts_ns[idx_keep], kind='mergesort')
        t_sorted = pd.to_datetime(ts_ns[idx_keep][order], unit='ns', errors='coerce')
        y_sorted = y_all[idx_keep][order]

        summary = _compute_rolling_time_summary(t_sorted, y_sorted, window_days=win_days, lower_q=0.25, upper_q=0.75, min_periods=3)
        if summary is None or len(summary['median']) < 2:
            continue
        x_roll = pd.to_datetime(summary['x'])
        y_roll = np.asarray(summary['median'], dtype=np.float64)
        dt_days = (x_roll.view('int64').astype(np.float64) - float(x_roll.view('int64')[0])) / (86400.0 * 1.0e9)
        fit = _fit_line_ols_huber(dt_days, y_roll, fit_mode='ols')
        if fit is not None:
            drift_values.append(float(fit['slope']))

    if len(drift_values) == 0:
        return {
            'window_days': float(win_days),
            'object_count': 0,
            'median_drift_deg_per_day': np.nan,
            'p10_drift_deg_per_day': np.nan,
            'p90_drift_deg_per_day': np.nan,
        }

    drift_arr = np.asarray(drift_values, dtype=np.float64)
    return {
        'window_days': float(win_days),
        'object_count': int(drift_arr.size),
        'median_drift_deg_per_day': float(np.nanmedian(drift_arr)),
        'p10_drift_deg_per_day': float(np.nanquantile(drift_arr, 0.10)),
        'p90_drift_deg_per_day': float(np.nanquantile(drift_arr, 0.90)),
    }


def _detect_negative_slope_segments(
    times,
    object_ids,
    y_vals,
    threshold_per_day=-1.0e-5,
    min_duration_days=7.0,
    min_value=1.8e-3,
    min_points=3,
):
    ts = pd.to_datetime(np.asarray(times), errors='coerce')
    ts_ns = ts.view('int64')
    valid_time = ~pd.isna(ts)
    y = np.asarray(y_vals, dtype=np.float64)
    highlight_mask = np.zeros(y.shape, dtype=bool)
    segments = []

    try:
        slope_threshold = float(threshold_per_day)
    except Exception:
        slope_threshold = -1.0e-5
    if not np.isfinite(slope_threshold):
        slope_threshold = -1.0e-5

    try:
        min_duration = float(min_duration_days)
    except Exception:
        min_duration = 7.0
    if not np.isfinite(min_duration) or min_duration < 0.0:
        min_duration = 7.0

    try:
        min_ecc_value = float(min_value)
    except Exception:
        min_ecc_value = 1.8e-3
    if not np.isfinite(min_ecc_value) or min_ecc_value < 0.0:
        min_ecc_value = 1.8e-3

    for obj, obj_idx in _iter_object_index_groups(object_ids):
        keep = valid_time[obj_idx] & np.isfinite(y[obj_idx])
        if np.sum(keep) < max(2, int(min_points)):
            continue

        idx_keep = obj_idx[keep]
        order = np.argsort(ts_ns[idx_keep], kind='mergesort')
        idx_sorted = idx_keep[order]
        t_ns_sorted = ts_ns[idx_sorted]
        y_sorted = y[idx_sorted]

        t_days = t_ns_sorted.astype(np.float64) / (86400.0 * 1.0e9)
        if _USE_NUMBA and _negative_slope_point_mask_numba is not None and idx_sorted.size >= 3:
            point_desc = _negative_slope_point_mask_numba(t_days, y_sorted, float(slope_threshold))
        else:
            dt = np.diff(t_days)
            dy = np.diff(y_sorted)
            slope = np.full_like(dt, np.nan)
            valid_dt = np.isfinite(dt) & (dt > 0.0)
            slope[valid_dt] = dy[valid_dt] / dt[valid_dt]
            edge_desc = np.isfinite(slope) & (slope <= slope_threshold)
            point_desc = np.zeros(y_sorted.shape, dtype=bool)
            point_desc[1:] = edge_desc

        # Keep descent highlights focused on physically meaningful elevated-e regimes.
        point_desc &= np.isfinite(y_sorted) & (y_sorted >= float(min_ecc_value))

        if not np.any(point_desc):
            continue

        start = None
        for i, flag in enumerate(point_desc.tolist() + [False]):
            if flag and start is None:
                start = i
            if (not flag) and (start is not None):
                end = i - 1
                n_pts = end - start + 1
                duration = float(t_days[end] - t_days[start]) if n_pts > 1 else 0.0
                if n_pts >= int(min_points) and duration >= float(min_duration):
                    highlight_mask[idx_sorted[start:end + 1]] = True
                    segments.append(
                        {
                            'object_id': str(obj),
                            'start_index': int(idx_sorted[start]),
                            'end_index': int(idx_sorted[end]),
                            'n_points': int(n_pts),
                            'duration_days': float(duration),
                            'threshold_per_day': float(slope_threshold),
                            'min_value': float(min_ecc_value),
                        }
                    )
                start = None

    return highlight_mask, {
        'threshold_per_day': float(slope_threshold),
        'min_duration_days': float(min_duration),
        'min_value': float(min_ecc_value),
        'segment_count': int(len(segments)),
        'segments': segments,
    }


def compute_inclination_reference_stats(y_vals, target_inclinations,
                                        assignment_tolerance_deg=0.4):
    y_arr = np.asarray(y_vals, dtype=np.float64)
    y_arr = y_arr[np.isfinite(y_arr)]

    targets = []
    for item in np.asarray(target_inclinations if target_inclinations is not None else [], dtype=np.float64):
        if np.isfinite(item):
            targets.append(float(item))
    targets = sorted(set(targets))

    tol = float(assignment_tolerance_deg)
    if not np.isfinite(tol):
        tol = 0.4
    tol = max(0.0, tol)

    if len(targets) == 0:
        return {
            'groups': [],
            'target_inclinations_deg': [],
            'assignment_tolerance_deg': tol,
            'assigned_count': 0,
            'unassigned_count': int(y_arr.size),
            'omitted_targets': [],
        }

    if y_arr.size == 0:
        return {
            'groups': [],
            'target_inclinations_deg': [float(t) for t in targets],
            'assignment_tolerance_deg': tol,
            'assigned_count': 0,
            'unassigned_count': 0,
            'omitted_targets': [float(t) for t in targets],
        }

    target_arr = np.asarray(targets, dtype=np.float64)
    distances = np.abs(y_arr[:, None] - target_arr[None, :])
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(y_arr.size), nearest_idx]
    assigned = nearest_dist <= tol

    groups = []
    omitted_targets = []
    for idx, target in enumerate(target_arr.tolist()):
        vals = y_arr[(nearest_idx == idx) & assigned]
        n = int(vals.size)
        if n == 0:
            omitted_targets.append(float(target))
            continue
        groups.append({
            'target_deg': float(target),
            'n': n,
            'mean_deg': float(np.nanmean(vals)),
            'median_deg': float(np.nanmedian(vals)),
            'min_deg': float(np.nanmin(vals)),
            'max_deg': float(np.nanmax(vals)),
        })

    assigned_count = int(np.sum(assigned))
    return {
        'groups': groups,
        'target_inclinations_deg': [float(t) for t in targets],
        'assignment_tolerance_deg': tol,
        'assigned_count': assigned_count,
        'unassigned_count': int(y_arr.size - assigned_count),
        'omitted_targets': omitted_targets,
    }


def _format_inclination_reference_annotation(reference_stats):
    groups = reference_stats.get('groups', [])
    if not groups:
        return "No target-assigned points"

    lines = []
    for group in sorted(groups, key=lambda item: item.get('target_deg', 0.0)):
        lines.append(
            f"{group['target_deg']:.3f}°: n={group['n']}"
        )
    return "\n".join(lines)

def orbital_elements_plot(inclinations, semi_major_axes, right_ascensions, args_of_perigee, eccentricities,
                           true_anomalies, timestamps, fileNames, filenames_array,
                           phase_mode=None, phase_series=None, altitude_series=None,
                           shell_series=None, low_ecc_mask=None,
                           show_plots=True, return_figures=False, return_results=False,
                           layout_mode='separate', decimation_max_points=None,
                           time_window=None, common_epoch_records=None,
                           phase_alt_mode='scatter',
                           inclination_time_render_mode='scatter',
                           inclination_reference_lines=True,
                           inclination_reference_assignment_tolerance_deg=0.4,
                           inclination_reference_annotation=False,
                           inclination_target_inclinations=None,
                           inclination_focused_views=True,
                           inclination_focus_windows=None,
                           raan_time_render_mode='scatter',
                           raan_display_mode='wrapped_scatter',
                           raan_residual_fit_mode='huber',
                           raan_residual_annotation=False,
                           raan_min_points_for_unwrap=3,
                           sma_time_render_mode='scatter',
                           sma_reference_lines=True,
                           sma_reference_values_km=None,
                           sma_rolling_window_days=30,
                           sma_show_envelope=True,
                           sma_operational_bands=None,
                           argp_time_render_mode='scatter',
                           argp_display_mode='raw_all',
                           argp_ecc_floor=1.0e-3,
                           argp_low_e_behavior='suppress',
                           argp_show_argument_of_latitude_companion=False,
                           argument_of_latitude_series=None,
                           eccentricity_time_render_mode='scatter',
                           eccentricity_highlight_descents=True,
                           eccentricity_descent_threshold_per_day=-5.0e-6,
                           eccentricity_descent_min_duration_days=7.0,
                           eccentricity_rolling_window_days=30,
                           eccentricity_show_envelope=True,
                           eccentricity_zoom_ylim=None,
                           eccentricity_descent_min_eccentricity=1.8e-3,
                           enable_common_epoch_shell_snapshot=False):
    """
    Create interactive plots for the orbital elements of the satellites.

    Parameters:
        inclinations (list): The inclinations of the satellites in degrees.
        semi_major_axes (list): The semi-major axes of the satellites in km.
        right_ascensions (list): The right ascensions of the satellites in degrees.
        args_of_perigee (list): The arguments of perigee of the satellites in degrees.
        eccentricities (list): The eccentricities of the satellites.
        true_anomalies (list): The true anomalies of the satellites in degrees.
        timestamps (list): The timestamps of the TLE data.
        fileNames (list): The names of the files containing the TLE data.
        filenames_array (np.array): The file index for each satellite.

    Returns:
        None

    Migration Notes:
        Existing six orbital-element time-series plots are unchanged by default.
        Optional phase/altitude/shell inputs add an extra additive view for
        low-e-safe analysis without removing legacy outputs.
    """
    t0 = perf_counter()
    inclination_time_render_mode = _normalize_time_render_mode(inclination_time_render_mode)
    raan_time_render_mode = _normalize_time_render_mode(raan_time_render_mode)
    sma_time_render_mode = _normalize_time_render_mode(sma_time_render_mode)
    argp_time_render_mode = _normalize_time_render_mode(argp_time_render_mode)
    eccentricity_time_render_mode = _normalize_time_render_mode(eccentricity_time_render_mode)

    raan_mode_requested = str(raan_display_mode or 'wrapped_scatter').strip().lower()
    if raan_mode_requested not in {'wrapped_scatter', 'wrapped_density', 'unwrapped_by_object', 'residual_by_object'}:
        raan_mode_requested = 'wrapped_scatter'
    raan_residual_fit_mode = _normalize_fit_mode(raan_residual_fit_mode)
    try:
        raan_min_points_for_unwrap = int(raan_min_points_for_unwrap)
    except Exception:
        raan_min_points_for_unwrap = 3
    raan_min_points_for_unwrap = max(2, int(raan_min_points_for_unwrap))

    argp_display_mode = str(argp_display_mode or 'raw_all').strip().lower()
    if argp_display_mode not in {'raw_all', 'ecc_filtered', 'split_validity', 'arglat_companion'}:
        argp_display_mode = 'raw_all'
    argp_mode_requested = str(argp_low_e_behavior or 'suppress').strip().lower()
    if argp_mode_requested not in {'suppress', 'highlight', 'split'}:
        argp_mode_requested = 'suppress'
    try:
        argp_ecc_floor = float(argp_ecc_floor)
    except Exception:
        argp_ecc_floor = 1.0e-3
    if not np.isfinite(argp_ecc_floor) or argp_ecc_floor < 0.0:
        argp_ecc_floor = 1.0e-3

    try:
        sma_rolling_window_days = float(sma_rolling_window_days)
    except Exception:
        sma_rolling_window_days = 30.0
    if not np.isfinite(sma_rolling_window_days) or sma_rolling_window_days <= 0.0:
        sma_rolling_window_days = 30.0

    try:
        eccentricity_rolling_window_days = float(eccentricity_rolling_window_days)
    except Exception:
        eccentricity_rolling_window_days = 30.0
    if not np.isfinite(eccentricity_rolling_window_days) or eccentricity_rolling_window_days <= 0.0:
        eccentricity_rolling_window_days = 30.0

    try:
        eccentricity_descent_threshold_per_day = float(eccentricity_descent_threshold_per_day)
    except Exception:
        eccentricity_descent_threshold_per_day = -1.0e-5
    if not np.isfinite(eccentricity_descent_threshold_per_day):
        eccentricity_descent_threshold_per_day = -1.0e-5

    try:
        eccentricity_descent_min_duration_days = float(eccentricity_descent_min_duration_days)
    except Exception:
        eccentricity_descent_min_duration_days = 7.0
    if not np.isfinite(eccentricity_descent_min_duration_days) or eccentricity_descent_min_duration_days < 0.0:
        eccentricity_descent_min_duration_days = 7.0

    try:
        eccentricity_descent_min_eccentricity = float(eccentricity_descent_min_eccentricity)
    except Exception:
        eccentricity_descent_min_eccentricity = 1.8e-3
    if not np.isfinite(eccentricity_descent_min_eccentricity) or eccentricity_descent_min_eccentricity < 0.0:
        eccentricity_descent_min_eccentricity = 1.8e-3

    eccentricity_zoom_limits = None
    if isinstance(eccentricity_zoom_ylim, (list, tuple)) and len(eccentricity_zoom_ylim) == 2:
        try:
            zoom_lo = float(eccentricity_zoom_ylim[0])
            zoom_hi = float(eccentricity_zoom_ylim[1])
        except Exception:
            zoom_lo, zoom_hi = np.nan, np.nan
        if np.isfinite(zoom_lo) and np.isfinite(zoom_hi):
            eccentricity_zoom_limits = [float(zoom_lo), float(zoom_hi)]

    if sma_reference_values_km is None:
        sma_reference_values = [6918.137, 6928.137, 6938.137, 6948.137]
    else:
        sma_reference_values = [
            float(v)
            for v in np.asarray(sma_reference_values_km, dtype=np.float64)
            if np.isfinite(v)
        ]
    sma_reference_values = sorted(set(sma_reference_values))

    normalized_sma_bands = []
    band_palette = ['#dff3e3', '#fef3c7', '#fde2e2']
    if isinstance(sma_operational_bands, dict):
        for idx, (label, span) in enumerate(sma_operational_bands.items()):
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            try:
                lo = float(span[0])
                hi = float(span[1])
            except Exception:
                continue
            if np.isfinite(lo) and np.isfinite(hi):
                normalized_sma_bands.append(
                    {
                        'label': str(label),
                        'min_km': min(lo, hi),
                        'max_km': max(lo, hi),
                        'color': band_palette[idx % len(band_palette)],
                        'alpha': 0.18,
                    }
                )
    elif isinstance(sma_operational_bands, (list, tuple)):
        for idx, band in enumerate(sma_operational_bands):
            if not isinstance(band, dict):
                continue
            try:
                lo = float(band.get('min_km'))
                hi = float(band.get('max_km'))
            except Exception:
                continue
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            normalized_sma_bands.append(
                {
                    'label': str(band.get('label', f'band_{idx + 1}')),
                    'min_km': min(lo, hi),
                    'max_km': max(lo, hi),
                    'color': str(band.get('color', band_palette[idx % len(band_palette)])),
                    'alpha': float(band.get('alpha', 0.18)),
                }
            )
    try:
        inclination_reference_assignment_tolerance_deg = float(inclination_reference_assignment_tolerance_deg)
    except Exception:
        inclination_reference_assignment_tolerance_deg = 0.4
    if not np.isfinite(inclination_reference_assignment_tolerance_deg):
        inclination_reference_assignment_tolerance_deg = 0.4
    inclination_reference_assignment_tolerance_deg = max(0.0, inclination_reference_assignment_tolerance_deg)

    configured_inclination_targets = []
    for item in np.asarray(
        GEN1_INCLINATION_TARGETS if inclination_target_inclinations is None else inclination_target_inclinations,
        dtype=np.float64,
    ):
        if np.isfinite(item):
            configured_inclination_targets.append(float(item))
    configured_inclination_targets = sorted(set(configured_inclination_targets))
    if not configured_inclination_targets:
        configured_inclination_targets = list(GEN1_INCLINATION_TARGETS)

    focus_windows = {
        key: {
            'xlim': tuple(value['xlim']),
            'ylim': tuple(value['ylim']),
        }
        for key, value in DEFAULT_INCLINATION_FOCUS_WINDOWS.items()
    }
    if isinstance(inclination_focus_windows, dict):
        for key, value in inclination_focus_windows.items():
            if key not in focus_windows or not isinstance(value, dict):
                continue
            xlim = value.get('xlim', focus_windows[key]['xlim'])
            ylim = value.get('ylim', focus_windows[key]['ylim'])
            if isinstance(xlim, (list, tuple)) and len(xlim) == 2:
                focus_windows[key]['xlim'] = (float(xlim[0]), float(xlim[1]))
            if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
                focus_windows[key]['ylim'] = (float(ylim[0]), float(ylim[1]))

    inclination_metadata = {
        'active_time_render_mode': str(inclination_time_render_mode),
        'target_inclinations_deg_default': [float(v) for v in configured_inclination_targets],
        'assignment_tolerance_deg': float(inclination_reference_assignment_tolerance_deg),
        'reference_lines_enabled': bool(inclination_reference_lines),
        'reference_annotation_enabled': bool(inclination_reference_annotation),
        'focused_views_enabled': bool(inclination_focused_views),
        'figures': {},
    }
    raan_metadata = {
        'requested_time_render_mode': str(raan_time_render_mode),
        'requested_display_mode': str(raan_mode_requested),
        'residual_fit_mode': str(raan_residual_fit_mode),
        'residual_annotation': bool(raan_residual_annotation),
        'min_points_for_unwrap': int(raan_min_points_for_unwrap),
        'figures': {},
    }
    sma_metadata = {
        'requested_time_render_mode': str(sma_time_render_mode),
        'reference_lines_enabled': bool(sma_reference_lines),
        'reference_values_km': [float(v) for v in sma_reference_values],
        'rolling_window_days': float(sma_rolling_window_days),
        'show_envelope': bool(sma_show_envelope),
        'operational_bands': normalized_sma_bands,
        'figures': {},
    }
    argp_metadata = {
        'requested_time_render_mode': str(argp_time_render_mode),
        'display_mode_requested': str(argp_display_mode),
        'ecc_floor': float(argp_ecc_floor),
        'low_e_behavior': str(argp_mode_requested),
        'show_argument_of_latitude_companion': bool(argp_show_argument_of_latitude_companion),
        'figures': {},
    }
    eccentricity_metadata = {
        'requested_time_render_mode': str(eccentricity_time_render_mode),
        'highlight_descents': bool(eccentricity_highlight_descents),
        'descent_threshold_per_day': float(eccentricity_descent_threshold_per_day),
        'descent_min_duration_days': float(eccentricity_descent_min_duration_days),
        'descent_min_eccentricity': float(eccentricity_descent_min_eccentricity),
        'rolling_window_days': float(eccentricity_rolling_window_days),
        'show_envelope': bool(eccentricity_show_envelope),
        'zoom_ylim': eccentricity_zoom_limits,
        'figures': {},
    }
    gui_data = {'timestamps': np.asarray(timestamps),
                'inclinations': np.asarray(inclinations),
                'semi_major_axes': np.asarray(semi_major_axes),
                'right_ascensions': np.asarray(right_ascensions),
                'args_of_perigee': np.asarray(args_of_perigee),
                'eccentricities': np.asarray(eccentricities),
                'true_anomalies': np.asarray(true_anomalies),
                'filenames_array': np.asarray(filenames_array)}

    if phase_series is not None:
        phase_arr = np.asarray(phase_series)
        if phase_arr.shape != gui_data['true_anomalies'].shape:
            raise ValueError("phase_series must match true_anomalies shape")
        gui_data['phase_series'] = phase_arr

    if altitude_series is not None:
        alt_arr = np.asarray(altitude_series)
        if alt_arr.shape != gui_data['true_anomalies'].shape:
            raise ValueError("altitude_series must match true_anomalies shape")
        gui_data['altitude_series'] = alt_arr

    if shell_series is not None:
        shell_arr = np.asarray(shell_series)
        if shell_arr.shape != gui_data['true_anomalies'].shape:
            raise ValueError("shell_series must match true_anomalies shape")
        gui_data['shell_series'] = shell_arr

    if low_ecc_mask is not None:
        low_e_arr = np.asarray(low_ecc_mask)
        if low_e_arr.shape != gui_data['true_anomalies'].shape:
            raise ValueError("low_ecc_mask must match true_anomalies shape")
        gui_data['low_ecc_mask'] = low_e_arr

    if argument_of_latitude_series is not None:
        arglat_arr = np.asarray(argument_of_latitude_series)
        if arglat_arr.shape != gui_data['true_anomalies'].shape:
            raise ValueError("argument_of_latitude_series must match true_anomalies shape")
        gui_data['argument_of_latitude_series'] = arglat_arr

    gui_data['timestamps_dt'] = pd.to_datetime(gui_data['timestamps'])

    plotted_phase_variable = 'true_anomaly_deg'
    if phase_mode is not None and 'phase_series' in gui_data:
        plotted_phase_variable = 'true_anomaly_deg'

    if time_window is not None:
        if not isinstance(time_window, (tuple, list)) or len(time_window) != 2:
            raise ValueError('time_window must be a (start, end) tuple/list when provided')
        t_start = pd.to_datetime(time_window[0])
        t_end = pd.to_datetime(time_window[1])
        keep = (gui_data['timestamps_dt'] >= t_start) & (gui_data['timestamps_dt'] <= t_end)
        if np.any(keep):
            for key in list(gui_data.keys()):
                if key == 'timestamps_dt':
                    continue
                gui_data[key] = gui_data[key][keep]
            gui_data['timestamps_dt'] = gui_data['timestamps_dt'][keep]

    display_names = list(fileNames)
    if 'All Files' not in display_names:
        display_names.append('All Files')

    order = np.argsort(gui_data['filenames_array'], kind='mergesort')
    sorted_names = gui_data['filenames_array'][order]
    unique_names, start_idx = np.unique(sorted_names, return_index=True)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = order.size
    bounds = {name: (int(s), int(e))
              for name, s, e in zip(unique_names.tolist(), start_idx.tolist(), end_idx.tolist())}
    all_indices = np.arange(gui_data['filenames_array'].size, dtype=np.int64)

    resolved_decimation_max_points = decimation_max_points
    auto_decimation_applied = False
    auto_decimation_enabled = str(os.getenv("ORBITAL_PLOT_ENABLE_AUTO_DECIMATION", "0")).strip().lower() in {
        '1', 'true', 'yes', 'on'
    }
    if resolved_decimation_max_points is None and auto_decimation_enabled:
        try:
            auto_trigger = int(os.getenv("ORBITAL_PLOT_AUTO_DECIMATION_TRIGGER", "250000"))
        except Exception:
            auto_trigger = 250000
        try:
            auto_cap = int(os.getenv("ORBITAL_PLOT_AUTO_DECIMATION_MAX_POINTS", "120000"))
        except Exception:
            auto_cap = 120000
        if gui_data['filenames_array'].size > max(0, auto_trigger) and auto_cap > 0:
            resolved_decimation_max_points = int(auto_cap)
            auto_decimation_applied = True
            print(
                f"[orbital_elements_plot] Auto-decimation enabled: "
                f"{gui_data['filenames_array'].size:,} points -> max {resolved_decimation_max_points:,} per frame"
            )

    def get_indices(selected_name):
        if selected_name == 'All Files':
            return all_indices
        window = bounds.get(selected_name)
        if window is None:
            return np.empty(0, dtype=np.int64)
        s, e = window
        return order[s:e]

    def decimate_indices(indices):
        if resolved_decimation_max_points is None:
            return indices
        max_points = int(resolved_decimation_max_points)
        if max_points <= 0 or indices.size <= max_points:
            return indices
        step = int(np.ceil(indices.size / float(max_points)))
        return indices[::step]

    # Plot-specific display constraints used by time-series orbital element views.
    time_series_marker_size = 4.0
    faceted_marker_size = 3.0
    phase_alt_marker_size = 3.0
    common_epoch_marker_size = 4.0

    def _valid_timeseries_mask(y_key, y_vals):
        y_arr = np.asarray(y_vals, dtype=np.float64)
        mask = np.isfinite(y_arr)
        if y_key == 'semi_major_axes':
            mask &= y_arr <= 7000.0
        elif y_key == 'inclinations':
            mask &= y_arr >= 40.0
        elif y_key == 'eccentricities':
            mask &= y_arr <= 0.05
        return mask

    def _filtered_timeseries_xy(y_key, indices):
        x_vals = gui_data['timestamps_dt'][indices]
        y_vals = gui_data[y_key][indices]
        if y_vals.size == 0:
            return x_vals, y_vals
        keep = _valid_timeseries_mask(y_key, y_vals)
        return x_vals[keep], y_vals[keep]

    limits_cache = {}

    def get_limits(y_key, selected_name):
        key = (y_key, selected_name)
        if key in limits_cache:
            return limits_cache[key]
        idx = get_indices(selected_name)
        if idx.size == 0:
            limits_cache[key] = None
            return None
        x, y = _filtered_timeseries_xy(y_key, idx)
        if y.size == 0:
            limits_cache[key] = None
            return None
        lims = (x.min(), x.max(), float(np.min(y)), float(np.max(y)))
        limits_cache[key] = lims
        return lims

    initial_idx = display_names.index('All Files') if 'All Files' in display_names else 0
    init_name = display_names[initial_idx]
    init_indices = get_indices(init_name)

    needs_unwrapped_raan = raan_mode_requested in {'unwrapped_by_object', 'residual_by_object'}
    needs_residual_raan = raan_mode_requested == 'residual_by_object'

    if needs_unwrapped_raan:
        raan_unwrapped_all, raan_unwrap_meta = _compute_objectwise_unwrapped_angles(
            gui_data['timestamps_dt'],
            gui_data['filenames_array'],
            gui_data['right_ascensions'],
            min_points_for_unwrap=raan_min_points_for_unwrap,
        )
        raan_unwrap_meta['computed'] = True
    else:
        raan_unwrapped_all = np.asarray([], dtype=np.float64)
        raan_unwrap_meta = {
            'objects_unwrapped': 0,
            'objects_short_tracks': 0,
            'computed': False,
        }

    if needs_residual_raan:
        raan_residual_all, raan_residual_meta = _compute_objectwise_residual_angles(
            gui_data['timestamps_dt'],
            gui_data['filenames_array'],
            raan_unwrapped_all,
            fit_mode=raan_residual_fit_mode,
            min_points=raan_min_points_for_unwrap,
        )
        raan_drift_meta = _estimate_objectwise_drift_metadata(
            gui_data['timestamps_dt'],
            gui_data['filenames_array'],
            raan_unwrapped_all,
            window_days=30.0,
        )
        raan_residual_meta['computed'] = True
        raan_drift_meta['computed'] = True
    else:
        raan_residual_all = np.asarray([], dtype=np.float64)
        raan_residual_meta = {
            'objects_fitted': 0,
            'fit_mode': _normalize_fit_mode(raan_residual_fit_mode),
            'computed': False,
        }
        raan_drift_meta = {
            'window_days': 30.0,
            'object_count': 0,
            'median_drift_deg_per_day': np.nan,
            'p10_drift_deg_per_day': np.nan,
            'p90_drift_deg_per_day': np.nan,
            'computed': False,
        }

    ecc_descent_mask_all, ecc_descent_meta = _detect_negative_slope_segments(
        gui_data['timestamps_dt'],
        gui_data['filenames_array'],
        gui_data['eccentricities'],
        threshold_per_day=eccentricity_descent_threshold_per_day,
        min_duration_days=eccentricity_descent_min_duration_days,
        min_value=eccentricity_descent_min_eccentricity,
        min_points=3,
    )
    # Treat sustained high-e points as the same operational highlight class.
    ecc_thruster_firing_threshold = 2.0e-3
    ecc_thruster_firing_mask_all = (
        np.isfinite(np.asarray(gui_data['eccentricities'], dtype=np.float64))
        & (np.asarray(gui_data['eccentricities'], dtype=np.float64) > float(ecc_thruster_firing_threshold))
    )
    ecc_highlight_mask_all = np.asarray(ecc_descent_mask_all, dtype=bool) | np.asarray(ecc_thruster_firing_mask_all, dtype=bool)

    def create_plot(y_data_key, title, ylabel, color,
                    time_render_mode='scatter',
                    reference_targets=None,
                    draw_reference_lines=False,
                    draw_reference_annotation=False,
                    fixed_xlim=None,
                    fixed_ylim=None,
                    metadata_key=None,
                    marker_size=time_series_marker_size,
                    metadata_store=None,
                    variable_options=None):
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        state = {
            'artist': None,
            'overlay_artists': [],
            'legend': None,
            'annotation': None,
        }
        resolved_metadata_key = str(metadata_key or y_data_key)
        meta_store = metadata_store if isinstance(metadata_store, dict) else inclination_metadata
        meta_store.setdefault('figures', {})
        variable_options = dict(variable_options or {})
        transform_fn = variable_options.get('transform_fn')
        overlay_fn = variable_options.get('overlay_fn')
        display_mode_requested = str(variable_options.get('display_mode_requested', 'default'))
        render_mode_requested = _normalize_time_render_mode(
            variable_options.get(
                'time_render_mode',
                time_render_mode if y_data_key == 'inclinations' else 'scatter',
            )
        )

        figure_meta = {
            'requested_time_render_mode': str(render_mode_requested),
            'effective_time_render_mode': str(render_mode_requested),
            'sparse_fallback_active': False,
            'display_mode_requested': display_mode_requested,
            'display_mode_effective': display_mode_requested,
            'target_inclinations_deg': [float(v) for v in (reference_targets or [])],
            'assignment_tolerance_deg': float(inclination_reference_assignment_tolerance_deg),
            'reference_stats': {
                'groups': [],
                'omitted_targets': [],
                'assigned_count': 0,
                'unassigned_count': 0,
            },
            'lines_drawn_successfully': {
                'target': False,
                'mean': False,
                'median': False,
            },
            'annotation_enabled': bool(draw_reference_annotation),
            'annotation_drawn': False,
            'variable_metadata': {},
        }

        def clear_overlay_state():
            for artist in state['overlay_artists']:
                _remove_artist_safe(artist)
            state['overlay_artists'] = []
            if state['legend'] is not None:
                try:
                    state['legend'].remove()
                except Exception:
                    pass
                state['legend'] = None
            if state['annotation'] is not None:
                _remove_artist_safe(state['annotation'])
                state['annotation'] = None

        def refresh_inclination_overlays(y_vals):
            clear_overlay_state()
            if y_data_key != 'inclinations':
                return

            targets = [float(v) for v in (reference_targets or [])]
            stats = compute_inclination_reference_stats(
                y_vals,
                targets,
                assignment_tolerance_deg=inclination_reference_assignment_tolerance_deg,
            )

            line_flags = {'target': False, 'mean': False, 'median': False}

            if bool(draw_reference_lines):
                for target in targets:
                    line = ax.axhline(
                        float(target),
                        linestyle='--',
                        linewidth=1.8,
                        color='black',
                        alpha=0.95,
                        zorder=2,
                    )
                    state['overlay_artists'].append(line)
                    line_flags['target'] = True

                legend_handles = [
                    Line2D([0], [0], color='black', linestyle='--', linewidth=1.8, label='Target Inclinations'),
                ]
                state['legend'] = ax.legend(handles=legend_handles, loc='upper right', fontsize=12, framealpha=0.85)

            if bool(draw_reference_annotation):
                annotation_text = _format_inclination_reference_annotation(stats)
                if annotation_text:
                    state['annotation'] = ax.text(
                        0.01,
                        0.99,
                        annotation_text,
                        transform=ax.transAxes,
                        va='top',
                        ha='left',
                        fontsize=12,
                        bbox={
                            'boxstyle': 'round,pad=0.25',
                            'facecolor': 'white',
                            'alpha': 0.7,
                            'edgecolor': 'none',
                        },
                    )

            figure_meta['target_inclinations_deg'] = [float(v) for v in targets]
            figure_meta['reference_stats'] = stats
            figure_meta['lines_drawn_successfully'] = line_flags
            figure_meta['annotation_drawn'] = bool(state['annotation'] is not None)
            meta_store['figures'][resolved_metadata_key] = figure_meta

        def apply_generic_overlay(payload):
            clear_overlay_state()
            if overlay_fn is None:
                return
            overlay_payload = overlay_fn(ax=ax, payload=payload, figure_meta=figure_meta)
            if not isinstance(overlay_payload, dict):
                return

            for artist in overlay_payload.get('artists', []):
                if artist is not None:
                    state['overlay_artists'].append(artist)

            legend_info = overlay_payload.get('legend')
            if isinstance(legend_info, dict):
                handles = legend_info.get('handles', [])
                if len(handles) > 0:
                    state['legend'] = ax.legend(
                        handles=handles,
                        loc=legend_info.get('loc', 'upper right'),
                        fontsize=legend_info.get('fontsize', 12),
                        framealpha=legend_info.get('framealpha', 0.85),
                    )

            annotation_info = overlay_payload.get('annotation')
            if isinstance(annotation_info, dict) and annotation_info.get('text'):
                state['annotation'] = ax.text(
                    annotation_info.get('x', 0.01),
                    annotation_info.get('y', 0.99),
                    annotation_info.get('text'),
                    transform=ax.transAxes,
                    va='top',
                    ha='left',
                    fontsize=annotation_info.get('fontsize', 12),
                    bbox=annotation_info.get(
                        'bbox',
                        {
                            'boxstyle': 'round,pad=0.25',
                            'facecolor': 'white',
                            'alpha': 0.7,
                            'edgecolor': 'none',
                        },
                    ),
                )

            meta_update = overlay_payload.get('meta')
            if isinstance(meta_update, dict):
                figure_meta['variable_metadata'].update(meta_update)

            figure_meta['annotation_drawn'] = bool(state['annotation'] is not None)

        def _build_payload(indices):
            idx_arr = np.asarray(indices, dtype=np.int64)
            x_raw = gui_data['timestamps_dt'][idx_arr]
            y_raw = np.asarray(gui_data[y_data_key][idx_arr], dtype=np.float64)
            if y_raw.size == 0:
                return {
                    'x': x_raw,
                    'y': y_raw,
                    'indices': idx_arr,
                    'metadata': {},
                    'render_mode': render_mode_requested,
                    'display_mode_effective': display_mode_requested,
                    'overlay_payload': {},
                }

            keep = _valid_timeseries_mask(y_data_key, y_raw)
            x_base = x_raw[keep]
            y_base = y_raw[keep]
            idx_base = idx_arr[keep]
            payload = {
                'x': x_base,
                'y': y_base,
                'indices': idx_base,
                'metadata': {},
                'render_mode': render_mode_requested,
                'display_mode_effective': display_mode_requested,
                'overlay_payload': {},
            }

            if transform_fn is None:
                return payload

            transformed = transform_fn(indices=idx_base, x_vals=x_base, y_vals=y_base)
            if not isinstance(transformed, dict):
                return payload

            x_new = transformed.get('x', x_base)
            y_new = transformed.get('y', y_base)
            idx_new = transformed.get('indices', idx_base)
            if len(x_new) != len(y_new):
                raise ValueError(f"Transformed x/y length mismatch for {y_data_key}")

            payload['x'] = x_new
            payload['y'] = y_new
            payload['indices'] = np.asarray(idx_new, dtype=np.int64)
            payload['metadata'] = dict(transformed.get('metadata', {}))
            payload['render_mode'] = _normalize_time_render_mode(transformed.get('render_mode', render_mode_requested))
            payload['display_mode_effective'] = str(transformed.get('display_mode_effective', display_mode_requested))
            payload['overlay_payload'] = dict(transformed.get('overlay_payload', {}))
            return payload

        def set_axis_limits(selected_name, x_vals=None, y_vals=None):
            if fixed_xlim is not None:
                ax.set_xlim(float(fixed_xlim[0]), float(fixed_xlim[1]))
            elif x_vals is not None and len(x_vals) > 0:
                xdt = pd.to_datetime(np.asarray(x_vals), errors='coerce')
                xdt = xdt[~pd.isna(xdt)]
                if len(xdt) > 0:
                    xmin = xdt.min()
                    xmax = xdt.max()
                    if xmin == xmax:
                        xmin = xmin - pd.Timedelta(hours=12)
                        xmax = xmax + pd.Timedelta(hours=12)
                    ax.set_xlim(xmin, xmax)
            else:
                lims = get_limits(y_data_key, selected_name)
                if lims is not None:
                    xmin = lims[0]
                    xmax = lims[1]
                    if xmin == xmax:
                        xmin = xmin - pd.Timedelta(hours=12)
                        xmax = xmax + pd.Timedelta(hours=12)
                    ax.set_xlim(xmin, xmax)

            if fixed_ylim is not None:
                ax.set_ylim(float(fixed_ylim[0]), float(fixed_ylim[1]))
            elif y_vals is not None and len(y_vals) > 0:
                y_arr = np.asarray(y_vals, dtype=np.float64)
                y_arr = y_arr[np.isfinite(y_arr)]
                if y_arr.size > 0:
                    ymin = float(np.min(y_arr))
                    ymax = float(np.max(y_arr))
                    if ymin == ymax:
                        pad = max(1.0e-6, abs(ymin) * 1.0e-3)
                        ymin -= pad
                        ymax += pad
                    ax.set_ylim(ymin, ymax)
            else:
                lims = get_limits(y_data_key, selected_name)
                if lims is not None:
                    ymin = float(lims[2])
                    ymax = float(lims[3])
                    if ymin == ymax:
                        pad = max(1.0e-6, abs(ymin) * 1.0e-3)
                        ymin -= pad
                        ymax += pad
                    ax.set_ylim(ymin, ymax)

        init_decimated = decimate_indices(init_indices)
        init_payload = _build_payload(init_decimated)
        artist_payload = _create_timeseries_artist(
            ax,
            init_payload['x'],
            init_payload['y'],
            render_mode=init_payload['render_mode'],
            color=color,
            marker_size=marker_size,
        )
        state['artist'] = artist_payload['artist']

        figure_meta['requested_time_render_mode'] = str(artist_payload['requested_mode'])
        figure_meta['effective_time_render_mode'] = str(artist_payload['effective_mode'])
        figure_meta['sparse_fallback_active'] = bool(artist_payload['sparse_fallback'])
        figure_meta['display_mode_effective'] = str(init_payload.get('display_mode_effective', display_mode_requested))
        figure_meta['variable_metadata'] = dict(init_payload.get('metadata', {}))

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        set_axis_limits(init_name, x_vals=init_payload['x'], y_vals=init_payload['y'])

        if y_data_key == 'inclinations':
            refresh_inclination_overlays(init_payload['y'])
        else:
            apply_generic_overlay(init_payload)
            meta_store['figures'][resolved_metadata_key] = figure_meta

        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig)
        slider = Slider(ax_slider, 'File Index', 0, len(display_names) - 1, valinit=initial_idx, valstep=1)

        def update_plot(val):
            idx = int(slider.val)
            selected_filename = display_names[idx]
            t_update = perf_counter()
            indices = decimate_indices(get_indices(selected_filename))

            payload = _build_payload(indices)
            _remove_artist_safe(state['artist'])
            artist_meta = _create_timeseries_artist(
                ax,
                payload['x'],
                payload['y'],
                render_mode=payload['render_mode'],
                color=color,
                marker_size=marker_size,
            )
            state['artist'] = artist_meta['artist']
            figure_meta['requested_time_render_mode'] = str(artist_meta['requested_mode'])
            figure_meta['effective_time_render_mode'] = str(artist_meta['effective_mode'])
            figure_meta['sparse_fallback_active'] = bool(artist_meta['sparse_fallback'])
            figure_meta['display_mode_effective'] = str(payload.get('display_mode_effective', display_mode_requested))
            figure_meta['variable_metadata'] = dict(payload.get('metadata', {}))

            if y_data_key == 'inclinations':
                refresh_inclination_overlays(payload['y'])
            else:
                apply_generic_overlay(payload)
                meta_store['figures'][resolved_metadata_key] = figure_meta

            set_axis_limits(selected_filename, x_vals=payload['x'], y_vals=payload['y'])
            fig.canvas.draw_idle()
            print(f"[orbital_elements_plot] {y_data_key} -> {selected_filename} in {perf_counter() - t_update:.2f}s")

        slider.on_changed(update_plot)
        if show_plots:
            plt.show()
            if not return_figures and not _preserve_open_figures_for_export():
                plt.close(fig)

        return fig

    def _transform_raan(indices, x_vals, y_vals):
        idx = np.asarray(indices, dtype=np.int64)
        x_arr = pd.to_datetime(np.asarray(x_vals), errors='coerce')
        y_wrapped = wrap_degrees_360(np.asarray(y_vals, dtype=np.float64))

        effective_mode = str(raan_mode_requested)
        render_mode = str(raan_time_render_mode)
        fallback_used = False

        if effective_mode == 'wrapped_density' and render_mode == 'scatter':
            render_mode = 'hexbin_time'

        if effective_mode == 'unwrapped_by_object':
            y_mode = raan_unwrapped_all[idx]
        elif effective_mode == 'residual_by_object':
            y_mode = raan_residual_all[idx]
        else:
            y_mode = y_wrapped

        keep = (~pd.isna(x_arr)) & np.isfinite(y_mode)
        x_out = x_arr[keep]
        y_out = y_mode[keep]
        idx_out = idx[keep]

        if y_out.size == 0 and effective_mode in {'unwrapped_by_object', 'residual_by_object'}:
            keep_wrap = (~pd.isna(x_arr)) & np.isfinite(y_wrapped)
            x_out = x_arr[keep_wrap]
            y_out = y_wrapped[keep_wrap]
            idx_out = idx[keep_wrap]
            effective_mode = 'wrapped_scatter'
            render_mode = 'scatter'
            fallback_used = True

        return {
            'x': x_out,
            'y': y_out,
            'indices': idx_out,
            'render_mode': render_mode,
            'display_mode_effective': effective_mode,
            'metadata': {
                'unwrap_applied': bool(effective_mode in {'unwrapped_by_object', 'residual_by_object'}),
                'residual_applied': bool(effective_mode == 'residual_by_object'),
                'residual_fit_mode': str(raan_residual_fit_mode),
                'dense_mode_requested': bool(raan_mode_requested == 'wrapped_density'),
                'fallback_to_wrapped_scatter': bool(fallback_used),
                'points_visible': int(len(y_out)),
            },
        }

    def _overlay_sma(ax, payload, figure_meta):
        artists = []
        handles = []

        if bool(sma_reference_lines) and len(sma_reference_values) > 0:
            for value in sma_reference_values:
                line = ax.axhline(float(value), linestyle='--', linewidth=1.2, color='black', alpha=0.75)
                artists.append(line)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color='black',
                    linestyle='--',
                    linewidth=1.2,
                    label='Target Semi-Major Axes',
                )
            )

        if len(normalized_sma_bands) > 0:
            for band in normalized_sma_bands:
                patch = ax.axhspan(
                    float(band['min_km']),
                    float(band['max_km']),
                    facecolor=str(band.get('color', '#dff3e3')),
                    alpha=float(band.get('alpha', 0.18)),
                    linewidth=0.0,
                )
                artists.append(patch)
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=str(band.get('color', '#dff3e3')),
                        linewidth=6,
                        alpha=float(band.get('alpha', 0.18)),
                        label=str(band.get('label', 'SMA band')),
                    )
                )

        summary = None

        y_vals = np.asarray(payload.get('y', []), dtype=np.float64)
        valid_y = y_vals[np.isfinite(y_vals)]
        return {
            'artists': artists,
            'legend': {
                'handles': handles,
                'loc': 'lower left',
                'fontsize': 12,
                'framealpha': 0.85,
            } if len(handles) > 0 else None,
            'annotation': None,
            'meta': {
                'reference_lines_enabled': bool(sma_reference_lines),
                'reference_values_km': [float(v) for v in sma_reference_values],
                'rolling_window_days': float(sma_rolling_window_days),
                'rolling_summary_available': bool(summary is not None),
                'show_envelope': bool(sma_show_envelope),
                'valid_count': int(valid_y.size),
                'active_median_sma_km': float(np.nanmedian(valid_y)) if valid_y.size > 0 else np.nan,
            },
        }

    def _transform_argp(indices, x_vals, y_vals):
        idx = np.asarray(indices, dtype=np.int64)
        x_arr = pd.to_datetime(np.asarray(x_vals), errors='coerce')
        y_wrapped = wrap_degrees_360(np.asarray(y_vals, dtype=np.float64))
        ecc_sel = np.asarray(gui_data['eccentricities'][idx], dtype=np.float64)

        valid_angle = (~pd.isna(x_arr)) & np.isfinite(y_wrapped)
        valid_ecc = np.isfinite(ecc_sel) & (ecc_sel >= float(argp_ecc_floor))
        low_e = valid_angle & (~valid_ecc)

        effective_display_mode = str(argp_display_mode)
        if effective_display_mode == 'ecc_filtered':
            keep = valid_angle & valid_ecc
        elif effective_display_mode == 'split_validity':
            keep = valid_angle & valid_ecc
        elif effective_display_mode == 'arglat_companion':
            if argp_mode_requested == 'suppress':
                keep = valid_angle & valid_ecc
            else:
                keep = valid_angle
        else:
            keep = valid_angle

        if argp_mode_requested == 'suppress' and effective_display_mode == 'raw_all':
            keep = valid_angle & valid_ecc
        if argp_mode_requested in {'highlight', 'split'} and effective_display_mode == 'raw_all':
            keep = valid_angle

        x_out = x_arr[keep]
        y_out = y_wrapped[keep]
        idx_out = idx[keep]

        low_keep = low_e
        if effective_display_mode == 'ecc_filtered':
            low_keep = np.zeros(low_e.shape, dtype=bool)
        low_x = x_arr[low_keep]
        low_y = y_wrapped[low_keep]

        companion_x = np.asarray([], dtype='datetime64[ns]')
        companion_y = np.asarray([], dtype=np.float64)
        companion_used = False
        if bool(argp_show_argument_of_latitude_companion) or effective_display_mode == 'arglat_companion':
            arglat_series = gui_data.get('argument_of_latitude_series')
            if arglat_series is not None:
                arglat_sel = wrap_degrees_360(np.asarray(arglat_series[idx], dtype=np.float64))
                comp_keep = (~pd.isna(x_arr)) & np.isfinite(arglat_sel)
                companion_x = x_arr[comp_keep]
                companion_y = arglat_sel[comp_keep]
                companion_used = companion_y.size > 0

        finite_count = int(np.sum(valid_angle))
        suppressed_count = int(np.sum(valid_angle & (~keep)))

        return {
            'x': x_out,
            'y': y_out,
            'indices': idx_out,
            'render_mode': str(argp_time_render_mode),
            'display_mode_effective': effective_display_mode,
            'overlay_payload': {
                'low_x': low_x,
                'low_y': low_y,
                'companion_x': companion_x,
                'companion_y': companion_y,
                'companion_used': bool(companion_used),
            },
            'metadata': {
                'ecc_floor': float(argp_ecc_floor),
                'low_e_behavior': str(argp_mode_requested),
                'display_mode_requested': str(argp_display_mode),
                'display_mode_effective': str(effective_display_mode),
                'fraction_suppressed': (float(suppressed_count) / float(finite_count)) if finite_count > 0 else np.nan,
                'suppressed_count': int(suppressed_count),
                'finite_count': int(finite_count),
                'companion_argument_of_latitude_used': bool(companion_used),
            },
        }

    def _overlay_argp(ax, payload, figure_meta):
        artists = []
        handles = []
        overlay = payload.get('overlay_payload', {})

        low_x = pd.to_datetime(np.asarray(overlay.get('low_x', [])), errors='coerce')
        low_y = np.asarray(overlay.get('low_y', []), dtype=np.float64)
        keep_low = (~pd.isna(low_x)) & np.isfinite(low_y)
        if np.any(keep_low):
            low_artist = ax.scatter(
                low_x[keep_low],
                low_y[keep_low],
                c='#8b8b8b',
                s=max(2.0, float(time_series_marker_size) * 1.2),
                linewidths=0,
                alpha=0.6,
                rasterized=True,
            )
            artists.append(low_artist)
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#8b8b8b', markersize=6, label='Low-e points'))

        companion_x = pd.to_datetime(np.asarray(overlay.get('companion_x', [])), errors='coerce')
        companion_y = np.asarray(overlay.get('companion_y', []), dtype=np.float64)
        keep_comp = (~pd.isna(companion_x)) & np.isfinite(companion_y)
        if np.any(keep_comp):
            companion_artist = ax.plot(
                companion_x[keep_comp],
                companion_y[keep_comp],
                color='#d97706',
                linewidth=1.0,
                alpha=0.65,
                label='Argument of latitude companion',
            )[0]
            artists.append(companion_artist)
            handles.append(Line2D([0], [0], color='#d97706', linewidth=1.0, label='Argument of latitude companion'))

        return {
            'artists': artists,
            'legend': {
                'handles': handles,
                'loc': 'upper right',
                'fontsize': 12,
                'framealpha': 0.85,
            } if len(handles) > 0 else None,
            'meta': {
                'low_e_points_highlighted': bool(np.any(keep_low)),
                'companion_argument_of_latitude_used': bool(np.any(keep_comp)),
            },
        }

    def _transform_eccentricity(indices, x_vals, y_vals):
        idx = np.asarray(indices, dtype=np.int64)
        x_arr = pd.to_datetime(np.asarray(x_vals), errors='coerce')
        y_arr = np.asarray(y_vals, dtype=np.float64)
        keep = (~pd.isna(x_arr)) & np.isfinite(y_arr)
        x_out = x_arr[keep]
        y_out = y_arr[keep]
        idx_out = idx[keep]

        desc_sel = np.asarray(ecc_highlight_mask_all[idx_out], dtype=bool) if idx_out.size > 0 else np.asarray([], dtype=bool)
        return {
            'x': x_out,
            'y': y_out,
            'indices': idx_out,
            'render_mode': str(eccentricity_time_render_mode),
            'display_mode_effective': 'baseline',
            'overlay_payload': {
                'descent_mask': desc_sel,
            },
            'metadata': {
                'descent_threshold_per_day': float(eccentricity_descent_threshold_per_day),
                'descent_min_duration_days': float(eccentricity_descent_min_duration_days),
                'descent_min_eccentricity': float(eccentricity_descent_min_eccentricity),
                'thruster_firing_eccentricity_threshold': float(ecc_thruster_firing_threshold),
            },
        }

    def _overlay_eccentricity(ax, payload, figure_meta):
        artists = []
        handles = []

        overlay = payload.get('overlay_payload', {})
        desc_mask = np.asarray(overlay.get('descent_mask', []), dtype=bool)
        x_vals = pd.to_datetime(np.asarray(payload.get('x', [])), errors='coerce')
        y_vals = np.asarray(payload.get('y', []), dtype=np.float64)
        desc_keep = np.zeros(y_vals.shape, dtype=bool)
        if desc_mask.shape == y_vals.shape and bool(eccentricity_highlight_descents):
            desc_keep = desc_mask & (~pd.isna(x_vals)) & np.isfinite(y_vals)
        if np.any(desc_keep):
            highlight = ax.scatter(
                x_vals[desc_keep],
                y_vals[desc_keep],
                c='#b91c1c',
                s=max(2.0, float(time_series_marker_size) * 1.4),
                linewidths=0,
                alpha=0.75,
                rasterized=True,
            )
            artists.append(highlight)
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#b91c1c', markersize=4, label='Potential Thruster Firings'))

        fraction_desc = float(np.mean(desc_keep)) if desc_keep.size > 0 else np.nan
        return {
            'artists': artists,
            'legend': {
                'handles': handles,
                'loc': 'upper right',
                'fontsize': 12,
                'framealpha': 0.85,
            } if len(handles) > 0 else None,
            'meta': {
                'rolling_summary_available': False,
                'segment_count_global': int(ecc_descent_meta.get('segment_count', 0)),
                'highlighted_fraction_current': fraction_desc,
                'highlight_descents_enabled': bool(eccentricity_highlight_descents),
            },
        }

    figures = {}

    if str(layout_mode).lower() == 'faceted':
        fig_faceted, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        axes = np.asarray(axes).reshape(-1)
        plot_defs = [
            ('inclinations', 'Inclination Over Time', 'Inclination (degrees)', 'r'),
            ('semi_major_axes', 'Semi-major Axis Over Time', 'Semi-major Axis (km)', 'g'),
            ('right_ascensions', 'Right Ascension of Ascending Node Over Time', 'Right Ascension (degrees)', 'b'),
            ('args_of_perigee', 'Argument of Perigee Over Time', 'Argument of Perigee (degrees)', 'm'),
            ('eccentricities', 'Eccentricity Over Time', 'Eccentricity', 'c'),
            ('true_anomalies', 'True Anomaly (TLE Kepler proxy) Over Time', 'True Anomaly (TLE Kepler proxy, degrees)', 'k'),
        ]

        init_decimated = decimate_indices(init_indices)
        artists = {}
        faceted_incl_state = {
            'reference_lines': [],
            'legend': None,
            'annotation': None,
        }
        faceted_incl_meta = {
            'requested_time_render_mode': str(inclination_time_render_mode),
            'effective_time_render_mode': 'scatter',
            'sparse_fallback_active': False,
            'target_inclinations_deg': [float(v) for v in configured_inclination_targets],
            'assignment_tolerance_deg': float(inclination_reference_assignment_tolerance_deg),
            'reference_stats': {
                'groups': [],
                'omitted_targets': [],
                'assigned_count': 0,
                'unassigned_count': 0,
            },
            'lines_drawn_successfully': {
                'target': False,
                'mean': False,
                'median': False,
            },
            'annotation_enabled': bool(inclination_reference_annotation),
            'annotation_drawn': False,
        }

        def _clear_faceted_incl_overlays(ax_incl):
            for line in faceted_incl_state['reference_lines']:
                _remove_artist_safe(line)
            faceted_incl_state['reference_lines'] = []
            if faceted_incl_state['legend'] is not None:
                try:
                    faceted_incl_state['legend'].remove()
                except Exception:
                    pass
                faceted_incl_state['legend'] = None
            if faceted_incl_state['annotation'] is not None:
                _remove_artist_safe(faceted_incl_state['annotation'])
                faceted_incl_state['annotation'] = None

        def _update_faceted_incl_overlays(ax_incl, y_vals):
            _clear_faceted_incl_overlays(ax_incl)
            stats = compute_inclination_reference_stats(
                y_vals,
                configured_inclination_targets,
                assignment_tolerance_deg=inclination_reference_assignment_tolerance_deg,
            )

            line_flags = {'target': False, 'mean': False, 'median': False}

            if bool(inclination_reference_lines):
                for target in configured_inclination_targets:
                    faceted_incl_state['reference_lines'].append(
                        ax_incl.axhline(float(target), linestyle='--', linewidth=1.8, color='black', alpha=0.95, zorder=2)
                    )
                    line_flags['target'] = True

                legend_handles = [
                    Line2D([0], [0], color='black', linestyle='--', linewidth=1.8, label='Target Inclinations'),
                ]
                faceted_incl_state['legend'] = ax_incl.legend(
                    handles=legend_handles,
                    loc='upper right',
                    fontsize=12,
                    framealpha=0.85,
                )

            if bool(inclination_reference_annotation):
                annotation_text = _format_inclination_reference_annotation(stats)
                if annotation_text:
                    faceted_incl_state['annotation'] = ax_incl.text(
                        0.01,
                        0.99,
                        annotation_text,
                        transform=ax_incl.transAxes,
                        va='top',
                        ha='left',
                        fontsize=12,
                        bbox={
                            'boxstyle': 'round,pad=0.2',
                            'facecolor': 'white',
                            'alpha': 0.7,
                            'edgecolor': 'none',
                        },
                    )

            faceted_incl_meta['reference_stats'] = stats
            faceted_incl_meta['lines_drawn_successfully'] = line_flags
            faceted_incl_meta['annotation_drawn'] = bool(faceted_incl_state['annotation'] is not None)
            inclination_metadata['figures']['faceted_inclinations'] = faceted_incl_meta

        for ax, (y_key, title, ylabel, color) in zip(axes, plot_defs):
            x_init, y_init = _filtered_timeseries_xy(y_key, init_decimated)
            if y_key == 'inclinations':
                payload = _create_timeseries_artist(
                    ax,
                    x_init,
                    y_init,
                    render_mode=inclination_time_render_mode,
                    color=color,
                    marker_size=faceted_marker_size,
                )
                artists[y_key] = payload['artist']
                faceted_incl_meta['requested_time_render_mode'] = str(payload['requested_mode'])
                faceted_incl_meta['effective_time_render_mode'] = str(payload['effective_mode'])
                faceted_incl_meta['sparse_fallback_active'] = bool(payload['sparse_fallback'])
                _update_faceted_incl_overlays(ax, y_init)
            else:
                artists[y_key] = ax.scatter(
                    x_init,
                    y_init,
                    c=color,
                    s=faceted_marker_size,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

            lims = get_limits(y_key, init_name)
            if lims is not None:
                ax.set_xlim(lims[0], lims[1])
                ax.set_ylim(lims[2], lims[3])

        for ax in axes[-2:]:
            ax.set_xlabel('Time')

        ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig_faceted)
        slider = Slider(ax_slider, 'File Index', 0, len(display_names) - 1, valinit=initial_idx, valstep=1)

        def update_faceted(val):
            idx = int(slider.val)
            selected_name = display_names[idx]
            t_update = perf_counter()
            idx_sel = decimate_indices(get_indices(selected_name))
            for ax, (y_key, _, _, _) in zip(axes, plot_defs):
                x_vals, y_vals = _filtered_timeseries_xy(y_key, idx_sel)
                if y_key == 'inclinations':
                    _remove_artist_safe(artists[y_key])
                    payload = _create_timeseries_artist(
                        ax,
                        x_vals,
                        y_vals,
                        render_mode=inclination_time_render_mode,
                        color='r',
                        marker_size=faceted_marker_size,
                    )
                    artists[y_key] = payload['artist']
                    faceted_incl_meta['requested_time_render_mode'] = str(payload['requested_mode'])
                    faceted_incl_meta['effective_time_render_mode'] = str(payload['effective_mode'])
                    faceted_incl_meta['sparse_fallback_active'] = bool(payload['sparse_fallback'])
                    _update_faceted_incl_overlays(ax, y_vals)
                else:
                    if y_vals.size > 0:
                        artists[y_key].set_offsets(np.column_stack((date2num(x_vals), y_vals)))
                    else:
                        artists[y_key].set_offsets(np.empty((0, 2), dtype=np.float64))
                lims = get_limits(y_key, selected_name)
                if lims is not None:
                    ax.set_xlim(lims[0], lims[1])
                    ax.set_ylim(lims[2], lims[3])
            fig_faceted.canvas.draw_idle()
            print(f"[orbital_elements_plot] faceted -> {selected_name} in {perf_counter() - t_update:.2f}s")

        slider.on_changed(update_faceted)
        figures['faceted_timeseries'] = fig_faceted
        if show_plots:
            plt.show()
            if not return_figures and not _preserve_open_figures_for_export():
                plt.close(fig_faceted)
    else:
        figures['inclinations'] = create_plot(
            'inclinations',
            'Inclination Over Time',
            'Inclination (degrees)',
            'r',
            time_render_mode=inclination_time_render_mode,
            reference_targets=configured_inclination_targets,
            draw_reference_lines=bool(inclination_reference_lines),
            draw_reference_annotation=bool(inclination_reference_annotation),
            metadata_key='inclinations_all',
            marker_size=time_series_marker_size,
            metadata_store=inclination_metadata,
        )
        figures['semi_major_axes'] = create_plot(
            'semi_major_axes',
            'Semi-major Axis Over Time',
            'Semi-major Axis (km)',
            'g',
            metadata_store=sma_metadata,
            metadata_key='semi_major_axes',
            variable_options={
                'time_render_mode': sma_time_render_mode,
                'display_mode_requested': 'lifecycle',
                'overlay_fn': _overlay_sma,
            },
        )
        figures['right_ascensions'] = create_plot(
            'right_ascensions',
            'Right Ascension of Ascending Node Over Time',
            'Right Ascension (degrees)',
            'b',
            metadata_store=raan_metadata,
            metadata_key='right_ascensions',
            variable_options={
                'time_render_mode': raan_time_render_mode,
                'display_mode_requested': raan_mode_requested,
                'transform_fn': _transform_raan,
            },
        )
        argp_title = 'Argument of Perigee Over Time'
        if argp_display_mode == 'ecc_filtered' or (argp_display_mode == 'raw_all' and argp_mode_requested == 'suppress'):
            argp_title = 'Argument of Perigee Over Time (low-e filtered)'

        figures['args_of_perigee'] = create_plot(
            'args_of_perigee',
            argp_title,
            'Argument of Perigee (degrees)',
            'm',
            metadata_store=argp_metadata,
            metadata_key='args_of_perigee',
            variable_options={
                'time_render_mode': argp_time_render_mode,
                'display_mode_requested': argp_display_mode,
                'transform_fn': _transform_argp,
                'overlay_fn': _overlay_argp,
            },
        )
        figures['eccentricities'] = create_plot(
            'eccentricities',
            'Eccentricity Over Time',
            'Eccentricity',
            'c',
            metadata_store=eccentricity_metadata,
            metadata_key='eccentricities',
            variable_options={
                'time_render_mode': eccentricity_time_render_mode,
                'display_mode_requested': 'baseline',
                'transform_fn': _transform_eccentricity,
                'overlay_fn': _overlay_eccentricity,
            },
        )

        if eccentricity_zoom_limits is not None:
            ecc_zoom_bounds = (float(eccentricity_zoom_limits[0]), float(eccentricity_zoom_limits[1]))
            figures['eccentricities_zoom'] = create_plot(
                'eccentricities',
                'Eccentricity Over Time (Zoomed)',
                'Eccentricity',
                'c',
                fixed_ylim=ecc_zoom_bounds,
                metadata_store=eccentricity_metadata,
                metadata_key='eccentricities_zoom',
                variable_options={
                    'time_render_mode': eccentricity_time_render_mode,
                    'display_mode_requested': 'baseline',
                    'transform_fn': _transform_eccentricity,
                    'overlay_fn': _overlay_eccentricity,
                },
            )

        figures['true_anomalies'] = create_plot('true_anomalies', 'True Anomaly (TLE Kepler proxy) Over Time', 'True Anomaly (TLE Kepler proxy, degrees)', 'k')

        if bool(inclination_focused_views):
            figures['inclinations_53'] = create_plot(
                'inclinations',
                'Inclination Over Time (53-degree Family)',
                'Inclination (degrees)',
                'r',
                time_render_mode=inclination_time_render_mode,
                reference_targets=[53.05, 53.217],
                draw_reference_lines=bool(inclination_reference_lines),
                draw_reference_annotation=bool(inclination_reference_annotation),
                fixed_xlim=focus_windows['53']['xlim'],
                fixed_ylim=focus_windows['53']['ylim'],
                metadata_key='inclinations_53',
                marker_size=time_series_marker_size,
                metadata_store=inclination_metadata,
            )
            figures['inclinations_70'] = create_plot(
                'inclinations',
                'Inclination Over Time (70-degree Family)',
                'Inclination (degrees)',
                'r',
                time_render_mode=inclination_time_render_mode,
                reference_targets=[70.0],
                draw_reference_lines=bool(inclination_reference_lines),
                draw_reference_annotation=bool(inclination_reference_annotation),
                fixed_xlim=focus_windows['70']['xlim'],
                fixed_ylim=focus_windows['70']['ylim'],
                metadata_key='inclinations_70',
                marker_size=time_series_marker_size,
                metadata_store=inclination_metadata,
            )
            figures['inclinations_97'] = create_plot(
                'inclinations',
                'Inclination Over Time (97-degree Family)',
                'Inclination (degrees)',
                'r',
                time_render_mode=inclination_time_render_mode,
                reference_targets=[97.655],
                draw_reference_lines=bool(inclination_reference_lines),
                draw_reference_annotation=bool(inclination_reference_annotation),
                fixed_xlim=focus_windows['97']['xlim'],
                fixed_ylim=focus_windows['97']['ylim'],
                metadata_key='inclinations_97',
                marker_size=time_series_marker_size,
                metadata_store=inclination_metadata,
            )

    if phase_mode is not None and 'phase_series' in gui_data and 'altitude_series' in gui_data:
        fig_phase_alt, ax_phase_alt = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)

        color_values = gui_data.get('shell_series', gui_data['eccentricities'])
        initial_colors = color_values[init_indices]
        shell_color_info = None
        if np.asarray(initial_colors).dtype.kind in {'U', 'S', 'O'}:
            shell_color_info = stable_category_color_map(color_values)
            initial_colors = shell_color_info['codes'][init_indices]
        shell_color_state = {'info': shell_color_info}

        init_decimated = decimate_indices(init_indices)
        x_phase_init = wrap_degrees_360(gui_data['phase_series'][init_decimated])
        y_sma_init = gui_data['semi_major_axes'][init_decimated]
        c_init = np.asarray(initial_colors)[np.searchsorted(init_indices, init_decimated)] if init_indices.size > 0 else np.asarray(initial_colors)

        scatter_phase_alt = ax_phase_alt.scatter(x_phase_init,
                             y_sma_init,
                                                 c=c_init, s=phase_alt_marker_size, linewidths=0, rasterized=True, cmap='viridis')
        ax_phase_alt.set_title('Semi-Major Axis vs True Anomaly (TLE Kepler proxy) (Circular-Linear)')
        ax_phase_alt.set_xlabel('True Anomaly (TLE Kepler proxy, degrees, wrapped 0-360)')
        ax_phase_alt.set_ylabel('Semi-Major Axis (km)')
        plt.colorbar(scatter_phase_alt, ax=ax_phase_alt, label='Shell/Eccentricity Color')

        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig_phase_alt)
        slider = Slider(ax_slider, 'File Index', 0, len(display_names) - 1, valinit=initial_idx, valstep=1)

        def update_phase_alt(val):
            idx = int(slider.val)
            selected_filename = display_names[idx]
            t_update = perf_counter()
            indices = get_indices(selected_filename)
            indices = decimate_indices(indices)

            x = wrap_degrees_360(gui_data['phase_series'][indices])
            y = gui_data['semi_major_axes'][indices]
            cvals = gui_data.get('shell_series', gui_data['eccentricities'])[indices]
            if np.asarray(cvals).dtype.kind in {'U', 'S', 'O'}:
                if shell_color_state['info'] is None:
                    shell_color_state['info'] = stable_category_color_map(gui_data.get('shell_series', gui_data['eccentricities']))
                cvals = shell_color_state['info']['codes'][indices]

            if str(phase_alt_mode) == 'density':
                h = np.histogram2d(x, y, bins=[72, 60])
                padded = circular_pad_histogram_2d(h[0], pad_x=2, pad_y=2, circular_x=True, circular_y=False)
                smooth = gaussian_filter(padded, sigma=1.0)[2:-2, 2:-2]
                ax_phase_alt.clear()
                ax_phase_alt.imshow(
                    smooth.T,
                    origin='lower',
                    extent=[h[1][0], h[1][-1], h[2][0], h[2][-1]],
                    aspect='auto',
                    cmap='viridis',
                    rasterized=True,
                )
                ax_phase_alt.set_title('Semi-Major Axis vs True Anomaly (TLE Kepler proxy) (density proxy)')
                ax_phase_alt.set_xlabel('True Anomaly (TLE Kepler proxy, degrees, wrapped 0-360)')
                ax_phase_alt.set_ylabel('Semi-Major Axis (km)')
            elif str(phase_alt_mode) == 'circular_linear_kde':
                kde = circular_linear_kde(x, y, circular_bins=72, linear_bins=60, kappa=25.0)
                ax_phase_alt.clear()
                ax_phase_alt.imshow(
                    kde['density'].T,
                    origin='lower',
                    extent=[kde['theta_grid_deg'][0], kde['theta_grid_deg'][-1],
                            kde['z_grid'][0], kde['z_grid'][-1]],
                    aspect='auto',
                    cmap='viridis',
                    rasterized=True,
                )
                ax_phase_alt.set_title('Semi-Major Axis vs True Anomaly (TLE Kepler proxy) (circular-linear KDE)')
                ax_phase_alt.set_xlabel('True Anomaly (TLE Kepler proxy, degrees, wrapped 0-360)')
                ax_phase_alt.set_ylabel('Semi-Major Axis (km)')
            else:
                scatter_phase_alt.set_offsets(np.column_stack((x, y)))
                scatter_phase_alt.set_array(np.asarray(cvals, dtype=np.float64))

            if x.size > 0 and y.size > 0:
                ax_phase_alt.set_xlim(float(np.min(x)), float(np.max(x)))
                ax_phase_alt.set_ylim(float(np.min(y)), float(np.max(y)))

            fig_phase_alt.canvas.draw_idle()
            print(f"[orbital_elements_plot] phase-alt -> {selected_filename} in {perf_counter() - t_update:.2f}s")

        slider.on_changed(update_phase_alt)
        figures['phase_altitude'] = fig_phase_alt
        if show_plots:
            plt.show()
            if not return_figures and not _preserve_open_figures_for_export():
                plt.close(fig_phase_alt)

    if bool(enable_common_epoch_shell_snapshot) and common_epoch_records is not None:
        frame = common_epoch_records if isinstance(common_epoch_records, pd.DataFrame) else pd.DataFrame(common_epoch_records)
        if {'phase', 'raan', 'shell'}.issubset(set(frame.columns)):
            fig_common, ax_common = plt.subplots()
            color_info = stable_category_color_map(frame['shell'].values)
            ax_common.scatter(
                wrap_degrees_360(frame['phase'].values),
                wrap_degrees_360(frame['raan'].values),
                c=color_info['codes'].astype(float),
                cmap=color_info['cmap'],
                s=common_epoch_marker_size,
                linewidths=0,
                rasterized=True,
            )
            ax_common.set_title('Common-Epoch Shell Snapshot (RAAN vs Phase)')
            ax_common.set_xlabel('True Anomaly (TLE Kepler proxy, deg, wrapped)')
            ax_common.set_ylabel('RAAN (deg, wrapped)')
            figures['common_epoch_shell_snapshot'] = fig_common
            if show_plots:
                plt.show()
                if not return_figures and not _preserve_open_figures_for_export():
                    plt.close(fig_common)

    print(f"[orbital_elements_plot] Ready in {perf_counter() - t0:.2f}s")

    payload = {
        'metadata': {
            'layout_mode': str(layout_mode),
            'plotted_phase_variable': plotted_phase_variable,
            'phase_semantics': 'TLE-derived Kepler proxy from mean anomaly',
            'phase_alt_mode': str(phase_alt_mode),
            'decimation_max_points_requested': None if decimation_max_points is None else int(decimation_max_points),
            'decimation_max_points': None if resolved_decimation_max_points is None else int(resolved_decimation_max_points),
            'auto_decimation_applied': bool(auto_decimation_applied),
            'inclination_time_series': inclination_metadata,
            'raan_time_series': {
                **raan_metadata,
                'unwrap_summary': raan_unwrap_meta,
                'residual_summary': raan_residual_meta,
                'drift_summary': raan_drift_meta,
            },
            'sma_time_series': sma_metadata,
            'argp_time_series': argp_metadata,
            'eccentricity_time_series': {
                **eccentricity_metadata,
                'descent_summary': {
                    'threshold_per_day': float(ecc_descent_meta.get('threshold_per_day', np.nan)),
                    'min_duration_days': float(ecc_descent_meta.get('min_duration_days', np.nan)),
                    'min_eccentricity': float(ecc_descent_meta.get('min_value', np.nan)),
                    'segment_count': int(ecc_descent_meta.get('segment_count', 0)),
                    'thruster_firing_eccentricity_threshold': float(ecc_thruster_firing_threshold),
                    'highlighted_fraction_global': float(np.mean(ecc_highlight_mask_all)) if ecc_highlight_mask_all.size > 0 else np.nan,
                },
            },
            'geometric_assumptions': {
                'angular_pairs_on_torus': True,
                'phase_altitude_is_circular_linear': True,
                'phase_wrapped_360_for_display': True,
            },
        },
        'figures': figures if return_figures else {},
    }

    if return_results:
        return payload
    return None

def _circular_distance_deg(a_deg, b_deg):
    a = float(a_deg)
    b = float(b_deg)
    d = abs(a - b) % 360.0
    return float(min(d, 360.0 - d))


def _normalize_family_mode(mode):
    key = str(mode or 'aggregate').strip().lower()
    if key in {'aggregate', 'all'}:
        return 'aggregate'
    if key in {'per_family', 'family', 'separate', 'by_family'}:
        return 'per_family'
    if key in {'both', 'aggregate_and_family'}:
        return 'both'
    return 'aggregate'


def _centers_to_edges(arr):
    vals = np.asarray(arr, dtype=np.float64)
    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    if vals.size == 1:
        return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=np.float64)
    mids = 0.5 * (vals[:-1] + vals[1:])
    left = vals[0] - (mids[0] - vals[0])
    right = vals[-1] + (vals[-1] - mids[-1])
    return np.concatenate(([left], mids, [right]))


def _prepare_density_windows(timestamps=None, explicit_windows=None,
                             rolling_window_days=None, step_days=None,
                             max_windows=10):
    windows = []

    if explicit_windows is not None:
        for i, pair in enumerate(explicit_windows):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            t0 = pd.to_datetime(pair[0], errors='coerce')
            t1 = pd.to_datetime(pair[1], errors='coerce')
            if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
                continue
            windows.append(
                {
                    'id': f"window_{i + 1}",
                    'start': t0,
                    'end': t1,
                    'label': f"{t0.date()} to {t1.date()}",
                    'mode': 'explicit',
                }
            )
        return windows

    if timestamps is None or rolling_window_days is None:
        return windows

    ts = pd.to_datetime(np.asarray(timestamps), errors='coerce')
    ts = ts[~pd.isna(ts)]
    if ts.size == 0:
        return windows

    try:
        window_days = float(rolling_window_days)
    except Exception:
        return windows
    if not np.isfinite(window_days) or window_days <= 0.0:
        return windows

    if step_days is None:
        step_days = window_days
    try:
        step_days = float(step_days)
    except Exception:
        step_days = window_days
    if not np.isfinite(step_days) or step_days <= 0.0:
        step_days = window_days

    t_min = pd.Timestamp(ts.min())
    t_max = pd.Timestamp(ts.max())
    duration = pd.Timedelta(days=float(window_days))
    step = pd.Timedelta(days=float(step_days))

    start = t_min
    count = 0
    while start < t_max and count < int(max(1, max_windows)):
        end = start + duration
        windows.append(
            {
                'id': f"rolling_{count + 1}",
                'start': start,
                'end': end,
                'label': f"{start.date()} to {end.date()}",
                'mode': 'rolling',
            }
        )
        start = start + step
        count += 1
    return windows


def assign_inclination_family_labels(inclinations,
                                     family_targets_deg=None,
                                     family_tolerance_deg=0.4,
                                     family_labels=None):
    if family_labels is not None:
        labels = np.asarray(family_labels, dtype=object)
        matched_target = np.full(labels.shape, np.nan, dtype=np.float64)
        labels_clean = np.asarray([
            str(v) if str(v).strip() != '' and str(v).lower() != 'nan' else 'unmatched'
            for v in labels
        ], dtype=object)
        unmatched = labels_clean == 'unmatched'
        unique_labels, counts = np.unique(labels_clean, return_counts=True)
        return {
            'family_labels': labels_clean,
            'matched_target_deg': matched_target,
            'unmatched_mask': unmatched,
            'family_targets_deg': None,
            'family_tolerance_deg': None,
            'assignment_mode': 'provided_labels',
            'family_counts': {str(k): int(v) for k, v in zip(unique_labels.tolist(), counts.tolist())},
        }

    inc = np.asarray(inclinations, dtype=np.float64)
    if family_targets_deg is None:
        targets = np.asarray(GEN1_INCLINATION_TARGETS, dtype=np.float64)
    else:
        targets = np.asarray(family_targets_deg, dtype=np.float64)
    targets = targets[np.isfinite(targets)]
    if targets.size == 0:
        targets = np.asarray(GEN1_INCLINATION_TARGETS, dtype=np.float64)
    targets = np.unique(targets)

    try:
        tol = float(family_tolerance_deg)
    except Exception:
        tol = 0.4
    if not np.isfinite(tol) or tol <= 0.0:
        tol = 0.4

    labels = np.full(inc.shape, 'unmatched', dtype=object)
    matched_target = np.full(inc.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(inc)
    if np.any(valid):
        distances = np.abs(inc[valid, None] - targets[None, :])
        nearest_idx = np.argmin(distances, axis=1)
        nearest_dist = distances[np.arange(distances.shape[0]), nearest_idx]
        matched = nearest_dist <= tol
        nearest_targets = targets[nearest_idx]
        valid_idx = np.flatnonzero(valid)
        matched_idx = valid_idx[matched]
        matched_target[matched_idx] = nearest_targets[matched]
        labels[matched_idx] = np.asarray([f"inc_{float(v):.3f}" for v in nearest_targets[matched]], dtype=object)

    unique_labels, counts = np.unique(labels, return_counts=True)
    return {
        'family_labels': labels,
        'matched_target_deg': matched_target,
        'unmatched_mask': labels == 'unmatched',
        'family_targets_deg': [float(v) for v in targets.tolist()],
        'family_tolerance_deg': float(tol),
        'assignment_mode': 'inclination_targets',
        'family_counts': {str(k): int(v) for k, v in zip(unique_labels.tolist(), counts.tolist())},
    }


def _compute_density_metrics(density_2d, compute_uniformity=True):
    d = np.asarray(density_2d, dtype=np.float64)
    finite = np.isfinite(d)
    if d.ndim != 2 or d.size == 0 or not np.any(finite):
        return {
            'phase_circular_entropy': np.nan,
            'raan_circular_entropy': np.nan,
            'concentration_ratio_top5pct': np.nan,
            'peak_to_median_density_ratio': np.nan,
            'phase_uniform_chi2_like': np.nan,
            'raan_uniform_chi2_like': np.nan,
            'phase_uniform_kl_divergence': np.nan,
            'raan_uniform_kl_divergence': np.nan,
        }

    d = np.where(finite, d, 0.0)
    d_sum = float(np.sum(d))
    if d_sum <= 0.0:
        return {
            'phase_circular_entropy': np.nan,
            'raan_circular_entropy': np.nan,
            'concentration_ratio_top5pct': np.nan,
            'peak_to_median_density_ratio': np.nan,
            'phase_uniform_chi2_like': np.nan,
            'raan_uniform_chi2_like': np.nan,
            'phase_uniform_kl_divergence': np.nan,
            'raan_uniform_kl_divergence': np.nan,
        }

    p = d / d_sum
    p_phase = np.sum(p, axis=1)
    p_raan = np.sum(p, axis=0)

    def _entropy(prob):
        q = np.asarray(prob, dtype=np.float64)
        q = q[q > 0.0]
        if q.size == 0:
            return np.nan
        return float(-np.sum(q * np.log(q)))

    flat = p.ravel()
    positive = flat[flat > 0.0]
    if positive.size == 0:
        concentration = np.nan
        peak_to_median = np.nan
    else:
        n_top = int(max(1, np.ceil(0.05 * positive.size)))
        concentration = float(np.sum(np.sort(positive)[-n_top:]) / np.sum(positive))
        med = float(np.median(positive))
        peak_to_median = float(np.max(positive) / med) if med > 0.0 else np.nan

    metrics = {
        'phase_circular_entropy': _entropy(p_phase),
        'raan_circular_entropy': _entropy(p_raan),
        'concentration_ratio_top5pct': concentration,
        'peak_to_median_density_ratio': peak_to_median,
        'phase_uniform_chi2_like': np.nan,
        'raan_uniform_chi2_like': np.nan,
        'phase_uniform_kl_divergence': np.nan,
        'raan_uniform_kl_divergence': np.nan,
    }

    if not bool(compute_uniformity):
        return metrics

    eps = 1.0e-12
    for key, marginal in (
        ('phase', p_phase),
        ('raan', p_raan),
    ):
        m = np.asarray(marginal, dtype=np.float64)
        if m.size == 0 or np.sum(m) <= 0.0:
            continue
        m = m / np.sum(m)
        u = np.full(m.shape, 1.0 / m.size, dtype=np.float64)
        chi_like = float(np.sum((m - u) ** 2 / (u + eps)))
        kl_div = float(np.sum(m * np.log((m + eps) / (u + eps))))
        metrics[f'{key}_uniform_chi2_like'] = chi_like
        metrics[f'{key}_uniform_kl_divergence'] = kl_div
    return metrics


def _build_raan_phase_density_product(phase_deg, raan_deg, density_mode):
    mode = str(density_mode or 'hist2d').strip().lower()
    if mode == 'torus_kde':
        kde = torus_kde_von_mises(phase_deg, raan_deg, bins_x=72, bins_y=72, kappa_x=25.0, kappa_y=25.0)
        x_centers = np.asarray(kde.get('xgrid_deg', np.linspace(0.0, 360.0, 72, endpoint=False)), dtype=np.float64)
        y_centers = np.asarray(kde.get('ygrid_deg', np.linspace(0.0, 360.0, 72, endpoint=False)), dtype=np.float64)
        return {
            'density': np.asarray(kde.get('density', np.zeros((72, 72), dtype=np.float64)), dtype=np.float64),
            'xedges': _centers_to_edges(x_centers),
            'yedges': _centers_to_edges(y_centers),
            'xcenters': x_centers,
            'ycenters': y_centers,
            'effective_mode': 'torus_kde',
            'sample_count': int(kde.get('sample_count', len(phase_deg))),
        }

    hist, xedges, yedges = np.histogram2d(phase_deg, raan_deg, bins=[72, 72], range=[[0.0, 360.0], [0.0, 360.0]])
    if mode == 'hist2d':
        pad = circular_pad_histogram_2d(hist, pad_x=2, pad_y=2, circular_x=True, circular_y=True)
        density = gaussian_filter(pad, sigma=1.0)[2:-2, 2:-2]
        eff_mode = 'hist2d'
    else:
        density = hist
        eff_mode = 'hexbin' if mode == 'hexbin' else 'hist2d'
    return {
        'density': np.asarray(density, dtype=np.float64),
        'xedges': np.asarray(xedges, dtype=np.float64),
        'yedges': np.asarray(yedges, dtype=np.float64),
        'xcenters': 0.5 * (np.asarray(xedges[:-1], dtype=np.float64) + np.asarray(xedges[1:], dtype=np.float64)),
        'ycenters': 0.5 * (np.asarray(yedges[:-1], dtype=np.float64) + np.asarray(yedges[1:], dtype=np.float64)),
        'effective_mode': eff_mode,
        'sample_count': int(len(phase_deg)),
    }


def plot_raan_vs_selected_phase_torus_density(right_ascensions, phase_series,
                                              shell_series=None, mode='hist2d',
                                              show_plots=True, return_results=False,
                                              timestamps=None,
                                              inclinations=None,
                                              family_labels=None,
                                              raan_phase_density_mode=None,
                                              raan_phase_density_family_mode='aggregate',
                                              raan_phase_density_family_targets_deg=None,
                                              raan_phase_density_family_tolerance_deg=0.4,
                                              raan_phase_density_time_windows=None,
                                              raan_phase_density_rolling_window_days=None,
                                              raan_phase_density_return_arrays=True,
                                              raan_phase_density_compute_uniformity_metrics=True,
                                              raan_phase_density_overlay_family_labels=False):
    """Plot RAAN vs True Anomaly (TLE Kepler proxy) with optional family/time conditioning."""
    del raan_phase_density_overlay_family_labels  # Labels are encoded in panel titles for compactness.

    effective_mode = str(raan_phase_density_mode or mode or 'hist2d').strip().lower()
    if effective_mode not in {'hist2d', 'hexbin', 'torus_kde'}:
        effective_mode = 'hist2d'

    raan = wrap_degrees_360(np.asarray(right_ascensions, dtype=np.float64))
    phase = wrap_degrees_360(np.asarray(phase_series, dtype=np.float64))

    finite = np.isfinite(raan) & np.isfinite(phase)
    if inclinations is not None:
        inc = np.asarray(inclinations, dtype=np.float64)
        if inc.shape != raan.shape:
            raise ValueError('inclinations must match right_ascensions shape')
    else:
        inc = None

    if timestamps is not None:
        ts = pd.to_datetime(np.asarray(timestamps), errors='coerce')
        if ts.shape != raan.shape:
            raise ValueError('timestamps must match right_ascensions shape')
    else:
        ts = np.asarray([pd.NaT] * raan.size, dtype='datetime64[ns]')
    ts_dt = pd.to_datetime(ts, errors='coerce')

    family_payload = assign_inclination_family_labels(
        inclinations=np.zeros_like(raan) if inc is None else inc,
        family_targets_deg=raan_phase_density_family_targets_deg,
        family_tolerance_deg=raan_phase_density_family_tolerance_deg,
        family_labels=family_labels,
    )
    family_arr = np.asarray(family_payload['family_labels'], dtype=object)

    family_mode = _normalize_family_mode(raan_phase_density_family_mode)
    family_entries = [('aggregate_all', np.ones(raan.shape, dtype=bool), 'Aggregate')]
    if family_mode in {'per_family', 'both'}:
        unique_families = [f for f in np.unique(family_arr).tolist() if str(f) != 'unmatched']
        if unique_families:
            entries = []
            for fam in unique_families:
                entries.append((str(fam), family_arr == fam, f"Family {fam}"))
            if family_mode == 'per_family':
                family_entries = entries
            else:
                family_entries.extend(entries)

    windows = _prepare_density_windows(
        timestamps=ts,
        explicit_windows=raan_phase_density_time_windows,
        rolling_window_days=raan_phase_density_rolling_window_days,
        step_days=None,
        max_windows=8,
    )
    if not windows:
        windows = [{'id': 'all', 'start': None, 'end': None, 'label': 'All times', 'mode': 'aggregate'}]

    panel_defs = []
    for win in windows:
        if win.get('start') is None:
            wmask = np.ones(raan.shape, dtype=bool)
        else:
            wmask = (ts_dt >= win['start']) & (ts_dt < win['end'])
        for fam_key, fam_mask, fam_label in family_entries:
            mask = finite & np.asarray(wmask, dtype=bool) & np.asarray(fam_mask, dtype=bool)
            panel_defs.append(
                {
                    'id': f"{win['id']}__{fam_key}",
                    'family_key': str(fam_key),
                    'family_label': str(fam_label),
                    'window_label': str(win.get('label', 'All times')),
                    'window_start': None if win.get('start') is None else str(win['start']),
                    'window_end': None if win.get('end') is None else str(win['end']),
                    'mask': mask,
                }
            )

    products = {}
    for panel in panel_defs:
        mask = np.asarray(panel['mask'], dtype=bool)
        p = phase[mask]
        r = raan[mask]
        product = _build_raan_phase_density_product(p, r, effective_mode)
        metrics = _compute_density_metrics(
            product['density'],
            compute_uniformity=bool(raan_phase_density_compute_uniformity_metrics),
        )
        products[panel['id']] = {
            'panel': {
                'family_key': panel['family_key'],
                'family_label': panel['family_label'],
                'window_label': panel['window_label'],
                'window_start': panel['window_start'],
                'window_end': panel['window_end'],
            },
            'sample_count': int(np.sum(mask)),
            'density_mode_effective': str(product['effective_mode']),
            'metrics': metrics,
        }
        if bool(raan_phase_density_return_arrays):
            products[panel['id']].update(
                {
                    'density': product['density'],
                    'xedges': product['xedges'],
                    'yedges': product['yedges'],
                    'xcenters': product['xcenters'],
                    'ycenters': product['ycenters'],
                }
            )

    fig = None
    if show_plots:
        n_panels = len(panel_defs)
        n_cols = int(min(3, max(1, n_panels)))
        n_rows = int(np.ceil(n_panels / float(n_cols)))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        for idx, panel in enumerate(panel_defs):
            ax = axes[idx]
            payload = products[panel['id']]
            mask = np.asarray(panel['mask'], dtype=bool)
            p = phase[mask]
            r = raan[mask]
            artist = None

            if payload['density_mode_effective'] == 'hexbin':
                artist = ax.hexbin(p, r, gridsize=70, mincnt=1, cmap='viridis', rasterized=True)
            else:
                d = np.asarray(payload.get('density', np.zeros((72, 72), dtype=np.float64)), dtype=np.float64)
                xedges = np.asarray(payload.get('xedges', np.linspace(0.0, 360.0, d.shape[0] + 1)), dtype=np.float64)
                yedges = np.asarray(payload.get('yedges', np.linspace(0.0, 360.0, d.shape[1] + 1)), dtype=np.float64)
                artist = ax.imshow(
                    d.T,
                    origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto',
                    cmap='viridis',
                    rasterized=True,
                )

            if shell_series is not None and len(panel_defs) == 1:
                shell_arr = np.asarray(shell_series)
                if shell_arr.shape == phase.shape:
                    shell_info = stable_category_color_map(shell_arr[mask])
                    ax.scatter(p, r, c=shell_info['codes'].astype(float), cmap=shell_info['cmap'],
                               s=2, alpha=0.25, linewidths=0, rasterized=True)

            ax.set_title(f"RAAN vs True Anomaly (TLE Kepler proxy)\n{panel['family_label']} | {panel['window_label']} | n={int(np.sum(mask))}")
            ax.set_xlabel('True Anomaly (TLE Kepler proxy, deg, wrapped)')
            ax.set_ylabel('RAAN (deg, wrapped)')
            fig.colorbar(artist, ax=ax, label='Density proxy')

        for j in range(len(panel_defs), axes.size):
            axes[j].set_visible(False)

        plt.show()
        if not return_results and not _preserve_open_figures_for_export():
            plt.close(fig)

    if return_results:
        return {
            'figure': fig,
            'mode': mode,
            'density_mode': effective_mode,
            'wrap_aware': True,
            'family_mode': family_mode,
            'family_assignment_mode': str(family_payload.get('assignment_mode', 'none')),
            'family_counts': family_payload.get('family_counts', {}),
            'family_targets_deg': family_payload.get('family_targets_deg'),
            'family_tolerance_deg': family_payload.get('family_tolerance_deg'),
            'time_window_count': int(len(windows)),
            'time_window_mode': str(windows[0].get('mode', 'aggregate') if windows else 'aggregate'),
            'sample_count': int(np.sum(finite)),
            'products': products,
            'phase_semantics': 'TLE-derived Kepler proxy from mean anomaly',
        }
    return None


def plot_sma_vs_inclination_density_shell_map(semi_major_axes, inclinations,
                                              shell_series=None, mode='hist2d',
                                              show_plots=True, return_results=False):
    """Plot semi-major axis vs inclination shell map/density."""
    sma = np.asarray(semi_major_axes, dtype=np.float64)
    inc = np.asarray(inclinations, dtype=np.float64)
    keep = np.isfinite(sma) & np.isfinite(inc) & (sma <= 7000.0) & (inc >= 40.0)
    sma_plot = sma[keep]
    inc_plot = inc[keep]

    fig, ax = plt.subplots(figsize=(8, 7))
    artist = None
    if sma_plot.size == 0:
        ax.set_title('Semi-major Axis vs Inclination Density/Shell Map (no data after filters)')
    elif str(mode) == 'hexbin':
        artist = ax.hexbin(sma_plot, inc_plot, gridsize=70, mincnt=1, cmap='viridis', rasterized=True)
    else:
        h = ax.hist2d(sma_plot, inc_plot, bins=70, cmap='viridis', rasterized=True)
        artist = h[-1]

    if shell_series is not None:
        shell_info = stable_category_color_map(shell_series)
        shell_codes = shell_info['codes'][keep]
        ax.scatter(sma_plot, inc_plot, c=shell_codes.astype(float), cmap=shell_info['cmap'],
                   s=2, alpha=0.30, linewidths=0, rasterized=True)

    if sma_plot.size > 0:
        ax.set_title('Semi-major Axis vs Inclination Density/Shell Map')
    ax.set_xlabel('Semi-major Axis (km)')
    ax.set_ylabel('Inclination (deg)')
    if artist is not None:
        plt.colorbar(artist, ax=ax, label='Density proxy')

    if show_plots:
        plt.show()
        if not return_results and not _preserve_open_figures_for_export():
            plt.close(fig)

    if return_results:
        return {'figure': fig, 'mode': mode}
    return None


def _local_maxima_mask_grid(surface_2d):
    arr = np.asarray(surface_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros_like(arr, dtype=bool)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=bool)

    work = np.where(finite, arr, -np.inf)
    if _USE_NUMBA and _local_maxima_mask_numba is not None and work.size > 0:
        try:
            return _local_maxima_mask_numba(work, finite)
        except Exception:
            pass

    n0, n1 = work.shape
    is_peak = np.zeros((n0, n1), dtype=bool)
    for i in range(n0):
        for j in range(n1):
            if not finite[i, j]:
                continue
            value = work[i, j]
            keep = True
            for di in (-1, 0, 1):
                ii = i + di
                if ii < 0 or ii >= n0:
                    continue
                for dj in (-1, 0, 1):
                    jj = j + dj
                    if di == 0 and dj == 0:
                        continue
                    if jj < 0 or jj >= n1:
                        continue
                    if value < work[ii, jj]:
                        keep = False
                        break
                if not keep:
                    break
            is_peak[i, j] = keep
    return is_peak


def _extract_density_hotspots(density, x_centers, y_centers, top_k=5,
                              support_counts=None, min_sep_x_bins=2,
                              min_sep_y_bins=2, circular_x=True):
    d = np.asarray(density, dtype=np.float64)
    x = np.asarray(x_centers, dtype=np.float64)
    y = np.asarray(y_centers, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != x.size or d.shape[1] != y.size:
        return []

    mask = _local_maxima_mask_grid(d)
    cand = np.argwhere(mask)
    if cand.size == 0:
        return []

    vals = d[cand[:, 0], cand[:, 1]]
    order = np.argsort(vals)[::-1]
    top_k = int(max(1, top_k))
    sep_x = int(max(0, min_sep_x_bins))
    sep_y = int(max(0, min_sep_y_bins))
    nx = int(x.size)

    if support_counts is not None:
        support = np.asarray(support_counts, dtype=np.float64)
        if support.shape != d.shape:
            support = None
    else:
        support = None

    out = []
    for idx in order.tolist():
        ix = int(cand[idx, 0])
        iy = int(cand[idx, 1])
        val = float(d[ix, iy])
        if not np.isfinite(val):
            continue

        too_close = False
        for row in out:
            jx = int(row['phase_index'])
            jy = int(row['altitude_index'])
            dx = abs(ix - jx)
            if circular_x and nx > 0:
                dx = min(dx, nx - dx)
            dy = abs(iy - jy)
            if dx < sep_x and dy < sep_y:
                too_close = True
                break
        if too_close:
            continue

        out.append(
            {
                'phase_index': ix,
                'altitude_index': iy,
                'phase_deg': float(x[ix]),
                'altitude_km': float(y[iy]),
                'density': val,
                'support_count': None if support is None else float(support[ix, iy]),
            }
        )
        if len(out) >= top_k:
            break

    for rank, row in enumerate(out, start=1):
        row['rank'] = int(rank)
    return out


def _build_phase_alt_density_product(phase_deg, alt_km, density_mode):
    mode = str(density_mode or 'hist2d').strip().lower()
    if mode in {'circular_linear_kde', 'circular_kde', 'circular_kde_placeholder'}:
        kde = circular_linear_density(phase_deg, alt_km, circular_bins=72, linear_bins=60, sigma=1.0)
        support, _, _ = np.histogram2d(phase_deg, alt_km, bins=[72, 60],
                                       range=[[0.0, 360.0], [kde['yedges'][0], kde['yedges'][-1]]])
        return {
            'density': np.asarray(kde['density'], dtype=np.float64),
            'xedges': np.asarray(kde['xedges'], dtype=np.float64),
            'yedges': np.asarray(kde['yedges'], dtype=np.float64),
            'xcenters': np.asarray(kde.get('theta_grid_deg', 0.5 * (kde['xedges'][:-1] + kde['xedges'][1:])), dtype=np.float64),
            'ycenters': np.asarray(kde.get('z_grid', 0.5 * (kde['yedges'][:-1] + kde['yedges'][1:])), dtype=np.float64),
            'effective_mode': 'circular_linear_kde',
            'support_counts': np.asarray(support, dtype=np.float64),
        }

    hist, xedges, yedges = np.histogram2d(phase_deg, alt_km, bins=[72, 60])
    if mode == 'hist2d':
        pad = circular_pad_histogram_2d(hist, pad_x=2, pad_y=2, circular_x=True, circular_y=False)
        density = gaussian_filter(pad, sigma=1.0)[2:-2, 2:-2]
        eff_mode = 'hist2d'
    else:
        density = hist
        eff_mode = 'hexbin' if mode == 'hexbin' else 'hist2d'

    return {
        'density': np.asarray(density, dtype=np.float64),
        'xedges': np.asarray(xedges, dtype=np.float64),
        'yedges': np.asarray(yedges, dtype=np.float64),
        'xcenters': 0.5 * (np.asarray(xedges[:-1], dtype=np.float64) + np.asarray(xedges[1:], dtype=np.float64)),
        'ycenters': 0.5 * (np.asarray(yedges[:-1], dtype=np.float64) + np.asarray(yedges[1:], dtype=np.float64)),
        'effective_mode': eff_mode,
        'support_counts': np.asarray(hist, dtype=np.float64),
    }


def _compute_hotspot_persistence(products_by_panel):
    by_family = {}
    for pid, item in products_by_panel.items():
        panel = item.get('panel', {})
        fam = str(panel.get('family_key', 'aggregate_all'))
        by_family.setdefault(fam, []).append((pid, item))

    out = {}
    for fam, rows in by_family.items():
        rows_sorted = sorted(
            rows,
            key=lambda kv: pd.to_datetime(kv[1].get('panel', {}).get('window_start'), errors='coerce')
            if kv[1].get('panel', {}).get('window_start') is not None else pd.Timestamp.min,
        )
        top_points = []
        for _, item in rows_sorted:
            hotspots = item.get('hotspots', [])
            top_points.append(hotspots[0] if hotspots else None)

        phase_drifts = []
        alt_drifts = []
        for a, b in zip(top_points[:-1], top_points[1:]):
            if a is None or b is None:
                continue
            phase_drifts.append(_circular_distance_deg(a.get('phase_deg', np.nan), b.get('phase_deg', np.nan)))
            alt_drifts.append(abs(float(a.get('altitude_km', np.nan)) - float(b.get('altitude_km', np.nan))))

        ref = next((x for x in top_points if x is not None), None)
        near_ref_count = 0
        valid_windows = 0
        if ref is not None:
            for item in rows_sorted:
                hotspots = item[1].get('hotspots', [])
                if not hotspots:
                    continue
                valid_windows += 1
                found = any(
                    _circular_distance_deg(h.get('phase_deg', np.nan), ref.get('phase_deg', np.nan)) <= 15.0
                    and abs(float(h.get('altitude_km', np.nan)) - float(ref.get('altitude_km', np.nan))) <= 20.0
                    for h in hotspots
                )
                if found:
                    near_ref_count += 1

        out[fam] = {
            'family_key': fam,
            'mean_nearest_hotspot_phase_drift_deg': float(np.nanmean(phase_drifts)) if phase_drifts else np.nan,
            'mean_nearest_hotspot_altitude_drift_km': float(np.nanmean(alt_drifts)) if alt_drifts else np.nan,
            'windows_with_hotspot_near_reference_fraction': float(near_ref_count / valid_windows) if valid_windows > 0 else np.nan,
            'valid_window_count': int(valid_windows),
        }
    return out


def plot_selected_phase_vs_altitude_density(phase_series, altitude_series,
                                            mode='hist2d', show_plots=True,
                                            return_results=False,
                                            timestamps=None,
                                            inclinations=None,
                                            family_labels=None,
                                            phase_alt_density_mode=None,
                                            phase_alt_density_family_mode='aggregate',
                                            phase_alt_density_family_targets_deg=None,
                                            phase_alt_density_family_tolerance_deg=0.4,
                                            phase_alt_density_time_windows=None,
                                            phase_alt_density_rolling_window_days=None,
                                            phase_alt_density_step_days=None,
                                            phase_alt_density_top_k_hotspots=5,
                                            phase_alt_density_return_arrays=True,
                                            phase_alt_density_normalization='per_panel',
                                            phase_alt_density_overlay_altitude_refs_km=None):
    """Plot True Anomaly (TLE Kepler proxy) vs altitude density with window/family diagnostics."""
    effective_mode = str(phase_alt_density_mode or mode or 'hist2d').strip().lower()
    if effective_mode not in {'hist2d', 'hexbin', 'circular_linear_kde', 'circular_kde', 'circular_kde_placeholder'}:
        effective_mode = 'hist2d'

    phase = wrap_degrees_360(np.asarray(phase_series, dtype=np.float64))
    alt = np.asarray(altitude_series, dtype=np.float64)
    finite = np.isfinite(phase) & np.isfinite(alt)

    if inclinations is not None:
        inc = np.asarray(inclinations, dtype=np.float64)
        if inc.shape != phase.shape:
            raise ValueError('inclinations must match phase_series shape')
    else:
        inc = None

    if timestamps is not None:
        ts = pd.to_datetime(np.asarray(timestamps), errors='coerce')
        if ts.shape != phase.shape:
            raise ValueError('timestamps must match phase_series shape')
    else:
        ts = np.asarray([pd.NaT] * phase.size, dtype='datetime64[ns]')
    ts_dt = pd.to_datetime(ts, errors='coerce')

    family_payload = assign_inclination_family_labels(
        inclinations=np.zeros_like(phase) if inc is None else inc,
        family_targets_deg=phase_alt_density_family_targets_deg,
        family_tolerance_deg=phase_alt_density_family_tolerance_deg,
        family_labels=family_labels,
    )
    family_arr = np.asarray(family_payload['family_labels'], dtype=object)

    family_mode = _normalize_family_mode(phase_alt_density_family_mode)
    family_entries = [('aggregate_all', np.ones(phase.shape, dtype=bool), 'Aggregate')]
    if family_mode in {'per_family', 'both'}:
        unique_families = [f for f in np.unique(family_arr).tolist() if str(f) != 'unmatched']
        if unique_families:
            entries = []
            for fam in unique_families:
                entries.append((str(fam), family_arr == fam, f"Family {fam}"))
            if family_mode == 'per_family':
                family_entries = entries
            else:
                family_entries.extend(entries)

    windows = _prepare_density_windows(
        timestamps=ts,
        explicit_windows=phase_alt_density_time_windows,
        rolling_window_days=phase_alt_density_rolling_window_days,
        step_days=phase_alt_density_step_days,
        max_windows=8,
    )
    if not windows:
        windows = [{'id': 'all', 'start': None, 'end': None, 'label': 'All times', 'mode': 'aggregate'}]

    panel_defs = []
    for win in windows:
        if win.get('start') is None:
            wmask = np.ones(phase.shape, dtype=bool)
        else:
            wmask = (ts_dt >= win['start']) & (ts_dt < win['end'])
        for fam_key, fam_mask, fam_label in family_entries:
            mask = finite & np.asarray(wmask, dtype=bool) & np.asarray(fam_mask, dtype=bool)
            panel_defs.append(
                {
                    'id': f"{win['id']}__{fam_key}",
                    'family_key': str(fam_key),
                    'family_label': str(fam_label),
                    'window_label': str(win.get('label', 'All times')),
                    'window_start': None if win.get('start') is None else str(win['start']),
                    'window_end': None if win.get('end') is None else str(win['end']),
                    'mask': mask,
                }
            )

    products = {}
    for panel in panel_defs:
        mask = np.asarray(panel['mask'], dtype=bool)
        product = _build_phase_alt_density_product(phase[mask], alt[mask], effective_mode)
        products[panel['id']] = {
            'panel': {
                'family_key': panel['family_key'],
                'family_label': panel['family_label'],
                'window_label': panel['window_label'],
                'window_start': panel['window_start'],
                'window_end': panel['window_end'],
            },
            'sample_count': int(np.sum(mask)),
            'density_mode_effective': str(product['effective_mode']),
            'density_raw': np.asarray(product['density'], dtype=np.float64),
            'support_counts': np.asarray(product['support_counts'], dtype=np.float64),
            'xedges': np.asarray(product['xedges'], dtype=np.float64),
            'yedges': np.asarray(product['yedges'], dtype=np.float64),
            'xcenters': np.asarray(product['xcenters'], dtype=np.float64),
            'ycenters': np.asarray(product['ycenters'], dtype=np.float64),
        }

    norm_mode = str(phase_alt_density_normalization or 'per_panel').strip().lower()
    family_raw_sum = {}
    for item in products.values():
        fam = item['panel']['family_key']
        family_raw_sum[fam] = float(family_raw_sum.get(fam, 0.0) + np.nansum(item['density_raw']))

    for pid, item in products.items():
        d = np.asarray(item['density_raw'], dtype=np.float64)
        if norm_mode == 'raw':
            d_norm = d
        elif norm_mode == 'per_family':
            denom = float(family_raw_sum.get(item['panel']['family_key'], np.nan))
            d_norm = d / denom if np.isfinite(denom) and denom > 0.0 else d
        else:
            denom = float(np.nansum(d))
            d_norm = d / denom if np.isfinite(denom) and denom > 0.0 else d

        item['density'] = d_norm
        item['hotspots'] = _extract_density_hotspots(
            d_norm,
            item['xcenters'],
            item['ycenters'],
            top_k=int(max(1, phase_alt_density_top_k_hotspots)),
            support_counts=item['support_counts'],
            min_sep_x_bins=2,
            min_sep_y_bins=2,
            circular_x=True,
        )

        if not bool(phase_alt_density_return_arrays):
            item.pop('density_raw', None)
            item.pop('support_counts', None)
            item.pop('xedges', None)
            item.pop('yedges', None)
            item.pop('xcenters', None)
            item.pop('ycenters', None)
            item.pop('density', None)

    persistence = _compute_hotspot_persistence(products)

    fig = None
    if show_plots:
        n_panels = len(panel_defs)
        n_cols = int(min(3, max(1, n_panels)))
        n_rows = int(np.ceil(n_panels / float(n_cols)))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        for idx, panel in enumerate(panel_defs):
            ax = axes[idx]
            payload = products[panel['id']]
            mask = np.asarray(panel['mask'], dtype=bool)
            p = phase[mask]
            z = alt[mask]
            artist = None

            if payload['density_mode_effective'] == 'hexbin':
                artist = ax.hexbin(p, z, gridsize=70, mincnt=1, cmap='viridis', rasterized=True)
            else:
                d = np.asarray(payload.get('density', payload.get('density_raw', np.zeros((72, 60), dtype=np.float64))), dtype=np.float64)
                xedges = np.asarray(payload.get('xedges', np.linspace(0.0, 360.0, d.shape[0] + 1)), dtype=np.float64)
                yedges = np.asarray(payload.get('yedges', np.linspace(np.nanmin(alt), np.nanmax(alt), d.shape[1] + 1)), dtype=np.float64)
                artist = ax.imshow(
                    d.T,
                    origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto',
                    cmap='viridis',
                    rasterized=True,
                )

            refs = phase_alt_density_overlay_altitude_refs_km
            if refs is not None:
                try:
                    refs_arr = np.asarray(refs, dtype=np.float64)
                except Exception:
                    refs_arr = np.asarray([], dtype=np.float64)
                for ref in refs_arr[np.isfinite(refs_arr)]:
                    ax.axhline(float(ref), color='#f97316', linestyle='--', linewidth=0.8, alpha=0.7)

            ax.set_title(
                f"True Anomaly (TLE Kepler proxy) vs Altitude\n"
                f"{panel['family_label']} | {panel['window_label']} | n={int(np.sum(mask))}"
            )
            ax.set_xlabel('True Anomaly (TLE Kepler proxy, deg, wrapped)')
            ax.set_ylabel('Altitude (km)')
            fig.colorbar(artist, ax=ax, label='Density proxy')

        for j in range(len(panel_defs), axes.size):
            axes[j].set_visible(False)

        plt.show()
        if not return_results and not _preserve_open_figures_for_export():
            plt.close(fig)

    if return_results:
        return {
            'figure': fig,
            'mode': mode,
            'density_mode': effective_mode,
            'circular_linear': True,
            'phase_wrap_mode': '0_to_360',
            'family_mode': family_mode,
            'family_assignment_mode': str(family_payload.get('assignment_mode', 'none')),
            'family_counts': family_payload.get('family_counts', {}),
            'family_targets_deg': family_payload.get('family_targets_deg'),
            'family_tolerance_deg': family_payload.get('family_tolerance_deg'),
            'time_window_count': int(len(windows)),
            'time_window_mode': str(windows[0].get('mode', 'aggregate') if windows else 'aggregate'),
            'density_normalization': norm_mode,
            'sample_count': int(np.sum(finite)),
            'altitude_range_km': [float(np.nanmin(alt[finite])) if np.any(finite) else np.nan,
                                  float(np.nanmax(alt[finite])) if np.any(finite) else np.nan],
            'hotspot_persistence_by_family': persistence,
            'products': products,
            'phase_semantics': 'TLE-derived Kepler proxy from mean anomaly',
        }
    return None


def plot_common_epoch_shell_snapshot(right_ascensions, phase_series, shell_series,
                                     altitude_series=None, seam_margin_deg=8.0,
                                     show_plots=True, return_results=False):
    """Plot a common-epoch snapshot with torus seam duplication and shell colors."""
    raan = wrap_degrees_360(np.asarray(right_ascensions, dtype=np.float64))
    phase = wrap_degrees_360(np.asarray(phase_series, dtype=np.float64))
    shell_info = stable_category_color_map(shell_series)

    dup = duplicate_torus_points_for_display(phase, raan, margin_deg=float(seam_margin_deg), wrap_mode='360')
    src = dup['source_index']

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        dup['x'],
        dup['y'],
        c=shell_info['codes'][src].astype(float),
        cmap=shell_info['cmap'],
        s=8,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title('Common-Epoch Shell Snapshot')
    ax.set_xlabel('True Anomaly (TLE Kepler proxy, deg, wrapped)')
    ax.set_ylabel('RAAN (deg, wrapped)')

    if altitude_series is not None:
        alt = np.asarray(altitude_series, dtype=np.float64)
        med_alt = float(np.nanmedian(alt)) if alt.size > 0 else float('nan')
        ax.text(0.01, 0.99, f'Median Altitude: {med_alt:.1f} km',
                transform=ax.transAxes, va='top', ha='left', fontsize=12)

    if show_plots:
        plt.show()
        if not return_results and not _preserve_open_figures_for_export():
            plt.close(fig)

    if return_results:
        return {'figure': fig, 'seam_margin_deg': seam_margin_deg}
    return None


def plot_starlink_dynamical_atlas(right_ascensions, phase_series,
                                  shell_series=None, altitude_residual=None,
                                  maneuver_score=None, resonance_residual=None,
                                  color_field='shell', show_plots=True,
                                  return_results=False):
    """Lightweight Starlink-oriented atlas on (RAAN, true-anomaly proxy)."""
    raan = wrap_degrees_360(np.asarray(right_ascensions, dtype=np.float64))
    phase = wrap_degrees_360(np.asarray(phase_series, dtype=np.float64))

    fig, ax = plt.subplots(figsize=(8, 7))

    color_field = str(color_field)
    cvals = None
    cmap = 'viridis'
    cbar_label = color_field

    if color_field == 'shell' and shell_series is not None:
        shell_info = stable_category_color_map(shell_series)
        cvals = shell_info['codes'].astype(float)
        cmap = shell_info['cmap']
        cbar_label = 'Shell ID code'
    elif color_field == 'altitude_residual' and altitude_residual is not None:
        cvals = np.asarray(altitude_residual, dtype=np.float64)
        cbar_label = 'Altitude residual (km)'
    elif color_field == 'maneuver_score' and maneuver_score is not None:
        cvals = np.asarray(maneuver_score, dtype=np.float64)
        cbar_label = 'Maneuver score proxy'
    elif color_field == 'resonance_residual' and resonance_residual is not None:
        cvals = np.asarray(resonance_residual, dtype=np.float64)
        cbar_label = 'Resonance residual proxy'
    else:
        cvals = np.zeros_like(phase)
        cbar_label = 'Default code'

    artist = ax.scatter(phase, raan, c=cvals, cmap=cmap, s=5, linewidths=0, rasterized=True)
    ax.set_title('Starlink-Oriented Dynamical Atlas (RAAN vs True Anomaly (TLE Kepler proxy))')
    ax.set_xlabel('True Anomaly (TLE Kepler proxy, deg, wrapped)')
    ax.set_ylabel('RAAN (deg, wrapped)')
    plt.colorbar(artist, ax=ax, label=cbar_label)

    if show_plots:
        plt.show()
        if not return_results and not _preserve_open_figures_for_export():
            plt.close(fig)

    if return_results:
        return {'figure': fig, 'color_field': color_field}
    return None

# Consolidated from inc_versus_sma.py
def inc_versus_sma(semi_major_axes, inclinations, fileNames, filenames_array,
                   metric_mode='euclidean', render_mode='scatter',
                   shell_series=None, show_shell_centroids=False,
                   show_plots=True, return_figures=False, return_results=False,
                   reference_lines=True, reference_annotation=False,
                   reference_markers=True,
                   target_sma_tolerance_km=25.0,
                   target_inclination_tolerance_deg=0.4,
                   target_profiles=None,
                   focused_profiles=True,
                   assignment_mode='joint_nearest',
                   standardized_robust_scale=False,
                   mean_lines=True,
                   median_lines=True):
    """
    Plot focused Gen1 Inclination-vs-Semi-major-Axis families and distance metrics.

    Existing inputs and return contract are preserved. New options are additive.
    """
    t0 = perf_counter()
    semi_major_axes = np.asarray(semi_major_axes, dtype=np.float64)
    inclinations = np.asarray(inclinations, dtype=np.float64)
    filenames_array = np.asarray(filenames_array)

    n_rows = semi_major_axes.size
    if n_rows == 0:
        print("[inc_versus_sma] No data to plot.")
        return

    metric_mode_requested_raw = str(metric_mode)
    render_mode_requested_raw = str(render_mode)
    metric_mode_requested = _normalize_inc_sma_metric_mode(metric_mode)
    render_mode_requested = _normalize_inc_sma_render_mode(render_mode)
    # Preserve signature compatibility while disabling mean/median overlays.
    mean_lines = False
    median_lines = False

    try:
        target_sma_tolerance_km = float(target_sma_tolerance_km)
    except Exception:
        target_sma_tolerance_km = 25.0
    if not np.isfinite(target_sma_tolerance_km) or target_sma_tolerance_km <= 0.0:
        target_sma_tolerance_km = 25.0

    try:
        target_inclination_tolerance_deg = float(target_inclination_tolerance_deg)
    except Exception:
        target_inclination_tolerance_deg = 0.4
    if not np.isfinite(target_inclination_tolerance_deg) or target_inclination_tolerance_deg <= 0.0:
        target_inclination_tolerance_deg = 0.4

    target_profiles_runtime = _normalize_inc_sma_target_profiles(target_profiles)
    if bool(focused_profiles):
        family_order = [key for key in ["53", "70", "97"] if key in target_profiles_runtime]
        if not family_order:
            family_order = sorted(target_profiles_runtime.keys())
    else:
        merged_targets = []
        seen_targets = set()
        for family_key in ["53", "70", "97"] + sorted(target_profiles_runtime.keys()):
            if family_key not in target_profiles_runtime:
                continue
            for point in target_profiles_runtime[family_key]:
                token = (
                    str(point.get("label", "")),
                    round(float(point.get("target_sma_km", np.nan)), 6),
                    round(float(point.get("target_inc_deg", np.nan)), 6),
                )
                if token in seen_targets:
                    continue
                seen_targets.add(token)
                merged_targets.append(dict(point))
        if merged_targets:
            target_profiles_runtime = {"all": merged_targets}
            family_order = ["all"]
        else:
            family_order = []

    if not family_order:
        family_order = ["all"]
        target_profiles_runtime = {
            "all": [
                {"label": "53.2@540", "target_sma_km": 6918.137, "target_inc_deg": 53.2},
                {"label": "53.0@550", "target_sma_km": 6928.137, "target_inc_deg": 53.0},
                {"label": "70.0@570", "target_sma_km": 6948.137, "target_inc_deg": 70.0},
                {"label": "97.6@560", "target_sma_km": 6938.137, "target_inc_deg": 97.6},
            ]
        }

    # Keep current compatibility and expose an explicit All Files option if missing.
    display_names = list(fileNames)
    if 'All Files' not in display_names:
        display_names.append('All Files')

    print(f"[inc_versus_sma] Preparing {n_rows:,} records...")

    # Memory-safe grouping: store one permutation + per-file slice bounds.
    order = np.argsort(filenames_array, kind='mergesort')
    sorted_names = filenames_array[order]
    unique_names, start_idx = np.unique(sorted_names, return_index=True)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = order.size
    bounds = {
        name: (int(s), int(e))
        for name, s, e in zip(unique_names.tolist(), start_idx.tolist(), end_idx.tolist())
    }
    all_indices = np.arange(n_rows, dtype=np.int64)

    def get_indices(selected_name):
        if selected_name == 'All Files':
            return all_indices
        window = bounds.get(selected_name)
        if window is None:
            return np.empty(0, dtype=np.int64)
        s, e = window
        return order[s:e]

    def safe_limits(x, y):
        if x.size == 0 or y.size == 0:
            return None
        return (
            float(np.min(x)), float(np.max(x)),
            float(np.min(y)), float(np.max(y)),
        )

    def apply_inc_sma_filters(x_vals, y_vals):
        x_arr = np.asarray(x_vals, dtype=np.float64)
        y_arr = np.asarray(y_vals, dtype=np.float64)
        keep = np.isfinite(x_arr) & np.isfinite(y_arr)
        keep &= x_arr <= 7000.0
        keep &= y_arr >= 40.0
        return keep

    def _set_family_limits(ax, family_key, target_points, x_vals, y_vals):
        fixed_window = DEFAULT_INC_SMA_FOCUS_WINDOWS.get(str(family_key))
        if isinstance(fixed_window, dict):
            xlim = fixed_window.get('xlim')
            ylim = fixed_window.get('ylim')
            if isinstance(xlim, (tuple, list)) and len(xlim) == 2 and isinstance(ylim, (tuple, list)) and len(ylim) == 2:
                ax.set_xlim(float(xlim[0]), float(xlim[1]))
                ax.set_ylim(float(ylim[0]), float(ylim[1]))
                return

        if target_points:
            target_sma_vals = np.asarray([p['target_sma_km'] for p in target_points], dtype=np.float64)
            target_inc_vals = np.asarray([p['target_inc_deg'] for p in target_points], dtype=np.float64)
            if target_sma_vals.size > 0 and target_inc_vals.size > 0:
                x_pad = max(85.0, float(target_sma_tolerance_km) * 3.5)
                y_pad = max(1.2, float(target_inclination_tolerance_deg) * 4.0)
                ax.set_xlim(float(np.min(target_sma_vals) - x_pad), float(np.max(target_sma_vals) + x_pad))
                ax.set_ylim(float(np.min(target_inc_vals) - y_pad), float(np.max(target_inc_vals) + y_pad))
                return

        lims = safe_limits(x_vals, y_vals)
        if lims is not None:
            ax.set_xlim(lims[0], lims[1])
            ax.set_ylim(lims[2], lims[3])

    dense_min_points = 80

    def _create_standard_artist(ax, x_vals, y_vals):
        x_arr = np.asarray(x_vals, dtype=np.float64)
        y_arr = np.asarray(y_vals, dtype=np.float64)
        requested_mode = render_mode_requested
        effective_mode = requested_mode
        sparse_fallback = False
        fallback_reason = None

        if effective_mode != 'scatter' and x_arr.size < dense_min_points:
            effective_mode = 'scatter'
            sparse_fallback = True
            fallback_reason = 'sparse_subset'

        if effective_mode == 'hexbin' and x_arr.size > 0:
            artist = ax.hexbin(
                x_arr,
                y_arr,
                gridsize=65,
                mincnt=1,
                cmap='viridis',
                rasterized=True,
            )
        elif effective_mode == 'hist2d' and x_arr.size > 0:
            if float(np.min(x_arr)) == float(np.max(x_arr)) or float(np.min(y_arr)) == float(np.max(y_arr)):
                effective_mode = 'scatter'
                sparse_fallback = True
                fallback_reason = 'degenerate_range'
                artist = ax.scatter(x_arr, y_arr, s=5, linewidths=0, rasterized=True, color='tab:blue', alpha=0.85)
            else:
                h = ax.hist2d(
                    x_arr,
                    y_arr,
                    bins=60,
                    cmin=1,
                    cmap='viridis',
                    rasterized=True,
                )
                artist = h[-1]
        else:
            artist = ax.scatter(
                x_arr,
                y_arr,
                s=5,
                linewidths=0,
                rasterized=True,
                color='tab:blue',
                alpha=0.85,
            )

        return {
            'artist': artist,
            'requested_render_mode': requested_mode,
            'effective_render_mode': effective_mode,
            'sparse_fallback': bool(sparse_fallback),
            'fallback_reason': fallback_reason,
            'point_count': int(x_arr.size),
        }

    def _create_metric_artist(ax, x_vals, y_vals, metric_vals):
        x_arr = np.asarray(x_vals, dtype=np.float64)
        y_arr = np.asarray(y_vals, dtype=np.float64)
        m_arr = np.asarray(metric_vals, dtype=np.float64)

        requested_mode = render_mode_requested
        effective_mode = requested_mode
        sparse_fallback = False
        fallback_reason = None

        if x_arr.size != m_arr.size:
            m_arr = np.zeros(x_arr.size, dtype=np.float64)

        if effective_mode != 'scatter' and x_arr.size < dense_min_points:
            effective_mode = 'scatter'
            sparse_fallback = True
            fallback_reason = 'sparse_subset'

        if effective_mode == 'hexbin' and x_arr.size > 0:
            artist = ax.hexbin(
                x_arr,
                y_arr,
                C=m_arr,
                reduce_C_function=np.mean,
                gridsize=65,
                mincnt=1,
                cmap='viridis',
                rasterized=True,
            )
        elif effective_mode == 'hist2d' and x_arr.size > 0:
            if float(np.min(x_arr)) == float(np.max(x_arr)) or float(np.min(y_arr)) == float(np.max(y_arr)):
                effective_mode = 'scatter'
                sparse_fallback = True
                fallback_reason = 'degenerate_range'
                artist = ax.scatter(x_arr, y_arr, c=m_arr, cmap='viridis', s=5, linewidths=0, rasterized=True)
            else:
                counts, x_edges, y_edges = np.histogram2d(x_arr, y_arr, bins=60)
                weighted, _, _ = np.histogram2d(x_arr, y_arr, bins=[x_edges, y_edges], weights=m_arr)
                with np.errstate(invalid='ignore', divide='ignore'):
                    mean_metric = np.divide(
                        weighted,
                        counts,
                        out=np.full_like(weighted, np.nan, dtype=np.float64),
                        where=counts > 0,
                    )
                masked = np.ma.masked_invalid(mean_metric)
                if masked.count() == 0:
                    effective_mode = 'scatter'
                    sparse_fallback = True
                    fallback_reason = 'no_hist_cells'
                    artist = ax.scatter(x_arr, y_arr, c=m_arr, cmap='viridis', s=5, linewidths=0, rasterized=True)
                else:
                    artist = ax.pcolormesh(
                        x_edges,
                        y_edges,
                        masked.T,
                        cmap='viridis',
                        shading='auto',
                        rasterized=True,
                    )
        else:
            artist = ax.scatter(x_arr, y_arr, c=m_arr, cmap='viridis', s=5, linewidths=0, rasterized=True)

        return {
            'artist': artist,
            'requested_render_mode': requested_mode,
            'effective_render_mode': effective_mode,
            'sparse_fallback': bool(sparse_fallback),
            'fallback_reason': fallback_reason,
            'point_count': int(x_arr.size),
        }

    def _clear_overlay_state(state):
        for key in ['target_lines', 'mean_lines', 'median_lines', 'target_markers', 'centroids']:
            artists = list(state.get(key, []))
            for artist in artists:
                _remove_artist_safe(artist)
            state[key] = []
        if state.get('legend') is not None:
            try:
                state['legend'].remove()
            except Exception:
                pass
            state['legend'] = None
        if state.get('annotation') is not None:
            _remove_artist_safe(state['annotation'])
            state['annotation'] = None

    def _apply_reference_overlays(ax, state, target_points, reference_stats):
        _clear_overlay_state(state)

        flags = {
            'target_lines_drawn': False,
            'mean_lines_drawn': False,
            'median_lines_drawn': False,
            'target_markers_drawn': False,
            'annotation_drawn': False,
        }

        if bool(reference_lines):
            for point in target_points:
                state['target_lines'].append(
                    ax.axvline(
                        float(point['target_sma_km']),
                        linestyle='--',
                        linewidth=1.2,
                        color='black',
                        alpha=0.7,
                        zorder=3,
                    )
                )
                state['target_lines'].append(
                    ax.axhline(
                        float(point['target_inc_deg']),
                        linestyle='--',
                        linewidth=1.2,
                        color='black',
                        alpha=0.7,
                        zorder=3,
                    )
                )
            flags['target_lines_drawn'] = len(state['target_lines']) > 0

        if bool(reference_markers):
            for point in target_points:
                state['target_markers'].append(
                    ax.scatter(
                        [float(point['target_sma_km'])],
                        [float(point['target_inc_deg'])],
                        marker='x',
                        c='black',
                        s=64,
                        linewidths=1.6,
                        zorder=6,
                    )
                )
            flags['target_markers_drawn'] = len(state['target_markers']) > 0

        flags['mean_lines_drawn'] = len(state['mean_lines']) > 0
        flags['median_lines_drawn'] = len(state['median_lines']) > 0

        legend_handles = []
        if flags['target_lines_drawn']:
            legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.2, label='Targets'))
        if flags['target_markers_drawn']:
            legend_handles.append(Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=8.5, label='Target Point'))

        if legend_handles:
            unique = {}
            for handle in legend_handles:
                unique[handle.get_label()] = handle
            state['legend'] = ax.legend(
                handles=list(unique.values()),
                labels=list(unique.keys()),
                loc='upper left',
                fontsize=12,
                framealpha=0.85,
            )

        if bool(reference_annotation):
            annotation_text = _format_inc_sma_reference_annotation(reference_stats)
            if annotation_text:
                state['annotation'] = ax.text(
                    0.01,
                    0.99,
                    annotation_text,
                    transform=ax.transAxes,
                    va='top',
                    ha='left',
                    fontsize=12,
                    bbox={
                        'boxstyle': 'round,pad=0.2',
                        'facecolor': 'white',
                        'alpha': 0.78,
                        'edgecolor': 'none',
                    },
                )
        flags['annotation_drawn'] = state.get('annotation') is not None
        return flags

    def _update_centroids(ax, state, shell_codes, x_vals, y_vals):
        for artist in list(state.get('centroids', [])):
            _remove_artist_safe(artist)
        state['centroids'] = []

        if not bool(show_shell_centroids) or shell_color_info is None:
            return
        if x_vals.size == 0 or y_vals.size == 0 or shell_codes.size == 0:
            return

        for code in np.unique(shell_codes):
            members = shell_codes == code
            if not np.any(members):
                continue
            cx = float(np.median(x_vals[members]))
            cy = float(np.median(y_vals[members]))
            state['centroids'].append(
                ax.scatter(
                    [cx],
                    [cy],
                    s=55,
                    c=[float(code)],
                    cmap=shell_color_info['cmap'],
                    marker='x',
                    linewidths=1.25,
                    zorder=6,
                )
            )

    def _build_figure_metadata(
        family_key,
        family_label,
        figure_role,
        selected_filename,
        artist_payload,
        metric_payload,
        reference_stats,
        overlay_flags,
        target_points,
    ):
        return {
            'family_key': str(family_key),
            'family_label': str(family_label),
            'figure_role': str(figure_role),
            'selected_file': str(selected_filename),
            'requested_metric_mode': metric_mode_requested_raw,
            'active_metric_mode': metric_payload.get('metric_mode_active'),
            'requested_render_mode': render_mode_requested_raw,
            'active_render_mode': artist_payload.get('effective_render_mode'),
            'sparse_fallback_occurred': bool(artist_payload.get('sparse_fallback', False)),
            'sparse_fallback_reason': artist_payload.get('fallback_reason'),
            'target_operating_points': [
                {
                    'label': point['label'],
                    'target_sma_km': float(point['target_sma_km']),
                    'target_inc_deg': float(point['target_inc_deg']),
                }
                for point in target_points
            ],
            'assignment_tolerances': {
                'sma_tolerance_km': float(reference_stats.get('sma_tolerance_km', target_sma_tolerance_km)),
                'inclination_tolerance_deg': float(reference_stats.get('inclination_tolerance_deg', target_inclination_tolerance_deg)),
                'assignment_mode_requested': reference_stats.get('assignment_mode_requested'),
                'assignment_mode_effective': reference_stats.get('assignment_mode_effective'),
                'joint_distance_threshold': float(reference_stats.get('joint_distance_threshold', np.sqrt(2.0))),
            },
            'per_target_stats': list(reference_stats.get('groups', [])),
            'omitted_targets': list(reference_stats.get('omitted_targets', [])),
            'assigned_count': int(reference_stats.get('assigned_count', 0)),
            'unassigned_count': int(reference_stats.get('unassigned_count', 0)),
            'target_lines_drawn': bool(overlay_flags.get('target_lines_drawn', False)),
            'mean_lines_drawn': bool(overlay_flags.get('mean_lines_drawn', False)),
            'median_lines_drawn': bool(overlay_flags.get('median_lines_drawn', False)),
            'target_markers_drawn': bool(overlay_flags.get('target_markers_drawn', False)),
            'annotation_drawn': bool(overlay_flags.get('annotation_drawn', False)),
            'metric_label': metric_payload.get('metric_label'),
            'metric_numerical_fallback': bool(metric_payload.get('numerical_fallback', False)),
            'metric_fallback_reason': metric_payload.get('fallback_reason'),
        }

    shell_color_info = None
    if shell_series is not None:
        shell_arr = np.asarray(shell_series)
        if shell_arr.shape != semi_major_axes.shape:
            raise ValueError('shell_series must match semi_major_axes shape')
        shell_color_info = stable_category_color_map(shell_arr)

    initial_idx = display_names.index('All Files') if 'All Files' in display_names else 0
    init_name = display_names[initial_idx]
    family_name_map = {
        '53': '53-degree Family',
        '70': '70-degree Family',
        '97': '97-degree Family',
        'all': 'All Families',
    }

    family_states = {}
    family_metadata = {}
    created_figures = []

    for family_key in family_order:
        target_points = list(target_profiles_runtime.get(family_key, []))
        family_label = family_name_map.get(str(family_key), str(family_key))

        init_indices = get_indices(init_name)
        x0_raw = semi_major_axes[init_indices]
        y0_raw = inclinations[init_indices]
        keep0 = apply_inc_sma_filters(x0_raw, y0_raw)
        x0 = x0_raw[keep0]
        y0 = y0_raw[keep0]

        metric0_payload = _compute_inc_sma_metric_values(
            x0,
            y0,
            metric_mode=metric_mode_requested,
            standardized_robust_scale=bool(standardized_robust_scale),
            nondim_sma_scale_km=float(target_sma_tolerance_km),
            nondim_inc_scale_deg=float(target_inclination_tolerance_deg),
        )
        reference0_stats = compute_inc_sma_reference_stats(
            x0,
            y0,
            target_points,
            sma_tolerance_km=float(target_sma_tolerance_km),
            inclination_tolerance_deg=float(target_inclination_tolerance_deg),
            assignment_mode=assignment_mode,
        )

        shell_codes0 = np.asarray([], dtype=np.float64)
        if shell_color_info is not None and init_indices.size > 0:
            shell_codes0 = np.asarray(shell_color_info['codes'][init_indices][keep0], dtype=np.float64)

        fig_metric, ax_metric = plt.subplots()
        created_figures.append(fig_metric)
        try:
            fig_metric.canvas.manager.set_window_title(f"Distance-Metric Analysis ({family_label})")
        except Exception:
            pass
        metric_artist_payload = _create_metric_artist(ax_metric, x0, y0, metric0_payload['distances'])
        cbar_metric = plt.colorbar(metric_artist_payload['artist'], ax=ax_metric, label=metric0_payload['metric_label'])
        ax_metric.set_title(f'Distance-Metric Analysis of Inclination vs. Semi-major Axis ({family_label})')
        ax_metric.set_xlabel('Semi-major Axis (km)')
        ax_metric.set_ylabel('Inclination (degrees)')

        ax_slider_metric = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig_metric)
        slider_metric = Slider(
            ax_slider_metric,
            f'File Index ({family_key} metric)',
            0,
            len(display_names) - 1,
            valinit=initial_idx,
            valstep=1,
        )
        _set_family_limits(ax_metric, family_key, target_points, x0, y0)

        fig_standard, ax_standard = plt.subplots()
        created_figures.append(fig_standard)
        try:
            fig_standard.canvas.manager.set_window_title(f"Inclination vs. Semi-major Axis ({family_label})")
        except Exception:
            pass
        standard_artist_payload = _create_standard_artist(ax_standard, x0, y0)
        ax_standard.set_title(f'Inclination vs. Semi-major Axis ({family_label})')
        ax_standard.set_xlabel('Semi-major Axis (km)')
        ax_standard.set_ylabel('Inclination (degrees)')

        shell_overlay_artist = None
        if shell_color_info is not None and x0.size > 0:
            shell_overlay_artist = ax_standard.scatter(
                x0,
                y0,
                c=shell_codes0.astype(float),
                cmap=shell_color_info['cmap'],
                s=3,
                linewidths=0,
                alpha=0.45,
                rasterized=True,
            )

        ax_slider_standard = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig_standard)
        slider_standard = Slider(
            ax_slider_standard,
            f'File Index ({family_key} standard)',
            0,
            len(display_names) - 1,
            valinit=initial_idx,
            valstep=1,
        )
        _set_family_limits(ax_standard, family_key, target_points, x0, y0)

        metric_state = {
            'artist': metric_artist_payload['artist'],
            'colorbar': cbar_metric,
            'target_lines': [],
            'mean_lines': [],
            'median_lines': [],
            'target_markers': [],
            'centroids': [],
            'legend': None,
            'annotation': None,
        }
        standard_state = {
            'artist': standard_artist_payload['artist'],
            'shell_overlay': shell_overlay_artist,
            'target_lines': [],
            'mean_lines': [],
            'median_lines': [],
            'target_markers': [],
            'centroids': [],
            'legend': None,
            'annotation': None,
        }

        metric_overlay_flags = _apply_reference_overlays(ax_metric, metric_state, target_points, reference0_stats)
        standard_overlay_flags = _apply_reference_overlays(ax_standard, standard_state, target_points, reference0_stats)
        _update_centroids(ax_metric, metric_state, shell_codes0, x0, y0)

        family_metadata[family_key] = {
            'family_label': family_label,
            'metric': _build_figure_metadata(
                family_key,
                family_label,
                'metric',
                init_name,
                metric_artist_payload,
                metric0_payload,
                reference0_stats,
                metric_overlay_flags,
                target_points,
            ),
            'standard': _build_figure_metadata(
                family_key,
                family_label,
                'standard',
                init_name,
                standard_artist_payload,
                metric0_payload,
                reference0_stats,
                standard_overlay_flags,
                target_points,
            ),
        }

        def update_metric_plot(
            val,
            _family_key=family_key,
            _target_points=target_points,
            _slider=slider_metric,
            _family_label=family_label,
            _ax=ax_metric,
            _fig=fig_metric,
            _state=metric_state,
        ):
            idx = int(_slider.val)
            selected_filename = display_names[idx]
            t_update = perf_counter()
            indices = get_indices(selected_filename)

            raw_sma = semi_major_axes[indices]
            raw_inc = inclinations[indices]
            keep = apply_inc_sma_filters(raw_sma, raw_inc)
            x_sel = raw_sma[keep]
            y_sel = raw_inc[keep]

            metric_payload = _compute_inc_sma_metric_values(
                x_sel,
                y_sel,
                metric_mode=metric_mode_requested,
                standardized_robust_scale=bool(standardized_robust_scale),
                nondim_sma_scale_km=float(target_sma_tolerance_km),
                nondim_inc_scale_deg=float(target_inclination_tolerance_deg),
            )
            reference_stats = compute_inc_sma_reference_stats(
                x_sel,
                y_sel,
                _target_points,
                sma_tolerance_km=float(target_sma_tolerance_km),
                inclination_tolerance_deg=float(target_inclination_tolerance_deg),
                assignment_mode=assignment_mode,
            )

            _remove_artist_safe(_state['artist'])
            artist_payload = _create_metric_artist(_ax, x_sel, y_sel, metric_payload['distances'])
            _state['artist'] = artist_payload['artist']
            _state['colorbar'].update_normal(_state['artist'])
            _state['colorbar'].set_label(metric_payload['metric_label'])

            overlay_flags = _apply_reference_overlays(_ax, _state, _target_points, reference_stats)

            shell_codes = np.asarray([], dtype=np.float64)
            if shell_color_info is not None and indices.size > 0:
                shell_codes = np.asarray(shell_color_info['codes'][indices][keep], dtype=np.float64)
            _update_centroids(_ax, _state, shell_codes, x_sel, y_sel)

            _set_family_limits(_ax, _family_key, _target_points, x_sel, y_sel)
            _fig.canvas.draw_idle()

            family_metadata[_family_key]['metric'] = _build_figure_metadata(
                _family_key,
                _family_label,
                'metric',
                selected_filename,
                artist_payload,
                metric_payload,
                reference_stats,
                overlay_flags,
                _target_points,
            )
            print(
                f"[inc_versus_sma] Metric plot ({_family_key}) -> {selected_filename} "
                f"({x_sel.size:,} pts) in {perf_counter() - t_update:.2f}s"
            )

        def update_standard_plot(
            val,
            _family_key=family_key,
            _target_points=target_points,
            _slider=slider_standard,
            _family_label=family_label,
            _ax=ax_standard,
            _fig=fig_standard,
            _state=standard_state,
        ):
            idx = int(_slider.val)
            selected_filename = display_names[idx]
            t_update = perf_counter()
            indices = get_indices(selected_filename)

            raw_sma = semi_major_axes[indices]
            raw_inc = inclinations[indices]
            keep = apply_inc_sma_filters(raw_sma, raw_inc)
            x_sel = raw_sma[keep]
            y_sel = raw_inc[keep]

            metric_payload = _compute_inc_sma_metric_values(
                x_sel,
                y_sel,
                metric_mode=metric_mode_requested,
                standardized_robust_scale=bool(standardized_robust_scale),
                nondim_sma_scale_km=float(target_sma_tolerance_km),
                nondim_inc_scale_deg=float(target_inclination_tolerance_deg),
            )
            reference_stats = compute_inc_sma_reference_stats(
                x_sel,
                y_sel,
                _target_points,
                sma_tolerance_km=float(target_sma_tolerance_km),
                inclination_tolerance_deg=float(target_inclination_tolerance_deg),
                assignment_mode=assignment_mode,
            )

            _remove_artist_safe(_state['artist'])
            artist_payload = _create_standard_artist(_ax, x_sel, y_sel)
            _state['artist'] = artist_payload['artist']

            if _state['shell_overlay'] is not None:
                _remove_artist_safe(_state['shell_overlay'])
                _state['shell_overlay'] = None

            if shell_color_info is not None and indices.size > 0 and x_sel.size > 0:
                shell_codes = np.asarray(shell_color_info['codes'][indices][keep], dtype=np.float64)
                _state['shell_overlay'] = _ax.scatter(
                    x_sel,
                    y_sel,
                    c=shell_codes.astype(float),
                    cmap=shell_color_info['cmap'],
                    s=3,
                    linewidths=0,
                    alpha=0.45,
                    rasterized=True,
                )

            overlay_flags = _apply_reference_overlays(_ax, _state, _target_points, reference_stats)
            _set_family_limits(_ax, _family_key, _target_points, x_sel, y_sel)
            _fig.canvas.draw_idle()

            family_metadata[_family_key]['standard'] = _build_figure_metadata(
                _family_key,
                _family_label,
                'standard',
                selected_filename,
                artist_payload,
                metric_payload,
                reference_stats,
                overlay_flags,
                _target_points,
            )
            print(
                f"[inc_versus_sma] Inc-vs-SMA plot ({_family_key}) -> {selected_filename} "
                f"({x_sel.size:,} pts) in {perf_counter() - t_update:.2f}s"
            )

        slider_metric.on_changed(update_metric_plot)
        slider_standard.on_changed(update_standard_plot)

        family_states[family_key] = {
            'metric_figure': fig_metric,
            'standard_figure': fig_standard,
            'metric_slider': slider_metric,
            'standard_slider': slider_standard,
        }

    print(f"[inc_versus_sma] Ready in {perf_counter() - t0:.2f}s")

    if show_plots:
        plt.show()

    first_family = family_order[0] if family_order else None
    primary_metric_meta = family_metadata.get(first_family, {}).get('metric', {}) if first_family else {}
    primary_standard_meta = family_metadata.get(first_family, {}).get('standard', {}) if first_family else {}

    figures_payload = {
        'metric_analysis': None,
        'inc_vs_sma': None,
        'focused_pairs': {},
    }

    if return_figures:
        for family_key in family_order:
            state = family_states[family_key]
            figures_payload['focused_pairs'][family_key] = {
                'metric_analysis': state['metric_figure'],
                'inc_vs_sma': state['standard_figure'],
            }
        if first_family is not None:
            figures_payload['metric_analysis'] = family_states[first_family]['metric_figure']
            figures_payload['inc_vs_sma'] = family_states[first_family]['standard_figure']

    payload = {
        'metadata': {
            'metric_mode': metric_mode_requested_raw,
            'metric_label': primary_metric_meta.get('metric_label', 'Euclidean distance'),
            'render_mode': render_mode_requested_raw,
            'points_total': int(n_rows),
            'has_shell_overlay': bool(shell_color_info is not None),
            'requested_metric_mode': metric_mode_requested_raw,
            'active_metric_mode': primary_metric_meta.get('active_metric_mode', metric_mode_requested),
            'requested_render_mode': render_mode_requested_raw,
            'active_render_mode': primary_standard_meta.get('active_render_mode', render_mode_requested),
            'sparse_fallback_occurred': bool(
                primary_metric_meta.get('sparse_fallback_occurred', False)
                or primary_standard_meta.get('sparse_fallback_occurred', False)
            ),
            'target_operating_points_used': {
                key: [
                    {
                        'label': point['label'],
                        'target_sma_km': float(point['target_sma_km']),
                        'target_inc_deg': float(point['target_inc_deg']),
                    }
                    for point in target_profiles_runtime.get(key, [])
                ]
                for key in family_order
            },
            'assignment_tolerances_used': {
                'sma_tolerance_km': float(target_sma_tolerance_km),
                'inclination_tolerance_deg': float(target_inclination_tolerance_deg),
                'assignment_mode': str(assignment_mode),
            },
            'reference_lines_enabled': bool(reference_lines),
            'reference_markers_enabled': bool(reference_markers),
            'reference_annotation_enabled': bool(reference_annotation),
            'mean_lines_enabled': False,
            'median_lines_enabled': False,
            'focused_profiles_enabled': bool(focused_profiles),
            'figure_metadata': family_metadata,
        },
        'figures': figures_payload,
    }

    if not return_figures and not _preserve_open_figures_for_export():
        for figure in created_figures:
            plt.close(figure)

    if return_results:
        return payload
    return None


# Consolidated from density_ra_versus_arg.py
def density_ra_versus_arg(
    args_of_perigee,
    right_ascensions,
    fileNames,
    filenames_array,
    phase_mode=None,
    phase_series=None,
    shell_series=None,
    mode='hist2d',
    smoothing_sigma=1.0,
    x_wrap_mode='360',
    y_wrap_mode='360',
    add_shell_overlay=False,
    show_plots=True,
    return_figures=False,
    return_results=False,
):
    """
    Plot the density of Right Ascension of the Ascending Node versus the
    Argument of Perigee for a set of satellites, with a slider to filter
    by file name.

    Parameters:
        args_of_perigee (np.array): The argument of perigee of the satellites in degrees.
        right_ascensions (np.array): The right ascension of the ascending node in degrees.
        fileNames (list): The names of the files containing the TLE data. 
                          Must include 'All Files' if you want to show all data.
        filenames_array (np.array): The file index or file name for each satellite.

    Returns:
        None

    Migration Notes:
        Default behavior remains RAAN vs argument of perigee.
        Optional `phase_mode` and `phase_series` allow plotting RAAN against a
        low-e-safe phase variable while keeping old callers unchanged.
    """
    t0 = perf_counter()
    # Convert data to NumPy arrays (if not already)
    args_of_perigee = np.asarray(args_of_perigee)
    right_ascensions = np.asarray(right_ascensions)
    filenames_array = np.asarray(filenames_array)

    x_data = args_of_perigee
    x_label = 'Argument of Perigee (degrees)'
    title = 'Density Plot: RAAN vs. Argument of Perigee'
    plotted_phase_variable = 'argument_of_perigee_deg'
    if phase_mode is not None and phase_series is not None:
        phase_arr = np.asarray(phase_series)
        if phase_arr.shape != args_of_perigee.shape:
            raise ValueError("phase_series must have same shape as args_of_perigee")
        x_data = phase_arr
        x_label = 'True Anomaly (TLE Kepler proxy, degrees)'
        title = 'Density Plot: RAAN vs. True Anomaly (TLE Kepler proxy)'
        plotted_phase_variable = 'true_anomaly_deg'

    if shell_series is not None:
        shell_arr = np.asarray(shell_series)
        if shell_arr.shape != args_of_perigee.shape:
            raise ValueError("shell_series must have same shape as args_of_perigee")
    else:
        shell_arr = None

    if args_of_perigee.size == 0:
        print("[density_ra_versus_arg] No data to plot.")
        if return_results:
            return {
                'metadata': {
                    'mode': mode,
                    'plotted_phase_variable': plotted_phase_variable,
                    'wrap_aware': True,
                    'points_total': 0,
                },
                'figures': {},
            }
        return None

    def wrap_axis(values, wrap_mode):
        if str(wrap_mode) == '180':
            return wrap_degrees_180(values)
        return wrap_degrees_360(values)

    x_data = wrap_axis(x_data, x_wrap_mode)
    y_data = wrap_axis(right_ascensions, y_wrap_mode)

    display_names = list(fileNames)
    if 'All Files' not in display_names:
        display_names.append('All Files')

    # Memory-safe index grouping.
    order = np.argsort(filenames_array, kind='mergesort')
    sorted_names = filenames_array[order]
    unique_names, start_idx = np.unique(sorted_names, return_index=True)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = order.size
    bounds = {
        name: (int(s), int(e))
        for name, s, e in zip(unique_names.tolist(), start_idx.tolist(), end_idx.tolist())
    }
    all_indices = np.arange(filenames_array.size, dtype=np.int64)

    def get_indices(selected_name):
        if selected_name == 'All Files':
            return all_indices
        window = bounds.get(selected_name)
        if window is None:
            return np.empty(0, dtype=np.int64)
        s, e = window
        return order[s:e]

    # Prepare fixed histogram edges for all slider states.
    number_of_bins_x = 60
    number_of_bins_y = 60
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)
    if str(x_wrap_mode) == '180':
        x_min, x_max = -180.0, 180.0
    else:
        x_min, x_max = 0.0, 360.0
    if str(y_wrap_mode) == '180':
        y_min, y_max = -180.0, 180.0
    else:
        y_min, y_max = 0.0, 360.0
    xedges = np.linspace(x_min, x_max, number_of_bins_x + 1)
    yedges = np.linspace(y_min, y_max, number_of_bins_y + 1)

    # Lazy histogram cache: compute only when a file is selected.
    histogram_cache = {}

    def get_histogram(selected_name):
        if selected_name in histogram_cache:
            return histogram_cache[selected_name]

        idx = get_indices(selected_name)
        if idx.size > 0:
            hist, _, _ = np.histogram2d(
                x_data[idx],
                y_data[idx],
                bins=[xedges, yedges]
            )
        else:
            hist = np.zeros((number_of_bins_x, number_of_bins_y), dtype=np.float64)

        padded = circular_pad_histogram_2d(hist, pad_x=2, pad_y=2, circular_x=True, circular_y=True)
        hist_smoothed = gaussian_filter(padded, sigma=float(smoothing_sigma))[2:-2, 2:-2]
        histogram_cache[selected_name] = {
            'hist_raw': hist,
            'hist_smoothed': hist_smoothed,
        }
        return hist

    initial_idx = display_names.index('All Files') if 'All Files' in display_names else 0
    init_filename = display_names[initial_idx]
    print(f"[density_ra_versus_arg] Preparing initial frame for {init_filename}...")
    get_histogram(init_filename)
    hist = histogram_cache[init_filename]['hist_smoothed']

    # -- Create figure and axes for the density plot --
    fig_density, ax_density = plt.subplots(figsize=(10, 8))
    fig_density.canvas.manager.set_window_title('Density RA vs Arg of Perigee')

    density_plot = None
    contour_overlay = None

    if mode == 'hexbin':
        idx0 = get_indices(init_filename)
        density_plot = ax_density.hexbin(
            x_data[idx0],
            y_data[idx0],
            gridsize=65,
            mincnt=1,
            cmap='viridis',
            rasterized=True,
        )
    elif mode == 'torus_kde':
        idx0 = get_indices(init_filename)
        kde = torus_kde_von_mises(
            x_data[idx0],
            y_data[idx0],
            bins_x=number_of_bins_x,
            bins_y=number_of_bins_y,
            kappa_x=25.0,
            kappa_y=25.0,
        )
        hist = kde['density']
        density_plot = ax_density.imshow(hist.T, origin='lower', cmap='viridis', aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='bilinear', rasterized=True)
    elif mode in {'circular_linear_kde', 'circular_kde', 'circular_kde_placeholder'}:
        idx0 = get_indices(init_filename)
        payload0 = circular_linear_density(
            x_data[idx0],
            y_data[idx0],
            circular_bins=number_of_bins_x,
            linear_bins=number_of_bins_y,
            sigma=float(smoothing_sigma),
        )
        hist = payload0['density']
        density_plot = ax_density.imshow(hist.T, origin='lower', cmap='viridis', aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='bilinear', rasterized=True)
    else:
        density_plot = ax_density.imshow(hist.T, origin='lower', cmap='viridis', aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='bilinear', rasterized=True)

    # Add colorbar, labels, etc.
    cbar = fig_density.colorbar(density_plot, ax=ax_density)
    cbar_label = 'Counts per bin (wrap-aware smoothed)'
    if mode == 'hexbin':
        cbar_label = 'Counts per hexbin'
    if mode == 'torus_kde':
        cbar_label = 'Torus KDE density (von-Mises x von-Mises)'
    if mode in {'circular_linear_kde', 'circular_kde', 'circular_kde_placeholder'}:
        cbar_label = 'Circular-linear KDE density'
    cbar.set_label(cbar_label)
    ax_density.set_title(title)
    ax_density.set_xlabel(x_label)
    ax_density.set_ylabel('Right Ascension (degrees)')

    xticks, xlabels = angular_axis_ticks('180' if str(x_wrap_mode) == '180' else '360')
    yticks, ylabels = angular_axis_ticks('180' if str(y_wrap_mode) == '180' else '360')
    ax_density.set_xticks(xticks)
    ax_density.set_xticklabels(xlabels)
    ax_density.set_yticks(yticks)
    ax_density.set_yticklabels(ylabels)

    if add_shell_overlay and shell_arr is not None:
        cat = stable_category_color_map(shell_arr)
        idx0 = get_indices(init_filename)
        if idx0.size > 0:
            contour_overlay = ax_density.scatter(
                x_data[idx0],
                y_data[idx0],
                c=cat['codes'][idx0].astype(float),
                cmap=cat['cmap'],
                s=2,
                linewidths=0,
                alpha=0.35,
                rasterized=True,
            )

    # -- Add a slider to filter by file index --
    # Adjust these [left, bottom, width, height] to fit your layout
    ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow',
                         figure=fig_density)
    slider_file = Slider(ax=ax_slider, label='File Index',
                        valmin=0, valmax=len(display_names) - 1,
                        valinit=initial_idx, valstep=1) 
    
    # -- Update function: re-compute histogram for selected file --
    def update(val):
        idx = int(slider_file.val)
        selected_filename = display_names[idx]
        t_update = perf_counter()

        get_histogram(selected_filename)
        hist_new = histogram_cache[selected_filename]['hist_smoothed']
        idx_new = get_indices(selected_filename)

        if mode == 'hexbin':
            nonlocal_artists['density'].remove()
            nonlocal_artists['density'] = ax_density.hexbin(
                x_data[idx_new],
                y_data[idx_new],
                gridsize=65,
                mincnt=1,
                cmap='viridis',
                rasterized=True,
            )
            cbar.update_normal(nonlocal_artists['density'])
        elif mode == 'torus_kde':
            kde = torus_kde_von_mises(
                x_data[idx_new],
                y_data[idx_new],
                bins_x=number_of_bins_x,
                bins_y=number_of_bins_y,
                kappa_x=25.0,
                kappa_y=25.0,
            )
            nonlocal_artists['density'].set_data(kde['density'].T)
            nonlocal_artists['density'].set_clim(vmin=float(np.min(kde['density'])),
                                                 vmax=float(np.max(kde['density'])))
            cbar.update_normal(nonlocal_artists['density'])
        elif mode in {'circular_linear_kde', 'circular_kde', 'circular_kde_placeholder'}:
            payload = circular_linear_density(
                x_data[idx_new],
                y_data[idx_new],
                circular_bins=number_of_bins_x,
                linear_bins=number_of_bins_y,
                sigma=float(smoothing_sigma),
            )
            nonlocal_artists['density'].set_data(payload['density'].T)
            nonlocal_artists['density'].set_clim(vmin=float(np.min(payload['density'])),
                                                 vmax=float(np.max(payload['density'])))
            cbar.update_normal(nonlocal_artists['density'])
        else:
            nonlocal_artists['density'].set_data(hist_new.T)
            nonlocal_artists['density'].set_clim(vmin=float(hist_new.min()), vmax=float(hist_new.max()))
            cbar.update_normal(nonlocal_artists['density'])

        if add_shell_overlay and shell_arr is not None:
            cat = stable_category_color_map(shell_arr)
            if nonlocal_artists['overlay'] is not None:
                nonlocal_artists['overlay'].remove()
            nonlocal_artists['overlay'] = ax_density.scatter(
                x_data[idx_new],
                y_data[idx_new],
                c=cat['codes'][idx_new].astype(float),
                cmap=cat['cmap'],
                s=2,
                linewidths=0,
                alpha=0.35,
                rasterized=True,
            )

        fig_density.canvas.draw_idle()
        print(f"[density_ra_versus_arg] Updated {selected_filename} in {perf_counter() - t_update:.2f}s")

    nonlocal_artists = {'density': density_plot, 'overlay': contour_overlay}

    slider_file.on_changed(update)

    print(f"[density_ra_versus_arg] Ready in {perf_counter() - t0:.2f}s")

    if show_plots:
        plt.show()
        if not return_figures and not _preserve_open_figures_for_export():
            plt.close(fig_density)

    payload = {
        'metadata': {
            'mode': mode,
            'smoothing_sigma': float(smoothing_sigma),
            'plotted_phase_variable': plotted_phase_variable,
            'x_wrap_mode': str(x_wrap_mode),
            'y_wrap_mode': str(y_wrap_mode),
            'wrap_aware': True,
            'bins': [number_of_bins_x, number_of_bins_y],
            'cache_entries': int(len(histogram_cache)),
        },
        'figures': {
            'density': fig_density if return_figures else None,
        },
    }

    if return_results:
        return payload
    return None


# Consolidated from ra_versus_arg.py
def ra_versus_arg(eccentricities, args_of_perigee, right_ascensions, fileNames, filenames_array,
                  phase_mode=None, phase_series=None, low_ecc_mask=None,
                  shell_series=None, show_plots=True, return_figures=False,
                  return_results=False, x_wrap_mode='360', y_wrap_mode='360',
                  torus_duplicate_margin_deg=0.0, density_mode='scatter',
                  stable_shell_colors=True, highlight_low_e=True):
    """
    Plot the Right Ascension of the Ascending Node versus the Argument of Perigee for a set of satellites.

    Parameters:
        eccentricities (np.array): The eccentricities of the satellites.
        args_of_perigee (np.array): The argument of perigee of the satellites in degrees.
        right_ascensions (np.array): The right ascension of the ascending node of the satellites in degrees.
        fileNames (list): The names of the files containing the TLE data.
        filenames_array (np.array): The file index for each satellite.

    Returns:
        None

    Notes:
        New optional parameters (`phase_mode`, `phase_series`, `low_ecc_mask`) enable
        low-e-safe phase plotting without changing existing call sites.
    """
    t0 = perf_counter()
    # Store GUI data in a dictionary
    gui_data = {'eccentricities': np.asarray(eccentricities),
                'args_of_perigee': np.asarray(args_of_perigee),
                'right_ascensions': np.asarray(right_ascensions),
                'filenames_array': np.asarray(filenames_array)}

    if phase_series is not None:
        phase_arr = np.asarray(phase_series)
        if phase_arr.shape != gui_data['args_of_perigee'].shape:
            raise ValueError("phase_series must have the same shape as args_of_perigee")
        gui_data['phase_series'] = phase_arr

    if low_ecc_mask is not None:
        low_e_arr = np.asarray(low_ecc_mask).astype(bool)
        if low_e_arr.shape != gui_data['args_of_perigee'].shape:
            raise ValueError("low_ecc_mask must have the same shape as args_of_perigee")
        gui_data['low_ecc_mask'] = low_e_arr

    if shell_series is not None:
        shell_arr = np.asarray(shell_series)
        if shell_arr.shape != gui_data['args_of_perigee'].shape:
            raise ValueError("shell_series must have the same shape as args_of_perigee")
        gui_data['shell_series'] = shell_arr

    x_key = 'args_of_perigee'
    x_label = 'Argument of Perigee (degrees)'
    title_main = 'Right Ascension of Ascending Node vs. Argument of Perigee'
    title_color = 'RAAN vs Argument of Perigee (In Relation to Eccentricity)'
    if phase_mode is not None and 'phase_series' in gui_data:
        x_key = 'phase_series'
        x_label = 'True Anomaly (TLE Kepler proxy, degrees)'
        title_main = 'RAAN vs True Anomaly (TLE Kepler proxy)'
        title_color = 'RAAN vs True Anomaly (TLE Kepler proxy) (In Relation to Eccentricity)'

    plotted_phase_variable = 'argument_of_perigee_deg'
    if x_key == 'phase_series':
        plotted_phase_variable = 'true_anomaly_deg'

    if x_key == 'args_of_perigee' and 'low_ecc_mask' in gui_data and np.any(gui_data['low_ecc_mask']):
        warnings.warn(
            "Using argument of perigee for low-e points can be ill-conditioned; "
            "consider supplying phase_series for circular-safe phase visualization.",
            UserWarning,
            stacklevel=2,
        )

    if gui_data['args_of_perigee'].size == 0:
        print("[ra_versus_arg] No data to plot.")
        if return_results:
            return {
                'metadata': {
                    'plotted_phase_variable': plotted_phase_variable,
                    'density_mode': density_mode,
                    'x_wrap_mode': x_wrap_mode,
                    'y_wrap_mode': y_wrap_mode,
                    'points_total': 0,
                },
                'figures': {},
            }
        return None

    display_names = list(fileNames)
    if 'All Files' not in display_names:
        display_names.append('All Files')

    order = np.argsort(gui_data['filenames_array'], kind='mergesort')
    sorted_names = gui_data['filenames_array'][order]
    unique_names, start_idx = np.unique(sorted_names, return_index=True)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = order.size
    bounds = {name: (int(s), int(e))
              for name, s, e in zip(unique_names.tolist(), start_idx.tolist(), end_idx.tolist())}
    all_indices = np.arange(gui_data['filenames_array'].size, dtype=np.int64)

    def get_indices(selected_name):
        if selected_name == 'All Files':
            return all_indices
        window = bounds.get(selected_name)
        if window is None:
            return np.empty(0, dtype=np.int64)
        s, e = window
        return order[s:e]

    def safe_limits(x, y):
        if x.size == 0 or y.size == 0:
            return None
        return (float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y)))

    def wrap_axis(values, wrap_mode):
        if str(wrap_mode) == '180':
            return wrap_degrees_180(values)
        return wrap_degrees_360(values)

    wrapped_x_all = wrap_axis(gui_data[x_key], x_wrap_mode)
    wrapped_y_all = wrap_axis(gui_data['right_ascensions'], y_wrap_mode)

    shell_color_info = None
    if stable_shell_colors and 'shell_series' in gui_data:
        shell_color_info = stable_category_color_map(gui_data['shell_series'])

    initial_idx = display_names.index('All Files') if 'All Files' in display_names else 0
    init_name = display_names[initial_idx]
    init_indices = get_indices(init_name)
    x0 = wrapped_x_all[init_indices]
    y0 = wrapped_y_all[init_indices]
    e0 = gui_data['eccentricities'][init_indices]

    def get_display_subset(indices):
        x_base = wrapped_x_all[indices]
        y_base = wrapped_y_all[indices]
        local_source = np.arange(indices.size, dtype=np.int64)
        if float(torus_duplicate_margin_deg) > 0.0:
            dup = duplicate_torus_points_for_display(
                x_base,
                y_base,
                margin_deg=float(torus_duplicate_margin_deg),
                wrap_mode=x_wrap_mode,
            )
            x_plot = dup['x']
            y_plot = dup['y']
            local_source = dup['source_index']
        else:
            x_plot = x_base
            y_plot = y_base

        ecc_plot = gui_data['eccentricities'][indices][local_source] if indices.size > 0 else np.empty(0)
        low_plot = None
        if 'low_ecc_mask' in gui_data and indices.size > 0:
            low_plot = gui_data['low_ecc_mask'][indices][local_source]
        shell_codes_plot = None
        if shell_color_info is not None and indices.size > 0:
            shell_codes_plot = shell_color_info['codes'][indices][local_source]

        return x_plot, y_plot, ecc_plot, low_plot, shell_codes_plot

    print(f"[ra_versus_arg] Initializing with {x0.size:,} points from {init_name}...")

    # Plot 1: Right Ascension vs Argument of Perigee
    fig_ra_vs_arg, ax_ra_vs_arg = plt.subplots()
    x_plot0, y_plot0, e_plot0, low_plot0, shell_plot0 = get_display_subset(init_indices)
    scatter_ra_vs_arg = ax_ra_vs_arg.scatter(x_plot0, y_plot0, s=4, linewidths=0, rasterized=True)
    density_artist = None
    ax_ra_vs_arg.set_title(title_main)
    ax_ra_vs_arg.set_xlabel(x_label)
    ax_ra_vs_arg.set_ylabel('Right Ascension of Ascending Node (degrees)')

    xticks, xlabels = angular_axis_ticks('180' if str(x_wrap_mode) == '180' else '360')
    yticks, ylabels = angular_axis_ticks('180' if str(y_wrap_mode) == '180' else '360')
    ax_ra_vs_arg.set_xticks(xticks)
    ax_ra_vs_arg.set_xticklabels(xlabels)
    ax_ra_vs_arg.set_yticks(yticks)
    ax_ra_vs_arg.set_yticklabels(ylabels)

    if density_mode == 'hexbin':
        scatter_ra_vs_arg.set_offsets(np.empty((0, 2), dtype=np.float64))
        density_artist = ax_ra_vs_arg.hexbin(
            x_plot0,
            y_plot0,
            gridsize=60,
            mincnt=1,
            cmap='viridis',
            rasterized=True,
        )
    elif density_mode == 'hist2d':
        scatter_ra_vs_arg.set_offsets(np.empty((0, 2), dtype=np.float64))
        h = ax_ra_vs_arg.hist2d(
            x_plot0,
            y_plot0,
            bins=60,
            cmap='viridis',
            rasterized=True,
        )
        density_artist = h[-1]

    # Slider for the first plot
    ax_slider_ra_vs_arg = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig_ra_vs_arg)
    slider_ra_vs_arg = Slider(ax_slider_ra_vs_arg, 'File Index', 0, len(display_names) - 1, valinit=initial_idx, valstep=1)

    lims0 = safe_limits(x_plot0, y_plot0)
    if lims0 is not None:
        ax_ra_vs_arg.set_xlim(lims0[0], lims0[1])
        ax_ra_vs_arg.set_ylim(lims0[2], lims0[3])

    def update_ra_vs_arg(val):
        idx = int(slider_ra_vs_arg.val)
        selected_filename = display_names[idx]
        t_update = perf_counter()
        indices = get_indices(selected_filename)

        x_plot, y_plot, _, low_plot, _ = get_display_subset(indices)

        if density_mode == 'scatter':
            scatter_ra_vs_arg.set_offsets(np.column_stack((x_plot, y_plot)))
        elif density_mode == 'hexbin':
            if nonlocal_density['artist'] is not None:
                nonlocal_density['artist'].remove()
            new_artist = ax_ra_vs_arg.hexbin(
                x_plot,
                y_plot,
                gridsize=60,
                mincnt=1,
                cmap='viridis',
                rasterized=True,
            )
            nonlocal_density['artist'] = new_artist
        elif density_mode == 'hist2d':
            if nonlocal_density['artist'] is not None:
                nonlocal_density['artist'].remove()
            h = ax_ra_vs_arg.hist2d(
                x_plot,
                y_plot,
                bins=60,
                cmap='viridis',
                rasterized=True,
            )
            nonlocal_density['artist'] = h[-1]

        if highlight_low_e and low_plot is not None and low_plot.size > 0:
            low_idx = low_plot.astype(bool)
            if np.any(low_idx):
                ax_ra_vs_arg.scatter(
                    x_plot[low_idx],
                    y_plot[low_idx],
                    s=10,
                    facecolors='none',
                    edgecolors='k',
                    linewidths=0.25,
                    rasterized=True,
                )

        lims = safe_limits(x_plot, y_plot)
        if lims is not None:
            ax_ra_vs_arg.set_xlim(lims[0], lims[1])
            ax_ra_vs_arg.set_ylim(lims[2], lims[3])
        fig_ra_vs_arg.canvas.draw_idle()
        print(f"[ra_versus_arg] Scatter updated for {selected_filename} in {perf_counter() - t_update:.2f}s")

    nonlocal_density = {'artist': density_artist}

    slider_ra_vs_arg.on_changed(update_ra_vs_arg)

    # Plot 2: Right Ascension vs Argument of Perigee with Eccentricity as color
    fig_ra_vs_arg_ecc, ax_ra_vs_arg_ecc = plt.subplots()
    c0 = e_plot0
    cmap0 = 'viridis'
    if shell_plot0 is not None:
        c0 = shell_plot0.astype(float)
        cmap0 = shell_color_info['cmap']
    elif phase_mode == 'shell_colored_snapshot' and low_plot0 is not None:
        c0 = low_plot0.astype(float)
        cmap0 = 'coolwarm'

    scatter_ra_vs_arg_ecc = ax_ra_vs_arg_ecc.scatter(x_plot0, y_plot0, c=c0, cmap=cmap0, s=4, linewidths=0,
                                                     rasterized=True)
    ax_ra_vs_arg_ecc.set_title(title_color)
    ax_ra_vs_arg_ecc.set_xlabel(x_label)
    ax_ra_vs_arg_ecc.set_ylabel('Right Ascension of Ascending Node (degrees)')
    cbar_label = 'Eccentricity'
    if shell_plot0 is not None:
        cbar_label = 'Shell ID (stable categorical code)'
    elif phase_mode == 'shell_colored_snapshot' and low_plot0 is not None:
        cbar_label = 'Low-e mask (0/1)'
    plt.colorbar(scatter_ra_vs_arg_ecc, ax=ax_ra_vs_arg_ecc, label=cbar_label)

    # Slider for the second plot
    ax_slider_ra_vs_arg_ecc = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow', figure=fig_ra_vs_arg_ecc)
    slider_ra_vs_arg_ecc = Slider(ax_slider_ra_vs_arg_ecc, 'File Index', 0, len(display_names) - 1, valinit=initial_idx, valstep=1)

    lims1 = safe_limits(x_plot0, y_plot0)
    if lims1 is not None:
        ax_ra_vs_arg_ecc.set_xlim(lims1[0], lims1[1])
        ax_ra_vs_arg_ecc.set_ylim(lims1[2], lims1[3])

    def update_ra_vs_arg_ecc(val):
        idx = int(slider_ra_vs_arg_ecc.val)
        selected_filename = display_names[idx]
        t_update = perf_counter()
        indices = get_indices(selected_filename)

        # Update the scatter plot data
        x_plot, y_plot, e_plot, low_plot, shell_plot = get_display_subset(indices)
        scatter_ra_vs_arg_ecc.set_offsets(np.column_stack((x_plot, y_plot)))
        if shell_plot is not None:
            scatter_ra_vs_arg_ecc.set_array(shell_plot.astype(float))
        elif phase_mode == 'shell_colored_snapshot' and low_plot is not None:
            scatter_ra_vs_arg_ecc.set_array(low_plot.astype(float))
        else:
            scatter_ra_vs_arg_ecc.set_array(e_plot)
        lims = safe_limits(x_plot, y_plot)
        if lims is not None:
            ax_ra_vs_arg_ecc.set_xlim(lims[0], lims[1])
            ax_ra_vs_arg_ecc.set_ylim(lims[2], lims[3])
        fig_ra_vs_arg_ecc.canvas.draw_idle()
        print(f"[ra_versus_arg] Ecc-color plot updated for {selected_filename} in {perf_counter() - t_update:.2f}s")

    slider_ra_vs_arg_ecc.on_changed(update_ra_vs_arg_ecc)
    print(f"[ra_versus_arg] Ready in {perf_counter() - t0:.2f}s")

    if show_plots:
        plt.show()
        if not return_figures and not _preserve_open_figures_for_export():
            plt.close(fig_ra_vs_arg)
            plt.close(fig_ra_vs_arg_ecc)

    payload = {
        'metadata': {
            'plotted_phase_variable': plotted_phase_variable,
            'phase_semantics': 'TLE-derived Kepler proxy from mean anomaly',
            'density_mode': density_mode,
            'x_wrap_mode': str(x_wrap_mode),
            'y_wrap_mode': str(y_wrap_mode),
            'torus_duplicate_margin_deg': float(torus_duplicate_margin_deg),
            'points_total': int(gui_data['args_of_perigee'].size),
        },
        'figures': {
            'raan_vs_phase': fig_ra_vs_arg if return_figures else None,
            'raan_vs_phase_colored': fig_ra_vs_arg_ecc if return_figures else None,
        },
    }

    if return_results:
        return payload
    return None