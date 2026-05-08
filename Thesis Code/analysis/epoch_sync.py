import numpy as np
import pandas as pd

import os

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    njit = None
    _HAS_NUMBA = False

from constants import (DEFAULT_SYNC_TOLERANCE, SYNC_MODE_EXACT_EPOCH_INTERSECTION,
                       SYNC_MODE_NEAREST_INTERSECTION, SYNC_MODE_SCALAR_INTERPOLATION,
                       SYNC_MODE_SGP4_COMMON_EPOCH, SYNC_MODE_TARGET_NEAREST, SYNC_MODES)


DEFAULT_TOLERANCE = DEFAULT_SYNC_TOLERANCE
_NUMBA_DISABLED_ENV = {"0", "false", "no", "off"}
_USE_NUMBA = _HAS_NUMBA and str(os.getenv("EPOCH_SYNC_USE_NUMBA", "1")).strip().lower() not in _NUMBA_DISABLED_ENV


if _HAS_NUMBA:

    @njit(cache=True)
    def _interpolate_numeric_at_target_numba(time_ns, values, target_ns):
        n = time_ns.size
        finite_count = 0
        for i in range(n):
            if np.isfinite(values[i]):
                finite_count += 1

        if finite_count == 0:
            return np.nan

        t = np.empty(finite_count, dtype=np.int64)
        v = np.empty(finite_count, dtype=np.float64)
        pos = 0
        for i in range(n):
            if np.isfinite(values[i]):
                t[pos] = time_ns[i]
                v[pos] = values[i]
                pos += 1

        if finite_count == 1:
            return float(v[0])

        order = np.argsort(t)
        t_sorted = np.empty(finite_count, dtype=np.int64)
        v_sorted = np.empty(finite_count, dtype=np.float64)
        for i in range(finite_count):
            idx = order[i]
            t_sorted[i] = t[idx]
            v_sorted[i] = v[idx]

        t_unique = np.empty(finite_count, dtype=np.int64)
        v_unique = np.empty(finite_count, dtype=np.float64)
        uniq_n = 0
        current_t = t_sorted[0]
        sum_v = v_sorted[0]
        count_v = 1

        for i in range(1, finite_count):
            if t_sorted[i] == current_t:
                sum_v += v_sorted[i]
                count_v += 1
            else:
                t_unique[uniq_n] = current_t
                v_unique[uniq_n] = sum_v / count_v
                uniq_n += 1
                current_t = t_sorted[i]
                sum_v = v_sorted[i]
                count_v = 1

        t_unique[uniq_n] = current_t
        v_unique[uniq_n] = sum_v / count_v
        uniq_n += 1

        if uniq_n == 1:
            return float(v_unique[0])

        if target_ns <= t_unique[0]:
            return float(v_unique[0])
        if target_ns >= t_unique[uniq_n - 1]:
            return float(v_unique[uniq_n - 1])

        lo = 0
        hi = uniq_n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if t_unique[mid] <= target_ns:
                lo = mid
            else:
                hi = mid

        t0 = float(t_unique[lo])
        t1 = float(t_unique[lo + 1])
        y0 = float(v_unique[lo])
        y1 = float(v_unique[lo + 1])
        if t1 == t0:
            return y0

        w = (float(target_ns) - t0) / (t1 - t0)
        return y0 + (y1 - y0) * w


def _to_timestamp(value):
    return pd.Timestamp(value)


def _to_timedelta(value):
    if isinstance(value, pd.Timedelta):
        return value
    return pd.to_timedelta(value)


def _empty_result(df):
    empty = pd.DataFrame(columns=df.columns)
    return empty, None, None


def _resolve_mode(mode, target_time):
    if mode in (None, "auto"):
        return SYNC_MODE_TARGET_NEAREST if target_time is not None else SYNC_MODE_NEAREST_INTERSECTION
    mode_str = str(mode).strip().lower()
    if mode_str not in SYNC_MODES:
        raise ValueError(f"Unsupported synchronization mode '{mode_str}'. Expected one of {SYNC_MODES}.")
    return mode_str


def _nearest_row_for_time(df, target_time, time_col="timestamp", tolerance=DEFAULT_TOLERANCE):
    if df.empty:
        return None

    t = _to_timestamp(target_time)
    tol = _to_timedelta(tolerance)

    times = pd.to_datetime(df[time_col], errors="coerce")
    valid = times.notna()
    if not valid.any():
        return None

    valid_df = df.loc[valid]
    valid_times = times.loc[valid]

    deltas = (valid_times - t).abs()
    best_idx = deltas.idxmin()
    if deltas.loc[best_idx] > tol:
        return None

    row = valid_df.loc[best_idx]
    return row


def _resolve_interpolation_target(groups, sat_ids, time_col, target_time=None):
    if target_time is not None:
        return _to_timestamp(target_time)

    min_times = []
    max_times = []
    for sat_id in sat_ids:
        times = pd.to_datetime(groups[sat_id][time_col], errors="coerce").dropna()
        if times.empty:
            return None
        min_times.append(times.min())
        max_times.append(times.max())

    overlap_start = max(min_times)
    overlap_end = min(max_times)
    if overlap_start <= overlap_end:
        return overlap_start + (overlap_end - overlap_start) / 2

    base_times = pd.to_datetime(groups[sat_ids[0]][time_col], errors="coerce").dropna().to_list()
    best_time = None
    best_score = None
    for candidate in base_times:
        deltas = []
        feasible = True
        for sat_id in sat_ids:
            times = pd.to_datetime(groups[sat_id][time_col], errors="coerce").dropna()
            if times.empty:
                feasible = False
                break
            delta = (times - candidate).abs().min()
            if pd.isna(delta):
                feasible = False
                break
            deltas.append(float(delta.total_seconds()))

        if not feasible:
            continue

        score = (max(deltas), sum(deltas))
        if best_score is None or score < best_score:
            best_score = score
            best_time = pd.Timestamp(candidate)

    return best_time


def _interpolate_numeric_at_target(time_ns, values, target_ns):
    t = np.asarray(time_ns, dtype=np.int64)
    v = np.asarray(values, dtype=np.float64)

    if _USE_NUMBA and t.ndim == 1 and v.ndim == 1 and t.shape == v.shape:
        return float(_interpolate_numeric_at_target_numba(t, v, np.int64(target_ns)))

    finite = np.isfinite(v)
    if not np.any(finite):
        return np.nan

    t = t[finite]
    v = v[finite]
    if t.size == 0:
        return np.nan
    if t.size == 1:
        return float(v[0])

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[order]

    if np.any(np.diff(t) == 0):
        uniq, inv = np.unique(t, return_inverse=True)
        sums = np.zeros(uniq.shape, dtype=np.float64)
        counts = np.zeros(uniq.shape, dtype=np.float64)
        np.add.at(sums, inv, v)
        np.add.at(counts, inv, 1.0)
        t = uniq
        v = sums / np.maximum(counts, 1.0)

    return float(np.interp(float(target_ns), t.astype(np.float64), v, left=v[0], right=v[-1]))


def _interpolate_row_for_time(grp, sat_id, target_time, time_col, object_col):
    times = pd.to_datetime(grp[time_col], errors="coerce")
    valid = times.notna()
    if not valid.any():
        return None, np.nan

    grp_valid = grp.loc[valid].copy()
    times_valid = pd.to_datetime(grp_valid[time_col], errors="coerce")
    t_ns = times_valid.astype("int64", copy=False).to_numpy(dtype=np.int64)
    target_ts = _to_timestamp(target_time)
    target_ns = np.int64(target_ts.value)

    deltas_sec = np.abs(t_ns.astype(np.float64) - float(target_ns)) / 1.0e9
    nearest_idx = int(np.argmin(deltas_sec))
    nearest_delta = float(deltas_sec[nearest_idx])
    nearest_row = grp_valid.iloc[nearest_idx]

    out = nearest_row.to_dict()
    out[object_col] = sat_id
    out[time_col] = target_ts

    for col in grp_valid.columns:
        if col in {time_col, object_col}:
            continue
        dtype = grp_valid[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            continue
        if not pd.api.types.is_numeric_dtype(dtype):
            continue
        vals = pd.to_numeric(grp_valid[col], errors="coerce").to_numpy(dtype=np.float64)
        interp_val = _interpolate_numeric_at_target(t_ns, vals, target_ns)
        if np.isfinite(interp_val):
            out[col] = float(interp_val)

    return out, nearest_delta


def find_common_epoch_records(
    df,
    sat_ids,
    target_time=None,
    tolerance=DEFAULT_TOLERANCE,
    object_col="sat_id",
    time_col="timestamp",
    mode="auto",
    return_metadata=False,
):
    """Return synchronized records for multiple objects under an explicit mode.

    Modes:
        - target_nearest: nearest row to target_time for each object.
        - nearest_intersection: minimize max absolute epoch delta using one object's
          epochs as candidates (legacy behavior).
        - exact_epoch_intersection: require one exact shared timestamp across all
          selected objects (strict intersection).
                - scalar_interpolation: interpolate numeric scalar columns at a common epoch.
                - sgp4_common_epoch: practical common-epoch proxy using scalar interpolation
                    when full SGP4 state propagation inputs are unavailable.

    Scientific semantics:
        This utility synchronizes rows in catalog-derived panels; it does not
        propagate states and should be treated as a descriptive/proxy alignment
        stage unless a dynamic propagation mode is used.

    Returns a tuple: (records_df, common_epoch_timestamp, max_abs_delta_seconds)
    where records_df is empty when no synchronized solution is found.
    """
    resolved_mode = _resolve_mode(mode, target_time=target_time)
    sat_ids = [str(s) for s in sat_ids]
    if len(sat_ids) == 0:
        records, common_epoch, max_delta = _empty_result(df)
        if return_metadata:
            return records, common_epoch, max_delta, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "empty_sat_ids"}
        return records, common_epoch, max_delta

    work = df[df[object_col].astype(str).isin(sat_ids)].copy()
    if work.empty:
        records, common_epoch, max_delta = _empty_result(df)
        if return_metadata:
            return records, common_epoch, max_delta, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "no_matching_objects"}
        return records, common_epoch, max_delta

    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    if work.empty:
        records, common_epoch, max_delta = _empty_result(df)
        if return_metadata:
            return records, common_epoch, max_delta, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "no_valid_timestamps"}
        return records, common_epoch, max_delta

    tol = _to_timedelta(tolerance)

    groups = {
        str(sat_id): grp.sort_values(time_col, kind="mergesort")
        for sat_id, grp in work.groupby(work[object_col].astype(str), sort=False)
    }
    if any(s not in groups for s in sat_ids):
        records, common_epoch, max_delta = _empty_result(df)
        if return_metadata:
            return records, common_epoch, max_delta, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "missing_object_group"}
        return records, common_epoch, max_delta

    if resolved_mode == SYNC_MODE_TARGET_NEAREST:
        if target_time is None:
            raise ValueError("target_nearest mode requires target_time.")
        target_ts = _to_timestamp(target_time)
        chosen = []
        deltas = []
        for sat_id in sat_ids:
            row = _nearest_row_for_time(groups[sat_id], target_ts, time_col=time_col, tolerance=tol)
            if row is None:
                records, common_epoch, max_delta = _empty_result(df)
                if return_metadata:
                    return records, common_epoch, max_delta, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "tolerance_reject"}
                return records, common_epoch, max_delta
            chosen.append(row)
            deltas.append(abs(pd.Timestamp(row[time_col]) - target_ts).total_seconds())

        records = pd.DataFrame(chosen).reset_index(drop=True)
        common_epoch = target_ts
        max_delta = float(max(deltas)) if deltas else 0.0
        if return_metadata:
            return records, common_epoch, max_delta, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "ok"}
        return records, common_epoch, max_delta

    if resolved_mode == SYNC_MODE_EXACT_EPOCH_INTERSECTION:
        intersection = None
        for sat_id in sat_ids:
            time_set = set(pd.to_datetime(groups[sat_id][time_col], errors="coerce").dropna().tolist())
            intersection = time_set if intersection is None else (intersection & time_set)
            if not intersection:
                records, common_epoch, max_delta = _empty_result(df)
                if return_metadata:
                    return records, common_epoch, max_delta, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "no_exact_intersection"}
                return records, common_epoch, max_delta

        common_times = sorted(pd.Timestamp(t) for t in intersection)
        if target_time is not None:
            target_ts = _to_timestamp(target_time)
            if target_ts not in intersection:
                records, common_epoch, max_delta = _empty_result(df)
                if return_metadata:
                    return records, common_epoch, max_delta, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "target_not_in_intersection"}
                return records, common_epoch, max_delta
            chosen_time = target_ts
        else:
            chosen_time = common_times[0]

        chosen = []
        for sat_id in sat_ids:
            grp = groups[sat_id]
            exact = grp[pd.to_datetime(grp[time_col], errors="coerce") == chosen_time]
            if exact.empty:
                records, common_epoch, max_delta = _empty_result(df)
                if return_metadata:
                    return records, common_epoch, max_delta, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "exact_row_missing"}
                return records, common_epoch, max_delta
            chosen.append(exact.iloc[-1])

        records = pd.DataFrame(chosen).reset_index(drop=True)
        if return_metadata:
            return records, chosen_time, 0.0, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "ok"}
        return records, chosen_time, 0.0

    if resolved_mode in {SYNC_MODE_SCALAR_INTERPOLATION, SYNC_MODE_SGP4_COMMON_EPOCH}:
        target_ts = _resolve_interpolation_target(groups, sat_ids, time_col=time_col, target_time=target_time)
        if target_ts is None:
            records, common_epoch, max_delta = _empty_result(df)
            if return_metadata:
                return records, common_epoch, max_delta, {
                    "mode": resolved_mode,
                    "tolerance": str(tolerance),
                    "status": "no_interpolation_anchor",
                }
            return records, common_epoch, max_delta

        chosen = []
        deltas = []
        tol_seconds = float(tol.total_seconds())
        for sat_id in sat_ids:
            row, nearest_delta = _interpolate_row_for_time(
                groups[sat_id],
                sat_id=sat_id,
                target_time=target_ts,
                time_col=time_col,
                object_col=object_col,
            )
            if row is None or not np.isfinite(nearest_delta):
                records, common_epoch, max_delta = _empty_result(df)
                if return_metadata:
                    return records, common_epoch, max_delta, {
                        "mode": resolved_mode,
                        "tolerance": str(tolerance),
                        "status": "interpolation_failed",
                    }
                return records, common_epoch, max_delta

            if nearest_delta > tol_seconds:
                records, common_epoch, max_delta = _empty_result(df)
                if return_metadata:
                    return records, common_epoch, max_delta, {
                        "mode": resolved_mode,
                        "tolerance": str(tolerance),
                        "status": "tolerance_reject",
                        "anchor_time": pd.Timestamp(target_ts),
                    }
                return records, common_epoch, max_delta

            chosen.append(row)
            deltas.append(nearest_delta)

        records = pd.DataFrame(chosen).reset_index(drop=True)
        max_delta = float(max(deltas)) if deltas else 0.0
        if return_metadata:
            return records, pd.Timestamp(target_ts), max_delta, {
                "mode": resolved_mode,
                "tolerance": str(tolerance),
                "status": "ok",
                "interpolation": "linear_scalar",
                "propagation": "sgp4_proxy_scalar_interpolation" if resolved_mode == SYNC_MODE_SGP4_COMMON_EPOCH else "none",
            }
        return records, pd.Timestamp(target_ts), max_delta

    # nearest_intersection style: pick epoch candidates from first satellite
    base_id = sat_ids[0]
    base_times = groups[base_id][time_col].to_list()

    best = None
    for candidate_time in base_times:
        chosen = []
        deltas = []
        ok = True
        for sat_id in sat_ids:
            row = _nearest_row_for_time(groups[sat_id], candidate_time, time_col=time_col, tolerance=tol)
            if row is None:
                ok = False
                break
            delta = abs(pd.Timestamp(row[time_col]) - pd.Timestamp(candidate_time)).total_seconds()
            chosen.append(row)
            deltas.append(delta)

        if not ok:
            continue

        max_delta = float(max(deltas)) if deltas else 0.0
        sum_delta = float(sum(deltas))
        if best is None or (max_delta < best["max_delta"]) or (
            np.isclose(max_delta, best["max_delta"]) and sum_delta < best["sum_delta"]
        ):
            best = {
                "records": pd.DataFrame(chosen).reset_index(drop=True),
                "common_epoch": pd.Timestamp(candidate_time),
                "max_delta": max_delta,
                "sum_delta": sum_delta,
            }

    if best is None:
        records, common_epoch, max_delta = _empty_result(df)
        if return_metadata:
            return records, common_epoch, max_delta, {"mode": resolved_mode, "tolerance": str(tolerance), "status": "no_feasible_solution"}
        return records, common_epoch, max_delta

    if return_metadata:
        return best["records"], best["common_epoch"], best["max_delta"], {"mode": resolved_mode, "tolerance": str(tolerance), "status": "ok"}
    return best["records"], best["common_epoch"], best["max_delta"]
