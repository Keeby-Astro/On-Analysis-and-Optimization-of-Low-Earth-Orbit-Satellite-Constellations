"""Shell occupancy and altitude-regime analytics for constellation archives.

Scope labels:
- Descriptive historical: occupancy, spread, entry/exit counts, transition products.
- Proxy: occupancy normalization and replenishment indicators.

Notes:
- Shell occupancy is distinct from altitude-bin occupancy.
- If candidate_shell_id is missing, fallback products are explicitly labeled as altitude-bin based.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors as mcolors

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    njit = None
    _HAS_NUMBA = False

from operational_metrics_utils import (SCOPE_COMPLIANCE,
                                       SCOPE_CONJUNCTION_INPUT_READINESS,
                                       SCOPE_CONJUNCTION_PLACEHOLDER,
                                       SCOPE_DESCRIPTIVE, SCOPE_PROXY,
                                       add_compliance_horizon_columns,
                                       add_occupancy_normalization,
                                       apply_duplicate_safe_conditioning,
                                       prepare_time_binned_panel,
                                       resolve_shell_identity, safe_ratio)

try:
    from scipy.interpolate import UnivariateSpline
except Exception:  # pragma: no cover - optional dependency fallback
    UnivariateSpline = None

try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
except Exception:  # pragma: no cover - optional dependency fallback
    _gaussian_filter = None


_NUMBA_DISABLED_ENV = {"0", "false", "no", "off"}
_USE_NUMBA = _HAS_NUMBA and str(os.getenv("SHELL_ANALYTICS_USE_NUMBA", "1")).strip().lower() not in _NUMBA_DISABLED_ENV


if _HAS_NUMBA:

    @njit(cache=True)
    def _local_slope_km_day_numba(t_seconds, altitude_km):
        n = altitude_km.size
        slope = np.empty(n, dtype=np.float64)
        for i in range(n):
            slope[i] = np.nan
        if n < 3:
            return slope

        for i in range(1, n - 1):
            dt = t_seconds[i + 1] - t_seconds[i - 1]
            dy = altitude_km[i + 1] - altitude_km[i - 1]
            if np.isfinite(dt) and np.isfinite(dy) and dt != 0.0:
                slope[i] = (dy / dt) * 86400.0

        d0 = t_seconds[1] - t_seconds[0]
        if d0 != 0.0 and np.isfinite(d0) and np.isfinite(altitude_km[1]) and np.isfinite(altitude_km[0]):
            slope[0] = ((altitude_km[1] - altitude_km[0]) / d0) * 86400.0

        d1 = t_seconds[n - 1] - t_seconds[n - 2]
        if d1 != 0.0 and np.isfinite(d1) and np.isfinite(altitude_km[n - 1]) and np.isfinite(altitude_km[n - 2]):
            slope[n - 1] = ((altitude_km[n - 1] - altitude_km[n - 2]) / d1) * 86400.0

        return slope


def _bin_midpoint_km(label: str) -> float:
    if isinstance(label, pd.Interval):
        lo = float(label.left)
        hi = float(label.right)
        if np.isfinite(lo) and np.isfinite(hi):
            return 0.5 * (lo + hi)
        return np.nan
    if not isinstance(label, str):
        return np.nan
    cleaned = label.strip()
    if "," not in cleaned:
        return np.nan
    cleaned = cleaned.strip("[]()")
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) != 2:
        return np.nan
    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except ValueError:
        return np.nan
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan
    return 0.5 * (lo + hi)


def _bin_bounds_km(label):
    if isinstance(label, pd.Interval):
        lo = float(label.left)
        hi = float(label.right)
        if np.isfinite(lo) and np.isfinite(hi):
            return lo, hi
        return np.nan, np.nan

    if not isinstance(label, str):
        return np.nan, np.nan

    cleaned = label.strip()
    if "," not in cleaned:
        return np.nan, np.nan
    cleaned = cleaned.strip("[]()")
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) != 2:
        return np.nan, np.nan
    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except ValueError:
        return np.nan, np.nan
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan, np.nan
    return lo, hi


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    mids = 0.5 * (c[:-1] + c[1:])
    left = c[0] - (mids[0] - c[0])
    right = c[-1] + (c[-1] - mids[-1])
    return np.concatenate(([left], mids, [right]))


def _resolve_heatmap_norm(grid: np.ndarray, mode: str = "linear", clip_percentile=None):
    data = np.asarray(grid, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None, np.nan, np.nan

    mode_key = str(mode or "linear").strip().lower()
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))

    if clip_percentile is not None:
        try:
            p_clip = float(clip_percentile)
        except Exception:
            p_clip = None
        if p_clip is not None and np.isfinite(p_clip):
            p_clip = min(100.0, max(0.0, p_clip))
            vmax_clip = float(np.nanpercentile(finite, p_clip))
            if np.isfinite(vmax_clip) and vmax_clip > vmin:
                vmax = vmax_clip

    if mode_key == "log":
        positive = finite[finite > 0.0]
        if positive.size == 0:
            return None, vmin, vmax
        vmin_pos = float(np.nanmin(positive))
        vmax_pos = float(np.nanmax(positive))
        if np.isfinite(vmax) and vmax > 0.0:
            vmax_pos = min(vmax_pos, float(vmax))
        if vmax_pos <= vmin_pos:
            vmax_pos = vmin_pos * 1.05
        return mcolors.LogNorm(vmin=vmin_pos, vmax=vmax_pos), vmin_pos, vmax_pos

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return mcolors.Normalize(vmin=vmin, vmax=vmax), vmin, vmax


def _transition_direction(delta_km: float, threshold_km: float = 2.5) -> str:
    if not np.isfinite(delta_km):
        return "unknown"
    if delta_km > threshold_km:
        return "upward"
    if delta_km < -threshold_km:
        return "downward"
    return "lateral"


def compute_common_epoch_shell_snapshot(df: pd.DataFrame,
                                        epoch: pd.Timestamp,
                                        shell_col: str = "shell_or_bin") -> pd.DataFrame:
    """Build a shell snapshot at a common epoch using nearest record per object.

    This is descriptive only and intended for shell occupancy snapshots in reports.
    """
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["object_id", "timestamp", shell_col, "altitude_km"])

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])
    if work.empty:
        return pd.DataFrame(columns=["object_id", "timestamp", shell_col, "altitude_km"])

    object_col = "norad_cat_id" if "norad_cat_id" in work.columns else ("sat_id" if "sat_id" in work.columns else None)
    if object_col is None:
        work["object_id"] = np.arange(len(work)).astype(str)
        object_col = "object_id"
    work[object_col] = work[object_col].astype(str)

    epoch = pd.to_datetime(epoch, errors="coerce")
    if pd.isna(epoch):
        return pd.DataFrame(columns=["object_id", "timestamp", shell_col, "altitude_km"])

    work["epoch_delta_abs"] = (work["timestamp"] - epoch).abs().dt.total_seconds()
    nearest = work.sort_values([object_col, "epoch_delta_abs", "timestamp"], kind="mergesort").drop_duplicates(
        subset=[object_col],
        keep="first",
    )
    out = nearest[[object_col, "timestamp", shell_col, "altitude_km"]].copy()
    out = out.rename(columns={object_col: "object_id"}).reset_index(drop=True)
    return out


def compute_shell_analytics(df: pd.DataFrame, altitude_bins: list[float] | None = None,
                            time_freq: str = "7D", object_col: str | None = None,
                            shell_col: str = "candidate_shell_id",
                            return_extras: bool = False,
                            common_epoch: pd.Timestamp | None = None,
                            include_replenishment_proxy: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute shell occupancy/spread metrics and heatmap-ready outputs.

    Returns:
    - shell_time_df: shell/altitude occupancy metrics over time.
    - entry_exit_df: entry/exit counts over time.
    - altitude_time_heatmap: pivot-ready long table.
    """
    if altitude_bins is None:
        altitude_bins = [300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500]

    work, object_col = prepare_time_binned_panel(df, time_freq=time_freq, object_col=object_col, sort_by_time_only=True)
    work = resolve_shell_identity(work, shell_col=shell_col, altitude_bins=altitude_bins, output_col="shell_or_bin")

    grouped = work.groupby(["time_bin", "shell_or_bin"], observed=False)
    shell_time_df = grouped.agg(
        n_records=(object_col, "size"),
        n_objects=(object_col, "nunique"),
        shell_width_km=("altitude_km", lambda x: float(np.nanmax(x) - np.nanmin(x)) if len(x) else np.nan),
        median_altitude_km=("altitude_km", "median"),
        altitude_spread_km=("altitude_km", "std"),
        raan_spread_deg=("raan", "std"),
        inc_spread_deg=("inc", "std"),
    ).reset_index()
    shell_time_df = shell_time_df.sort_values(["time_bin", "shell_or_bin"], kind="mergesort").reset_index(drop=True)

    source_mix = work.groupby(["time_bin", "shell_or_bin"], observed=False)["shell_identity_source"].agg(
        lambda s: "mixed" if pd.Series(s).nunique(dropna=True) > 1 else (pd.Series(s).iloc[0] if len(s) else "unknown")
    ).rename("shell_basis_label").reset_index()
    shell_time_df = shell_time_df.merge(source_mix, on=["time_bin", "shell_or_bin"], how="left")

    if "shell_assignment_basis" in work.columns:
        assign_basis = work.groupby(["time_bin", "shell_or_bin"], observed=False)["shell_assignment_basis"].agg(
            lambda s: "mixed" if pd.Series(s).nunique(dropna=True) > 1 else (pd.Series(s).dropna().iloc[0] if pd.Series(s).dropna().size else "unknown")
        ).rename("shell_assignment_basis").reset_index()
        shell_time_df = shell_time_df.merge(assign_basis, on=["time_bin", "shell_or_bin"], how="left")
    else:
        shell_time_df["shell_assignment_basis"] = "unknown"

    if "shell_profile_semantics" in work.columns:
        profile_sem = work.groupby(["time_bin", "shell_or_bin"], observed=False)["shell_profile_semantics"].agg(
            lambda s: pd.Series(s).dropna().iloc[0] if pd.Series(s).dropna().size else "unknown"
        ).rename("shell_profile_semantics").reset_index()
        shell_time_df = shell_time_df.merge(profile_sem, on=["time_bin", "shell_or_bin"], how="left")
    else:
        shell_time_df["shell_profile_semantics"] = "unknown"

    shell_time_df["scope_label"] = SCOPE_DESCRIPTIVE

    shell_time_df = add_occupancy_normalization(shell_time_df)
    shell_time_df["occupancy_proxy_scope"] = SCOPE_PROXY

    shell_time_df["shell_centroid_altitude_km"] = shell_time_df["median_altitude_km"]
    prev_centroid = shell_time_df.groupby("shell_or_bin", sort=False)["shell_centroid_altitude_km"].shift(1)
    dt_days = shell_time_df.groupby("shell_or_bin", sort=False)["time_bin"].diff().dt.total_seconds() / 86400.0
    shell_time_df["shell_centroid_drift_km_day"] = safe_ratio(
        shell_time_df["shell_centroid_altitude_km"] - prev_centroid,
        dt_days,
    )

    first_seen = shell_time_df.groupby("shell_or_bin", observed=False)["time_bin"].transform("min")
    shell_time_df["shell_age_days"] = (shell_time_df["time_bin"] - first_seen).dt.total_seconds() / 86400.0

    if include_replenishment_proxy:
        prev_objects = shell_time_df.groupby("shell_or_bin", sort=False)["n_objects"].shift(1)
        shell_time_df["shell_replenishment_count"] = (shell_time_df["n_objects"] - prev_objects).clip(lower=0).fillna(0).astype(int)
    else:
        shell_time_df["shell_replenishment_count"] = 0

    occ = work[[object_col, "time_bin", "shell_or_bin", "altitude_km"]].drop_duplicates().sort_values(
        [object_col, "time_bin"],
        kind="mergesort",
    )
    occ["prev_shell"] = occ.groupby(object_col, sort=False)["shell_or_bin"].shift(1)
    occ["prev_altitude_km"] = occ.groupby(object_col, sort=False)["altitude_km"].shift(1)
    changed = occ["prev_shell"].notna() & (occ["shell_or_bin"] != occ["prev_shell"])

    entry_exit_df = occ.loc[changed, ["time_bin"]].copy()
    entry_exit_df["entry_count"] = 1
    entry_exit_df["exit_count"] = 1
    entry_exit_df = entry_exit_df.groupby("time_bin", as_index=False).agg(entry_count=("entry_count", "sum"),
                                                                           exit_count=("exit_count", "sum"))
    entry_exit_df["scope_label"] = SCOPE_DESCRIPTIVE

    transition_events = occ.loc[changed, [object_col, "time_bin", "prev_shell", "shell_or_bin", "prev_altitude_km", "altitude_km"]].copy()
    transition_events = transition_events.rename(columns={"prev_shell": "shell_from", "shell_or_bin": "shell_to"})
    transition_events["delta_altitude_km"] = transition_events["altitude_km"] - transition_events["prev_altitude_km"]
    transition_events["directional_migration"] = transition_events["delta_altitude_km"].map(_transition_direction)
    transition_events["scope_label"] = SCOPE_DESCRIPTIVE

    transition_counts_df = transition_events.groupby(["time_bin", "shell_from", "shell_to", "directional_migration"], observed=False).agg(
        transition_count=(object_col, "size"),
        median_delta_altitude_km=("delta_altitude_km", "median"),
        n_transition_objects=(object_col, "nunique"),
    ).reset_index()
    transition_counts_df["sankey_source"] = transition_counts_df["shell_from"].astype(str)
    transition_counts_df["sankey_target"] = transition_counts_df["shell_to"].astype(str)

    transition_matrix_df = transition_counts_df.copy()

    altitude_time_heatmap = work.groupby(["time_bin", "altitude_bin"], observed=False).agg(
        n_records=(object_col, "size"),
        n_objects=(object_col, "nunique"),
    ).reset_index()
    altitude_time_heatmap["scope_label"] = SCOPE_DESCRIPTIVE

    snapshot_df = pd.DataFrame(columns=["object_id", "timestamp", "shell_or_bin", "altitude_km"])
    if common_epoch is not None:
        snapshot_df = compute_common_epoch_shell_snapshot(work, common_epoch, shell_col="shell_or_bin")
        snapshot_df["scope_label"] = SCOPE_DESCRIPTIVE

    if return_extras:
        return shell_time_df, entry_exit_df, altitude_time_heatmap, transition_counts_df, transition_matrix_df, snapshot_df
    return shell_time_df, entry_exit_df, altitude_time_heatmap


def plot_altitude_time_occupancy_heatmap(altitude_time_heatmap: pd.DataFrame,
                                         value_col: str = "n_objects",
                                         title: str = "Altitude-Time Occupancy Heatmap",
                                         occupancy_heatmap_time_axis_mode: str = "datetime",
                                         occupancy_heatmap_y_label_mode: str = "bin_labels",
                                         occupancy_heatmap_norm: str = "linear",
                                         occupancy_heatmap_clip_percentile=None,
                                         occupancy_heatmap_overlay_altitude_refs_km=None,
                                         occupancy_heatmap_use_pcolormesh: bool = True,
                                         occupancy_heatmap_smoothing_sigma=None,
                                         occupancy_heatmap_value_normalization: str = "raw_counts",
                                         show_plots: bool = False,
                                         return_results: bool = False):
    if altitude_time_heatmap.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title(f"{title} (no data)")
        if show_plots:
            plt.show()
        if return_results:
            return {
                "figure": fig,
                "metadata": {
                    "status": "empty",
                    "value_col": str(value_col),
                },
            }
        return fig

    basis_col = "altitude_bin" if "altitude_bin" in altitude_time_heatmap.columns else "shell_or_bin"
    if basis_col not in altitude_time_heatmap.columns:
        raise KeyError("altitude_time_heatmap must include altitude_bin or shell_or_bin column")

    if value_col not in altitude_time_heatmap.columns:
        raise KeyError(f"value_col '{value_col}' is not present in altitude_time_heatmap")

    pivot = altitude_time_heatmap.pivot(index=basis_col, columns="time_bin", values=value_col)
    time_idx = pd.to_datetime(pivot.columns, errors="coerce")
    good_time = time_idx.notna()
    if not np.all(good_time):
        pivot = pivot.loc[:, good_time]
        time_idx = pd.to_datetime(pivot.columns, errors="coerce")

    order_time = np.argsort(time_idx.values)
    pivot = pivot.iloc[:, order_time]
    time_idx = time_idx[order_time]

    raw_labels = list(pivot.index.tolist())
    valid_label_mask = np.asarray(
        [not (pd.isna(lbl) or str(lbl).strip().lower() == "nan") for lbl in raw_labels],
        dtype=bool,
    )
    if valid_label_mask.size and not np.all(valid_label_mask):
        pivot = pivot.iloc[valid_label_mask, :]
        raw_labels = [lbl for lbl, keep in zip(raw_labels, valid_label_mask) if keep]

    mids = np.asarray([_bin_midpoint_km(v) for v in raw_labels], dtype=np.float64)
    lo_hi = np.asarray([_bin_bounds_km(v) for v in raw_labels], dtype=np.float64)

    if np.any(np.isfinite(mids)):
        missing_fill = np.where(np.isfinite(mids), mids, np.nanmax(np.where(np.isfinite(mids), mids, -np.inf)) + np.arange(mids.size) + 1.0)
        order_alt = np.argsort(missing_fill)
    else:
        order_alt = np.arange(len(raw_labels), dtype=np.int64)

    pivot = pivot.iloc[order_alt, :]
    raw_labels = [raw_labels[i] for i in order_alt]
    mids = mids[order_alt]
    lo_hi = lo_hi[order_alt]

    grid_raw = pivot.to_numpy(dtype=np.float64)

    normalization_mode = str(occupancy_heatmap_value_normalization or "raw_counts").strip().lower()
    if normalization_mode in {"per_time_total", "occupancy_fraction", "fraction"}:
        col_sum = np.nansum(grid_raw, axis=0)
        denom = np.where(col_sum > 0.0, col_sum, np.nan)
        grid = grid_raw / denom[None, :]
    else:
        normalization_mode = "raw_counts"
        grid = grid_raw.copy()

    smoothing_applied = False
    if occupancy_heatmap_smoothing_sigma is not None:
        try:
            sigma = float(occupancy_heatmap_smoothing_sigma)
        except Exception:
            sigma = None
        if sigma is not None and np.isfinite(sigma) and sigma > 0.0 and _gaussian_filter is not None:
            grid = _gaussian_filter(np.nan_to_num(grid, nan=0.0), sigma=float(sigma), mode="nearest")
            smoothing_applied = True

    fig, ax = plt.subplots(figsize=(11, 6))

    norm_obj, norm_vmin, norm_vmax = _resolve_heatmap_norm(
        grid,
        mode=occupancy_heatmap_norm,
        clip_percentile=occupancy_heatmap_clip_percentile,
    )

    use_pcolormesh = bool(occupancy_heatmap_use_pcolormesh)
    artist = None
    if use_pcolormesh and np.all(np.isfinite(mids)) and time_idx.size > 0:
        x_num = mdates.date2num(time_idx.to_pydatetime())
        x_edges = _centers_to_edges(x_num)

        lo = lo_hi[:, 0]
        hi = lo_hi[:, 1]
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            y_edges = np.concatenate(([lo[0]], hi))
        else:
            y_edges = _centers_to_edges(mids)

        artist = ax.pcolormesh(
            x_edges,
            y_edges,
            grid,
            shading="auto",
            cmap="viridis",
            norm=norm_obj,
            rasterized=True,
        )
        ax.xaxis_date()
    else:
        use_pcolormesh = False
        artist = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            norm=norm_obj,
            rasterized=True,
        )

    ax.set_title(title)

    y_mode = str(occupancy_heatmap_y_label_mode or "bin_labels").strip().lower()
    y_ticks = np.arange(len(raw_labels), dtype=np.float64)
    if use_pcolormesh and np.all(np.isfinite(mids)):
        y_ticks = mids

    if y_mode in {"midpoints", "midpoint", "km_midpoints"}:
        y_labels = [f"{m:.1f} km" if np.isfinite(m) else str(lbl) for m, lbl in zip(mids, raw_labels)]
    else:
        y_labels = [str(lbl) for lbl in raw_labels]

    max_y_ticks = 12
    y_step = int(max(1, np.ceil(len(y_ticks) / float(max_y_ticks))))
    ax.set_yticks(y_ticks[::y_step])
    ax.set_yticklabels(y_labels[::y_step])
    ax.set_ylabel("Altitude bin")

    time_mode = str(occupancy_heatmap_time_axis_mode or "datetime").strip().lower()
    if use_pcolormesh and time_mode == "datetime":
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    else:
        n_cols = int(time_idx.size)
        max_ticks = 10
        step = int(max(1, np.ceil(n_cols / float(max_ticks))))
        tick_pos = np.arange(0, n_cols, step, dtype=np.int64)
        if use_pcolormesh:
            x_num = mdates.date2num(time_idx.to_pydatetime())
            ax.set_xticks(x_num[tick_pos])
        else:
            ax.set_xticks(tick_pos)

        if time_mode == "datetime":
            tick_labels = [time_idx[i].strftime("%Y-%m-%d") for i in tick_pos]
        else:
            tick_labels = [str(i) for i in tick_pos]
        ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    ax.set_xlabel("Time")

    refs = occupancy_heatmap_overlay_altitude_refs_km
    if refs is not None:
        try:
            refs_arr = np.asarray(refs, dtype=np.float64)
        except Exception:
            refs_arr = np.asarray([], dtype=np.float64)
        for ref in refs_arr[np.isfinite(refs_arr)]:
            if use_pcolormesh:
                ax.axhline(float(ref), color="#f97316", linestyle="--", linewidth=0.8, alpha=0.7)
            elif np.any(np.isfinite(mids)):
                nearest = int(np.nanargmin(np.abs(mids - float(ref))))
                ax.axhline(float(nearest), color="#f97316", linestyle="--", linewidth=0.8, alpha=0.7)

    cb = fig.colorbar(artist, ax=ax)
    value_label = "Objects" if str(value_col).strip().lower() == "n_objects" else str(value_col)
    cb.set_label(value_label if normalization_mode == "raw_counts" else f"{value_label} ({normalization_mode})")
    fig.tight_layout()

    if show_plots:
        plt.show()

    metadata = {
        "status": "ok",
        "value_col": str(value_col),
        "basis_col": str(basis_col),
        "occupancy_basis": "altitude_bin_based" if basis_col == "altitude_bin" else "shell_fallback_based",
        "ordered_altitude_bin_labels": [str(v) for v in raw_labels],
        "ordered_time_bin_labels": [str(ts) for ts in time_idx.tolist()],
        "altitude_bin_midpoints_km": [float(v) if np.isfinite(v) else np.nan for v in mids.tolist()],
        "time_axis_mode": str(time_mode),
        "y_label_mode": str(y_mode),
        "value_normalization_mode": str(normalization_mode),
        "color_norm_mode": str(occupancy_heatmap_norm),
        "color_clip_percentile": occupancy_heatmap_clip_percentile,
        "color_vmin": float(norm_vmin) if np.isfinite(norm_vmin) else np.nan,
        "color_vmax": float(norm_vmax) if np.isfinite(norm_vmax) else np.nan,
        "use_pcolormesh": bool(use_pcolormesh),
        "smoothing_sigma": occupancy_heatmap_smoothing_sigma,
        "smoothing_applied": bool(smoothing_applied),
        "overlay_altitude_refs_km": None if refs is None else [float(v) for v in np.asarray(refs, dtype=np.float64) if np.isfinite(v)],
    }

    if return_results:
        return {
            "figure": fig,
            "metadata": metadata,
            "grid": grid,
        }
    return fig


def plot_shell_width_vs_time(shell_time_df: pd.DataFrame,
                             title: str = "Shell Width Over Time"):
    fig, ax = plt.subplots(figsize=(10, 5))
    if shell_time_df.empty:
        ax.set_title(f"{title} (no data)")
        return fig

    y_axis_max_km = 20.0
    plotted_any = False
    plotted_labels = []

    def _exclude_shell_label(shell_id) -> bool:
        if pd.isna(shell_id):
            return True
        label = str(shell_id).strip()
        if label == "" or label.lower() in {"nan", "none", "null"}:
            return True
        # Exclude altitude-bin style interval labels, e.g. (400.0, 500.0] or [299.999, 400.0]
        if "," in label and ((label.startswith("(") and label.endswith("]")) or (label.startswith("[") and label.endswith("]"))):
            return True
        return False

    for shell_id, grp in shell_time_df.groupby("shell_or_bin", sort=True):
        if _exclude_shell_label(shell_id):
            continue
        widths = pd.to_numeric(grp["shell_width_km"], errors="coerce")
        keep = np.isfinite(widths.to_numpy(dtype=np.float64))
        grp_plot = grp.loc[keep]
        if grp_plot.empty:
            continue
        plotted_any = True
        label = str(shell_id)
        plotted_labels.append(label)
        ax.plot(grp_plot["time_bin"], grp_plot["shell_width_km"], label=label, alpha=0.8)

    if plotted_any:
        ax.set_title(title)
        ax.set_ylim(bottom=0.0, top=y_axis_max_km)
    else:
        ax.set_title(f"{title} (no finite width data)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Shell Width (km)")
    if plotted_any and len(set(plotted_labels)) <= 15:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# Consolidated from disposal_corridors.py
def _local_slope_km_day(times: np.ndarray, altitude_km: np.ndarray) -> np.ndarray:
    t = times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    y = altitude_km.astype(np.float64)

    if _USE_NUMBA and t.ndim == 1 and y.ndim == 1 and t.shape == y.shape:
        return _local_slope_km_day_numba(t, y)

    slope = np.full(y.shape, np.nan, dtype=np.float64)
    if y.size < 3:
        return slope

    dt = t[2:] - t[:-2]
    dy = y[2:] - y[:-2]
    valid = np.isfinite(dt) & (dt != 0.0) & np.isfinite(dy)
    slope_mid = np.full_like(dt, np.nan, dtype=np.float64)
    slope_mid[valid] = dy[valid] / dt[valid]
    slope[1:-1] = slope_mid * 86400.0

    if y.size >= 2:
        d0 = t[1] - t[0]
        d1 = t[-1] - t[-2]
        if d0 != 0:
            slope[0] = (y[1] - y[0]) / d0 * 86400.0
        if d1 != 0:
            slope[-1] = (y[-1] - y[-2]) / d1 * 86400.0

    return slope


def _smooth_altitude_series(times: np.ndarray,
                            altitude_km: np.ndarray,
                            smoothing_method: str,
                            smoothing_window_points: int) -> np.ndarray:
    if altitude_km.size < 3:
        return altitude_km

    method = (smoothing_method or "none").lower()
    if method == "none":
        return altitude_km

    if method == "rolling":
        win = max(3, int(smoothing_window_points))
        return pd.Series(altitude_km).rolling(window=win, center=True, min_periods=1).median().to_numpy(dtype=np.float64)

    if method == "spline":
        if UnivariateSpline is None:
            return altitude_km
        t = times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
        t0 = float(t[0])
        x = (t - t0) / 86400.0
        if np.nanmax(x) <= np.nanmin(x):
            return altitude_km
        try:
            spline = UnivariateSpline(x, altitude_km, s=max(1e-6, 0.1 * altitude_km.size))
            return spline(x).astype(np.float64)
        except Exception:
            return altitude_km

    return altitude_km


def compute_disposal_corridor_metrics(df: pd.DataFrame, object_col: str | None = None,
                                      monotonic_window: int = 8,
                                      lowering_slope_threshold_km_day: float = -0.15,
                                      drop_from_km: float = 600.0,
                                      drop_to_km: float = 400.0,
                                      shell_floor_km: float = 500.0,
                                      smoothing_method: str = "none",
                                      smoothing_window_points: int = 5,
                                      active_band_km: tuple[float, float] = (500.0, 1400.0),
                                      protected_band_km: tuple[float, float] = (450.0, 750.0),
                                      uncertainty_quantiles: tuple[float, float] = (0.16, 0.84),
                                      family_col: str = "candidate_shell_id",
                                      cohort_col: str | None = None,
                                      launch_epoch_col: str | None = "launch_epoch",
                                      include_age_at_onset: bool = True,
                                      compliance_horizons_years: list[int] | tuple[int, ...] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-record and per-object empirical disposal metrics.

    Scope notes:
    - Descriptive/proxy metrics are derived from TLE-history trends.
    - Compliance-oriented estimates are heuristic and not legal determinations.
    """
    work, object_col = prepare_time_binned_panel(df, time_freq="7D", object_col=object_col)
    work = apply_duplicate_safe_conditioning(work, subset=[object_col, "timestamp"], keep="last")

    if work.empty:
        empty_record = pd.DataFrame(columns=["object_id", "timestamp", "altitude_km", "local_decay_slope_km_day",
                                             "monotonic_lowering_run", "likely_disposal_phase"])
        empty_summary = pd.DataFrame(columns=["object_id", "n_records", "disposal_onset_estimate",
                                              "shell_exit_date_estimate", "time_to_drop_days",
                                              "candidate_passive_decay", "lifetime_proxy_days",
                                              "shell_family", "cohort_label",
                                              "first_observed_timestamp", "age_at_disposal_onset_days"])
        return empty_record, empty_summary

    record_chunks = []
    summary_rows = []

    for obj, grp in work.groupby(object_col, sort=True):
        sat = grp.copy().reset_index(drop=True)
        sat["object_id"] = str(obj)

        alt = pd.to_numeric(sat["altitude_km"], errors="coerce").to_numpy(dtype=np.float64)
        ts = sat["timestamp"].to_numpy(dtype="datetime64[ns]")
        alt_smoothed = _smooth_altitude_series(ts, alt, smoothing_method=smoothing_method,
                                               smoothing_window_points=smoothing_window_points)
        slope = _local_slope_km_day(ts, alt_smoothed)

        dalt = np.diff(alt_smoothed)
        lowering_step = np.r_[False, dalt <= 0.0]
        lowering_run = pd.Series(lowering_step).rolling(window=max(2, int(monotonic_window)), min_periods=1).mean().to_numpy()

        likely = (np.isfinite(slope) & (slope <= float(lowering_slope_threshold_km_day))) & (lowering_run >= 0.8)
        sat["local_decay_slope_km_day"] = slope
        sat["altitude_km_smoothed"] = alt_smoothed
        sat["monotonic_lowering_run"] = lowering_run
        sat["likely_disposal_phase"] = likely

        if family_col in sat.columns:
            fam = sat[family_col].dropna().astype(str)
            family_label = fam.mode().iloc[0] if not fam.empty else "unknown"
        else:
            family_label = "unknown"

        cohort_label = "unknown"
        if cohort_col is not None and cohort_col in sat.columns:
            c = sat[cohort_col].dropna().astype(str)
            if not c.empty:
                cohort_label = c.mode().iloc[0]
        elif launch_epoch_col is not None and launch_epoch_col in sat.columns:
            launch_series = pd.to_datetime(sat[launch_epoch_col], errors="coerce").dropna()
            if not launch_series.empty:
                cohort_label = str(int(launch_series.iloc[0].year))

        sat["scope_label"] = SCOPE_DESCRIPTIVE
        sat["metric_scope"] = np.where(sat["likely_disposal_phase"], SCOPE_PROXY, SCOPE_DESCRIPTIVE)
        sat["shell_family"] = family_label
        sat["cohort_label"] = cohort_label

        slope_valid = slope[np.isfinite(slope)]
        slope_q_lo = float(np.nanquantile(slope_valid, uncertainty_quantiles[0])) if slope_valid.size else np.nan
        slope_q_hi = float(np.nanquantile(slope_valid, uncertainty_quantiles[1])) if slope_valid.size else np.nan

        onset_ts = pd.NaT
        likely_idx = np.flatnonzero(likely)
        if likely_idx.size > 0:
            onset_ts = sat.loc[int(likely_idx[0]), "timestamp"]

        shell_exit_ts = pd.NaT
        shell_mask = np.isfinite(alt_smoothed) & (alt_smoothed <= float(shell_floor_km))
        shell_idx = np.flatnonzero(shell_mask)
        if shell_idx.size > 0:
            shell_exit_ts = sat.loc[int(shell_idx[0]), "timestamp"]

        t_drop = np.nan
        above_from = np.flatnonzero(np.isfinite(alt_smoothed) & (alt_smoothed >= float(drop_from_km)))
        below_to = np.flatnonzero(np.isfinite(alt_smoothed) & (alt_smoothed <= float(drop_to_km)))
        if above_from.size > 0 and below_to.size > 0:
            i0 = int(above_from[0])
            i1_candidates = below_to[below_to >= i0]
            if i1_candidates.size > 0:
                i1 = int(i1_candidates[0])
                dt_days = (sat.loc[i1, "timestamp"] - sat.loc[i0, "timestamp"]).total_seconds() / 86400.0
                t_drop = float(dt_days) if np.isfinite(dt_days) else np.nan

        median_slope = np.nanmedian(slope) if np.isfinite(slope).any() else np.nan
        candidate_passive = bool(np.isfinite(median_slope) and (median_slope <= lowering_slope_threshold_km_day * 0.8))

        lifetime_proxy = np.nan
        if np.isfinite(median_slope) and median_slope < -1e-6 and np.isfinite(alt[-1]):
            lifetime_proxy = max(0.0, float((alt[-1] - 120.0) / abs(median_slope)))

        active_lo, active_hi = float(active_band_km[0]), float(active_band_km[1])
        protected_lo, protected_hi = float(protected_band_km[0]), float(protected_band_km[1])
        is_active_band = np.isfinite(alt_smoothed) & (alt_smoothed >= active_lo) & (alt_smoothed <= active_hi)
        is_protected_band = np.isfinite(alt_smoothed) & (alt_smoothed >= protected_lo) & (alt_smoothed <= protected_hi)

        day_steps = np.r_[0.0, np.diff(sat["timestamp"]).astype("timedelta64[s]").astype(np.float64) / 86400.0]
        transit_active_days = float(np.nansum(np.where(is_active_band, day_steps, 0.0)))
        transit_protected_days = float(np.nansum(np.where(is_protected_band, day_steps, 0.0)))

        est_5y = bool(np.isfinite(lifetime_proxy) and lifetime_proxy <= 5.0 * 365.25)
        est_25y = bool(np.isfinite(lifetime_proxy) and lifetime_proxy <= 25.0 * 365.25)

        onset_to_exit_days = np.nan
        if pd.notna(onset_ts) and pd.notna(shell_exit_ts):
            onset_to_exit_days = (shell_exit_ts - onset_ts).total_seconds() / 86400.0

        first_observed_ts = pd.to_datetime(sat["timestamp"], errors="coerce").dropna().min()
        launch_ref_ts = first_observed_ts
        if launch_epoch_col is not None and launch_epoch_col in sat.columns:
            launch_series = pd.to_datetime(sat[launch_epoch_col], errors="coerce").dropna()
            if not launch_series.empty:
                launch_ref_ts = launch_series.min()

        age_at_onset_days = np.nan
        if bool(include_age_at_onset) and pd.notna(onset_ts) and pd.notna(launch_ref_ts):
            age_at_onset_days = float((pd.Timestamp(onset_ts) - pd.Timestamp(launch_ref_ts)).total_seconds() / 86400.0)

        eol_indicator = sat.get("phase_state", pd.Series([None] * len(sat))).isin(["disposal_lowering", "passive_decay"]).any()

        natural_highway_indicator = bool(sat.get("natural_highway_indicator", pd.Series([False] * len(sat))).fillna(False).any())
        resonance_proximate_indicator = bool(sat.get("is_resonance_proximate", pd.Series([False] * len(sat))).fillna(False).any())
        secular_drift_indicator = bool(sat.get("secular_drift_indicator", pd.Series([False] * len(sat))).fillna(False).any())

        summary_rows.append(
            {
                "object_id": str(obj),
                "n_records": int(len(sat)),
                "median_decay_slope_km_day": median_slope,
                "decay_slope_q16_km_day": slope_q_lo,
                "decay_slope_q84_km_day": slope_q_hi,
                "disposal_onset_estimate": onset_ts,
                "shell_exit_date_estimate": shell_exit_ts,
                "onset_to_shell_exit_days": onset_to_exit_days,
                "time_to_drop_days": t_drop,
                "candidate_passive_decay": candidate_passive,
                "lifetime_proxy_days": lifetime_proxy,
                "estimated_5yr_disposal_compliance": est_5y,
                "estimated_25yr_disposal_compliance": est_25y,
                "transit_active_band_days": transit_active_days,
                "transit_protected_band_days": transit_protected_days,
                "eol_phase_observed": bool(eol_indicator),
                "shell_family": str(family_label),
                "cohort_label": str(cohort_label),
                "first_observed_timestamp": first_observed_ts,
                "age_at_disposal_onset_days": age_at_onset_days,
                "natural_highway_indicator": natural_highway_indicator,
                "resonance_proximate_indicator": resonance_proximate_indicator,
                "secular_drift_indicator": secular_drift_indicator,
                "scope_label": SCOPE_PROXY,
                "compliance_scope_label": SCOPE_COMPLIANCE,
            }
        )
        record_chunks.append(sat)

    record_df = pd.concat(record_chunks, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = add_compliance_horizon_columns(
        summary_df,
        reference_col="disposal_onset_estimate",
        prefix="estimated",
        compliance_horizons_years=compliance_horizons_years,
    )

    summary_df["estimated_5yr_transition_burden"] = safe_ratio(
        summary_df["transit_protected_band_days"],
        np.full(len(summary_df), 365.25 * 5.0),
    )
    summary_df["estimated_25yr_transition_burden"] = safe_ratio(
        summary_df["transit_protected_band_days"],
        np.full(len(summary_df), 365.25 * 25.0),
    )

    return record_df, summary_df


def summarize_disposal_onset_timeline(
    disposal_summary_df: pd.DataFrame,
    *,
    time_freq: str = "M",
    group_by: str | None = None,
    candidate_only: bool = False,
    include_age_stats: bool = True,
) -> pd.DataFrame:
    """Aggregate disposal-onset counts over time with optional cohort/family grouping."""
    cols = [
        "time_bin",
        "onset_object_count",
        "group_label",
        "mean_age_at_onset_days",
        "median_age_at_onset_days",
    ]
    if disposal_summary_df is None or disposal_summary_df.empty:
        return pd.DataFrame(columns=cols)

    work = disposal_summary_df.copy()
    work["disposal_onset_estimate"] = pd.to_datetime(work.get("disposal_onset_estimate"), errors="coerce")
    work = work.dropna(subset=["disposal_onset_estimate"])
    if work.empty:
        return pd.DataFrame(columns=cols)

    if candidate_only and "candidate_passive_decay" in work.columns:
        work = work[pd.Series(work["candidate_passive_decay"]).fillna(False).astype(bool)]
        if work.empty:
            return pd.DataFrame(columns=cols)

    work["time_bin"] = work["disposal_onset_estimate"].dt.to_period(str(time_freq)).dt.to_timestamp()

    if group_by is not None and str(group_by) in work.columns:
        group_col = str(group_by)
        work["group_label"] = work[group_col].astype(str).fillna("unknown")
    else:
        group_col = None
        work["group_label"] = "all"

    if group_col is not None:
        grouped = work.groupby(["time_bin", "group_label"], observed=False)
    else:
        grouped = work.groupby(["time_bin"], observed=False)

    out = grouped.agg(onset_object_count=("object_id", "nunique")).reset_index()
    if "group_label" not in out.columns:
        out["group_label"] = "all"

    if include_age_stats and "age_at_disposal_onset_days" in work.columns:
        age_stats = grouped["age_at_disposal_onset_days"].agg(
            mean_age_at_onset_days="mean",
            median_age_at_onset_days="median",
        ).reset_index()
        merge_keys = [c for c in ["time_bin", "group_label"] if c in out.columns and c in age_stats.columns]
        out = out.merge(age_stats, on=merge_keys, how="left")
    else:
        out["mean_age_at_onset_days"] = np.nan
        out["median_age_at_onset_days"] = np.nan

    return out.sort_values(["time_bin", "group_label"], kind="mergesort").reset_index(drop=True)


# Consolidated from sustainability_metrics.py
def compute_sustainability_metrics(df: pd.DataFrame, shell_time_df: pd.DataFrame | None = None,
                                   disposal_summary_df: pd.DataFrame | None = None,
                                   time_freq: str = "7D",
                                   count_basis: str = "both",
                                   return_shell_summary: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate sustainability indicators over time.

    Active-like/disposal-like labels are mapped from phase_state where available.
    """
    if "timestamp" not in df.columns:
        if return_shell_summary:
            return pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame()

    work, object_col = prepare_time_binned_panel(
        df,
        time_freq=time_freq,
        object_col=None,
        sort_by_time_only=True,
        ensure_altitude_features=False,
    )

    if "phase_state" not in work.columns:
        work["phase_state"] = "unknown"

    active_states = {"operational_shell", "transition", "insertion_or_orbit_raise", "relocation"}
    disposal_states = {"disposal_lowering", "passive_decay", "likely_nonoperational"}

    work["active_like"] = work["phase_state"].isin(active_states)
    work["disposal_like"] = work["phase_state"].isin(disposal_states)
    work["drifting_like"] = work["phase_state"].isin({"likely_nonoperational", "passive_decay"})

    grouped = work.groupby("time_bin", observed=False)
    out = grouped.agg(
        n_records=(object_col, "size"),
        n_objects=(object_col, "nunique"),
        active_like_count=("active_like", "sum"),
        disposal_like_count=("disposal_like", "sum"),
        drifting_like_count=("drifting_like", "sum"),
    ).reset_index()

    object_states = work.groupby(["time_bin", object_col], observed=False).agg(
        active_like=("active_like", "max"),
        disposal_like=("disposal_like", "max"),
        drifting_like=("drifting_like", "max"),
    ).reset_index()
    by_obj = object_states.groupby("time_bin", observed=False).agg(
        active_like_objects=("active_like", "sum"),
        disposal_like_objects=("disposal_like", "sum"),
        drifting_like_objects=("drifting_like", "sum"),
    ).reset_index()
    out = out.merge(by_obj, on="time_bin", how="left")

    out["disposal_fraction_objects"] = safe_ratio(out["disposal_like_objects"], out["n_objects"])
    out["drifting_fraction_objects"] = safe_ratio(out["drifting_like_objects"], out["n_objects"])
    out["active_fraction_objects"] = safe_ratio(out["active_like_objects"], out["n_objects"])

    out["disposal_fraction"] = out["disposal_fraction_objects"]
    out["drifting_fraction"] = out["drifting_fraction_objects"]
    out["disposal_fraction_record_compat"] = safe_ratio(out["disposal_like_count"], out["n_records"])
    out["drifting_fraction_record_compat"] = safe_ratio(out["drifting_like_count"], out["n_records"])
    out["normalization_scope"] = SCOPE_PROXY

    if disposal_summary_df is not None and not disposal_summary_df.empty:
        ds = disposal_summary_df.copy()
        ds["candidate_passive_decay"] = pd.Series(ds.get("candidate_passive_decay", False)).fillna(False).astype(bool)
        out["passive_decay_candidate_objects"] = int(ds["candidate_passive_decay"].sum())

        comp5 = pd.Series(ds.get("estimated_5yr_disposal_compliance", False)).fillna(False).astype(bool)
        comp25 = pd.Series(ds.get("estimated_25yr_disposal_compliance", False)).fillna(False).astype(bool)
        out["pmd_5yr_compliance_fraction"] = float(comp5.mean()) if len(comp5) else np.nan
        out["pmd_25yr_compliance_fraction"] = float(comp25.mean()) if len(comp25) else np.nan
        out["compliance_scope_label"] = SCOPE_COMPLIANCE
        out["disposal_transit_burden_mean_days"] = float(pd.to_numeric(ds.get("transit_protected_band_days"), errors="coerce").mean())
        out["active_to_disposal_object_ratio"] = safe_ratio(out["active_like_objects"], out["disposal_like_objects"])
    else:
        out["pmd_5yr_compliance_fraction"] = np.nan
        out["pmd_25yr_compliance_fraction"] = np.nan
        out["compliance_scope_label"] = SCOPE_COMPLIANCE
        out["disposal_transit_burden_mean_days"] = np.nan
        out["active_to_disposal_object_ratio"] = safe_ratio(out["active_like_objects"], out["disposal_like_objects"])

    shell_summary_df = pd.DataFrame()
    if shell_time_df is not None and not shell_time_df.empty:
        shell_work = shell_time_df.copy()
        if "time_bin" in shell_work.columns:
            shell_work["time_bin"] = pd.to_datetime(shell_work["time_bin"], errors="coerce")

        congestion = shell_work.groupby("time_bin", observed=False)["n_objects"].max().rename("shell_congestion_proxy")
        out = out.merge(congestion, on="time_bin", how="left")

        if "object_share_in_time_bin" in shell_work.columns:
            out = out.merge(
                shell_work.groupby("time_bin", observed=False)["object_share_in_time_bin"].max().rename("max_shell_object_share"),
                on="time_bin",
                how="left",
            )
        else:
            out["max_shell_object_share"] = np.nan

        out["shell_congestion_proxy_v2"] = out["shell_congestion_proxy"] * out["max_shell_object_share"].fillna(1.0)

        shell_col = "shell_or_bin" if "shell_or_bin" in shell_work.columns else None
        if shell_col is not None:
            shell_summary_df = shell_work.groupby(["time_bin", shell_col], observed=False).agg(
                shell_n_objects=("n_objects", "sum"),
                shell_n_records=("n_records", "sum"),
                shell_replenishment_count=("shell_replenishment_count", "sum") if "shell_replenishment_count" in shell_work.columns else ("n_objects", "sum"),
            ).reset_index().rename(columns={shell_col: "shell_id"})
            shell_summary_df["shell_object_fraction"] = safe_ratio(
                shell_summary_df["shell_n_objects"],
                shell_summary_df.groupby("time_bin", observed=False)["shell_n_objects"].transform("sum"),
            )
            shell_summary_df["scope_label"] = SCOPE_PROXY
    else:
        out["shell_congestion_proxy"] = np.nan
        out["max_shell_object_share"] = np.nan
        out["shell_congestion_proxy_v2"] = np.nan

    new_objects = work.groupby("time_bin", observed=False)[object_col].nunique().reset_index(name="n_unique_objects_in_bin")
    new_objects["prev_unique"] = new_objects["n_unique_objects_in_bin"].shift(1)
    new_objects["replenishment_intensity"] = (new_objects["n_unique_objects_in_bin"] - new_objects["prev_unique"]).clip(lower=0)
    out = out.merge(new_objects[["time_bin", "replenishment_intensity"]], on="time_bin", how="left")

    out["scope_label"] = SCOPE_DESCRIPTIVE
    out["proxy_scope_label"] = SCOPE_PROXY

    basis_mode = str(count_basis or "both").strip().lower()
    if basis_mode not in {"records", "objects", "both"}:
        basis_mode = "both"
    out["count_basis_mode"] = basis_mode
    if basis_mode == "records":
        out["active_like_count_selected"] = out["active_like_count"]
        out["disposal_like_count_selected"] = out["disposal_like_count"]
        out["drifting_like_count_selected"] = out["drifting_like_count"]
    else:
        out["active_like_count_selected"] = out["active_like_objects"]
        out["disposal_like_count_selected"] = out["disposal_like_objects"]
        out["drifting_like_count_selected"] = out["drifting_like_objects"]

    if return_shell_summary:
        return out, shell_summary_df
    return out


# Consolidated from risk_screening.py
def compute_conjunction_input_readiness(
    df: pd.DataFrame,
    *,
    has_covariance: bool = False,
    has_encounter_geometry: bool = False,
    has_relative_state_uncertainty: bool = False,
) -> dict[str, object]:
    """Return readiness metadata for covariance-aware conjunction workflows.

    This helper does not compute collision probability; it grades whether
    required inputs appear available for downstream conjunction assessment.
    """
    has_timestamp = "timestamp" in df.columns
    has_object_id = ("norad_cat_id" in df.columns) or ("sat_id" in df.columns)

    covariance_hint_cols = {
        "cov_r", "cov_t", "cov_n", "cov_rr", "cov_tt", "cov_nn", "sigma_r", "sigma_t", "sigma_n",
    }
    encounter_hint_cols = {
        "tca_timestamp", "miss_distance_km", "relative_speed_km_s", "relative_position_rtn_km",
    }
    uncertainty_hint_cols = {
        "relative_covariance", "combined_covariance", "relative_position_sigma_km", "relative_velocity_sigma_km_s",
    }

    cols_lower = {str(c).strip().lower() for c in df.columns}
    has_covariance = bool(has_covariance) or bool(covariance_hint_cols & cols_lower)
    has_encounter_geometry = bool(has_encounter_geometry) or bool(encounter_hint_cols & cols_lower)
    has_relative_state_uncertainty = bool(has_relative_state_uncertainty) or bool(uncertainty_hint_cols & cols_lower)

    readiness_score = sum(
        [
            bool(has_timestamp),
            bool(has_object_id),
            bool(has_covariance),
            bool(has_encounter_geometry),
            bool(has_relative_state_uncertainty),
        ]
    )
    readiness_level = "proxy_only"
    if readiness_score >= 3:
        readiness_level = "screening_ready"
    if readiness_score >= 4:
        readiness_level = "encounter_ready_not_pc"
    if readiness_score == 5:
        readiness_level = "conjunction_input_complete"

    return {
        "scope_label": SCOPE_CONJUNCTION_INPUT_READINESS,
        "readiness_level": readiness_level,
        "readiness_score": float(readiness_score) / 5.0,
        "requires_covariance_for_pc": not bool(has_covariance),
        "features": {
            "has_timestamp": bool(has_timestamp),
            "has_object_id": bool(has_object_id),
            "has_covariance": bool(has_covariance),
            "has_encounter_geometry": bool(has_encounter_geometry),
            "has_relative_state_uncertainty": bool(has_relative_state_uncertainty),
        },
    }


def compute_conjunction_grade_placeholder(
    df: pd.DataFrame,
    *,
    has_covariance: bool = False,
    has_encounter_geometry: bool = False,
    has_relative_state_uncertainty: bool = False,
) -> dict[str, object]:
    """Backward-compatible alias for conjunction input-readiness grading."""
    payload = compute_conjunction_input_readiness(
        df,
        has_covariance=has_covariance,
        has_encounter_geometry=has_encounter_geometry,
        has_relative_state_uncertainty=has_relative_state_uncertainty,
    )
    payload["scope_label"] = SCOPE_CONJUNCTION_PLACEHOLDER
    return payload


def _relative_inclination_proxy(df: pd.DataFrame) -> np.ndarray:
    if "inc" not in df.columns:
        return np.full(len(df), np.nan)
    inc = pd.to_numeric(df["inc"], errors="coerce")
    return np.abs(inc - inc.groupby(df["time_bin"]).transform("median")).to_numpy(dtype=np.float64)


def compute_risk_screening(df: pd.DataFrame, altitude_bins: list[float] | None = None,
                           time_freq: str = "7D", include_composite_score: bool = True,
                           shell_col: str = "candidate_shell_id",
                           return_severity_table: bool = False) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute lightweight density/crossing/severity proxy metrics.

    Warning: This function is not conjunction assessment and does not estimate Pc.
    """
    if altitude_bins is None:
        altitude_bins = [300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500]

    work, object_col = prepare_time_binned_panel(df, time_freq=time_freq, object_col=None, sort_by_time_only=True)
    if work.empty:
        if return_severity_table:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame(), pd.DataFrame()

    work = resolve_shell_identity(work, shell_col=shell_col, altitude_bins=altitude_bins, output_col="shell_or_bin")
    work["altitude_bin"] = pd.cut(pd.to_numeric(work["altitude_km"], errors="coerce"), bins=altitude_bins, include_lowest=True)
    work["relative_inclination_proxy_deg"] = _relative_inclination_proxy(work)

    if "phase_state" not in work.columns:
        work["phase_state"] = "unknown"

    density = work.groupby(["time_bin", "altitude_bin"], observed=False).agg(
        n_records=(object_col, "size"),
        n_objects=(object_col, "nunique"),
        active_like_density=("phase_state", lambda s: int(pd.Series(s).isin(["operational_shell", "transition", "relocation"]).sum()) if len(s) else 0),
        relative_inclination_proxy_deg=("relative_inclination_proxy_deg", "median"),
    ).reset_index()

    density["object_gradient"] = density.groupby("time_bin", observed=False)["n_objects"].diff().abs()
    density["scope_label"] = SCOPE_DESCRIPTIVE

    trans = work[[object_col, "time_bin", "shell_or_bin", "altitude_bin", "altitude_km", "shell_identity_source"]].drop_duplicates().sort_values(
        [object_col, "time_bin"],
        kind="mergesort",
    )
    trans["prev_shell"] = trans.groupby(object_col, sort=False)["shell_or_bin"].shift(1)
    trans["prev_altitude_km"] = trans.groupby(object_col, sort=False)["altitude_km"].shift(1)
    trans["crossed"] = trans["prev_shell"].notna() & (trans["shell_or_bin"] != trans["prev_shell"])
    trans["delta_altitude_km"] = trans["altitude_km"] - trans["prev_altitude_km"]
    trans["crossing_direction"] = np.where(
        trans["delta_altitude_km"] > 2.5,
        "upward",
        np.where(trans["delta_altitude_km"] < -2.5, "downward", "lateral"),
    )

    def _safe_median_abs_delta(series):
        vals = np.abs(pd.to_numeric(series, errors="coerce")).to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return np.nan
        return float(np.median(finite))

    crossing = trans.groupby("time_bin", observed=False).agg(
        shell_crossing_intensity=("crossed", "sum"),
        active_objects=(object_col, "nunique"),
        shell_overlap_proxy=("shell_or_bin", "nunique"),
        mean_abs_altitude_delta_km=("delta_altitude_km", _safe_median_abs_delta),
        shell_basis_label=("shell_identity_source", lambda s: "mixed" if pd.Series(s).nunique(dropna=True) > 1 else (pd.Series(s).iloc[0] if len(s) else "unknown")),
    ).reset_index()
    crossing["scope_label"] = SCOPE_PROXY

    x = crossing["shell_crossing_intensity"].to_numpy(dtype=np.float64)
    y = crossing["active_objects"].to_numpy(dtype=np.float64)
    z = crossing["shell_overlap_proxy"].to_numpy(dtype=np.float64)
    xn = x / np.nanmax(x) if np.nanmax(x) > 0 else np.zeros_like(x)
    yn = y / np.nanmax(y) if np.nanmax(y) > 0 else np.zeros_like(y)
    zn = z / np.nanmax(z) if np.nanmax(z) > 0 else np.zeros_like(z)

    crossing["risk_component_crossing_intensity_raw"] = x
    crossing["risk_component_active_objects_raw"] = y
    crossing["risk_component_shell_overlap_raw"] = z
    crossing["risk_component_crossing_intensity_norm"] = xn
    crossing["risk_component_active_objects_norm"] = yn
    crossing["risk_component_shell_overlap_norm"] = zn
    crossing["risk_component_weights"] = "crossing=0.5,active=0.3,overlap=0.2"
    crossing["risk_component_weighted_sum"] = 0.5 * xn + 0.3 * yn + 0.2 * zn
    crossing["risk_component_scope_label"] = SCOPE_PROXY

    if include_composite_score:
        crossing["traffic_instability_score"] = crossing["risk_component_weighted_sum"]
        crossing["shell_transit_burden_score"] = safe_ratio(
            crossing["shell_crossing_intensity"],
            crossing["active_objects"],
        )
        crossing["heuristic_risk_score"] = crossing["traffic_instability_score"]
        crossing["proxy_risk_score"] = crossing["heuristic_risk_score"]
    else:
        crossing["traffic_instability_score"] = np.nan
        crossing["shell_transit_burden_score"] = np.nan
        crossing["heuristic_risk_score"] = np.nan
        crossing["proxy_risk_score"] = np.nan

    severity = crossing.copy()
    if not severity.empty:
        score_col = "traffic_instability_score" if "traffic_instability_score" in severity.columns else "shell_crossing_intensity"
        score = pd.to_numeric(severity[score_col], errors="coerce")
        q33 = float(score.quantile(0.33)) if len(score) else 0.0
        q66 = float(score.quantile(0.66)) if len(score) else 0.0
        if not np.isfinite(q33):
            q33 = 0.0
        if not np.isfinite(q66):
            q66 = q33 + 1e-9
        if q66 <= q33:
            q66 = q33 + 1e-9

        severity["severity_bucket"] = pd.cut(
            score,
            bins=[-np.inf, q33, q66, np.inf],
            labels=["low", "medium", "high"],
            include_lowest=True,
        ).astype(str)
        severity["scope_label"] = SCOPE_PROXY

    if return_severity_table:
        return density, crossing, severity

    return density, crossing

