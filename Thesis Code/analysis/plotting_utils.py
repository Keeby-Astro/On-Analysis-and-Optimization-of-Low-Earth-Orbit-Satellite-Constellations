"""Consolidated plotting and summary helpers for event/phase analytics.

Plotting utilities are intentionally headless-safe by default and preserve
existing return signatures while adding optional readability upgrades:
- stable phase colors
- event-time uncertainty overlays
- central, readable phase interval timeline
- optional small-multiple phase views
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, Optional, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from time import perf_counter


PHASE_COLOR_MAP = {
    "unknown": "#8D99AE",
    "insertion_or_orbit_raise": "#2A9D8F",
    "transition": "#E9C46A",
    "operational_shell": "#1D3557",
    "relocation": "#F4A261",
    "disposal_lowering": "#E76F51",
    "passive_decay": "#6D597A",
    "likely_nonoperational": "#7F1D1D",
}

EVENT_TYPE_COLOR_MAP = {
    "semi_major_axis_raise": "#1f77b4",
    "semi_major_axis_lower": "#d62728",
    "inclination_change": "#ff7f0e",
    "combined_raise_and_plane_change": "#9467bd",
    "possible_stationkeeping": "#2ca02c",
    "possible_disposal_start": "#8c564b",
    "possible_anomaly": "#7f7f7f",
    "unknown_event": "#bcbd22",
}

MISSION_PHASE_LEGEND_STATES = (
    "insertion_or_orbit_raise",
    "operational_shell",
    "disposal_lowering",
)
MISSION_PHASE_DISPLAY_LABELS = {
    "insertion_or_orbit_raise": "Insertion / Orbit Raise",
    "operational_shell": "Operational Shell",
    "disposal_lowering": "Disposal Lowering",
}


def summarize_object_time_panel(df, low_ecc_col="low_eccentricity", object_col="sat_id", altitude_regime_col="altitude_regime", candidate_shell_col="candidate_shell_id"):
    """Print and return a compact summary of the object-time panel."""
    n_objects = int(df[object_col].nunique()) if object_col in df.columns else 0
    n_records = int(len(df))

    if low_ecc_col in df.columns and n_records > 0:
        fraction_low_e = float(pd.Series(df[low_ecc_col]).fillna(False).astype(bool).mean())
    else:
        fraction_low_e = 0.0

    altitude_counts = df[altitude_regime_col].value_counts(dropna=False).to_dict() if altitude_regime_col in df.columns else {}
    shell_counts = df[candidate_shell_col].value_counts(dropna=False).to_dict() if candidate_shell_col in df.columns else {}

    summary = {
        "number_of_objects": n_objects,
        "number_of_records": n_records,
        "fraction_low_e": fraction_low_e,
        "altitude_regime_counts": altitude_counts,
        "candidate_shell_counts": shell_counts,
    }

    print("Object-Time Summary")
    print(f"  objects: {summary['number_of_objects']}")
    print(f"  records: {summary['number_of_records']}")
    print(f"  fraction low-e: {summary['fraction_low_e']:.4f}")
    print(f"  altitude regimes: {summary['altitude_regime_counts']}")
    print(f"  candidate shells: {summary['candidate_shell_counts']}")

    return summary


def _to_time(df: pd.DataFrame, time_col: str = "timestamp") -> pd.Series:
    return pd.to_datetime(df[time_col], errors="coerce")


def _select_events_for_layer(
    events_df: Optional[pd.DataFrame],
    layer_mode: str = "accepted_high_confidence",
    high_confidence_threshold: float = 0.8,
) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame()

    out = events_df.copy()
    if "event_score" in out.columns:
        score = pd.to_numeric(out["event_score"], errors="coerce").fillna(0.0)
    else:
        score = pd.Series(np.zeros(len(out), dtype=np.float64), index=out.index)
    quality = out.get("quality_flag", pd.Series([""] * len(out), index=out.index)).astype(str)
    layer = out.get("event_layer", pd.Series(["accepted"] * len(out), index=out.index)).astype(str)

    mode = str(layer_mode or "accepted_high_confidence").strip().lower()
    if mode == "all":
        keep = np.ones(len(out), dtype=bool)
    elif mode == "accepted_plus_raw":
        keep = layer.isin(["accepted", "raw_candidate", "accepted_candidate"]).to_numpy(dtype=bool)
    elif mode == "accepted_only":
        keep = (~quality.str.contains("reject", case=False, na=False)).to_numpy(dtype=bool)
    elif mode == "raw_only":
        keep = layer.str.contains("raw", case=False, na=False).to_numpy(dtype=bool)
    elif mode == "rejects_only":
        keep = quality.str.contains("reject|anomaly_screened", case=False, na=False).to_numpy(dtype=bool)
    else:
        keep = (
            (~quality.str.contains("reject", case=False, na=False))
            & (score >= float(high_confidence_threshold))
        ).to_numpy(dtype=bool)

    out = out.loc[keep].copy()
    if out.empty:
        return out
    if "estimated_event_time" in out.columns:
        out["estimated_event_time"] = pd.to_datetime(out["estimated_event_time"], errors="coerce")
        out = out.dropna(subset=["estimated_event_time"])
    return out


def _event_times(events_df: Optional[pd.DataFrame]) -> pd.Series:
    if events_df is None or events_df.empty or "estimated_event_time" not in events_df.columns:
        return pd.Series([], dtype="datetime64[ns]")
    return pd.to_datetime(events_df["estimated_event_time"], errors="coerce").dropna()


def _event_uncertainty_bounds(events_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=["event_time_lower", "event_time_upper", "estimated_event_time"])

    out = events_df.copy()
    out["estimated_event_time"] = pd.to_datetime(out.get("estimated_event_time"), errors="coerce")

    if "event_time_lower" in out.columns:
        out["event_time_lower"] = pd.to_datetime(out["event_time_lower"], errors="coerce")
    else:
        out["event_time_lower"] = out["estimated_event_time"]

    if "event_time_upper" in out.columns:
        out["event_time_upper"] = pd.to_datetime(out["event_time_upper"], errors="coerce")
    else:
        out["event_time_upper"] = out["estimated_event_time"]

    return out.dropna(subset=["estimated_event_time"]).reset_index(drop=True)


def _overlay_event_markers(
    ax: plt.Axes,
    events_df: Optional[pd.DataFrame],
    *,
    with_uncertainty: bool = True,
    layer_mode: str = "accepted_high_confidence",
    high_confidence_threshold: float = 0.8,
    color_mode: str = "event_type",
):
    if events_df is None or events_df.empty:
        return

    selected = _select_events_for_layer(
        events_df,
        layer_mode=layer_mode,
        high_confidence_threshold=high_confidence_threshold,
    )
    if selected.empty:
        return

    et = _event_times(selected)
    if et.empty:
        return

    ymin, ymax = ax.get_ylim()
    mode = str(color_mode or "event_type").strip().lower()

    if mode == "event_score":
        sc = pd.to_numeric(selected.get("event_score", np.nan), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        sc_norm = np.clip(sc, 0.0, 1.0)
        cmap = plt.get_cmap("viridis")
        for t_val, s_val in zip(et.to_numpy(), sc_norm):
            ax.vlines(t_val, ymin=ymin, ymax=ymax, color=cmap(float(s_val)), alpha=0.35, linewidth=1.2)
    elif mode == "detector_support_count":
        supp = pd.to_numeric(selected.get("detector_family_support_count", 1), errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        supp = np.clip(supp, 1.0, 5.0)
        cmap = plt.get_cmap("plasma")
        for t_val, s_val in zip(et.to_numpy(), supp):
            ax.vlines(t_val, ymin=ymin, ymax=ymax, color=cmap((s_val - 1.0) / 4.0), alpha=0.35, linewidth=1.2)
    else:
        et_df = selected.copy()
        et_df["_event_time"] = pd.to_datetime(et_df["estimated_event_time"], errors="coerce")
        for event_type, part in et_df.groupby(et_df.get("event_type", "unknown_event"), dropna=False):
            color = EVENT_TYPE_COLOR_MAP.get(str(event_type), "#B22222")
            ax.vlines(part["_event_time"].to_numpy(), ymin=ymin, ymax=ymax, color=color, alpha=0.35, linewidth=1.2)

    if not with_uncertainty:
        return

    bounds = _event_uncertainty_bounds(selected)
    if bounds.empty:
        return

    for _, row in bounds.iterrows():
        lo = row["event_time_lower"]
        hi = row["event_time_upper"]
        if pd.isna(lo) or pd.isna(hi):
            continue
        ax.axvspan(lo, hi, color="#B22222", alpha=0.06, linewidth=0)


def plot_altitude_inclination_with_events(
    df_sat: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    with_uncertainty: bool = True,
    event_layer_mode: str = "accepted_high_confidence",
    event_high_confidence_threshold: float = 0.8,
    event_color_mode: str = "event_type",
    phase_intervals_df: Optional[pd.DataFrame] = None,
    show_plots: bool = False,
) -> Dict[str, plt.Figure]:
    t = _to_time(df_sat)

    fig_alt, ax_alt = plt.subplots(figsize=(10, 4))
    if phase_intervals_df is not None and not phase_intervals_df.empty:
        p = phase_intervals_df.copy()
        p["phase_start"] = pd.to_datetime(p.get("phase_start"), errors="coerce")
        p["phase_end"] = pd.to_datetime(p.get("phase_end"), errors="coerce")
        p = p.dropna(subset=["phase_start", "phase_end"])
        for _, row in p.iterrows():
            phase_name = str(row.get("phase_state", "unknown"))
            color = PHASE_COLOR_MAP.get(phase_name, PHASE_COLOR_MAP["unknown"])
            ax_alt.axvspan(row["phase_start"], row["phase_end"], color=color, alpha=0.5, linewidth=0)
    ax_alt.plot(t, pd.to_numeric(df_sat.get("altitude_km", np.nan), errors="coerce"), label="altitude_km", color="#1D3557")
    _overlay_event_markers(
        ax_alt,
        events_df,
        with_uncertainty=with_uncertainty,
        layer_mode=event_layer_mode,
        high_confidence_threshold=event_high_confidence_threshold,
        color_mode=event_color_mode,
    )
    ax_alt.set_title("Altitude with detected events")
    ax_alt.set_xlabel("Time")
    ax_alt.set_ylabel("Altitude (km)")

    fig_inc, ax_inc = plt.subplots(figsize=(10, 4))
    ax_inc.plot(t, pd.to_numeric(df_sat.get("inc", np.nan), errors="coerce"), label="inc", color="#6D597A")
    _overlay_event_markers(
        ax_inc,
        events_df,
        with_uncertainty=with_uncertainty,
        layer_mode=event_layer_mode,
        high_confidence_threshold=event_high_confidence_threshold,
        color_mode=event_color_mode,
    )
    ax_inc.set_title("Inclination with detected events")
    ax_inc.set_xlabel("Time")
    ax_inc.set_ylabel("Inclination (deg)")

    if show_plots:
        plt.show()

    return {"altitude": fig_alt, "inclination": fig_inc}


def plot_bstar_with_events(
    df_sat: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    with_uncertainty: bool = True,
    event_layer_mode: str = "accepted_high_confidence",
    event_high_confidence_threshold: float = 0.8,
    event_color_mode: str = "event_type",
    show_plots: bool = False,
) -> plt.Figure:
    t = _to_time(df_sat)
    y = pd.to_numeric(df_sat.get("bstar_effective", df_sat.get("bstar", df_sat.get("drag_term", np.nan))), errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, color="#2A9D8F", label="bstar")
    _overlay_event_markers(
        ax,
        events_df,
        with_uncertainty=with_uncertainty,
        layer_mode=event_layer_mode,
        high_confidence_threshold=event_high_confidence_threshold,
        color_mode=event_color_mode,
    )
    ax.set_title("BSTAR with detected events")
    ax.set_xlabel("Time")
    ax.set_ylabel("BSTAR")

    if show_plots:
        plt.show()

    return fig


def plot_phase_colored_timeseries(
    df_sat: pd.DataFrame,
    phase_df: pd.DataFrame,
    *,
    alpha_by_confidence: bool = True,
    use_interval_overlay: bool = False,
    show_plots: bool = False,
) -> plt.Figure:
    data = df_sat.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    phase_cols = ["timestamp", "phase_state"]
    if "phase_confidence" in phase_df.columns:
        phase_cols.append("phase_confidence")

    phase_small = phase_df[phase_cols].copy()
    phase_small["timestamp"] = pd.to_datetime(phase_small["timestamp"], errors="coerce")
    phase_small = (
        phase_small.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset=["timestamp"], keep="last")
    )

    merged = data.merge(phase_small, on="timestamp", how="left")
    merged["phase_state"] = merged["phase_state"].fillna("unknown")

    fig, ax = plt.subplots(figsize=(10, 4))
    if use_interval_overlay and not phase_df.empty and {"phase_state", "timestamp"}.issubset(phase_df.columns):
        summary = phase_df.copy()
        summary["timestamp"] = pd.to_datetime(summary["timestamp"], errors="coerce")
        summary = summary.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        if not summary.empty:
            changes = np.flatnonzero(np.r_[True, summary["phase_state"].to_numpy(dtype=object)[1:] != summary["phase_state"].to_numpy(dtype=object)[:-1]])
            ends = np.r_[changes[1:] - 1, len(summary) - 1]
            for s_idx, e_idx in zip(changes, ends):
                phase_name = str(summary.loc[int(s_idx), "phase_state"])
                color = PHASE_COLOR_MAP.get(phase_name, PHASE_COLOR_MAP["unknown"])
                ax.axvspan(summary.loc[int(s_idx), "timestamp"], summary.loc[int(e_idx), "timestamp"], color=color, alpha=0.5, linewidth=0)

    for phase_name, color in PHASE_COLOR_MAP.items():
        part = merged[merged["phase_state"] == phase_name]
        if part.empty:
            continue
        alpha = 0.9
        if alpha_by_confidence and "phase_confidence" in part.columns:
            alpha_vals = np.clip(pd.to_numeric(part["phase_confidence"], errors="coerce").fillna(0.6).to_numpy(dtype=np.float64), 0.25, 1.0)
            alpha = float(np.nanmean(alpha_vals)) if alpha_vals.size else 0.8
        else:
            alpha = 0.9
        ax.scatter(
            part["timestamp"],
            pd.to_numeric(part.get("altitude_km", np.nan), errors="coerce"),
            color=color,
            s=9,
            alpha=alpha,
            label=phase_name,
            linewidths=0,
        )

    ax.set_title("Altitude colored by inferred phase")
    ax.set_xlabel("Time")
    ax.set_ylabel("Altitude (km)")
    ax.legend(loc="best", fontsize=12, ncol=2)

    if show_plots:
        plt.show()

    return fig


def plot_phase_small_multiples(
    df_sat: pd.DataFrame,
    phase_df: pd.DataFrame,
    *,
    value_col: str = "altitude_km",
    max_panels: int = 6,
    show_plots: bool = False,
) -> plt.Figure:
    """Optional small-multiples plot for phase-specific time slices."""
    data = df_sat.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    merged = data.merge(phase_df[["timestamp", "phase_state"]], on="timestamp", how="left")
    merged["phase_state"] = merged["phase_state"].fillna("unknown")

    phases_present = [p for p in PHASE_COLOR_MAP if (merged["phase_state"] == p).any()]
    phases_present = phases_present[: max(1, int(max_panels))]

    n_panels = len(phases_present)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, max(2.5, 2.3 * n_panels)), sharex=True)
    if n_panels == 1:
        axes = [axes]

    y = pd.to_numeric(merged.get(value_col, np.nan), errors="coerce")
    for ax, phase_name in zip(axes, phases_present):
        part = merged[merged["phase_state"] == phase_name]
        ax.scatter(part["timestamp"], pd.to_numeric(part.get(value_col, np.nan), errors="coerce"), color=PHASE_COLOR_MAP[phase_name], s=9)
        ax.plot(merged["timestamp"], y, color="#B0B7C3", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(value_col)
        ax.set_title(phase_name)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Phase small multiples", y=0.995)
    fig.tight_layout()

    if show_plots:
        plt.show()

    return fig


def plot_event_rate_histogram(
    events_df: pd.DataFrame,
    object_col: str = "object_id",
    *,
    count_basis: str = "accepted",
    high_confidence_threshold: float = 0.8,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    if events_df is None or events_df.empty:
        ax.set_title("Event counts per object (no events)")
        ax.set_xticks([])
        ax.set_xticklabels([])
        return fig

    mode = str(count_basis or "accepted").strip().lower()
    data = events_df.copy()
    if mode == "raw_segments":
        selected = _select_events_for_layer(data, layer_mode="all", high_confidence_threshold=high_confidence_threshold)
        title_suffix = "raw candidate segments"
    elif mode == "high_confidence":
        selected = _select_events_for_layer(data, layer_mode="accepted_high_confidence", high_confidence_threshold=high_confidence_threshold)
        title_suffix = "high-confidence accepted"
    else:
        selected = _select_events_for_layer(data, layer_mode="accepted_only", high_confidence_threshold=high_confidence_threshold)
        title_suffix = "merged accepted"

    if selected.empty:
        ax.set_title(f"Event counts per object ({title_suffix}; no rows)")
        ax.set_xticks([])
        ax.set_xticklabels([])
        return fig

    obj_col = object_col
    if obj_col not in selected.columns:
        obj_col = "object_id" if "object_id" in selected.columns else None
    if obj_col is None:
        ax.set_title(f"Event counts per object ({title_suffix}; missing object column)")
        ax.set_xticks([])
        ax.set_xticklabels([])
        return fig

    counts = selected[obj_col].astype(str).value_counts()
    ax.bar(counts.index, counts.values, color="#457B9D")
    ax.set_title(f"Event counts per object ({title_suffix})")
    ax.set_xlabel("")
    ax.set_ylabel("Event count")
    ax.set_xticks([])
    ax.set_xticklabels([])
    return fig


def plot_shell_entry_exit_timeline(
    phase_summary_df: pd.DataFrame,
    *,
    show_legend: bool = True,
    sort_mode: str = "first_epoch",  # first_epoch|object_id|disposal_onset|launch_epoch
    confidence_shading: bool = False,
    zoom_inset: bool = True,
    zoom_start=None,
    zoom_end="2025-12-31",
    zoom_group_window_days: float = 14.0,
    zoom_size_inches: float = 2.35,
    show_plots: bool = False,
) -> plt.Figure:
    """Render phase interval timeline with stable colors and readable row labels."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if phase_summary_df is None or phase_summary_df.empty:
        ax.set_title("Phase interval timeline (no data)")
        ax.set_yticks([])
        ax.set_yticklabels([])
        return fig

    summary = phase_summary_df.copy()
    summary["phase_start"] = pd.to_datetime(summary["phase_start"], errors="coerce")
    summary["phase_end"] = pd.to_datetime(summary["phase_end"], errors="coerce")
    summary["phase_state"] = summary.get("phase_state", "unknown").fillna("unknown").astype(str)
    summary["object_id"] = summary.get("object_id", "").astype(str)

    summary = summary.dropna(subset=["phase_start", "phase_end"])
    if summary.empty:
        ax.set_title("Phase interval timeline (no valid intervals)")
        ax.set_yticks([])
        ax.set_yticklabels([])
        return fig

    mode = str(sort_mode or "first_epoch").strip().lower()
    if mode == "object_id":
        obj_order = summary[["object_id"]].drop_duplicates().sort_values("object_id", kind="mergesort")
    elif mode == "disposal_onset":
        tmp = summary.copy()
        tmp["_disposal_start"] = np.where(
            tmp["phase_state"].astype(str) == "disposal_lowering",
            tmp["phase_start"],
            pd.NaT,
        )
        obj_order = (
            tmp.groupby("object_id", as_index=False)["_disposal_start"]
            .min()
            .sort_values("_disposal_start", kind="mergesort", na_position="last")
        )
    elif mode == "launch_epoch" and "launch_epoch" in summary.columns:
        obj_order = (
            summary[["object_id", "launch_epoch"]]
            .drop_duplicates()
            .sort_values("launch_epoch", kind="mergesort", na_position="last")
        )
    else:
        obj_order = (
            summary.groupby("object_id", as_index=False)["phase_start"]
            .min()
            .sort_values("phase_start", kind="mergesort")
        )

    object_ids = obj_order["object_id"].astype(str).tolist()
    y_map = {obj: i for i, obj in enumerate(object_ids)}

    segments_by_color: Dict[str, List[np.ndarray]] = {}
    for _, row in summary.iterrows():
        obj = str(row["object_id"])
        phase = str(row["phase_state"])
        color = PHASE_COLOR_MAP.get(phase, PHASE_COLOR_MAP["unknown"])
        alpha = 0.9
        if confidence_shading:
            conf = float(pd.to_numeric(row.get("interval_confidence", np.nan), errors="coerce"))
            if np.isfinite(conf):
                alpha = float(np.clip(conf, 0.25, 1.0))

        y = float(y_map[obj])
        x0 = mdates.date2num(row["phase_start"])
        x1 = mdates.date2num(row["phase_end"])
        seg = np.array([[x0, y], [x1, y]], dtype=np.float64)
        segments_by_color.setdefault((color, alpha), []).append(seg)

    for (color, alpha), segs in segments_by_color.items():
        arr = np.asarray(segs, dtype=np.float64)
        ax.add_collection(LineCollection(arr, linewidths=5.0, alpha=alpha, colors=color))

    x_min = mdates.date2num(summary["phase_start"].min())
    x_max = mdates.date2num(summary["phase_end"].max())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.75, float(len(object_ids) - 1) + 0.75)
    ax.xaxis_date()

    if len(object_ids) <= 20:
        ax.set_yticks(np.arange(len(object_ids), dtype=np.float64))
        ax.set_yticklabels(object_ids)
        ax.set_ylabel("Object")
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("Objects")

    ax.set_title("Mission Phase Interval Timeline")
    ax.set_xlabel("Time")

    if bool(zoom_inset):
        object_first_start = summary.groupby("object_id", sort=False)["phase_start"].min()
        first_launch = object_first_start.min() if not object_first_start.empty else pd.NaT
        if pd.notna(first_launch):
            try:
                group_window = pd.to_timedelta(float(zoom_group_window_days), unit="D")
            except Exception:
                group_window = pd.to_timedelta(14.0, unit="D")
            first_group_ids = set(
                object_first_start[object_first_start <= first_launch + group_window]
                .index.astype(str)
                .tolist()
            )
            first_group_order = [obj for obj in object_ids if obj in first_group_ids]

            zoom_start_ts = first_launch if zoom_start is None or str(zoom_start).strip().lower() in {"", "launch", "first_launch"} else pd.to_datetime(zoom_start, errors="coerce")
            zoom_end_ts = pd.to_datetime(zoom_end, errors="coerce")
            if pd.isna(zoom_end_ts) or zoom_end_ts <= zoom_start_ts:
                zoom_end_ts = zoom_start_ts + pd.DateOffset(months=2)

            zoom_rows = summary[
                summary["object_id"].astype(str).isin(first_group_ids)
                & (summary["phase_end"] >= zoom_start_ts)
                & (summary["phase_start"] <= zoom_end_ts)
            ].copy()
            if first_group_order and not zoom_rows.empty:
                try:
                    inset_size = max(1.0, float(zoom_size_inches))
                except Exception:
                    inset_size = 2.35
                ax_zoom = inset_axes(
                    ax,
                    width=inset_size,
                    height=inset_size,
                    loc="upper left",
                    borderpad=0.85,
                )
                try:
                    ax_zoom.set_in_layout(False)
                except Exception:
                    pass
                try:
                    ax_zoom.set_box_aspect(1)
                except Exception:
                    pass

                zoom_y_map = {obj: i for i, obj in enumerate(first_group_order)}
                zoom_segments_by_color: Dict[str, List[np.ndarray]] = {}
                for _, row in zoom_rows.iterrows():
                    obj = str(row["object_id"])
                    if obj not in zoom_y_map:
                        continue
                    phase = str(row["phase_state"])
                    color = PHASE_COLOR_MAP.get(phase, PHASE_COLOR_MAP["unknown"])
                    y = float(zoom_y_map[obj])
                    x0 = mdates.date2num(max(row["phase_start"], zoom_start_ts))
                    x1 = mdates.date2num(min(row["phase_end"], zoom_end_ts))
                    if x1 < x0:
                        continue
                    zoom_segments_by_color.setdefault(color, []).append(np.array([[x0, y], [x1, y]], dtype=np.float64))

                for color, segs in zoom_segments_by_color.items():
                    ax_zoom.add_collection(LineCollection(np.asarray(segs, dtype=np.float64), linewidths=3.0, colors=color, alpha=0.95))

                ax_zoom.set_xlim(mdates.date2num(zoom_start_ts), mdates.date2num(zoom_end_ts))
                ax_zoom.set_ylim(-0.75, float(len(first_group_order) - 1) + 0.75)
                ax_zoom.xaxis_date()
                ax_zoom.set_yticks([])
                ax_zoom.tick_params(axis="x", which="both", labelbottom=False, length=2)
                ax_zoom.tick_params(axis="y", length=0)
                ax_zoom.grid(True, linewidth=0.35, alpha=0.45)
                ax_zoom.set_facecolor("white")
                ax_zoom.patch.set_alpha(0.96)
                try:
                    locator_box, connector1, connector2 = mark_inset(
                        ax,
                        ax_zoom,
                        loc1=2,
                        loc2=4,
                        fc="none",
                        ec="0.25",
                        linewidth=0.8,
                    )
                    for artist in (locator_box, connector1, connector2):
                        artist.set_zorder(6)
                except Exception:
                    pass

    if show_legend:
        handles = [
            Patch(
                facecolor=PHASE_COLOR_MAP[state],
                edgecolor="none",
                label=MISSION_PHASE_DISPLAY_LABELS[state],
            )
            for state in MISSION_PHASE_LEGEND_STATES
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=12)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout()

    if show_plots:
        plt.show()

    return fig


# Consolidated from plot_style.py
COLORS = ['#15528e', '#b25800', '#1e701e', '#951c1c', '#673284',
          '#623c34', '#9e5387', '#585858', '#848417', '#108590',
          '#798ba2', '#b28254', '#6a9c60', '#b26a68', '#8a7b94',
          '#896d67', '#ac7f93', '#8b8b8b', '#999962', '#6f989f']

def apply_plot_style():
    plt.rcParams.update({
        'figure.figsize': (10.0, 7.5),
        'xtick.direction': 'in', 'xtick.labelsize': 14, 'xtick.major.size': 3,
        'xtick.major.width': 0.5, 'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5,
        'xtick.minor.visible': True, 'xtick.top': True,
        'ytick.direction': 'in', 'ytick.labelsize': 14, 'ytick.major.size': 3,
        'ytick.major.width': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
        'ytick.minor.visible': True, 'ytick.right': True,
        'axes.linewidth': 0.5, 'grid.linewidth': 0.5, 'lines.linewidth': 1.0,
        'legend.fontsize': 12, 'legend.frameon': False,
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'dejavuserif',
        'font.size': 12, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'axes.grid': True, 'grid.linestyle': '--', 'grid.color': '0.5',
        'lines.markersize': 8, 'axes.spines.top': True, 'axes.spines.right': True,
        'path.simplify': True,
        'path.simplify_threshold': 0.2,
        'agg.path.chunksize': 20000,
    })
    plt.rcParams['axes.prop_cycle'] = cycler(color=COLORS)


# Consolidated from precession_rates.py
def _unit_factor_to_rad_s(input_units: str) -> float:
    key = str(input_units).lower()
    if key == "rad_s":
        return 1.0
    if key == "rad_day":
        return 1.0 / 86400.0
    if key == "deg_day":
        return np.pi / (180.0 * 86400.0)
    raise ValueError("input_units must be one of {'rad_s', 'rad_day', 'deg_day'}")


def _unit_factor_from_rad_s(output_units: str) -> float:
    key = str(output_units).lower()
    if key == "rad_s":
        return 1.0
    if key == "rad_day":
        return 86400.0
    if key == "deg_day":
        return 86400.0 * (180.0 / np.pi)
    raise ValueError("output_units must be one of {'rad_s', 'rad_day', 'deg_day'}")


def _normalize_precession_trend_mode(mode: str) -> str:
    value = str(mode or "none").strip().lower()
    if value in {"none", "ols", "huber"}:
        return value
    return "none"


def _precession_output_unit_label(output_units: str) -> str:
    key = str(output_units).strip().lower()
    labels = {
        "rad_s": "rad/s",
        "rad_day": "rad/day",
        "deg_day": "deg/day",
    }
    return labels.get(key, str(output_units))


def _precession_group_numeric_value(group_label) -> Optional[float]:
    try:
        value = float(group_label)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(group_label))
    if match is None:
        return None
    try:
        value = float(match.group(0))
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _format_precession_group_label(group_label) -> str:
    value = _precession_group_numeric_value(group_label)
    if value is None:
        return str(group_label)
    return f"{value:.1f} deg"


def _ordered_precession_groups(group_labels, resolved_targets=None) -> List[object]:
    groups = [group for group in pd.unique(group_labels) if pd.notna(group)]
    target_order = {}
    for target_index, target in enumerate(resolved_targets or []):
        try:
            target_value = float(target)
        except Exception:
            continue
        if np.isfinite(target_value):
            target_order[round(target_value, 6)] = target_index

    def sort_key(group_label):
        value = _precession_group_numeric_value(group_label)
        if value is None:
            return (2, str(group_label))
        rounded_value = round(value, 6)
        if rounded_value in target_order:
            return (0, target_order[rounded_value])
        return (1, value)

    return sorted(groups, key=sort_key)


def _assign_inclination_groups(
    inclinations_deg,
    target_inclinations_deg,
    assignment_tolerance_deg,
):
    inc = np.asarray(inclinations_deg, dtype=np.float64)
    labels = np.full(inc.shape, np.nan, dtype=object)

    targets = []
    for item in np.asarray(target_inclinations_deg if target_inclinations_deg is not None else [], dtype=np.float64):
        if np.isfinite(item):
            targets.append(float(item))
    targets = sorted(set(targets))

    tol = float(assignment_tolerance_deg)
    if not np.isfinite(tol):
        tol = 0.5
    tol = max(0.0, tol)

    if inc.size == 0 or len(targets) == 0:
        return labels, targets, tol

    target_arr = np.asarray(targets, dtype=np.float64)
    dist = np.abs(inc[:, None] - target_arr[None, :])
    nearest_idx = np.argmin(dist, axis=1)
    nearest_dist = dist[np.arange(inc.size), nearest_idx]
    keep = np.isfinite(inc) & (nearest_dist <= tol)

    for idx, target in enumerate(target_arr.tolist()):
        mask = keep & (nearest_idx == idx)
        if np.any(mask):
            labels[mask] = _format_precession_group_label(target)

    return labels, targets, tol


def _fit_huber_line_days(x_days: np.ndarray, y_vals: np.ndarray, max_iter: int = 50, c: float = 1.345):
    if x_days.size < 2:
        return None
    X = np.column_stack((x_days, np.ones_like(x_days)))
    beta = np.linalg.lstsq(X, y_vals, rcond=None)[0]
    for _ in range(int(max_iter)):
        resid = y_vals - X @ beta
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1.0e-12:
            break
        u = resid / scale
        w = np.ones_like(u)
        large = np.abs(u) > c
        w[large] = c / np.maximum(np.abs(u[large]), 1.0e-12)
        Xw = X * w[:, None]
        yw = y_vals * w
        beta_new = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < 1.0e-10:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def _fit_precession_trend(x_days: np.ndarray, y_vals: np.ndarray, trend_mode: str):
    mode = _normalize_precession_trend_mode(trend_mode)
    if mode == "none" or x_days.size < 2:
        return None
    if mode == "ols":
        coeff = np.polyfit(x_days, y_vals, deg=1)
        return float(coeff[0]), float(coeff[1]), "ols"
    huber_fit = _fit_huber_line_days(x_days, y_vals)
    if huber_fit is None:
        return None
    slope, intercept = huber_fit
    return float(slope), float(intercept), "huber"


def _compute_rolling_quantiles(
    dates,
    y_vals,
    *,
    window_days: Optional[float] = 30.0,
    window_samples: Optional[int] = None,
    lower_q: float = 0.10,
    upper_q: float = 0.90,
    min_periods: int = 5,
):
    ts = pd.to_datetime(np.asarray(dates), errors="coerce")
    y = np.asarray(y_vals, dtype=np.float64)
    keep = (~pd.isna(ts)) & np.isfinite(y)
    if np.sum(keep) < max(3, int(min_periods)):
        return None

    ts_keep = pd.to_datetime(ts[keep])
    y_keep = y[keep]
    order = np.argsort(ts_keep.view("int64"), kind="mergesort")
    ts_sorted = ts_keep[order]
    y_sorted = y_keep[order]

    series = pd.Series(y_sorted, index=pd.DatetimeIndex(ts_sorted))
    use_samples = False
    win_obj = None
    if window_samples is not None:
        try:
            ws = int(window_samples)
            if ws >= 2:
                win_obj = ws
                use_samples = True
        except Exception:
            win_obj = None

    if win_obj is None:
        try:
            wd = float(window_days)
        except Exception:
            wd = 30.0
        if not np.isfinite(wd) or wd <= 0.0:
            wd = 30.0
        win_obj = f"{wd:.6f}D"

    roll = series.rolling(win_obj, min_periods=max(2, int(min_periods)))
    med = roll.median()
    low = roll.quantile(float(lower_q))
    high = roll.quantile(float(upper_q))

    valid = np.isfinite(med.values) & np.isfinite(low.values) & np.isfinite(high.values)
    if np.sum(valid) < 2:
        return None

    return {
        "x": med.index[valid],
        "median": med.values[valid],
        "low": low.values[valid],
        "high": high.values[valid],
        "window_mode": "samples" if use_samples else "days",
        "window_value": int(win_obj) if use_samples else str(win_obj),
    }


def _compute_group_rate_stats(group_labels, y_vals, finite_mask):
    if group_labels is None:
        if np.any(finite_mask):
            return {
                "all": {
                    "n": int(np.sum(finite_mask)),
                    "mean": float(np.nanmean(y_vals[finite_mask])),
                    "median": float(np.nanmedian(y_vals[finite_mask])),
                }
            }
        return {}

    out = {}
    for grp in _ordered_precession_groups(group_labels):
        mask = finite_mask & (group_labels == grp)
        if not np.any(mask):
            continue
        out[_format_precession_group_label(grp)] = {
            "n": int(np.sum(mask)),
            "mean": float(np.nanmean(y_vals[mask])),
            "median": float(np.nanmedian(y_vals[mask])),
        }
    return out


def precession_rates(
    node_precession_rates,
    perigee_precession_rates,
    dates,
    *,
    show_plots=True,
    return_figures=True,
    model_label="J2_first_order",
    input_units="rad_s",
    output_units="deg_day",
    apsidal_rate_valid=None,
    suppress_apsidal_when_invalid=False,
    fit_trends=True,
    group_labels=None,
    comparison_model_rates=None,
    inclinations_deg=None,
    eccentricities=None,
    precession_group_targets_deg=None,
    precession_group_assignment_tolerance_deg=0.5,
    precession_group_trend_mode="none",
    precession_group_rolling_window_days=30,
    precession_group_rolling_window_samples=None,
    precession_show_group_envelopes=True,
    precession_apsidal_ecc_floor=1.0e-3,
    precession_low_e_behavior="suppress",
    precession_node_xlim=(18103.234478763217, 20565.8996289785),
    precession_node_ylim=(-6.010353780510761, 1.7029920575427253),
    precession_apsidal_xlim=(18103.579595386887, 20565.8996289785),
    precession_apsidal_ylim=(-5.042730375994507, 5.026006480170245),
):
    """
    Plot J2 nodal precession rates over time.

    Parameters:
        node_precession_rates (list): Nodal rates in units specified by input_units.
        perigee_precession_rates (list): Unused legacy argument.
        dates (list): The dates corresponding to the precession rates.
        show_plots (bool): Whether to call plt.show() for interactive sessions.
        return_figures (bool): Whether to return figure/axes handles.
        model_label (str): Label printed in plot titles.
        input_units (str): Units of provided rates: rad_s, rad_day, deg_day.
        output_units (str): Units to display on plots: rad_s, rad_day, deg_day.
        apsidal_rate_valid (array-like|None): Unused legacy argument.
        suppress_apsidal_when_invalid (bool): Unused legacy argument.
        fit_trends (bool): Add linear trend fits per series (or per group when labels provided).
        group_labels (array-like|None): Optional group labels for grouped plotting/fits.
        comparison_model_rates (dict|None): Legacy argument retained for compatibility; ignored.

    Returns:
        dict|None: Figure/axes payload when return_figures is True, else None.
    """
    t0 = perf_counter()
    print("[precession_rates] Preparing J2 precession plot...")

    to_rad_s = _unit_factor_to_rad_s(input_units)
    from_rad_s = _unit_factor_from_rad_s(output_units)
    output_unit_label = _precession_output_unit_label(output_units)

    node_precession_rates = np.asarray(node_precession_rates, dtype=np.float64) * to_rad_s * from_rad_s
    perigee_precession_rates = np.asarray(perigee_precession_rates, dtype=np.float64) * to_rad_s * from_rad_s
    dates = pd.to_datetime(dates)

    if len(node_precession_rates) != len(dates):
        raise ValueError("node_precession_rates and dates must have equal length")
    if len(perigee_precession_rates) != len(dates):
        raise ValueError("perigee_precession_rates and dates must have equal length")

    if group_labels is not None:
        group_labels = np.asarray(group_labels)
        if len(group_labels) != len(dates):
            raise ValueError("group_labels must have same length as dates")
    elif inclinations_deg is not None:
        inc_arr = np.asarray(inclinations_deg, dtype=np.float64)
        if len(inc_arr) != len(dates):
            raise ValueError("inclinations_deg must have same length as dates")
        default_targets = [53.0, 70.0, 97.6]
        targets_input = default_targets if precession_group_targets_deg is None else precession_group_targets_deg
        group_labels, resolved_targets, resolved_tol = _assign_inclination_groups(
            inc_arr,
            targets_input,
            precession_group_assignment_tolerance_deg,
        )
    else:
        resolved_targets = []
        try:
            resolved_tol = float(precession_group_assignment_tolerance_deg)
        except Exception:
            resolved_tol = 0.5

    if group_labels is not None and "resolved_targets" not in locals():
        targets_input = [53.0, 70.0, 97.6] if precession_group_targets_deg is None else precession_group_targets_deg
        resolved_targets = [float(v) for v in np.asarray(targets_input, dtype=np.float64) if np.isfinite(v)]
        resolved_tol = float(precession_group_assignment_tolerance_deg) if np.isfinite(float(precession_group_assignment_tolerance_deg)) else 0.5

    trend_mode = _normalize_precession_trend_mode(precession_group_trend_mode)

    if eccentricities is not None:
        ecc_arr = np.asarray(eccentricities, dtype=np.float64)
        if len(ecc_arr) != len(dates):
            raise ValueError("eccentricities must have same length as dates")
    else:
        ecc_arr = None

    low_e_behavior = str(precession_low_e_behavior or "suppress").strip().lower()
    if low_e_behavior not in {"suppress", "highlight"}:
        low_e_behavior = "suppress"

    gui_data = {
        'node_precession_rates': node_precession_rates,
        'perigee_precession_rates': perigee_precession_rates,
        'dates': dates,
    }

    def _time_axis_days(ts: pd.DatetimeIndex) -> np.ndarray:
        t_seconds = (ts.view("int64") / 1.0e9).astype(np.float64)
        return (t_seconds - t_seconds[0]) / 86400.0

    # Function to plot precession rates
    panel_group_stats = {}

    def plot_precession_rate(
        rate_key,
        title,
        ylabel,
        color,
        validity_mask=None,
        legend_loc='upper right',
        legend_anchor_y=None,
        low_conf_mask=None,
        x_limits=None,
        y_limits=None,
    ):
        t_plot = perf_counter()
        y = np.asarray(gui_data[rate_key], dtype=np.float64)
        x_days = _time_axis_days(gui_data['dates'])
        finite = np.isfinite(y)
        if validity_mask is not None:
            finite &= np.asarray(validity_mask, dtype=bool)

        mean_rate = np.nan
        if np.any(finite):
            mean_rate = float(np.nanmean(y[finite]))

        fig_pr, ax_pr = plt.subplots()
        if group_labels is None:
            ax_pr.scatter(gui_data['dates'][finite], y[finite],
                          color=color, label='Precession Rate', alpha=0.75, s=5)
            if fit_trends and np.sum(finite) >= 2:
                coeff = np.polyfit(x_days[finite], y[finite], deg=1)
                trend = coeff[0] * x_days + coeff[1]
                ax_pr.plot(gui_data['dates'], trend, color='k', linestyle='-.', linewidth=1.2,
                           label=f"Trend ({coeff[0]:.3e} {output_unit_label}/day)")
            if np.isfinite(mean_rate):
                ax_pr.axhline(y=mean_rate, color='k', linestyle='--', label='Average Rate')
        else:
            uniq = _ordered_precession_groups(group_labels, resolved_targets)
            cmap = plt.get_cmap('tab10')
            for idx, grp in enumerate(uniq):
                grp_mask = (group_labels == grp) & finite
                if not np.any(grp_mask):
                    continue
                display_label = _format_precession_group_label(grp)
                ax_pr.scatter(gui_data['dates'][grp_mask], y[grp_mask],
                              s=5, color=cmap(idx % 10), alpha=0.8, label=display_label)

                grp_mean_rate = float(np.nanmean(y[grp_mask])) if np.any(grp_mask) else np.nan
                if np.isfinite(grp_mean_rate):
                    ax_pr.axhline(
                        y=grp_mean_rate,
                        color=cmap(idx % 10),
                        linestyle=':',
                        linewidth=1.2,
                        alpha=0.95,
                        label=f"{display_label} Average Rate",
                    )

                if fit_trends:
                    fit_payload = _fit_precession_trend(x_days[grp_mask], y[grp_mask], trend_mode)
                    if fit_payload is not None:
                        slope, intercept, mode_used = fit_payload
                        trend = slope * x_days + intercept
                        ax_pr.plot(
                            gui_data['dates'],
                            trend,
                            color=cmap(idx % 10),
                            linestyle='--',
                            linewidth=1.0,
                            alpha=0.9,
                            label=f"{display_label} {mode_used} trend ({slope:.3e} {output_unit_label}/day)",
                        )

                if bool(precession_show_group_envelopes):
                    summary = _compute_rolling_quantiles(
                        gui_data['dates'][grp_mask],
                        y[grp_mask],
                        window_days=precession_group_rolling_window_days,
                        window_samples=precession_group_rolling_window_samples,
                        lower_q=0.10,
                        upper_q=0.90,
                        min_periods=5,
                    )
                    if summary is not None:
                        ax_pr.plot(
                            summary['x'],
                            summary['median'],
                            color=cmap(idx % 10),
                            linewidth=1.4,
                            alpha=0.85,
                            label=f"{display_label} Rolling Median",
                        )
                        ax_pr.fill_between(
                            summary['x'],
                            summary['low'],
                            summary['high'],
                            color=cmap(idx % 10),
                            alpha=0.12,
                            linewidth=0.0,
                        )

        if low_conf_mask is not None and np.any(low_conf_mask):
            low_mask = np.asarray(low_conf_mask, dtype=bool) & np.isfinite(y)
            if np.any(low_mask):
                ax_pr.scatter(
                    gui_data['dates'][low_mask],
                    y[low_mask],
                    s=9,
                    color='#7a7a7a',
                    alpha=0.55,
                    marker='x',
                    label='Low-confidence apsidal points',
                )

        ax_pr.set_title(title)
        ax_pr.set_xlabel('Time')
        ax_pr.set_ylabel(ylabel)

        if isinstance(x_limits, (tuple, list)) and len(x_limits) == 2:
            try:
                ax_pr.set_xlim(float(x_limits[0]), float(x_limits[1]))
            except Exception:
                pass
        if isinstance(y_limits, (tuple, list)) and len(y_limits) == 2:
            try:
                ax_pr.set_ylim(float(y_limits[0]), float(y_limits[1]))
            except Exception:
                pass

        handles, labels = ax_pr.get_legend_handles_labels()
        if len(labels) > 0:
            legend_kwargs = {"loc": legend_loc, "fontsize": 12}
            if legend_anchor_y is not None:
                legend_kwargs["bbox_to_anchor"] = (0.01, float(legend_anchor_y))
                legend_kwargs["bbox_transform"] = ax_pr.get_yaxis_transform()
            ax_pr.legend(**legend_kwargs)
        ax_pr.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        fig_pr.tight_layout()
        panel_group_stats[rate_key] = _compute_group_rate_stats(group_labels, y, finite)
        print(f"[precession_rates] {rate_key} figure ready in {perf_counter() - t_plot:.2f}s")
        if show_plots:
            plt.show()
        return fig_pr, ax_pr

    node_fig, node_ax = plot_precession_rate(
        'node_precession_rates',
        'J2 Node Precession Rates Over Time',
        f'$J_2$ Node Precession Rate ({output_unit_label})',
        'b',
        validity_mask=np.isfinite(node_precession_rates),
        legend_loc='upper left',
        legend_anchor_y=0.95,
        x_limits=precession_node_xlim,
        y_limits=precession_node_ylim,
    )

    apsidal_base_mask = np.isfinite(perigee_precession_rates)
    apsidal_valid_mask = apsidal_base_mask.copy()
    if apsidal_rate_valid is not None:
        apsidal_valid_mask &= np.asarray(apsidal_rate_valid, dtype=bool)
    if ecc_arr is not None:
        apsidal_valid_mask &= np.isfinite(ecc_arr) & (ecc_arr >= float(precession_apsidal_ecc_floor))

    finite_apsidal_count = int(np.sum(apsidal_base_mask))
    valid_apsidal_count = int(np.sum(apsidal_valid_mask))
    apsidal_valid_fraction = (float(valid_apsidal_count) / float(finite_apsidal_count)) if finite_apsidal_count > 0 else np.nan

    apsidal_low_conf_mask = apsidal_base_mask & (~apsidal_valid_mask)

    if low_e_behavior == "highlight":
        apsidal_plot_mask = apsidal_base_mask
        apsidal_low_conf_overlay = apsidal_low_conf_mask
    else:
        apsidal_plot_mask = apsidal_valid_mask
        apsidal_low_conf_overlay = None

    should_skip_apsidal = bool(suppress_apsidal_when_invalid and np.isfinite(apsidal_valid_fraction) and apsidal_valid_fraction < 0.5)
    if should_skip_apsidal:
        apsidal_fig, apsidal_ax = None, None
    else:
        apsidal_fig, apsidal_ax = plot_precession_rate(
            'perigee_precession_rates',
            'J2 Apsidal Precession Rates Over Time',
            f'$J_2$ Apsidal Precession Rate ({output_unit_label})',
            'r',
            validity_mask=apsidal_plot_mask,
            legend_loc='lower left',
            legend_anchor_y=-1.85,
            low_conf_mask=apsidal_low_conf_overlay,
            x_limits=precession_apsidal_xlim,
            y_limits=precession_apsidal_ylim,
        )

    print(f"[precession_rates] Ready in {perf_counter() - t0:.2f}s")

    if not return_figures:
        return None
    return {
        "node": {"fig": node_fig, "ax": node_ax},
        "apsidal": {"fig": apsidal_fig, "ax": apsidal_ax, "skipped": bool(should_skip_apsidal)},
        "comparison": {"fig": None, "axes": None},
        "metadata": {
            "model_label": model_label,
            "input_units": input_units,
            "output_units": output_units,
            "apsidal_valid_fraction": apsidal_valid_fraction,
            "precession_group_targets_deg": [float(v) for v in np.asarray(resolved_targets, dtype=np.float64) if np.isfinite(v)],
            "precession_group_assignment_tolerance_deg": float(resolved_tol),
            "precession_group_trend_mode": trend_mode,
            "precession_group_rolling_window_days": None if precession_group_rolling_window_days is None else float(precession_group_rolling_window_days),
            "precession_group_rolling_window_samples": None if precession_group_rolling_window_samples is None else int(precession_group_rolling_window_samples),
            "precession_show_group_envelopes": bool(precession_show_group_envelopes),
            "precession_apsidal_ecc_floor": float(precession_apsidal_ecc_floor),
            "precession_low_e_behavior": low_e_behavior,
            "precession_node_xlim": None if precession_node_xlim is None else [float(precession_node_xlim[0]), float(precession_node_xlim[1])],
            "precession_node_ylim": None if precession_node_ylim is None else [float(precession_node_ylim[0]), float(precession_node_ylim[1])],
            "precession_apsidal_xlim": None if precession_apsidal_xlim is None else [float(precession_apsidal_xlim[0]), float(precession_apsidal_xlim[1])],
            "precession_apsidal_ylim": None if precession_apsidal_ylim is None else [float(precession_apsidal_ylim[0]), float(precession_apsidal_ylim[1])],
            "apsidal_low_confidence_fraction": (float(np.sum(apsidal_low_conf_mask)) / float(finite_apsidal_count)) if finite_apsidal_count > 0 else np.nan,
            "apsidal_low_confidence_points_hidden": bool(low_e_behavior == "suppress"),
            "group_rate_statistics": {
                "node_precession_rates": panel_group_stats.get('node_precession_rates', {}),
                "perigee_precession_rates": panel_group_stats.get('perigee_precession_rates', {}),
            },
        },
    }

