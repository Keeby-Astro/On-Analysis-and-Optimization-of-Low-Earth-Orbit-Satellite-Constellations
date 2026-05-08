import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.dates import DateFormatter
import pandas as pd
from time import perf_counter
from resonance_diagnostics import (
    compute_secular_rates,
    estimate_resonance_width_proxy_detailed,
    evaluate_resonance_proximity,
    get_resonance_registry,
    map_resonance_proximity_over_ai_grid,
    plot_resonance_proximity_ai_map as _plot_resonance_proximity_ai_map,
    summarize_resonant_objects,
)

def resonance(
    args_of_perigee,
    right_ascensions,
    fileNames,
    filenames_array,
    timestamps,
    phase_mode=None,
    phase_series=None,
):
    """Legacy resonance visualization for continuity with older workflows.

    Plot the kinematic phase metric 2*(Argument of Perigee) + (RAAN) over time.
    This view is retained for backward compatibility and exploratory use; physically
    grounded resonance interpretation should rely on ``resonance_physical_diagnostics``
    plus the plotting helpers in this module.

    Parameters:
        args_of_perigee (np.array): The argument of perigee of the satellites in degrees.
        right_ascensions (np.array): The right ascension of the ascending node of the satellites in degrees.
        fileNames (list): The names of the files containing the TLE data.
        filenames_array (np.array): The file index for each satellite.
        timestamps (list): The timestamps of the TLE data.

    Returns:
        None

    Migration Notes:
        Default metric remains `2*aop + raan`.
        Optional phase inputs preserve low-e-safe plotting compatibility.
    """
    t0 = perf_counter()
    # Calculate resonance-like phase combinations.
    if phase_mode is not None and phase_series is not None:
        phase_arr = np.asarray(phase_series)
        if phase_arr.shape != np.asarray(args_of_perigee).shape:
            raise ValueError("phase_series must match args_of_perigee shape")
        two_omega_plus_Omega = (2 * phase_arr + np.asarray(right_ascensions)) % 360
        y_label = '2*(True Anomaly (TLE Kepler proxy)) + (RAAN) [degrees]'
        plot_title = '2*(True Anomaly (TLE Kepler proxy)) + (RAAN) over Time'
    else:
        two_omega_plus_Omega = (2 * np.asarray(args_of_perigee) + np.asarray(right_ascensions)) % 360
        y_label = '2*(Arg of Perigee) + (RAAN) [degrees]'
        plot_title = '2*(Argument of Perigee) + (RAAN) over Time'

    # Store GUI data
    gui_data = {
        'timestamps': np.asarray(timestamps),
        'two_omega_plus_Omega': two_omega_plus_Omega,
        'filenames_array': np.asarray(filenames_array)
    }
    gui_data['timestamps_dt'] = pd.to_datetime(gui_data['timestamps'])

    display_names = list(fileNames)
    if 'All Files' not in display_names:
        display_names.append('All Files')

    order = np.argsort(gui_data['filenames_array'], kind='mergesort')
    sorted_names = gui_data['filenames_array'][order]
    unique_names, start_idx = np.unique(sorted_names, return_index=True)
    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = order.size
    bounds = {
        name: (int(s), int(e))
        for name, s, e in zip(unique_names.tolist(), start_idx.tolist(), end_idx.tolist())
    }
    all_indices = np.arange(gui_data['filenames_array'].size, dtype=np.int64)

    def get_indices(selected_name):
        if selected_name == 'All Files':
            return all_indices
        window = bounds.get(selected_name)
        if window is None:
            return np.empty(0, dtype=np.int64)
        s, e = window
        return order[s:e]

    limits_cache = {}

    def get_limits(selected_name):
        if selected_name in limits_cache:
            return limits_cache[selected_name]
        idx = get_indices(selected_name)
        if idx.size == 0:
            limits_cache[selected_name] = None
            return None
        x = gui_data['timestamps_dt'][idx]
        y = gui_data['two_omega_plus_Omega'][idx]
        lims = (x.min(), x.max(), float(np.min(y)), float(np.max(y)))
        limits_cache[selected_name] = lims
        return lims

    initial_idx = display_names.index('All Files') if 'All Files' in display_names else 0
    init_name = display_names[initial_idx]
    init_indices = get_indices(init_name)

    # Initial plot setup
    fig_resonance, ax_resonance = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    line_resonance, = ax_resonance.plot(
        gui_data['timestamps_dt'][init_indices],
        gui_data['two_omega_plus_Omega'][init_indices],
        'b-',
        linewidth=0.75,
    )
    ax_resonance.set_title(plot_title)
    ax_resonance.set_xlabel('Time')
    ax_resonance.set_ylabel(y_label)
    ax_resonance.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    # Slider setup
    ax_slider_resonance = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow', figure=fig_resonance)
    slider_resonance = Slider(ax_slider_resonance, 'File Index', 0, len(display_names) - 1, valinit=initial_idx, valstep=1)

    def update_resonance_plot(val):
        idx = int(slider_resonance.val)
        selected_filename = display_names[idx]
        t_update = perf_counter()
        indices = get_indices(selected_filename)

        filtered_timestamps = gui_data['timestamps_dt'][indices]
        filtered_two_omega_plus_Omega = gui_data['two_omega_plus_Omega'][indices]

        # Update the plot data
        line_resonance.set_data(
            filtered_timestamps,
            filtered_two_omega_plus_Omega
        )
        lims = get_limits(selected_filename)
        if lims is not None:
            ax_resonance.set_xlim(lims[0], lims[1])
            ax_resonance.set_ylim(lims[2], lims[3])
        fig_resonance.canvas.draw_idle()
        print(f"[resonance] Updated {selected_filename} in {perf_counter() - t_update:.2f}s")

    # Connect the slider to the update function
    slider_resonance.on_changed(update_resonance_plot)

    print(f"[resonance] Ready in {perf_counter() - t0:.2f}s")
    plt.show()


def resonance_physical_diagnostics(panel_df, resonance_definitions=None, tolerance_rad_day=1.0e-3):
    """Compatibility bridge to physically grounded secular resonance diagnostics.

    This wrapper keeps the legacy `resonance(...)` plot entry point unchanged while
    exposing J2/SRP-aware diagnostics for newer workflows.
    """
    registry = resonance_definitions if resonance_definitions is not None else get_resonance_registry()
    out = compute_secular_rates(panel_df)
    out = evaluate_resonance_proximity(
        out,
        resonance_definitions=registry,
        tolerance_rad_day=tolerance_rad_day,
    )
    out = estimate_resonance_width_proxy_detailed(out, method="rolling_local")
    ai_grid = map_resonance_proximity_over_ai_grid(out)
    obj_summary = summarize_resonant_objects(out)
    return {
        "panel": out,
        "ai_grid": ai_grid,
        "object_summary": obj_summary,
        "registry": registry,
        "family_counts": out["best_resonance_family"].value_counts(dropna=False).to_dict() if "best_resonance_family" in out.columns else {},
        "angle_behavior_counts": out["best_resonance_angle_behavior"].value_counts(dropna=False).to_dict() if "best_resonance_angle_behavior" in out.columns else {},
        "resonance_proximity_only": True,
        "capture_not_proven": True,
    }


def _resolve_object_col(panel_df, preferred=None):
    if preferred is not None and preferred in panel_df.columns:
        return preferred
    for c in ("sat_id", "norad_cat_id", "object_id"):
        if c in panel_df.columns:
            return c
    raise KeyError("No object identifier column found; expected one of sat_id, norad_cat_id, object_id")


def plot_resonant_angle_timeseries(
    panel_df,
    angle_col,
    object_col=None,
    timestamp_col="timestamp",
    title=None,
):
    """Plot wrapped resonant angle time series."""
    if object_col is None:
        object_col = "sat_id" if "sat_id" in panel_df.columns else "norad_cat_id"

    work = panel_df[[timestamp_col, object_col, angle_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col])

    fig, ax = plt.subplots(figsize=(10, 5))
    for obj_id, grp in work.groupby(object_col, sort=False):
        ax.plot(grp[timestamp_col], grp[angle_col], alpha=0.6, label=str(obj_id))
    ax.set_ylabel("Angle [deg, wrapped]")
    ax.set_xlabel("Time")
    ax.set_title(title or f"Resonant Angle Time Series: {angle_col}")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.tight_layout()
    return fig


def plot_resonance_residual_timeseries(
    panel_df,
    residual_col="best_resonance_abs_residual_rad_day",
    timestamp_col="timestamp",
    object_col=None,
    title="Resonance Residual Time Series",
):
    """Plot residual time series for resonance screening metrics."""
    if object_col is None:
        object_col = "sat_id" if "sat_id" in panel_df.columns else "norad_cat_id"

    work = panel_df[[timestamp_col, object_col, residual_col]].copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col])

    fig, ax = plt.subplots(figsize=(10, 5))
    for obj_id, grp in work.groupby(object_col, sort=False):
        ax.plot(grp[timestamp_col], grp[residual_col], alpha=0.6)
    ax.set_ylabel("|Residual| [rad/day]")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.tight_layout()
    return fig


def plot_resonance_phase_portrait(
    panel_df,
    angle_col,
    residual_col="best_resonance_abs_residual_rad_day",
    title="Resonance Phase Portrait",
):
    """Plot residual versus resonant angle to inspect circulation/libration patterns."""
    work = panel_df[[angle_col, residual_col]].copy()
    work[angle_col] = pd.to_numeric(work[angle_col], errors="coerce")
    work[residual_col] = pd.to_numeric(work[residual_col], errors="coerce")
    work = work.dropna()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(work[angle_col], work[residual_col], s=8, alpha=0.4)
    ax.set_xlabel(f"{angle_col} [deg]")
    ax.set_ylabel(f"{residual_col} [rad/day]")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_resonance_proximity_ai_map(
    ai_summary_df,
    value_col="median_abs_residual_rad_day",
    title="Resonance Proximity in (a, i)",
):
    """Compatibility wrapper to diagnostics heatmap plotting."""
    return _plot_resonance_proximity_ai_map(ai_summary_df, value_col=value_col, title=title)


def plot_resonance_ai_subplots(
    ai_summary_df,
    median_col="median_abs_residual_rad_day",
    fraction_col="proximate_fraction",
    title="Resonance AI Diagnostics",
):
    """Render side-by-side AI maps for residual median and proximate fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if ai_summary_df.empty:
        axes[0].set_title("Median residual (no data)")
        axes[1].set_title("Proximate fraction (no data)")
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    p1 = ai_summary_df.pivot(index="i_bin", columns="a_bin", values=median_col).to_numpy(dtype=np.float64)
    p2 = ai_summary_df.pivot(index="i_bin", columns="a_bin", values=fraction_col).to_numpy(dtype=np.float64)

    im0 = axes[0].imshow(p1, aspect="auto", origin="lower", interpolation="nearest")
    axes[0].set_title("Median |Residual| [rad/day]")
    axes[0].set_xlabel("semi-major axis bins")
    axes[0].set_ylabel("inclination bins")
    cb0 = fig.colorbar(im0, ax=axes[0])
    cb0.set_label(median_col)

    im1 = axes[1].imshow(p2, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("Proximate fraction")
    axes[1].set_xlabel("semi-major axis bins")
    axes[1].set_ylabel("inclination bins")
    cb1 = fig.colorbar(im1, ax=axes[1])
    cb1.set_label(fraction_col)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_resonance_diagnostics_dashboard(
    panel_df,
    ai_summary_df=None,
    object_col=None,
    timestamp_col="timestamp",
    angle_col=None,
    residual_col="best_resonance_abs_residual_rad_day",
    max_objects=8,
    title="Resonance Diagnostics Dashboard",
):
    """Create a multi-panel dashboard with key resonance diagnostics."""
    obj_col = _resolve_object_col(panel_df, preferred=object_col)
    work = panel_df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col])

    if angle_col is None:
        wrapped_cols = [c for c in work.columns if c.startswith("psi_") and c.endswith("_deg_wrapped")]
        if not wrapped_cols:
            raise KeyError("No wrapped resonant angle column found (expected psi_*_deg_wrapped)")
        angle_col = wrapped_cols[0]

    unwrapped_col = angle_col.replace("_deg_wrapped", "_deg_unwrapped")
    if unwrapped_col not in work.columns:
        work[unwrapped_col] = np.rad2deg(np.unwrap(np.deg2rad(pd.to_numeric(work[angle_col], errors="coerce").to_numpy(dtype=np.float64))))

    selected_ids = work[obj_col].astype(str).drop_duplicates().head(int(max_objects)).tolist()
    subset = work[work[obj_col].astype(str).isin(selected_ids)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for obj_id, grp in subset.groupby(obj_col, sort=False):
        axes[0, 0].plot(grp[timestamp_col], grp[angle_col], alpha=0.6)
        axes[0, 1].plot(grp[timestamp_col], grp[unwrapped_col], alpha=0.6)
        axes[0, 2].plot(grp[timestamp_col], grp[residual_col], alpha=0.6)

    axes[0, 0].set_title("Wrapped resonant angle")
    axes[0, 0].set_ylabel("deg")
    axes[0, 1].set_title("Unwrapped resonant angle")
    axes[0, 1].set_ylabel("deg")
    axes[0, 2].set_title("Residual time series")
    axes[0, 2].set_ylabel("rad/day")
    for ax in axes[0, :]:
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        ax.set_xlabel("Time")

    phase = subset[[angle_col, residual_col]].copy()
    phase[angle_col] = pd.to_numeric(phase[angle_col], errors="coerce")
    phase[residual_col] = pd.to_numeric(phase[residual_col], errors="coerce")
    phase = phase.dropna()
    axes[1, 0].scatter(phase[angle_col], phase[residual_col], s=8, alpha=0.35)
    axes[1, 0].set_title("Phase portrait")
    axes[1, 0].set_xlabel("angle [deg]")
    axes[1, 0].set_ylabel("|residual| [rad/day]")

    width_proxy_series = subset["resonance_width_proxy"] if "resonance_width_proxy" in subset.columns else pd.Series([], dtype=np.float64)
    hist_vals = pd.to_numeric(width_proxy_series, errors="coerce").dropna()
    axes[1, 1].hist(hist_vals.to_numpy(dtype=np.float64), bins=24, alpha=0.8, color="tab:blue")
    axes[1, 1].set_title("Width proxy distribution")
    axes[1, 1].set_xlabel("proxy")
    axes[1, 1].set_ylabel("count")

    behavior_counts = subset.get("best_resonance_angle_behavior", pd.Series([], dtype=object)).value_counts(dropna=False)
    if behavior_counts.empty:
        axes[1, 2].text(0.5, 0.5, "No behavior labels", ha="center", va="center")
        axes[1, 2].set_axis_off()
    else:
        axes[1, 2].bar(behavior_counts.index.astype(str), behavior_counts.values, color="tab:orange")
        axes[1, 2].set_title("Angle-behavior classes")
        axes[1, 2].set_ylabel("count")
        axes[1, 2].tick_params(axis="x", rotation=20)

    fig.suptitle(title)
    fig.tight_layout()

    if ai_summary_df is not None:
        fig2 = plot_resonance_ai_subplots(ai_summary_df, title=f"{title} (AI maps)")
        return fig, fig2
    return fig