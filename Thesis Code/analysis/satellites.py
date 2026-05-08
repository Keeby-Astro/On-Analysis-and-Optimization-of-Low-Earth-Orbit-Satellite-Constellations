"""Starlink TLE workflow orchestration

This module keeps a thin CLI wrapper for interactive runs and exposes a
reusable dict-driven pipeline function for scripted/reproducible execution.
"""

from __future__ import annotations

import os
import gc
import json
import re
import subprocess
from datetime import datetime
from copy import deepcopy
from time import perf_counter
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import (DEFAULT_SYNC_TOLERANCE, J2_EARTH, LOW_ECCENTRICITY_THRESHOLD,
                       PHASE_SEMANTICS_TRUE_ANOMALY_PROXY,
                       PHASE_VARIABLE_TRUE_ANOMALY, RADIUS_EARTH,
                       SYNC_MODE_NEAREST_INTERSECTION, SYNC_MODE_TARGET_NEAREST)
from epoch_sync import find_common_epoch_records
from load_all_tle_data import load_all_tle_data
from orbital_features import add_standard_tle_proxy_enrichment
from orbital_visualization import orbital_visualization

from plot_data_export import (
    DEFAULT_COMPRESS_THRESHOLD_BYTES as PLOT_DATA_COMPRESS_THRESHOLD_BYTES,
    capture_open_figures as _capture_open_figures_for_export,
)
from plotting_utils import apply_plot_style
from secular_perturbations import calculate_precession_rates
from state_models import SUPPORTED_STATE_MODELS, _satrec_from_lines_cached, normalize_state_model


DEFAULT_FOLDER_PATH = r"C:\Users\PC\Code\Research\backup_starlink"
DEFAULT_GENERATION_SPLIT_PATH = r"C:\Users\PC\Code\starlink_generation_split"
DEFAULT_FULL_EXPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "full_exports"))
DEFAULT_CACHED_PHASE_LABELS_CSV = os.path.join(DEFAULT_FULL_EXPORTS_DIR, "maneuver_phase_labels_gen1_full.csv")
DEFAULT_CACHED_PHASE_INTERVALS_CSV = os.path.join(DEFAULT_FULL_EXPORTS_DIR, "maneuver_phase_intervals_gen1_full.csv")
MAX_ANALYSIS_SMA_KM = 7000.0
DEFAULT_MANEUVER_RELEVANT_PHASE_STATES = (
    "insertion_or_orbit_raise",
    "transition",
    "operational_shell",
    "disposal_lowering",
    "passive_decay",
)
DEFAULT_PLOT_SAVE_SKIP_FILENAMES = frozenset(
    {
        "007_Density_Plot_RAAN_vs._True_Anomaly_TLE_Kepler_proxy.png",
        "008_RAAN_vs_True_Anomaly_TLE_Kepler_proxy.png",
        "009_RAAN_vs_True_Anomaly_TLE_Kepler_proxy_In_Relation_to_Eccentricity.png",
        "012_Right_Ascension_of_Ascending_Node_Over_Time.png",
        "013_Argument_of_Perigee_Over_Time_low-e_filtered.png",
        "015_True_Anomaly_TLE_Kepler_proxy_Over_Time.png",
        "019_Semi-Major_Axis_vs_True_Anomaly_TLE_Kepler_proxy_Circular-Linear.png"
        "020_RAAN_vs_True_Anomaly_TLE_Kepler_proxy_Aggregate__All_times__n119384.png",
        "021_True_Anomaly_TLE_Kepler_proxy_vs_Altitude_Aggregate__All_times__n119384.png",
        "024_Altitude-Time_Occupancy_Heatmap.png",
        "027_Risk_Proxy_Timeline.png",
        "028_Altitude_with_detected_events.png",
        "029_Inclination_with_detected_events.png",
        "030_BSTAR_with_detected_events.png",
        "031_Altitude_colored_by_inferred_phase.png",
        "032_Event_counts_per_object_no_events.png"
    }
)


def get_default_pipeline_config() -> Dict[str, Any]:
    """Return conservative defaults for the dict-driven pipeline."""
    return {
        "folder_paths": [DEFAULT_FOLDER_PATH],  # One or more TLE folder roots to ingest.
        "only_files": None,  # Optional explicit filename allowlist (basename match).
        "ingest": {
            "validate_checksum": None,  # None=use env/default; True/False overrides TLE checksum validation.
        },
        "derived": {"sma", "true_anomaly", "specific_angular_momentum"},  # Derived columns from raw TLE fields.
        "quality_control": {
            "drop_invalid_timestamp": True,  # Drop rows where timestamp cannot be parsed.
            "max_sma_km": MAX_ANALYSIS_SMA_KM,  # Filter out high-altitude records before all analytics/plots.
            "required_columns": ["sat_id", "timestamp", "ecc", "inc", "sma", "true_anomaly"],  # Hard-required columns post-ingest.
        },
        "deduplication": {
            "enabled": False,  # Enable deterministic duplicate removal.
            "subset": ["norad_cat_id", "timestamp"],  # De-duplication key columns.
            "keep": "last",  # Keep policy: first|last.
        },
        "feature_derivation": {
            "ecc_threshold": LOW_ECCENTRICITY_THRESHOLD,  # Low-e threshold for phase-safe feature logic.
            "include_radians": True,  # Include radian angle companion columns.
            "include_unwrapped": True,  # Include unwrapped angle/time-series columns.
        },
        "synchronization": {
            "enabled": False,  # Enable common-epoch synchronization utilities.
            "mode": "auto",  # Sync strategy.
            "sat_ids": None,  # Satellite IDs to synchronize.
            "target_time": None,  # Optional target epoch/time anchor.
            "tolerance": DEFAULT_SYNC_TOLERANCE,  # Temporal tolerance window.
            "object_col": "sat_id",  # Object ID column for sync grouping.
            "time_col": "timestamp",  # Timestamp column for sync logic.
        },
        "analysis_dispatch": {
            "apply_plot_style": True,  # Apply shared plotting style before analysis runs.
            "show_plots": False,  # Render/show interactive figures.
            "save_plots": False,  # Save generated figures to disk.
            "plot_output_dir": "plots",  # Output root for saved plots (relative paths resolve from this module directory).
            "plot_save_dpi": 600,  # Export resolution for saved plots.
            "export_plot_data": False,  # Capture per-figure data + manifest.json for offline replotting.
            "plot_data_output_dir": "plot_data",  # Output root for captured plot data (run subdir auto-appended).
            "plot_data_compress_threshold_kb": 256,  # CSVs larger than this are recompressed with zstandard if available.
            "return_results": True,  # Return structured analysis payload.

            "print_summary": False,  # Print panel/object summary tables.
            "altitude_bins": None,  # Optional altitude bins for regime/shell analytics.
            "shell_definitions": None,  # Optional shell definitions for labeling/metrics.
            "starlink_shell_profile": "gen1",  # Default profile for historical Gen1 Starlink workflows.
            "shell_refine_with_inclination": False,  # Optional altitude+inclination refinement for shell labels.
            "inclination_time_render_mode": "scatter",  # Inclination time-series rendering: scatter|hexbin_time|hist2d_time.
            "inclination_reference_lines": True,  # Draw target/mean/median reference lines on inclination time-series views.
            "inclination_reference_assignment_tolerance_deg": 0.4,  # Nearest-target assignment tolerance for inclination summaries.
            "inclination_reference_annotation": False,  # Optional compact per-target stats annotation on inclination figures.
            "precession_group_targets_deg": [53.0, 70.0, 97.6],  # Inclination-family targets for grouped J2 diagnostics.
            "precession_group_assignment_tolerance_deg": 0.5,  # Nearest-target tolerance for J2 grouping.
            "precession_group_trend_mode": "none",  # Group trend mode: none|ols|huber.
            "precession_group_rolling_window_days": 30,  # Rolling summary window for grouped precession diagnostics.
            "precession_show_group_envelopes": True,  # Show grouped rolling envelope bands for precession rates.
            "precession_apsidal_ecc_floor": 1.0e-3,  # Low-e threshold for apsidal precession confidence.
            "precession_low_e_behavior": "suppress",  # suppress|highlight low-confidence apsidal points.
            "raan_time_render_mode": "scatter",  # RAAN time-series rendering: scatter|hexbin_time|hist2d_time.
            "raan_display_mode": "wrapped_scatter",  # wrapped_scatter|wrapped_density|unwrapped_by_object|residual_by_object.
            "raan_residual_fit_mode": "huber",  # Residual trend fit mode: ols|huber.
            "raan_residual_annotation": False,  # Optional compact residual annotation overlay.
            "raan_min_points_for_unwrap": 3,  # Minimum points per object for RAAN unwrapping/residualization.
            "sma_time_render_mode": "scatter",  # SMA time-series rendering: scatter|hexbin_time|hist2d_time.
            "sma_reference_lines": True,  # Draw SMA operating-level reference lines.
            "sma_reference_values_km": [6918.137, 6928.137, 6938.137, 6948.137],  # Gen1 operating-level SMA references.
            "sma_rolling_window_days": 30,  # Rolling SMA summary window in days.
            "sma_show_envelope": True,  # Show rolling SMA spread envelope.
            "sma_operational_bands": None,  # Optional generic SMA altitude bands (not shell overlays).
            "argp_time_render_mode": "scatter",  # Argument-of-perigee time-series rendering mode.
            "argp_ecc_floor": 1.0e-3,  # Low-e threshold for argument-of-perigee validity.
            "argp_low_e_behavior": "suppress",  # suppress|highlight|split low-e argp handling.
            "argp_show_argument_of_latitude_companion": False,  # Overlay argument-of-latitude companion diagnostic.
            "eccentricity_time_render_mode": "scatter",  # Eccentricity time-series rendering mode.
            "eccentricity_highlight_descents": True,  # Highlight negative-slope eccentricity transition segments.
            "eccentricity_descent_threshold_per_day": -5.0e-6,  # Looser threshold for eccentricity descent highlighting.
            "eccentricity_descent_min_duration_days": 7.0,  # Minimum duration for a highlighted descent segment.
            "eccentricity_descent_min_eccentricity": 1.8e-3,  # Ignore tiny-e descent noise below this level.
            "eccentricity_rolling_window_days": 30,  # Rolling eccentricity summary window.
            "eccentricity_show_envelope": True,  # Show eccentricity rolling envelope.
            "eccentricity_zoom_ylim": None,  # Optional [ymin, ymax] zoomed eccentricity panel bounds.
            "inc_sma_render_mode": "scatter",  # Inc-vs-SMA rendering: scatter|hexbin|hist2d.
            "inc_sma_metric_mode": "standardized_euclidean",  # Inc-vs-SMA metric mode.
            "inc_sma_reference_lines": True,  # Draw target reference lines for Inc-vs-SMA.
            "inc_sma_reference_annotation": False,  # Optional compact per-target annotation in Inc-vs-SMA figures.
            "inc_sma_reference_markers": True,  # Draw target operating-point markers for Inc-vs-SMA.
            "inc_sma_target_sma_tolerance_km": 25.0,  # Target assignment tolerance in semi-major-axis dimension.
            "inc_sma_target_inclination_tolerance_deg": 0.4,  # Target assignment tolerance in inclination dimension.
            "inc_sma_target_profiles": None,  # Optional explicit Inc-vs-SMA target-profile override.
            "inc_sma_focused_profiles": True,  # Plot focused 53/70/97 figure pairs by default.
            "enable_sma_inclination_density_shell_map": False,  # Disable geometric SMA-vs-inc shell map panel.
            "enable_common_epoch_shell_snapshot": False,  # Disable geometric common-epoch shell snapshot panel.
            "enable_starlink_dynamical_atlas": False,  # Disable Starlink-oriented dynamical atlas panel.

            "run_maneuver_analysis": False,  # Enable maneuver/event detection workflow.
            "maneuver_config": None,  # Optional ManeuverDetectionConfig override.
            "phase_config": None,  # Optional PhaseClassificationConfig override.
            "use_cached_maneuver_phase": True,  # Reuse cached mission-phase CSVs when available.
            "cached_phase_labels_csv": DEFAULT_CACHED_PHASE_LABELS_CSV,  # Cached per-epoch mission-phase labels.
            "cached_phase_intervals_csv": DEFAULT_CACHED_PHASE_INTERVALS_CSV,  # Cached mission-phase intervals.
            "cached_maneuver_events_csv": None,  # Optional cached maneuver/event table when exported separately.
            "plot_maneuver_layers": False,  # Plot maneuver overlays/panels when analysis enabled.
            "maneuver_event_layer_mode": "accepted_high_confidence",  # accepted_high_confidence|accepted_only|all|raw_only overlays.
            "maneuver_event_high_confidence_threshold": 0.8,  # Threshold for high-confidence event filtering.
            "maneuver_event_color_mode": "event_type",  # event_type|detector_support_count|event_score coloring.
            "maneuver_show_event_uncertainty": True,  # Shade estimated event-time uncertainty windows.
            "maneuver_event_histogram_basis": "accepted",  # accepted|high_confidence|raw_segments basis for counts.
            "maneuver_phase_use_interval_overlay": True,  # Shade phase intervals behind phase-colored altitude scatter.
            "maneuver_phase_alpha_by_confidence": True,  # Scale phase scatter alpha by inferred confidence.
            "maneuver_timeline_sort_mode": "first_epoch",  # first_epoch|object_id|disposal_onset|launch_epoch.
            "maneuver_timeline_confidence_shading": False,  # Modulate phase timeline alpha using interval confidence.
            "maneuver_timeline_zoom_inset": True,  # Add square inset for the first launch group on the phase timeline.
            "maneuver_timeline_zoom_start": None,  # None/launch starts inset at the first launch epoch.
            "maneuver_timeline_zoom_end": "2025-12-31",  # End of first-launch inset window; accepts datetime-like text.
            "maneuver_timeline_zoom_group_window_days": 14.0,  # Objects launched within this many days are included in the inset.
            "maneuver_timeline_zoom_size_inches": 2.35,  # Physical inset size; width and height are kept equal.

            "spectral_mode": "uniform_fft",  # Fixed spectral method.
            "spectral_cadence_seconds": None,  # Optional cadence override for resampling.
            "spectral_selected_satellites": None,  # Optional satellite subset for spectral runs.
            "plot_stacked_periodogram": False,  # Convenience toggle for median spectral stack.
            "run_fft": False,  # Gate FFT stage for full-history safety.
            "run_wavelet": False,  # Gate wavelet stage for full-history safety.
            "run_crosscorr": False,  # Gate cross-correlation stage for full-history safety.
            "run_relative_motion": False,  # Gate relative-motion stage for full-history safety.

            "fft_backend": "auto",  # FFT backend: auto|numpy|cupy|torch.
            "fft_gpu_min_samples": 4096,  # Minimum sample count before GPU auto-selection.
            "fft_interpolation": "linear",  # Uniform-grid interpolation method.
            "fft_stack_mode": None,  # Explicit stack mode override; None follows convenience toggle.
            "fft_normalize_before_stack": True,  # Normalize each spectrum before stacking.
            "fft_show_period_axis": True,  # Show period axis companion view in FFT plots.
            "fft_extract_peaks": False,  # Extract dominant spectral peaks.
            "fft_peak_top_k": 5,  # Peak count when extraction is enabled.
            "fft_peak_min_prominence": None,  # Optional peak prominence threshold.
            "fft_peak_min_distance_bins": 1,  # Minimum bin spacing between peaks.
            "fft_peak_overlay": False,  # Overlay detected peaks on spectral plots.
            "fft_bootstrap_replicates": 0,  # Bootstrap replicates for stacked confidence bands.
            "fft_bootstrap_seed": 0,  # RNG seed for spectral bootstrap.

            "wavelet_method": "wwz",  # Fixed wavelet method.
            "wavelet_irregular_policy": "resample",  # Fixed irregular cadence handling.
            "wavelet_irregular_warning_mode": "once",  # Warning policy: once|always|never.
            "wwz_use_gpu": True,  # Prefer GPU WWZ when CUDA is available.
            "wwz_freq_batch_size": 16,  # WWZ frequency batch size for GPU memory/perf tuning.
            "wwz_extract_peaks": True,  # Export dominant WWZ local maxima from combined maps.
            "wwz_peak_top_k": 5,  # Number of local WWZ maxima to export per element.
            "wwz_peak_min_prominence": None,  # Optional minimum local prominence threshold.
            "wwz_peak_min_separation_tau_bins": 2,  # Tau-bin separation for WWZ peak deduplication.
            "wwz_peak_min_separation_period_bins": 2,  # Period-bin separation for WWZ peak deduplication.
            "wwz_overlay_ridge": False,  # Optional thin ridge overlay on combined WWZ panels.
            "wwz_annotate_panels": False,  # Optional compact dominant-feature text on WWZ panels.
            "wwz_min_effective_n": None,  # Optional low-support mask threshold from WWZ effective_n.
            "wwz_export_combined_summary": True,  # Return combined per-element WWZ summary/peak/ridge tables.

            "crosscorr_preprocessing": "zscored",  # Series preprocessing before cross-correlation.
            "plot_crosscorr_heatmap": False,  # Plot pairwise cross-correlation heatmap.
            "crosscorr_min_overlap": 32,  # Minimum overlap sample count for pair analysis.
            "crosscorr_max_grid_points": None,  # Optional override for resampled pair grid size.
            "crosscorr_max_plot_points": None,  # Optional decimation target for plotting speed.
            "crosscorr_include_frequency_products": False,  # Include CSD/coherence products.
            "crosscorr_include_cross_wavelet": False,  # Include cross-wavelet/coherence payload.
            "crosscorr_freq_nperseg": 256,  # Segment length for frequency-domain coupling metrics.
            "crosscorr_freq_noverlap": None,  # Overlap length for coupling metrics.
            "crosscorr_normalization": "legacy",  # Cross-correlation normalization: legacy|overlap_aware.
            "crosscorr_interpolation": "linear",  # Uniform-grid interpolation for pairwise coupling products.

            "run_resonance_diagnostics": False,  # Enable resonance diagnostics stage.
            "resonance_definitions": None,  # Optional custom resonance definition list.
            "resonance_tolerance_rad_day": 1.0e-3,  # Proximity tolerance in rad/day.

            "run_shell_analytics": False,  # Enable shell occupancy/transition analytics.
            "run_disposal_metrics": False,  # Enable disposal corridor metrics.
            "run_sustainability_metrics": False,  # Enable sustainability summary metrics.
            "run_risk_screening": False,  # Enable risk proxy screening metrics.
            "analytics_time_freq": "7D",  # Time bin frequency for analytics rollups.
            "plot_new_analytics": False,  # Plot optional newer analytics figures.
            "sustainability_count_basis": "objects",  # records|objects|both selection for sustainability timelines.
            "risk_timeline_mode": "crossing_and_score",  # crossing_and_score|components|score_only plot mode.
            "disposal_family_col": "candidate_shell_id",  # Optional family/group label source for disposal grouping.
            "disposal_cohort_col": None,  # Optional cohort label column for disposal onset grouping.
            "disposal_include_age_at_onset": True,  # Compute disposal onset age using launch/reference epoch when available.
            "disposal_onset_group_by": None,  # Optional disposal onset grouping key (e.g., shell_family/cohort_label).
            "disposal_onset_candidate_only": False,  # Restrict onset timeline to passive-decay candidates.
            "disposal_onset_include_age_stats": True,  # Include mean/median onset age statistics in onset timeline output.
            "occupancy_heatmap_time_axis_mode": "datetime",  # Occupancy heatmap time axis mode.
            "occupancy_heatmap_y_label_mode": "bin_labels",  # Occupancy heatmap y-axis labeling mode.
            "occupancy_heatmap_norm": "linear",  # Occupancy heatmap color normalization mode.
            "occupancy_heatmap_clip_percentile": None,  # Optional color clipping percentile.
            "occupancy_heatmap_overlay_altitude_refs_km": None,  # Optional generic altitude reference lines.
            "occupancy_heatmap_use_pcolormesh": True,  # Prefer pcolormesh for physical axis labeling.
            "occupancy_heatmap_smoothing_sigma": None,  # Optional smoothing sigma (off by default).
            "raan_phase_density_mode": "hist2d",  # RAAN-vs-phase density rendering mode.
            "raan_phase_density_family_mode": "aggregate",  # aggregate|per_family|both family diagnostics.
            "raan_phase_density_family_targets_deg": [53.05, 53.217, 70.0, 97.655],  # Inclination-family targets.
            "raan_phase_density_family_tolerance_deg": 0.4,  # Inclination-family assignment tolerance.
            "raan_phase_density_time_windows": None,  # Optional explicit time windows for RAAN-phase density.
            "raan_phase_density_rolling_window_days": None,  # Optional rolling window width in days.
            "raan_phase_density_return_arrays": True,  # Return density arrays and diagnostics payloads.
            "raan_phase_density_compute_uniformity_metrics": True,  # Compute descriptive residual-to-uniform diagnostics.
            "phase_alt_density_mode": "circular_linear_kde",  # Phase-vs-altitude density mode.
            "phase_alt_density_family_mode": "aggregate",  # aggregate|per_family|both family diagnostics.
            "phase_alt_density_family_targets_deg": [53.05, 53.217, 70.0, 97.655],  # Inclination-family targets.
            "phase_alt_density_family_tolerance_deg": 0.4,  # Inclination-family assignment tolerance.
            "phase_alt_density_time_windows": None,  # Optional explicit time windows for phase-altitude density.
            "phase_alt_density_rolling_window_days": None,  # Optional rolling window width in days.
            "phase_alt_density_step_days": None,  # Optional rolling step size in days.
            "phase_alt_density_top_k_hotspots": 5,  # Top-K hotspots per panel/window.
            "phase_alt_density_return_arrays": True,  # Return panel density arrays for downstream analysis.
            "phase_alt_density_normalization": "per_panel",  # raw|per_panel|per_family density scaling.
            "phase_alt_density_overlay_altitude_refs_km": None,  # Optional generic altitude reference lines.
            "include_risk_composite_score": False,  # Backward-compatible alias for include_proxy_risk_score.
            "include_proxy_risk_score": False,  # Include heuristic proxy risk score (not conjunction-grade truth).
            "compliance_horizons_years": [5, 25],  # Disposal framing horizons in years.

            # TLEs are mean elements; SGP4-compatible reconstruction is safer for relative-state analysis.
            "state_model": "sgp4_preferred",  # State reconstruction policy: classical|sgp4_preferred|sgp4_required.
            "relative_model": "exact_lvlh",  # Relative dynamics model family: exact_lvlh|sgp4|keplerian.
            "relative_n_periods": 10,  # Relative trajectory horizon in orbital periods.
            "relative_samples_per_period": 100,  # Temporal resolution for relative trajectories.
            "relative_max_duration_seconds": None,  # Optional hard cap on relative propagation duration.
            "relative_tolerance_seconds": 300,  # Epoch matching tolerance for relative pairing.
            "relative_pair_list": None,  # Optional explicit object pair list.
            "relative_pair_mode": "explicit_only",  # explicit_only: do not auto-generate pairs.
            "relative_max_pairs": None,  # Optional cap when pair generation modes are introduced.
            "relative_same_shell_only": False,  # Optional shell-based pair filtering control.
        },
        "export": {
            "enabled": False,  # Enable CSV export of enriched panel.
            "output_csv": None,  # Output CSV path when export is enabled.
            "phase_intervals_csv": None,  # Optional mission-phase interval CSV export path.
            "phase_labels_csv": None,  # Optional per-epoch mission-phase label CSV export path.
            "phase_filter_mode": "maneuver_relevant_only",  # all|maneuver_relevant_only export phase filtering.
            "phase_column_profile": "core",  # core|full exported phase column profile.
            "maneuver_relevant_states": list(DEFAULT_MANEUVER_RELEVANT_PHASE_STATES),  # Curated phase states retained in the mission-phase CSV exports.
            "include_only_objects_with_maneuver_labels": True,  # Keep only satellites with exported phase labels.
            "export_rejected_rows": False,  # Optionally export QC-rejected rows.
            "rejected_rows_csv": None,  # Optional explicit rejected-row CSV path.
            "provenance_sidecar_json": None,  # Optional JSON sidecar for reproducibility provenance.
        },
    }


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _normalize_state_model_compat(state_model_value: str) -> str:
    try:
        normalized = normalize_state_model(state_model_value or "classical")
        if state_model_value and state_model_value != normalized:
            print(f"Interpreting '{state_model_value}' as '{normalized}'.")
        return normalized
    except ValueError:
        print(f"Invalid state model; expected one of {SUPPORTED_STATE_MODELS}. Defaulting to classical.")
        return "classical"


def _resolve_generation_only_files(
    generation_selection: str,
    generation_split_root: str = DEFAULT_GENERATION_SPLIT_PATH,
) -> Optional[list[str]]:
    selection = (generation_selection or "all").strip().lower()
    if selection in {"", "all"}:
        return None

    if selection not in {"gen1", "gen2"}:
        print(f"Unknown generation selection '{generation_selection}'. Falling back to all.")
        return None

    generation_dir = os.path.join(generation_split_root, selection)
    if not os.path.isdir(generation_dir):
        print(f"Generation directory not found: {generation_dir}. Falling back to all.")
        return None

    allowed = sorted(
        {
            name
            for name in os.listdir(generation_dir)
            if name.lower().endswith(".txt")
        }
    )
    if not allowed:
        print(f"No .txt files found in {generation_dir}. Falling back to all.")
        return None

    return allowed


def _run_ingest_stage(config: Dict[str, Any]) -> Dict[str, Any]:
    folder_paths = config["folder_paths"]
    ingest_cfg = config.get("ingest") or {}

    env_validate_checksum = ingest_cfg.get("validate_checksum")
    old_validate_checksum = os.environ.get("TLE_VALIDATE_CHECKSUM")
    env_override_active = env_validate_checksum is not None
    if env_override_active:
        os.environ["TLE_VALIDATE_CHECKSUM"] = "1" if bool(env_validate_checksum) else "0"

    total_files = 0
    available_txt_files = []
    for folder in folder_paths:
        if os.path.isdir(folder):
            folder_txt_files = [name for name in os.listdir(folder) if name.lower().endswith(".txt")]
            total_files += len(folder_txt_files)
            available_txt_files.extend(folder_txt_files)

    only_files = config.get("only_files")
    if only_files is not None:
        only_files = sorted({str(name) for name in only_files if str(name).lower().endswith(".txt")})

    tle_file_limit = config.get("tle_file_limit")
    selected_file_count = None
    if tle_file_limit is not None:
        try:
            tle_file_limit_int = int(tle_file_limit)
        except (TypeError, ValueError):
            tle_file_limit_int = None

        if tle_file_limit_int is not None and tle_file_limit_int > 0:
            if only_files is None:
                candidate_files = sorted({name for name in available_txt_files})
            else:
                available_set = {name for name in available_txt_files}
                candidate_files = [name for name in only_files if name in available_set]

            only_files = candidate_files[:tle_file_limit_int]

    if only_files is not None:
        selected_file_count = len(only_files)

    print(f"Total Files Processed: {total_files}")

    try:
        df, filenames = load_all_tle_data(
            folder_paths,
            only_files=only_files,
            derived=config.get("derived"),
        )
    finally:
        if env_override_active:
            if old_validate_checksum is None:
                os.environ.pop("TLE_VALIDATE_CHECKSUM", None)
            else:
                os.environ["TLE_VALIDATE_CHECKSUM"] = old_validate_checksum

    print(f"Total TLE Records Loaded: {len(df)}")

    return {
        "panel": df,
        "filenames_array": filenames,
        "fileNames": np.unique(filenames).tolist(),
        "metadata": {
            "total_files_processed": int(total_files),
            "selected_file_count": selected_file_count,
            "total_records_loaded": int(len(df)),
            "validate_checksum": None if env_validate_checksum is None else bool(env_validate_checksum),
        },
    }


def _run_quality_control_stage(df: pd.DataFrame, qc_cfg: Dict[str, Any]) -> Dict[str, Any]:
    work = df.copy()

    required_columns = qc_cfg.get("required_columns") or []
    missing_required = [c for c in required_columns if c not in work.columns]
    if missing_required:
        available_preview = sorted(work.columns.tolist())
        raise KeyError(
            "Missing required columns after ingest: "
            f"{missing_required}. Required={required_columns}. "
            f"Available columns sample={available_preview[:25]} (total={len(available_preview)})."
        )

    numeric_cols = ["inc", "raan", "ecc", "aop", "mean_anomaly", "mean_motion", "sma", "true_anomaly"]
    coerced_cols = []
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            coerced_cols.append(col)

    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")

    max_sma_km = qc_cfg.get("max_sma_km", MAX_ANALYSIS_SMA_KM)
    try:
        max_sma_km = float(max_sma_km)
    except (TypeError, ValueError):
        max_sma_km = MAX_ANALYSIS_SMA_KM

    reason_masks: Dict[str, np.ndarray] = {}
    n_rows = len(work)

    if qc_cfg.get("drop_invalid_timestamp", True):
        reason_masks["invalid_timestamp"] = ~work["timestamp"].notna().to_numpy(dtype=bool)

    def _col_values(name: str) -> np.ndarray:
        if name not in work.columns:
            return np.full(n_rows, np.nan, dtype=np.float64)
        return pd.to_numeric(work[name], errors="coerce").to_numpy(dtype=np.float64)

    ecc = _col_values("ecc")
    inc = _col_values("inc")
    mean_motion = _col_values("mean_motion")
    sma = _col_values("sma")
    raan = _col_values("raan")
    aop = _col_values("aop")
    mean_anomaly = _col_values("mean_anomaly")
    true_anomaly = _col_values("true_anomaly")

    if "ecc" in work.columns:
        reason_masks["invalid_ecc_bounds"] = (~np.isfinite(ecc)) | (ecc < 0.0) | (ecc >= 1.0)
    if "inc" in work.columns:
        reason_masks["invalid_inc_bounds"] = (~np.isfinite(inc)) | (inc < 0.0) | (inc > 180.0)
    if "mean_motion" in work.columns:
        reason_masks["invalid_mean_motion"] = (~np.isfinite(mean_motion)) | (mean_motion <= 0.0)
    if "sma" in work.columns:
        reason_masks["invalid_sma"] = (~np.isfinite(sma)) | (sma <= 0.0)
        reason_masks["sma_above_max"] = np.isfinite(sma) & (sma > float(max_sma_km))

    for angle_name, angle_values in (
        ("raan", raan),
        ("aop", aop),
        ("mean_anomaly", mean_anomaly),
        ("true_anomaly", true_anomaly),
    ):
        if angle_name in work.columns:
            reason_masks[f"nonfinite_{angle_name}"] = ~np.isfinite(angle_values)

    rejection_mask = np.zeros(n_rows, dtype=bool)
    for mask in reason_masks.values():
        rejection_mask |= np.asarray(mask, dtype=bool)

    rejected_rows = work.loc[rejection_mask].copy()
    if not rejected_rows.empty:
        reason_names = list(reason_masks.keys())
        mask_matrix = np.vstack([np.asarray(reason_masks[name], dtype=bool) for name in reason_names])
        rejected_idx = np.flatnonzero(rejection_mask)
        rejected_reasons = []
        for ridx in rejected_idx:
            row_reasons = [name for j, name in enumerate(reason_names) if bool(mask_matrix[j, ridx])]
            rejected_reasons.append(";".join(row_reasons))
        rejected_rows["rejection_reasons"] = rejected_reasons

    work = work.loc[~rejection_mask].reset_index(drop=True)
    removed = int(np.sum(rejection_mask))
    removed_sma = int(np.sum(reason_masks.get("sma_above_max", np.zeros(n_rows, dtype=bool))))

    rows_removed_by_cause = {
        cause: int(np.sum(mask))
        for cause, mask in reason_masks.items()
        if int(np.sum(mask)) > 0
    }

    sma_min_before = float(np.nanmin(sma)) if np.isfinite(sma).any() else np.nan
    sma_max_before = float(np.nanmax(sma)) if np.isfinite(sma).any() else np.nan

    return {
        "panel": work,
        "rejected_rows": rejected_rows,
        "metadata": {
            "rows_removed": int(removed),
            "rows_removed_sma": int(removed_sma),
            "rows_removed_by_cause": rows_removed_by_cause,
            "sma_max_km_applied": float(max_sma_km),
            "sma_min_before_filter": sma_min_before,
            "sma_max_before_filter": sma_max_before,
            "missing_required_columns": missing_required,
            "numeric_columns_coerced": coerced_cols,
        },
    }


def _run_dedup_stage(df: pd.DataFrame, dedup_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not dedup_cfg.get("enabled", True):
        return {"panel": df.copy(), "metadata": {"enabled": False, "rows_dropped": 0}}

    subset = [c for c in dedup_cfg.get("subset", ["norad_cat_id", "timestamp"]) if c in df.columns]
    if not subset:
        return {"panel": df.copy(), "metadata": {"enabled": True, "rows_dropped": 0, "note": "no_valid_subset_columns"}}

    keep = "last" if str(dedup_cfg.get("keep", "last")).lower() == "last" else "first"
    work = df.copy().sort_values(subset, kind="mergesort")
    before = len(work)
    work = work.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

    return {
        "panel": work,
        "metadata": {
            "enabled": True,
            "subset": subset,
            "keep": keep,
            "rows_dropped": int(before - len(work)),
        },
    }


def _run_feature_stage(df: pd.DataFrame, feature_cfg: Dict[str, Any]) -> Dict[str, Any]:
    work = df.copy()

    work["node_precession_rate"], work["perigee_precession_rate"] = calculate_precession_rates(
        work["sma"].values,
        work["ecc"].values,
        work["inc"].values,
        J2_EARTH,
        RADIUS_EARTH,
    )

    work = add_standard_tle_proxy_enrichment(
        work,
        ecc_threshold=float(feature_cfg.get("ecc_threshold", LOW_ECCENTRICITY_THRESHOLD)),
        include_radians=bool(feature_cfg.get("include_radians", True)),
        include_unwrapped=bool(feature_cfg.get("include_unwrapped", True)),
        # Keep pipeline phase semantics pinned to true_anomaly for backward-compatible
        # downstream schemas, even in low-e rows where geometric singularities are known.
        requested_phase_variable=PHASE_VARIABLE_TRUE_ANOMALY,
        low_e_choice=PHASE_VARIABLE_TRUE_ANOMALY,
    )

    return {
        "panel": work,
        "metadata": {
            "features_added": [
                "node_precession_rate",
                "perigee_precession_rate",
                "low_eccentricity",
                "argument_of_latitude_deg",
                "mean_longitude_deg",
                "longitude_of_perigee_deg",
                "mean_argument_of_latitude_deg",
                "selected_phase_deg",
                "recommended_phase_variable",
                "true_anomaly_kepler_proxy_deg",
                "phase_variable",
                "phase_semantics",
            ],
        },
    }


def _run_sync_stage(df: pd.DataFrame, sync_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not sync_cfg.get("enabled", False):
        return {
            "records": pd.DataFrame(columns=df.columns),
            "common_epoch": None,
            "max_abs_delta_seconds": None,
            "metadata": {"enabled": False},
        }

    sat_ids = sync_cfg.get("sat_ids")
    if not sat_ids:
        return {
            "records": pd.DataFrame(columns=df.columns),
            "common_epoch": None,
            "max_abs_delta_seconds": None,
            "metadata": {"enabled": True, "status": "no_sat_ids"},
        }

    records, common_epoch, max_abs_delta_seconds, metadata = find_common_epoch_records(
        df,
        sat_ids=sat_ids,
        target_time=sync_cfg.get("target_time"),
        tolerance=sync_cfg.get("tolerance", DEFAULT_SYNC_TOLERANCE),
        object_col=sync_cfg.get("object_col", "sat_id"),
        time_col=sync_cfg.get("time_col", "timestamp"),
        mode=sync_cfg.get("mode", "auto"),
        return_metadata=True,
    )

    return {
        "records": records,
        "common_epoch": common_epoch,
        "max_abs_delta_seconds": max_abs_delta_seconds,
        "metadata": metadata,
    }


def _resolve_plot_output_dir(plot_output_dir: Optional[str]) -> str:
    base_dir = str(plot_output_dir).strip() if plot_output_dir is not None else ""
    if not base_dir:
        base_dir = "plots"
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.path.dirname(__file__), base_dir)
    run_label = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return os.path.join(base_dir, run_label)


def _sanitize_plot_stem(text: str, fallback: str) -> str:
    cleaned = re.sub(r"\s+", "_", str(text or "").strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _figure_display_name(fig, fallback: str) -> str:
    label = str(fig.get_label() or "").strip()
    if label:
        return label
    for ax in fig.get_axes():
        title = str(ax.get_title() or "").strip()
        if title:
            return title
    return fallback


def _is_probable_control_axes(axes) -> bool:
    try:
        if axes.get_title() or axes.get_xlabel() or axes.get_ylabel():
            return False
        position = axes.get_position()
        if position.height > 0.08 or position.width < 0.25 or position.y0 > 0.16:
            return False
        if len(axes.collections) > 0 or len(axes.images) > 0 or len(axes.containers) > 0:
            return False
        return True
    except Exception:
        return False


def _axes_title_artists(axes) -> list:
    return [
        getattr(axes, "_left_title", None),
        getattr(axes, "title", None),
        getattr(axes, "_right_title", None),
    ]


def _prepare_figure_for_saved_export(fig):
    restorers = []

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        text = str(suptitle.get_text() or "")
        visible = bool(suptitle.get_visible())
        if text or visible:
            suptitle.set_visible(False)

            def restore_suptitle(title_artist=suptitle, was_visible=visible):
                title_artist.set_visible(was_visible)

            restorers.append(restore_suptitle)

    for axes in fig.get_axes():
        if _is_probable_control_axes(axes):
            visible = bool(axes.get_visible())
            axes.set_visible(False)

            def restore_axes(target_axes=axes, was_visible=visible):
                target_axes.set_visible(was_visible)

            restorers.append(restore_axes)

        for title_artist in _axes_title_artists(axes):
            if title_artist is None:
                continue
            title_text = str(title_artist.get_text() or "")
            title_visible = bool(title_artist.get_visible())
            if not title_text and not title_visible:
                continue
            title_artist.set_text("")
            title_artist.set_visible(False)

            def restore_title(target_artist=title_artist, text=title_text, visible=title_visible):
                target_artist.set_text(text)
                target_artist.set_visible(visible)

            restorers.append(restore_title)

    def restore():
        for restore_item in reversed(restorers):
            restore_item()

    return restore


def _save_open_matplotlib_figures(
    output_dir: str,
    dpi: int,
    close_after_save: bool = False,
    skip_filenames: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    figure_numbers = sorted(int(n) for n in plt.get_fignums())
    saved_files = []
    skipped_files = []
    errors = []
    used_stems: Dict[str, int] = {}
    skip_set = set(DEFAULT_PLOT_SAVE_SKIP_FILENAMES if skip_filenames is None else skip_filenames)

    for ordinal, figure_number in enumerate(figure_numbers, start=1):
        fig = plt.figure(figure_number)
        title_hint = _figure_display_name(fig, fallback=f"figure_{ordinal:03d}")
        stem = _sanitize_plot_stem(title_hint, fallback=f"figure_{ordinal:03d}")

        used_stems[stem] = used_stems.get(stem, 0) + 1
        stem_count = used_stems[stem]
        if stem_count > 1:
            stem = f"{stem}_{stem_count:02d}"

        file_name = f"{ordinal:03d}_{stem}.png"
        file_path = os.path.join(output_dir, file_name)
        if file_name in skip_set:
            skipped_files.append(file_path)
            continue

        restore_figure = _prepare_figure_for_saved_export(fig)
        try:
            fig.savefig(file_path, dpi=int(dpi), bbox_inches="tight")
            saved_files.append(file_path)
        except Exception as exc:
            errors.append(
                {
                    "figure_number": int(figure_number),
                    "file_path": file_path,
                    "error": str(exc),
                }
            )
        finally:
            restore_figure()
            if close_after_save:
                try:
                    plt.close(fig)
                except Exception:
                    pass

    print(f"[plot_export] Saved {len(saved_files)} figure(s) to {output_dir}")
    if skipped_files:
        print(f"[plot_export] Skipped {len(skipped_files)} configured figure(s)")
    if errors:
        print(f"[plot_export] {len(errors)} figure(s) failed to save")

    return {
        "enabled": True,
        "output_dir": output_dir,
        "dpi": int(dpi),
        "saved_count": int(len(saved_files)),
        "saved_files": saved_files,
        "skipped_count": int(len(skipped_files)),
        "skipped_files": skipped_files,
        "errors": errors,
    }


def _resolve_cached_csv_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        return None
    path_text = str(path_value).strip()
    if not path_text:
        return None
    if os.path.isabs(path_text):
        return path_text
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path_text))


def _load_cached_dataframe(path_value: Optional[str], date_columns: Iterable[str]) -> tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    resolved_path = _resolve_cached_csv_path(path_value)
    info: Dict[str, Any] = {
        "path": resolved_path,
        "loaded": False,
        "rows": 0,
        "reason": None,
    }
    if resolved_path is None:
        info["reason"] = "path_not_configured"
        return None, info
    if not os.path.isfile(resolved_path):
        info["reason"] = "file_missing"
        return None, info

    try:
        df = pd.read_csv(resolved_path)
    except Exception as exc:
        info["reason"] = f"read_failed: {exc}"
        return None, info

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    for column in ("object_id", "sat_id", "norad_cat_id"):
        if column in df.columns:
            df[column] = df[column].where(df[column].isna(), df[column].astype(str))

    info["loaded"] = True
    info["rows"] = int(len(df))
    return df, info


def _load_cached_maneuver_tables(cfg: Dict[str, Any]) -> tuple[Dict[str, Optional[pd.DataFrame]], Dict[str, Any]]:
    summary: Dict[str, Any] = {
        "enabled": bool(cfg.get("use_cached_maneuver_phase", True)),
        "used": False,
        "phase_labels": {},
        "phase_intervals": {},
        "maneuver_events": {},
    }
    tables: Dict[str, Optional[pd.DataFrame]] = {
        "cached_phase_df": None,
        "cached_phase_summary_df": None,
        "cached_maneuver_events_df": None,
    }
    if not summary["enabled"]:
        return tables, summary

    labels_df, labels_info = _load_cached_dataframe(
        cfg.get("cached_phase_labels_csv", DEFAULT_CACHED_PHASE_LABELS_CSV),
        date_columns=("timestamp",),
    )
    intervals_df, intervals_info = _load_cached_dataframe(
        cfg.get("cached_phase_intervals_csv", DEFAULT_CACHED_PHASE_INTERVALS_CSV),
        date_columns=("phase_start", "phase_end"),
    )
    events_df, events_info = _load_cached_dataframe(
        cfg.get("cached_maneuver_events_csv"),
        date_columns=("estimated_event_time", "event_time_lower", "event_time_upper"),
    )

    tables["cached_phase_df"] = labels_df
    tables["cached_phase_summary_df"] = intervals_df
    tables["cached_maneuver_events_df"] = events_df
    summary["phase_labels"] = labels_info
    summary["phase_intervals"] = intervals_info
    summary["maneuver_events"] = events_info
    summary["used"] = any(isinstance(df, pd.DataFrame) for df in tables.values())

    if summary["used"]:
        loaded_parts = [
            name
            for name, info in (
                ("phase labels", labels_info),
                ("phase intervals", intervals_info),
                ("maneuver events", events_info),
            )
            if bool(info.get("loaded"))
        ]
        print(f"[maneuver_cache] using cached {', '.join(loaded_parts)}")
    else:
        print("[maneuver_cache] no cached maneuver/phase CSVs loaded; falling back to computation")

    return tables, summary


def _run_analysis_dispatch_stage(
    panel: pd.DataFrame,
    filenames_array: Iterable[str],
    file_names: Iterable[str],
    dispatch_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = dict(dispatch_cfg)
    cfg["state_model"] = _normalize_state_model_compat(str(cfg.get("state_model", "sgp4_preferred")))

    show_plots_requested = bool(cfg.get("show_plots", False))
    save_plots_enabled = bool(cfg.get("save_plots", False))
    export_plot_data_enabled = bool(cfg.get("export_plot_data", False))
    render_plots = bool(show_plots_requested or save_plots_enabled or export_plot_data_enabled)
    plot_output_dir_cfg = cfg.get("plot_output_dir", "plots")
    plot_data_output_dir_cfg = cfg.get("plot_data_output_dir", "plot_data")
    try:
        plot_data_compress_threshold_bytes = int(
            float(cfg.get("plot_data_compress_threshold_kb", 256)) * 1024
        )
    except (TypeError, ValueError):
        plot_data_compress_threshold_bytes = PLOT_DATA_COMPRESS_THRESHOLD_BYTES
    if plot_data_compress_threshold_bytes <= 0:
        plot_data_compress_threshold_bytes = PLOT_DATA_COMPRESS_THRESHOLD_BYTES
    try:
        plot_save_dpi = int(cfg.get("plot_save_dpi", 600))
    except (TypeError, ValueError):
        plot_save_dpi = 600
    if plot_save_dpi <= 0:
        plot_save_dpi = 600

    plot_save_summary: Dict[str, Any] = {
        "enabled": bool(save_plots_enabled),
        "output_dir": None,
        "dpi": int(plot_save_dpi),
        "saved_count": 0,
        "saved_files": [],
        "errors": [],
    }
    plot_data_export_summary: Dict[str, Any] = {
        "enabled": bool(export_plot_data_enabled),
        "output_dir": None,
        "manifest_path": None,
        "figure_count": 0,
        "errors": [],
        "compression_threshold_bytes": int(plot_data_compress_threshold_bytes),
    }
    maneuver_cache_tables: Dict[str, Optional[pd.DataFrame]] = {
        "cached_phase_df": None,
        "cached_phase_summary_df": None,
        "cached_maneuver_events_df": None,
    }
    maneuver_cache_summary: Dict[str, Any] = {
        "enabled": bool(cfg.get("use_cached_maneuver_phase", True)),
        "used": False,
    }
    maneuver_analysis_requested = bool(cfg.get("run_maneuver_analysis", False))
    maneuver_plot_requested = bool(cfg.get("plot_maneuver_layers", False))
    if maneuver_analysis_requested or maneuver_plot_requested:
        maneuver_cache_tables, maneuver_cache_summary = _load_cached_maneuver_tables(cfg)
    run_maneuver_analysis_effective = bool(
        maneuver_analysis_requested or (maneuver_plot_requested and maneuver_cache_summary.get("used", False))
    )

    # Keep plotting/grouping labels aligned with the post-QC/post-dedup panel rows.
    if "sat_id" in panel.columns:
        filenames_runtime = panel["sat_id"].astype(str).tolist()
    elif "source_filename" in panel.columns:
        filenames_runtime = panel["source_filename"].astype(str).tolist()
    else:
        filenames_runtime = list(filenames_array)

    if len(filenames_runtime) != len(panel):
        print(
            "[analysis_dispatch] filename labels were misaligned after filtering; "
            "using row-index fallback labels."
        )
        filenames_runtime = [f"row_{i}" for i in range(len(panel))]

    file_names_runtime = list(dict.fromkeys(filenames_runtime))
    if not file_names_runtime:
        file_names_runtime = list(file_names)

    # Hard-cleanup policy: fixed analysis methods.
    cfg["spectral_mode"] = "uniform_fft"
    cfg["wavelet_method"] = "wwz"
    cfg["wavelet_irregular_policy"] = "resample"

    # Backward-compatibility alias: retain include_risk_composite_score while preferring include_proxy_risk_score.
    if "include_proxy_risk_score" not in cfg:
        cfg["include_proxy_risk_score"] = bool(cfg.get("include_risk_composite_score", True))

    if cfg.get("apply_plot_style", True):
        apply_plot_style()

    original_show = None
    preserve_fig_env_var = "ORBITAL_PLOT_PRESERVE_FIGURES_FOR_EXPORT"
    preserve_fig_env_old = None
    preserve_fig_env_overridden = False
    if save_plots_enabled or export_plot_data_enabled:
        # Render plotting code paths while suppressing intermediate interactive popups.
        original_show = plt.show
        plt.show = lambda *args, **kwargs: None
        preserve_fig_env_old = os.environ.get(preserve_fig_env_var)
        os.environ[preserve_fig_env_var] = "1"
        preserve_fig_env_overridden = True

    try:
        result = orbital_visualization(
            panel,
            filenames_runtime,
            file_names_runtime,
            ecc_threshold=LOW_ECCENTRICITY_THRESHOLD,
            altitude_bins=cfg.get("altitude_bins"),
            shell_definitions=cfg.get("shell_definitions"),
            starlink_shell_profile=cfg.get("starlink_shell_profile"),
            shell_refine_with_inclination=bool(cfg.get("shell_refine_with_inclination", False)),
            inclination_time_render_mode=cfg.get("inclination_time_render_mode", "scatter"),
            inclination_reference_lines=bool(cfg.get("inclination_reference_lines", True)),
            inclination_reference_assignment_tolerance_deg=float(cfg.get("inclination_reference_assignment_tolerance_deg", 0.4)),
            inclination_reference_annotation=bool(cfg.get("inclination_reference_annotation", False)),
            precession_group_targets_deg=cfg.get("precession_group_targets_deg", [53.0, 70.0, 97.6]),
            precession_group_assignment_tolerance_deg=float(cfg.get("precession_group_assignment_tolerance_deg", 0.5)),
            precession_group_trend_mode=cfg.get("precession_group_trend_mode", "none"),
            precession_group_rolling_window_days=float(cfg.get("precession_group_rolling_window_days", 30)),
            precession_show_group_envelopes=bool(cfg.get("precession_show_group_envelopes", True)),
            precession_apsidal_ecc_floor=float(cfg.get("precession_apsidal_ecc_floor", 1.0e-3)),
            precession_low_e_behavior=cfg.get("precession_low_e_behavior", "suppress"),
            raan_time_render_mode=cfg.get("raan_time_render_mode", "scatter"),
            raan_display_mode=cfg.get("raan_display_mode", "wrapped_scatter"),
            raan_residual_fit_mode=cfg.get("raan_residual_fit_mode", "huber"),
            raan_residual_annotation=bool(cfg.get("raan_residual_annotation", False)),
            raan_min_points_for_unwrap=int(cfg.get("raan_min_points_for_unwrap", 3)),
            sma_time_render_mode=cfg.get("sma_time_render_mode", "scatter"),
            sma_reference_lines=bool(cfg.get("sma_reference_lines", True)),
            sma_reference_values_km=cfg.get("sma_reference_values_km", [6918.137, 6928.137, 6938.137, 6948.137]),
            sma_rolling_window_days=float(cfg.get("sma_rolling_window_days", 30)),
            sma_show_envelope=bool(cfg.get("sma_show_envelope", True)),
            sma_operational_bands=cfg.get("sma_operational_bands"),
            argp_time_render_mode=cfg.get("argp_time_render_mode", "scatter"),
            argp_ecc_floor=float(cfg.get("argp_ecc_floor", 1.0e-3)),
            argp_low_e_behavior=cfg.get("argp_low_e_behavior", "suppress"),
            argp_show_argument_of_latitude_companion=bool(cfg.get("argp_show_argument_of_latitude_companion", False)),
            eccentricity_time_render_mode=cfg.get("eccentricity_time_render_mode", "scatter"),
            eccentricity_highlight_descents=bool(cfg.get("eccentricity_highlight_descents", True)),
            eccentricity_descent_threshold_per_day=float(cfg.get("eccentricity_descent_threshold_per_day", -5.0e-6)),
            eccentricity_descent_min_duration_days=float(cfg.get("eccentricity_descent_min_duration_days", 7.0)),
            eccentricity_descent_min_eccentricity=float(cfg.get("eccentricity_descent_min_eccentricity", 1.8e-3)),
            eccentricity_rolling_window_days=float(cfg.get("eccentricity_rolling_window_days", 30)),
            eccentricity_show_envelope=bool(cfg.get("eccentricity_show_envelope", True)),
            eccentricity_zoom_ylim=cfg.get("eccentricity_zoom_ylim"),
            inc_sma_render_mode=cfg.get("inc_sma_render_mode", "scatter"),
            inc_sma_metric_mode=cfg.get("inc_sma_metric_mode", "standardized_euclidean"),
            inc_sma_reference_lines=bool(cfg.get("inc_sma_reference_lines", True)),
            inc_sma_reference_annotation=bool(cfg.get("inc_sma_reference_annotation", False)),
            inc_sma_reference_markers=bool(cfg.get("inc_sma_reference_markers", True)),
            inc_sma_target_sma_tolerance_km=float(cfg.get("inc_sma_target_sma_tolerance_km", 25.0)),
            inc_sma_target_inclination_tolerance_deg=float(cfg.get("inc_sma_target_inclination_tolerance_deg", 0.4)),
            inc_sma_target_profiles=cfg.get("inc_sma_target_profiles"),
            inc_sma_focused_profiles=bool(cfg.get("inc_sma_focused_profiles", True)),
            enable_sma_inclination_density_shell_map=bool(cfg.get("enable_sma_inclination_density_shell_map", False)),
            enable_common_epoch_shell_snapshot=bool(cfg.get("enable_common_epoch_shell_snapshot", False)),
            enable_starlink_dynamical_atlas=bool(cfg.get("enable_starlink_dynamical_atlas", False)),
            print_summary=bool(cfg.get("print_summary", False)),
            run_maneuver_analysis=run_maneuver_analysis_effective,
            maneuver_config=cfg.get("maneuver_config"),
            phase_config=cfg.get("phase_config"),
            cached_maneuver_events_df=maneuver_cache_tables.get("cached_maneuver_events_df"),
            cached_phase_df=maneuver_cache_tables.get("cached_phase_df"),
            cached_phase_summary_df=maneuver_cache_tables.get("cached_phase_summary_df"),
            maneuver_cache_metadata=maneuver_cache_summary,
            plot_maneuver_layers=bool(cfg.get("plot_maneuver_layers", False)),
            show_plots=bool(render_plots),
            maneuver_event_layer_mode=cfg.get("maneuver_event_layer_mode", "accepted_high_confidence"),
            maneuver_event_high_confidence_threshold=float(cfg.get("maneuver_event_high_confidence_threshold", 0.8)),
            maneuver_event_color_mode=cfg.get("maneuver_event_color_mode", "event_type"),
            maneuver_show_event_uncertainty=bool(cfg.get("maneuver_show_event_uncertainty", True)),
            maneuver_event_histogram_basis=cfg.get("maneuver_event_histogram_basis", "accepted"),
            maneuver_phase_use_interval_overlay=bool(cfg.get("maneuver_phase_use_interval_overlay", True)),
            maneuver_phase_alpha_by_confidence=bool(cfg.get("maneuver_phase_alpha_by_confidence", True)),
            maneuver_timeline_sort_mode=cfg.get("maneuver_timeline_sort_mode", "first_epoch"),
            maneuver_timeline_confidence_shading=bool(cfg.get("maneuver_timeline_confidence_shading", False)),
            maneuver_timeline_zoom_inset=bool(cfg.get("maneuver_timeline_zoom_inset", True)),
            maneuver_timeline_zoom_start=cfg.get("maneuver_timeline_zoom_start"),
            maneuver_timeline_zoom_end=cfg.get("maneuver_timeline_zoom_end", "2025-12-31"),
            maneuver_timeline_zoom_group_window_days=float(cfg.get("maneuver_timeline_zoom_group_window_days", 14.0)),
            maneuver_timeline_zoom_size_inches=float(cfg.get("maneuver_timeline_zoom_size_inches", 2.35)),
            spectral_mode=cfg.get("spectral_mode", "uniform_fft"),
            spectral_cadence_seconds=cfg.get("spectral_cadence_seconds"),
            spectral_selected_satellites=cfg.get("spectral_selected_satellites"),
            plot_stacked_periodogram=bool(cfg.get("plot_stacked_periodogram", False)),
            run_fft=bool(cfg.get("run_fft", False)),
            run_wavelet=bool(cfg.get("run_wavelet", False)),
            run_crosscorr=bool(cfg.get("run_crosscorr", False)),
            run_relative_motion=bool(cfg.get("run_relative_motion", False)),
            fft_backend=cfg.get("fft_backend", "auto"),
            fft_gpu_min_samples=int(cfg.get("fft_gpu_min_samples", 4096)),
            fft_interpolation=cfg.get("fft_interpolation", "linear"),
            fft_stack_mode=cfg.get("fft_stack_mode"),
            fft_normalize_before_stack=bool(cfg.get("fft_normalize_before_stack", True)),
            fft_show_period_axis=bool(cfg.get("fft_show_period_axis", True)),
            fft_extract_peaks=bool(cfg.get("fft_extract_peaks", False)),
            fft_peak_top_k=int(cfg.get("fft_peak_top_k", 5)),
            fft_peak_min_prominence=cfg.get("fft_peak_min_prominence"),
            fft_peak_min_distance_bins=int(cfg.get("fft_peak_min_distance_bins", 1)),
            fft_peak_overlay=bool(cfg.get("fft_peak_overlay", False)),
            fft_bootstrap_replicates=int(cfg.get("fft_bootstrap_replicates", 0)),
            fft_bootstrap_seed=int(cfg.get("fft_bootstrap_seed", 0)),
            wavelet_irregular_policy=cfg.get("wavelet_irregular_policy", "resample"),
            wavelet_method=cfg.get("wavelet_method", "wwz"),
            wavelet_irregular_warning_mode=cfg.get("wavelet_irregular_warning_mode", "once"),
            wwz_use_gpu=bool(cfg.get("wwz_use_gpu", True)),
            wwz_freq_batch_size=int(cfg.get("wwz_freq_batch_size", 16)),
            wwz_extract_peaks=bool(cfg.get("wwz_extract_peaks", True)),
            wwz_peak_top_k=int(cfg.get("wwz_peak_top_k", 5)),
            wwz_peak_min_prominence=cfg.get("wwz_peak_min_prominence"),
            wwz_peak_min_separation_tau_bins=int(cfg.get("wwz_peak_min_separation_tau_bins", 2)),
            wwz_peak_min_separation_period_bins=int(cfg.get("wwz_peak_min_separation_period_bins", 2)),
            wwz_overlay_ridge=bool(cfg.get("wwz_overlay_ridge", False)),
            wwz_annotate_panels=bool(cfg.get("wwz_annotate_panels", False)),
            wwz_min_effective_n=cfg.get("wwz_min_effective_n"),
            wwz_export_combined_summary=bool(cfg.get("wwz_export_combined_summary", True)),
            crosscorr_preprocessing=cfg.get("crosscorr_preprocessing", "zscored"),
            plot_crosscorr_heatmap=bool(cfg.get("plot_crosscorr_heatmap", False)),
            crosscorr_min_overlap=int(cfg.get("crosscorr_min_overlap", 32)),
            crosscorr_max_grid_points=cfg.get("crosscorr_max_grid_points"),
            crosscorr_max_plot_points=cfg.get("crosscorr_max_plot_points"),
            crosscorr_include_frequency_products=bool(cfg.get("crosscorr_include_frequency_products", False)),
            crosscorr_include_cross_wavelet=bool(cfg.get("crosscorr_include_cross_wavelet", False)),
            crosscorr_freq_nperseg=int(cfg.get("crosscorr_freq_nperseg", 256)),
            crosscorr_freq_noverlap=cfg.get("crosscorr_freq_noverlap"),
            crosscorr_normalization=cfg.get("crosscorr_normalization", "legacy"),
            crosscorr_interpolation=cfg.get("crosscorr_interpolation", "linear"),
            run_resonance_diagnostics=bool(cfg.get("run_resonance_diagnostics", False)),
            resonance_definitions=cfg.get("resonance_definitions"),
            resonance_tolerance_rad_day=float(cfg.get("resonance_tolerance_rad_day", 1.0e-3)),
            run_shell_analytics=bool(cfg.get("run_shell_analytics", True)),
            run_disposal_metrics=bool(cfg.get("run_disposal_metrics", True)),
            run_sustainability_metrics=bool(cfg.get("run_sustainability_metrics", True)),
            run_risk_screening=bool(cfg.get("run_risk_screening", True)),
            analytics_time_freq=cfg.get("analytics_time_freq", "7D"),
            plot_new_analytics=bool(cfg.get("plot_new_analytics", False)),
            sustainability_count_basis=cfg.get("sustainability_count_basis", "objects"),
            risk_timeline_mode=cfg.get("risk_timeline_mode", "crossing_and_score"),
            disposal_family_col=cfg.get("disposal_family_col", "candidate_shell_id"),
            disposal_cohort_col=cfg.get("disposal_cohort_col"),
            disposal_include_age_at_onset=bool(cfg.get("disposal_include_age_at_onset", True)),
            disposal_onset_group_by=cfg.get("disposal_onset_group_by"),
            disposal_onset_candidate_only=bool(cfg.get("disposal_onset_candidate_only", False)),
            disposal_onset_include_age_stats=bool(cfg.get("disposal_onset_include_age_stats", True)),
            occupancy_heatmap_time_axis_mode=cfg.get("occupancy_heatmap_time_axis_mode", "datetime"),
            occupancy_heatmap_y_label_mode=cfg.get("occupancy_heatmap_y_label_mode", "bin_labels"),
            occupancy_heatmap_norm=cfg.get("occupancy_heatmap_norm", "linear"),
            occupancy_heatmap_clip_percentile=cfg.get("occupancy_heatmap_clip_percentile"),
            occupancy_heatmap_overlay_altitude_refs_km=cfg.get("occupancy_heatmap_overlay_altitude_refs_km"),
            occupancy_heatmap_use_pcolormesh=bool(cfg.get("occupancy_heatmap_use_pcolormesh", True)),
            occupancy_heatmap_smoothing_sigma=cfg.get("occupancy_heatmap_smoothing_sigma"),
            raan_phase_density_mode=cfg.get("raan_phase_density_mode", "hist2d"),
            raan_phase_density_family_mode=cfg.get("raan_phase_density_family_mode", "aggregate"),
            raan_phase_density_family_targets_deg=cfg.get("raan_phase_density_family_targets_deg", [53.05, 53.217, 70.0, 97.655]),
            raan_phase_density_family_tolerance_deg=float(cfg.get("raan_phase_density_family_tolerance_deg", 0.4)),
            raan_phase_density_time_windows=cfg.get("raan_phase_density_time_windows"),
            raan_phase_density_rolling_window_days=cfg.get("raan_phase_density_rolling_window_days"),
            raan_phase_density_return_arrays=bool(cfg.get("raan_phase_density_return_arrays", True)),
            raan_phase_density_compute_uniformity_metrics=bool(cfg.get("raan_phase_density_compute_uniformity_metrics", True)),
            phase_alt_density_mode=cfg.get("phase_alt_density_mode", "circular_linear_kde"),
            phase_alt_density_family_mode=cfg.get("phase_alt_density_family_mode", "aggregate"),
            phase_alt_density_family_targets_deg=cfg.get("phase_alt_density_family_targets_deg", [53.05, 53.217, 70.0, 97.655]),
            phase_alt_density_family_tolerance_deg=float(cfg.get("phase_alt_density_family_tolerance_deg", 0.4)),
            phase_alt_density_time_windows=cfg.get("phase_alt_density_time_windows"),
            phase_alt_density_rolling_window_days=cfg.get("phase_alt_density_rolling_window_days"),
            phase_alt_density_step_days=cfg.get("phase_alt_density_step_days"),
            phase_alt_density_top_k_hotspots=int(cfg.get("phase_alt_density_top_k_hotspots", 5)),
            phase_alt_density_return_arrays=bool(cfg.get("phase_alt_density_return_arrays", True)),
            phase_alt_density_normalization=cfg.get("phase_alt_density_normalization", "per_panel"),
            phase_alt_density_overlay_altitude_refs_km=cfg.get("phase_alt_density_overlay_altitude_refs_km"),
            include_risk_composite_score=bool(cfg.get("include_risk_composite_score", cfg.get("include_proxy_risk_score", True))),
            include_proxy_risk_score=bool(cfg.get("include_proxy_risk_score", cfg.get("include_risk_composite_score", True))),
            compliance_horizons_years=cfg.get("compliance_horizons_years", [5, 25]),
            state_model=cfg["state_model"],
            relative_model=cfg.get("relative_model", "exact_lvlh"),
            relative_n_periods=int(cfg.get("relative_n_periods", 10)),
            relative_samples_per_period=int(cfg.get("relative_samples_per_period", 100)),
            relative_max_duration_seconds=cfg.get("relative_max_duration_seconds"),
            relative_tolerance_seconds=float(cfg.get("relative_tolerance_seconds", 300)),
            relative_pair_list=cfg.get("relative_pair_list"),
            relative_pair_mode=cfg.get("relative_pair_mode", "explicit_only"),
            relative_max_pairs=cfg.get("relative_max_pairs"),
            relative_same_shell_only=bool(cfg.get("relative_same_shell_only", False)),
            return_results=bool(cfg.get("return_results", True)),
        )
        if save_plots_enabled or export_plot_data_enabled:
            if export_plot_data_enabled:
                data_output_dir = _resolve_plot_output_dir(plot_data_output_dir_cfg)
                try:
                    capture_summary = _capture_open_figures_for_export(
                        data_output_dir,
                        compress_threshold_bytes=plot_data_compress_threshold_bytes,
                        extra_metadata={
                            "state_model": cfg["state_model"],
                            "spectral_mode": cfg.get("spectral_mode"),
                            "wavelet_method": cfg.get("wavelet_method"),
                        },
                    )
                except Exception as exc:
                    capture_summary = {
                        "enabled": True,
                        "output_dir": data_output_dir,
                        "manifest_path": None,
                        "figure_count": 0,
                        "errors": [f"capture failed: {exc}"],
                    }
                plot_data_export_summary.update(capture_summary)
                plot_data_export_summary["compression_threshold_bytes"] = int(
                    plot_data_compress_threshold_bytes
                )

            if save_plots_enabled:
                output_dir = _resolve_plot_output_dir(plot_output_dir_cfg)
                plot_save_summary = _save_open_matplotlib_figures(
                    output_dir,
                    plot_save_dpi,
                    close_after_save=not show_plots_requested,
                )

            if isinstance(result, dict):
                metadata = result.get("metadata")
                if isinstance(metadata, dict):
                    metadata["show_plots"] = bool(show_plots_requested)
                    metadata["plots_rendered_for_saving"] = True
                    metadata["saved_plot_count"] = int(plot_save_summary.get("saved_count", 0))
                    metadata["saved_plot_output_dir"] = plot_save_summary.get("output_dir")
                    metadata["saved_plot_dpi"] = int(plot_save_dpi)
                    metadata["plot_data_export_enabled"] = bool(export_plot_data_enabled)
                    metadata["plot_data_export_dir"] = plot_data_export_summary.get("output_dir")
                    metadata["plot_data_export_figure_count"] = int(
                        plot_data_export_summary.get("figure_count", 0)
                    )

            if show_plots_requested and original_show is not None:
                original_show()
    finally:
        if original_show is not None:
            plt.show = original_show
        if preserve_fig_env_overridden:
            if preserve_fig_env_old is None:
                os.environ.pop(preserve_fig_env_var, None)
            else:
                os.environ[preserve_fig_env_var] = preserve_fig_env_old
        _satrec_from_lines_cached.cache_clear()
        gc.collect()

    return {
        "result": result,
        "metadata": {
            "show_plots": bool(show_plots_requested),
            "plots_rendered": bool(render_plots),
            "save_plots": bool(save_plots_enabled),
            "plot_output_dir": plot_save_summary.get("output_dir"),
            "plot_save_dpi": int(plot_save_dpi),
            "saved_plot_count": int(plot_save_summary.get("saved_count", 0)),
            "plot_save_errors": plot_save_summary.get("errors", []),
            "export_plot_data": bool(export_plot_data_enabled),
            "plot_data_export_dir": plot_data_export_summary.get("output_dir"),
            "plot_data_manifest_path": plot_data_export_summary.get("manifest_path"),
            "plot_data_figure_count": int(plot_data_export_summary.get("figure_count", 0)),
            "plot_data_export_errors": plot_data_export_summary.get("errors", []),
            "maneuver_cache": maneuver_cache_summary,
            "run_maneuver_analysis_effective": bool(run_maneuver_analysis_effective),
            "state_model": cfg["state_model"],
        },
    }


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def _extract_phase_tables(payload: Dict[str, Any]) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    stage = payload.get("stages", {}).get("analysis_dispatch", {})

    phase_df = stage.get("phase_df") if isinstance(stage, dict) else None
    phase_summary_df = stage.get("phase_summary_df") if isinstance(stage, dict) else None

    if not isinstance(phase_df, pd.DataFrame) or not isinstance(phase_summary_df, pd.DataFrame):
        result = stage.get("result") if isinstance(stage, dict) else None
        if isinstance(result, dict):
            analytics = result.get("analytics") if isinstance(result.get("analytics"), dict) else result
            if not isinstance(phase_df, pd.DataFrame):
                candidate = analytics.get("phase_df") if isinstance(analytics, dict) else None
                if isinstance(candidate, pd.DataFrame):
                    phase_df = candidate
            if not isinstance(phase_summary_df, pd.DataFrame):
                candidate = analytics.get("phase_summary_df") if isinstance(analytics, dict) else None
                if isinstance(candidate, pd.DataFrame):
                    phase_summary_df = candidate

    if not isinstance(phase_df, pd.DataFrame):
        phase_df = None
    if not isinstance(phase_summary_df, pd.DataFrame):
        phase_summary_df = None
    return phase_df, phase_summary_df


def _normalize_phase_states(raw_states: Any) -> set[str]:
    if raw_states is None:
        return set(DEFAULT_MANEUVER_RELEVANT_PHASE_STATES)
    if isinstance(raw_states, (str, bytes)):
        raw_states = [raw_states]
    states = {
        str(state).strip()
        for state in raw_states
        if str(state).strip()
    }
    return states or set(DEFAULT_MANEUVER_RELEVANT_PHASE_STATES)


def _apply_phase_filter(df: Optional[pd.DataFrame], mode: str, states: set[str]) -> Optional[pd.DataFrame]:
    if not isinstance(df, pd.DataFrame):
        return None
    out = df.copy()
    if mode == "maneuver_relevant_only" and "phase_state" in out.columns:
        out = out[out["phase_state"].astype(str).isin(states)].copy()
    return out


def _derive_object_scope(
    phase_df: Optional[pd.DataFrame],
    phase_summary_df: Optional[pd.DataFrame],
) -> set[str]:
    if isinstance(phase_summary_df, pd.DataFrame) and "object_id" in phase_summary_df.columns and not phase_summary_df.empty:
        return {str(v) for v in phase_summary_df["object_id"].dropna().astype(str).tolist()}
    if isinstance(phase_df, pd.DataFrame) and "object_id" in phase_df.columns and not phase_df.empty:
        return {str(v) for v in phase_df["object_id"].dropna().astype(str).tolist()}
    return set()


def _apply_object_scope(df: Optional[pd.DataFrame], object_scope: set[str]) -> Optional[pd.DataFrame]:
    if not isinstance(df, pd.DataFrame):
        return None
    if "object_id" not in df.columns:
        return df.copy()
    out = df.copy()
    if not object_scope:
        return out.iloc[0:0].copy()
    return out[out["object_id"].astype(str).isin(object_scope)].copy()


def _with_required_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns]


def _prepare_phase_interval_export_df(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    out = df.copy()
    if "phase_start" in out.columns:
        out["phase_start"] = pd.to_datetime(out["phase_start"], errors="coerce")
    if "phase_end" in out.columns:
        out["phase_end"] = pd.to_datetime(out["phase_end"], errors="coerce")
    if "phase_start" in out.columns and "phase_end" in out.columns:
        duration_hours = (out["phase_end"] - out["phase_start"]).dt.total_seconds() / 3600.0
        out["duration_hours"] = pd.to_numeric(duration_hours, errors="coerce")
    else:
        out["duration_hours"] = np.nan

    core_columns = [
        "object_id",
        "sat_id",
        "norad_cat_id",
        "phase_state",
        "phase_start",
        "phase_end",
        "duration_hours",
        "n_records",
    ]
    if profile == "core":
        return _with_required_columns(out, core_columns)
    front = [col for col in core_columns if col in out.columns]
    trailing = [col for col in out.columns if col not in front]
    return out[front + trailing]


def _prepare_phase_label_export_df(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    core_columns = [
        "object_id",
        "sat_id",
        "norad_cat_id",
        "timestamp",
        "phase_state",
    ]
    if profile == "core":
        return _with_required_columns(out, core_columns)
    front = [col for col in core_columns if col in out.columns]
    trailing = [col for col in out.columns if col not in front]
    return out[front + trailing]


def _serialize_object_columns_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(
                lambda value: json.dumps(sorted(value), ensure_ascii=True, default=str)
                if isinstance(value, set)
                else (
                    json.dumps(value, ensure_ascii=True, default=str)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
            )
    return out


def _default_sidecar_path(output_path: Optional[str]) -> Optional[str]:
    if not output_path:
        return None
    base, _ = os.path.splitext(output_path)
    return f"{base}_provenance.json"


def _run_export_stage(payload: Dict[str, Any], export_cfg: Dict[str, Any]) -> Dict[str, Any]:
    output_csv = export_cfg.get("output_csv")
    phase_intervals_csv = export_cfg.get("phase_intervals_csv")
    phase_labels_csv = export_cfg.get("phase_labels_csv")

    export_requested = bool(
        export_cfg.get("enabled", False)
        or output_csv
        or phase_intervals_csv
        or phase_labels_csv
    )
    if not export_requested:
        return {"enabled": False}

    export_result: Dict[str, Any] = {
        "enabled": True,
        "status": "ok",
        "output_csv": output_csv,
        "phase_intervals_csv": phase_intervals_csv,
        "phase_labels_csv": phase_labels_csv,
    }
    skipped_reasons: list[str] = []
    wrote_csv_artifact = False

    panel = payload["stages"]["feature_derivation"]["panel"]
    if output_csv:
        _ensure_parent_dir(output_csv)
        panel.to_csv(output_csv, index=False)
        export_result["panel_rows_exported"] = int(len(panel))
        wrote_csv_artifact = True
    elif bool(export_cfg.get("enabled", False)):
        skipped_reasons.append("panel_output_csv_not_provided")

    if bool(export_cfg.get("export_rejected_rows", False)):
        qc_stage = payload["stages"].get("quality_control", {})
        rejected = qc_stage.get("rejected_rows")
        if isinstance(rejected, pd.DataFrame):
            rejected_path = export_cfg.get("rejected_rows_csv")
            if not rejected_path:
                fallback = output_csv or phase_intervals_csv or phase_labels_csv
                if fallback:
                    base, ext = os.path.splitext(fallback)
                    rejected_path = f"{base}_rejected{ext or '.csv'}"
            if rejected_path:
                _ensure_parent_dir(rejected_path)
                rejected.to_csv(rejected_path, index=False)
                export_result["rejected_rows_csv"] = rejected_path
                export_result["rejected_rows_count"] = int(len(rejected))
                wrote_csv_artifact = True
            else:
                skipped_reasons.append("rejected_rows_csv_not_provided")

    phase_df_raw, phase_summary_raw = _extract_phase_tables(payload)
    phase_filter_mode = str(export_cfg.get("phase_filter_mode", "maneuver_relevant_only")).strip().lower()
    if phase_filter_mode not in {"all", "maneuver_relevant_only"}:
        phase_filter_mode = "maneuver_relevant_only"

    phase_column_profile = str(export_cfg.get("phase_column_profile", "core")).strip().lower()
    if phase_column_profile not in {"core", "full"}:
        phase_column_profile = "core"

    maneuver_states = _normalize_phase_states(export_cfg.get("maneuver_relevant_states"))
    include_only_objects = bool(export_cfg.get("include_only_objects_with_maneuver_labels", True))

    phase_df_filtered = _apply_phase_filter(phase_df_raw, phase_filter_mode, maneuver_states)
    phase_summary_filtered = _apply_phase_filter(phase_summary_raw, phase_filter_mode, maneuver_states)

    if include_only_objects:
        object_scope = _derive_object_scope(phase_df_filtered, phase_summary_filtered)
        phase_df_filtered = _apply_object_scope(phase_df_filtered, object_scope)
        phase_summary_filtered = _apply_object_scope(phase_summary_filtered, object_scope)

    if phase_intervals_csv:
        if not isinstance(phase_summary_filtered, pd.DataFrame):
            skipped_reasons.append("phase_intervals_unavailable")
        else:
            interval_export_df = _prepare_phase_interval_export_df(phase_summary_filtered, phase_column_profile)
            interval_export_df = _serialize_object_columns_for_csv(interval_export_df)
            _ensure_parent_dir(phase_intervals_csv)
            interval_export_df.to_csv(phase_intervals_csv, index=False)
            export_result["phase_intervals_rows"] = int(len(interval_export_df))
            wrote_csv_artifact = True

    if phase_labels_csv:
        if not isinstance(phase_df_filtered, pd.DataFrame):
            skipped_reasons.append("phase_labels_unavailable")
        else:
            label_export_df = _prepare_phase_label_export_df(phase_df_filtered, phase_column_profile)
            label_export_df = _serialize_object_columns_for_csv(label_export_df)
            _ensure_parent_dir(phase_labels_csv)
            label_export_df.to_csv(phase_labels_csv, index=False)
            export_result["phase_labels_rows"] = int(len(label_export_df))
            wrote_csv_artifact = True

    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance_path = export_cfg.get("provenance_sidecar_json")
        if not provenance_path and wrote_csv_artifact:
            provenance_path = (
                _default_sidecar_path(output_csv)
                or _default_sidecar_path(phase_intervals_csv)
                or _default_sidecar_path(phase_labels_csv)
            )
        if provenance_path:
            _ensure_parent_dir(provenance_path)
            with open(provenance_path, "w", encoding="utf-8") as fh:
                json.dump(provenance, fh, indent=2, default=str)
            export_result["provenance_sidecar_json"] = provenance_path

    if skipped_reasons:
        export_result["skipped_reasons"] = sorted(set(skipped_reasons))

    if not wrote_csv_artifact:
        export_result["status"] = "skipped"
    elif skipped_reasons:
        export_result["status"] = "ok_with_skips"

    return export_result


def _safe_code_version_hash() -> Optional[str]:
    try:
        repo_dir = os.path.dirname(__file__)
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def run_starlink_pipeline(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute the Stage-1/2 pipeline using a dict configuration.

    Stages:
        ingest -> quality_control -> deduplication -> feature_derivation ->
        synchronization_utilities -> analysis_dispatch -> plotting -> export
    """
    cfg = _deep_update(get_default_pipeline_config(), config or {})

    phase_exports_requested = bool(
        cfg.get("export", {}).get("phase_intervals_csv")
        or cfg.get("export", {}).get("phase_labels_csv")
    )
    if phase_exports_requested and not bool(cfg.get("analysis_dispatch", {}).get("return_results", True)):
        cfg["analysis_dispatch"]["return_results"] = True

    t0 = perf_counter()
    stage_times = {}

    def timed(stage_name: str, fn, *args, **kwargs):
        ts = perf_counter()
        out = fn(*args, **kwargs)
        stage_times[stage_name] = float(perf_counter() - ts)
        return out

    ingest = timed("ingest", _run_ingest_stage, cfg)
    quality = timed("quality_control", _run_quality_control_stage, ingest["panel"], cfg["quality_control"])
    dedup = timed("deduplication", _run_dedup_stage, quality["panel"], cfg["deduplication"])
    features = timed("feature_derivation", _run_feature_stage, dedup["panel"], cfg["feature_derivation"])
    sync = timed("synchronization_utilities", _run_sync_stage, features["panel"], cfg["synchronization"])
    analysis = timed(
        "analysis_dispatch",
        _run_analysis_dispatch_stage,
        features["panel"],
        ingest["filenames_array"],
        ingest["fileNames"],
        cfg["analysis_dispatch"],
    )

    feat_panel = features.get("panel", pd.DataFrame())
    time_min = None
    time_max = None
    if isinstance(feat_panel, pd.DataFrame) and "timestamp" in feat_panel.columns and not feat_panel.empty:
        ts = pd.to_datetime(feat_panel["timestamp"], errors="coerce")
        if ts.notna().any():
            time_min = ts.min()
            time_max = ts.max()

    source_satellites = 0
    ingest_panel = ingest.get("panel", pd.DataFrame())
    if isinstance(ingest_panel, pd.DataFrame):
        if "sat_id" in ingest_panel.columns:
            source_satellites = int(ingest_panel["sat_id"].astype(str).nunique())
        elif "norad_cat_id" in ingest_panel.columns:
            source_satellites = int(ingest_panel["norad_cat_id"].astype(str).nunique())

    filters_applied = [
        "numeric_coercion_and_physical_bounds",
        f"max_sma_km<={cfg['quality_control'].get('max_sma_km', MAX_ANALYSIS_SMA_KM)}",
    ]
    if bool(cfg["quality_control"].get("drop_invalid_timestamp", True)):
        filters_applied.append("drop_invalid_timestamp")
    if bool(cfg["deduplication"].get("enabled", False)):
        filters_applied.append("deduplicate_rows")

    provenance = {
        "source_rows": int(ingest.get("metadata", {}).get("total_records_loaded", 0)),
        "source_satellites": int(source_satellites),
        "time_window": {
            "start": None if time_min is None else str(pd.Timestamp(time_min)),
            "end": None if time_max is None else str(pd.Timestamp(time_max)),
        },
        "filters_applied": filters_applied,
        "phase_variable": PHASE_VARIABLE_TRUE_ANOMALY,
        "phase_semantics": PHASE_SEMANTICS_TRUE_ANOMALY_PROXY,
        "shell_profile": cfg["analysis_dispatch"].get("starlink_shell_profile", "gen1"),
        "state_model": analysis.get("metadata", {}).get("state_model", cfg["analysis_dispatch"].get("state_model", "sgp4_preferred")),
        "resampling_info": {
            "spectral_mode": "uniform_fft",
            "wavelet_method": "wwz",
            "wavelet_irregular_policy": "resample",
            "spectral_cadence_seconds": cfg["analysis_dispatch"].get("spectral_cadence_seconds"),
            "fft_interpolation": cfg["analysis_dispatch"].get("fft_interpolation", "linear"),
            "crosscorr_interpolation": cfg["analysis_dispatch"].get("crosscorr_interpolation", "linear"),
            "crosscorr_normalization": cfg["analysis_dispatch"].get("crosscorr_normalization", "legacy"),
        },
        "code_version_hash": _safe_code_version_hash(),
    }

    export_info = timed(
        "export",
        _run_export_stage,
        {
            "stages": {
                "quality_control": quality,
                "feature_derivation": features,
                "analysis_dispatch": analysis,
            },
            "provenance": provenance,
        },
        cfg["export"],
    )

    total_runtime = float(perf_counter() - t0)

    return {
        "config": cfg,
        "runtime_seconds": total_runtime,
        "stage_timings_seconds": stage_times,
        "provenance": provenance,
        "stages": {
            "ingest": ingest,
            "quality_control": quality,
            "deduplication": dedup,
            "feature_derivation": features,
            "synchronization_utilities": sync,
            "analysis_dispatch": analysis,
            "plotting": {"handled_in_analysis_dispatch": True},
            "export": export_info,
        },
    }


def _prompt_bool(prompt: str, default: bool = False) -> bool:
    default_num = "1" if default else "0"
    raw = input(f"{prompt} (1=yes, 0=no, default={default_num}): ").strip()
    if raw == "":
        return default
    return bool(int(raw))


def _prompt_cli_config() -> Dict[str, Any]:
    """Keep CLI behavior as a thin wrapper around pipeline config."""
    plot_data = int(input("Enter 1 to plot the data, 0 to skip plotting: "))

    generation_selection = input("Select Starlink generation [all/gen1/gen2] (default=gen1): ").strip().lower() or "gen1"
    generation_only_files = _resolve_generation_only_files(generation_selection)

    tle_file_limit_input = input("Number of TLE satellite files to load (blank=all): ").strip()
    tle_file_limit = None
    if tle_file_limit_input:
        tle_file_limit = int(tle_file_limit_input)
        if tle_file_limit <= 0:
            tle_file_limit = None

    selected_satellites = None
    plot_stacked_periodogram = False
    plot_crosscorr_heatmap = False
    run_fft = False
    run_wavelet = False
    run_crosscorr = False
    run_relative_motion = False
    run_resonance_diagnostics = False
    run_shell_analytics = False
    run_disposal_metrics = False
    run_sustainability_metrics = False
    run_risk_screening = False
    include_proxy_risk_score = False
    plot_new_analytics = False
    inc_sma_render_mode = "scatter"
    inc_sma_metric_mode = "standardized_euclidean"
    inc_sma_reference_lines = True
    inc_sma_reference_annotation = False
    inc_sma_reference_markers = True
    inc_sma_target_sma_tolerance_km = 25.0
    inc_sma_target_inclination_tolerance_deg = 0.4
    inc_sma_target_profiles = None
    inc_sma_focused_profiles = True
    save_plots = True
    export_plot_data = False
    phase_intervals_csv = None
    phase_labels_csv = None

    state_model_input = input("State model [classical/sgp4_preferred/sgp4_required] (default=sgp4_preferred): ").strip().lower()
    state_model = _normalize_state_model_compat(state_model_input or "sgp4_preferred")

    if plot_data == 1:
        print_summary = _prompt_bool("Print object-time summary?", default=True)
        run_maneuver_analysis = _prompt_bool("Run maneuver/mission-phase analysis?", default=False)
        plot_maneuver_layers = _prompt_bool("Plot maneuver analysis layers?", default=False)
        run_fft = _prompt_bool("Run FFT stage?", default=False)
        run_wavelet = _prompt_bool("Run wavelet stage?", default=False)
        run_crosscorr = _prompt_bool("Run cross-correlation stage?", default=False)
        run_relative_motion = _prompt_bool("Run relative motion stage?", default=False)

        if run_fft or run_wavelet or run_crosscorr:
            plot_stacked_periodogram = _prompt_bool("Plot median stacked periodogram?", default=False)
            plot_crosscorr_heatmap = _prompt_bool("Plot aligned cross-correlation heatmap?", default=False)

        run_resonance_diagnostics = _prompt_bool("Run J2/SRP-aware resonance diagnostics?", default=False)
        run_shell_analytics = _prompt_bool("Run shell occupancy/regime analytics?", default=False)
        run_disposal_metrics = _prompt_bool("Run disposal corridor metrics?", default=False)
        run_sustainability_metrics = _prompt_bool("Run sustainability metrics?", default=False)
        run_risk_screening = _prompt_bool("Run coarse risk screening proxies?", default=False)
        include_proxy_risk_score = _prompt_bool("Include heuristic proxy risk score?", default=False)
        plot_new_analytics = _prompt_bool("Plot new analytics panels?", default=False)

        inc_sma_metric_mode_input = input(
            "Inc-vs-SMA metric mode [euclidean/standardized_euclidean/mahalanobis/nondimensional_constellation] "
            "(default=standardized_euclidean): "
        ).strip().lower()
        if inc_sma_metric_mode_input in {
            "euclidean",
            "standardized_euclidean",
            "mahalanobis",
            "nondimensional_constellation",
        }:
            inc_sma_metric_mode = inc_sma_metric_mode_input

        show_plots = _prompt_bool("Show plots interactively?", default=False)
        save_plots = _prompt_bool("Save all generated plots to a folder at 600 DPI?", default=True)
        export_plot_data = _prompt_bool(
            "Export per-figure plot data (CSV + manifest.json) for offline replotting?",
            default=False,
        )

    else:
        print_summary = False
        run_maneuver_analysis = False
        plot_maneuver_layers = False
        show_plots = False

    export_enabled = bool(phase_intervals_csv or phase_labels_csv)

    return {
        "only_files": generation_only_files,
        "tle_file_limit": tle_file_limit,
        "analysis_dispatch": {
            "print_summary": print_summary,
            "run_maneuver_analysis": run_maneuver_analysis,
            "plot_maneuver_layers": plot_maneuver_layers,
            "show_plots": show_plots,
            "save_plots": save_plots,
            "plot_output_dir": "plots",
            "plot_save_dpi": 600,
            "export_plot_data": export_plot_data,
            "plot_data_output_dir": "plot_data",
            "plot_data_compress_threshold_kb": 256,
            "spectral_mode": "uniform_fft",
            "spectral_cadence_seconds": None,
            "spectral_selected_satellites": selected_satellites,
            "plot_stacked_periodogram": plot_stacked_periodogram,
            "run_fft": run_fft,
            "run_wavelet": run_wavelet,
            "run_crosscorr": run_crosscorr,
            "run_relative_motion": run_relative_motion,
            "fft_backend": "auto",
            "fft_gpu_min_samples": 4096,
            "wavelet_irregular_policy": "resample",
            "wavelet_method": "wwz",
            "wavelet_irregular_warning_mode": "once",
            "wwz_use_gpu": True,
            "wwz_freq_batch_size": 16,
            "crosscorr_preprocessing": "zscored",
            "plot_crosscorr_heatmap": plot_crosscorr_heatmap,
            "crosscorr_normalization": "legacy",
            "crosscorr_interpolation": "linear",
            "inc_sma_render_mode": inc_sma_render_mode,
            "inc_sma_metric_mode": inc_sma_metric_mode,
            "inc_sma_reference_lines": inc_sma_reference_lines,
            "inc_sma_reference_annotation": inc_sma_reference_annotation,
            "inc_sma_reference_markers": inc_sma_reference_markers,
            "inc_sma_target_sma_tolerance_km": inc_sma_target_sma_tolerance_km,
            "inc_sma_target_inclination_tolerance_deg": inc_sma_target_inclination_tolerance_deg,
            "inc_sma_target_profiles": inc_sma_target_profiles,
            "inc_sma_focused_profiles": inc_sma_focused_profiles,
            "run_resonance_diagnostics": run_resonance_diagnostics,
            "run_shell_analytics": run_shell_analytics,
            "run_disposal_metrics": run_disposal_metrics,
            "run_sustainability_metrics": run_sustainability_metrics,
            "run_risk_screening": run_risk_screening,
            "resonance_tolerance_rad_day": 1.0e-3,
            "analytics_time_freq": "7D",
            "include_proxy_risk_score": include_proxy_risk_score,
            "include_risk_composite_score": include_proxy_risk_score,
            "plot_new_analytics": plot_new_analytics,
            "starlink_shell_profile": "gen1",
            "compliance_horizons_years": [5, 25],
            "state_model": state_model,
            "relative_pair_mode": "explicit_only",
            "relative_max_pairs": None,
            "relative_same_shell_only": False,
            "return_results": True,
            "apply_plot_style": True,
        },
        "export": {
            "enabled": export_enabled,
            "output_csv": None,
            "phase_intervals_csv": phase_intervals_csv,
            "phase_labels_csv": phase_labels_csv,
            "phase_filter_mode": "maneuver_relevant_only",
            "phase_column_profile": "core",
            "maneuver_relevant_states": list(DEFAULT_MANEUVER_RELEVANT_PHASE_STATES),
            "include_only_objects_with_maneuver_labels": True,
            "export_rejected_rows": False,
            "rejected_rows_csv": None,
            "provenance_sidecar_json": None,
        },
        "synchronization": {
            "enabled": False,
            "mode": SYNC_MODE_TARGET_NEAREST if plot_data == 1 else SYNC_MODE_NEAREST_INTERSECTION,
            "tolerance": DEFAULT_SYNC_TOLERANCE,
        },
    }


def main() -> None:
    cli_config = _prompt_cli_config()
    payload = run_starlink_pipeline(cli_config)
    print(f"Pipeline completed in {payload['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
