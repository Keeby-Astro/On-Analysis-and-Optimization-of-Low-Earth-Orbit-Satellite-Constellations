from resonance import (plot_resonance_diagnostics_dashboard,
                       plot_resonance_proximity_ai_map,
                       resonance)
from orbital_elements_plot import (density_ra_versus_arg, inc_versus_sma,
                                   orbital_elements_plot,
                                   plot_common_epoch_shell_snapshot,
                                   plot_raan_vs_selected_phase_torus_density,
                                   plot_selected_phase_vs_altitude_density,
                                   plot_sma_vs_inclination_density_shell_map,
                                   plot_starlink_dynamical_atlas,
                                   ra_versus_arg)
from fft_orbital_elements import fft_orbital_elements
from cross_correlation import cross_correlation
from wavelet_transform_orbital_elements import wavelet_transform_orbital_elements
from relative_motion import relative_motion
from orbital_features import (add_altitude_features, add_altitude_regime,
                              add_low_eccentricity_flag,
                              assign_candidate_shell_id, select_phase_series)
from event_detection import (classify_mission_phase_all, detect_maneuvers_all,
                             summarize_phase_intervals)
from plotting_utils import (plot_altitude_inclination_with_events,
                            plot_bstar_with_events,
                            plot_event_rate_histogram,
                            plot_phase_colored_timeseries,
                            precession_rates,
                            plot_shell_entry_exit_timeline,
                            summarize_object_time_panel)
from resonance_diagnostics import (compute_secular_rates, evaluate_resonance_proximity,
                                   map_resonance_proximity_over_ai_grid, summarize_resonant_objects)
from shell_analytics import (compute_conjunction_grade_placeholder,
                             compute_conjunction_input_readiness,
                             compute_disposal_corridor_metrics,
                             compute_risk_screening,
                             compute_shell_analytics,
                             summarize_disposal_onset_timeline,
                             compute_sustainability_metrics,
                             plot_altitude_time_occupancy_heatmap,
                             plot_shell_width_vs_time)
from state_models import DEFAULT_STATE_MODEL, SUPPORTED_STATE_MODELS, normalize_state_model
from constants import (
    LOW_ECCENTRICITY_THRESHOLD,
    PHASE_SEMANTICS_TRUE_ANOMALY_PROXY,
    PHASE_VARIABLE_TRUE_ANOMALY,
)
import matplotlib.pyplot as plt
import os
import subprocess
from time import perf_counter
import numpy as np
import pandas as pd


STARLINK_SHELL_PROFILES = {
    # Nominal FCC-informed shell families for analytics labeling only.
    'gen1': [
        {'id': 'gen1_540', 'min_altitude_km': 535.0, 'max_altitude_km': 545.0, 'inclination_deg': 53.0},
        {'id': 'gen1_550', 'min_altitude_km': 545.0, 'max_altitude_km': 555.0, 'inclination_deg': 53.0},
        {'id': 'gen1_560', 'min_altitude_km': 555.0, 'max_altitude_km': 565.0, 'inclination_deg': 53.0},
        {'id': 'gen1_570', 'min_altitude_km': 565.0, 'max_altitude_km': 575.0, 'inclination_deg': 53.0},
    ],
    'gen2_2022': [
        {'id': 'gen2_2022_525', 'min_altitude_km': 520.0, 'max_altitude_km': 528.0, 'inclination_deg': 53.2},
        {'id': 'gen2_2022_530', 'min_altitude_km': 528.0, 'max_altitude_km': 533.0, 'inclination_deg': 53.2},
        {'id': 'gen2_2022_535', 'min_altitude_km': 533.0, 'max_altitude_km': 540.0, 'inclination_deg': 53.2},
    ],
    'gen2_current': [
        {'id': 'gen2_current_480', 'min_altitude_km': 470.0, 'max_altitude_km': 490.0, 'inclination_deg': 43.0},
        {'id': 'gen2_current_525', 'min_altitude_km': 520.0, 'max_altitude_km': 528.0, 'inclination_deg': 53.2},
        {'id': 'gen2_current_530', 'min_altitude_km': 528.0, 'max_altitude_km': 533.0, 'inclination_deg': 53.2},
        {'id': 'gen2_current_535', 'min_altitude_km': 533.0, 'max_altitude_km': 540.0, 'inclination_deg': 53.2},
    ],
}
STARLINK_SHELL_PROFILES['combined_known'] = (
    STARLINK_SHELL_PROFILES['gen1']
    + STARLINK_SHELL_PROFILES['gen2_2022']
    + STARLINK_SHELL_PROFILES['gen2_current']
)


def _resolve_starlink_shell_definitions(starlink_shell_profile):
    if starlink_shell_profile is None:
        return None
    profile_key = str(starlink_shell_profile).strip().lower()
    return STARLINK_SHELL_PROFILES.get(profile_key)


def _normalize_compliance_horizons_years(compliance_horizons_years):
    if compliance_horizons_years is None:
        return [5, 25]
    try:
        values = [int(v) for v in compliance_horizons_years]
    except Exception:
        return [5, 25]
    values = [v for v in values if v > 0]
    if not values:
        return [5, 25]
    return sorted(set(values))


def _best_effort_git_hash():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _normalize_cached_table(df, date_columns=(), id_columns=("object_id", "sat_id", "norad_cat_id")):
    if not isinstance(df, pd.DataFrame):
        return None
    out = df.copy()
    for column in date_columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors='coerce')
    for column in id_columns:
        if column in out.columns:
            out[column] = out[column].where(out[column].isna(), out[column].astype(str))
    return out


def _empty_maneuver_events_df():
    return pd.DataFrame(
        columns=[
            'object_id',
            'estimated_event_time',
            'event_time_lower',
            'event_time_upper',
            'event_type',
            'event_score',
            'quality_flag',
            'event_layer',
        ]
    )


def _has_cached_phase_tables(events_df, phase_df, phase_summary_df):
    return any(isinstance(df, pd.DataFrame) for df in (events_df, phase_df, phase_summary_df))


def orbital_visualization(all_tle_data, filenames_array, fileNames,
                          ecc_threshold=LOW_ECCENTRICITY_THRESHOLD, altitude_bins=None, shell_definitions=None, print_summary=False,
                          starlink_shell_profile='gen1',
                          run_maneuver_analysis=False, maneuver_config=None, phase_config=None,
                          cached_maneuver_events_df=None, cached_phase_df=None, cached_phase_summary_df=None,
                          maneuver_cache_metadata=None,
                          plot_maneuver_layers=False, show_plots=False,
                          spectral_mode='uniform_fft', spectral_cadence_seconds=None,
                          spectral_selected_satellites=None, plot_stacked_periodogram=False,
                          run_fft=False, run_wavelet=False, run_crosscorr=False, run_relative_motion=False,
                          fft_backend='auto', fft_gpu_min_samples=4096,
                          fft_interpolation='linear', fft_stack_mode=None,
                          fft_normalize_before_stack=True, fft_show_period_axis=True,
                          fft_extract_peaks=False, fft_peak_top_k=5,
                          fft_peak_min_prominence=None, fft_peak_min_distance_bins=1,
                          fft_peak_overlay=False, fft_bootstrap_replicates=0,
                          fft_bootstrap_seed=0,
                          wavelet_irregular_policy='resample',
                          wavelet_method='wwz',
                          wavelet_irregular_warning_mode='once',
                          wwz_use_gpu=True,
                          wwz_freq_batch_size=16,
                          wwz_extract_peaks=True,
                          wwz_peak_top_k=5,
                          wwz_peak_min_prominence=None,
                          wwz_peak_min_separation_tau_bins=2,
                          wwz_peak_min_separation_period_bins=2,
                          wwz_overlay_ridge=False,
                          wwz_annotate_panels=False,
                          wwz_min_effective_n=None,
                          wwz_export_combined_summary=True,
                          crosscorr_preprocessing='zscored',
                          plot_crosscorr_heatmap=False,
                          crosscorr_min_overlap=32,
                          crosscorr_max_grid_points=None,
                          crosscorr_max_plot_points=None,
                          crosscorr_include_frequency_products=False,
                          crosscorr_include_cross_wavelet=False,
                          crosscorr_freq_nperseg=256,
                          crosscorr_freq_noverlap=None,
                          run_resonance_diagnostics=False, resonance_definitions=None,
                          resonance_tolerance_rad_day=1.0e-3,
                          run_shell_analytics=True, run_disposal_metrics=True,
                          run_sustainability_metrics=True, run_risk_screening=True,
                          analytics_time_freq='7D', plot_new_analytics=False,
                          occupancy_heatmap_time_axis_mode='datetime',
                          occupancy_heatmap_y_label_mode='bin_labels',
                          occupancy_heatmap_norm='linear',
                          occupancy_heatmap_clip_percentile=None,
                          occupancy_heatmap_overlay_altitude_refs_km=None,
                          occupancy_heatmap_use_pcolormesh=True,
                          occupancy_heatmap_smoothing_sigma=None,
                          raan_phase_density_mode='hist2d',
                          raan_phase_density_family_mode='aggregate',
                          raan_phase_density_family_targets_deg=None,
                          raan_phase_density_family_tolerance_deg=0.4,
                          raan_phase_density_time_windows=None,
                          raan_phase_density_rolling_window_days=None,
                          raan_phase_density_return_arrays=True,
                          raan_phase_density_compute_uniformity_metrics=True,
                          phase_alt_density_mode='circular_linear_kde',
                          phase_alt_density_family_mode='aggregate',
                          phase_alt_density_family_targets_deg=None,
                          phase_alt_density_family_tolerance_deg=0.4,
                          phase_alt_density_time_windows=None,
                          phase_alt_density_rolling_window_days=None,
                          phase_alt_density_step_days=None,
                          phase_alt_density_top_k_hotspots=5,
                          phase_alt_density_return_arrays=True,
                          phase_alt_density_normalization='per_panel',
                          phase_alt_density_overlay_altitude_refs_km=None,
                          include_risk_composite_score=True,
                          include_proxy_risk_score=None,
                          compliance_horizons_years=None,
                          state_model='sgp4_preferred',
                          relative_model='exact_lvlh',
                          relative_n_periods=10,
                          relative_samples_per_period=100,
                          relative_max_duration_seconds=None,
                          relative_tolerance_seconds=300,
                          relative_pair_list=None,
                          relative_pair_mode='explicit_only',
                          relative_max_pairs=None,
                          relative_same_shell_only=False,
                          shell_refine_with_inclination=False,
                          inclination_time_render_mode='scatter',
                          inclination_reference_lines=True,
                          inclination_reference_assignment_tolerance_deg=0.4,
                          inclination_reference_annotation=False,
                          precession_group_targets_deg=(53.0, 70.0, 97.6),
                          precession_group_assignment_tolerance_deg=0.5,
                          precession_group_trend_mode='none',
                          precession_group_rolling_window_days=30,
                          precession_show_group_envelopes=True,
                          precession_apsidal_ecc_floor=1.0e-3,
                          precession_low_e_behavior='suppress',
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
                          argp_ecc_floor=1.0e-3,
                          argp_low_e_behavior='suppress',
                          argp_show_argument_of_latitude_companion=False,
                          eccentricity_time_render_mode='scatter',
                          eccentricity_highlight_descents=True,
                          eccentricity_descent_threshold_per_day=-5.0e-6,
                          eccentricity_descent_min_duration_days=7.0,
                          eccentricity_rolling_window_days=30,
                          eccentricity_show_envelope=True,
                          eccentricity_zoom_ylim=None,
                          inc_sma_render_mode='scatter',
                          inc_sma_metric_mode='standardized_euclidean',
                          inc_sma_reference_lines=True,
                          inc_sma_reference_annotation=False,
                          inc_sma_reference_markers=True,
                          inc_sma_target_sma_tolerance_km=25.0,
                          inc_sma_target_inclination_tolerance_deg=0.4,
                          inc_sma_target_profiles=None,
                          inc_sma_focused_profiles=True,
                          crosscorr_normalization='legacy',
                          crosscorr_interpolation='linear',
                          return_results=True,
                          maneuver_event_layer_mode='accepted_high_confidence',
                          maneuver_event_high_confidence_threshold=0.8,
                          maneuver_event_color_mode='event_type',
                          maneuver_show_event_uncertainty=True,
                          maneuver_event_histogram_basis='accepted',
                          maneuver_phase_use_interval_overlay=True,
                          maneuver_phase_alpha_by_confidence=True,
                          maneuver_timeline_sort_mode='first_epoch',
                          maneuver_timeline_confidence_shading=False,
                          maneuver_timeline_zoom_inset=True,
                          maneuver_timeline_zoom_start=None,
                          maneuver_timeline_zoom_end='2025-12-31',
                          maneuver_timeline_zoom_group_window_days=1825.0,
                          maneuver_timeline_zoom_size_inches=2.35,
                          sustainability_count_basis='objects',
                          risk_timeline_mode='crossing_and_score',
                          disposal_family_col='candidate_shell_id',
                          disposal_cohort_col=None,
                          disposal_include_age_at_onset=True,
                          disposal_onset_group_by=None,
                          disposal_onset_candidate_only=False,
                          disposal_onset_include_age_stats=True,
                          eccentricity_descent_min_eccentricity=1.8e-3,
                          enable_sma_inclination_density_shell_map=False,
                          enable_common_epoch_shell_snapshot=False,
                          enable_starlink_dynamical_atlas=False):
    """Run the orbital-visualization pipeline.

    Migration Notes:
        Existing call sites remain valid; all new parameters are optional.
        Phase handling is fixed to true anomaly across the pipeline.
        If an enriched panel already includes low-e-safe columns
        (`true_anomaly`, `recommended_phase_variable`, `low_eccentricity`),
        those values are consumed directly.
        Set `run_maneuver_analysis=True` to enable TLE-history event/phase
        inference (optional, default off for compatibility).
        Set `show_plots=False` to suppress interactive figure display.
        `spectral_mode` is fixed to `uniform_fft`.
        `fft_backend` supports `auto|numpy|cupy|torch` for uniform-grid FFT
        acceleration path.
        Set `return_results=True` to receive a structured payload containing
        tables and execution metadata.
        `state_model` controls relative-motion initial-state generation policy:
        `classical|sgp4_preferred|sgp4_required`.
    """
    t_all = perf_counter()
    viz_df = all_tle_data.copy()
    stage_timings = {}
    stage_execution = {}

    def mark_stage_skipped(label, reason):
        stage_timings[label] = None
        stage_execution[label] = {'executed': False, 'reason': str(reason)}

    # Hard-cleanup policy: fixed spectral/wavelet dispatch behavior.
    spectral_mode = 'uniform_fft'
    wavelet_method = 'wwz'
    wavelet_irregular_policy = 'resample'

    use_proxy_risk_score = bool(include_risk_composite_score)
    if include_proxy_risk_score is not None:
        use_proxy_risk_score = bool(include_proxy_risk_score)
    compliance_horizons_years = _normalize_compliance_horizons_years(compliance_horizons_years)

    viz_df = add_low_eccentricity_flag(viz_df, threshold=ecc_threshold)
    viz_df = add_altitude_features(viz_df)
    if altitude_bins is not None:
        viz_df = add_altitude_regime(viz_df, bins=altitude_bins)
    effective_shell_definitions = shell_definitions
    if effective_shell_definitions is None:
        effective_shell_definitions = _resolve_starlink_shell_definitions(starlink_shell_profile)
    if effective_shell_definitions is not None:
        # Shell profiles are nominal labeling aids, not hard-truth or filtering rules.
        viz_df = assign_candidate_shell_id(
            viz_df,
            shell_definitions=effective_shell_definitions,
            use_inclination_refinement=bool(shell_refine_with_inclination),
        )
    viz_df['shell_profile_semantics'] = 'Nominal shell-profile labeling only (descriptive/proxy; not conjunction truth).'
    if 'shell_assignment_basis' not in viz_df.columns:
        viz_df['shell_assignment_basis'] = 'none'

    # Keep dispatch behavior scientifically explicit and stable: downstream products
    # are built on true_anomaly as a TLE-derived Kepler proxy for compatibility.
    viz_df = select_phase_series(
        viz_df,
        requested_variable=PHASE_VARIABLE_TRUE_ANOMALY,
        ecc_threshold=ecc_threshold,
        low_e_choice=PHASE_VARIABLE_TRUE_ANOMALY,
    )
    if 'true_anomaly' in viz_df.columns and 'true_anomaly_kepler_proxy_deg' not in viz_df.columns:
        viz_df['true_anomaly_kepler_proxy_deg'] = pd.to_numeric(viz_df['true_anomaly'], errors='coerce')
    viz_df['phase_variable'] = PHASE_VARIABLE_TRUE_ANOMALY
    viz_df['phase_semantics'] = PHASE_SEMANTICS_TRUE_ANOMALY_PROXY

    if print_summary:
        summarize_object_time_panel(viz_df)

    # Optional physically grounded analytics are additive and default-off.
    resonance_diag_df = None
    resonance_object_summary = None
    resonance_ai_map_df = None
    shell_time_df = None
    shell_entry_exit_df = None
    shell_heatmap_df = None
    shell_transition_df = None
    shell_transition_matrix_df = None
    shell_snapshot_df = None
    disposal_record_df = None
    disposal_summary_df = None
    disposal_onset_timeline_df = None
    sustainability_df = None
    sustainability_shell_df = None
    events_df = None
    phase_df = None
    phase_summary_df = None
    maneuver_analysis_source = 'not_run'
    risk_density_df = None
    risk_crossing_df = None
    risk_severity_df = None
    conjunction_input_readiness = None
    conjunction_placeholder = None
    fft_results_payload = None
    crosscorr_results_payload = None
    wavelet_results_payload = None
    relative_motion_payload = None
    inc_sma_results_payload = None
    precession_results_payload = None
    raan_phase_density_payload = None
    phase_alt_density_payload = None
    occupancy_heatmap_payload = None
    new_analytics_figures = 0
    collect_stage_results = bool(return_results)

    sma = viz_df['sma'].values
    inc = viz_df['inc'].values
    raan = viz_df['raan'].values
    aop = viz_df['aop'].values
    ecc = viz_df['ecc'].values
    mean_anomaly = viz_df['mean_anomaly'].values
    node_precession = viz_df['node_precession_rate'].values
    perigee_precession = viz_df['perigee_precession_rate'].values
    true_anomaly = viz_df['true_anomaly'].values
    timestamp_values = viz_df['timestamp'].values
    if 'norad_cat_id' in viz_df.columns:
        object_ids = viz_df['norad_cat_id'].astype(str).values
    elif 'sat_id' in viz_df.columns:
        object_ids = viz_df['sat_id'].astype(str).values
    else:
        object_ids = np.asarray(filenames_array).astype(str)
    selected_phase = viz_df['selected_phase_deg'].values
    low_e_mask = viz_df['low_eccentricity'].values
    altitude_km = viz_df['altitude_km'].values
    shell_ids = viz_df['candidate_shell_id'].values if 'candidate_shell_id' in viz_df.columns else None
    argument_of_latitude_deg = viz_df['argument_of_latitude_deg'].values if 'argument_of_latitude_deg' in viz_df.columns else None

    relative_pair_list_runtime = relative_pair_list
    if relative_pair_list_runtime is not None:
        try:
            relative_pair_list_runtime = [tuple(p) for p in relative_pair_list_runtime if len(tuple(p)) >= 2]
        except Exception:
            relative_pair_list_runtime = None
    if relative_pair_list_runtime is not None and bool(relative_same_shell_only) and shell_ids is not None:
        pair_shell_df = pd.DataFrame({'object_id': object_ids.astype(str), 'shell_id': shell_ids})
        pair_shell_df = pair_shell_df.dropna(subset=['shell_id'])
        shell_map = pair_shell_df.groupby('object_id')['shell_id'].agg(lambda s: s.iloc[0]).to_dict()
        relative_pair_list_runtime = [
            p for p in relative_pair_list_runtime
            if shell_map.get(str(p[0])) is not None and shell_map.get(str(p[0])) == shell_map.get(str(p[1]))
        ]
    if relative_pair_list_runtime is not None and relative_max_pairs is not None:
        try:
            max_pairs = int(relative_max_pairs)
        except Exception:
            max_pairs = None
        if max_pairs is not None and max_pairs >= 0:
            relative_pair_list_runtime = relative_pair_list_runtime[:max_pairs]

    print(f"Generating orbital visualizations for {len(viz_df):,} rows...")

    effective_show_plots = bool(show_plots)
    spectral_selected_runtime = spectral_selected_satellites
    try:
        state_model = normalize_state_model(state_model)
    except ValueError:
        print(f"[state_model] Unsupported value '{state_model}'. Defaulting to {DEFAULT_STATE_MODEL}. "
              f"Supported values: {SUPPORTED_STATE_MODELS}")
        state_model = DEFAULT_STATE_MODEL

    def run_step(label, fn, *args, **kwargs):
        t_step = perf_counter()
        print(f"[{label}] start")
        out = fn(*args, **kwargs)
        elapsed = perf_counter() - t_step
        stage_timings[label] = elapsed
        stage_execution[label] = {'executed': True, 'reason': None}
        print(f"[{label}] done in {elapsed:.2f}s")
        return out

    if effective_show_plots:
        inc_sma_results_payload = run_step(
            'inc_versus_sma',
            inc_versus_sma,
            sma,
            inc,
            fileNames,
            filenames_array,
            metric_mode=inc_sma_metric_mode,
            render_mode=inc_sma_render_mode,
            show_plots=effective_show_plots,
            return_figures=False,
            return_results=collect_stage_results,
            reference_lines=bool(inc_sma_reference_lines),
            reference_annotation=bool(inc_sma_reference_annotation),
            reference_markers=bool(inc_sma_reference_markers),
            target_sma_tolerance_km=float(inc_sma_target_sma_tolerance_km),
            target_inclination_tolerance_deg=float(inc_sma_target_inclination_tolerance_deg),
            target_profiles=inc_sma_target_profiles,
            focused_profiles=bool(inc_sma_focused_profiles),
        )
        run_step('density_ra_versus_arg', density_ra_versus_arg, aop, raan, fileNames, filenames_array,
                 'raan_vs_true_anomaly', selected_phase)
        run_step('ra_versus_arg', ra_versus_arg, ecc, aop, raan, fileNames, filenames_array,
                 'raan_vs_true_anomaly', selected_phase, low_e_mask)
        mark_stage_skipped('resonance', 'resonance plotting disabled by workflow policy')

        argp_display_mode = 'raw_all'
        argp_behavior_key = str(argp_low_e_behavior or 'suppress').strip().lower()
        if bool(argp_show_argument_of_latitude_companion):
            argp_display_mode = 'arglat_companion'
        elif argp_behavior_key == 'split':
            argp_display_mode = 'split_validity'
        elif argp_behavior_key == 'suppress':
            argp_display_mode = 'ecc_filtered'

        run_step('orbital_elements_plot', orbital_elements_plot, inc, sma, raan, aop, ecc, true_anomaly, timestamp_values,
                 fileNames, filenames_array, 'altitude_vs_true_anomaly', selected_phase,
                 altitude_km, shell_ids, low_e_mask,
                 inclination_time_render_mode=inclination_time_render_mode,
                 inclination_reference_lines=inclination_reference_lines,
                 inclination_reference_assignment_tolerance_deg=inclination_reference_assignment_tolerance_deg,
                 inclination_reference_annotation=inclination_reference_annotation,
                 raan_time_render_mode=raan_time_render_mode,
                 raan_display_mode=raan_display_mode,
                 raan_residual_fit_mode=raan_residual_fit_mode,
                 raan_residual_annotation=raan_residual_annotation,
                 raan_min_points_for_unwrap=raan_min_points_for_unwrap,
                 sma_time_render_mode=sma_time_render_mode,
                 sma_reference_lines=sma_reference_lines,
                 sma_reference_values_km=sma_reference_values_km,
                 sma_rolling_window_days=sma_rolling_window_days,
                 sma_show_envelope=sma_show_envelope,
                 sma_operational_bands=sma_operational_bands,
                 argp_time_render_mode=argp_time_render_mode,
                 argp_display_mode=argp_display_mode,
                 argp_ecc_floor=argp_ecc_floor,
                 argp_low_e_behavior=argp_low_e_behavior,
                 argp_show_argument_of_latitude_companion=argp_show_argument_of_latitude_companion,
                 argument_of_latitude_series=argument_of_latitude_deg,
                 eccentricity_time_render_mode=eccentricity_time_render_mode,
                 eccentricity_highlight_descents=eccentricity_highlight_descents,
                 eccentricity_descent_threshold_per_day=eccentricity_descent_threshold_per_day,
                 eccentricity_descent_min_duration_days=eccentricity_descent_min_duration_days,
                 eccentricity_descent_min_eccentricity=eccentricity_descent_min_eccentricity,
                 eccentricity_rolling_window_days=eccentricity_rolling_window_days,
                 eccentricity_show_envelope=eccentricity_show_envelope,
                 eccentricity_zoom_ylim=eccentricity_zoom_ylim,
                 enable_common_epoch_shell_snapshot=bool(enable_common_epoch_shell_snapshot))

        # Geometric atlas views with one cached decimation for interactivity.
        atlas_cache = {}

        def get_atlas_slice(max_points=120_000):
            key = int(max_points)
            if key in atlas_cache:
                return atlas_cache[key]
            n = len(selected_phase)
            if n <= key:
                idx = np.arange(n, dtype=np.int64)
            else:
                step = int(np.ceil(n / float(key)))
                idx = np.arange(0, n, step, dtype=np.int64)
            atlas_cache[key] = idx
            return idx

        atlas_idx = get_atlas_slice(120_000)
        phase_atlas = selected_phase[atlas_idx]
        raan_atlas = raan[atlas_idx]
        sma_atlas = sma[atlas_idx]
        inc_atlas = inc[atlas_idx]
        alt_atlas = altitude_km[atlas_idx]
        shell_atlas = shell_ids[atlas_idx] if shell_ids is not None else None

        raan_phase_density_payload = run_step(
            'plot_raan_vs_true_anomaly_torus_density',
            plot_raan_vs_selected_phase_torus_density,
            raan_atlas,
            phase_atlas,
            shell_series=shell_atlas,
            mode=raan_phase_density_mode,
            show_plots=effective_show_plots,
            return_results=collect_stage_results,
            timestamps=timestamp_values[atlas_idx],
            inclinations=inc_atlas,
            raan_phase_density_mode=raan_phase_density_mode,
            raan_phase_density_family_mode=raan_phase_density_family_mode,
            raan_phase_density_family_targets_deg=raan_phase_density_family_targets_deg,
            raan_phase_density_family_tolerance_deg=raan_phase_density_family_tolerance_deg,
            raan_phase_density_time_windows=raan_phase_density_time_windows,
            raan_phase_density_rolling_window_days=raan_phase_density_rolling_window_days,
            raan_phase_density_return_arrays=bool(raan_phase_density_return_arrays),
            raan_phase_density_compute_uniformity_metrics=bool(raan_phase_density_compute_uniformity_metrics),
        )
        if bool(enable_sma_inclination_density_shell_map):
            run_step(
                'plot_sma_vs_inclination_density_shell_map',
                plot_sma_vs_inclination_density_shell_map,
                sma_atlas,
                inc_atlas,
                shell_series=shell_atlas,
                mode='hist2d',
                show_plots=effective_show_plots,
                return_results=False,
            )
        else:
            mark_stage_skipped('plot_sma_vs_inclination_density_shell_map', 'disabled by configuration')
        phase_alt_density_payload = run_step(
            'plot_true_anomaly_vs_altitude_density',
            plot_selected_phase_vs_altitude_density,
            phase_atlas,
            alt_atlas,
            mode=phase_alt_density_mode,
            show_plots=effective_show_plots,
            return_results=collect_stage_results,
            timestamps=timestamp_values[atlas_idx],
            inclinations=inc_atlas,
            phase_alt_density_mode=phase_alt_density_mode,
            phase_alt_density_family_mode=phase_alt_density_family_mode,
            phase_alt_density_family_targets_deg=phase_alt_density_family_targets_deg,
            phase_alt_density_family_tolerance_deg=phase_alt_density_family_tolerance_deg,
            phase_alt_density_time_windows=phase_alt_density_time_windows,
            phase_alt_density_rolling_window_days=phase_alt_density_rolling_window_days,
            phase_alt_density_step_days=phase_alt_density_step_days,
            phase_alt_density_top_k_hotspots=int(phase_alt_density_top_k_hotspots),
            phase_alt_density_return_arrays=bool(phase_alt_density_return_arrays),
            phase_alt_density_normalization=phase_alt_density_normalization,
            phase_alt_density_overlay_altitude_refs_km=phase_alt_density_overlay_altitude_refs_km,
        )

        if bool(enable_common_epoch_shell_snapshot):
            common_epoch_df = pd.DataFrame({
                'phase': phase_atlas,
                'raan': raan_atlas,
                'shell': shell_atlas if shell_atlas is not None else np.full(phase_atlas.shape, 'unknown', dtype=object),
                'altitude': alt_atlas,
            })
            run_step(
                'plot_common_epoch_shell_snapshot',
                plot_common_epoch_shell_snapshot,
                common_epoch_df['raan'].values,
                common_epoch_df['phase'].values,
                common_epoch_df['shell'].values,
                altitude_series=common_epoch_df['altitude'].values,
                show_plots=effective_show_plots,
                return_results=False,
            )
        else:
            mark_stage_skipped('plot_common_epoch_shell_snapshot', 'disabled by configuration')

        if bool(enable_starlink_dynamical_atlas):
            alt_med = float(np.nanmedian(alt_atlas)) if alt_atlas.size > 0 else 0.0
            alt_residual = alt_atlas - alt_med
            resonance_residual = np.abs(node_precession[atlas_idx] - np.nanmedian(node_precession[atlas_idx]))
            run_step(
                'plot_starlink_dynamical_atlas',
                plot_starlink_dynamical_atlas,
                raan_atlas,
                phase_atlas,
                shell_series=shell_atlas,
                altitude_residual=alt_residual,
                resonance_residual=resonance_residual,
                color_field='shell' if shell_atlas is not None else 'altitude_residual',
                show_plots=effective_show_plots,
                return_results=False,
            )
        else:
            mark_stage_skipped('plot_starlink_dynamical_atlas', 'disabled by configuration')
    else:
        print('[visualization] show_plots is disabled; skipping legacy plotting panels.')
        mark_stage_skipped('legacy_plot_panels', 'show_plots is False')

    if collect_stage_results and not effective_show_plots:
        atlas_cache = {}

        def get_atlas_slice(max_points=120_000):
            key = int(max_points)
            if key in atlas_cache:
                return atlas_cache[key]
            n = len(selected_phase)
            if n <= key:
                idx = np.arange(n, dtype=np.int64)
            else:
                step = int(np.ceil(n / float(key)))
                idx = np.arange(0, n, step, dtype=np.int64)
            atlas_cache[key] = idx
            return idx

        atlas_idx = get_atlas_slice(120_000)
        phase_atlas = selected_phase[atlas_idx]
        raan_atlas = raan[atlas_idx]
        inc_atlas = inc[atlas_idx]
        alt_atlas = altitude_km[atlas_idx]
        shell_atlas = shell_ids[atlas_idx] if shell_ids is not None else None

        raan_phase_density_payload = run_step(
            'plot_raan_vs_true_anomaly_torus_density',
            plot_raan_vs_selected_phase_torus_density,
            raan_atlas,
            phase_atlas,
            shell_series=shell_atlas,
            mode=raan_phase_density_mode,
            show_plots=False,
            return_results=True,
            timestamps=timestamp_values[atlas_idx],
            inclinations=inc_atlas,
            raan_phase_density_mode=raan_phase_density_mode,
            raan_phase_density_family_mode=raan_phase_density_family_mode,
            raan_phase_density_family_targets_deg=raan_phase_density_family_targets_deg,
            raan_phase_density_family_tolerance_deg=raan_phase_density_family_tolerance_deg,
            raan_phase_density_time_windows=raan_phase_density_time_windows,
            raan_phase_density_rolling_window_days=raan_phase_density_rolling_window_days,
            raan_phase_density_return_arrays=bool(raan_phase_density_return_arrays),
            raan_phase_density_compute_uniformity_metrics=bool(raan_phase_density_compute_uniformity_metrics),
        )

        phase_alt_density_payload = run_step(
            'plot_true_anomaly_vs_altitude_density',
            plot_selected_phase_vs_altitude_density,
            phase_atlas,
            alt_atlas,
            mode=phase_alt_density_mode,
            show_plots=False,
            return_results=True,
            timestamps=timestamp_values[atlas_idx],
            inclinations=inc_atlas,
            phase_alt_density_mode=phase_alt_density_mode,
            phase_alt_density_family_mode=phase_alt_density_family_mode,
            phase_alt_density_family_targets_deg=phase_alt_density_family_targets_deg,
            phase_alt_density_family_tolerance_deg=phase_alt_density_family_tolerance_deg,
            phase_alt_density_time_windows=phase_alt_density_time_windows,
            phase_alt_density_rolling_window_days=phase_alt_density_rolling_window_days,
            phase_alt_density_step_days=phase_alt_density_step_days,
            phase_alt_density_top_k_hotspots=int(phase_alt_density_top_k_hotspots),
            phase_alt_density_return_arrays=bool(phase_alt_density_return_arrays),
            phase_alt_density_normalization=phase_alt_density_normalization,
            phase_alt_density_overlay_altitude_refs_km=phase_alt_density_overlay_altitude_refs_km,
        )

    fft_stack_mode_runtime = 'median' if plot_stacked_periodogram else 'none'
    if fft_stack_mode is not None:
        fft_stack_mode_runtime = str(fft_stack_mode)

    if run_fft:
        fft_results_payload = run_step('fft_orbital_elements', fft_orbital_elements, inc, sma, raan, aop, ecc, true_anomaly, fileNames,
                                       filenames_array, phase_mode='true_anomaly_spectrum', phase_series=selected_phase,
                                       timestamps=timestamp_values, satellite_ids=object_ids, selected_satellites=spectral_selected_runtime,
                                       mode=spectral_mode, cadence_seconds=spectral_cadence_seconds, interpolation=fft_interpolation,
                                       stack_mode=fft_stack_mode_runtime, normalize_before_stack=bool(fft_normalize_before_stack),
                                       show_period_axis=bool(fft_show_period_axis), show_plots=effective_show_plots,
                                       return_results=collect_stage_results,
                                       fft_backend=fft_backend, gpu_min_samples=fft_gpu_min_samples,
                                       extract_peaks=bool(fft_extract_peaks), peak_top_k=int(fft_peak_top_k),
                                       peak_min_prominence=fft_peak_min_prominence,
                                       peak_min_distance_bins=int(fft_peak_min_distance_bins),
                                       peak_overlay=bool(fft_peak_overlay),
                                       bootstrap_replicates=int(fft_bootstrap_replicates),
                                       bootstrap_seed=int(fft_bootstrap_seed))
    else:
        mark_stage_skipped('fft_orbital_elements', 'run_fft is False')

    if run_crosscorr:
        t_cc = perf_counter()
        print('[cross_correlation_stage] start')
        # Keep cross-correlation responsive for full-catalog runs.
        cc_max_grid_points = 120_000 if effective_show_plots else 200_000
        cc_max_plot_points = 30_000 if effective_show_plots else 50_000
        if crosscorr_max_grid_points is not None:
            cc_max_grid_points = int(crosscorr_max_grid_points)
        if crosscorr_max_plot_points is not None:
            cc_max_plot_points = int(crosscorr_max_plot_points)

        crosscorr_results_payload = run_step('cross_correlation', cross_correlation, inc, sma, aop, raan, use_fft=True, timestamps=timestamp_values,
                             satellite_ids=object_ids, selected_satellites=spectral_selected_runtime, preprocessing=crosscorr_preprocessing,
                             cadence_seconds=spectral_cadence_seconds,  min_overlap=int(crosscorr_min_overlap), plot_heatmap=plot_crosscorr_heatmap,
                             show_plots=effective_show_plots,
                             return_results=collect_stage_results, phase_mode='true_anomaly_crosscorr', phase_series=selected_phase,
                             eccentricities=ecc, ecc_threshold=ecc_threshold,
                             max_grid_points=cc_max_grid_points, max_plot_points=cc_max_plot_points,
                             include_frequency_products=bool(crosscorr_include_frequency_products),
                             include_cross_wavelet=bool(crosscorr_include_cross_wavelet),
                             freq_nperseg=int(crosscorr_freq_nperseg), freq_noverlap=crosscorr_freq_noverlap,
                             normalization=crosscorr_normalization, interpolation=crosscorr_interpolation)
        print(f"[cross_correlation_stage] done in {perf_counter() - t_cc:.2f}s")
    else:
        mark_stage_skipped('cross_correlation', 'run_crosscorr is False')

    if effective_show_plots or collect_stage_results:
        precession_results_payload = run_step(
            'precession_rates',
            precession_rates,
            node_precession,
            perigee_precession,
            timestamp_values,
            show_plots=effective_show_plots,
            return_figures=bool(collect_stage_results),
            inclinations_deg=inc,
            eccentricities=ecc,
            precession_group_targets_deg=precession_group_targets_deg,
            precession_group_assignment_tolerance_deg=precession_group_assignment_tolerance_deg,
            precession_group_trend_mode=precession_group_trend_mode,
            precession_group_rolling_window_days=precession_group_rolling_window_days,
            precession_show_group_envelopes=precession_show_group_envelopes,
            precession_apsidal_ecc_floor=precession_apsidal_ecc_floor,
            precession_low_e_behavior=precession_low_e_behavior,
        )
    else:
        mark_stage_skipped('precession_rates', 'show_plots and return_results are both disabled')

    if run_wavelet:
        wavelet_results_payload = run_step('wavelet_transform_orbital_elements', wavelet_transform_orbital_elements, inc, sma, raan, aop, ecc, true_anomaly,
                                           timestamps=timestamp_values,
                                           satellite_ids=object_ids, selected_satellites=spectral_selected_runtime, cadence_seconds=spectral_cadence_seconds,
                                           interpolation='linear', irregular_policy=wavelet_irregular_policy,
                                           method=wavelet_method, irregular_warning_mode=wavelet_irregular_warning_mode,
                                           wwz_use_gpu=bool(wwz_use_gpu),
                                           wwz_freq_batch_size=int(wwz_freq_batch_size),
                                           wwz_extract_peaks=bool(wwz_extract_peaks),
                                           wwz_peak_top_k=int(wwz_peak_top_k),
                                           wwz_peak_min_prominence=wwz_peak_min_prominence,
                                           wwz_peak_min_separation_tau_bins=int(wwz_peak_min_separation_tau_bins),
                                           wwz_peak_min_separation_period_bins=int(wwz_peak_min_separation_period_bins),
                                           wwz_overlay_ridge=bool(wwz_overlay_ridge),
                                           wwz_annotate_panels=bool(wwz_annotate_panels),
                                           wwz_min_effective_n=wwz_min_effective_n,
                                           wwz_export_combined_summary=bool(wwz_export_combined_summary),
                                           show_plots=effective_show_plots,
                                           return_results=collect_stage_results)
    else:
        mark_stage_skipped('wavelet_transform_orbital_elements', 'run_wavelet is False')

    if run_relative_motion:
        if relative_pair_list_runtime is None and str(relative_pair_mode).strip().lower() == 'explicit_only':
            mark_stage_skipped('relative_motion', 'relative_pair_mode=explicit_only requires relative_pair_list')
        else:
            relative_motion_payload = run_step('relative_motion', relative_motion, viz_df, state_model=state_model,
                                               show_plots=effective_show_plots, return_results=collect_stage_results,
                                               relative_model=relative_model,
                                               n_periods=int(relative_n_periods),
                                               samples_per_period=int(relative_samples_per_period),
                                               max_duration_seconds=relative_max_duration_seconds,
                                               tolerance_seconds=float(relative_tolerance_seconds),
                                               pair_list=relative_pair_list_runtime)
    else:
        mark_stage_skipped('relative_motion', 'run_relative_motion is False')

    if run_resonance_diagnostics:
        t_res_diag = perf_counter()
        resonance_diag_df = compute_secular_rates(viz_df)
        resonance_diag_df = evaluate_resonance_proximity(
            resonance_diag_df,
            resonance_definitions=resonance_definitions,
            tolerance_rad_day=resonance_tolerance_rad_day,
            include_unwrapped_angles=False,
            include_apsidal_warning_text=False,
        )
        resonance_object_summary = summarize_resonant_objects(resonance_diag_df)
        ai_map = map_resonance_proximity_over_ai_grid(resonance_diag_df)
        resonance_ai_map_df = ai_map

        prox_count = int(resonance_diag_df['is_resonance_proximate'].fillna(False).astype(bool).sum())
        print(f"[resonance_diagnostics] proximate_records={prox_count:,}, objects={len(resonance_object_summary):,}")
        # Resonance diagnostics plotting intentionally disabled in this workflow.
        # Keep computations and tabular outputs available without generating figures.
        elapsed_res = perf_counter() - t_res_diag
        stage_timings['resonance_diagnostics'] = elapsed_res
        stage_execution['resonance_diagnostics'] = {'executed': True, 'reason': None}
        print(f"[resonance_diagnostics] done in {elapsed_res:.2f}s")
    else:
        mark_stage_skipped('resonance_diagnostics', 'run_resonance_diagnostics is False')

    if run_shell_analytics:
        t_shell = perf_counter()
        shell_time_df, shell_entry_exit_df, shell_heatmap_df, shell_transition_df, shell_transition_matrix_df, shell_snapshot_df = compute_shell_analytics(
            viz_df,
            altitude_bins=altitude_bins,
            time_freq=analytics_time_freq,
            return_extras=True,
        )
        print(f"[shell_analytics] shell_time_rows={len(shell_time_df):,}, entry_exit_rows={len(shell_entry_exit_df):,}, "
              f"transitions={len(shell_transition_df):,}")
        heatmap_needed = bool(collect_stage_results) or bool(effective_show_plots and plot_new_analytics)
        if heatmap_needed:
            hm_result = plot_altitude_time_occupancy_heatmap(
                shell_heatmap_df,
                value_col='n_objects',
                occupancy_heatmap_time_axis_mode=occupancy_heatmap_time_axis_mode,
                occupancy_heatmap_y_label_mode=occupancy_heatmap_y_label_mode,
                occupancy_heatmap_norm=occupancy_heatmap_norm,
                occupancy_heatmap_clip_percentile=occupancy_heatmap_clip_percentile,
                occupancy_heatmap_overlay_altitude_refs_km=occupancy_heatmap_overlay_altitude_refs_km,
                occupancy_heatmap_use_pcolormesh=bool(occupancy_heatmap_use_pcolormesh),
                occupancy_heatmap_smoothing_sigma=occupancy_heatmap_smoothing_sigma,
                show_plots=bool(effective_show_plots and plot_new_analytics),
                return_results=bool(collect_stage_results),
            )
            fig_hm = hm_result.get('figure') if collect_stage_results else hm_result
            if collect_stage_results:
                occupancy_heatmap_payload = hm_result

        if effective_show_plots and plot_new_analytics:
            fig_w = plot_shell_width_vs_time(shell_time_df)
            if fig_hm is not None:
                new_analytics_figures += 1
            if fig_w is not None:
                new_analytics_figures += 1
        elapsed_shell = perf_counter() - t_shell
        stage_timings['shell_analytics'] = elapsed_shell
        stage_execution['shell_analytics'] = {'executed': True, 'reason': None}
        print(f"[shell_analytics] done in {elapsed_shell:.2f}s")
    else:
        mark_stage_skipped('shell_analytics', 'run_shell_analytics is False')

    if run_disposal_metrics:
        t_disposal = perf_counter()
        disposal_record_df, disposal_summary_df = compute_disposal_corridor_metrics(
            viz_df,
            family_col=(str(disposal_family_col) if disposal_family_col is not None else 'candidate_shell_id'),
            cohort_col=disposal_cohort_col,
            include_age_at_onset=bool(disposal_include_age_at_onset),
            compliance_horizons_years=compliance_horizons_years,
        )
        disposal_onset_timeline_df = summarize_disposal_onset_timeline(
            disposal_summary_df,
            time_freq="M",
            group_by=disposal_onset_group_by,
            candidate_only=bool(disposal_onset_candidate_only),
            include_age_stats=bool(disposal_onset_include_age_stats),
        )
        n_candidates = int(disposal_summary_df['candidate_passive_decay'].fillna(False).astype(bool).sum()) if not disposal_summary_df.empty else 0
        print(f"[disposal_corridors] record_rows={len(disposal_record_df):,}, passive_candidates={n_candidates:,}")
        if effective_show_plots and plot_new_analytics and disposal_onset_timeline_df is not None and not disposal_onset_timeline_df.empty:
            fig_disp, ax_disp = plt.subplots(figsize=(10, 5))
            grouped = disposal_onset_timeline_df.groupby('group_label', sort=True)
            if len(grouped) <= 1:
                part = disposal_onset_timeline_df.sort_values('time_bin', kind='mergesort')
                ax_disp.plot(part['time_bin'], part['onset_object_count'], marker='o', color='tab:orange')
            else:
                for label, part in grouped:
                    part = part.sort_values('time_bin', kind='mergesort')
                    ax_disp.plot(part['time_bin'], part['onset_object_count'], marker='o', label=str(label), alpha=0.9)
                if disposal_onset_timeline_df['group_label'].nunique(dropna=False) <= 12:
                    ax_disp.legend(loc='best', fontsize=12)
            suffix = ""
            if disposal_onset_group_by:
                suffix = f" grouped by {disposal_onset_group_by}"
            ax_disp.set_title(f'Disposal Onset Timeline (objects/month{suffix})')
            ax_disp.set_xlabel('Time')
            ax_disp.set_ylabel('Object Count')
            fig_disp.tight_layout()
            new_analytics_figures += 1
        elapsed_disposal = perf_counter() - t_disposal
        stage_timings['disposal_corridors'] = elapsed_disposal
        stage_execution['disposal_corridors'] = {'executed': True, 'reason': None}
        print(f"[disposal_corridors] done in {elapsed_disposal:.2f}s")
    else:
        mark_stage_skipped('disposal_corridors', 'run_disposal_metrics is False')

    if run_risk_screening:
        t_risk = perf_counter()
        # Proxy risk scores are heuristic and not conjunction-grade products.
        risk_density_df, risk_crossing_df, risk_severity_df = compute_risk_screening(
            viz_df,
            altitude_bins=altitude_bins,
            time_freq=analytics_time_freq,
            include_composite_score=use_proxy_risk_score,
            return_severity_table=True,
        )
        conjunction_input_readiness = compute_conjunction_input_readiness(viz_df)
        conjunction_placeholder = compute_conjunction_grade_placeholder(viz_df)
        print(f"[risk_screening] density_rows={len(risk_density_df):,}, crossing_rows={len(risk_crossing_df):,}, "
              f"severity_rows={len(risk_severity_df):,}")
        if effective_show_plots and plot_new_analytics and not risk_crossing_df.empty:
            fig_risk, ax_risk = plt.subplots(figsize=(10, 5))
            ax2 = None
            risk_mode = str(risk_timeline_mode or 'crossing_and_score').strip().lower()

            if risk_mode == 'components':
                ax_risk.plot(
                    risk_crossing_df['time_bin'],
                    pd.to_numeric(risk_crossing_df.get('risk_component_crossing_intensity_norm', np.nan), errors='coerce'),
                    label='Crossing intensity (norm)',
                    color='tab:red',
                )
                ax_risk.plot(
                    risk_crossing_df['time_bin'],
                    pd.to_numeric(risk_crossing_df.get('risk_component_active_objects_norm', np.nan), errors='coerce'),
                    label='Active objects (norm)',
                    color='tab:green',
                    alpha=0.8,
                )
                ax_risk.plot(
                    risk_crossing_df['time_bin'],
                    pd.to_numeric(risk_crossing_df.get('risk_component_shell_overlap_norm', np.nan), errors='coerce'),
                    label='Shell overlap (norm)',
                    color='tab:purple',
                    alpha=0.8,
                )
                score_col = 'proxy_risk_score' if 'proxy_risk_score' in risk_crossing_df.columns else 'heuristic_risk_score'
                if score_col in risk_crossing_df.columns:
                    ax_risk.plot(
                        risk_crossing_df['time_bin'],
                        pd.to_numeric(risk_crossing_df.get(score_col, np.nan), errors='coerce'),
                        label='Proxy risk score',
                        color='tab:blue',
                        linewidth=2.0,
                    )
                ax_risk.set_ylabel('Normalized component / proxy score')
            elif risk_mode == 'score_only':
                score_col = 'proxy_risk_score' if 'proxy_risk_score' in risk_crossing_df.columns else 'heuristic_risk_score'
                if score_col in risk_crossing_df.columns:
                    ax_risk.plot(risk_crossing_df['time_bin'], risk_crossing_df[score_col],
                                 label='Proxy risk score', color='tab:blue')
                    ax_risk.set_ylabel('Proxy risk score')
                else:
                    ax_risk.plot(risk_crossing_df['time_bin'], risk_crossing_df['shell_crossing_intensity'],
                                 label='Shell-crossing intensity', color='tab:red')
                    ax_risk.set_ylabel('Crossing intensity')
            else:
                ax_risk.plot(risk_crossing_df['time_bin'], risk_crossing_df['shell_crossing_intensity'],
                             label='Shell-crossing intensity', color='tab:red')
                score_col = 'proxy_risk_score' if 'proxy_risk_score' in risk_crossing_df.columns else 'heuristic_risk_score'
                if score_col in risk_crossing_df.columns:
                    ax2 = ax_risk.twinx()
                    ax2.plot(risk_crossing_df['time_bin'], risk_crossing_df[score_col],
                             label='Proxy risk score', color='tab:blue', alpha=0.8)
                    ax2.set_ylabel('Proxy risk score')
                ax_risk.set_ylabel('Crossing intensity')

            ax_risk.set_title('Risk Proxy Timeline')
            ax_risk.set_xlabel('Time')
            handles, labels = ax_risk.get_legend_handles_labels()
            if ax2 is not None:
                handles2, labels2 = ax2.get_legend_handles_labels()
                handles.extend(handles2)
                labels.extend(labels2)
            if labels:
                ax_risk.legend(handles, labels, loc='best')
            fig_risk.tight_layout()
            new_analytics_figures += 1
        elapsed_risk = perf_counter() - t_risk
        stage_timings['risk_screening'] = elapsed_risk
        stage_execution['risk_screening'] = {'executed': True, 'reason': None}
        print(f"[risk_screening] done in {elapsed_risk:.2f}s")
    else:
        mark_stage_skipped('risk_screening', 'run_risk_screening is False')

    if run_maneuver_analysis:
        t_maneuver = perf_counter()
        # Build a lean analysis frame to reduce memory pressure on large archives.
        analysis_cols = ['sat_id', 'norad_cat_id', 'timestamp', 'sma', 'inc', 'ecc', 'raan',
                         'mean_motion', 'mean_motion_dot', 'ballistic_coefficient',
                         'bstar', 'drag_term', 'mean_longitude_deg', 'altitude_km']
        keep_cols = [c for c in analysis_cols if c in viz_df.columns]
        analysis_df = viz_df[keep_cols].copy()

        obj_col = 'norad_cat_id' if 'norad_cat_id' in analysis_df.columns else ('sat_id' if 'sat_id' in analysis_df.columns else None)
        object_count = int(analysis_df[obj_col].astype(str).nunique()) if obj_col is not None else 0
        print(f"[maneuver_analysis] start rows={len(analysis_df):,}, objects={object_count:,}")

        cached_events = _normalize_cached_table(
            cached_maneuver_events_df,
            date_columns=('estimated_event_time', 'event_time_lower', 'event_time_upper'),
        )
        cached_phase = _normalize_cached_table(cached_phase_df, date_columns=('timestamp',))
        cached_phase_summary = _normalize_cached_table(
            cached_phase_summary_df,
            date_columns=('phase_start', 'phase_end'),
        )

        if _has_cached_phase_tables(cached_events, cached_phase, cached_phase_summary):
            events_df = cached_events if isinstance(cached_events, pd.DataFrame) else _empty_maneuver_events_df()
            if 'object_id' not in events_df.columns:
                events_df['object_id'] = ''
            phase_df = cached_phase if isinstance(cached_phase, pd.DataFrame) else pd.DataFrame()
            if isinstance(cached_phase_summary, pd.DataFrame):
                phase_summary_df = cached_phase_summary
            elif isinstance(phase_df, pd.DataFrame) and not phase_df.empty:
                phase_summary_df = summarize_phase_intervals(phase_df)
            else:
                phase_summary_df = pd.DataFrame()
            maneuver_analysis_source = 'cache'
            cache_rows = {
                'events': int(len(events_df)),
                'phase_rows': int(len(phase_df)),
                'phase_intervals': int(len(phase_summary_df)),
            }
            print(f"[maneuver_analysis] using cached tables {cache_rows}")
        else:
            events_df = detect_maneuvers_all(analysis_df, config=maneuver_config)
            phase_df = classify_mission_phase_all(analysis_df, events_df=events_df, config=phase_config)
            phase_summary_df = summarize_phase_intervals(phase_df)
            maneuver_analysis_source = 'computed'

        # Make phase labels available for optional sustainability metrics.
        if not phase_df.empty and 'timestamp' in viz_df.columns:
            phase_small = phase_df[['object_id', 'timestamp', 'phase_state']].copy()
            phase_small['timestamp'] = pd.to_datetime(phase_small['timestamp'], errors='coerce')
            phase_small = phase_small.dropna(subset=['timestamp'])

            join_id_col = 'norad_cat_id' if 'norad_cat_id' in viz_df.columns else ('sat_id' if 'sat_id' in viz_df.columns else None)
            if join_id_col is not None:
                phase_small['object_id'] = phase_small['object_id'].astype(str)
                viz_df[join_id_col] = viz_df[join_id_col].astype(str)
                viz_df['timestamp'] = pd.to_datetime(viz_df['timestamp'], errors='coerce')
                phase_small = phase_small.rename(columns={'object_id': join_id_col})
                viz_df = viz_df.merge(phase_small, on=[join_id_col, 'timestamp'], how='left')

        print(f"[maneuver_analysis] events={len(events_df):,}, phase_rows={len(phase_df):,}, "
              f"phase_intervals={len(phase_summary_df):,}")

        if plot_maneuver_layers and not phase_df.empty:
            created_figures = 0
            object_ids = phase_df['object_id'].astype(str).unique().tolist()
            if object_ids:
                selected_object = object_ids[0]
                sat_view = analysis_df.copy()
                object_col = 'norad_cat_id' if 'norad_cat_id' in sat_view.columns else 'sat_id'
                sat_view = sat_view[sat_view[object_col].astype(str) == str(selected_object)]
                sat_events = events_df[events_df['object_id'].astype(str) == str(selected_object)]
                sat_phase = phase_df[phase_df['object_id'].astype(str) == str(selected_object)]
                sat_phase_summary = phase_summary_df[phase_summary_df['object_id'].astype(str) == str(selected_object)]

                figs = plot_altitude_inclination_with_events(
                    sat_view,
                    sat_events,
                    with_uncertainty=bool(maneuver_show_event_uncertainty),
                    event_layer_mode=maneuver_event_layer_mode,
                    event_high_confidence_threshold=float(maneuver_event_high_confidence_threshold),
                    event_color_mode=maneuver_event_color_mode,
                    phase_intervals_df=sat_phase_summary,
                )
                created_figures += len(figs)
                plot_bstar_with_events(
                    sat_view,
                    sat_events,
                    with_uncertainty=bool(maneuver_show_event_uncertainty),
                    event_layer_mode=maneuver_event_layer_mode,
                    event_high_confidence_threshold=float(maneuver_event_high_confidence_threshold),
                    event_color_mode=maneuver_event_color_mode,
                )
                created_figures += 1
                plot_phase_colored_timeseries(
                    sat_view,
                    sat_phase,
                    alpha_by_confidence=bool(maneuver_phase_alpha_by_confidence),
                    use_interval_overlay=bool(maneuver_phase_use_interval_overlay),
                )
                created_figures += 1

            plot_event_rate_histogram(
                events_df,
                object_col='object_id',
                count_basis=maneuver_event_histogram_basis,
                high_confidence_threshold=float(maneuver_event_high_confidence_threshold),
            )
            plot_shell_entry_exit_timeline(
                phase_summary_df,
                sort_mode=maneuver_timeline_sort_mode,
                confidence_shading=bool(maneuver_timeline_confidence_shading),
                zoom_inset=bool(maneuver_timeline_zoom_inset),
                zoom_start=maneuver_timeline_zoom_start,
                zoom_end=maneuver_timeline_zoom_end,
                zoom_group_window_days=float(maneuver_timeline_zoom_group_window_days),
                zoom_size_inches=float(maneuver_timeline_zoom_size_inches),
            )
            created_figures += 2

            if effective_show_plots and created_figures > 0:
                plt.show()

        elapsed_maneuver = perf_counter() - t_maneuver
        stage_timings['maneuver_analysis'] = elapsed_maneuver
        stage_execution['maneuver_analysis'] = {
            'executed': True,
            'reason': None,
            'source': maneuver_analysis_source,
            'cache_metadata': maneuver_cache_metadata if isinstance(maneuver_cache_metadata, dict) else None,
        }
        print(f"[maneuver_analysis] done in {elapsed_maneuver:.2f}s")
    else:
        mark_stage_skipped('maneuver_analysis', 'run_maneuver_analysis is False')

    if run_sustainability_metrics:
        t_sustain = perf_counter()
        sustainability_df, sustainability_shell_df = compute_sustainability_metrics(
            viz_df,
            shell_time_df=shell_time_df,
            disposal_summary_df=disposal_summary_df,
            time_freq=analytics_time_freq,
            count_basis=sustainability_count_basis,
            return_shell_summary=True,
        )
        print(f"[sustainability_metrics] rows={len(sustainability_df):,}")
        if effective_show_plots and plot_new_analytics and not sustainability_df.empty:
            fig_sus, ax_sus = plt.subplots(figsize=(10, 5))
            basis_mode = str(sustainability_count_basis or 'objects').strip().lower()
            if basis_mode == 'records':
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['active_like_count'], label='Active-like (records)')
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['disposal_like_count'], label='Disposal-like (records)')
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['drifting_like_count'], label='Drifting-like (records)')
                ax_sus.set_ylabel('Record count')
            elif basis_mode == 'both':
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['active_like_objects'], label='Active-like (objects)', color='tab:blue')
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['disposal_like_objects'], label='Disposal-like (objects)', color='tab:orange')
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['drifting_like_objects'], label='Drifting-like (objects)', color='tab:green')
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['active_like_count'], label='Active-like (records)', linestyle='--', color='tab:blue', alpha=0.6)
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['disposal_like_count'], label='Disposal-like (records)', linestyle='--', color='tab:orange', alpha=0.6)
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['drifting_like_count'], label='Drifting-like (records)', linestyle='--', color='tab:green', alpha=0.6)
                ax_sus.set_ylabel('Count')
            else:
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['active_like_objects'], label='Active-like (objects)')
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['disposal_like_objects'], label='Disposal-like (objects)')
                ax_sus.plot(sustainability_df['time_bin'], sustainability_df['drifting_like_objects'], label='Drifting-like (objects)')
                ax_sus.set_ylabel('Object Count')
            ax_sus.set_title('Sustainability State Counts Over Time')
            ax_sus.legend(loc='best')
            fig_sus.tight_layout()
            new_analytics_figures += 1
        elapsed_sustain = perf_counter() - t_sustain
        stage_timings['sustainability_metrics'] = elapsed_sustain
        stage_execution['sustainability_metrics'] = {'executed': True, 'reason': None}
        print(f"[sustainability_metrics] done in {elapsed_sustain:.2f}s")
    else:
        mark_stage_skipped('sustainability_metrics', 'run_sustainability_metrics is False')

    if effective_show_plots and plot_new_analytics and new_analytics_figures > 0:
        print(f"[new_analytics_plots] showing {new_analytics_figures} figure(s)")
        plt.show()

    total_runtime = perf_counter() - t_all
    print(f"All orbital visualizations completed in {total_runtime:.2f}s")

    shell_assignment_basis = None
    if 'shell_assignment_basis' in viz_df.columns:
        basis_counts = viz_df['shell_assignment_basis'].fillna('unknown').astype(str).value_counts(dropna=False).to_dict()
        shell_assignment_basis = basis_counts

    time_window = {"start": None, "end": None}
    if 'timestamp' in viz_df.columns and len(viz_df) > 0:
        ts = pd.to_datetime(viz_df['timestamp'], errors='coerce')
        if ts.notna().any():
            time_window = {
                "start": str(ts.min()),
                "end": str(ts.max()),
            }

    provenance = {
        'source_rows': int(len(all_tle_data)),
        'source_satellites': int(pd.Series(object_ids).astype(str).nunique()),
        'time_window': time_window,
        'filters_applied': [
            'add_low_eccentricity_flag',
            'add_altitude_features',
            'select_phase_series_true_anomaly_enforced',
            'spectral_mode_uniform_fft_enforced',
        ],
        'phase_variable': PHASE_VARIABLE_TRUE_ANOMALY,
        'phase_semantics': PHASE_SEMANTICS_TRUE_ANOMALY_PROXY,
        'shell_profile': str(starlink_shell_profile),
        'state_model': str(state_model),
        'resampling_info': {
            'spectral_mode': 'uniform_fft',
            'wavelet_method': 'wwz',
            'wavelet_irregular_policy': 'resample',
            'spectral_cadence_seconds': spectral_cadence_seconds,
            'fft_interpolation': fft_interpolation,
            'crosscorr_interpolation': crosscorr_interpolation,
            'crosscorr_normalization': crosscorr_normalization,
        },
        'code_version_hash': _best_effort_git_hash(),
    }

    results_payload = {
        'panel_enriched': viz_df,
        'provenance': provenance,
        'metadata': {
            'runtime_seconds': float(total_runtime),
            'stage_timings_seconds': stage_timings,
            'stage_execution': stage_execution,
            'state_model': state_model,
            'phase_variable': PHASE_VARIABLE_TRUE_ANOMALY,
            'phase_semantics': PHASE_SEMANTICS_TRUE_ANOMALY_PROXY,
            'show_plots': bool(effective_show_plots),
            'run_fft': bool(run_fft),
            'run_wavelet': bool(run_wavelet),
            'run_crosscorr': bool(run_crosscorr),
            'run_relative_motion': bool(run_relative_motion),
            'spectral_mode': spectral_mode,
            'wavelet_method': wavelet_method,
            'wavelet_irregular_policy': wavelet_irregular_policy,
            'run_shell_analytics': bool(run_shell_analytics),
            'run_disposal_metrics': bool(run_disposal_metrics),
            'run_sustainability_metrics': bool(run_sustainability_metrics),
            'run_risk_screening': bool(run_risk_screening),
            'maneuver_event_layer_mode': str(maneuver_event_layer_mode),
            'maneuver_event_high_confidence_threshold': float(maneuver_event_high_confidence_threshold),
            'maneuver_event_color_mode': str(maneuver_event_color_mode),
            'maneuver_show_event_uncertainty': bool(maneuver_show_event_uncertainty),
            'maneuver_event_histogram_basis': str(maneuver_event_histogram_basis),
            'maneuver_phase_use_interval_overlay': bool(maneuver_phase_use_interval_overlay),
            'maneuver_phase_alpha_by_confidence': bool(maneuver_phase_alpha_by_confidence),
            'maneuver_timeline_sort_mode': str(maneuver_timeline_sort_mode),
            'maneuver_timeline_confidence_shading': bool(maneuver_timeline_confidence_shading),
            'maneuver_timeline_zoom_inset': bool(maneuver_timeline_zoom_inset),
            'maneuver_timeline_zoom_start': maneuver_timeline_zoom_start,
            'maneuver_timeline_zoom_end': maneuver_timeline_zoom_end,
            'maneuver_timeline_zoom_group_window_days': float(maneuver_timeline_zoom_group_window_days),
            'maneuver_timeline_zoom_size_inches': float(maneuver_timeline_zoom_size_inches),
            'maneuver_analysis_source': maneuver_analysis_source,
            'maneuver_cache_metadata': maneuver_cache_metadata if isinstance(maneuver_cache_metadata, dict) else None,
            'sustainability_count_basis': str(sustainability_count_basis),
            'risk_timeline_mode': str(risk_timeline_mode),
            'disposal_family_col': None if disposal_family_col is None else str(disposal_family_col),
            'disposal_cohort_col': None if disposal_cohort_col is None else str(disposal_cohort_col),
            'disposal_include_age_at_onset': bool(disposal_include_age_at_onset),
            'disposal_onset_group_by': None if disposal_onset_group_by is None else str(disposal_onset_group_by),
            'disposal_onset_candidate_only': bool(disposal_onset_candidate_only),
            'disposal_onset_include_age_stats': bool(disposal_onset_include_age_stats),
            'starlink_shell_profile': starlink_shell_profile,
            'shell_profile_semantics': 'Nominal shell-profile labeling only (descriptive/proxy; not conjunction truth).',
            'shell_assignment_basis': shell_assignment_basis,
            'shell_refine_with_inclination': bool(shell_refine_with_inclination),
            'inclination_time_render_mode': str(inclination_time_render_mode),
            'inclination_reference_lines': bool(inclination_reference_lines),
            'inclination_reference_assignment_tolerance_deg': float(inclination_reference_assignment_tolerance_deg),
            'inclination_reference_annotation': bool(inclination_reference_annotation),
            'precession_group_targets_deg': None if precession_group_targets_deg is None else [
                float(v) for v in np.asarray(precession_group_targets_deg, dtype=np.float64) if np.isfinite(v)
            ],
            'precession_group_assignment_tolerance_deg': float(precession_group_assignment_tolerance_deg),
            'precession_group_trend_mode': str(precession_group_trend_mode),
            'precession_group_rolling_window_days': None if precession_group_rolling_window_days is None else float(precession_group_rolling_window_days),
            'precession_show_group_envelopes': bool(precession_show_group_envelopes),
            'precession_apsidal_ecc_floor': float(precession_apsidal_ecc_floor),
            'precession_low_e_behavior': str(precession_low_e_behavior),
            'raan_time_render_mode': str(raan_time_render_mode),
            'raan_display_mode': str(raan_display_mode),
            'raan_residual_fit_mode': str(raan_residual_fit_mode),
            'raan_residual_annotation': bool(raan_residual_annotation),
            'raan_min_points_for_unwrap': int(raan_min_points_for_unwrap),
            'sma_time_render_mode': str(sma_time_render_mode),
            'sma_reference_lines': bool(sma_reference_lines),
            'sma_reference_values_km': None if sma_reference_values_km is None else [
                float(v) for v in np.asarray(sma_reference_values_km, dtype=np.float64) if np.isfinite(v)
            ],
            'sma_rolling_window_days': None if sma_rolling_window_days is None else float(sma_rolling_window_days),
            'sma_show_envelope': bool(sma_show_envelope),
            'sma_operational_bands': sma_operational_bands,
            'argp_time_render_mode': str(argp_time_render_mode),
            'argp_ecc_floor': float(argp_ecc_floor),
            'argp_low_e_behavior': str(argp_low_e_behavior),
            'argp_show_argument_of_latitude_companion': bool(argp_show_argument_of_latitude_companion),
            'eccentricity_time_render_mode': str(eccentricity_time_render_mode),
            'eccentricity_highlight_descents': bool(eccentricity_highlight_descents),
            'eccentricity_descent_threshold_per_day': float(eccentricity_descent_threshold_per_day),
            'eccentricity_descent_min_duration_days': float(eccentricity_descent_min_duration_days),
            'eccentricity_descent_min_eccentricity': float(eccentricity_descent_min_eccentricity),
            'eccentricity_rolling_window_days': None if eccentricity_rolling_window_days is None else float(eccentricity_rolling_window_days),
            'eccentricity_show_envelope': bool(eccentricity_show_envelope),
            'eccentricity_zoom_ylim': eccentricity_zoom_ylim,
            'inc_sma_render_mode': str(inc_sma_render_mode),
            'inc_sma_metric_mode': str(inc_sma_metric_mode),
            'inc_sma_reference_lines': bool(inc_sma_reference_lines),
            'inc_sma_reference_annotation': bool(inc_sma_reference_annotation),
            'inc_sma_reference_markers': bool(inc_sma_reference_markers),
            'inc_sma_target_sma_tolerance_km': float(inc_sma_target_sma_tolerance_km),
            'inc_sma_target_inclination_tolerance_deg': float(inc_sma_target_inclination_tolerance_deg),
            'inc_sma_focused_profiles': bool(inc_sma_focused_profiles),
            'enable_sma_inclination_density_shell_map': bool(enable_sma_inclination_density_shell_map),
            'enable_common_epoch_shell_snapshot': bool(enable_common_epoch_shell_snapshot),
            'enable_starlink_dynamical_atlas': bool(enable_starlink_dynamical_atlas),
            'wwz_extract_peaks': bool(wwz_extract_peaks),
            'wwz_peak_top_k': int(wwz_peak_top_k),
            'wwz_peak_min_prominence': wwz_peak_min_prominence,
            'wwz_peak_min_separation_tau_bins': int(wwz_peak_min_separation_tau_bins),
            'wwz_peak_min_separation_period_bins': int(wwz_peak_min_separation_period_bins),
            'wwz_overlay_ridge': bool(wwz_overlay_ridge),
            'wwz_annotate_panels': bool(wwz_annotate_panels),
            'wwz_min_effective_n': wwz_min_effective_n,
            'wwz_export_combined_summary': bool(wwz_export_combined_summary),
            'occupancy_heatmap_time_axis_mode': str(occupancy_heatmap_time_axis_mode),
            'occupancy_heatmap_y_label_mode': str(occupancy_heatmap_y_label_mode),
            'occupancy_heatmap_norm': str(occupancy_heatmap_norm),
            'occupancy_heatmap_clip_percentile': occupancy_heatmap_clip_percentile,
            'occupancy_heatmap_use_pcolormesh': bool(occupancy_heatmap_use_pcolormesh),
            'occupancy_heatmap_smoothing_sigma': occupancy_heatmap_smoothing_sigma,
            'raan_phase_density_mode': str(raan_phase_density_mode),
            'raan_phase_density_family_mode': str(raan_phase_density_family_mode),
            'raan_phase_density_family_targets_deg': raan_phase_density_family_targets_deg,
            'raan_phase_density_family_tolerance_deg': float(raan_phase_density_family_tolerance_deg),
            'raan_phase_density_time_windows': raan_phase_density_time_windows,
            'raan_phase_density_rolling_window_days': raan_phase_density_rolling_window_days,
            'raan_phase_density_return_arrays': bool(raan_phase_density_return_arrays),
            'raan_phase_density_compute_uniformity_metrics': bool(raan_phase_density_compute_uniformity_metrics),
            'phase_alt_density_mode': str(phase_alt_density_mode),
            'phase_alt_density_family_mode': str(phase_alt_density_family_mode),
            'phase_alt_density_family_targets_deg': phase_alt_density_family_targets_deg,
            'phase_alt_density_family_tolerance_deg': float(phase_alt_density_family_tolerance_deg),
            'phase_alt_density_time_windows': phase_alt_density_time_windows,
            'phase_alt_density_rolling_window_days': phase_alt_density_rolling_window_days,
            'phase_alt_density_step_days': phase_alt_density_step_days,
            'phase_alt_density_top_k_hotspots': int(phase_alt_density_top_k_hotspots),
            'phase_alt_density_return_arrays': bool(phase_alt_density_return_arrays),
            'phase_alt_density_normalization': str(phase_alt_density_normalization),
            'phase_alt_density_overlay_altitude_refs_km': phase_alt_density_overlay_altitude_refs_km,
            'compliance_horizons_years': compliance_horizons_years,
            'include_proxy_risk_score': bool(use_proxy_risk_score),
            'relative_pair_mode': str(relative_pair_mode),
            'relative_max_pairs': relative_max_pairs,
            'relative_same_shell_only': bool(relative_same_shell_only),
            'resonance_proximity_only': True,
            'capture_not_proven': True,
        },
        'analytics': {
            'resonance_diag_df': resonance_diag_df,
            'resonance_object_summary': resonance_object_summary,
            'resonance_ai_map_df': resonance_ai_map_df,
            'shell_time_df': shell_time_df,
            'shell_entry_exit_df': shell_entry_exit_df,
            'shell_heatmap_df': shell_heatmap_df,
            'shell_transition_df': shell_transition_df,
            'shell_transition_matrix_df': shell_transition_matrix_df,
            'shell_snapshot_df': shell_snapshot_df,
            'disposal_record_df': disposal_record_df,
            'disposal_summary_df': disposal_summary_df,
            'disposal_onset_timeline_df': disposal_onset_timeline_df,
            'risk_density_df': risk_density_df,
            'risk_crossing_df': risk_crossing_df,
            'risk_severity_df': risk_severity_df,
            'conjunction_input_readiness': conjunction_input_readiness,
            'conjunction_placeholder': conjunction_placeholder,
            'sustainability_df': sustainability_df,
            'sustainability_shell_df': sustainability_shell_df,
            'fft_results': fft_results_payload,
            'crosscorr_results': crosscorr_results_payload,
            'wavelet_results': wavelet_results_payload,
            'occupancy_heatmap_results': occupancy_heatmap_payload,
            'raan_phase_density_results': raan_phase_density_payload,
            'phase_alt_density_results': phase_alt_density_payload,
            'relative_motion_results': relative_motion_payload,
            'inc_versus_sma_results': inc_sma_results_payload,
            'precession_results': precession_results_payload,
            'events_df': events_df,
            'phase_df': phase_df,
            'phase_summary_df': phase_summary_df,
        },
    }

    if return_results:
        return results_payload
    return None