"""Relative-motion analysis entrypoint for synchronized satellite pairs.

- exact_lvlh: nonlinear chief-centered LVLH kinematics
- hcw: circular-chief Clohessy-Wiltshire linearized model
- yamanaka_ankersen: elliptical-chief linearized STM-style interface

Frame discipline:
- Relative states require a common state frame.
- If requested state generation yields mixed frames (for example TEME from SGP4
    and proxy_inertial from classical fallback), this module reconciles by
    regenerating both initial states with the classical model and records that
    choice in payload metadata.

References:
- Clohessy and Wiltshire (1960)
- Yamanaka and Ankersen (2002)
- Astropy TEME docs: https://docs.astropy.org/en/latest/api/astropy.coordinates.TEME.html
- sgp4 package docs: https://pypi.org/project/sgp4/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import warnings
from matplotlib.widgets import Slider
from orbital_mechanics import (
    relative_motion_exact_lvlh,
    relative_motion_hcw,
    relative_motion_yamanaka_ankersen,
    rv_from_r0v0,
)
from time import perf_counter
from state_models import DEFAULT_STATE_MODEL, state_from_row

DEFAULT_EPOCH_TOLERANCE_SECONDS = 300
DEFAULT_RELATIVE_MODEL = "exact_lvlh"
SUPPORTED_RELATIVE_MODELS = (
    "exact_lvlh",
    "hcw",
    "yamanaka_ankersen",
    "th",
    "ya",
)

def _compute_period_seconds(row, mu):
    """Compute orbital period in seconds from sma when available, else from h/e."""
    sma = row.get('sma', np.nan)
    if np.isfinite(sma) and sma > 0.0:
        return 2.0 * np.pi * np.sqrt((sma ** 3) / mu)

    h = row.get('specific_angular_momentum', np.nan)
    e = row.get('ecc', np.nan)
    if np.isfinite(h) and h > 0.0 and np.isfinite(e) and 0.0 <= e < 1.0:
        return (2.0 * np.pi * (h ** 3)) / (mu ** 2 * (1.0 - e ** 2) ** 1.5)

    return np.nan


def _normalize_relative_model(relative_model):
    value = DEFAULT_RELATIVE_MODEL if relative_model is None else str(relative_model).strip().lower()
    aliases = {
        "exact": "exact_lvlh",
        "lvlh": "exact_lvlh",
        "cw": "hcw",
        "cwh": "hcw",
        "th": "yamanaka_ankersen",
        "ya": "yamanaka_ankersen",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_RELATIVE_MODELS:
        raise ValueError(f"Unsupported relative_model='{relative_model}'. Supported: {SUPPORTED_RELATIVE_MODELS}")
    if value in ("th", "ya"):
        return "yamanaka_ankersen"
    return value

def _coerce_timestamp_ns(values):
    ts = pd.to_datetime(values, errors='coerce').to_numpy(dtype='datetime64[ns]')
    ns = ts.astype('int64')
    valid = ns != np.iinfo(np.int64).min
    return ns, valid


def _base_error_payload(message, *, state_model=None, relative_model=None, sat_pair=None):
    payload = {
        'error': message,
        'rel': np.empty((0, 3), dtype=np.float64),
        'orbit_A': np.empty((0, 3), dtype=np.float64),
        'orbit_B': np.empty((0, 3), dtype=np.float64),
        'extent': 1.0,
        'label': 'No synchronized epoch',
    }
    if state_model is not None:
        payload['requested_state_model'] = state_model
    if relative_model is not None:
        payload['relative_model'] = relative_model
    if sat_pair is not None:
        payload['satellite_pair'] = sat_pair
    return payload

def _find_nearest_common_epoch_pair(sat_a_df, sat_b_df, tolerance_seconds=DEFAULT_EPOCH_TOLERANCE_SECONDS):
    """Find nearest synchronized rows for a satellite pair within tolerance."""
    if sat_a_df.empty or sat_b_df.empty:
        return None

    a_ns, valid_a = _coerce_timestamp_ns(sat_a_df['timestamp'])
    b_ns, valid_b = _coerce_timestamp_ns(sat_b_df['timestamp'])
    if not np.any(valid_a) or not np.any(valid_b):
        return None

    a_idx_valid = np.flatnonzero(valid_a)
    b_idx_valid = np.flatnonzero(valid_b)
    a_ns_valid = a_ns[a_idx_valid]
    b_ns_valid = b_ns[b_idx_valid]

    order_b = np.argsort(b_ns_valid)
    b_ns_sorted = b_ns_valid[order_b]
    b_idx_sorted = b_idx_valid[order_b]

    tol_ns = int(float(tolerance_seconds) * 1e9)
    best = None

    for pos_a, t_a in enumerate(a_ns_valid):
        insert_pos = np.searchsorted(b_ns_sorted, t_a)
        candidates = []
        if insert_pos < b_ns_sorted.size:
            candidates.append(insert_pos)
        if insert_pos > 0:
            candidates.append(insert_pos - 1)

        for cand_pos in candidates:
            t_b = b_ns_sorted[cand_pos]
            delta_ns = abs(int(t_a) - int(t_b))
            if delta_ns > tol_ns:
                continue

            candidate = {
                'row_a_idx': int(a_idx_valid[pos_a]),
                'row_b_idx': int(b_idx_sorted[cand_pos]),
                'delta_ns': int(delta_ns),
                'epoch_ns': int((int(t_a) + int(t_b)) // 2),
                'a_ns': int(t_a),
                'b_ns': int(t_b),
            }
            if best is None:
                best = candidate
                continue

            # Deterministic tie-break for duplicate timestamps.
            cand_key = (candidate['delta_ns'], candidate['epoch_ns'], candidate['row_a_idx'], candidate['row_b_idx'])
            best_key = (best['delta_ns'], best['epoch_ns'], best['row_a_idx'], best['row_b_idx'])
            if cand_key < best_key:
                best = candidate

    if best is None:
        return None

    row_a = sat_a_df.iloc[best['row_a_idx']]
    row_b = sat_b_df.iloc[best['row_b_idx']]
    epoch = np.datetime64(best['epoch_ns'], 'ns')
    delta_seconds = best['delta_ns'] / 1e9
    metadata = {
        'pair_tolerance_seconds': float(tolerance_seconds),
        'delta_seconds': float(delta_seconds),
        'row_a_idx': int(best['row_a_idx']),
        'row_b_idx': int(best['row_b_idx']),
        'duplicate_safe_tiebreak': True,
        'a_timestamp_ns': int(best['a_ns']),
        'b_timestamp_ns': int(best['b_ns']),
    }
    return row_a, row_b, epoch, delta_seconds, metadata


def relative_motion(
    all_tle_data,
    state_model=DEFAULT_STATE_MODEL,
    show_plots=True,
    return_results=False,
    relative_model=DEFAULT_RELATIVE_MODEL,
    n_periods=10,
    samples_per_period=100,
    max_duration_seconds=None,
    tolerance_seconds=DEFAULT_EPOCH_TOLERANCE_SECONDS,
    pair_list=None,
):
    """
    Function to visualize the relative motion between two satellites using matplotlib sliders.
    This keeps the legacy default behavior and adds an optional state-model switch.
    
    Parameters:
        all_tle_data (pd.DataFrame): DataFrame containing TLE-like data (including orbital elements).
                                     Required columns:
                                     ['sat_id', 'specific_angular_momentum', 'ecc', 'inc', 
                                      'raan', 'aop', 'true_anomaly']
    Additional Parameters:
        relative_model (str): exact_lvlh, hcw, yamanaka_ankersen (aliases: th, ya).
        n_periods (int): Number of chief periods to simulate.
        samples_per_period (int): Temporal samples per chief period.
        max_duration_seconds (float | None): Optional cap on propagation horizon.
        tolerance_seconds (float): Epoch synchronization tolerance.
        pair_list (list[tuple] | None): Optional list of (satA_id, satB_id) pairs for
                                        batch headless evaluation.

    Returns:
        None by default. Returns a payload dictionary when show_plots=False
        or when return_results=True.
    """
    t0 = perf_counter()
    print("[relative_motion] Preparing satellite records...")

    model_name = _normalize_relative_model(relative_model)

    if n_periods <= 0:
        raise ValueError("n_periods must be positive")
    if samples_per_period <= 1:
        raise ValueError("samples_per_period must be greater than 1")
    if tolerance_seconds <= 0:
        raise ValueError("tolerance_seconds must be positive")

    sat_data = all_tle_data.copy()
    sat_data['timestamp'] = pd.to_datetime(sat_data['timestamp'], errors='coerce')
    sat_data = sat_data.dropna(subset=['sat_id', 'timestamp'])
    sat_data = sat_data.sort_values(['sat_id', 'timestamp'], kind='mergesort')

    sat_groups = {sat_id: grp.reset_index(drop=True) for sat_id, grp in sat_data.groupby('sat_id', sort=True)}
    unique_sat_ids = np.array(list(sat_groups.keys()))
    if len(unique_sat_ids) < 2:
        message = "Need at least two satellites for relative motion visualization."
        print(message)
        if not show_plots:
            return _base_error_payload(
                message,
                state_model=state_model,
                relative_model=model_name,
            )
        return
    
    fig = None
    ax = None
    slider_satA = None
    slider_satB = None
    line_rel = None
    line_a = None
    line_b = None
    last_payload = None

    if show_plots:
        # Create the plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create sliders for selecting satellites
        axcolor = 'lightgoldenrodyellow'
        ax_satA = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor=axcolor)
        ax_satB = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor=axcolor)

        # Sliders for satellite selection
        slider_satA = Slider(ax_satA, 'Satellite A', 0, len(unique_sat_ids) - 1, valinit=0, valstep=1)
        slider_satB = Slider(ax_satB, 'Satellite B', 0, len(unique_sat_ids) - 1, valinit=1, valstep=1)

    # Gravitational parameter (km^3/s^2)
    mu = 398600.4418

    trajectory_cache = {}

    def _cleanup_cache():
        trajectory_cache.clear()
        gc.collect()

    def calculate_orbit(r0, v0, mu, period_seconds):
        n = int(samples_per_period)
        dt = period_seconds / n
        orbit = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            t = (i + 1) * dt
            r, _ = rv_from_r0v0(r0, v0, t, mu)
            orbit[i, :] = r
        return orbit

    def _resolve_states_with_frame_policy(satA_record, satB_record, common_epoch):
        rA0, vA0, meta_a = state_from_row(satA_record, epoch=common_epoch, mu=mu, state_model=state_model)
        rB0, vB0, meta_b = state_from_row(satB_record, epoch=common_epoch, mu=mu, state_model=state_model)

        frame_a = meta_a.get('state_frame', 'unknown')
        frame_b = meta_b.get('state_frame', 'unknown')
        fallback_info = {
            'frame_reconciled': False,
            'frame_policy': 'same_frame_required',
            'frame_reconcile_reason': None,
            'initial_frame_a': frame_a,
            'initial_frame_b': frame_b,
            'explicit_frame_transform_applied': False,
        }

        if frame_a == frame_b:
            return rA0, vA0, meta_a, rB0, vB0, meta_b, fallback_info

        fallback_info['frame_reconciled'] = True
        fallback_info['frame_reconcile_reason'] = 'frame_mismatch_auto_fallback_to_classical'
        warnings.warn(
            "relative_motion: mixed state frames detected (for example TEME and proxy_inertial) "
            "without explicit transform; falling back to classical/proxy_inertial for consistency.",
            RuntimeWarning,
            stacklevel=2,
        )
        rA0_c, vA0_c, meta_a_c = state_from_row(satA_record, epoch=common_epoch, mu=mu, state_model='classical')
        rB0_c, vB0_c, meta_b_c = state_from_row(satB_record, epoch=common_epoch, mu=mu, state_model='classical')

        meta_a_c = dict(meta_a_c)
        meta_b_c = dict(meta_b_c)
        meta_a_c['frame_reconciled_from'] = frame_a
        meta_b_c['frame_reconciled_from'] = frame_b
        meta_a_c['fallback_used'] = True
        meta_b_c['fallback_used'] = True

        return rA0_c, vA0_c, meta_a_c, rB0_c, vB0_c, meta_b_c, fallback_info

    def _run_relative_model(rA0, vA0, rB0, vB0, times):
        if model_name == 'exact_lvlh':
            return relative_motion_exact_lvlh(rA0, vA0, rB0, vB0, times, mu, return_diagnostics=True)
        if model_name == 'hcw':
            return relative_motion_hcw(rA0, vA0, rB0, vB0, times, mu, return_diagnostics=True)
        return relative_motion_yamanaka_ankersen(rA0, vA0, rB0, vB0, times, mu, return_diagnostics=True)

    def compute_trajectory_pair(satA_id, satB_id):
        pair_key = (satA_id, satB_id)
        if pair_key in trajectory_cache:
            print(f"[relative_motion] Cache hit for pair ({satA_id}, {satB_id})")
            return trajectory_cache[pair_key]

        t_pair = perf_counter()
        print(f"[relative_motion] Computing pair ({satA_id}, {satB_id})...")

        sat_a_df = sat_groups[satA_id]
        sat_b_df = sat_groups[satB_id]

        sync = _find_nearest_common_epoch_pair(sat_a_df, sat_b_df, tolerance_seconds=tolerance_seconds)
        if sync is None:
            message = (f"No common epoch found within {tolerance_seconds}s for pair "
                       f"({satA_id}, {satB_id}); refusing to compute unsynchronized relative motion.")
            payload = _base_error_payload(
                message,
                state_model=state_model,
                relative_model=model_name,
                sat_pair=(satA_id, satB_id),
            )
            trajectory_cache[pair_key] = payload
            print(f"[relative_motion] {message}")
            return payload

        satA_record, satB_record, common_epoch, delta_seconds, sync_meta = sync

        try:
            rA0, vA0, meta_a, rB0, vB0, meta_b, frame_policy = _resolve_states_with_frame_policy(
                satA_record,
                satB_record,
                common_epoch,
            )
        except Exception as exc:
            message = f"Unable to construct initial states for pair ({satA_id}, {satB_id}): {exc}"
            payload = _base_error_payload(
                message,
                state_model=state_model,
                relative_model=model_name,
                sat_pair=(satA_id, satB_id),
            )
            trajectory_cache[pair_key] = payload
            print(f"[relative_motion] {message}")
            return payload

        if meta_a.get('fallback_used') or meta_b.get('fallback_used'):
            print('[relative_motion] state_model=sgp4_preferred fell back to classical for this pair.')

        used_a = meta_a.get('state_model_used', 'unknown')
        used_b = meta_b.get('state_model_used', 'unknown')
        frame_a = meta_a.get('state_frame', 'unknown')
        frame_b = meta_b.get('state_frame', 'unknown')
        if frame_a != frame_b:
            message = (
                f"Frame reconciliation failed for pair ({satA_id}, {satB_id}): "
                f"A={frame_a}, B={frame_b}"
            )
            payload = _base_error_payload(
                message,
                state_model=state_model,
                relative_model=model_name,
                sat_pair=(satA_id, satB_id),
            )
            trajectory_cache[pair_key] = payload
            print(f"[relative_motion] {message}")
            return payload

        model_label = (f"requested={state_model}; used A={used_a}, B={used_b}; "
                       f"frame A={frame_a}, B={frame_b}")

        period_a = _compute_period_seconds(satA_record, mu)
        period_b = _compute_period_seconds(satB_record, mu)
        if np.isfinite(period_a) and period_a > 0.0:
            period = period_a
        elif np.isfinite(period_b) and period_b > 0.0:
            period = period_b
        else:
            message = f"Unable to determine orbital period for pair ({satA_id}, {satB_id})."
            payload = _base_error_payload(
                message,
                state_model=state_model,
                relative_model=model_name,
                sat_pair=(satA_id, satB_id),
            )
            payload['label'] = model_label
            trajectory_cache[pair_key] = payload
            print(f"[relative_motion] {message}")
            return payload

        sim_duration = float(period * float(n_periods))
        if max_duration_seconds is not None:
            sim_duration = min(sim_duration, float(max_duration_seconds))

        times = np.linspace(0.0, sim_duration, int(n_periods) * int(samples_per_period), endpoint=False)
        model_payload = _run_relative_model(rA0, vA0, rB0, vB0, times)
        rel_xyz = np.asarray(model_payload['r_rel_lvlh'], dtype=np.float64)

        orbit_A = calculate_orbit(rA0, vA0, mu, period)
        orbit_B = calculate_orbit(rB0, vB0, mu, period)

        max_extent = np.max(np.abs(np.vstack((rel_xyz, orbit_A, orbit_B))))
        if max_extent <= 0:
            max_extent = 1.0

        sgp4_seeded = (
            meta_a.get('source') == 'sgp4_tle'
            or meta_b.get('source') == 'sgp4_tle'
        )
        propagation_note = (
            'initial_states_from_sgp4_then_two_body_propagation'
            if sgp4_seeded
            else 'initial_states_from_classical_then_two_body_propagation'
        )

        payload = {
            'rel': rel_xyz,
            'orbit_A': orbit_A,
            'orbit_B': orbit_B,
            'extent': max_extent,
            'state_meta_a': meta_a,
            'state_meta_b': meta_b,
            'relative_model': model_name,
            'relative_model_diagnostics': model_payload,
            'sync_metadata': sync_meta,
            'frame_policy': frame_policy,
            'chief_id': satA_id,
            'deputy_id': satB_id,
            'label': (
                f"{model_label}; rel_model={model_name}; nearest-epoch delta={delta_seconds:.2f}s "
                f"at {pd.Timestamp(common_epoch)}"
            ),
            'propagation_assumption': propagation_note,
            'requested_state_model': state_model,
            'satellite_pair': (satA_id, satB_id),
        }
        trajectory_cache[pair_key] = payload
        print(f"[relative_motion] Pair ({satA_id}, {satB_id}) ready in {perf_counter() - t_pair:.2f}s")
        return payload

    if show_plots:
        line_rel, = ax.plot([], [], [], label='Relative Motion (B wrt A)', linewidth=1.5, color='m')
        line_a, = ax.plot([], [], [], 'r', linewidth=1.5, label='Satellite A Orbit')
        line_b, = ax.plot([], [], [], 'b', linewidth=1.5, label='Satellite B Orbit')
        ax.set_title('Relative Motion between Satellites')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.legend()
        ax.grid(True)
        ax.set_box_aspect([1, 1, 1])

    def update_relative_motion_plot():
        """Update the 3D relative-motion plot for current slider satellite selections."""
        nonlocal last_payload
        satA_id = unique_sat_ids[int(slider_satA.val)]
        satB_id = unique_sat_ids[int(slider_satB.val)]

        if satA_id == satB_id:
            print("Please select two different satellites.")
            return

        t_update = perf_counter()
        data = compute_trajectory_pair(satA_id, satB_id)
        if 'error' in data:
            line_rel.set_data([], [])
            line_rel.set_3d_properties([])
            line_a.set_data([], [])
            line_a.set_3d_properties([])
            line_b.set_data([], [])
            line_b.set_3d_properties([])
            ax.set_title(f"Relative Motion between Satellites\n{data['error']}")
            fig.canvas.draw_idle()
            last_payload = data
            return

        rel_xyz = data['rel']
        orbit_A = data['orbit_A']
        orbit_B = data['orbit_B']

        line_rel.set_data(rel_xyz[:, 0], rel_xyz[:, 1])
        line_rel.set_3d_properties(rel_xyz[:, 2])
        line_a.set_data(orbit_A[:, 0], orbit_A[:, 1])
        line_a.set_3d_properties(orbit_A[:, 2])
        line_b.set_data(orbit_B[:, 0], orbit_B[:, 1])
        line_b.set_3d_properties(orbit_B[:, 2])

        extent = data['extent']
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_zlim(-extent, extent)
        ax.set_title(f"Relative Motion between Satellites\n{data['label']}")

        fig.canvas.draw_idle()
        last_payload = data
        print(f"[relative_motion] Updated view in {perf_counter() - t_update:.2f}s")
    
    if show_plots:
        # Update function for sliders
        def update(val):
            update_relative_motion_plot()

        # Connect sliders to the update function
        slider_satA.on_changed(update)
        slider_satB.on_changed(update)

        # Initial plot
        update_relative_motion_plot()
        print(f"[relative_motion] Ready in {perf_counter() - t0:.2f}s")

        # Show the interactive plot
        plt.show()
        out = last_payload if return_results else None
        _cleanup_cache()
        return out

    if pair_list is not None:
        results = []
        for satA_id, satB_id in pair_list:
            if satA_id not in sat_groups or satB_id not in sat_groups:
                results.append(
                    _base_error_payload(
                        f"Unknown satellite IDs in pair ({satA_id}, {satB_id})",
                        state_model=state_model,
                        relative_model=model_name,
                        sat_pair=(satA_id, satB_id),
                    )
                )
                continue
            if satA_id == satB_id:
                results.append(
                    _base_error_payload(
                        f"Pair ({satA_id}, {satB_id}) must contain two distinct satellites",
                        state_model=state_model,
                        relative_model=model_name,
                        sat_pair=(satA_id, satB_id),
                    )
                )
                continue
            results.append(compute_trajectory_pair(satA_id, satB_id))

        payload = {
            'results': results,
            'requested_state_model': state_model,
            'relative_model': model_name,
            'show_plots': False,
            'batch_mode': True,
        }
        print(f"[relative_motion] Noninteractive batch payload ready in {perf_counter() - t0:.2f}s")
        _cleanup_cache()
        return payload

    satA_id = unique_sat_ids[0]
    satB_id = unique_sat_ids[1]
    payload = compute_trajectory_pair(satA_id, satB_id)
    print(f"[relative_motion] Noninteractive payload ready in {perf_counter() - t0:.2f}s")
    _cleanup_cache()
    return payload