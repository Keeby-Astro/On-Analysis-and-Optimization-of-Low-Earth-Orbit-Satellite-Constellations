"""
thrust_helpers.py — Per-satellite Hall-thruster schedule helpers.

Provides:
  - Thruster calibration loading (JSON)
  - Maneuver phase interval loading (CSV)
  - Per-case schedule construction (clip, convert, coast-fill)
  - Numba-compiled along-track thrust math
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G0_M_S2 = 9.80665  # standard gravity, m/s^2

# Default data file locations (relative to *this* module's parent directory)
_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_THRUSTER_JSON = _MODULE_DIR.parent / "outputs" / "latest" / "reports" / "thruster_performance_summary.json"
_DEFAULT_PHASE_CSV = _MODULE_DIR.parent / "full_exports" / "maneuver_phase_intervals_gen1_full.csv"

_SCHEDULE_POLICY_KEYS = (
    'cluster_policy_applied',
    'cluster_policy_source',
    'policy_tau_keep_s',
    'policy_deadband_a_km',
    'policy_deadband_lambda_rad',
    'policy_reserve_prop_frac',
    'policy_disposal_altitude_trigger_km',
    'target_a_km',
    'target_raan_rad',
    'target_mean_anomaly_rad',
    'target_lambda_rad',
)


# ---------------------------------------------------------------------------
# Satellite ID normalisation
# ---------------------------------------------------------------------------
def canonical_sat_id(raw_id):
    """Normalise a satellite identifier for matching TLE ↔ CSV.

    Strips whitespace, lowercases, removes a trailing '.txt' suffix.
    Example: "sat1008.txt" → "sat1008"
    """
    s = str(raw_id).strip().lower()
    if s.endswith('.txt'):
        s = s[:-4]
    return s


# ---------------------------------------------------------------------------
# Thruster calibration loading
# ---------------------------------------------------------------------------
def load_thruster_defaults(json_path=None):
    """Load thruster calibration JSON and return a structured dict.

    Returns dict with keys:
        fitted_parameters  – dict with mass_kg, dry_mass_kg, isp_s, eta_total
        per_phase          – dict keyed by phase_state string, each value is a dict
                             with duty_cycle, mean_thrust_N, mean_isp_s,
                             mean_efficiency, sign
    """
    if json_path is None:
        json_path = _DEFAULT_THRUSTER_JSON
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Thruster JSON not found: {json_path}")

    with open(json_path, 'r') as fh:
        raw = json.load(fh)

    fitted = raw['fitted_parameters']

    per_phase = {}
    # Map from JSON section name → (phase_state key, thrust sign)
    _phase_map = {
        'insertion_or_orbit_raise': ('insertion_or_orbit_raise', +1),
        'operational_shell': ('operational_shell', +1),
        'disposal_lowering': ('disposal_lowering', -1),
    }
    for json_key, (phase_key, sign) in _phase_map.items():
        section = raw['per_phase'][json_key]
        per_phase[phase_key] = {
            'duty_cycle': float(section['duty_cycle']),
            'mean_thrust_N': float(section['mean_thrust_N']),
            'mean_isp_s': float(section['mean_isp_s']),
            'mean_efficiency': float(section['mean_efficiency']),
            'sign': sign,
        }

    return {
        'fitted_parameters': {
            'mass_kg': float(fitted['mass_kg']),
            'dry_mass_kg': float(fitted['dry_mass_kg']),
            'isp_s': float(fitted['isp_s']),
            'eta_total': float(fitted['eta_total']),
        },
        'per_phase': per_phase,
    }


# ---------------------------------------------------------------------------
# Phase-interval CSV loading
# ---------------------------------------------------------------------------
def load_phase_intervals(csv_path=None):
    """Load maneuver_phase_intervals CSV, parse timestamps, normalise sat_ids.

    Returns a pandas DataFrame with columns:
        sat_id_canonical, phase_state, phase_start (datetime64), phase_end (datetime64)
    Rows with zero-length intervals (start == end) are dropped.
    """
    if csv_path is None:
        csv_path = _DEFAULT_PHASE_CSV
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Phase-interval CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Defensive column selection — accept slight naming variants
    col_map = {}
    for needed, variants in [('sat_id', ['sat_id', 'sat_name', 'satellite_id']),
                              ('phase_state', ['phase_state', 'phase', 'maneuver_phase']),
                              ('phase_start', ['phase_start', 'start', 'interval_start']),
                              ('phase_end', ['phase_end', 'end', 'interval_end'])]:
        found = None
        for v in variants:
            if v in df.columns:
                found = v
                break
        if found is None:
            raise KeyError(f"Cannot find column for '{needed}' in CSV.  "
                           f"Available: {list(df.columns)}")
        col_map[needed] = found

    df = df.rename(columns={col_map[k]: k for k in col_map if col_map[k] != k})
    df['phase_start'] = pd.to_datetime(df['phase_start'], utc=False)
    df['phase_end'] = pd.to_datetime(df['phase_end'], utc=False)
    df['sat_id_canonical'] = df['sat_id'].apply(canonical_sat_id)

    # Drop zero-length intervals (start == end)
    df = df[df['phase_start'] < df['phase_end']].copy()

    # Sort for deterministic schedule building
    df = df.sort_values(['sat_id_canonical', 'phase_start']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Phase parameter map
# ---------------------------------------------------------------------------
def build_phase_parameter_map(thruster_defaults):
    """Build a dict mapping phase_state → control scalars.

    Returns dict[str, dict] with keys:
        T_eff_N  – continuous-equivalent effective thrust (N)
        Isp_s    – specific impulse (s)
        eta      – total efficiency (dimensionless)
        sign     – +1 for raising/maintenance, -1 for disposal
        duty_cycle
    """
    result = {}
    for phase_key, pdata in thruster_defaults['per_phase'].items():
        T_eff = pdata['duty_cycle'] * pdata['mean_thrust_N']
        result[phase_key] = {
            'T_eff_N': T_eff,
            'Isp_s': pdata['mean_isp_s'],
            'eta': pdata['mean_efficiency'],
            'sign': pdata['sign'],
            'duty_cycle': pdata['duty_cycle'],
        }
    return result


def _attach_policy_context(segment, policy_context=None):
    """Attach selected policy keys to a schedule segment."""
    if not policy_context:
        return segment
    for key in _SCHEDULE_POLICY_KEYS:
        if key in policy_context:
            segment[key] = policy_context[key]
    return segment


# ---------------------------------------------------------------------------
# Per-case schedule construction
# ---------------------------------------------------------------------------
def build_case_schedule(sat_id, intervals_df, phase_params,
                        case_start_ts, t_final_s, policy_context=None):
    """Build a chronological segment list for one satellite case.

    Parameters
    ----------
    sat_id : str
        Raw satellite ID (will be canonicalised internally).
    intervals_df : DataFrame
        Output of load_phase_intervals().
    phase_params : dict
        Output of build_phase_parameter_map().
    case_start_ts : pd.Timestamp
        Absolute start time for this propagation case.
    t_final_s : float
        Duration of propagation in seconds.

    Returns
    -------
    list[dict]
        Chronological segments, each with keys:
            t0_s, t1_s  – seconds since case_start_ts
            phase        – phase_state string or 'coast'
            T_eff_N, Isp_s, eta, sign  – control scalars (0 for coast)
    """
    canon = canonical_sat_id(sat_id)
    case_start = pd.Timestamp(case_start_ts)
    case_end = case_start + pd.Timedelta(seconds=t_final_s)

    # Filter to this satellite
    mask = intervals_df['sat_id_canonical'] == canon
    sat_df = intervals_df.loc[mask].copy()

    if sat_df.empty:
        # No schedule data — entire propagation is coast
        return [_attach_policy_context(_coast_segment(0.0, t_final_s), policy_context)]

    # Clip intervals to propagation window
    sat_df = sat_df[(sat_df['phase_end'] > case_start) &
                    (sat_df['phase_start'] < case_end)].copy()

    if sat_df.empty:
        return [_attach_policy_context(_coast_segment(0.0, t_final_s), policy_context)]

    sat_df['clip_start'] = sat_df['phase_start'].clip(lower=case_start)
    sat_df['clip_end'] = sat_df['phase_end'].clip(upper=case_end)

    # Convert to seconds since case_start
    sat_df['t0_s'] = (sat_df['clip_start'] - case_start).dt.total_seconds()
    sat_df['t1_s'] = (sat_df['clip_end'] - case_start).dt.total_seconds()

    # Drop intervals that became zero-length after clipping
    sat_df = sat_df[sat_df['t1_s'] - sat_df['t0_s'] > 0.0].copy()
    sat_df = sat_df.sort_values('t0_s').reset_index(drop=True)

    if sat_df.empty:
        return [_attach_policy_context(_coast_segment(0.0, t_final_s), policy_context)]

    segments = []
    cursor = 0.0

    for _, row in sat_df.iterrows():
        t0 = float(row['t0_s'])
        t1 = float(row['t1_s'])
        phase = str(row['phase_state']).strip().lower()

        # Insert coast gap before this interval if needed
        if t0 > cursor + 0.5:  # tolerance for tiny gaps
            segments.append(_attach_policy_context(_coast_segment(cursor, t0), policy_context))

        pdata = phase_params.get(phase)
        if pdata is not None:
            segments.append(_attach_policy_context({
                't0_s': t0,
                't1_s': t1,
                'phase': phase,
                'T_eff_N': pdata['T_eff_N'],
                'Isp_s': pdata['Isp_s'],
                'eta': pdata['eta'],
                'sign': pdata['sign'],
            }, policy_context))
        else:
            # Unknown phase → treat as coast
            segments.append(_attach_policy_context(_coast_segment(t0, t1), policy_context))

        cursor = t1

    # Trailing coast if schedule ends before t_final
    if cursor < t_final_s - 0.5:
        segments.append(_attach_policy_context(_coast_segment(cursor, t_final_s), policy_context))

    return segments


def _coast_segment(t0, t1):
    """Return a zero-thrust coast segment dict."""
    return {
        't0_s': float(t0),
        't1_s': float(t1),
        'phase': 'coast',
        'T_eff_N': 0.0,
        'Isp_s': 0.0,
        'eta': 0.0,
        'sign': 0,
    }


# ---------------------------------------------------------------------------
# Schedule boundary helpers
# ---------------------------------------------------------------------------
def get_schedule_boundaries(schedule, t_final_s=None, msis_day_boundaries=None):
    """Return sorted unique boundary times from a segment schedule.

    Merges phase boundaries, MSIS day boundaries, and [0, t_final].

    Parameters
    ----------
    schedule : list[dict]
        Output of build_case_schedule().
    t_final_s : float or None
        End time; if None, inferred from schedule.
    msis_day_boundaries : array-like or None
        Additional boundary times (e.g. MSIS daily grid change points).

    Returns
    -------
    np.ndarray
        Sorted unique boundary times in seconds.
    """
    bds = set()
    bds.add(0.0)
    for seg in schedule:
        bds.add(seg['t0_s'])
        bds.add(seg['t1_s'])
    if t_final_s is not None:
        bds.add(float(t_final_s))
    if msis_day_boundaries is not None:
        for b in msis_day_boundaries:
            bds.add(float(b))
    arr = np.array(sorted(bds), dtype=np.float64)
    return arr


def lookup_segment_for_time(schedule, t_s):
    """Return the schedule segment active at time *t_s* (seconds).

    Falls back to coast parameters if no segment covers *t_s*.

    Returns
    -------
    dict with T_eff_N, Isp_s, eta, sign, phase
    """
    for seg in schedule:
        if seg['t0_s'] <= t_s < seg['t1_s']:
            return seg
    # Fallback — coast
    return _coast_segment(t_s, t_s)


def _wrap_angle_rad(angle_rad):
    """Wrap an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def resolve_segment_command(segment, *, current_a_km=None, current_lambda_rad=None,
                            current_alt_km=None, current_mass_kg=None,
                            dry_mass_kg=None, initial_mass_kg=None,
                            segment_start_s=None, controller_state=None):
    """Resolve the live thrust command for one schedule segment.

    The historical phase schedule still selects the active mode, but the
    optimized policy terms can now modulate the command at runtime:
        - `deadband_a` / `deadband_lambda`: disable or reverse thrust in
          `operational_shell` when the current state is inside/outside the
          target deadbands.
        - `tau_keep`: minimum dwell time before the operational-shell command
          is allowed to switch sign or switch off.
        - `reserve_prop_frac`: preserves a propellant reserve for non-disposal
          phases.
        - `disposal_altitude_trigger`: turns disposal thrust off once the
          current altitude is below the trigger.
    """
    cmd = dict(segment)
    phase = str(cmd.get('phase', 'coast'))
    T_eff = float(cmd.get('T_eff_N', 0.0) or 0.0)
    sign = int(cmd.get('sign', 0) or 0)

    if T_eff <= 0.0:
        cmd['T_eff_N'] = 0.0
        cmd['sign'] = 0
        return cmd

    if current_mass_kg is not None and dry_mass_kg is not None and initial_mass_kg is not None:
        reserve_frac = float(cmd.get('policy_reserve_prop_frac', 0.0) or 0.0)
        usable_prop = max(float(initial_mass_kg) - float(dry_mass_kg), 0.0)
        reserve_mass = float(dry_mass_kg) + reserve_frac * usable_prop
        if phase != 'disposal_lowering' and float(current_mass_kg) <= reserve_mass + 1.0e-9:
            cmd['T_eff_N'] = 0.0
            cmd['sign'] = 0
            cmd['reserve_mass_kg'] = reserve_mass
            return cmd
        cmd['reserve_mass_kg'] = reserve_mass

    if phase == 'disposal_lowering':
        trigger_alt = cmd.get('policy_disposal_altitude_trigger_km')
        if trigger_alt is not None and current_alt_km is not None:
            if float(current_alt_km) <= float(trigger_alt):
                cmd['T_eff_N'] = 0.0
                cmd['sign'] = 0
                return cmd

    if phase == 'operational_shell' and bool(cmd.get('cluster_policy_applied', False)):
        target_a = cmd.get('target_a_km')
        target_lambda = cmd.get('target_lambda_rad')
        if target_a is not None and target_lambda is not None and current_a_km is not None and current_lambda_rad is not None:
            deadband_a = max(float(cmd.get('policy_deadband_a_km', 5.0) or 5.0), 1.0e-6)
            deadband_lambda = max(float(cmd.get('policy_deadband_lambda_rad', 0.01) or 0.01), 1.0e-6)
            a_err = float(current_a_km) - float(target_a)
            lambda_err = _wrap_angle_rad(float(current_lambda_rad) - float(target_lambda))

            if abs(a_err) <= deadband_a and abs(lambda_err) <= deadband_lambda:
                desired_sign = 0
            else:
                a_score = abs(a_err) / deadband_a
                lambda_score = abs(lambda_err) / deadband_lambda
                if a_score >= lambda_score:
                    desired_sign = -1 if a_err > 0.0 else +1
                else:
                    desired_sign = -1 if lambda_err > 0.0 else +1

            if controller_state is not None:
                tau_keep = max(float(cmd.get('policy_tau_keep_s', 0.0) or 0.0), 0.0)
                previous_sign = int(controller_state.get('active_sign', 0) or 0)
                last_change_t = float(controller_state.get('last_change_t_s', -1.0e30) or -1.0e30)
                if (segment_start_s is not None and tau_keep > 0.0 and desired_sign != previous_sign and
                        float(segment_start_s) - last_change_t < tau_keep):
                    desired_sign = previous_sign
                if desired_sign != previous_sign:
                    controller_state['last_change_t_s'] = float(segment_start_s or 0.0)
                controller_state['active_sign'] = int(desired_sign)
                controller_state['last_a_err_km'] = a_err
                controller_state['last_lambda_err_rad'] = lambda_err

            if desired_sign == 0:
                cmd['T_eff_N'] = 0.0
                cmd['sign'] = 0
                return cmd

            sign = int(desired_sign)

    cmd['T_eff_N'] = float(T_eff)
    cmd['sign'] = int(sign)
    return cmd


# ---------------------------------------------------------------------------
# Numba-compiled thrust math
# ---------------------------------------------------------------------------
@njit(cache=True)
def tangential_unit_from_state(r_vec, v_vec):
    """Compute along-track (tangential) unit vector in RTN frame.

    Parameters
    ----------
    r_vec : array (3,) — position vector (km)
    v_vec : array (3,) — velocity vector (km/s)

    Returns
    -------
    t_hat : array (3,) — unit tangential direction in inertial frame
    """
    # r_hat
    r_mag = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
    if r_mag < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    r_hat = r_vec / r_mag

    # h = r × v
    hx = r_vec[1]*v_vec[2] - r_vec[2]*v_vec[1]
    hy = r_vec[2]*v_vec[0] - r_vec[0]*v_vec[2]
    hz = r_vec[0]*v_vec[1] - r_vec[1]*v_vec[0]
    h_mag = np.sqrt(hx*hx + hy*hy + hz*hz)
    if h_mag < 1.0e-12:
        return np.zeros(3, dtype=np.float64)

    h_hat_x = hx / h_mag
    h_hat_y = hy / h_mag
    h_hat_z = hz / h_mag

    # t_hat = h_hat × r_hat
    t_hat = np.empty(3, dtype=np.float64)
    t_hat[0] = h_hat_y*r_hat[2] - h_hat_z*r_hat[1]
    t_hat[1] = h_hat_z*r_hat[0] - h_hat_x*r_hat[2]
    t_hat[2] = h_hat_x*r_hat[1] - h_hat_y*r_hat[0]

    t_mag = np.sqrt(t_hat[0]**2 + t_hat[1]**2 + t_hat[2]**2)
    if t_mag < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    t_hat[0] /= t_mag
    t_hat[1] /= t_mag
    t_hat[2] /= t_mag
    return t_hat


@njit(cache=True)
def thrust_accel_and_mdot(r_vec, v_vec, T_eff_N, Isp_s, sign,
                          mass_kg, dry_mass_kg):
    """Compute along-track thrust acceleration and mass flow rate.

    Parameters
    ----------
    r_vec     : array (3,) — satellite position (km)
    v_vec     : array (3,) — satellite velocity (km/s)
    T_eff_N   : float — continuous-equivalent thrust (N)
    Isp_s     : float — specific impulse (s)
    sign      : int   — +1 raising/maintenance, -1 disposal
    mass_kg   : float — current spacecraft mass (kg)
    dry_mass_kg : float — minimum mass (kg)

    Returns
    -------
    ax, ay, az : float — thrust acceleration components (km/s^2)
    dm_dt      : float — mass depletion rate (kg/s, always ≤ 0)
    """
    # Guard: no thrust if at or below dry mass, or zero T_eff
    if T_eff_N <= 0.0 or mass_kg <= dry_mass_kg:
        return 0.0, 0.0, 0.0, 0.0

    t_hat = tangential_unit_from_state(r_vec, v_vec)

    # Acceleration magnitude: T_eff / m  [N / kg = m/s^2]
    # Convert to km/s^2: multiply by 1e-3
    a_mag = (T_eff_N / mass_kg) * 1.0e-3  # km/s^2

    ax = sign * a_mag * t_hat[0]
    ay = sign * a_mag * t_hat[1]
    az = sign * a_mag * t_hat[2]

    # Mass depletion: dm/dt = -T_eff / (g0 * Isp)  [kg/s]
    g0 = 9.80665  # m/s^2
    dm_dt = -T_eff_N / (g0 * Isp_s)

    return ax, ay, az, dm_dt


# ---------------------------------------------------------------------------
# Segment-level summary accumulators
# ---------------------------------------------------------------------------
def compute_case_thrust_summary(schedule, times, mass_series):
    """Compute per-case impulse, energy, and phase-time summaries.

    Parameters
    ----------
    schedule : list[dict]
        Output of build_case_schedule().
    times : np.ndarray
        Time array in seconds (from the propagation).
    mass_series : np.ndarray
        Spacecraft mass at each time point (kg).

    Returns
    -------
    dict with summary scalars.
    """
    initial_mass = float(mass_series[0]) if mass_series.size > 0 else 0.0
    final_mass = float(mass_series[-1]) if mass_series.size > 0 else 0.0
    propellant_used = initial_mass - final_mass

    total_impulse_Ns = 0.0
    total_energy_Wh = 0.0
    total_thrust_on_s = 0.0
    total_raise_s = 0.0
    total_shell_s = 0.0
    total_disposal_s = 0.0

    for seg in schedule:
        dt_seg = seg['t1_s'] - seg['t0_s']
        T_eff = seg['T_eff_N']
        Isp_s = seg['Isp_s']
        eta = seg['eta']

        if T_eff > 0.0 and dt_seg > 0.0:
            total_impulse_Ns += T_eff * dt_seg
            total_thrust_on_s += dt_seg

            # P_in = T_eff * g0 * Isp / (2 * eta)   [W]
            if eta > 0.0 and Isp_s > 0.0:
                P_in_W = T_eff * G0_M_S2 * Isp_s / (2.0 * eta)
                total_energy_Wh += P_in_W * dt_seg / 3600.0

        phase = seg.get('phase', 'coast')
        if phase == 'insertion_or_orbit_raise':
            total_raise_s += dt_seg
        elif phase == 'operational_shell':
            total_shell_s += dt_seg
        elif phase == 'disposal_lowering':
            total_disposal_s += dt_seg

    return {
        'initial_mass_kg': initial_mass,
        'final_mass_kg': final_mass,
        'propellant_used_kg': propellant_used,
        'cumulative_impulse_Ns': total_impulse_Ns,
        'cumulative_energy_Wh': total_energy_Wh,
        'total_thrust_on_time_s': total_thrust_on_s,
        'total_raise_time_s': total_raise_s,
        'total_shell_time_s': total_shell_s,
        'total_disposal_time_s': total_disposal_s,
    }
