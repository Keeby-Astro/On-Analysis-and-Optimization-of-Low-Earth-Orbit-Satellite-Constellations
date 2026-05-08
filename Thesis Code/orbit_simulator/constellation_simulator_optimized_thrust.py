# Orbit simulator
#
# In this program, a satellite orbit around the Earth is simulated considering the
# the Earth as the central body and the perturbations from the Sun, the Moon,
# the solar radiation pressure, and the gravitational effects of the
# Earth's spherical harmonics.
#
# Author: Diogo Merguizo Sanchez
#         The University of Oklahoma
#         2024
#
# Input - initial orbital elements: a, initial semi-major axis (km)
#                                   e, initial eccentricity
#                                   i, initial inclination (deg)
#                                   omega, initial argument of perigee (deg)
#                                   Omega, initial RAAN (deg)
#                                   Ma, initial mean anomaly (deg)
#                                   JD0, initial Julian date for the natural bodies' initial conditions
#                                   tu, time unit: 0: seconds, 1: minutes, 2: hours, 3: days, 4: years
#                                   tf, final time (tu)
#                                   t_step, time step (tu)
#
# Output - orbital elements propagated over time (a, e, i, omega, Omega, Ma)
#          state vector (x, y, z, vx, vy, vz) propagated over time

# Libraries
import numpy as np
import pandas as pd
import timeit
from scipy.integrate import solve_ivp
import os
import multiprocessing as mp
from pathlib import Path
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from numba import njit
from datetime import datetime, timedelta, date # Added for MSIS date arithmetic
import json as _json

# Self-made libraries
from control_optimized import pause
from load_all_tle_data import load_all_tle_data

# Hall-thruster helpers (schedule building, Numba math)
from thrust_helpers import (
    canonical_sat_id, load_thruster_defaults, load_phase_intervals,
    build_phase_parameter_map, build_case_schedule,
    get_schedule_boundaries, lookup_segment_for_time, resolve_segment_command,
    tangential_unit_from_state, thrust_accel_and_mdot,
    compute_case_thrust_summary, G0_M_S2,
)

# Import the constants from the file "major_bodies_parameters.py"
from major_bodies_parameters_optimized import constants

# Import the major bodies' initial conditions
from major_bodies_optimized import load_mb

# Import the function to convert orbital elements to Cartesian coordinates
from functions_optimized import orb2xyz

# Import the function to convert Cartesian coordinates to orbital elements
from functions_optimized import xyz2orb

# Import the function to change the reference plane (ecliptic to equatorial - if needed)
from functions_optimized import ecl2equ

# Import the function to calculate the GST at a given epoch
from functions_optimized import gst0

# Import function with the perturbations acting on the satellite
from perturbation_optimized import SRPacc
from perturbation_optimized import AC3b
from perturbation_optimized import EGM2008
from perturbation_optimized import atm_drag

# Optional: precomputed NRLMSIS grid atmosphere
from msis_optimized import load_meta as msis_load_meta
from msis_optimized import MsisGridIndex as MsisGridIndex
# Removed: memmap_grid, parse_date, date_add_days (replaced by standard lib and load_grid)
from msis_optimized import load_grid as msis_load_grid
from msis_optimized import atm_drag_msis_grid

# Import function to calculate the Earth's J2 (oblateness) perturbation on the Moon
from perturbation_optimized import J2acc

# On Windows spawn, child workers re-import this module as __mp_main__.
# Use this flag to avoid expensive import-time work in worker processes.
_IS_MAIN_PROCESS = mp.current_process().name == "MainProcess"
########################################################################
########################################################################
########################################################################
#----------------------### Initial conditions ###-----------------------

import matplotlib.pyplot as plt
# UPDATE FIGURE SETTINGS & DEFINE CUSTOM PALETTE
plt.rcParams.update({'figure.figsize': (9.9, 7.5),
                     'xtick.direction': 'in', 'xtick.labelsize': 14, 'xtick.major.size': 3,
                     'xtick.major.width': 0.5, 'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5,
                     'xtick.minor.visible': True, 'xtick.top': True,
                     'ytick.direction': 'in', 'ytick.labelsize': 14, 'ytick.major.size': 3,
                     'ytick.major.width': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
                     'ytick.minor.visible': True, 'ytick.right': True,
                     'axes.linewidth': 0.5, 'grid.linewidth': 0.5, 'lines.linewidth': 1.0,
                     'legend.fontsize': 14, 'legend.frameon': False,
                     'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 12, 'axes.labelsize': 16, 'axes.titlesize': 18,
                     'axes.grid': True, 'grid.linestyle': '--', 'grid.color': '0.5',
                     'lines.markersize': 8, 'axes.spines.top': True, 'axes.spines.right': True,
                     'agg.path.chunksize': 10000})

# Define the custom 20-color palette (darkened colors)
colors = ['#15528e', '#b25800', '#1e701e', '#951c1c', '#673284', 
          '#623c34', '#9e5387', '#585858', '#848417', '#108590',
          '#798ba2', '#b28254', '#6a9c60', '#b26a68', '#8a7b94',
          '#896d67', '#ac7f93', '#8b8b8b', '#999962', '#6f989f']

fd = 'optimization_final' # folder where the results are stored

# Satellite's initial conditions

central_body = 'earth' # central body
frame = 'earth' # Options: Earth's mean equator [earth] or ecliptic [ecliptic] reference frame


a = 550 + 6378.1366 # a, initial semi-major axis (km)
e = 0.05  # e, initial eccentricity
i = 56.06  # i, initial inclination (deg)
w = 0.0  # w, initial argument of perigee (deg)
OM = 0.0  # OM, initial RAAN (deg)
Ma = 0.0  # Ma, initial mean anomaly (deg)
epoch = '2023-07-01'  # initial epoch (yyyy-mm-dd) for the natural bodies' initial conditions

# TLE-based initialization
# 0: use manual orbital elements above
# 1: load initial orbital elements from TLE files
use_tle_initial_conditions = 1

# TLE source and selection controls
_default_tle_data_dir = Path(__file__).resolve().parent.parent / "starlink_backup"
tle_data_folders = [str(_default_tle_data_dir)]
tle_only_files = None          # Example: ['sat1010_decay.txt', 'sat1053_decay.txt']s
tle_satellite_limit = 5       # 0: all satellites found, N>0: first N satellites (sorted by sat_id)

# Global calendar cutoffs for propagation window
tle_earliest_start_epoch = '2019-10-01'
simulation_date_cutoff = '2035-01-01'

tu = 2 # tu, time unit: 0: seconds, 1: minutes, 2: hours, 3: days, 4: years
# 2 and half years equals days of 
tf = 365 * 5  # tf, final time (tu)
t_step = 0.01  # t_step, time step (tu)

# Batch simulation controls (parallel Monte Carlo)
batch_mode = 1  # 0: single run (default), 1: run many simulations in parallel
num_simulations = 32  # used when batch_mode = 1
max_parallel_workers = 12  # tuned from batch benchmark
batch_random_seed = 42

# 1-sigma Gaussian dispersion around nominal ballistic coefficient for batch_mode
batch_sigma_ballistic_coef = 0.0  # m^2/kg

# Batch post-processing cutoff altitude (km)
batch_cutoff_alt_km = 115.0

# Save full trajectories for each batch case (expensive for 100s-1000s)
save_batch_trajectories = 0

########################################################################
# Solver configuration (overridable per batch via run_batch_cases kwargs)
########################################################################
_solver_rtol: float = 1.e-10
_solver_atol: float = 1.e-12
_max_prop_time_s: float | None = None  # None = use _seconds_until_date_cutoff
_output_stride: int = 1  # Thin t_eval by this factor (1=full, 100=every 100th pt)

########################################################################
# Global cluster integration
########################################################################
# 0: disable cluster features entirely (recover current behavior)
# 1: enable cluster-aware analysis, plotting, and optional medoid pooling
enable_global_cluster_features = 1

# Options: "labels" (run all active sats, attach cluster metadata)
#          "medoids" (reduce to one representative per active cluster)
cluster_run_mode = "labels"

# Path to global cluster assignment table
_base_dir_cluster = Path(__file__).resolve().parent.parent
cluster_assignments_csv = str(_base_dir_cluster / "global_analysis" / "tables" / "combined_cluster_labels_global.csv")

# Optional stats table for extra reporting only
cluster_stats_csv = str(_base_dir_cluster / "global_analysis" / "tables" / "global_cluster_stats.csv")

# Apply stitched cluster-policy results from the latest optimization study.
# The currently implemented runtime effect is the orbital-element offset
# triplet (delta_a, delta_Omega, delta_lambda); the remaining policy terms
# are attached to metadata and outputs for downstream control-law use.
enable_optimized_cluster_policy_defaults = 1
optimized_cluster_policy_csv = str(_base_dir_cluster / "optimization_outputs" / "Data" / "stitched_policy_table.csv")
optimized_cluster_overview_csv = str(_base_dir_cluster / "optimization_outputs" / "Data" / "cluster_overview.csv")

# In medoid mode:
# 0 = respect tle_satellite_limit and existing active-set filters, compute medoids from that active set
# 1 = override: build medoids from all loaded satellites after TLE loading
cluster_pool_ignore_satellite_limit = 0

# Keep noise points as direct runs in medoid mode
cluster_keep_noise_unpooled = 1

# Plot styling for cluster-aware outputs
cluster_noise_color = "#8b8b8b"
cluster_palette_name = "tab20"
cluster_alpha_individual = 0.45
cluster_linewidth_individual = 0.7
cluster_summary_quantiles = (0.10, 0.50, 0.90)

# Module-level cluster data (populated during TLE init if cluster features enabled)
_CLUSTER_ASSIGNMENTS_CACHE = {}
_CLUSTER_POLICY_CACHE = {}
cluster_metadata_by_sat = {}
cluster_policy_by_cluster = {}

_CLUSTER_BASE_OUTPUT_FIELDS = [
    'global_cluster_id',
    'cluster_weight_active',
    'cluster_weight_global',
    'pooled_role',
    'is_cluster_noise',
    'representative_sat_id',
    'cluster_color_hex',
]

_CLUSTER_POLICY_OUTPUT_FIELDS = [
    'cluster_policy_applied',
    'cluster_policy_source',
    'policy_delta_a_km',
    'policy_delta_Omega_rad',
    'policy_delta_lambda_rad',
    'policy_tau_keep_s',
    'policy_deadband_a_km',
    'policy_deadband_lambda_rad',
    'policy_reserve_prop_frac',
    'policy_disposal_altitude_trigger_km',
    'policy_total_cost',
    'policy_propellant_kg',
]

_CLUSTER_OUTPUT_FIELDS = _CLUSTER_BASE_OUTPUT_FIELDS + _CLUSTER_POLICY_OUTPUT_FIELDS

_CLUSTER_POLICY_DEFAULTS = {
    'policy_delta_a_km': 0.0,
    'policy_delta_Omega_rad': 0.0,
    'policy_delta_lambda_rad': 0.0,
    'policy_tau_keep_s': 86400.0,
    'policy_deadband_a_km': 5.0,
    'policy_deadband_lambda_rad': 0.01,
    'policy_reserve_prop_frac': 0.10,
    'policy_disposal_altitude_trigger_km': 300.0,
}

_CLUSTER_POLICY_COLUMN_MAP = [
    ('delta_a', 'policy_delta_a_km'),
    ('delta_Omega', 'policy_delta_Omega_rad'),
    ('delta_lambda', 'policy_delta_lambda_rad'),
    ('tau_keep', 'policy_tau_keep_s'),
    ('deadband_a', 'policy_deadband_a_km'),
    ('deadband_lambda', 'policy_deadband_lambda_rad'),
    ('reserve_prop_frac', 'policy_reserve_prop_frac'),
    ('disposal_altitude_trigger', 'policy_disposal_altitude_trigger_km'),
]

# Satellite properties for SRP calculation
frontal_area = 30  # rectangular cross-sectional area in m^2
mass = 260.0  # kg
AtoM = frontal_area / mass  # m^2/kg
Cr = 1.5 # reflectivity coefficient (1.0 <= ref_co <= 2.0)

# Satellite properties for atmospheric drag calculation
Cd = 2.2 # drag coefficient
# Use the Model B combined posterior mean beta instead of the geometry-only Cd*A/m estimate.
ballistic_coefficient_nominal = 0.0334047233755809  # posterior mean beta (Cd*A/m), m^2/kg

# Earth's spherical harmonics (gravity model)
nmax = 2
mmax = 0

### Perturbations ###
# Flags to turn on/off the perturbations
# Solar radiation pressure
k_SRP = 1 # 1: on, 0: off
# Moon's perturbation
k_moon = 1 # 1: on, 0: off
# Sun's perturbation
k_sun = 1 # 1: on, 0: off
# Earth's spherical harmonics
k_EGM2008 = 1 # 1: on, 0: off
# Atmospheric drag
k_atm_drag = 1 # 1: on, 0: off

### Hall-thruster configuration ###
# 0: no thrust (classic coast-only propagation, 18-state)
# 1: per-satellite phase-conditioned thrust (19-state with variable mass)
k_thrust = 1

# Paths for thruster calibration and phase-interval data
_base_dir_thrust = Path(__file__).resolve().parent.parent
thrust_json_path = str(_base_dir_thrust / "outputs" / "latest" / "reports" / "thruster_performance_summary.json")
phase_csv_path   = str(_base_dir_thrust / "full_exports" / "maneuver_phase_intervals_gen1_full.csv")

# Plotting flags for thrust diagnostics
plot_thrust_mag   = 1  # thrust mag vs time: 0: no, 1: yes
plot_mass         = 1  # spacecraft mass vs time: 0: no, 1: yes
plot_impulse      = 1  # cumulative impulse vs time: 0: no, 1: yes
plot_power        = 1  # electrical power vs time: 0: no, 1: yes
plot_phase_overlay = 1 # SMA with phase-colored background: 0: no, 1: yes

# Atmosphere model selection for drag
# 0: USSA76
# 1: NRLMSIS
atm_model = 1

# Precomputed NRLMSIS grid configuration (used when atm_model=1)
if atm_model == 1:
    # Directory containing grid_meta.txt + grid_index.csv + rho_*.bin(.zst)
    # Override without editing this file by setting environment variable MSIS_GRID_DIR.
    _default_msis_grid_dir = Path(__file__).resolve().parent.parent / "full_out"
    msis_grid_dir = os.getenv("MSIS_GRID_DIR", str(_default_msis_grid_dir))
    msis_grid_start_date = '2019-10-01'  # date for t=0 (yyyy-mm-dd)

    # Validate the grid directory only when MSIS drag is actually used.
    if k_atm_drag == 1:
        _candidates = [Path(msis_grid_dir), Path(__file__).resolve().parent / "full_out",
                       Path(__file__).resolve().parent.parent / "full_out", Path.cwd() / "full_out"]
        _picked = None
        for _cand in _candidates:
            try:
                if (_cand / "grid_meta.txt").is_file():
                    _picked = _cand
                    break
            except OSError:
                # Ignore invalid paths and keep trying fallbacks.
                pass

        if _picked is None:
            tried = "\n".join(f"- {p}" for p in _candidates)
            raise FileNotFoundError("MSIS grid metadata not found (grid_meta.txt).\n"
                                    "Set MSIS_GRID_DIR to the folder containing grid_meta.txt and grid_index.csv,\n"
                                    "or ensure a full_out folder exists at the workspace root (../full_out relative to this script).\n"
                                    f"Tried:\n{tried}")

        msis_grid_dir = str(_picked)

########################################################################
# Thruster and phase-interval data loading (main process only)
########################################################################
thruster_config = None
phase_intervals_df = None
phase_param_map = None
thrust_initial_mass_kg = mass  # fallback to satellite config mass (kg)
thrust_dry_mass_kg = 150.0     # fallback

if k_thrust == 1 and _IS_MAIN_PROCESS:
    try:
        thruster_config = load_thruster_defaults(thrust_json_path)
        thrust_initial_mass_kg = thruster_config['fitted_parameters']['mass_kg']
        thrust_dry_mass_kg = thruster_config['fitted_parameters']['dry_mass_kg']
        phase_param_map = build_phase_parameter_map(thruster_config)
        print(f"[Thrust] Loaded thruster calibration: "
              f"mass={thrust_initial_mass_kg:.2f} kg, "
              f"dry={thrust_dry_mass_kg:.2f} kg, "
              f"Isp={thruster_config['fitted_parameters']['isp_s']:.1f} s")
    except Exception as _e:
        print(f"[Thrust] WARNING: Could not load thruster JSON ({_e}). "
              f"Using fallback mass values.")
        # Build hardcoded fallback
        phase_param_map = {
            'insertion_or_orbit_raise': {'T_eff_N': 0.4458 * 0.02714, 'Isp_s': 1499.76,
                                         'eta': 0.4752, 'sign': +1, 'duty_cycle': 0.4458},
            'operational_shell':       {'T_eff_N': 0.0446 * 0.001634, 'Isp_s': 1499.76,
                                         'eta': 0.4752, 'sign': +1, 'duty_cycle': 0.0446},
            'disposal_lowering':       {'T_eff_N': 0.343 * 0.013505, 'Isp_s': 1499.76,
                                         'eta': 0.4752, 'sign': -1, 'duty_cycle': 0.343},
        }
    try:
        phase_intervals_df = load_phase_intervals(phase_csv_path)
        _n_sats_csv = phase_intervals_df['sat_id_canonical'].nunique()
        print(f"[Thrust] Loaded phase intervals: "
              f"{len(phase_intervals_df)} rows, {_n_sats_csv} satellites")
    except Exception as _e:
        print(f"[Thrust] WARNING: Could not load phase CSV ({_e}). "
              f"All satellites will coast (no thrust schedule).")
        phase_intervals_df = None

########################################################################
################## Control output and data analysis ####################
# Flags to plot the results
plot_sv = 0 # plot the state vector: 0: no, 1: yes
plot_sv_3d_sat = 0 # plot the state vector for satellite in 3D: 0: no, 1: yes
plot_sv_3d_sun = 0 # plot the state vector for the Sun in 3D: 0: no, 1: yes 
plot_rp_ra = 1 # plot the perigee and apogee altitudes vs. time: 0: no, 1: yes
plot_sma = 1 # plot the semi-major axis vs. time: 0: no, 1: yes
plot_ecc = 1 # plot the eccentricity vs. time: 0: no, 1: yes
save_plots = 1 # save generated plots into fd: 0: no, 1: yes
show_plots = 1 # display generated plots interactively: 0: no, 1: yes
plot_dpi = 600
# Batch time axis mode: 0 = relative time per case, 1 = absolute days since tle_earliest_start_epoch
plot_batch_absolute_time = 1
#
# Flags to save the results
save_mb_sv = 1  # save major bodies' state vector: 0: no, 1: yes

# Save satellite's orbital elements
save_sat_oe = 1  # save satellite's orbital elements: 0: no, 1: yes
#
########################################################################
########################################################################
########################################################################
if tu == 0:
    tu_conv = 1.0
    unit = 'seconds'
elif tu == 1:
    tu_conv = 60.0
    unit = 'minutes'
elif tu == 2:
    tu_conv = 3600.0
    unit = 'hours'
elif tu == 3:
    tu_conv = 86400.0
    unit = 'days'
elif tu == 4:
    tu_conv = 86400.0 * 365.25
    unit = 'years'

tf = tf * tu_conv
dt = t_step * tu_conv

nt = round(tf / dt)
tspan = np.arange(0.0, tf + dt, dt)

def _finalize_plot(filename, fig=None):
    if fig is None:
        fig = plt.gcf()

    if save_plots == 1:
        os.makedirs(fd, exist_ok=True)
        fig.savefig(f'{fd}/{filename}', dpi=plot_dpi, bbox_inches='tight')

    if show_plots == 1:
        plt.show()

    plt.close(fig)


def _extract_guidance_state(y_state):
    """Return (a_km, lambda_rad, altitude_km) from the current segment state."""
    sat_state = np.asarray(y_state[12:18], dtype=np.float64)
    oe_now = xyz2orb(earth_GM, sat_state[0:3], sat_state[3:6])
    a_now = float(oe_now[0])
    lambda_now = float((oe_now[3] + oe_now[4] + oe_now[5]) % (2.0 * np.pi))
    altitude_now = float(np.linalg.norm(sat_state[0:3]) - earth_Re)
    return a_now, lambda_now, altitude_now

########################################################################
# Cluster helper functions
########################################################################

def _normalize_cluster_sat_id(sat_id_value):
    """Normalize a satellite identifier to a canonical filename key.

    Lowercase, basename only, converts ``_decay.txt`` / ``_decay`` to
    ``.txt`` and appends ``.txt`` when missing.
    """
    s = str(sat_id_value).strip().lower()
    s = os.path.basename(s)
    if s.endswith("_decay.txt"):
        s = s[:-10] + ".txt"
    elif s.endswith("_decay"):
        s = s[:-6] + ".txt"
    elif not s.endswith(".txt"):
        s = s + ".txt"
    return s


def _resolve_cluster_assignments_csv_path(csv_path_value):
    """Resolve the cluster assignment CSV, trying local dir then parent."""
    csv_path = Path(csv_path_value)
    if csv_path.is_absolute() and csv_path.is_file():
        return csv_path
    if csv_path.is_absolute():
        return csv_path  # let caller fail with FileNotFoundError
    candidate_local = (Path(__file__).resolve().parent / csv_path).resolve()
    if candidate_local.is_file():
        return candidate_local
    candidate_workspace = (Path(__file__).resolve().parent.parent / csv_path).resolve()
    return candidate_workspace


def _load_cluster_assignments_map(csv_path_value):
    """Load cluster CSV and return ``{sat_key: global_cluster_id}`` dict.

    File-stat based caching avoids re-reads within the same process.
    """
    csv_path = _resolve_cluster_assignments_csv_path(csv_path_value)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cluster assignment CSV not found: {csv_path}")

    cache_key = str(csv_path)
    stat = csv_path.stat()
    sig = (int(stat.st_mtime_ns), int(stat.st_size))
    cached = _CLUSTER_ASSIGNMENTS_CACHE.get(cache_key)
    if cached is not None and cached.get("sig") == sig:
        return cached.get("map", {})

    df = pd.read_csv(csv_path, usecols=["sat_id", "global_cluster_id"])
    if df.empty:
        out = {}
        _CLUSTER_ASSIGNMENTS_CACHE[cache_key] = {"sig": sig, "map": out}
        return out

    df["sat_key"] = df["sat_id"].map(_normalize_cluster_sat_id)
    df["global_cluster_id"] = (pd.to_numeric(df["global_cluster_id"], errors="coerce")
                               .fillna(0).astype(np.int64))

    counts = (df.groupby(["sat_key", "global_cluster_id"], as_index=False)
              .size()
              .rename(columns={"size": "count"}))

    chosen = (counts.sort_values(["sat_key", "count", "global_cluster_id"],
                                 ascending=[True, False, True])
              .drop_duplicates(subset=["sat_key"], keep="first"))

    out = {str(row.sat_key): int(row.global_cluster_id)
           for row in chosen.itertuples(index=False)}
    _CLUSTER_ASSIGNMENTS_CACHE[cache_key] = {"sig": sig, "map": out}
    return out


def _load_global_cluster_stats(csv_path_value):
    """Load global cluster stats CSV for reporting. Returns empty dict if missing."""
    try:
        csv_path = _resolve_cluster_assignments_csv_path(csv_path_value)
        if not csv_path.is_file():
            return {}
        df = pd.read_csv(csv_path)
        if df.empty or "global_cluster_id" not in df.columns:
            return {}
        df["global_cluster_id"] = (pd.to_numeric(df["global_cluster_id"], errors="coerce")
                                   .fillna(0).astype(np.int64))
        return {int(row.global_cluster_id): row._asdict()
                for row in df.itertuples(index=False)}
    except Exception:
        return {}


def _find_matching_column(columns, *prefixes):
    """Return the first exact/prefix column match from *columns*."""
    clean_to_raw = {str(col).strip(): col for col in columns}
    for prefix in prefixes:
        if prefix in clean_to_raw:
            return clean_to_raw[prefix]
    for prefix in prefixes:
        for col in columns:
            clean = str(col).strip()
            if clean.startswith(prefix):
                return col
    return None


def _coerce_float_or_default(value, default_value):
    """Convert to float when possible, otherwise fall back to *default_value*."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default_value)
    if not np.isfinite(out):
        return float(default_value)
    return out


def _coerce_optional_float(value):
    """Convert to float when possible, otherwise return None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _load_cluster_policy_map():
    """Load stitched per-cluster policy defaults from optimization outputs."""
    candidates = []
    for candidate in (
        optimized_cluster_policy_csv,
        str(_base_dir_cluster / "optimization_outputs" / "stitched_policy_table.csv"),
        optimized_cluster_overview_csv,
        str(_base_dir_cluster / "optimization_outputs" / "cluster_overview.csv"),
    ):
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        csv_path = _resolve_cluster_assignments_csv_path(candidate)
        if not csv_path.is_file():
            continue

        cache_key = str(csv_path)
        stat = csv_path.stat()
        sig = (int(stat.st_mtime_ns), int(stat.st_size))
        cached = _CLUSTER_POLICY_CACHE.get(cache_key)
        if cached is not None and cached.get("sig") == sig:
            return cached.get("map", {}), cached.get("source")

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        cluster_col = _find_matching_column(df.columns, "cluster_id", "global_cluster_id")
        if cluster_col is None:
            continue

        field_cols = {
            field_name: _find_matching_column(df.columns, field_name, f"policy_{field_name}")
            for field_name, _ in _CLUSTER_POLICY_COLUMN_MAP
        }
        total_cost_col = _find_matching_column(df.columns, "total_cost", "cost_scalar")
        propellant_col = _find_matching_column(df.columns, "propellant_kg")

        policy_map = {}
        for _, row in df.iterrows():
            cid = _coerce_optional_float(row.get(cluster_col))
            if cid is None:
                continue
            cluster_id = int(cid)
            if cluster_id <= 0:
                continue

            record = {
                'cluster_policy_applied': True,
                'cluster_policy_source': str(csv_path),
                'policy_total_cost': _coerce_optional_float(row.get(total_cost_col)) if total_cost_col is not None else None,
                'policy_propellant_kg': _coerce_optional_float(row.get(propellant_col)) if propellant_col is not None else None,
            }
            for field_name, output_name in _CLUSTER_POLICY_COLUMN_MAP:
                col = field_cols.get(field_name)
                default_value = _CLUSTER_POLICY_DEFAULTS[output_name]
                record[output_name] = _coerce_float_or_default(
                    row.get(col) if col is not None else None,
                    default_value,
                )
            policy_map[cluster_id] = record

        if policy_map:
            _CLUSTER_POLICY_CACHE[cache_key] = {
                'sig': sig,
                'map': policy_map,
                'source': str(csv_path),
            }
            return policy_map, str(csv_path)

    return {}, None


def _build_pool_feature_matrix(df):
    """Build a (N,9) orbital-element feature matrix for medoid computation."""
    sma = pd.to_numeric(df["sma"], errors="coerce").to_numpy(dtype=np.float64)
    ecc = pd.to_numeric(df["ecc"], errors="coerce").to_numpy(dtype=np.float64)
    inc = pd.to_numeric(df["inc"], errors="coerce").to_numpy(dtype=np.float64)

    aop = np.deg2rad(pd.to_numeric(df["aop"], errors="coerce").to_numpy(dtype=np.float64))
    raan = np.deg2rad(pd.to_numeric(df["raan"], errors="coerce").to_numpy(dtype=np.float64))
    ma = np.deg2rad(pd.to_numeric(df["mean_anomaly"], errors="coerce").to_numpy(dtype=np.float64))

    features = np.column_stack([
        sma, ecc, inc,
        np.sin(aop), np.cos(aop),
        np.sin(raan), np.cos(raan),
        np.sin(ma), np.cos(ma),
    ])

    col_means = np.nanmean(np.where(np.isfinite(features), features, np.nan), axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    for j in range(features.shape[1]):
        bad = ~np.isfinite(features[:, j])
        if np.any(bad):
            features[bad, j] = col_means[j]
    return features


def _pick_medoid_index(feature_matrix):
    """Return the index of the medoid (closest to centroid under L2)."""
    if feature_matrix.shape[0] <= 1:
        return 0
    centroid = np.mean(feature_matrix, axis=0)
    diff = feature_matrix - centroid
    scores = np.sum(diff * diff, axis=1)
    return int(np.argmin(scores))


def _build_cluster_color_map(cluster_ids):
    """Map sorted unique cluster IDs to stable hex colors.

    Noise (id <= 0) gets ``cluster_noise_color``.  Non-noise IDs use the
    custom 20-color palette first, then fall back to matplotlib cmap.
    """
    unique_ids = sorted(set(int(c) for c in cluster_ids))
    positive_ids = [c for c in unique_ids if c > 0]
    n_pos = len(positive_ids)
    color_map = {}
    for cid in unique_ids:
        if cid <= 0:
            color_map[cid] = cluster_noise_color
    if n_pos <= len(colors):
        for idx_c, cid in enumerate(positive_ids):
            color_map[cid] = colors[idx_c % len(colors)]
    else:
        try:
            cmap = plt.get_cmap(cluster_palette_name, n_pos)
        except ValueError:
            cmap = plt.get_cmap("tab20", n_pos)
        for idx_c, cid in enumerate(positive_ids):
            if idx_c < len(colors):
                color_map[cid] = colors[idx_c]
            else:
                rgba = cmap(idx_c / max(n_pos - 1, 1))
                color_map[cid] = '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    return color_map


def _select_cluster_pooled_representatives(tle_latest_df, cluster_assignments=None, verbose=True):
    """Active-set medoid pooling.

    Returns ``(selected_df, metadata_by_sat, medoid_by_cluster, summary)``.
    Only clusters represented in the active set produce medoids.
    """
    if tle_latest_df is None or len(tle_latest_df) == 0:
        return tle_latest_df, {}, {}, {"input_rows": 0, "selected_rows": 0,
            "selected_noise_rows": 0, "selected_medoid_rows": 0,
            "non_noise_cluster_count": 0, "cluster_run_mode": "medoids"}

    t_pool0 = timeit.default_timer()

    work = tle_latest_df.copy()
    work["sat_id"] = work["sat_id"].astype(str)
    work["sat_key"] = work["sat_id"].map(_normalize_cluster_sat_id)

    if cluster_assignments is None:
        cluster_assignments = _load_cluster_assignments_map(cluster_assignments_csv)

    work["cluster_id"] = work["sat_key"].map(lambda k: int(cluster_assignments.get(str(k), 0)))
    work["is_noise"] = work["cluster_id"] <= 0

    # Load global stats for weight reporting
    global_stats = _load_global_cluster_stats(cluster_stats_csv)
    all_cluster_sizes_global = {int(k): int(v.get("count", 0))
                                for k, v in global_stats.items()} if global_stats else {}

    selected_indices = set()
    metadata_by_sat = {}
    medoid_by_cluster = {}

    # Active-set cluster sizes
    cluster_sizes_active = (work[~work["is_noise"]]
                            .groupby("cluster_id").size().to_dict())

    # Handle noise points
    noise_rows = work[work["is_noise"]]
    if cluster_keep_noise_unpooled == 1:
        selected_indices.update(noise_rows.index.tolist())
        for row in noise_rows.itertuples(index=False):
            metadata_by_sat[str(row.sat_id)] = {
                "cluster_id": int(row.cluster_id),
                "cluster_weight_active": 1,
                "cluster_weight_global": int(all_cluster_sizes_global.get(int(row.cluster_id), 1)),
                "pooled_role": "noise",
                "is_noise": True,
                "is_applicable": True,
                "representative_sat_id": str(row.sat_id),
                "color_hex": cluster_noise_color,
            }

    # Handle non-noise clusters: compute one medoid per active cluster
    non_noise = work[~work["is_noise"]].sort_values("sat_id", kind="mergesort")
    if len(non_noise) > 0:
        feats = _build_pool_feature_matrix(non_noise)
        feat_mean = np.mean(feats, axis=0)
        feat_std = np.std(feats, axis=0)
        feat_std = np.where(feat_std > 1e-12, feat_std, 1.0)
        feats_z = (feats - feat_mean) / feat_std

        cluster_ids_arr = non_noise["cluster_id"].to_numpy(dtype=np.int64)
        row_indices = non_noise.index.to_numpy(dtype=np.int64)
        sat_ids_sorted = non_noise["sat_id"].astype(str).to_numpy()

        order = np.argsort(cluster_ids_arr, kind="mergesort")
        cluster_ids_ord = cluster_ids_arr[order]
        change_idx = np.where(np.diff(cluster_ids_ord) != 0)[0] + 1
        starts = np.concatenate(([0], change_idx))
        ends = np.concatenate((change_idx, [order.size]))

        all_cluster_ids = sorted(set(cluster_ids_arr.tolist()))
        cmap = _build_cluster_color_map(list(set(work["cluster_id"].tolist())))

        for s, e in zip(starts, ends):
            idx_local = order[s:e]
            if idx_local.size == 0:
                continue
            cluster_id = int(cluster_ids_ord[s])
            medoid_rel = _pick_medoid_index(feats_z[idx_local, :])
            medoid_abs = int(idx_local[medoid_rel])
            selected_row_index = int(row_indices[medoid_abs])
            selected_indices.add(selected_row_index)

            medoid_sat_id = str(sat_ids_sorted[medoid_abs])
            medoid_by_cluster[cluster_id] = medoid_sat_id
            weight_active = int(cluster_sizes_active.get(cluster_id, 1))
            weight_global = int(all_cluster_sizes_global.get(cluster_id, weight_active))
            metadata_by_sat[medoid_sat_id] = {
                "cluster_id": cluster_id,
                "cluster_weight_active": weight_active,
                "cluster_weight_global": weight_global,
                "pooled_role": "medoid",
                "is_noise": False,
                "is_applicable": True,
                "representative_sat_id": medoid_sat_id,
                "color_hex": cmap.get(cluster_id, cluster_noise_color),
            }

    selected_df = (work.loc[sorted(selected_indices)]
                   .copy()
                   .sort_values("sat_id", kind="mergesort")
                   .reset_index(drop=True))

    summary = {
        "input_rows": int(len(work)),
        "selected_rows": int(len(selected_df)),
        "selected_noise_rows": int(sum(1 for v in metadata_by_sat.values() if v.get("is_noise"))),
        "selected_medoid_rows": int(len(medoid_by_cluster)),
        "non_noise_cluster_count": int(len(medoid_by_cluster)),
        "cluster_run_mode": "medoids",
    }

    if verbose:
        print("\n--- Cluster medoid pooling ---")
        print(f"  Assignment source : {_resolve_cluster_assignments_csv_path(cluster_assignments_csv)}")
        print(f"  Input satellites  : {summary['input_rows']}")
        print(f"  Non-noise clusters: {summary['non_noise_cluster_count']}")
        print(f"  Selected medoids  : {summary['selected_medoid_rows']}")
        print(f"  Selected noise    : {summary['selected_noise_rows']} (unpooled)")
        print(f"  Total selected    : {summary['selected_rows']}")
        print(f"  Pooling time      : {timeit.default_timer() - t_pool0:.3f} s")

    return selected_df, metadata_by_sat, medoid_by_cluster, summary


def _build_cluster_metadata_for_labels(tle_latest_df, cluster_assignments=None, verbose=True):
    """Attach cluster metadata to all active sats without filtering (labels mode)."""
    if tle_latest_df is None or len(tle_latest_df) == 0:
        return {}

    if cluster_assignments is None:
        cluster_assignments = _load_cluster_assignments_map(cluster_assignments_csv)

    global_stats = _load_global_cluster_stats(cluster_stats_csv)
    all_cluster_sizes_global = {int(k): int(v.get("count", 0))
                                for k, v in global_stats.items()} if global_stats else {}

    work = tle_latest_df.copy()
    work["sat_id"] = work["sat_id"].astype(str)
    work["sat_key"] = work["sat_id"].map(_normalize_cluster_sat_id)
    work["cluster_id"] = work["sat_key"].map(lambda k: int(cluster_assignments.get(str(k), 0)))
    work["is_noise"] = work["cluster_id"] <= 0

    cluster_sizes_active = work.groupby("cluster_id").size().to_dict()
    all_cluster_ids = sorted(set(work["cluster_id"].tolist()))
    cmap = _build_cluster_color_map(all_cluster_ids)

    metadata_by_sat = {}
    for row in work.itertuples(index=False):
        sid = str(row.sat_id)
        cid = int(row.cluster_id)
        is_noise = bool(cid <= 0)
        metadata_by_sat[sid] = {
            "cluster_id": cid,
            "cluster_weight_active": int(cluster_sizes_active.get(cid, 1)),
            "cluster_weight_global": int(all_cluster_sizes_global.get(cid, 1)),
            "pooled_role": "full_member",
            "is_noise": is_noise,
            "is_applicable": True,
            "representative_sat_id": sid,
            "color_hex": cmap.get(cid, cluster_noise_color),
        }

    n_noise = sum(1 for v in metadata_by_sat.values() if v.get("is_noise"))
    n_clusters = len([c for c in all_cluster_ids if c > 0])
    if verbose:
        print("\n--- Cluster labels mode ---")
        print(f"  Assignment source : {_resolve_cluster_assignments_csv_path(cluster_assignments_csv)}")
        print(f"  Active satellites : {len(metadata_by_sat)}")
        print(f"  Non-noise clusters: {n_clusters}")
        print(f"  Noise satellites  : {n_noise}")

    return metadata_by_sat


def _attach_cluster_policy_metadata(metadata_by_sat, policy_by_cluster):
    """Merge per-cluster policy defaults into the selected-satellite metadata map."""
    if not metadata_by_sat:
        return 0

    n_applied = 0
    for meta in metadata_by_sat.values():
        cid = int(meta.get("cluster_id", 0) or 0)
        policy_meta = policy_by_cluster.get(cid)
        if policy_meta is None:
            meta['cluster_policy_applied'] = False
            meta.setdefault('cluster_policy_source', '')
            continue
        meta.update(policy_meta)
        n_applied += 1

    return n_applied


def _append_cluster_policy_columns_to_df(df, cluster_assignments, policy_by_cluster):
    """Attach per-cluster policy metadata columns to a selection DataFrame."""
    if df is None or len(df) == 0:
        return df

    work = df.copy()
    if 'global_cluster_id' in work.columns:
        cluster_ids = pd.to_numeric(work['global_cluster_id'], errors='coerce').fillna(0).astype(np.int64)
    elif 'cluster_id' in work.columns:
        cluster_ids = pd.to_numeric(work['cluster_id'], errors='coerce').fillna(0).astype(np.int64)
        work['global_cluster_id'] = cluster_ids
    elif 'sat_id' in work.columns:
        sat_keys = work['sat_id'].astype(str).map(_normalize_cluster_sat_id)
        cluster_ids = sat_keys.map(lambda k: int(cluster_assignments.get(str(k), 0))).astype(np.int64)
        work['global_cluster_id'] = cluster_ids
    else:
        return work

    work['cluster_policy_applied'] = cluster_ids.map(lambda cid: int(cid) in policy_by_cluster)
    work['cluster_policy_source'] = cluster_ids.map(
        lambda cid: policy_by_cluster.get(int(cid), {}).get('cluster_policy_source', '')
    )
    for field_name in _CLUSTER_POLICY_OUTPUT_FIELDS:
        if field_name in ('cluster_policy_applied', 'cluster_policy_source'):
            continue
        work[field_name] = cluster_ids.map(
            lambda cid, _field=field_name: policy_by_cluster.get(int(cid), {}).get(_field, None)
        )

    return work


def _apply_cluster_policy_offsets_to_oe_cases(oe_cases, sat_ids, cluster_assignments, policy_by_cluster):
    """Apply stitched delta-a / delta-RAAN / delta-phase defaults to TLE cases."""
    if oe_cases is None or len(oe_cases) == 0 or not sat_ids or not policy_by_cluster:
        return oe_cases, 0, 0

    oe_adj = np.ascontiguousarray(np.array(oe_cases, dtype=np.float64, copy=True))
    n_applied = 0
    touched_clusters = set()

    for idx, sat_id in enumerate(sat_ids):
        sat_key = _normalize_cluster_sat_id(sat_id)
        cid = int(cluster_assignments.get(str(sat_key), 0))
        policy_meta = policy_by_cluster.get(cid)
        if policy_meta is None:
            continue

        oe_adj[idx, 0] = max(6378.1366 + 120.0, oe_adj[idx, 0] + policy_meta['policy_delta_a_km'])
        oe_adj[idx, 4] = (oe_adj[idx, 4] + policy_meta['policy_delta_Omega_rad']) % (2.0 * np.pi)
        oe_adj[idx, 5] = (oe_adj[idx, 5] + policy_meta['policy_delta_lambda_rad']) % (2.0 * np.pi)

        n_applied += 1
        touched_clusters.add(cid)

    return oe_adj, n_applied, len(touched_clusters)


def _write_cluster_csvs(batch_results, output_dir, active_df=None,
                        medoid_by_cluster=None, summary_info=None):
    """Write cluster-specific CSV files after batch simulation."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. active_cluster_selection.csv
    if active_df is not None and len(active_df) > 0:
        keep_cols = [c for c in active_df.columns
                     if c not in ("state_sat", "times")]
        active_df[keep_cols].to_csv(f"{output_dir}/active_cluster_selection.csv", index=False)

    # 2. batch_results_with_clusters.csv
    cluster_fields = _CLUSTER_OUTPUT_FIELDS
    base_cols = ['case_id', 'sat_id', 'start_timestamp', 'start_day_offset',
                 'a_km', 'e', 'i_deg', 'w_deg', 'OM_deg', 'Ma_deg',
                 'ballistic_coeff_m2_per_kg',
                 'n_points', 'terminated_at_115km', 't_115_s',
                 'final_x_km', 'final_y_km', 'final_z_km',
                 'final_vx_kms', 'final_vy_kms', 'final_vz_kms',
                 'initial_mass_kg', 'final_mass_kg', 'propellant_used_kg']
    header = base_cols + cluster_fields
    with open(f"{output_dir}/batch_results_with_clusters.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item in batch_results:
            vals = [item.get(c, '') for c in header]
            writer.writerow(vals)

    # 3. cluster_medoids_manifest.csv (medoid mode only)
    if medoid_by_cluster and len(medoid_by_cluster) > 0:
        with open(f"{output_dir}/cluster_medoids_manifest.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['global_cluster_id', 'representative_sat_id',
                             'cluster_weight_active', 'cluster_weight_global', 'is_noise'])
            for cid in sorted(medoid_by_cluster.keys()):
                sid = medoid_by_cluster[cid]
                meta = cluster_metadata_by_sat.get(sid, {})
                writer.writerow([cid, sid,
                                 meta.get("cluster_weight_active", 1),
                                 meta.get("cluster_weight_global", 1),
                                 False])

    # 4. cluster_summary_runtime.csv
    if batch_results and len(batch_results) > 0 and any('global_cluster_id' in r for r in batch_results):
        from collections import defaultdict
        by_cluster = defaultdict(list)
        for r in batch_results:
            cid = r.get('global_cluster_id', 0)
            by_cluster[cid].append(r)

        with open(f"{output_dir}/cluster_summary_runtime.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            header_rt = ['global_cluster_id', 'n_simulated_cases',
                         'cluster_weight_active', 'cluster_weight_global',
                         'is_noise', 'median_final_sma_km', 'median_cutoff_time_days',
                         'pooled_role_mode']
            writer.writerow(header_rt)
            for cid in sorted(by_cluster.keys()):
                cases_in_cluster = by_cluster[cid]
                n_cases = len(cases_in_cluster)
                meta0 = cluster_metadata_by_sat.get(cases_in_cluster[0].get('sat_id', ''), {})
                w_active = meta0.get("cluster_weight_active", 1)
                w_global = meta0.get("cluster_weight_global", 1)
                is_noise = meta0.get("is_noise", cid <= 0)
                role = meta0.get("pooled_role", "unknown")

                # Final SMA from last state sample
                final_smas = []
                cutoff_days = []
                for r in cases_in_cluster:
                    xs = r.get('state_sat')
                    if xs is not None and xs.shape[1] > 0:
                        r_final = np.sqrt(xs[0, -1]**2 + xs[1, -1]**2 + xs[2, -1]**2)
                        final_smas.append(r_final)
                    t115 = r.get('t_115_s', -1.0)
                    if t115 >= 0:
                        cutoff_days.append(t115 / 86400.0)

                median_sma = float(np.median(final_smas)) if final_smas else ''
                median_cut = float(np.median(cutoff_days)) if cutoff_days else ''

                writer.writerow([cid, n_cases, w_active, w_global, is_noise,
                                 median_sma, median_cut, role])


# Load the satellite's initial orbital elements
oe_sat = np.zeros(6)
oe_sat[0] = a
oe_sat[1] = e
oe_sat[2] = np.deg2rad(i)
oe_sat[3] = np.deg2rad(w)
oe_sat[4] = np.deg2rad(OM)
oe_sat[5] = np.deg2rad(Ma)

# TLE-selected initialization cases (used in batch and single-run when enabled)
tle_oe_cases = None
tle_sat_ids_selected = None
tle_start_datetimes_selected = None
tle_data_loaded = False

if use_tle_initial_conditions == 1 and _IS_MAIN_PROCESS:
    base_dir = Path(__file__).resolve().parent.parent
    tle_folder_paths = []
    for p in tle_data_folders:
        pp = Path(p)
        if not pp.is_absolute():
            pp = base_dir / pp
        tle_folder_paths.append(str(pp))

    tle_df, _ = load_all_tle_data(tle_folder_paths, only_files=tle_only_files, derived={"sma"})
    if tle_df is None or tle_df.empty:
        raise RuntimeError("No TLE data loaded from configured TLE folders.")

    tle_latest = (tle_df.sort_values('timestamp')
                  .groupby('sat_id', as_index=False)
                  .head(1)
                  .sort_values('sat_id')
                  .reset_index(drop=True))

    if tle_satellite_limit > 0:
        tle_latest = tle_latest.iloc[:int(tle_satellite_limit), :].copy()

    if tle_latest.empty:
        raise RuntimeError("TLE selection produced zero satellites. Check tle_only_files/tle_satellite_limit.")

    # ------------------------------------------------------------------
    # Global cluster integration (after normal TLE selection/filtering)
    # ------------------------------------------------------------------
    _cluster_active_df = None     # saved for CSV export
    _cluster_medoid_map = None    # medoid mode only
    _cluster_summary = None
    _cluster_assign_map = None

    if enable_global_cluster_features == 1 or enable_optimized_cluster_policy_defaults == 1:
        try:
            _cluster_assign_map = _load_cluster_assignments_map(cluster_assignments_csv)
        except FileNotFoundError as _cfe:
            print(f"[Cluster] WARNING: {_cfe}")
            print("[Cluster] Falling back to non-cluster behavior.")
            _cluster_assign_map = None

    if enable_global_cluster_features == 1 and _cluster_assign_map is not None:
            if cluster_run_mode == "labels":
                cluster_metadata_by_sat.update(
                    _build_cluster_metadata_for_labels(tle_latest,
                                                      cluster_assignments=_cluster_assign_map))
                # Attach helper columns to tle_latest for CSV export
                tle_latest["sat_key"] = tle_latest["sat_id"].astype(str).map(_normalize_cluster_sat_id)
                tle_latest["global_cluster_id"] = tle_latest["sat_key"].map(
                    lambda k: int(_cluster_assign_map.get(str(k), 0)))
                tle_latest["is_cluster_noise"] = tle_latest["global_cluster_id"] <= 0
                _cluster_active_df = tle_latest.copy()
                _cluster_summary = {"input_rows": len(tle_latest),
                                    "selected_rows": len(tle_latest),
                                    "selected_noise_rows": int(tle_latest["is_cluster_noise"].sum()),
                                    "selected_medoid_rows": 0,
                                    "non_noise_cluster_count": int(
                                        tle_latest.loc[tle_latest["global_cluster_id"] > 0,
                                                       "global_cluster_id"].nunique()),
                                    "cluster_run_mode": "labels"}

            elif cluster_run_mode == "medoids":
                # Optionally expand active set to all loaded sats
                if cluster_pool_ignore_satellite_limit == 1 and tle_satellite_limit > 0:
                    tle_pool = (tle_df.sort_values('timestamp')
                                .groupby('sat_id', as_index=False)
                                .head(1)
                                .sort_values('sat_id')
                                .reset_index(drop=True))
                    print(f"[Cluster] Medoid pool override: using all {len(tle_pool)} loaded sats "
                          f"(ignoring tle_satellite_limit={tle_satellite_limit})")
                else:
                    tle_pool = tle_latest

                (selected_df, meta_map, medoid_map,
                 pool_summary) = _select_cluster_pooled_representatives(
                    tle_pool, cluster_assignments=_cluster_assign_map, verbose=True)

                cluster_metadata_by_sat.update(meta_map)
                _cluster_medoid_map = medoid_map
                _cluster_summary = pool_summary
                _cluster_active_df = selected_df.copy()

                # Replace tle_latest with medoid selection
                tle_latest = selected_df
            else:
                print(f"[Cluster] Unknown cluster_run_mode='{cluster_run_mode}'; ignoring.")

    if enable_optimized_cluster_policy_defaults == 1 and _cluster_assign_map is not None and not cluster_metadata_by_sat:
        cluster_metadata_by_sat.update(
            _build_cluster_metadata_for_labels(
                tle_latest,
                cluster_assignments=_cluster_assign_map,
                verbose=False,
            )
        )

    n_tle = int(len(tle_latest))
    tle_oe_cases = np.zeros((n_tle, 6), dtype=np.float64)
    tle_oe_cases[:, 0] = tle_latest['sma'].to_numpy(dtype=np.float64)
    tle_oe_cases[:, 1] = np.clip(tle_latest['ecc'].to_numpy(dtype=np.float64), 0.0, 0.95)
    tle_oe_cases[:, 2] = np.deg2rad(tle_latest['inc'].to_numpy(dtype=np.float64))
    tle_oe_cases[:, 3] = np.deg2rad(tle_latest['aop'].to_numpy(dtype=np.float64))
    tle_oe_cases[:, 4] = np.deg2rad(tle_latest['raan'].to_numpy(dtype=np.float64))
    tle_oe_cases[:, 5] = np.deg2rad(tle_latest['mean_anomaly'].to_numpy(dtype=np.float64))
    tle_oe_cases[:, 0] = np.maximum(6378.1366 + 120.0, tle_oe_cases[:, 0])

    cluster_policy_by_cluster.clear()
    if enable_optimized_cluster_policy_defaults == 1:
        if _cluster_assign_map is None:
            print("[Cluster Policy] WARNING: Cluster assignments unavailable; stitched policy defaults were not applied.")
        else:
            _cluster_policy_map, _cluster_policy_source = _load_cluster_policy_map()
            cluster_policy_by_cluster.update(_cluster_policy_map)
            if cluster_policy_by_cluster:
                tle_oe_cases, _n_policy_sats, _n_policy_clusters = _apply_cluster_policy_offsets_to_oe_cases(
                    tle_oe_cases,
                    tle_latest['sat_id'].astype(str).tolist(),
                    _cluster_assign_map,
                    cluster_policy_by_cluster,
                )
                tle_latest = _append_cluster_policy_columns_to_df(
                    tle_latest,
                    _cluster_assign_map,
                    cluster_policy_by_cluster,
                )
                if _cluster_active_df is not None:
                    _cluster_active_df = _append_cluster_policy_columns_to_df(
                        _cluster_active_df,
                        _cluster_assign_map,
                        cluster_policy_by_cluster,
                    )
                _n_policy_meta = _attach_cluster_policy_metadata(
                    cluster_metadata_by_sat,
                    cluster_policy_by_cluster,
                )
                print("\n--- Stitched cluster policy defaults ---")
                print(f"  Policy source     : {_cluster_policy_source}")
                print(f"  Clusters loaded   : {len(cluster_policy_by_cluster)}")
                print(f"  Cases adjusted    : {_n_policy_sats}")
                print(f"  Clusters adjusted : {_n_policy_clusters}")
                if cluster_metadata_by_sat:
                    print(f"  Metadata attached : {_n_policy_meta}/{len(cluster_metadata_by_sat)} selected satellites")
            else:
                print("[Cluster Policy] WARNING: No stitched cluster policy table found; using raw TLE initial conditions.")

    tle_sat_ids_selected = tle_latest['sat_id'].astype(str).tolist()
    tle_start_datetimes_selected = pd.to_datetime(tle_latest['timestamp']).tolist()

    if cluster_metadata_by_sat:
        for _idx, _sid in enumerate(tle_sat_ids_selected):
            _meta = cluster_metadata_by_sat.get(_sid)
            if _meta is None:
                continue
            _meta['target_a_km'] = float(tle_oe_cases[_idx, 0])
            _meta['target_raan_rad'] = float(tle_oe_cases[_idx, 4])
            _meta['target_mean_anomaly_rad'] = float(tle_oe_cases[_idx, 5])
            _meta['target_lambda_rad'] = float((tle_oe_cases[_idx, 3] + tle_oe_cases[_idx, 4] + tle_oe_cases[_idx, 5]) % (2.0 * np.pi))
            _meta.setdefault('cluster_policy_applied', False)
            _meta.setdefault('cluster_policy_source', '')

    # Use first selected satellite as nominal single-run initial condition
    oe_sat[:] = tle_oe_cases[0, :]
    first_ts = tle_latest['timestamp'].iloc[0]
    if pd.notna(first_ts):
        epoch = pd.Timestamp(first_ts).strftime('%Y-%m-%d')

    tle_data_loaded = True
########################################################################

# -------------------- Constants --------------------
const = constants()
G = const.G  # gravitational constant [km^3/s^2]
au = const.au  # astronomical unit [km]

# Module-level constants
# Earth constants
earth_mass = 5.9722e24       # kg
earth_GM = G * earth_mass    # km^3/s^2
earth_Re = 6378.1366         # km
earth_J2 = 1.0826359e-3
earth_spin = 7.292115e-5     # rad/s

# Sun constants
sun_mass = 1.9884e30         # kg
sun_GM = G * sun_mass        # km^3/s^2

# Moon constants
moon_mass = 7.345828157e22   # kg
moon_GM = G * moon_mass      # km^3/s^2

#----------------------### Major bodies parameters ###-------------------------
# Major bodies' gravitational parameters
GM = np.zeros(3)
GM[0] = sun_GM  # Sun's gravitational parameter [km^3/s^2]
GM[1] = earth_GM  # Earth's gravitational parameter [km^3/s^2]
GM[2] = moon_GM  # Moon's gravitational parameter [km^3/s^2]

# Earth Oblateness (for the J2 perturbation in the Moon)
J2 = earth_J2  # Earth's J2 coefficient

# Load the Earth's spherical harmonics coefficients
# The coefficients are stored in the file "EGM2008_upto50_TideFree.in"
# Full normalized coefficients up to degree and order 50
input = np.loadtxt('EGM2008_upto50_TideFree.in', skiprows=1)

C = np.zeros((nmax + 1, mmax + 1))
S = np.zeros((nmax + 1, mmax + 1))

# print(f"len(input[:,0]) = {len(input[:,0])}")

for i in range(0, len(input[:, 0])):
    degree = int(input[i, 0])
    order = int(input[i, 1])
    if degree <= nmax and order <= mmax:
        C[degree, order] = input[i, 2]
        S[degree, order] = input[i, 3]
    if degree == nmax and order == mmax:
        break

# Test if the variables were correctly loaded
# print(f"\nTest if the variables were correctly loaded:")
# for n in range(2, nmax + 1):
#     for m in range(0, nmax + 1):
#       if m <= n:
#         # print(f"C({n},{m}) = {C[n, m]:20.14e} | S({n},{m}) = {S[n, m]:20.14e}")
#         print("C({0},{1}) = {2:20.14e}, S({0},{1}) = {3:20.14e}".format(n, m, C[n, m], S[n, m]))
#---------------------------------------------------------------------------------------
# Convert the satellite's initial orbital elements to Cartesian coordinates
init_sc = orb2xyz(earth_GM, oe_sat)

# Satellite's period
period = 2.0 * np.pi * np.sqrt(oe_sat[0] ** 3 / earth_GM)

if _IS_MAIN_PROCESS:
    print(f"\nSatellite's initial conditions with respect to the Earth:")
    print(f"---------------------------------------------------------")
    if use_tle_initial_conditions == 1 and tle_data_loaded and tle_sat_ids_selected:
        print(f"TLE source enabled: {len(tle_sat_ids_selected)} satellites loaded")
        print(f"Nominal satellite ID: {tle_sat_ids_selected[0]}")
    print(f"Initial semi-major axis (a): {oe_sat[0]:.5e} km")
    print(f"Initial eccentricity (e): {oe_sat[1]:.5e}")
    print(f"Initial inclination (i): {np.rad2deg(oe_sat[2]):.3f} deg")
    print(f"Initial argument of perigee (w): {np.rad2deg(oe_sat[3]):.2f} deg")
    print(f"Initial RAAN (OM): {np.rad2deg(oe_sat[4]):.2f} deg")
    print(f"Initial mean anomaly (Ma): {np.rad2deg(oe_sat[5]):.2f} deg")
    print(f"Initial position vector (x, y, z): {init_sc[0]:.5e}, {init_sc[1]:.5e}, {init_sc[2]:.5e} (km)")
    print(f"Initial velocity vector (vx, vy, vz): {init_sc[3]:.5e}, {init_sc[4]:.5e}, {init_sc[5]:.5e} (km/s)")
    print(f"Satellite's period: {period / 3600:.2f} hours")
    print(f"\nSatellite's physical properties:")
    print(f"Area-to-mass ratio (A/m): {AtoM:.5e} m^2/kg")
    print(f"Reflectivity coefficient (Cr): {Cr}")
    print(f"Drag coefficient (Cd): {Cd}")
    print(f"Nominal drag beta (posterior mean Cd*A/m): {ballistic_coefficient_nominal:.5e} m^2/kg")
    print(f"---------------------------------------------------------\n")

# Load the major bodies' initial conditions (mb) and the initial Julian date (jd) at the epoch
mb, jd0 = load_mb(frame, epoch)

# Load the initial GST at the epoch
GST0 = np.deg2rad(gst0(jd0))

if _IS_MAIN_PROCESS:
    print(f"Initial Julian date: {jd0}")
    print(f"Initial GST: {np.rad2deg(GST0)} deg")

# Major bodies' initial conditions
# Sun
Xb_sun = mb[0, :].astype(np.float64)
# Earth
Xb_earth = mb[1, :].astype(np.float64)
# Moon
Xb_moon = mb[2, :].astype(np.float64)
# Convert the Sun's initial conditions to orbital elements
mu = GM[0] + GM[1]
oe_sun = xyz2orb(mu, -Xb_sun[0:3], -Xb_sun[3:6])
# Earth is the central body, so the orbital elements are not defined
# Convert the Moon's initial conditions to orbital elements
mu = GM[1] + GM[2]
oe_moon = xyz2orb(mu, Xb_moon[0:3], Xb_moon[3:6])

if _IS_MAIN_PROCESS:
    print(f"\nMajor bodies' initial conditions:")
    print(f"---------------------------------")
    print(f"Sun:")
    print(f"Position vector: {Xb_sun[0]:.5e}, {Xb_sun[1]:.5e}, {Xb_sun[2]:.5e} (km)")
    print(f"R_sun = {np.linalg.norm(Xb_sun[0:2]):.8e} km")
    print(f"Velocity vector: {Xb_sun[3]:.5e}, {Xb_sun[4]:.5e}, {Xb_sun[5]:.5e} (km/s)")
    print(f"V_sun = {np.linalg.norm(Xb_sun[3:5]):.8e} km/s")
    print(f"Orbital elements from state vector (Earth viewed from the Sun):")
    print(f"a = {(oe_sun[0]):.8e} km")
    print(f"e = {oe_sun[1]:.8e}")
    print(f"i = {np.rad2deg(oe_sun[2]):.2f} deg")
    print(f"w = {np.rad2deg(oe_sun[3]):.2f} deg")
    print(f"Omega = {np.rad2deg(oe_sun[4]):.2f} deg")
    print(f"Ma = {np.rad2deg(oe_sun[5]):.2f} deg")

    print(f"\nMoon:")
    print(f"Position vector: {Xb_moon[0]:.5e}, {Xb_moon[1]:.5e}, {Xb_moon[2]:.5e} (km)")
    print(f"R_moon = {np.linalg.norm(Xb_moon[0:2]):.8e} km")
    print(f"Velocity vector: {Xb_moon[3]:.5e}, {Xb_moon[4]:.5e}, {Xb_moon[5]:.5e} (km/s)")
    print(f"V_moon = {np.linalg.norm(Xb_moon[3:5]):.8e} km/s")
    print(f"Orbital elements from state vector:")
    print(f"a = {(oe_moon[0]):.8e} km")
    print(f"e = {oe_moon[1]:.8e}")
    print(f"i = {np.rad2deg(oe_moon[2]):.2f} deg")
    print(f"w = {np.rad2deg(oe_moon[3]):.2f} deg")
    print(f"Omega = {np.rad2deg(oe_moon[4]):.2f} deg")
    print(f"Ma = {np.rad2deg(oe_moon[5]):.2f} deg")
    print(f"---------------------------------\n")

# Load the initial state vectors of all bodies into a single array (Xb)
# This array is used to initialize the integration process

# Size of the array Xb
# When k_thrust == 1 the propagated state has 19 elements (appended mass);
# otherwise the classic 18-element state is used.
if k_thrust == 1:
    neq = 6 * 3 + 1  # Sun(6) + Moon(6) + Sat(6) + mass(1)
else:
    neq = 6 * 3
Xb_init = np.zeros(neq)

# Sun
Xb_init[0:6] = Xb_sun
# Moon
Xb_init[6:12] = Xb_moon
# Satellite
Xb_init[12:18] = init_sc
# Spacecraft mass (only used when k_thrust == 1)
if k_thrust == 1:
    Xb_init[18] = thrust_initial_mass_kg

# Ensure stable dtypes/contiguity for Numba (avoids accidental recompiles)
Xb_init = np.ascontiguousarray(Xb_init, dtype=np.float64)
C = np.ascontiguousarray(C, dtype=np.float64)
S = np.ascontiguousarray(S, dtype=np.float64)

#-------------------------------------------------------------------------------
#-------------#### Event Functions to control the integration ####--------------
critical = 0
def Reentry(t, f, *args):
    r = np.sqrt(f[12] ** 2 + f[13] ** 2 + f[14] ** 2)
    alt = r - earth_Re
    return alt - 100.0
Reentry.direction = -1
Reentry.terminal = True

def BatchCutoff(t, f, *args):
    r = np.sqrt(f[12] ** 2 + f[13] ** 2 + f[14] ** 2)
    alt = r - earth_Re
    return alt - batch_cutoff_alt_km
BatchCutoff.direction = -1
BatchCutoff.terminal = True

def _resolve_batch_worker_count():
    cpu_total = os.cpu_count() or 1
    if max_parallel_workers is None:
        return max(1, cpu_total - 1)
    return max(1, int(max_parallel_workers))

_numba_warmed = False
_batch_epoch_cache: dict = {}   # Persistent across run_batch_cases() calls

def _warmup_numba_derivs():
    global _numba_warmed
    if _numba_warmed:
        return

    mm = nmax + 1
    p_pre = np.ascontiguousarray(np.zeros((mm, mm), dtype=np.float64))
    pl_pre = np.ascontiguousarray(np.zeros((mm, mm), dtype=np.float64))
    sml_pre_local = np.ascontiguousarray(np.zeros(mm, dtype=np.float64))
    cml_pre_local = np.ascontiguousarray(np.zeros(mm, dtype=np.float64))
    tmp3_pre_local = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

    # Warm up classic 18-state Derivs
    _xb18 = np.ascontiguousarray(Xb_init[:18], dtype=np.float64) if len(Xb_init) > 18 else Xb_init
    _ = Derivs(0.0, _xb18, p_pre, pl_pre, sml_pre_local, cml_pre_local, tmp3_pre_local,
               ballistic_coefficient_nominal, GST0)

    # Warm up thrust-enabled 19-state Derivs_thrust
    if k_thrust == 1:
        _xb19 = np.zeros(19, dtype=np.float64)
        _xb19[:18] = _xb18
        _xb19[18] = thrust_initial_mass_kg
        _ = Derivs_thrust(0.0, _xb19,
                          np.zeros((mm, mm), dtype=np.float64),
                          np.zeros((mm, mm), dtype=np.float64),
                          np.zeros(mm, dtype=np.float64),
                          np.zeros(mm, dtype=np.float64),
                          np.zeros(3, dtype=np.float64),
                          ballistic_coefficient_nominal, GST0,
                          0.0, 1500.0, 1, thrust_dry_mass_kg,
                          frontal_area, thrust_initial_mass_kg)

    _numba_warmed = True
#-------------------------------------------------------------------------------
#-----------------#### Function with derivatives (model) ####-------------------
@njit(cache=True, fastmath=True, nogil=True)
def Derivs(t, f, P, Pl, sml, cml, tmp3, ballistic_coefficient, gst0_case):
    Re = earth_Re

    x_sun = f[0:6]
    r_sun = np.sqrt(x_sun[0] ** 2 + x_sun[1] ** 2 + x_sun[2] ** 2)

    x_moon = f[6:12]
    r_moon = np.sqrt(x_moon[0] ** 2 + x_moon[1] ** 2 + x_moon[2] ** 2)

    x_sat = f[12:18]
    r = np.sqrt(x_sat[0] ** 2 + x_sat[1] ** 2 + x_sat[2] ** 2)

    # Sun EOM
    dxsundt = x_sun[3]
    dysundt = x_sun[4]
    dzsundt = x_sun[5]

    mu_sun_earth = sun_GM + earth_GM
    ddxsundt = -mu_sun_earth * x_sun[0] / r_sun ** 3
    ddysundt = -mu_sun_earth * x_sun[1] / r_sun ** 3
    ddzsundt = -mu_sun_earth * x_sun[2] / r_sun ** 3

    # Moon EOM
    dxmoondt = x_moon[3]
    dymoondt = x_moon[4]
    dzmoondt = x_moon[5]

    # Sun's perturbation on the Moon
    AC3b(x_moon[0:3], x_sun[0:3], sun_GM, tmp3)
    ac3b_sun0 = tmp3[0]
    ac3b_sun1 = tmp3[1]
    ac3b_sun2 = tmp3[2]

    # Earth's J2 perturbation on the Moon
    J2acc(earth_GM, earth_J2, earth_Re, x_moon[0:3], tmp3)
    acj20 = tmp3[0]
    acj21 = tmp3[1]
    acj22 = tmp3[2]
    mu_moon_earth = moon_GM + earth_GM
    ddxmoondt = -mu_moon_earth * x_moon[0] / r_moon ** 3 + acj20 + ac3b_sun0
    ddymoondt = -mu_moon_earth * x_moon[1] / r_moon ** 3 + acj21 + ac3b_sun1
    ddzmoondt = -mu_moon_earth * x_moon[2] / r_moon ** 3 + acj22 + ac3b_sun2

    # Accumulate spacecraft perturbations into tmp3 (used as accumulator)
    axp = 0.0
    ayp = 0.0
    azp = 0.0

    # Earth's spherical harmonics
    if k_EGM2008 == 1:
        EGM2008(nmax, mmax, x_sat[0:3], C, S, t, earth_GM, earth_Re, earth_spin, gst0_case, P, Pl, sml, cml, tmp3)
        axp += tmp3[0]
        ayp += tmp3[1]
        azp += tmp3[2]

    # Third-body: Moon
    if k_moon == 1:
        AC3b(x_sat[0:3], x_moon[0:3], moon_GM, tmp3)
        axp += tmp3[0]
        ayp += tmp3[1]
        azp += tmp3[2]

    # Third-body: Sun
    if k_sun == 1:
        AC3b(x_sat[0:3], x_sun[0:3], sun_GM, tmp3)
        axp += tmp3[0]
        ayp += tmp3[1]
        azp += tmp3[2]

    # Solar radiation pressure
    if k_SRP == 1:
        SRPacc(x_sat[0:3], x_sun[0:3], AtoM, Cr, Re, tmp3)
        axp += tmp3[0]
        ayp += tmp3[1]
        azp += tmp3[2]

    # Atmospheric drag
    if r <= (900.0 + Re) and k_atm_drag == 1:
        atm_drag(x_sat, 1.0, ballistic_coefficient, earth_spin, tmp3)
        axp += tmp3[0]
        ayp += tmp3[1]
        azp += tmp3[2]

    # Spacecraft EOM
    dxdt = x_sat[3]
    dydt = x_sat[4]
    dzdt = x_sat[5]

    mu = earth_GM
    ddxdt = -mu * x_sat[0] / r ** 3 + axp
    ddydt = -mu * x_sat[1] / r ** 3 + ayp
    ddzdt = -mu * x_sat[2] / r ** 3 + azp

    # Allocate locally the derivatives
    derivatives = np.empty(18, dtype=np.float64)

    # derivatives = np.zeros(18)
    derivatives[0] = dxsundt
    derivatives[1] = dysundt
    derivatives[2] = dzsundt
    derivatives[3] = ddxsundt
    derivatives[4] = ddysundt
    derivatives[5] = ddzsundt
    derivatives[6] = dxmoondt
    derivatives[7] = dymoondt
    derivatives[8] = dzmoondt
    derivatives[9] = ddxmoondt
    derivatives[10] = ddymoondt
    derivatives[11] = ddzmoondt
    derivatives[12] = dxdt
    derivatives[13] = dydt
    derivatives[14] = dzdt
    derivatives[15] = ddxdt
    derivatives[16] = ddydt
    derivatives[17] = ddzdt

    return derivatives
#-------------------------------------------------------------------------------
#----------#### Thrust-enabled RHS (19-state: Cartesian + mass) ####-----------
@njit(cache=True, fastmath=True, nogil=True)
def Derivs_thrust(t, f, P, Pl, sml, cml, tmp3,
                  ballistic_coefficient_nominal_arg, gst0_case,
                  T_eff_N, Isp_s, thrust_sign, dry_mass_kg,
                  frontal_area_val, initial_mass_kg):
    """RHS for 19-element state: [sun(6), moon(6), sat(6), mass(1)].

    Thrust control scalars (T_eff_N, Isp_s, thrust_sign) are constant
    within each piecewise segment. Dynamic SRP A/m is recomputed from the
    configured frontal area, while drag beta scales from the calibrated
    initial nominal value with the current spacecraft mass.
    """
    Re = earth_Re
    g0 = 9.80665  # m/s^2 — standard gravity

    x_sun = f[0:6]
    r_sun = np.sqrt(x_sun[0]**2 + x_sun[1]**2 + x_sun[2]**2)

    x_moon = f[6:12]
    r_moon = np.sqrt(x_moon[0]**2 + x_moon[1]**2 + x_moon[2]**2)

    x_sat = f[12:18]
    r = np.sqrt(x_sat[0]**2 + x_sat[1]**2 + x_sat[2]**2)

    m_sc = f[18]  # current spacecraft mass (kg)

    # --- Dynamic SRP area-to-mass and drag beta ---
    m_ref = m_sc if m_sc > 1.0 else 1.0
    AtoM_now = frontal_area_val / m_ref
    bc_now = ballistic_coefficient_nominal_arg * (initial_mass_kg / m_ref)

    # ===================== Sun EOM =====================
    dxsundt = x_sun[3]
    dysundt = x_sun[4]
    dzsundt = x_sun[5]

    mu_sun_earth = sun_GM + earth_GM
    ddxsundt = -mu_sun_earth * x_sun[0] / r_sun**3
    ddysundt = -mu_sun_earth * x_sun[1] / r_sun**3
    ddzsundt = -mu_sun_earth * x_sun[2] / r_sun**3

    # ===================== Moon EOM =====================
    dxmoondt = x_moon[3]
    dymoondt = x_moon[4]
    dzmoondt = x_moon[5]

    AC3b(x_moon[0:3], x_sun[0:3], sun_GM, tmp3)
    ac3b_sun0, ac3b_sun1, ac3b_sun2 = tmp3[0], tmp3[1], tmp3[2]

    J2acc(earth_GM, earth_J2, earth_Re, x_moon[0:3], tmp3)
    acj20, acj21, acj22 = tmp3[0], tmp3[1], tmp3[2]

    mu_moon_earth = moon_GM + earth_GM
    ddxmoondt = -mu_moon_earth * x_moon[0] / r_moon**3 + acj20 + ac3b_sun0
    ddymoondt = -mu_moon_earth * x_moon[1] / r_moon**3 + acj21 + ac3b_sun1
    ddzmoondt = -mu_moon_earth * x_moon[2] / r_moon**3 + acj22 + ac3b_sun2

    # ============= Spacecraft perturbations =============
    axp = 0.0
    ayp = 0.0
    azp = 0.0

    if k_EGM2008 == 1:
        EGM2008(nmax, mmax, x_sat[0:3], C, S, t, earth_GM, earth_Re,
                earth_spin, gst0_case, P, Pl, sml, cml, tmp3)
        axp += tmp3[0]; ayp += tmp3[1]; azp += tmp3[2]

    if k_moon == 1:
        AC3b(x_sat[0:3], x_moon[0:3], moon_GM, tmp3)
        axp += tmp3[0]; ayp += tmp3[1]; azp += tmp3[2]

    if k_sun == 1:
        AC3b(x_sat[0:3], x_sun[0:3], sun_GM, tmp3)
        axp += tmp3[0]; ayp += tmp3[1]; azp += tmp3[2]

    # SRP with dynamic AtoM
    if k_SRP == 1:
        SRPacc(x_sat[0:3], x_sun[0:3], AtoM_now, Cr, Re, tmp3)
        axp += tmp3[0]; ayp += tmp3[1]; azp += tmp3[2]

    # Atmospheric drag with dynamic ballistic coefficient
    if r <= (900.0 + Re) and k_atm_drag == 1:
        atm_drag(x_sat, 1.0, bc_now, earth_spin, tmp3)
        axp += tmp3[0]; ayp += tmp3[1]; azp += tmp3[2]

    # ============= Along-track thrust =============
    dm_dt = 0.0
    if T_eff_N > 0.0 and m_sc > dry_mass_kg:
        # RTN tangential direction (inlined for Numba)
        r_mag = r
        r_hat0 = x_sat[0] / r_mag
        r_hat1 = x_sat[1] / r_mag
        r_hat2 = x_sat[2] / r_mag
        # h = r × v
        hx = x_sat[1]*x_sat[5] - x_sat[2]*x_sat[4]
        hy = x_sat[2]*x_sat[3] - x_sat[0]*x_sat[5]
        hz = x_sat[0]*x_sat[4] - x_sat[1]*x_sat[3]
        h_mag = np.sqrt(hx*hx + hy*hy + hz*hz)
        if h_mag > 1.0e-12:
            h_hat0 = hx / h_mag
            h_hat1 = hy / h_mag
            h_hat2 = hz / h_mag
            # t_hat = h_hat × r_hat
            t0 = h_hat1*r_hat2 - h_hat2*r_hat1
            t1 = h_hat2*r_hat0 - h_hat0*r_hat2
            t2 = h_hat0*r_hat1 - h_hat1*r_hat0
            t_mag = np.sqrt(t0*t0 + t1*t1 + t2*t2)
            if t_mag > 1.0e-12:
                t0 /= t_mag; t1 /= t_mag; t2 /= t_mag
                a_thrust_mag = (T_eff_N / m_sc) * 1.0e-3  # km/s^2
                axp += thrust_sign * a_thrust_mag * t0
                ayp += thrust_sign * a_thrust_mag * t1
                azp += thrust_sign * a_thrust_mag * t2
                dm_dt = -T_eff_N / (g0 * Isp_s)  # kg/s (always ≤ 0)

    # ============= Spacecraft EOM =============
    dxdt = x_sat[3]
    dydt = x_sat[4]
    dzdt = x_sat[5]

    mu = earth_GM
    ddxdt = -mu * x_sat[0] / r**3 + axp
    ddydt = -mu * x_sat[1] / r**3 + ayp
    ddzdt = -mu * x_sat[2] / r**3 + azp

    # ============= Pack 19-element derivative vector =============
    derivatives = np.empty(19, dtype=np.float64)
    derivatives[0]  = dxsundt
    derivatives[1]  = dysundt
    derivatives[2]  = dzsundt
    derivatives[3]  = ddxsundt
    derivatives[4]  = ddysundt
    derivatives[5]  = ddzsundt
    derivatives[6]  = dxmoondt
    derivatives[7]  = dymoondt
    derivatives[8]  = dzmoondt
    derivatives[9]  = ddxmoondt
    derivatives[10] = ddymoondt
    derivatives[11] = ddzmoondt
    derivatives[12] = dxdt
    derivatives[13] = dydt
    derivatives[14] = dzdt
    derivatives[15] = ddxdt
    derivatives[16] = ddydt
    derivatives[17] = ddzdt
    derivatives[18] = dm_dt
    return derivatives
#-------------------------------------------------------------------------------

def _build_initial_state_from_oe(oe_case, x_sun_case, x_moon_case,
                                 initial_mass_kg=None):
    """Build initial state vector from orbital elements.

    When *initial_mass_kg* is not None a 19-element state is returned
    (appended mass); otherwise the classic 18-element vector.
    """
    init_sc_case = orb2xyz(earth_GM, oe_case)
    if initial_mass_kg is not None:
        xb_case = np.zeros(19, dtype=np.float64)
        xb_case[0:6] = x_sun_case
        xb_case[6:12] = x_moon_case
        xb_case[12:18] = init_sc_case
        xb_case[18] = initial_mass_kg
    else:
        xb_case = np.zeros(18, dtype=np.float64)
        xb_case[0:6] = x_sun_case
        xb_case[6:12] = x_moon_case
        xb_case[12:18] = init_sc_case
    return np.ascontiguousarray(xb_case, dtype=np.float64)

def _truncate_state_at_altitude(times_in, state_in, cutoff_alt_km):
    r_norm = np.sqrt(state_in[12, :] ** 2 + state_in[13, :] ** 2 + state_in[14, :] ** 2)
    alt = r_norm - earth_Re
    hit = np.where(alt <= cutoff_alt_km)[0]

    if hit.size == 0:
        return times_in, state_in, -1.0

    idx = int(hit[0])
    return times_in[:idx + 1], state_in[:, :idx + 1], float(times_in[idx])

def _seconds_until_date_cutoff(start_timestamp):
    if start_timestamp is None:
        return float(tf + dt)

    start_ts = pd.Timestamp(start_timestamp)
    if pd.isna(start_ts):
        return float(tf + dt)

    min_start = pd.Timestamp(tle_earliest_start_epoch)
    cutoff_ts = pd.Timestamp(simulation_date_cutoff)

    # Guard in case of older-than-expected epochs.
    if start_ts < min_start:
        start_ts = min_start

    dt_sec = (cutoff_ts - start_ts).total_seconds()
    if dt_sec <= 0.0:
        return 0.0
    return float(min(tf + dt, dt_sec))

def _build_t_eval_with_cutoff(t_final_case):
    if t_final_case <= 0.0:
        return np.array([0.0], dtype=np.float64)

    base = tspan[::_output_stride] if _output_stride > 1 else tspan
    t_eval_case = base[base <= (t_final_case + 1e-12)]
    if t_eval_case.size == 0:
        return np.array([0.0], dtype=np.float64)
    if t_eval_case[0] != 0.0:
        t_eval_case = np.concatenate((np.array([0.0], dtype=np.float64), t_eval_case))
    return np.asarray(t_eval_case, dtype=np.float64)

def _integrate_single_case(oe_case, ballistic_coefficient_case,
                           x_sun_case, x_moon_case, gst0_case,
                           start_timestamp=None, event_mode='reentry',
                           case_schedule=None):
    """Dispatch a single propagation case.

    When *case_schedule* is provided and k_thrust==1, a 19-state segmented
    integration with thrust is used.  Otherwise the classic 18-state
    Derivs path (USSA76 or MSIS) is used unchanged.
    """
    use_thrust = (k_thrust == 1 and case_schedule is not None)

    if use_thrust:
        xb_case = _build_initial_state_from_oe(
            oe_case, x_sun_case, x_moon_case,
            initial_mass_kg=thrust_initial_mass_kg)
    else:
        xb_case = _build_initial_state_from_oe(oe_case, x_sun_case, x_moon_case)

    t_final_case = _seconds_until_date_cutoff(start_timestamp)
    # Apply optional propagation-time cap (used by optimizer for horizon_fraction)
    if _max_prop_time_s is not None and _max_prop_time_s > 0:
        t_final_case = min(t_final_case, float(_max_prop_time_s))
    t_eval_case = _build_t_eval_with_cutoff(t_final_case)
    n_state = xb_case.size  # 18 or 19

    if t_final_case <= 0.0:
        return (np.array([0.0], dtype=np.float64),
                xb_case.reshape((n_state, 1)), -1.0)

    # ---- Thrust-enabled segmented path ----
    if use_thrust:
        if atm_model == 1 and k_atm_drag == 1:
            return _run_with_msis_grids(
                t_eval_override=t_eval_case,
                initial_state=xb_case,
                start_timestamp=start_timestamp,
                ballistic_coefficient_case=ballistic_coefficient_case,
                gst0_case=gst0_case,
                event_mode=event_mode,
                case_schedule=case_schedule)
        else:
            return _run_segmented_thrust_case(
                t_eval_full=t_eval_case,
                initial_state=xb_case,
                gst0_case=gst0_case,
                ballistic_coefficient_case=ballistic_coefficient_case,
                event_mode=event_mode,
                case_schedule=case_schedule)

    # ---- Classic no-thrust path ----
    if atm_model == 1 and k_atm_drag == 1:
        return _run_with_msis_grids(t_eval_override=t_eval_case,
                                    initial_state=xb_case,
                                    start_timestamp=start_timestamp,
                                    ballistic_coefficient_case=ballistic_coefficient_case,
                                    gst0_case=gst0_case,
                                    event_mode=event_mode)

    MM = nmax + 1
    p_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    pl_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    sml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    cml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    tmp3_pre_local = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

    if event_mode == 'reentry':
        events = [Reentry]
    elif event_mode == 'batch_cutoff':
        events = [BatchCutoff]
    else:
        events = None

    sol = solve_ivp(Derivs, [0.0, t_final_case], xb_case, events=events, method='DOP853',
                    t_eval=t_eval_case, rtol=_solver_rtol, atol=_solver_atol,
                  args=(p_pre, pl_pre, sml_pre_local, cml_pre_local, tmp3_pre_local,
                                            ballistic_coefficient_case, gst0_case))

    t_event = -1.0
    if events is not None and getattr(sol, 't_events', None) is not None:
        if len(sol.t_events) > 0 and np.asarray(sol.t_events[0]).size > 0:
            t_event = float(sol.t_events[0][0])

    return np.asarray(sol.t, dtype=np.float64), np.asarray(sol.y, dtype=np.float64), t_event


# ---------------------------------------------------------------------------
# Segmented thrust propagation (USSA76 atmosphere path — non-MSIS)
# ---------------------------------------------------------------------------
def _run_segmented_thrust_case(t_eval_full, initial_state, gst0_case,
                               ballistic_coefficient_case, event_mode,
                               case_schedule):
    """Piecewise-constant thrust segmentation over phase boundaries (USSA76).

    Each segment uses Derivs_thrust with constant control scalars.
    """
    if event_mode == 'reentry':
        events = [Reentry]
    elif event_mode == 'batch_cutoff':
        events = [BatchCutoff]
    else:
        events = None

    t_final = float(t_eval_full[-1]) if t_eval_full.size > 0 else 0.0
    boundaries = get_schedule_boundaries(case_schedule, t_final_s=t_final)

    MM = nmax + 1
    p_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    pl_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    sml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    cml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    tmp3_pre_local = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

    y0 = np.ascontiguousarray(initial_state, dtype=np.float64)
    t_all = []
    y_all = []
    t_event_case = -1.0
    controller_state = {}

    for seg_idx in range(len(boundaries) - 1):
        seg_t0 = boundaries[seg_idx]
        seg_t1 = boundaries[seg_idx + 1]
        if seg_t1 <= seg_t0:
            continue

        # Slice t_eval to this segment
        mask = (t_eval_full >= seg_t0 - 1e-12) & (t_eval_full <= seg_t1 + 1e-12)
        t_seg_eval = t_eval_full[mask]
        if t_seg_eval.size == 0:
            t_seg_eval = np.array([seg_t0, seg_t1], dtype=np.float64)

        # Look up phase control for this segment's midpoint
        seg_mid = 0.5 * (seg_t0 + seg_t1)
        seg_ctrl = lookup_segment_for_time(case_schedule, seg_mid)
        a_now, lambda_now, alt_now = _extract_guidance_state(y0)
        seg_ctrl = resolve_segment_command(
            seg_ctrl,
            current_a_km=a_now,
            current_lambda_rad=lambda_now,
            current_alt_km=alt_now,
            current_mass_kg=float(y0[18]) if y0.size > 18 else float(thrust_initial_mass_kg),
            dry_mass_kg=float(thrust_dry_mass_kg),
            initial_mass_kg=float(thrust_initial_mass_kg),
            segment_start_s=float(seg_t0),
            controller_state=controller_state,
        )
        T_eff_seg = float(seg_ctrl['T_eff_N'])
        Isp_seg = float(seg_ctrl['Isp_s']) if seg_ctrl['Isp_s'] > 0 else 1500.0
        sign_seg = int(seg_ctrl['sign'])

        # Local time within segment
        t_seg_local = t_seg_eval - seg_t0

        sol = solve_ivp(
            Derivs_thrust, [0.0, float(seg_t1 - seg_t0)], y0,
            events=events, method='DOP853',
            t_eval=t_seg_local, rtol=_solver_rtol, atol=_solver_atol,
            args=(p_pre, pl_pre, sml_pre_local, cml_pre_local, tmp3_pre_local,
                  ballistic_coefficient_case, gst0_case,
                  T_eff_seg, Isp_seg, sign_seg,
                  thrust_dry_mass_kg, frontal_area, thrust_initial_mass_kg))

        t_seg_out = np.atleast_1d(np.asarray(sol.t, dtype=np.float64))
        y_seg_out = np.asarray(sol.y, dtype=np.float64)
        if y_seg_out.ndim == 1:
            y_seg_out = y_seg_out.reshape((y_seg_out.size, 1))

        t_all.append(seg_t0 + t_seg_out)
        y_all.append(y_seg_out)

        if y_seg_out.shape[1] > 0:
            y0 = np.ascontiguousarray(y_seg_out[:, -1], dtype=np.float64)

        # Terminal event triggered?
        if getattr(sol, 'status', 0) == 1:
            if (getattr(sol, 't_events', None) is not None and
                    len(sol.t_events) > 0 and
                    np.asarray(sol.t_events[0]).size > 0):
                t_event_case = seg_t0 + float(sol.t_events[0][0])
            break

    if len(t_all) == 0:
        n_state = initial_state.size
        return (np.array([0.0], dtype=np.float64),
                initial_state.reshape((n_state, 1)), -1.0)

    t_cat = np.concatenate(t_all)
    y_cat = np.concatenate(y_all, axis=1)

    # Remove duplicate boundary rows (from segment joins)
    if t_cat.size > 1:
        keep = np.ones(t_cat.size, dtype=np.bool_)
        for j in range(1, t_cat.size):
            if t_cat[j] - t_cat[j-1] < 1e-12:
                keep[j] = False
        if not np.all(keep):
            t_cat = t_cat[keep]
            y_cat = y_cat[:, keep]

    return t_cat, y_cat, t_event_case

def _sample_batch_ballistic_coefficients(nsims, seed):
    rng = np.random.default_rng(seed)
    bc = rng.normal(ballistic_coefficient_nominal, batch_sigma_ballistic_coef, size=nsims)
    return np.clip(bc, 1e-12, None)

def _run_one_batch_case(case_id, oe_case, sat_id, start_timestamp,
                        x_sun_case, x_moon_case, gst0_case,
                        ballistic_coefficient_case, case_schedule=None,
                        cluster_meta=None):
    _warmup_numba_derivs()
    t_case, y_case, t_cut = _integrate_single_case(oe_case, ballistic_coefficient_case,
                                                   x_sun_case, x_moon_case, gst0_case,
                                                   start_timestamp=start_timestamp,
                                                   event_mode='batch_cutoff',
                                                   case_schedule=case_schedule)

    x_sat_case = np.ascontiguousarray(y_case[12:18, :], dtype=np.float64)
    final_state = x_sat_case[:, -1]
    terminated = 1 if t_cut >= 0.0 else 0
    start_ts_case = pd.Timestamp(start_timestamp)
    if pd.isna(start_ts_case):
        start_ts_case = pd.Timestamp(epoch)
    start_day_offset = (start_ts_case - pd.Timestamp(tle_earliest_start_epoch)).total_seconds() / 86400.0

    result = {'case_id': case_id, 'sat_id': str(sat_id),
              'a_km': float(oe_case[0]), 'e': float(oe_case[1]),
              'i_deg': float(np.rad2deg(oe_case[2])), 'w_deg': float(np.rad2deg(oe_case[3])),
              'OM_deg': float(np.rad2deg(oe_case[4])), 'Ma_deg': float(np.rad2deg(oe_case[5])),
              'start_timestamp': str(start_ts_case), 'start_day_offset': float(start_day_offset),
              'ballistic_coeff_m2_per_kg': float(ballistic_coefficient_case),
              'n_points': int(t_case.size), 'terminated_at_115km': terminated, 't_115_s': float(t_cut),
              'final_x_km': float(final_state[0]), 'final_y_km': float(final_state[1]),
              'final_z_km': float(final_state[2]),
              'final_vx_kms': float(final_state[3]), 'final_vy_kms': float(final_state[4]),
              'final_vz_kms': float(final_state[5]),
              'times': t_case, 'state_sat': x_sat_case}

    # Cluster metadata enrichment
    if cluster_meta is not None:
        result['global_cluster_id'] = cluster_meta.get('cluster_id', 0)
        result['cluster_weight_active'] = cluster_meta.get('cluster_weight_active', 1)
        result['cluster_weight_global'] = cluster_meta.get('cluster_weight_global', 1)
        result['pooled_role'] = cluster_meta.get('pooled_role', '')
        result['is_cluster_noise'] = cluster_meta.get('is_noise', False)
        result['representative_sat_id'] = cluster_meta.get('representative_sat_id', '')
        result['cluster_color_hex'] = cluster_meta.get('color_hex', cluster_noise_color)
        for field_name in _CLUSTER_POLICY_OUTPUT_FIELDS:
            if field_name in cluster_meta:
                result[field_name] = cluster_meta.get(field_name)

    # Thrust diagnostics (mass series and summary)
    if k_thrust == 1 and y_case.shape[0] >= 19:
        mass_series = np.asarray(y_case[18, :], dtype=np.float64)
        result['mass_series'] = mass_series
        if case_schedule is not None:
            thrust_summary = compute_case_thrust_summary(case_schedule, t_case, mass_series)
            result.update(thrust_summary)
        else:
            result['initial_mass_kg'] = float(mass_series[0])
            result['final_mass_kg'] = float(mass_series[-1])
            result['propellant_used_kg'] = float(mass_series[0] - mass_series[-1])
    else:
        result['initial_mass_kg'] = 0.0
        result['final_mass_kg'] = 0.0
        result['propellant_used_kg'] = 0.0

    return result

def _run_parallel_batch(oe_cases=None, sat_ids=None, start_timestamps=None, ballistic_coefficients=None,
                        workers_override=None, show_case_progress=True,
                        write_summary=True, write_trajectories=None):
    if oe_cases is None:
        if use_tle_initial_conditions == 1 and tle_oe_cases is not None and tle_oe_cases.shape[0] > 0:
            oe_cases = tle_oe_cases
            if sat_ids is None:
                sat_ids = tle_sat_ids_selected
        else:
            oe_cases = np.tile(oe_sat, (num_simulations, 1))

    oe_cases = np.ascontiguousarray(oe_cases, dtype=np.float64)
    nsims = int(oe_cases.shape[0])

    if sat_ids is None:
        sat_ids = [f"case_{k:05d}" for k in range(nsims)]
    elif len(sat_ids) != nsims:
        raise RuntimeError("Length of sat_ids must match number of oe_cases.")

    if start_timestamps is None:
        if use_tle_initial_conditions == 1 and tle_start_datetimes_selected is not None and len(tle_start_datetimes_selected) == nsims:
            start_timestamps = tle_start_datetimes_selected
        else:
            start_timestamps = [epoch] * nsims
    elif len(start_timestamps) != nsims:
        raise RuntimeError("Length of start_timestamps must match number of oe_cases.")

    if ballistic_coefficients is None:
        ballistic_coefficients = _sample_batch_ballistic_coefficients(nsims, batch_random_seed)
    ballistic_coefficients = np.asarray(ballistic_coefficients, dtype=np.float64)
    if ballistic_coefficients.size != nsims:
        raise RuntimeError("Length of ballistic_coefficients must match number of oe_cases.")

    # Build per-case major-body and GST initial conditions using each case epoch date.
    epoch_cache = {}
    x_sun_cases = np.zeros((nsims, 6), dtype=np.float64)
    x_moon_cases = np.zeros((nsims, 6), dtype=np.float64)
    gst0_cases = np.zeros(nsims, dtype=np.float64)
    epoch_date_keys = []

    for idx in range(nsims):
        ts = pd.Timestamp(start_timestamps[idx])
        if pd.isna(ts):
            ts = pd.Timestamp(epoch)
        date_key = ts.strftime('%Y-%m-%d')
        epoch_date_keys.append(date_key)

        cached = epoch_cache.get(date_key)
        if cached is None:
            mb_case, jd_case = load_mb(frame, date_key)
            x_sun_c = np.ascontiguousarray(mb_case[0, :], dtype=np.float64)
            x_moon_c = np.ascontiguousarray(mb_case[2, :], dtype=np.float64)
            gst0_c = float(np.deg2rad(gst0(jd_case)))
            cached = (x_sun_c, x_moon_c, gst0_c)
            epoch_cache[date_key] = cached

        x_sun_cases[idx, :] = cached[0]
        x_moon_cases[idx, :] = cached[1]
        gst0_cases[idx] = cached[2]

    # -------------------------------------------------------------------
    # Build per-satellite thrust schedules (main process, before dispatch)
    # -------------------------------------------------------------------
    case_schedules = [None] * nsims
    if k_thrust == 1 and phase_param_map is not None:
        n_with_schedule = 0
        for idx in range(nsims):
            ts = pd.Timestamp(start_timestamps[idx])
            if pd.isna(ts):
                ts = pd.Timestamp(epoch)
            t_final_idx = _seconds_until_date_cutoff(ts)
            if t_final_idx <= 0.0:
                continue
            policy_context = cluster_metadata_by_sat.get(str(sat_ids[idx])) if cluster_metadata_by_sat else None
            sched = build_case_schedule(
                sat_ids[idx],
                phase_intervals_df if phase_intervals_df is not None else pd.DataFrame(),
                phase_param_map, ts, t_final_idx,
                policy_context=policy_context)
            case_schedules[idx] = sched
            # Count satellites that have at least one non-coast segment
            if any(s['T_eff_N'] > 0 for s in sched):
                n_with_schedule += 1
        print(f"[Thrust] Built schedules: {n_with_schedule}/{nsims} satellites have thrust phases")

    if workers_override is None:
        workers = _resolve_batch_worker_count()
    else:
        workers = max(1, int(workers_override))

    if write_trajectories is None:
        write_trajectories = (save_batch_trajectories == 1)

    print(f"\nBatch mode enabled: {nsims} simulations")
    print(f"Max workers: {workers}")
    print(f"Thrust mode: {'ON (variable-mass, 19-state)' if k_thrust == 1 else 'OFF (classic 18-state)'}")
    print(f"Ballistic coefficient (nominal): {ballistic_coefficient_nominal:.5e} m^2/kg")
    print(f"Ballistic coefficient sigma: {batch_sigma_ballistic_coef:.5e} m^2/kg")
    print(f"Altitude cutoff after solve: {batch_cutoff_alt_km:.1f} km")
    print(f"Date cutoff after solve: {simulation_date_cutoff}")
    print(f"Unique start dates loaded: {len(epoch_cache)}")

    # Build per-case cluster metadata for worker dispatch
    cluster_meta_list = [None] * nsims
    if cluster_metadata_by_sat:
        for idx in range(nsims):
            sid = str(sat_ids[idx])
            meta = cluster_metadata_by_sat.get(sid)
            if meta is not None:
                cluster_meta_list[idx] = meta
        n_with_meta = sum(1 for m in cluster_meta_list if m is not None)
        print(f"[Cluster] Metadata attached: {n_with_meta}/{nsims} cases")

    batch_results = [None] * nsims

    _warmup_numba_derivs()

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_warmup_numba_derivs) as executor:
        futures = {executor.submit(_run_one_batch_case, idx, oe_cases[idx], sat_ids[idx],
                                   start_timestamps[idx], x_sun_cases[idx], x_moon_cases[idx],
                                   float(gst0_cases[idx]),
                                   float(ballistic_coefficients[idx]),
                                   case_schedules[idx],
                                   cluster_meta_list[idx]): idx
                   for idx in range(nsims)}

        completed = 0
        for fut in as_completed(futures):
            result = fut.result()
            batch_results[result['case_id']] = result
            completed += 1
            if show_case_progress:
                mass_info = ""
                if 'propellant_used_kg' in result and result['propellant_used_kg'] > 0.0:
                    mass_info = f" | prop={result['propellant_used_kg']:.3f} kg"
                if result['terminated_at_115km'] == 1:
                    print(f"Batch progress: {completed}/{nsims} | "
                          f"sat {result['sat_id']} done | cutoff at t={result['t_115_s']:.2f} s{mass_info}")
                else:
                    print(f"Batch progress: {completed}/{nsims} | "
                          f"sat {result['sat_id']} done | no cutoff{mass_info}")

    # ---- Write summary CSV ----
    if write_summary:
        os.makedirs(fd, exist_ok=True)
        summary_file = f'{fd}/batch_summary.csv'
        # Build header — base columns + optional thrust columns
        base_cols = ['case_id', 'sat_id', 'start_timestamp', 'start_day_offset',
                     'a_km', 'e', 'i_deg', 'w_deg', 'OM_deg', 'Ma_deg',
                     'ballistic_coeff_m2_per_kg',
                     'n_points', 'terminated_at_115km', 't_115_s',
                     'final_x_km', 'final_y_km', 'final_z_km',
                     'final_vx_kms', 'final_vy_kms', 'final_vz_kms']
        thrust_cols = ['initial_mass_kg', 'final_mass_kg', 'propellant_used_kg',
                       'cumulative_impulse_Ns', 'cumulative_energy_Wh',
                       'total_thrust_on_time_s', 'total_raise_time_s',
                       'total_shell_time_s', 'total_disposal_time_s']
        header = base_cols + thrust_cols
        # Append cluster columns when cluster features are active
        cluster_csv_cols = []
        if enable_global_cluster_features == 1 and cluster_metadata_by_sat:
            cluster_csv_cols = _CLUSTER_OUTPUT_FIELDS
            header = header + cluster_csv_cols

        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for item in batch_results:
                base_vals = [item['case_id'], item['sat_id'], item['start_timestamp'],
                             item['start_day_offset'],
                             item['a_km'], item['e'], item['i_deg'], item['w_deg'],
                             item['OM_deg'], item['Ma_deg'],
                             item['ballistic_coeff_m2_per_kg'],
                             item['n_points'], item['terminated_at_115km'], item['t_115_s'],
                             item['final_x_km'], item['final_y_km'], item['final_z_km'],
                             item['final_vx_kms'], item['final_vy_kms'], item['final_vz_kms']]
                thrust_vals = [item.get(c, '') for c in thrust_cols]
                cluster_vals = [item.get(c, '') for c in cluster_csv_cols]
                writer.writerow(base_vals + thrust_vals + cluster_vals)

    if write_trajectories:
        batch_dir = f'{fd}/batch_trajectories'
        os.makedirs(batch_dir, exist_ok=True)
        for item in batch_results:
            fn = f"{batch_dir}/state_vectors_sat_case_{item['case_id']:05d}.dat"
            tt = item['times']
            xx = item['state_sat']
            with open(fn, 'w') as fcase:
                for j in range(tt.size):
                    fcase.write(f"{tt[j]:<15.8e} {xx[0, j]:<15.8e} {xx[1, j]:<15.8e} {xx[2, j]:<15.8e} "
                                f"{xx[3, j]:<15.8e} {xx[4, j]:<15.8e} {xx[5, j]:<15.8e}\n")

    if write_summary:
        print(f"Batch summary saved to: {summary_file}")

    # ---- Write cluster-specific CSVs ----
    if enable_global_cluster_features == 1 and cluster_metadata_by_sat and write_summary:
        _write_cluster_csvs(batch_results, fd,
                            active_df=globals().get('_cluster_active_df'),
                            medoid_by_cluster=globals().get('_cluster_medoid_map'),
                            summary_info=globals().get('_cluster_summary'))
        print(f"[Cluster] Cluster CSVs saved to: {fd}/")

    return batch_results

# Pre-allocate arrays for EGM2008 and Derivs
MM = nmax + 1
P_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
Pl_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
sml_pre = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
cml_pre = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
derivatives_pre = np.ascontiguousarray(np.zeros(neq, dtype=np.float64))
tmp3_pre = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

state = None

@njit(cache=True)
def _state_to_orbital_series(x_state):
    n_samples = x_state.shape[1]
    orb_case = np.empty((n_samples, 6), dtype=np.float64)
    for j in range(n_samples):
        orb_case[j, :] = xyz2orb(earth_GM, x_state[0:3, j], x_state[3:6, j])
    return orb_case

def _run_with_msis_grids(t_eval_override=None, initial_state=None,
                         start_timestamp=None, ballistic_coefficient_case=None,
                         gst0_case=None, event_mode='reentry',
                         case_schedule=None):
    """Segmented integration so Numba RHS can use daily MSIS grids.

    When *case_schedule* is provided (list of segment dicts from
    thrust_helpers.build_case_schedule), the solver uses the 19-state
    Derivs_msis_thrust RHS and breaks at both MSIS day boundaries **and**
    phase boundaries so that thrust parameters remain constant within each
    piecewise segment.
    """

    meta = msis_load_meta(msis_grid_dir)
    index = MsisGridIndex(msis_grid_dir)

    # Replaced msis_parse_date with standard datetime
    day0 = datetime.strptime(msis_grid_start_date, '%Y-%m-%d').date()

    if initial_state is None:
        y0 = np.ascontiguousarray(Xb_init, dtype=np.float64)
    else:
        y0 = np.ascontiguousarray(initial_state, dtype=np.float64)

    if ballistic_coefficient_case is None:
        ballistic_coefficient_case = ballistic_coefficient_nominal
    ballistic_coefficient_case = float(ballistic_coefficient_case)

    if gst0_case is None:
        gst0_case = GST0
    gst0_case = float(gst0_case)

    start_ts = pd.Timestamp(start_timestamp) if start_timestamp is not None else pd.Timestamp(epoch)
    if pd.isna(start_ts):
        start_ts = pd.Timestamp(epoch)
    min_start = pd.Timestamp(tle_earliest_start_epoch)
    if start_ts < min_start:
        start_ts = min_start
    msis_ref_ts = pd.Timestamp(msis_grid_start_date)
    t_abs_shift = float((start_ts - msis_ref_ts).total_seconds())

    if event_mode == 'reentry':
        events = [Reentry]
    elif event_mode == 'batch_cutoff':
        events = [BatchCutoff]
    else:
        events = None

    # Pre-allocate arrays for EGM2008 and Derivs (same as the default path)
    MM = nmax + 1
    P_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    Pl_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    sml_pre = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    cml_pre = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    tmp3_pre = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

    @njit(cache=True, fastmath=True, nogil=True)
    def Derivs_msis(t, f, P, Pl, sml, cml, tmp3,
                    t_abs0,
                    ballistic_coefficient,
                    gst0_local,
                    grid_ut00, grid_ut06, grid_ut12, grid_ut18, grid_tomorrow_ut00,
                    lat0, dlat, lon0, dlon, alt_min_km, alt_step_km):
        Re = earth_Re

        x_sun = f[0:6]
        r_sun = np.sqrt(x_sun[0] ** 2 + x_sun[1] ** 2 + x_sun[2] ** 2)

        x_moon = f[6:12]
        r_moon = np.sqrt(x_moon[0] ** 2 + x_moon[1] ** 2 + x_moon[2] ** 2)

        x_sat = f[12:18]
        r = np.sqrt(x_sat[0] ** 2 + x_sat[1] ** 2 + x_sat[2] ** 2)

        # Sun EOM
        dxsundt = x_sun[3]
        dysundt = x_sun[4]
        dzsundt = x_sun[5]

        mu_sun_earth = sun_GM + earth_GM
        ddxsundt = -mu_sun_earth * x_sun[0] / r_sun ** 3
        ddysundt = -mu_sun_earth * x_sun[1] / r_sun ** 3
        ddzsundt = -mu_sun_earth * x_sun[2] / r_sun ** 3

        # Moon EOM
        dxmoondt = x_moon[3]
        dymoondt = x_moon[4]
        dzmoondt = x_moon[5]

        # Sun's perturbation on the Moon
        AC3b(x_moon[0:3], x_sun[0:3], sun_GM, tmp3)
        ac3b_sun0 = tmp3[0]
        ac3b_sun1 = tmp3[1]
        ac3b_sun2 = tmp3[2]

        # Earth's J2 perturbation on the Moon
        J2acc(earth_GM, earth_J2, earth_Re, x_moon[0:3], tmp3)
        acj20 = tmp3[0]
        acj21 = tmp3[1]
        acj22 = tmp3[2]

        mu_moon_earth = moon_GM + earth_GM
        ddxmoondt = -mu_moon_earth * x_moon[0] / r_moon ** 3 + acj20 + ac3b_sun0
        ddymoondt = -mu_moon_earth * x_moon[1] / r_moon ** 3 + acj21 + ac3b_sun1
        ddzmoondt = -mu_moon_earth * x_moon[2] / r_moon ** 3 + acj22 + ac3b_sun2

        # Accumulate spacecraft perturbations
        axp = 0.0
        ayp = 0.0
        azp = 0.0

        if k_EGM2008 == 1:
            EGM2008(nmax, mmax, x_sat[0:3], C, S, t_abs0 + t, earth_GM, earth_Re, earth_spin, gst0_local,
                    P, Pl, sml, cml, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if k_moon == 1:
            AC3b(x_sat[0:3], x_moon[0:3], moon_GM, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if k_sun == 1:
            AC3b(x_sat[0:3], x_sun[0:3], sun_GM, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if k_SRP == 1:
            SRPacc(x_sat[0:3], x_sun[0:3], AtoM, Cr, Re, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if r <= (900.0 + Re) and k_atm_drag == 1:
            atm_drag_msis_grid(x_sat, 1.0, ballistic_coefficient, earth_spin, gst0_local, t_abs0 + t,
                              grid_ut00, grid_ut06, grid_ut12, grid_ut18, grid_tomorrow_ut00,
                              earth_Re,
                              lat0, dlat, lon0, dlon,
                              alt_min_km, alt_step_km,
                              tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        # Spacecraft EOM
        dxdt = x_sat[3]
        dydt = x_sat[4]
        dzdt = x_sat[5]

        mu = earth_GM
        ddxdt = -mu * x_sat[0] / r ** 3 + axp
        ddydt = -mu * x_sat[1] / r ** 3 + ayp
        ddzdt = -mu * x_sat[2] / r ** 3 + azp

        derivatives = np.empty(18, dtype=np.float64)
        derivatives[0] = dxsundt
        derivatives[1] = dysundt
        derivatives[2] = dzsundt
        derivatives[3] = ddxsundt
        derivatives[4] = ddysundt
        derivatives[5] = ddzsundt
        derivatives[6] = dxmoondt
        derivatives[7] = dymoondt
        derivatives[8] = dzmoondt
        derivatives[9] = ddxmoondt
        derivatives[10] = ddymoondt
        derivatives[11] = ddzmoondt
        derivatives[12] = dxdt
        derivatives[13] = dydt
        derivatives[14] = dzdt
        derivatives[15] = ddxdt
        derivatives[16] = ddydt
        derivatives[17] = ddzdt
        return derivatives

    # ---- Thrust-enabled MSIS RHS (19-state) ----
    @njit(cache=True, fastmath=True, nogil=True)
    def Derivs_msis_thrust(t, f, P, Pl, sml, cml, tmp3,
                           t_abs0,
                           ballistic_coefficient,
                           gst0_local,
                           grid_ut00, grid_ut06, grid_ut12, grid_ut18, grid_tomorrow_ut00,
                           lat0, dlat, lon0, dlon, alt_min_km, alt_step_km,
                           T_eff_N, Isp_s, thrust_sign, dry_mass_kg,
                           frontal_area_val, initial_mass_kg):
        Re = earth_Re
        g0 = 9.80665

        x_sun = f[0:6]
        r_sun = np.sqrt(x_sun[0]**2 + x_sun[1]**2 + x_sun[2]**2)
        x_moon = f[6:12]
        r_moon = np.sqrt(x_moon[0]**2 + x_moon[1]**2 + x_moon[2]**2)
        x_sat = f[12:18]
        r = np.sqrt(x_sat[0]**2 + x_sat[1]**2 + x_sat[2]**2)
        m_sc = f[18]

        m_ref = m_sc if m_sc > 1.0 else 1.0
        AtoM_now = frontal_area_val / m_ref
        bc_now = ballistic_coefficient * (initial_mass_kg / m_ref)

        # Sun EOM
        dxsundt = x_sun[3]; dysundt = x_sun[4]; dzsundt = x_sun[5]
        mu_sun_earth = sun_GM + earth_GM
        ddxsundt = -mu_sun_earth * x_sun[0] / r_sun**3
        ddysundt = -mu_sun_earth * x_sun[1] / r_sun**3
        ddzsundt = -mu_sun_earth * x_sun[2] / r_sun**3

        # Moon EOM
        dxmoondt = x_moon[3]; dymoondt = x_moon[4]; dzmoondt = x_moon[5]
        AC3b(x_moon[0:3], x_sun[0:3], sun_GM, tmp3)
        ac3b_sun0, ac3b_sun1, ac3b_sun2 = tmp3[0], tmp3[1], tmp3[2]
        J2acc(earth_GM, earth_J2, earth_Re, x_moon[0:3], tmp3)
        acj20, acj21, acj22 = tmp3[0], tmp3[1], tmp3[2]
        mu_moon_earth = moon_GM + earth_GM
        ddxmoondt = -mu_moon_earth * x_moon[0] / r_moon**3 + acj20 + ac3b_sun0
        ddymoondt = -mu_moon_earth * x_moon[1] / r_moon**3 + acj21 + ac3b_sun1
        ddzmoondt = -mu_moon_earth * x_moon[2] / r_moon**3 + acj22 + ac3b_sun2

        # Spacecraft perturbations
        axp = 0.0; ayp = 0.0; azp = 0.0

        if k_EGM2008 == 1:
            EGM2008(nmax, mmax, x_sat[0:3], C, S, t_abs0 + t, earth_GM, earth_Re,
                    earth_spin, gst0_local, P, Pl, sml, cml, tmp3)
            axp += tmp3[0]; ayp += tmp3[1]; azp += tmp3[2]
        if k_moon == 1:
            AC3b(x_sat[0:3], x_moon[0:3], moon_GM, tmp3)
            axp += tmp3[0]; ayp += tmp3[1]; azp += tmp3[2]
        if k_sun == 1:
            AC3b(x_sat[0:3], x_sun[0:3], sun_GM, tmp3)
            axp += tmp3[0]; ayp += tmp3[1]; azp += tmp3[2]
        if k_SRP == 1:
            SRPacc(x_sat[0:3], x_sun[0:3], AtoM_now, Cr, Re, tmp3)
            axp += tmp3[0]; ayp += tmp3[1]; azp += tmp3[2]
        if r <= (900.0 + Re) and k_atm_drag == 1:
            atm_drag_msis_grid(x_sat, 1.0, bc_now, earth_spin, gst0_local, t_abs0 + t,
                              grid_ut00, grid_ut06, grid_ut12, grid_ut18, grid_tomorrow_ut00,
                              earth_Re, lat0, dlat, lon0, dlon, alt_min_km, alt_step_km, tmp3)
            axp += tmp3[0]; ayp += tmp3[1]; azp += tmp3[2]

        # Along-track thrust
        dm_dt = 0.0
        if T_eff_N > 0.0 and m_sc > dry_mass_kg:
            r_mag = r
            r_hat0 = x_sat[0]/r_mag; r_hat1 = x_sat[1]/r_mag; r_hat2 = x_sat[2]/r_mag
            hx = x_sat[1]*x_sat[5] - x_sat[2]*x_sat[4]
            hy = x_sat[2]*x_sat[3] - x_sat[0]*x_sat[5]
            hz = x_sat[0]*x_sat[4] - x_sat[1]*x_sat[3]
            h_mag = np.sqrt(hx*hx + hy*hy + hz*hz)
            if h_mag > 1.0e-12:
                h_hat0 = hx/h_mag; h_hat1 = hy/h_mag; h_hat2 = hz/h_mag
                t0 = h_hat1*r_hat2 - h_hat2*r_hat1
                t1 = h_hat2*r_hat0 - h_hat0*r_hat2
                t2 = h_hat0*r_hat1 - h_hat1*r_hat0
                t_mag_v = np.sqrt(t0*t0 + t1*t1 + t2*t2)
                if t_mag_v > 1.0e-12:
                    t0 /= t_mag_v; t1 /= t_mag_v; t2 /= t_mag_v
                    a_thrust_mag = (T_eff_N / m_sc) * 1.0e-3
                    axp += thrust_sign * a_thrust_mag * t0
                    ayp += thrust_sign * a_thrust_mag * t1
                    azp += thrust_sign * a_thrust_mag * t2
                    dm_dt = -T_eff_N / (g0 * Isp_s)

        dxdt = x_sat[3]; dydt = x_sat[4]; dzdt = x_sat[5]
        mu = earth_GM
        ddxdt = -mu * x_sat[0] / r**3 + axp
        ddydt = -mu * x_sat[1] / r**3 + ayp
        ddzdt = -mu * x_sat[2] / r**3 + azp

        derivatives = np.empty(19, dtype=np.float64)
        derivatives[0]  = dxsundt;  derivatives[1]  = dysundt;  derivatives[2]  = dzsundt
        derivatives[3]  = ddxsundt; derivatives[4]  = ddysundt; derivatives[5]  = ddzsundt
        derivatives[6]  = dxmoondt; derivatives[7]  = dymoondt; derivatives[8]  = dzmoondt
        derivatives[9]  = ddxmoondt; derivatives[10] = ddymoondt; derivatives[11] = ddzmoondt
        derivatives[12] = dxdt;  derivatives[13] = dydt;  derivatives[14] = dzdt
        derivatives[15] = ddxdt; derivatives[16] = ddydt; derivatives[17] = ddzdt
        derivatives[18] = dm_dt
        return derivatives

    # Build a segmented solve_ivp over day boundaries (+ phase boundaries when thrust active)
    use_thrust_msis = (case_schedule is not None and len(case_schedule) > 0)

    if t_eval_override is None:
        t_eval = np.asarray(tspan, dtype=np.float64)
    else:
        t_eval = np.asarray(t_eval_override, dtype=np.float64)

    neq_local = 19 if use_thrust_msis else 18
    if t_eval.size == 0:
        return np.array([0.0], dtype=np.float64), y0.reshape((neq_local, 1)), -1.0

    # If thrust active but y0 is only 18 elements, pad with initial mass
    if use_thrust_msis and y0.size == 18:
        y0_ext = np.zeros(19, dtype=np.float64)
        y0_ext[:18] = y0
        y0_ext[18] = thrust_initial_mass_kg
        y0 = y0_ext

    y_all = []
    t_all = []
    t_event_case = -1.0
    grid_cache = {}
    controller_state = {}

    # Compute MSIS day boundary times in simulation-relative coordinates
    t_eval_abs = t_eval + t_abs_shift
    day_idx_arr = np.floor(t_eval_abs / 86400.0).astype(np.int64)

    # Build unified breakpoints: MSIS day boundaries + phase boundaries
    day_boundary_set = set()
    for i in range(1, len(day_idx_arr)):
        if day_idx_arr[i] != day_idx_arr[i-1]:
            day_boundary_set.add(float(t_eval[i]))

    phase_boundary_set = set()
    if use_thrust_msis:
        from thrust_helpers import get_schedule_boundaries
        phase_boundary_set = set(get_schedule_boundaries(case_schedule))

    all_boundaries = sorted(day_boundary_set | phase_boundary_set)

    # Build segment ranges: [seg_start, seg_end) based on boundary times
    # Each segment gets a contiguous slice of t_eval
    t_eval_f = t_eval.astype(np.float64)
    seg_starts = [0]
    for bnd in all_boundaries:
        # Find the first t_eval index >= boundary
        idx = int(np.searchsorted(t_eval_f, bnd))
        if 0 < idx < len(t_eval_f) and idx not in seg_starts:
            seg_starts.append(idx)
    seg_ranges = []
    for si in range(len(seg_starts)):
        s = seg_starts[si]
        e = seg_starts[si+1] if si+1 < len(seg_starts) else len(t_eval)
        if s < e:
            seg_ranges.append((s, e))

    for s_idx, e_idx in seg_ranges:
        t_seg_case = t_eval[s_idx:e_idx]
        if len(t_seg_case) < 2:
            continue  # Skip single-point segments (can occur with large output_stride)
        t_case0 = float(t_seg_case[0])
        t_seg_local = t_seg_case - t_case0
        if t_seg_local[-1] <= 0.0:
            continue
        t0_abs = t_abs_shift + t_case0

        # Determine MSIS day for this segment
        di = int(np.floor((t_abs_shift + t_case0) / 86400.0))
        date_today = day0 + timedelta(days=di)
        date_tomorrow = day0 + timedelta(days=di + 1)

        p00 = index.path_for_date_ut(date_today, 0)
        p06 = index.path_for_date_ut(date_today, 21600)
        p12 = index.path_for_date_ut(date_today, 43200)
        p18 = index.path_for_date_ut(date_today, 64800)
        try:
            p00n = index.path_for_date_ut(date_tomorrow, 0)
        except Exception:
            p00n = p18

        def _load(p):
            key = str(p)
            cached = grid_cache.get(key)
            if cached is not None:
                return cached
            loaded = msis_load_grid(p, meta)
            grid_cache[key] = loaded
            return loaded

        g00 = _load(p00)
        g06 = _load(p06)
        g12 = _load(p12)
        g18 = _load(p18)
        g00n = _load(p00n)

        if use_thrust_msis:
            # Lookup thrust parameters for this segment
            seg_info = lookup_segment_for_time(case_schedule, t_case0)
            a_now, lambda_now, alt_now = _extract_guidance_state(y0)
            seg_info = resolve_segment_command(
                seg_info,
                current_a_km=a_now,
                current_lambda_rad=lambda_now,
                current_alt_km=alt_now,
                current_mass_kg=float(y0[18]) if y0.size > 18 else float(thrust_initial_mass_kg),
                dry_mass_kg=float(thrust_dry_mass_kg),
                initial_mass_kg=float(thrust_initial_mass_kg),
                segment_start_s=float(t_case0),
                controller_state=controller_state,
            )
            T_eff = float(seg_info['T_eff_N'])
            Isp_val = float(seg_info['Isp_s']) if seg_info['Isp_s'] > 0 else 1500.0
            ts_val = int(seg_info['sign'])
            dry_m = float(thrust_dry_mass_kg)
            fa_val = float(frontal_area)
            m0_val = float(thrust_initial_mass_kg)

            sol = solve_ivp(Derivs_msis_thrust, [0.0, float(t_seg_local[-1])], y0,
                            events=events, method='DOP853',
                            t_eval=t_seg_local, rtol=_solver_rtol, atol=_solver_atol,
                            args=(P_pre, Pl_pre, sml_pre, cml_pre, tmp3_pre,
                                  t0_abs, ballistic_coefficient_case, gst0_case,
                                  g00, g06, g12, g18, g00n,
                                  meta.lat0, meta.dlat, meta.lon0, meta.dlon,
                                  meta.alt_min_km, meta.alt_step_km,
                                  T_eff, Isp_val, ts_val, dry_m, fa_val, m0_val))
        else:
            sol = solve_ivp(Derivs_msis, [0.0, float(t_seg_local[-1])], y0,
                            events=events, method='DOP853',
                            t_eval=t_seg_local, rtol=_solver_rtol, atol=_solver_atol,
                            args=(P_pre, Pl_pre, sml_pre, cml_pre, tmp3_pre,
                                  t0_abs, ballistic_coefficient_case, gst0_case,
                                  g00, g06, g12, g18, g00n,
                                  meta.lat0, meta.dlat, meta.lon0, meta.dlon,
                                  meta.alt_min_km, meta.alt_step_km))

        t_seg_out = np.atleast_1d(np.asarray(sol.t, dtype=np.float64))
        y_seg_out = np.asarray(sol.y, dtype=np.float64)

        # Be robust to odd shapes (some environments may yield list-like outputs).
        if y_seg_out.ndim == 1:
            y_seg_out = y_seg_out.reshape((y_seg_out.size, 1))
        elif y_seg_out.ndim == 2 and y_seg_out.shape[0] != y0.size and y_seg_out.shape[1] == y0.size:
            y_seg_out = y_seg_out.T

        t_all.append(t_case0 + t_seg_out)
        if y_seg_out.size == 0 or y_seg_out.shape[0] == 0 or y_seg_out.shape[1] == 0:
            t_all.pop()  # remove the just-appended empty time array
            continue

        y_all.append(y_seg_out)
        y0 = y_seg_out[:, -1]

        # If a terminal event triggered (e.g., reentry), stop segmenting.
        if getattr(sol, "status", 0) == 1:
            if getattr(sol, 't_events', None) is not None and len(sol.t_events) > 0 and np.asarray(sol.t_events[0]).size > 0:
                t_event_case = t_case0 + float(sol.t_events[0][0])
            break

    if len(t_all) == 0 or len(y_all) == 0:
        raise RuntimeError("MSIS segmented integration produced no output")

    t_cat = np.concatenate(t_all)
    y_cat = np.concatenate(y_all, axis=1)
    return t_cat, y_cat, t_event_case

def _plot_batch_results(batch_results):
    if batch_results is None or len(batch_results) == 0:
        return

    if (plot_sv == 0 and plot_sv_3d_sat == 0 and plot_sma == 0 and
            plot_ecc == 0 and plot_rp_ra == 0):
        return

    import matplotlib.pyplot as plt

    n_cases_total = len(batch_results)
    case_indices = np.arange(n_cases_total, dtype=np.int64)

    time_series = []
    orb_series = []

    # XY trajectory overlay
    if plot_sv == 1:
        fig = plt.figure()
        for idx in case_indices:
            item = batch_results[int(idx)]
            xx = item['state_sat']
            plt.plot(xx[0, :], xx[1, :], linewidth=0.5, alpha=0.45)
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.title('Batch trajectories (x-y)')
        plt.grid(True)
        _finalize_plot('batch_trajectories_xy.png', fig)

    # 3D trajectory overlay
    if plot_sv_3d_sat == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for idx in case_indices:
            item = batch_results[int(idx)]
            xx = item['state_sat']
            ax.plot(xx[0, :], xx[1, :], xx[2, :], linewidth=0.5, alpha=0.45)
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_zlabel('z (km)')
        ax.set_title('Batch trajectories (3D)')
        _finalize_plot('batch_trajectories_3d.png', fig)

    # Orbital element trend overlays (computed from sampled points)
    if plot_sma == 1 or plot_ecc == 1 or plot_rp_ra == 1:
        for idx in case_indices:
            item = batch_results[int(idx)]
            times_case = item['times']
            xx = item['state_sat']

            if plot_batch_absolute_time == 1:
                t_plot = item['start_day_offset'] + (times_case / 86400.0)
            else:
                t_plot = times_case / tu_conv
            orb_case = _state_to_orbital_series(xx)

            time_series.append(t_plot)
            orb_series.append(orb_case)

        if plot_sma == 1:
            fig = plt.figure()
            for t_plot, orb_case in zip(time_series, orb_series):
                y = orb_case[:, 0]
                valid = np.isfinite(t_plot) & np.isfinite(y)
                if np.any(valid):
                    plt.plot(t_plot[valid], y[valid], linewidth=0.5, alpha=0.45)
            if plot_batch_absolute_time == 1:
                plt.xlabel('Time since ' + tle_earliest_start_epoch + ' (days)')
            else:
                plt.xlabel('Time (' + unit + ')')
            plt.ylabel('Semi-major axis (km)')
            plt.title('Batch semi-major axis vs. Time')
            plt.grid(True)
            _finalize_plot('batch_sma_vs_time.png', fig)

        if plot_ecc == 1:
            fig = plt.figure()
            any_ecc = False
            for t_plot, orb_case in zip(time_series, orb_series):
                y = orb_case[:, 1]
                valid = np.isfinite(t_plot) & np.isfinite(y)
                if np.any(valid):
                    plt.plot(t_plot[valid], y[valid], linewidth=0.5, alpha=0.45)
                    any_ecc = True

            if not any_ecc:
                print("Batch eccentricity plot: no finite eccentricity samples to draw.")
            if plot_batch_absolute_time == 1:
                plt.xlabel('Time since ' + tle_earliest_start_epoch + ' (days)')
            else:
                plt.xlabel('Time (' + unit + ')')
            plt.ylabel('Eccentricity')
            plt.title('Batch eccentricity vs. Time')
            plt.grid(True)
            _finalize_plot('batch_eccentricity_vs_time.png', fig)

        if plot_rp_ra == 1:
            fig = plt.figure()
            for t_plot, orb_case in zip(time_series, orb_series):
                rp_case = orb_case[:, 0] * (1.0 - orb_case[:, 1]) - earth_Re
                ra_case = orb_case[:, 0] * (1.0 + orb_case[:, 1]) - earth_Re

                valid_rp = np.isfinite(t_plot) & np.isfinite(rp_case)
                valid_ra = np.isfinite(t_plot) & np.isfinite(ra_case)

                if np.any(valid_rp):
                    plt.plot(t_plot[valid_rp], rp_case[valid_rp], linewidth=0.5, alpha=0.45)
                if np.any(valid_ra):
                    plt.plot(t_plot[valid_ra], ra_case[valid_ra], linewidth=0.5, alpha=0.25)
            if plot_batch_absolute_time == 1:
                plt.xlabel('Time since ' + tle_earliest_start_epoch + ' (days)')
            else:
                plt.xlabel('Time (' + unit + ')')
            plt.ylabel('Altitude (km)')
            plt.title('Batch perigee/apogee altitude vs. Time')
            plt.grid(True)
            _finalize_plot('batch_perigee_apogee_vs_time.png', fig)

    # ------------------------------------------------------------------
    # Cluster-aware plots
    # ------------------------------------------------------------------
    _has_cluster = (enable_global_cluster_features == 1 and cluster_metadata_by_sat
                    and any('global_cluster_id' in r for r in batch_results))
    if _has_cluster and (plot_sma == 1 or plot_ecc == 1 or plot_rp_ra == 1):
        # Build color list per case
        case_colors = []
        case_cids = []
        for idx in case_indices:
            item = batch_results[int(idx)]
            cid = item.get('global_cluster_id', 0)
            case_cids.append(cid)
            case_colors.append(item.get('cluster_color_hex', cluster_noise_color))

        unique_cids = sorted(set(case_cids))
        cmap_cluster = _build_cluster_color_map(unique_cids)
        n_legend = len([c for c in unique_cids if c > 0])
        show_legend = n_legend <= 15

        is_medoid = (cluster_run_mode == "medoids")
        title_suffix = " (medoid representatives)" if is_medoid else ""

        # Precompute time/orb series if not already done
        if len(time_series) == 0:
            for idx in case_indices:
                item = batch_results[int(idx)]
                times_case = item['times']
                xx = item['state_sat']
                if plot_batch_absolute_time == 1:
                    t_plot = item['start_day_offset'] + (times_case / 86400.0)
                else:
                    t_plot = times_case / tu_conv
                orb_case = _state_to_orbital_series(xx)
                time_series.append(t_plot)
                orb_series.append(orb_case)

        xlabel_str = ('Time since ' + tle_earliest_start_epoch + ' (days)'
                      if plot_batch_absolute_time == 1 else 'Time (' + unit + ')')

        # 1. batch_sma_vs_time_by_cluster.png
        if plot_sma == 1:
            fig = plt.figure()
            plotted_labels = set()
            for idx in range(n_cases_total):
                t_plot = time_series[idx]
                y = orb_series[idx][:, 0]
                valid = np.isfinite(t_plot) & np.isfinite(y)
                if not np.any(valid):
                    continue
                cid = case_cids[idx]
                lbl = f"Cluster {cid}" if (cid > 0 and cid not in plotted_labels and show_legend) else \
                      ("Noise" if (cid <= 0 and "Noise" not in plotted_labels and show_legend) else None)
                if lbl:
                    plotted_labels.add(cid if cid > 0 else "Noise")
                plt.plot(t_plot[valid], y[valid],
                         color=case_colors[idx],
                         linewidth=cluster_linewidth_individual,
                         alpha=cluster_alpha_individual,
                         label=lbl)
            plt.xlabel(xlabel_str)
            plt.ylabel('Semi-major axis (km)')
            plt.title('SMA vs. Time by Cluster' + title_suffix)
            plt.grid(True)
            if show_legend:
                plt.legend(fontsize=8, ncol=max(1, n_legend // 8))
            _finalize_plot('batch_sma_vs_time_by_cluster.png', fig)

        # 2. batch_sma_cluster_envelopes.png
        if plot_sma == 1:
            from collections import defaultdict
            cluster_traces = defaultdict(list)
            cluster_time_traces = defaultdict(list)
            for idx in range(n_cases_total):
                cid = case_cids[idx]
                cluster_traces[cid].append(orb_series[idx][:, 0])
                cluster_time_traces[cid].append(time_series[idx])

            fig = plt.figure()
            q_lo, q_med, q_hi = cluster_summary_quantiles
            for cid in sorted(cluster_traces.keys()):
                traces = cluster_traces[cid]
                t_traces = cluster_time_traces[cid]
                if len(traces) == 0:
                    continue
                color = cmap_cluster.get(cid, cluster_noise_color)
                # Use shortest common time grid (interpolation-free: just plot each)
                if len(traces) == 1:
                    vld = np.isfinite(t_traces[0]) & np.isfinite(traces[0])
                    if np.any(vld):
                        lbl = f"Cluster {cid}" if cid > 0 else "Noise"
                        plt.plot(t_traces[0][vld], traces[0][vld],
                                 color=color, linewidth=1.0, label=lbl)
                    continue
                # For multiple traces, build common grid via min/max bounds
                t_min = max(np.nanmin(t) for t in t_traces if len(t) > 0)
                t_max = min(np.nanmax(t) for t in t_traces if len(t) > 0)
                if t_max <= t_min:
                    continue
                n_grid = min(500, min(len(t) for t in t_traces))
                t_grid = np.linspace(t_min, t_max, n_grid)
                interp_vals = np.full((len(traces), n_grid), np.nan)
                for j, (tt, yy) in enumerate(zip(t_traces, traces)):
                    vld = np.isfinite(tt) & np.isfinite(yy)
                    if np.sum(vld) >= 2:
                        interp_vals[j, :] = np.interp(t_grid, tt[vld], yy[vld])
                med = np.nanmedian(interp_vals, axis=0)
                lo = np.nanquantile(interp_vals, q_lo, axis=0)
                hi = np.nanquantile(interp_vals, q_hi, axis=0)
                lbl = f"Cluster {cid}" if cid > 0 else "Noise"
                plt.fill_between(t_grid, lo, hi, color=color, alpha=0.2)
                plt.plot(t_grid, med, color=color, linewidth=1.0, label=lbl)
            plt.xlabel(xlabel_str)
            plt.ylabel('Semi-major axis (km)')
            plt.title('SMA Cluster Envelopes' + title_suffix)
            plt.grid(True)
            if show_legend:
                plt.legend(fontsize=8, ncol=max(1, n_legend // 8))
            _finalize_plot('batch_sma_cluster_envelopes.png', fig)

        # 3. batch_cluster_counts.png
        fig = plt.figure()
        from collections import Counter
        cid_counts = Counter(case_cids)
        cids_sorted = sorted(cid_counts.keys())
        bar_colors = [cmap_cluster.get(c, cluster_noise_color) for c in cids_sorted]
        bar_labels = [f"C{c}" if c > 0 else "Noise" for c in cids_sorted]
        plt.bar(range(len(cids_sorted)), [cid_counts[c] for c in cids_sorted],
                color=bar_colors, edgecolor='black', linewidth=0.3)
        plt.xticks(range(len(cids_sorted)), bar_labels, rotation=45, fontsize=8)
        plt.xlabel('Global Cluster ID')
        plt.ylabel('Simulated Cases')
        plt.title('Active Cases per Cluster' + title_suffix)
        plt.grid(True, axis='y')
        _finalize_plot('batch_cluster_counts.png', fig)

        # 4. batch_cluster_terminal_altitude_summary.png
        from collections import defaultdict as _dd
        final_alt_by_cluster = _dd(list)
        for idx in range(n_cases_total):
            item = batch_results[int(idx)]
            xs = item.get('state_sat')
            if xs is not None and xs.shape[1] > 0:
                r_final = np.sqrt(xs[0, -1]**2 + xs[1, -1]**2 + xs[2, -1]**2)
                alt_km = r_final - earth_Re
                cid = case_cids[idx]
                final_alt_by_cluster[cid].append(alt_km)

        if final_alt_by_cluster:
            fig = plt.figure()
            cids_s = sorted(final_alt_by_cluster.keys())
            box_data = [final_alt_by_cluster[c] for c in cids_s]
            bp = plt.boxplot(box_data, patch_artist=True, tick_labels=[f"C{c}" if c > 0 else "N" for c in cids_s])
            for patch, cid in zip(bp['boxes'], cids_s):
                patch.set_facecolor(cmap_cluster.get(cid, cluster_noise_color))
                patch.set_alpha(0.6)
            plt.xlabel('Global Cluster ID')
            plt.ylabel('Final Altitude (km)')
            plt.title('Terminal Altitude by Cluster' + title_suffix)
            plt.xticks(rotation=45, fontsize=8)
            plt.grid(True, axis='y')
            _finalize_plot('batch_cluster_terminal_altitude_summary.png', fig)


# ======================================================================
# Public batch API for external callers (Chapter 7 optimization, etc.)
# ======================================================================

def run_batch_cases(
    oe_cases,
    sat_ids,
    start_timestamps,
    ballistic_coefficients,
    case_schedules=None,
    cluster_meta_map=None,
    write_outputs=False,
    return_trajectories=True,
    return_mass_series=True,
    compact_payload=False,
    workers_override=None,
    solver_rtol=None,
    solver_atol=None,
    max_prop_time_s=None,
    output_stride=None,
):
    """Run a batch of orbit propagation cases and return result dicts.

    This is the primary programmatic entry-point for the simulator,
    designed for use by optimizers and analysis scripts.  It delegates to
    the same integration machinery used by ``_run_parallel_batch`` /
    ``_run_one_batch_case`` but avoids plot generation and file I/O
    unless explicitly requested.

    Parameters
    ----------
    oe_cases : ndarray, shape (N, 6)
        Orbital elements per case [a(km), e, i(rad), w(rad), OM(rad), Ma(rad)].
    sat_ids : list[str]
        Satellite identifiers (length N).
    start_timestamps : list[str | pd.Timestamp]
        Start epoch per case (length N).
    ballistic_coefficients : ndarray, shape (N,)
        Cd*A/m in m^2/kg per case.
    case_schedules : list[list[dict] | None] | None
        Per-case thrust schedules (from ``build_case_schedule``).  When
        *None*, schedules are built automatically if thrust is enabled.
    cluster_meta_map : dict | None
        ``{sat_id: metadata_dict}`` for cluster enrichment.
    write_outputs : bool
        Write summary CSV and trajectory files (default False).
    return_trajectories : bool
        Include ``times`` and ``state_sat`` arrays in each result dict.
    return_mass_series : bool
        Include ``mass_series`` array in each result dict (thrust mode).
    compact_payload : bool
        When True, strip bulky arrays (times, state_sat, mass_series) to
        reduce memory.  Overrides *return_trajectories* and
        *return_mass_series*.
    workers_override : int | None
        Max parallel workers (default: use module setting).
    solver_rtol : float | None
        Override integrator relative tolerance (default: module _solver_rtol).
    solver_atol : float | None
        Override integrator absolute tolerance (default: module _solver_atol).
    max_prop_time_s : float | None
        Cap propagation time per case in seconds (e.g. for horizon_fraction).
    output_stride : int | None
        Thin t_eval by this factor (1=full res, 100=every 100th pt).

    Returns
    -------
    list[dict]
        One result dict per case, same structure as ``_run_one_batch_case``
        output.  When *compact_payload* is True, large arrays are removed.
    """
    # ---- Apply solver configuration overrides ----
    global _solver_rtol, _solver_atol, _max_prop_time_s, _output_stride
    saved_rtol, saved_atol, saved_max_t = _solver_rtol, _solver_atol, _max_prop_time_s
    saved_stride = _output_stride
    try:
        if solver_rtol is not None:
            _solver_rtol = float(solver_rtol)
        if solver_atol is not None:
            _solver_atol = float(solver_atol)
        if max_prop_time_s is not None:
            _max_prop_time_s = float(max_prop_time_s)
        if output_stride is not None:
            _output_stride = int(output_stride)

        return _run_batch_cases_impl(
            oe_cases, sat_ids, start_timestamps, ballistic_coefficients,
            case_schedules=case_schedules,
            cluster_meta_map=cluster_meta_map,
            write_outputs=write_outputs,
            return_trajectories=return_trajectories,
            return_mass_series=return_mass_series,
            compact_payload=compact_payload,
            workers_override=workers_override,
        )
    finally:
        _solver_rtol, _solver_atol, _max_prop_time_s = saved_rtol, saved_atol, saved_max_t
        _output_stride = saved_stride


def _run_batch_cases_impl(
    oe_cases,
    sat_ids,
    start_timestamps,
    ballistic_coefficients,
    case_schedules=None,
    cluster_meta_map=None,
    write_outputs=False,
    return_trajectories=True,
    return_mass_series=True,
    compact_payload=False,
    workers_override=None,
):
    oe_cases = np.ascontiguousarray(oe_cases, dtype=np.float64)
    nsims = int(oe_cases.shape[0])
    ballistic_coefficients = np.asarray(ballistic_coefficients, dtype=np.float64)

    if len(sat_ids) != nsims:
        raise ValueError("sat_ids length must match oe_cases rows")
    if len(start_timestamps) != nsims:
        raise ValueError("start_timestamps length must match oe_cases rows")
    if ballistic_coefficients.size != nsims:
        raise ValueError("ballistic_coefficients size must match oe_cases rows")

    # ---- Build per-case major-body and GST initial conditions ----
    global _batch_epoch_cache
    x_sun_cases = np.zeros((nsims, 6), dtype=np.float64)
    x_moon_cases = np.zeros((nsims, 6), dtype=np.float64)
    gst0_cases = np.zeros(nsims, dtype=np.float64)

    for idx in range(nsims):
        ts = pd.Timestamp(start_timestamps[idx])
        if pd.isna(ts):
            ts = pd.Timestamp(epoch)
        date_key = ts.strftime('%Y-%m-%d')
        cached = _batch_epoch_cache.get(date_key)
        if cached is None:
            mb_case, jd_case = load_mb(frame, date_key)
            x_sun_c = np.ascontiguousarray(mb_case[0, :], dtype=np.float64)
            x_moon_c = np.ascontiguousarray(mb_case[2, :], dtype=np.float64)
            gst0_c = float(np.deg2rad(gst0(jd_case)))
            cached = (x_sun_c, x_moon_c, gst0_c)
            _batch_epoch_cache[date_key] = cached
        x_sun_cases[idx, :] = cached[0]
        x_moon_cases[idx, :] = cached[1]
        gst0_cases[idx] = cached[2]

    # ---- Build or accept thrust schedules ----
    if case_schedules is None:
        case_schedules = [None] * nsims
        if k_thrust == 1 and phase_param_map is not None:
            for idx in range(nsims):
                ts = pd.Timestamp(start_timestamps[idx])
                if pd.isna(ts):
                    ts = pd.Timestamp(epoch)
                t_final_idx = _seconds_until_date_cutoff(ts)
                if t_final_idx <= 0.0:
                    continue
                policy_context = cluster_meta_map.get(str(sat_ids[idx])) if cluster_meta_map is not None else None
                case_schedules[idx] = build_case_schedule(
                    sat_ids[idx],
                    phase_intervals_df if phase_intervals_df is not None else pd.DataFrame(),
                    phase_param_map, ts, t_final_idx,
                    policy_context=policy_context)

    # ---- Build per-case cluster metadata ----
    cluster_meta_list = [None] * nsims
    if cluster_meta_map is not None:
        for idx in range(nsims):
            meta = cluster_meta_map.get(str(sat_ids[idx]))
            if meta is not None:
                cluster_meta_list[idx] = meta

    # ---- Dispatch propagation ----
    _warmup_numba_derivs()
    workers = workers_override if workers_override is not None else _resolve_batch_worker_count()
    workers = max(1, int(workers))

    batch_results = [None] * nsims

    # Fast path: use threads for small batches (avoids process-spawn overhead;
    # Numba nogil=True releases the GIL during derivative evaluation so
    # concurrent solve_ivp calls overlap their heavy computation).
    if nsims <= 1:
        for idx in range(nsims):
            result = _run_one_batch_case(
                idx, oe_cases[idx], sat_ids[idx],
                start_timestamps[idx], x_sun_cases[idx], x_moon_cases[idx],
                float(gst0_cases[idx]), float(ballistic_coefficients[idx]),
                case_schedules[idx], cluster_meta_list[idx],
            )
            batch_results[result['case_id']] = result
    elif nsims <= max(3, workers):
        with ThreadPoolExecutor(max_workers=min(nsims, workers)) as texec:
            futures = {
                texec.submit(
                    _run_one_batch_case, idx, oe_cases[idx], sat_ids[idx],
                    start_timestamps[idx], x_sun_cases[idx], x_moon_cases[idx],
                    float(gst0_cases[idx]), float(ballistic_coefficients[idx]),
                    case_schedules[idx], cluster_meta_list[idx],
                ): idx
                for idx in range(nsims)
            }
            for fut in as_completed(futures):
                result = fut.result()
                batch_results[result['case_id']] = result
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=_warmup_numba_derivs) as executor:
            futures = {
                executor.submit(
                    _run_one_batch_case, idx, oe_cases[idx], sat_ids[idx],
                    start_timestamps[idx], x_sun_cases[idx], x_moon_cases[idx],
                    float(gst0_cases[idx]), float(ballistic_coefficients[idx]),
                    case_schedules[idx], cluster_meta_list[idx],
                ): idx
                for idx in range(nsims)
            }
            for fut in as_completed(futures):
                result = fut.result()
                batch_results[result['case_id']] = result

    # ---- Optional: write outputs ----
    if write_outputs:
        os.makedirs(fd, exist_ok=True)
        summary_file = f'{fd}/batch_summary.csv'
        base_cols = ['case_id', 'sat_id', 'start_timestamp', 'start_day_offset',
                     'a_km', 'e', 'i_deg', 'w_deg', 'OM_deg', 'Ma_deg',
                     'ballistic_coeff_m2_per_kg',
                     'n_points', 'terminated_at_115km', 't_115_s',
                     'final_x_km', 'final_y_km', 'final_z_km',
                     'final_vx_kms', 'final_vy_kms', 'final_vz_kms']
        thrust_cols = ['initial_mass_kg', 'final_mass_kg', 'propellant_used_kg']
        header = base_cols + thrust_cols
        if any('global_cluster_id' in item for item in batch_results):
            header = header + _CLUSTER_OUTPUT_FIELDS
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for item in batch_results:
                row = [item.get(c, '') for c in header]
                writer.writerow(row)

    # ---- Payload trimming ----
    if compact_payload:
        for item in batch_results:
            item.pop('times', None)
            item.pop('state_sat', None)
            item.pop('mass_series', None)
    else:
        if not return_trajectories:
            for item in batch_results:
                item.pop('times', None)
                item.pop('state_sat', None)
        if not return_mass_series:
            for item in batch_results:
                item.pop('mass_series', None)

    return batch_results


def main():
    start = timeit.default_timer()

    print(f"\nIntegration process:")
    print(f"TF = {tf / tu_conv:.2f} {unit}")
    print(f"dt = {dt / tu_conv:.2e} {unit}")
    print(f"nt = {nt}")
    print('\nRunning...\n')

    if batch_mode == 1:
        batch_results = _run_parallel_batch()
        stop = timeit.default_timer()
        runtime = stop - start
        if runtime < 60.0:
            print(f"Batch runtime = {runtime:.2f} seconds.\n")
        elif runtime >= 60.0 and runtime < 3600.0:
            print(f"Batch runtime = {runtime / 60.0:.2f} minutes.\n")
        else:
            print(f"Batch runtime = {runtime / 3600.0:.2f} hours.\n")

        _plot_batch_results(batch_results)
        return

    t_final_main = _seconds_until_date_cutoff(epoch)
    t_eval_main = _build_t_eval_with_cutoff(t_final_main)
    print(f"Date cutoff after solve: {simulation_date_cutoff}")
    if k_thrust == 1:
        print(f"Thrust mode: ON (variable-mass, 19-state)")

    # ---- Build single-run thrust schedule (if applicable) ----
    single_schedule = None
    if k_thrust == 1 and phase_param_map is not None:
        # Use first sat_id if available from TLE data
        single_sat_id = tle_sat_ids_selected[0] if (use_tle_initial_conditions == 1
            and tle_sat_ids_selected is not None and len(tle_sat_ids_selected) > 0) else "single_run"
        single_policy_context = cluster_metadata_by_sat.get(single_sat_id) if cluster_metadata_by_sat else None
        single_schedule = build_case_schedule(
            single_sat_id,
            phase_intervals_df if phase_intervals_df is not None else pd.DataFrame(),
            phase_param_map, pd.Timestamp(epoch), t_final_main,
            policy_context=single_policy_context)
        n_active = sum(1 for s in single_schedule if s['T_eff_N'] > 0)
        print(f"[Thrust] Single-run schedule: {n_active} active / {len(single_schedule)} total segments")

    # ---- Single-run cluster metadata ----
    if enable_global_cluster_features == 1 and cluster_metadata_by_sat:
        _single_sid = (tle_sat_ids_selected[0] if (use_tle_initial_conditions == 1
            and tle_sat_ids_selected is not None and len(tle_sat_ids_selected) > 0) else None)
        if _single_sid and _single_sid in cluster_metadata_by_sat:
            _sm = cluster_metadata_by_sat[_single_sid]
            print(f"\n--- Single-run cluster info ---")
            print(f"  Satellite ID      : {_single_sid}")
            print(f"  Global cluster ID : {_sm.get('cluster_id', 'N/A')}")
            print(f"  Is noise          : {_sm.get('is_noise', 'N/A')}")
            print(f"  Pooled role       : {_sm.get('pooled_role', 'N/A')}")
            print(f"  Cluster weight (active) : {_sm.get('cluster_weight_active', 'N/A')}")
            print(f"  Cluster weight (global) : {_sm.get('cluster_weight_global', 'N/A')}")
            if _sm.get('cluster_policy_applied'):
                print(f"  Policy source     : {_sm.get('cluster_policy_source', 'N/A')}")
                print(f"  Policy offsets    : da={_sm.get('policy_delta_a_km', 0.0):+.3f} km, "
                      f"dOM={np.rad2deg(_sm.get('policy_delta_Omega_rad', 0.0)):+.3f} deg, "
                      f"dlam={np.rad2deg(_sm.get('policy_delta_lambda_rad', 0.0)):+.3f} deg")
            if _sm.get('pooled_role') == 'medoid':
                print(f"  *** This satellite is a MEDOID representative ***")

    if atm_model == 1 and k_atm_drag == 1:
        times, state, reentry_t_abs = _run_with_msis_grids(
            t_eval_override=t_eval_main, case_schedule=single_schedule)
    elif k_thrust == 1 and single_schedule is not None:
        # USSA76 + thrust: segmented propagation
        times, state, reentry_t_abs = _run_segmented_thrust_case(
            t_eval_full=t_eval_main,
            initial_state=Xb_init,
            gst0_case=GST0,
            ballistic_coefficient_case=ballistic_coefficient_nominal,
            event_mode='reentry',
            case_schedule=single_schedule)
    else:
        if t_final_main <= 0.0:
            times = np.array([0.0], dtype=np.float64)
            state = Xb_init.reshape((neq, 1))
            reentry_t_abs = -1.0
        else:
            solution = solve_ivp(Derivs, [0.0, t_final_main], Xb_init, events=[Reentry], method='DOP853',
                             t_eval=t_eval_main, rtol=_solver_rtol, atol=_solver_atol,
                         args=(P_pre, Pl_pre, sml_pre, cml_pre, tmp3_pre,
                             ballistic_coefficient_nominal, GST0))
            state = solution.y
            times = solution.t
            if getattr(solution, 't_events', None) is not None and len(solution.t_events) > 0 and np.asarray(solution.t_events[0]).size > 0:
                reentry_t_abs = float(solution.t_events[0][0])
            else:
                reentry_t_abs = -1.0

    if reentry_t_abs >= 0.0:
        r_reentry = np.sqrt(state[12, -1] ** 2 + state[13, -1] ** 2 + state[14, -1] ** 2)
        print(f"Reentry: t = {reentry_t_abs / tu_conv} tu, r = {r_reentry} km")

    x_sun = state[0:6, :]
    x_moon = state[6:12, :]
    x_sat = state[12:18, :]
    mass_series = state[18, :] if state.shape[0] > 18 else None

    stop = timeit.default_timer()
    runtime = stop - start
    if runtime < 60.0:
        print(f"Runtime = {runtime:.2f} seconds.\n")
    elif runtime >= 60.0 and runtime < 3600.0:
        print(f"Runtime = {runtime / 60.0:.2f} minutes.\n")
    else:
        print(f"Runtime = {runtime / 3600.0:.2f} hours.\n")

    ################################################################################
    ################################################################################
    ################################################################################
    # Data analysis

    n = len(times)

    # Plot the satellite's orbit (x-y plane)
    if plot_sv == 1:
        fig = plt.figure()
        plt.plot(x_sat[0, :], x_sat[1, :], linewidth=0.5)
        _finalize_plot('single_trajectory_xy.png', fig)

    # Plot the satellite's orbit (3D)
    if plot_sv_3d_sat == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_sat[0, :], x_sat[1, :], x_sat[2, :], linewidth=0.5)
        _finalize_plot('single_trajectory_3d.png', fig)

    # Plot the sun's orbit (3D)
    if plot_sv_3d_sun == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_sun[0, :], x_sun[1, :], x_sun[2, :], linewidth=0.5)
        _finalize_plot('single_sun_trajectory_3d.png', fig)

    # Create a folder to store the results
    os.makedirs(fd, exist_ok=True)

    # Save the state vectors of the satellite in a file
    filename = f'{fd}/state_vectors_sat.dat'
    with open(filename, 'w') as f:
        for j in range(n):
            f.write(f"{times[j]:<15.8e} {x_sat[0, j]:<15.8e} {x_sat[1, j]:<15.8e} {x_sat[2, j]:<15.8e} {x_sat[3, j]:<15.8e} {x_sat[4, j]:<15.8e} {x_sat[5, j]:<15.8e}\n")

    # Save the state vectors of the major bodies in a file
    if save_mb_sv == 1:
        filename1 = f'{fd}/state_vectors_sun.dat'
        filename2 = f'{fd}/state_vectors_moon.dat'
        with open(filename1, 'w') as f:
            for j in range(n):
                f.write(f"{times[j]:<15.8e} {x_sun[0, j]:<15.8e} {x_sun[1, j]:<15.8e} {x_sun[2, j]:<15.8e} {x_sun[3, j]:<15.8e} {x_sun[4, j]:<15.8e} {x_sun[5, j]:<15.8e}\n")
        with open(filename2, 'w') as f:
            for j in range(n):
                f.write(f"{times[j]:<15.8e} {x_moon[0, j]:<15.8e} {x_moon[1, j]:<15.8e} {x_moon[2, j]:<15.8e} {x_moon[3, j]:<15.8e} {x_moon[4, j]:<15.8e} {x_moon[5, j]:<15.8e}\n")

    # Convert the state vector to orbital elements
    orb = []
    rp = np.zeros(n)
    ra = np.zeros(n)
    for j in range(n):
        oe_sat_local = xyz2orb(earth_GM, x_sat[0:3, j], x_sat[3:6, j])
        orb.append(oe_sat_local)

        # Calculate the perigee and apogee altitudes to test the decay due to the atmospheric drag
        rp[j] = oe_sat_local[0] * (1.0 - oe_sat_local[1]) - earth_Re
        ra[j] = oe_sat_local[0] * (1.0 + oe_sat_local[1]) - earth_Re
    orb = np.array(orb)
    time = np.array(times) / tu_conv

    # Plot semi-major axis vs. time
    if plot_sma == 1:
        fig = plt.figure()
        plt.plot(time, orb[:, 0], linewidth=0.5)
        plt.xlabel('Time (' + unit + ')')
        plt.ylabel('Semi-major axis (km)')
        plt.title('Semi-major axis vs. Time')
        plt.grid(True)
        _finalize_plot('single_sma_vs_time.png', fig)

    # Plot eccentricity vs. time
    if plot_ecc == 1:
        fig = plt.figure()
        plt.plot(time, orb[:, 1], linewidth=0.5)
        plt.xlabel('Time (' + unit + ')')
        plt.ylabel('Eccentricity')
        plt.title('Eccentricity vs. Time')
        plt.grid(True)
        _finalize_plot('single_eccentricity_vs_time.png', fig)

    # Plot perigee and apogee altitudes vs. time
    if plot_rp_ra == 1:
        fig = plt.figure()
        plt.plot(time, rp, label='Perigee', linewidth=0.5)
        plt.plot(time, ra, label='Apogee', linewidth=0.5)
        plt.xlabel('Time (' + unit + ')')
        plt.ylabel('Altitude (km)')
        plt.title('Perigee and Apogee Altitudes vs. Time')
        plt.legend()
        plt.grid(True)
        _finalize_plot('single_perigee_apogee_vs_time.png', fig)

    # Thrust diagnostic plots (single-run)
    if mass_series is not None and k_thrust == 1:
        if plot_mass == 1:
            fig = plt.figure()
            plt.plot(time, mass_series, linewidth=0.8)
            plt.xlabel('Time (' + unit + ')')
            plt.ylabel('Spacecraft mass (kg)')
            plt.title('Mass vs. Time')
            plt.grid(True)
            _finalize_plot('single_mass_vs_time.png', fig)

        if plot_thrust_mag == 1 and single_schedule is not None:
            from thrust_helpers import lookup_segment_for_time
            thrust_mag = np.zeros(n, dtype=np.float64)
            for j in range(n):
                seg_info = lookup_segment_for_time(single_schedule, float(times[j]))
                if mass_series[j] > seg_info.get('dry_mass_kg', 0.0):
                    thrust_mag[j] = seg_info['T_eff_N']
            fig = plt.figure()
            plt.plot(time, thrust_mag * 1e3, linewidth=0.8)  # mN
            plt.xlabel('Time (' + unit + ')')
            plt.ylabel('Thrust (mN)')
            plt.title('Effective Thrust vs. Time')
            plt.grid(True)
            _finalize_plot('single_thrust_magnitude_vs_time.png', fig)

        if plot_impulse == 1 and single_schedule is not None:
            from thrust_helpers import compute_case_thrust_summary
            summary = compute_case_thrust_summary(single_schedule,
                                                  np.asarray(times, dtype=np.float64),
                                                  mass_series)
            print(f"[Thrust summary] propellant = {summary['propellant_used_kg']:.3f} kg, "
                  f"impulse = {summary['cumulative_impulse_Ns']:.1f} Ns, "
                  f"energy = {summary['cumulative_energy_Wh']:.1f} Wh")

        # Save mass time-series
        mass_file = f'{fd}/mass_timeseries.dat'
        with open(mass_file, 'w') as f:
            for j in range(n):
                f.write(f"{times[j]:<15.8e} {mass_series[j]:<15.8e}\n")

    # Save the orbital elements of the satellite in a file
    if save_sat_oe == 1:
        filename = f'{fd}/orbital_elements_sat.dat'
        with open(filename, 'w') as f:
            for j in range(n):
                f.write(f"{times[j] / tu_conv:<15.8e} {orb[j][0]:<15.8e} "
                        f"{orb[j][1]:<15.8e} {np.rad2deg(orb[j][2]):<15.8e} "
                        f"{np.rad2deg(orb[j][3]):<15.8e} {np.rad2deg(orb[j][4]):<15.8e} "
                        f"{np.rad2deg(orb[j][5]):<15.8e}\n")

if __name__ == '__main__':
    main()