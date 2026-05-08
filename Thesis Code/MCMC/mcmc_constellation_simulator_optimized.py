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
from scipy.special import logsumexp 
import os
import sys
import multiprocessing as mp
from pathlib import Path
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit
from datetime import datetime, timedelta, date

_WORKER_QUIET_ENV = "MCMC_SIM_WORKER_QUIET"
if os.getenv(_WORKER_QUIET_ENV, "0") == "1":
    _WORKER_STDOUT_SINK = open(os.devnull, "w")
    sys.stdout = _WORKER_STDOUT_SINK

# Self-made libraries
from control_optimized import pause
from load_all_tle_data import load_all_tle_data

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
from perturbation_optimized import _EGM2008_n2_m0_reference_generic

# Optional: precomputed NRLMSIS grid atmosphere
from msis_precomputed import load_meta as msis_load_meta
from msis_precomputed import MsisGridIndex as MsisGridIndex
# Removed: memmap_grid, parse_date, date_add_days (replaced by standard lib and load_grid)
from msis_precomputed import load_grid as msis_load_grid
from atmosphere_msis_grid_optimized import atm_drag_msis_grid
from atmosphere_msis_grid_optimized import _atm_drag_msis_grid_daysec

# Import function to calculate the Earth's J2 (oblateness) perturbation on the Moon
from perturbation_optimized import J2acc
########################################################################
########################################################################
########################################################################
#----------------------### Initial conditions ###-----------------------

fd = '20240910A6' # folder where the results are stored

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
_default_tle_data_dir = Path(__file__).resolve().parent.parent / "starlink_decay"
tle_data_folders = [str(_default_tle_data_dir)]
tle_only_files = None          # Example: ['sat1010_decay.txt', 'sat1053_decay.txt']
tle_satellite_limit = 10       # 0: all satellites found, N>0: first N satellites (sorted by sat_id)

# Global calendar cutoffs for propagation window
tle_earliest_start_epoch = '2023-07-01'
simulation_date_cutoff = '2026-01-01'

tu = 3 # tu, time unit: 0: seconds, 1: minutes, 2: hours, 3: days, 4: years
# 2 and half years equals days of 
tf = 914.0  # tf, final time (tu)
t_step = 0.001  # t_step, time step (tu)

# Batch simulation controls (parallel Monte Carlo)
batch_mode = 1  # 0: single run (default), 1: run many simulations in parallel
num_simulations = 32  # used when batch_mode = 1
max_parallel_workers = 12  # tuned from batch benchmark
batch_random_seed = 42

# 1-sigma Gaussian dispersion around nominal ballistic coefficient for batch_mode
batch_sigma_ballistic_coef = 0.002  # m^2/kg

# Batch post-processing cutoff altitude (km)
batch_cutoff_alt_km = 115.0

# Save full trajectories for each batch case (expensive for 100s-1000s)
save_batch_trajectories = 0

# Satellite properties for SRP calculation
diameter = 6.1801  # m
radius = diameter / 2.0  # m
frontal_area = np.pi * (radius ** 2)  # circular cross-sectional area in m^2
mass = 260.0  # kg
AtoM = frontal_area / mass  # m^2/kg
Cr = 1.5 # reflectivity coefficient (1.0 <= ref_co <= 2.0)

# Satellite properties for atmospheric drag calculation
Cd = 2.2 # drag coefficient
ballistic_coefficient_nominal = Cd * AtoM  # (Cd * A / m), m^2/kg

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

# Atmosphere model selection for drag
# 0: USSA76
# 1: NRLMSIS
atm_model = 0

# Precomputed NRLMSIS grid configuration (used when atm_model=1)
if atm_model == 1:
    # Directory containing grid_meta.txt + grid_index.csv + rho_*.bin(.zst)
    # Override without editing this file by setting environment variable MSIS_GRID_DIR.
    _default_msis_grid_dir = Path(__file__).resolve().parent.parent / "out_grid"
    msis_grid_dir = os.getenv("MSIS_GRID_DIR", str(_default_msis_grid_dir))
    msis_grid_start_date = '2023-07-01'  # date for t=0 (yyyy-mm-dd)

    # Validate the grid directory only when MSIS drag is actually used.
    if k_atm_drag == 1:
        _candidates = [Path(msis_grid_dir), Path(__file__).resolve().parent / "out_grid",
                       Path(__file__).resolve().parent.parent / "out_grid", Path.cwd() / "out_grid"]
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
                                    "or ensure an out_grid folder exists at the workspace root (../out_grid relative to this script).\n"
                                    f"Tried:\n{tried}")

        msis_grid_dir = str(_picked)

########################################################################
################## Control output and data analysis ####################
# Flags to plot the results
plot_sv = 0 # plot the state vector: 0: no, 1: yes
plot_sv_3d_sat = 0 # plot the state vector for satellite in 3D: 0: no, 1: yes
plot_sv_3d_sun = 0 # plot the state vector for the Sun in 3D: 0: no, 1: yes 
plot_rp_ra = 0 # plot the perigee and apogee altitudes vs. time: 0: no, 1: yes
plot_sma = 1 # plot the semi-major axis vs. time: 0: no, 1: yes
plot_ecc = 1 # plot the eccentricity vs. time: 0: no, 1: yes
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
tle_df_all = None

if use_tle_initial_conditions == 1:
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
    tle_df_all = tle_df.copy()

    tle_latest = (tle_df.sort_values('timestamp')
                  .groupby('sat_id', as_index=False)
                  .head(1)
                  .sort_values('sat_id')
                  .reset_index(drop=True))

    if tle_satellite_limit > 0:
        tle_latest = tle_latest.iloc[:int(tle_satellite_limit), :].copy()

    if tle_latest.empty:
        raise RuntimeError("TLE selection produced zero satellites. Check tle_only_files/tle_satellite_limit.")

    n_tle = int(len(tle_latest))
    tle_oe_cases = np.zeros((n_tle, 6), dtype=np.float64)
    tle_oe_cases[:, 0] = tle_latest['sma'].to_numpy(dtype=np.float64)
    tle_oe_cases[:, 1] = np.clip(tle_latest['ecc'].to_numpy(dtype=np.float64), 0.0, 0.95)
    tle_oe_cases[:, 2] = np.deg2rad(tle_latest['inc'].to_numpy(dtype=np.float64))
    tle_oe_cases[:, 3] = np.deg2rad(tle_latest['aop'].to_numpy(dtype=np.float64))
    tle_oe_cases[:, 4] = np.deg2rad(tle_latest['raan'].to_numpy(dtype=np.float64))
    tle_oe_cases[:, 5] = np.deg2rad(tle_latest['mean_anomaly'].to_numpy(dtype=np.float64))
    tle_oe_cases[:, 0] = np.maximum(6378.1366 + 120.0, tle_oe_cases[:, 0])

    tle_sat_ids_selected = tle_latest['sat_id'].astype(str).tolist()
    tle_start_datetimes_selected = pd.to_datetime(tle_latest['timestamp']).tolist()

    # Use first selected satellite as nominal single-run initial condition
    oe_sat[:] = tle_oe_cases[0, :]
    first_ts = tle_latest['timestamp'].iloc[0]
    if pd.notna(first_ts):
        epoch = pd.Timestamp(first_ts).strftime('%Y-%m-%d')
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

print(f"\nSatellite's initial conditions with respect to the Earth:")
print(f"---------------------------------------------------------")
if use_tle_initial_conditions == 1:
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
print(f"Ballistic coefficient (Cd*A/m): {ballistic_coefficient_nominal:.5e} m^2/kg")
print(f"---------------------------------------------------------\n")

# Load the major bodies' initial conditions (mb) and the initial Julian date (jd) at the epoch
mb, jd0 = load_mb(frame, epoch)

# Load the initial GST at the epoch
GST0 = np.deg2rad(gst0(jd0))

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
neq = 6 * 3  # 6 elements for each of the 3 bodies (Sun, Moon, and satellite)
Xb_init = np.zeros(neq)

# Sun
Xb_init[0:6] = Xb_sun
# Moon
Xb_init[6:12] = Xb_moon
# Satellite
Xb_init[12:18] = init_sc

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

_BATCH_SHARED_SAT_IDS = None
_BATCH_SHARED_START_TIMESTAMPS = None
_BATCH_SHARED_X_SUN = None
_BATCH_SHARED_X_MOON = None
_BATCH_SHARED_GST0 = None

_MCMC_SHARED_SAT_CASES = None
_MCMC_SHARED_CFG = None
_MCMC_PROFILE_ENABLED = False

_MCMC_MSIS_CACHE_META = None
_MCMC_MSIS_CACHE_INDEX = None
_MCMC_MSIS_CACHE_DIR = None
_MCMC_MSIS_CACHE_START_DATE = None

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
    _ = Derivs(0.0, Xb_init, p_pre, pl_pre, sml_pre_local, cml_pre_local, tmp3_pre_local,
               ballistic_coefficient_nominal, GST0)

    _numba_warmed = True

def _batch_worker_init(sat_ids, start_timestamps, x_sun_cases, x_moon_cases, gst0_cases):
    global _BATCH_SHARED_SAT_IDS, _BATCH_SHARED_START_TIMESTAMPS
    global _BATCH_SHARED_X_SUN, _BATCH_SHARED_X_MOON, _BATCH_SHARED_GST0

    _BATCH_SHARED_SAT_IDS = tuple(sat_ids)
    _BATCH_SHARED_START_TIMESTAMPS = tuple(start_timestamps)
    _BATCH_SHARED_X_SUN = np.ascontiguousarray(x_sun_cases, dtype=np.float64)
    _BATCH_SHARED_X_MOON = np.ascontiguousarray(x_moon_cases, dtype=np.float64)
    _BATCH_SHARED_GST0 = np.ascontiguousarray(gst0_cases, dtype=np.float64)
    _warmup_numba_derivs()

def _mcmc_worker_init(sat_cases_shared, cfg_shared, profile_enabled):
    global _MCMC_SHARED_SAT_CASES, _MCMC_SHARED_CFG, _MCMC_PROFILE_ENABLED
    global model_b_cfg

    _MCMC_SHARED_SAT_CASES = tuple(sat_cases_shared)
    _MCMC_SHARED_CFG = dict(cfg_shared)
    if "model_b_cfg" in _MCMC_SHARED_CFG:
        # Safety: ensure spawned workers use exactly the caller's Model B prior/config.
        model_b_cfg = dict(_MCMC_SHARED_CFG["model_b_cfg"])
    _MCMC_PROFILE_ENABLED = bool(profile_enabled)
    _warmup_numba_derivs()

#-------------------------------------------------------------------------------
#-----------------#### Function with derivatives (model) ####-------------------
@njit(cache=True)
def Derivs(t, f, P, Pl, sml, cml, tmp3, ballistic_coefficient, gst0_case):
    Re = earth_Re

    x_sun = f[0:6]
    x_sun3 = f[0:3]
    sun_x = x_sun[0]
    sun_y = x_sun[1]
    sun_z = x_sun[2]
    r_sun2 = sun_x * sun_x + sun_y * sun_y + sun_z * sun_z
    r_sun = np.sqrt(r_sun2)
    inv_rsun3 = 1.0 / (r_sun2 * r_sun)

    x_moon = f[6:12]
    x_moon3 = f[6:9]
    moon_x = x_moon[0]
    moon_y = x_moon[1]
    moon_z = x_moon[2]
    r_moon2 = moon_x * moon_x + moon_y * moon_y + moon_z * moon_z
    r_moon = np.sqrt(r_moon2)
    inv_rmoon3 = 1.0 / (r_moon2 * r_moon)

    x_sat = f[12:18]
    x_sat3 = f[12:15]
    sat_x = x_sat[0]
    sat_y = x_sat[1]
    sat_z = x_sat[2]
    r2 = sat_x * sat_x + sat_y * sat_y + sat_z * sat_z
    r = np.sqrt(r2)
    inv_r3 = 1.0 / (r2 * r)

    # Sun EOM
    dxsundt = x_sun[3]
    dysundt = x_sun[4]
    dzsundt = x_sun[5]

    mu_sun_earth = sun_GM + earth_GM
    ddxsundt = -mu_sun_earth * sun_x * inv_rsun3
    ddysundt = -mu_sun_earth * sun_y * inv_rsun3
    ddzsundt = -mu_sun_earth * sun_z * inv_rsun3

    # Moon EOM
    dxmoondt = x_moon[3]
    dymoondt = x_moon[4]
    dzmoondt = x_moon[5]

    # Sun's perturbation on the Moon
    AC3b(x_moon3, x_sun3, sun_GM, tmp3)
    ac3b_sun0 = tmp3[0]
    ac3b_sun1 = tmp3[1]
    ac3b_sun2 = tmp3[2]

    # Earth's J2 perturbation on the Moon
    J2acc(earth_GM, earth_J2, earth_Re, x_moon3, tmp3)
    acj20 = tmp3[0]
    acj21 = tmp3[1]
    acj22 = tmp3[2]
    mu_moon_earth = moon_GM + earth_GM
    ddxmoondt = -mu_moon_earth * moon_x * inv_rmoon3 + acj20 + ac3b_sun0
    ddymoondt = -mu_moon_earth * moon_y * inv_rmoon3 + acj21 + ac3b_sun1
    ddzmoondt = -mu_moon_earth * moon_z * inv_rmoon3 + acj22 + ac3b_sun2

    # Accumulate spacecraft perturbations into tmp3 (used as accumulator)
    axp = 0.0
    ayp = 0.0
    azp = 0.0

    # Earth's spherical harmonics
    if k_EGM2008 == 1:
        EGM2008(nmax, mmax, x_sat3, C, S, t, earth_GM, earth_Re, earth_spin, gst0_case, P, Pl, sml, cml, tmp3)
        axp += tmp3[0]
        ayp += tmp3[1]
        azp += tmp3[2]

    # Third-body: Moon
    if k_moon == 1:
        AC3b(x_sat3, x_moon3, moon_GM, tmp3)
        axp += tmp3[0]
        ayp += tmp3[1]
        azp += tmp3[2]

    # Third-body: Sun
    if k_sun == 1:
        AC3b(x_sat3, x_sun3, sun_GM, tmp3)
        axp += tmp3[0]
        ayp += tmp3[1]
        azp += tmp3[2]

    # Solar radiation pressure
    if k_SRP == 1:
        SRPacc(x_sat3, x_sun3, AtoM, Cr, Re, tmp3)
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
    ddxdt = -mu * sat_x * inv_r3 + axp
    ddydt = -mu * sat_y * inv_r3 + ayp
    ddzdt = -mu * sat_z * inv_r3 + azp

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
# -----------------#### Batch processing functions ####-------------------
def _build_initial_state_from_oe(oe_case, x_sun_case, x_moon_case):
    init_sc_case = orb2xyz(earth_GM, oe_case)
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

    t_eval_case = tspan[tspan <= (t_final_case + 1e-12)]
    if t_eval_case.size == 0:
        return np.array([0.0], dtype=np.float64)
    if t_eval_case[0] != 0.0:
        t_eval_case = np.concatenate((np.array([0.0], dtype=np.float64), t_eval_case))
    return np.asarray(t_eval_case, dtype=np.float64)

def _integrate_single_case(oe_case, ballistic_coefficient_case,
                           x_sun_case, x_moon_case, gst0_case,
                           start_timestamp=None, event_mode='reentry'):
    xb_case = _build_initial_state_from_oe(oe_case, x_sun_case, x_moon_case)

    MM = nmax + 1
    p_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    pl_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    sml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    cml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    tmp3_pre_local = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

    if atm_model == 1 and k_atm_drag == 1:
        raise RuntimeError("Parallel batch mode currently supports atm_model=0 (USSA76) only.")

    if event_mode == 'reentry':
        events = [Reentry]
    elif event_mode == 'batch_cutoff':
        events = [BatchCutoff]
    else:
        events = None

    t_final_case = _seconds_until_date_cutoff(start_timestamp)
    t_eval_case = _build_t_eval_with_cutoff(t_final_case)

    if t_final_case <= 0.0:
        return np.array([0.0], dtype=np.float64), xb_case.reshape((18, 1)), -1.0

    sol = solve_ivp(Derivs, [0.0, t_final_case], xb_case, events=events, method='DOP853',
                    t_eval=t_eval_case, rtol=1.e-10, atol=1.e-12,
                  args=(p_pre, pl_pre, sml_pre_local, cml_pre_local, tmp3_pre_local,
                                            ballistic_coefficient_case, gst0_case))

    t_event = -1.0
    if events is not None and getattr(sol, 't_events', None) is not None:
        if len(sol.t_events) > 0 and np.asarray(sol.t_events[0]).size > 0:
            t_event = float(sol.t_events[0][0])

    return np.asarray(sol.t, dtype=np.float64), np.asarray(sol.y, dtype=np.float64), t_event

def _sample_batch_ballistic_coefficients(nsims, seed):
    rng = np.random.default_rng(seed)
    bc = rng.normal(ballistic_coefficient_nominal, batch_sigma_ballistic_coef, size=nsims)
    return np.clip(bc, 1e-12, None)

def _run_one_batch_case(case_id, oe_case, sat_id, start_timestamp,
                        x_sun_case, x_moon_case, gst0_case,
                        ballistic_coefficient_case):
    _warmup_numba_derivs()
    t_case, y_case, t_cut = _integrate_single_case(oe_case, ballistic_coefficient_case,
                                                   x_sun_case, x_moon_case, gst0_case,
                                                   start_timestamp=start_timestamp,
                                                   event_mode='batch_cutoff')

    x_sat_case = np.ascontiguousarray(y_case[12:18, :], dtype=np.float64)
    final_state = x_sat_case[:, -1]
    terminated = 1 if t_cut >= 0.0 else 0
    start_ts_case = pd.Timestamp(start_timestamp)
    if pd.isna(start_ts_case):
        start_ts_case = pd.Timestamp(epoch)
    start_day_offset = (start_ts_case - pd.Timestamp(tle_earliest_start_epoch)).total_seconds() / 86400.0

    return {'case_id': case_id, 'sat_id': str(sat_id),
            'a_km': float(oe_case[0]), 'e': float(oe_case[1]),
            'i_deg': float(np.rad2deg(oe_case[2])), 'w_deg': float(np.rad2deg(oe_case[3])),
            'OM_deg': float(np.rad2deg(oe_case[4])), 'Ma_deg': float(np.rad2deg(oe_case[5])),
            'start_timestamp': str(start_ts_case), 'start_day_offset': float(start_day_offset),
            'ballistic_coeff_m2_per_kg': float(ballistic_coefficient_case),
            'n_points': int(t_case.size), 'terminated_at_115km': terminated, 't_115_s': float(t_cut),
            'final_x_km': float(final_state[0]), 'final_y_km': float(final_state[1]), 'final_z_km': float(final_state[2]),
            'final_vx_kms': float(final_state[3]), 'final_vy_kms': float(final_state[4]), 'final_vz_kms': float(final_state[5]),
            'times': t_case, 'state_sat': x_sat_case}

def _run_one_batch_case_compact(case_id, oe_case, ballistic_coefficient_case):
    if (_BATCH_SHARED_SAT_IDS is None or _BATCH_SHARED_START_TIMESTAMPS is None or
            _BATCH_SHARED_X_SUN is None or _BATCH_SHARED_X_MOON is None or _BATCH_SHARED_GST0 is None):
        raise RuntimeError("Batch worker shared state not initialized")

    return _run_one_batch_case(case_id, oe_case, _BATCH_SHARED_SAT_IDS[case_id],
                               _BATCH_SHARED_START_TIMESTAMPS[case_id], _BATCH_SHARED_X_SUN[case_id],
                               _BATCH_SHARED_X_MOON[case_id], float(_BATCH_SHARED_GST0[case_id]),
                               ballistic_coefficient_case)

def _run_parallel_batch(oe_cases=None, sat_ids=None, start_timestamps=None, ballistic_coefficients=None,
                        workers_override=None, show_case_progress=True,
                        write_summary=True, write_trajectories=None,
                        compact_payload=True):
    if atm_model == 1 and k_atm_drag == 1:
        raise RuntimeError("batch_mode=1 currently supports atm_model=0 (USSA76) only.")

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

    if workers_override is None:
        workers = _resolve_batch_worker_count()
    else:
        workers = max(1, int(workers_override))

    if write_trajectories is None:
        write_trajectories = (save_batch_trajectories == 1)

    print(f"\nBatch mode enabled: {nsims} simulations")
    print(f"Max workers: {workers}")
    print(f"Ballistic coefficient (nominal): {ballistic_coefficient_nominal:.5e} m^2/kg")
    print(f"Ballistic coefficient sigma: {batch_sigma_ballistic_coef:.5e} m^2/kg")
    print(f"Altitude cutoff after solve: {batch_cutoff_alt_km:.1f} km")
    print(f"Date cutoff after solve: {simulation_date_cutoff}")
    print(f"Unique start dates loaded: {len(epoch_cache)}")

    batch_results = [None] * nsims

    _warmup_numba_derivs()

    _prev_worker_quiet = os.environ.get(_WORKER_QUIET_ENV)
    os.environ[_WORKER_QUIET_ENV] = "1"
    try:
        if compact_payload:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=_batch_worker_init,
                                     initargs=(sat_ids, start_timestamps, x_sun_cases, x_moon_cases, gst0_cases)) as executor:
                futures = {executor.submit(_run_one_batch_case_compact, idx, oe_cases[idx],
                               float(ballistic_coefficients[idx])): idx
                           for idx in range(nsims)}

                completed = 0
                for fut in as_completed(futures):
                    result = fut.result()
                    batch_results[result['case_id']] = result
                    completed += 1
                    if show_case_progress:
                        if result['terminated_at_115km'] == 1:
                            print(f"Batch progress: {completed}/{nsims} | "
                                  f"sat {result['sat_id']} done | cutoff at t={result['t_115_s']:.2f} s")
                        else:
                            print(f"Batch progress: {completed}/{nsims} | "
                                  f"sat {result['sat_id']} done | no cutoff")
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=_warmup_numba_derivs) as executor:
                futures = {executor.submit(_run_one_batch_case, idx, oe_cases[idx], sat_ids[idx],
                                           start_timestamps[idx], x_sun_cases[idx], x_moon_cases[idx], float(gst0_cases[idx]),
                               float(ballistic_coefficients[idx])): idx
                           for idx in range(nsims)}

                completed = 0
                for fut in as_completed(futures):
                    result = fut.result()
                    batch_results[result['case_id']] = result
                    completed += 1
                    if show_case_progress:
                        if result['terminated_at_115km'] == 1:
                            print(f"Batch progress: {completed}/{nsims} | "
                                  f"sat {result['sat_id']} done | cutoff at t={result['t_115_s']:.2f} s")
                        else:
                            print(f"Batch progress: {completed}/{nsims} | "
                                  f"sat {result['sat_id']} done | no cutoff")
    finally:
        if _prev_worker_quiet is None:
            os.environ.pop(_WORKER_QUIET_ENV, None)
        else:
            os.environ[_WORKER_QUIET_ENV] = _prev_worker_quiet

    if write_summary:
        os.makedirs(fd, exist_ok=True)
        summary_file = f'{fd}/batch_summary.csv'
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['case_id', 'sat_id', 'start_timestamp', 'start_day_offset',
                             'a_km', 'e', 'i_deg', 'w_deg', 'OM_deg', 'Ma_deg',
                             'ballistic_coeff_m2_per_kg',
                             'n_points', 'terminated_at_115km', 't_115_s',
                             'final_x_km', 'final_y_km', 'final_z_km',
                             'final_vx_kms', 'final_vy_kms', 'final_vz_kms'])
            for item in batch_results:
                writer.writerow([item['case_id'], item['sat_id'], item['start_timestamp'], item['start_day_offset'],
                                 item['a_km'], item['e'], item['i_deg'], item['w_deg'], item['OM_deg'], item['Ma_deg'],
                                 item['ballistic_coeff_m2_per_kg'],
                                 item['n_points'], item['terminated_at_115km'], item['t_115_s'],
                                 item['final_x_km'], item['final_y_km'], item['final_z_km'],
                                 item['final_vx_kms'], item['final_vy_kms'], item['final_vz_kms']])

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
    return batch_results

# Pre-allocate arrays for EGM2008 and Derivs
MM = nmax + 1
P_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
Pl_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
sml_pre = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
cml_pre = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
derivatives_pre = np.ascontiguousarray(np.zeros(18, dtype=np.float64))
tmp3_pre = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

state = None

@njit(cache=True)
def _state_to_orbital_series(x_state):
    n_samples = x_state.shape[1]
    orb_case = np.empty((n_samples, 6), dtype=np.float64)
    for j in range(n_samples):
        orb_case[j, :] = xyz2orb(earth_GM, x_state[0:3, j], x_state[3:6, j])
    return orb_case

def _run_with_msis_grids(t_eval_override=None):
    """Segmented integration so Numba RHS can use two memmapped daily grids."""
    meta = msis_load_meta(msis_grid_dir)
    index = MsisGridIndex(msis_grid_dir)
    
    day0_ord = datetime.strptime(msis_grid_start_date, '%Y-%m-%d').date().toordinal()

    # Pre-allocate arrays for EGM2008 and Derivs (same as the default path)
    MM = nmax + 1
    P_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    Pl_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    sml_pre = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    cml_pre = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    tmp3_pre = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

    @njit(cache=True)
    def Derivs_msis(t, f, P, Pl, sml, cml, tmp3,
                    t_abs0,
                    ballistic_coefficient,
                    grid_ut00, grid_ut06, grid_ut12, grid_ut18, grid_tomorrow_ut00,
                    lat0, dlat, lon0, dlon, alt_min_km, alt_step_km):
        Re = earth_Re

        x_sun = f[0:6]
        x_sun3 = f[0:3]
        sun_x = x_sun[0]
        sun_y = x_sun[1]
        sun_z = x_sun[2]
        r_sun2 = sun_x * sun_x + sun_y * sun_y + sun_z * sun_z
        r_sun = np.sqrt(r_sun2)
        inv_rsun3 = 1.0 / (r_sun2 * r_sun)

        x_moon = f[6:12]
        x_moon3 = f[6:9]
        moon_x = x_moon[0]
        moon_y = x_moon[1]
        moon_z = x_moon[2]
        r_moon2 = moon_x * moon_x + moon_y * moon_y + moon_z * moon_z
        r_moon = np.sqrt(r_moon2)
        inv_rmoon3 = 1.0 / (r_moon2 * r_moon)

        x_sat = f[12:18]
        x_sat3 = f[12:15]
        sat_x = x_sat[0]
        sat_y = x_sat[1]
        sat_z = x_sat[2]
        r2 = sat_x * sat_x + sat_y * sat_y + sat_z * sat_z
        r = np.sqrt(r2)
        inv_r3 = 1.0 / (r2 * r)

        # Sun EOM
        dxsundt = x_sun[3]
        dysundt = x_sun[4]
        dzsundt = x_sun[5]

        mu_sun_earth = sun_GM + earth_GM
        ddxsundt = -mu_sun_earth * sun_x * inv_rsun3
        ddysundt = -mu_sun_earth * sun_y * inv_rsun3
        ddzsundt = -mu_sun_earth * sun_z * inv_rsun3

        # Moon EOM
        dxmoondt = x_moon[3]
        dymoondt = x_moon[4]
        dzmoondt = x_moon[5]

        # Sun's perturbation on the Moon
        AC3b(x_moon3, x_sun3, sun_GM, tmp3)
        ac3b_sun0 = tmp3[0]
        ac3b_sun1 = tmp3[1]
        ac3b_sun2 = tmp3[2]

        # Earth's J2 perturbation on the Moon
        J2acc(earth_GM, earth_J2, earth_Re, x_moon3, tmp3)
        acj20 = tmp3[0]
        acj21 = tmp3[1]
        acj22 = tmp3[2]

        mu_moon_earth = moon_GM + earth_GM
        ddxmoondt = -mu_moon_earth * moon_x * inv_rmoon3 + acj20 + ac3b_sun0
        ddymoondt = -mu_moon_earth * moon_y * inv_rmoon3 + acj21 + ac3b_sun1
        ddzmoondt = -mu_moon_earth * moon_z * inv_rmoon3 + acj22 + ac3b_sun2

        # Accumulate spacecraft perturbations
        axp = 0.0
        ayp = 0.0
        azp = 0.0

        if k_EGM2008 == 1:
            EGM2008(nmax, mmax, x_sat3, C, S, t_abs0 + t, earth_GM, earth_Re, earth_spin, GST0,
                    P, Pl, sml, cml, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if k_moon == 1:
            AC3b(x_sat3, x_moon3, moon_GM, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if k_sun == 1:
            AC3b(x_sat3, x_sun3, sun_GM, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if k_SRP == 1:
            SRPacc(x_sat3, x_sun3, AtoM, Cr, Re, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if r <= (900.0 + Re) and k_atm_drag == 1:
            sec_in_day = t
            if sec_in_day < 0.0 or sec_in_day >= 86400.0:
                sec_in_day = sec_in_day % 86400.0
            _atm_drag_msis_grid_daysec(x_sat, 1.0, ballistic_coefficient, earth_spin, GST0, t_abs0 + t, sec_in_day,
                                       grid_ut00, grid_ut06, grid_ut12, grid_ut18, grid_tomorrow_ut00,
                                       earth_Re, lat0, dlat, lon0, dlon,
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
        ddxdt = -mu * sat_x * inv_r3 + axp
        ddydt = -mu * sat_y * inv_r3 + ayp
        ddzdt = -mu * sat_z * inv_r3 + azp

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

    # Build a segmented solve_ivp over day boundaries
    if t_eval_override is None:
        t_eval = tspan
    else:
        t_eval = np.asarray(t_eval_override, dtype=np.float64)
    y0 = Xb_init

    y_all = []
    t_all = []
    reentry_t_abs = -1.0
    grid_cache = {}

    # Group contiguous times by integer day index
    day_idx = (t_eval * (1.0 / 86400.0)).astype(np.int64)
    last_di = -9223372036854775807
    last_paths = None
    last_grids = None
    start = 0
    while start < len(t_eval):
        di = int(day_idx[start])
        end = start
        while end < len(t_eval) and int(day_idx[end]) == di:
            end += 1

        t0_abs = float(di) * 86400.0
        t_seg_abs = t_eval[start:end]
        t_seg_local = t_seg_abs - t0_abs

        if di == last_di and last_paths is not None and last_grids is not None:
            p00, p06, p12, p18, p00n = last_paths
            g00, g06, g12, g18, g00n = last_grids
        else:
            date_today = date.fromordinal(day0_ord + di)
            date_tomorrow = date.fromordinal(day0_ord + di + 1)

            def _grid_path(d, ut):
                try:
                    return index.path_for_date_ut(d, ut)
                except Exception:
                    return index.path_for_date(d)

            p00 = _grid_path(date_today, 0)
            p06 = _grid_path(date_today, 21600)
            p12 = _grid_path(date_today, 43200)
            p18 = _grid_path(date_today, 64800)
            try:
                p00n = _grid_path(date_tomorrow, 0)
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

            last_di = di
            last_paths = (p00, p06, p12, p18, p00n)
            last_grids = (g00, g06, g12, g18, g00n)

        sol = solve_ivp(Derivs_msis, [0.0, float(t_seg_local[-1])], y0, events=[Reentry], method='DOP853',
                t_eval=t_seg_local, rtol=1.e-10, atol=1.e-12,
                args=(P_pre, Pl_pre, sml_pre, cml_pre, tmp3_pre,
                                        t0_abs, ballistic_coefficient_nominal, g00, g06, g12, g18, g00n,
                      meta.lat0, meta.dlat, meta.lon0, meta.dlon,
                      meta.alt_min_km, meta.alt_step_km))

        t_seg_out = np.atleast_1d(np.asarray(sol.t, dtype=np.float64))
        y_seg_out = np.asarray(sol.y, dtype=np.float64)

        # Be robust to odd shapes (some environments may yield list-like outputs).
        if y_seg_out.ndim == 1:
            y_seg_out = y_seg_out.reshape((y_seg_out.size, 1))
        elif y_seg_out.ndim == 2 and y_seg_out.shape[0] != y0.size and y_seg_out.shape[1] == y0.size:
            y_seg_out = y_seg_out.T

        t_all.append(t0_abs + t_seg_out)
        if y_seg_out.size == 0 or y_seg_out.shape[0] == 0 or y_seg_out.shape[1] == 0:
            # Nothing returned (can happen if a terminal event triggers immediately).
            break

        y_all.append(y_seg_out)
        y0 = y_seg_out[:, -1]

        # If a terminal event triggered (e.g., reentry), stop segmenting.
        if getattr(sol, "status", 0) == 1:
            if getattr(sol, 't_events', None) is not None and len(sol.t_events) > 0 and np.asarray(sol.t_events[0]).size > 0:
                reentry_t_abs = t0_abs + float(sol.t_events[0][0])
            break

        start = end

    if len(t_all) == 0 or len(y_all) == 0:
        raise RuntimeError("MSIS segmented integration produced no output")

    t_cat = np.concatenate(t_all)
    y_cat = np.concatenate(y_all, axis=1)
    return t_cat, y_cat, reentry_t_abs

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
        plt.figure()
        for idx in case_indices:
            item = batch_results[int(idx)]
            xx = item['state_sat']
            plt.plot(xx[0, :], xx[1, :], linewidth=0.5, alpha=0.45)
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.title('Batch trajectories (x-y)')
        plt.grid(True)
        plt.show()

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
        plt.show()

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
            plt.figure()
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
            plt.show()

        if plot_ecc == 1:
            plt.figure()
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
            plt.show()

        if plot_rp_ra == 1:
            plt.figure()
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
            plt.show()

########################################################################
######################## Model A MCMC (USSA76) #########################
########################################################################

# Mode switch for Bayesian BC inference (Model A)
run_model_a_mcmc = 1  # 1: run Model A MCMC and return, 0: use existing modes below

# MCMC model variant selector
# "A": existing USSA76 constant-beta inference (default behavior)
# "B": MSIS-grid drag with inferred flat-plate one-sided face area and random-tumbling mean projection
mcmc_model_variant = "B"

# Model A assumptions (kept lean)
# - USSA76 atmosphere (atm_model must be 0)
# - Constant ballistic coefficient
# - Cannonball drag representation already implied by Cd*A/m in atm_drag

# MCMC controls
mcmc_satellite_limit = tle_satellite_limit  # max satellites to run MCMC on (None for no limit)
mcmc_satellite_ids = None                   # optional explicit list of sat_id strings
mcmc_chains_per_sat = 4                     # independent MCMC chains per satellite (for convergence checks)
mcmc_steps = 1000                           # total iterations per chain
mcmc_adapt_steps = 500                      # warmup adaptation length
mcmc_burn = 500                             # burn-in/warmup iterations per chain
mcmc_thin = 1                               # no thinning (thin=1 means keep every sample)    
mcmc_init_log_step = 0.20                   # proposal std in log(beta)
mcmc_target_accept = 0.40                   # practical target for 1D RWMH
mcmc_seed = 42                              # random seed for reproducibility

# Parallelization controls
mcmc_max_workers = 12                       # total workers across all satellites/chains

# Observation preprocessing controls
mcmc_min_obs_spacing_days = 2.0       # downsample to reduce repeated expensive solves
mcmc_max_obs_per_sat = 60             # cap observation count for speed
mcmc_use_tail_days = None             # e.g., 120.0 to use only last 120 days before last TLE
mcmc_min_points = 8                   # minimum points required to run MCMC for a satellite
mcmc_sigma_floor_km = 0.25            # likelihood noise floor in km (semi-major axis)
mcmc_outlier_sigma_clip = 5.0         # robust clipping on detrended residuals

# Prior on log(beta)
mcmc_prior_log_sigma = 1.25           # broad log-space prior width
mcmc_beta_min = 1e-5                  # m^2/kg
mcmc_beta_max = 5e-1                  # m^2/kg

# Output
mcmc_output_dirname = "modelA_mcmc"
mcmc_save_chain_traces = 0                 # 1: save full per-step chain traces (large/slow), 0: skip
mcmc_progress_print_every = 8              # print progress every N finished chains
mcmc_enable_phase_timing = 1               # print phase timings for profiling/optimization
mcmc_make_plots = 1                        # 1: save MCMC/statistics plots, 0: skip plotting
mcmc_show_plots = 0                        # 1: display plots interactively, 0: save and close
mcmc_profile_timing = 1                    # 1: print worker timing breakdowns, 0: skip
mcmc_use_compact_worker_context = 1        # 1: shared per-worker sat-case context, 0: legacy job payloads

# Model B (MSIS + random tumbling flat-plate) configuration
model_b_cfg = {"solar_array_planform_m2": 30.0, # Flat-plate nominal geometry (Starlink v1 approximation)
               "chassis_dims_m": (3.2, 1.6, 0.2),
               "include_chassis_largest_face": True,
               "Cd": Cd,                        # Drag conversion parameters
               "mass_kg": mass,
               "attitude_scale": 1.0,           # Deterministic scaling on theorem mean projected area
               "area_face_min_m2": 0.5,
               "area_face_max_m2": 200.0,
               "logA_sigma": 0.75}              # Prior support and width for inferred one-sided face area A_face
    
@njit(cache=True)
def _sma_from_state_series(x_state):
    """Compute osculating semi-major axis directly from Cartesian state history.
    x_state shape: (6, N), units km and km/s, returns a in km.
    """
    n_samples = x_state.shape[1]
    a_out = np.empty(n_samples, dtype=np.float64)
    for j in range(n_samples):
        rx = x_state[0, j]
        ry = x_state[1, j]
        rz = x_state[2, j]
        vx = x_state[3, j]
        vy = x_state[4, j]
        vz = x_state[5, j]

        r = np.sqrt(rx * rx + ry * ry + rz * rz)
        v2 = vx * vx + vy * vy + vz * vz
        denom = (2.0 / r) - (v2 / earth_GM)

        if denom <= 0.0 or not np.isfinite(denom):
            a_out[j] = np.nan
        else:
            a_out[j] = 1.0 / denom
    return a_out

def _robust_mad_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def _downsample_by_min_spacing(df_in, min_spacing_days):
    if df_in is None or df_in.empty:
        return df_in
    if min_spacing_days is None or min_spacing_days <= 0.0:
        return df_in

    dt_min_sec = float(min_spacing_days) * 86400.0
    ts = pd.to_datetime(df_in["timestamp"]).to_numpy()
    keep = [0]
    t_last = pd.Timestamp(ts[0])

    for idx in range(1, len(df_in) - 1):
        t_now = pd.Timestamp(ts[idx])
        if (t_now - t_last).total_seconds() >= dt_min_sec:
            keep.append(idx)
            t_last = t_now

    if (len(df_in) - 1) not in keep:
        keep.append(len(df_in) - 1)

    keep = np.unique(np.asarray(keep, dtype=np.int64))
    return df_in.iloc[keep].copy().reset_index(drop=True)

def _cap_obs_count(df_in, max_obs):
    if df_in is None or df_in.empty:
        return df_in
    if max_obs is None or max_obs <= 0 or len(df_in) <= max_obs:
        return df_in

    idx = np.linspace(0, len(df_in) - 1, int(max_obs), dtype=np.int64)
    idx = np.unique(idx)
    if idx[0] != 0:
        idx = np.concatenate(([0], idx))
    if idx[-1] != len(df_in) - 1:
        idx = np.concatenate((idx, [len(df_in) - 1]))
    idx = np.unique(idx)
    return df_in.iloc[idx].copy().reset_index(drop=True)

def _build_oe_from_tle_row(row):
    oe = np.zeros(6, dtype=np.float64)
    oe[0] = max(float(row["sma"]), earth_Re + 120.0)
    oe[1] = float(np.clip(row["ecc"], 0.0, 0.95))
    oe[2] = np.deg2rad(float(row["inc"]))
    oe[3] = np.deg2rad(float(row["aop"]))
    oe[4] = np.deg2rad(float(row["raan"]))
    oe[5] = np.deg2rad(float(row["mean_anomaly"]))
    return oe

def _prepare_sat_epoch_context(start_timestamp):
    ts = pd.Timestamp(start_timestamp)
    if pd.isna(ts):
        ts = pd.Timestamp(epoch)
    date_key = ts.strftime("%Y-%m-%d")
    mb_case, jd_case = load_mb(frame, date_key)
    x_sun_case = np.ascontiguousarray(mb_case[0, :], dtype=np.float64)
    x_moon_case = np.ascontiguousarray(mb_case[2, :], dtype=np.float64)
    gst0_case = float(np.deg2rad(gst0(jd_case)))
    return x_sun_case, x_moon_case, gst0_case, date_key

def _prepare_t_eval_seconds(t_eval_seconds):
    t_eval_seconds = np.asarray(t_eval_seconds, dtype=np.float64)
    t_eval_seconds = t_eval_seconds[np.isfinite(t_eval_seconds)]
    if t_eval_seconds.size == 0:
        return np.array([0.0], dtype=np.float64)

    t_eval_seconds = np.unique(t_eval_seconds)
    t_eval_seconds.sort()

    if t_eval_seconds[0] > 0.0:
        t_eval_seconds = np.concatenate((np.array([0.0], dtype=np.float64), t_eval_seconds))
    elif t_eval_seconds[0] < 0.0:
        t_eval_seconds = t_eval_seconds[t_eval_seconds >= 0.0]
        if t_eval_seconds.size == 0 or t_eval_seconds[0] != 0.0:
            t_eval_seconds = np.concatenate((np.array([0.0], dtype=np.float64), t_eval_seconds))

    return np.ascontiguousarray(t_eval_seconds, dtype=np.float64)

def _model_b_nominal_face_area_m2():
    cfg = model_b_cfg
    a, b, c = cfg["chassis_dims_m"]
    chassis_largest_face = max(a * b, a * c, b * c)
    area_face = float(cfg["solar_array_planform_m2"])
    if bool(cfg.get("include_chassis_largest_face", True)):
        area_face += float(chassis_largest_face)
    return float(area_face)

def _model_b_effective_area_m2(area_face_m2):
    # Safety: random tumbling mean projected area theorem for flat plate with two sides:
    # A_eff = (2*A_face)/4 = A_face/2.
    return float(0.5 * float(area_face_m2) * float(model_b_cfg.get("attitude_scale", 1.0)))

def _model_b_beta_from_log_area(log_area_face):
    if not np.isfinite(log_area_face):
        return np.nan
    area_face = float(np.exp(log_area_face))
    a_min = float(model_b_cfg["area_face_min_m2"])
    a_max = float(model_b_cfg["area_face_max_m2"])
    if area_face < a_min or area_face > a_max:
        return np.nan

    area_eff = _model_b_effective_area_m2(area_face)
    if area_eff <= 0.0 or not np.isfinite(area_eff):
        return np.nan

    cd_local = float(model_b_cfg.get("Cd", Cd))
    mass_local = float(model_b_cfg.get("mass_kg", mass))
    if mass_local <= 0.0 or not np.isfinite(mass_local):
        return np.nan
    return float((cd_local * area_eff) / mass_local)

def _log_prior_log_area_face(log_area_face):
    if not np.isfinite(log_area_face):
        return -np.inf

    area_face = float(np.exp(log_area_face))
    a_min = float(model_b_cfg["area_face_min_m2"])
    a_max = float(model_b_cfg["area_face_max_m2"])
    if area_face < a_min or area_face > a_max or (not np.isfinite(area_face)):
        return -np.inf

    mu = float(np.log(_model_b_nominal_face_area_m2()))
    sig = float(model_b_cfg.get("logA_sigma", 0.75))
    if sig <= 0.0 or not np.isfinite(sig):
        return -np.inf
    z = (float(log_area_face) - mu) / sig
    return float(-0.5 * z * z - np.log(sig) - 0.5 * np.log(2.0 * np.pi))

def _ensure_msis_mcmc_cache():
    global _MCMC_MSIS_CACHE_META, _MCMC_MSIS_CACHE_INDEX, _MCMC_MSIS_CACHE_DIR, _MCMC_MSIS_CACHE_START_DATE

    if _MCMC_MSIS_CACHE_META is not None and _MCMC_MSIS_CACHE_INDEX is not None:
        return _MCMC_MSIS_CACHE_META, _MCMC_MSIS_CACHE_INDEX, _MCMC_MSIS_CACHE_DIR, _MCMC_MSIS_CACHE_START_DATE

    env_dir = os.getenv("MSIS_GRID_DIR", "").strip()
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))

    script_dir = Path(__file__).resolve().parent
    candidates.extend([script_dir / "out_grid", script_dir.parent / "out_grid", Path.cwd() / "out_grid"])

    picked = None
    for cand in candidates:
        try:
            if (cand / "grid_meta.txt").is_file():
                picked = cand
                break
        except OSError:
            pass

    if picked is None:
        tried = "\n".join(f"- {p}" for p in candidates)
        raise FileNotFoundError("Model B MSIS grid metadata not found (grid_meta.txt).\n"
                                "Set MSIS_GRID_DIR to the folder containing grid_meta.txt and grid_index.csv,\n"
                                "or place an out_grid folder at script/workspace root.\n"
                                f"Tried:\n{tried}")

    meta = msis_load_meta(picked)
    index = MsisGridIndex(picked)

    _MCMC_MSIS_CACHE_META = meta
    _MCMC_MSIS_CACHE_INDEX = index
    _MCMC_MSIS_CACHE_DIR = str(picked)
    _MCMC_MSIS_CACHE_START_DATE = os.getenv("MCMC_MSIS_GRID_START_DATE", tle_earliest_start_epoch)
    return _MCMC_MSIS_CACHE_META, _MCMC_MSIS_CACHE_INDEX, _MCMC_MSIS_CACHE_DIR, _MCMC_MSIS_CACHE_START_DATE


def _integrate_single_case_custom_t_eval_msis(oe_case, ballistic_coefficient_case,
                                              x_sun_case, x_moon_case, gst0_case,
                                              t_eval_seconds, start_timestamp=None, event_mode='reentry',
                                              t_eval_prepared=False, profile_stats=None):
    meta, index, _, _ = _ensure_msis_mcmc_cache()

    xb_case = _build_initial_state_from_oe(oe_case, x_sun_case, x_moon_case)
    MM = nmax + 1
    p_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    pl_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    sml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    cml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    tmp3_pre_local = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

    if t_eval_prepared:
        t_eval_seconds = np.ascontiguousarray(t_eval_seconds, dtype=np.float64)
    else:
        t_eval_seconds = _prepare_t_eval_seconds(t_eval_seconds)

    if t_eval_seconds.size == 0:
        return np.array([0.0], dtype=np.float64), xb_case.reshape((18, 1)), -1.0

    t_final_req = float(t_eval_seconds[-1])
    t_final_date = _seconds_until_date_cutoff(start_timestamp)
    t_final_case = float(min(t_final_req, t_final_date))
    if t_final_case <= 0.0:
        return np.array([0.0], dtype=np.float64), xb_case.reshape((18, 1)), -1.0

    t_eval_case = t_eval_seconds[t_eval_seconds <= (t_final_case + 1e-9)]
    if t_eval_case.size == 0:
        t_eval_case = np.array([0.0], dtype=np.float64)

    if event_mode == 'reentry':
        events = [Reentry]
    elif event_mode == 'batch_cutoff':
        events = [BatchCutoff]
    else:
        events = None

    start_ts = pd.Timestamp(start_timestamp)
    if pd.isna(start_ts):
        start_ts = pd.Timestamp(epoch)
    day0 = start_ts.date()
    # Use the true epoch time-of-day so MSIS UT interpolation is aligned to the case epoch.
    start_sec_of_day = (float(start_ts.hour) * 3600.0 + float(start_ts.minute) * 60.0 +
                        float(start_ts.second) + float(start_ts.microsecond) * 1e-6)

    @njit(cache=True)
    def Derivs_msis_mcmc(t, f, P, Pl, sml, cml, tmp3,
                         t_abs0,
                         sec_in_day0,
                         ballistic_coefficient,
                         grid_ut00, grid_ut06, grid_ut12, grid_ut18, grid_tomorrow_ut00,
                         lat0, dlat, lon0, dlon, alt_min_km, alt_step_km,
                         gst0_local):
        Re = earth_Re

        x_sun = f[0:6]
        x_sun3 = f[0:3]
        sun_x = x_sun[0]
        sun_y = x_sun[1]
        sun_z = x_sun[2]
        r_sun2 = sun_x * sun_x + sun_y * sun_y + sun_z * sun_z
        r_sun = np.sqrt(r_sun2)
        inv_rsun3 = 1.0 / (r_sun2 * r_sun)

        x_moon = f[6:12]
        x_moon3 = f[6:9]
        moon_x = x_moon[0]
        moon_y = x_moon[1]
        moon_z = x_moon[2]
        r_moon2 = moon_x * moon_x + moon_y * moon_y + moon_z * moon_z
        r_moon = np.sqrt(r_moon2)
        inv_rmoon3 = 1.0 / (r_moon2 * r_moon)

        x_sat = f[12:18]
        x_sat3 = f[12:15]
        sat_x = x_sat[0]
        sat_y = x_sat[1]
        sat_z = x_sat[2]
        r2 = sat_x * sat_x + sat_y * sat_y + sat_z * sat_z
        r = np.sqrt(r2)
        inv_r3 = 1.0 / (r2 * r)

        dxsundt = x_sun[3]
        dysundt = x_sun[4]
        dzsundt = x_sun[5]

        mu_sun_earth = sun_GM + earth_GM
        ddxsundt = -mu_sun_earth * sun_x * inv_rsun3
        ddysundt = -mu_sun_earth * sun_y * inv_rsun3
        ddzsundt = -mu_sun_earth * sun_z * inv_rsun3

        dxmoondt = x_moon[3]
        dymoondt = x_moon[4]
        dzmoondt = x_moon[5]

        AC3b(x_moon3, x_sun3, sun_GM, tmp3)
        ac3b_sun0 = tmp3[0]
        ac3b_sun1 = tmp3[1]
        ac3b_sun2 = tmp3[2]

        J2acc(earth_GM, earth_J2, earth_Re, x_moon3, tmp3)
        acj20 = tmp3[0]
        acj21 = tmp3[1]
        acj22 = tmp3[2]
        mu_moon_earth = moon_GM + earth_GM
        ddxmoondt = -mu_moon_earth * moon_x * inv_rmoon3 + acj20 + ac3b_sun0
        ddymoondt = -mu_moon_earth * moon_y * inv_rmoon3 + acj21 + ac3b_sun1
        ddzmoondt = -mu_moon_earth * moon_z * inv_rmoon3 + acj22 + ac3b_sun2

        axp = 0.0
        ayp = 0.0
        azp = 0.0

        if k_EGM2008 == 1:
            EGM2008(nmax, mmax, x_sat3, C, S, t_abs0 + t, earth_GM, earth_Re, earth_spin, gst0_local,
                    P, Pl, sml, cml, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if k_moon == 1:
            AC3b(x_sat3, x_moon3, moon_GM, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if k_sun == 1:
            AC3b(x_sat3, x_sun3, sun_GM, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if k_SRP == 1:
            SRPacc(x_sat3, x_sun3, AtoM, Cr, Re, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        if r <= (900.0 + Re) and k_atm_drag == 1:
            sec_in_day = sec_in_day0 + t
            if sec_in_day < 0.0 or sec_in_day >= 86400.0:
                sec_in_day = sec_in_day % 86400.0
            _atm_drag_msis_grid_daysec(x_sat, 1.0, ballistic_coefficient, earth_spin, gst0_local, t_abs0 + t, sec_in_day,
                                       grid_ut00, grid_ut06, grid_ut12, grid_ut18, grid_tomorrow_ut00,
                                       earth_Re, lat0, dlat, lon0, dlon,
                                       alt_min_km, alt_step_km,
                                       tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

        dxdt = x_sat[3]
        dydt = x_sat[4]
        dzdt = x_sat[5]

        mu = earth_GM
        ddxdt = -mu * sat_x * inv_r3 + axp
        ddydt = -mu * sat_y * inv_r3 + ayp
        ddzdt = -mu * sat_z * inv_r3 + azp

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

    def _path_for_day_ut(d, ut):
        try:
            return index.path_for_date_ut(d, int(ut))
        except KeyError:
            if int(ut) != 0:
                return index.path_for_date_ut(d, 0)
            raise

    y0 = xb_case
    t_all = []
    y_all = []
    t_event_abs = -1.0
    grid_cache = {}

    # Continuous propagation across entire elapsed time; only sampling output at requested t_eval.
    # This avoids skipping dynamics between sparse observation days.
    t_cursor = 0.0
    while t_cursor < t_final_case - 1e-12:
        sec_abs = start_sec_of_day + t_cursor
        day_index = int(np.floor(sec_abs / 86400.0))
        sec_in_day0 = sec_abs - 86400.0 * day_index
        if sec_in_day0 < 0.0:
            sec_in_day0 += 86400.0

        date_today = day0 + timedelta(days=day_index)
        date_tomorrow = day0 + timedelta(days=day_index + 1)
        day_remaining = 86400.0 - sec_in_day0
        seg_end_abs = min(t_final_case, t_cursor + day_remaining)
        seg_dur = float(seg_end_abs - t_cursor)
        if seg_dur <= 0.0:
            break

        # Observation times that fall in this day-segment.
        mask = (t_eval_case >= (t_cursor - 1e-12)) & (t_eval_case <= (seg_end_abs + 1e-12))
        t_obs_seg_abs = t_eval_case[mask]
        obs_local = np.ascontiguousarray(t_obs_seg_abs - t_cursor, dtype=np.float64) if t_obs_seg_abs.size > 0 else np.array([], dtype=np.float64)
        if t_obs_seg_abs.size > 0:
            t_eval_local = np.ascontiguousarray(obs_local, dtype=np.float64)
            t_eval_local = t_eval_local[t_eval_local >= -1e-12]
            if t_eval_local.size > 0 and t_eval_local[0] < 0.0:
                t_eval_local[0] = 0.0
        else:
            t_eval_local = np.array([], dtype=np.float64)

        # Always request the segment endpoint so propagation across observation gaps remains continuous.
        if t_eval_local.size == 0:
            t_eval_local = np.array([seg_dur], dtype=np.float64)
        elif abs(float(t_eval_local[-1]) - seg_dur) > 1e-10:
            t_eval_local = np.concatenate((t_eval_local, np.array([seg_dur], dtype=np.float64)))

        n_obs_local = int(obs_local.size)

        p00 = _path_for_day_ut(date_today, 0)
        p06 = _path_for_day_ut(date_today, 21600)
        p12 = _path_for_day_ut(date_today, 43200)
        p18 = _path_for_day_ut(date_today, 64800)
        try:
            p00n = _path_for_day_ut(date_tomorrow, 0)
        except KeyError:
            p00n = p18

        key00 = str(p00)
        g00 = grid_cache.get(key00)
        if g00 is None:
            g00 = msis_load_grid(p00, meta)
            grid_cache[key00] = g00

        key06 = str(p06)
        g06 = grid_cache.get(key06)
        if g06 is None:
            g06 = msis_load_grid(p06, meta)
            grid_cache[key06] = g06

        key12 = str(p12)
        g12 = grid_cache.get(key12)
        if g12 is None:
            g12 = msis_load_grid(p12, meta)
            grid_cache[key12] = g12

        key18 = str(p18)
        g18 = grid_cache.get(key18)
        if g18 is None:
            g18 = msis_load_grid(p18, meta)
            grid_cache[key18] = g18

        key00n = str(p00n)
        g00n = grid_cache.get(key00n)
        if g00n is None:
            g00n = msis_load_grid(p00n, meta)
            grid_cache[key00n] = g00n

        t_solve0 = timeit.default_timer() if profile_stats is not None else 0.0
        sol = solve_ivp(Derivs_msis_mcmc, [0.0, seg_dur], y0, events=events, method='DOP853',
                        t_eval=t_eval_local, rtol=1.e-10, atol=1.e-12,
                        args=(p_pre, pl_pre, sml_pre_local, cml_pre_local, tmp3_pre_local,
                              float(t_cursor), float(sec_in_day0), float(ballistic_coefficient_case),
                              g00, g06, g12, g18, g00n,
                              meta.lat0, meta.dlat, meta.lon0, meta.dlon,
                              meta.alt_min_km, meta.alt_step_km,
                              float(gst0_case)))
        if profile_stats is not None:
            profile_stats["solve_ivp_s"] = profile_stats.get("solve_ivp_s", 0.0) + (timeit.default_timer() - t_solve0)

        t_seg_out = np.atleast_1d(np.asarray(sol.t, dtype=np.float64))
        y_seg_out = np.asarray(sol.y, dtype=np.float64)
        if y_seg_out.ndim == 1:
            y_seg_out = y_seg_out.reshape((y_seg_out.size, 1))
        elif y_seg_out.ndim == 2 and y_seg_out.shape[0] != y0.size and y_seg_out.shape[1] == y0.size:
            y_seg_out = y_seg_out.T

        if (n_obs_local > 0 and t_seg_out.size >= n_obs_local and
                y_seg_out.size > 0 and y_seg_out.shape[1] >= n_obs_local):
            t_all.append(t_cursor + np.asarray(t_seg_out[:n_obs_local], dtype=np.float64))
            y_all.append(np.asarray(y_seg_out[:, :n_obs_local], dtype=np.float64))

        # Always advance state to segment end so intermediate days without observations are propagated.
        if y_seg_out.size == 0 or y_seg_out.shape[1] == 0:
            return np.array([0.0], dtype=np.float64), xb_case.reshape((18, 1)), -1.0
        y0 = np.asarray(y_seg_out[:, -1], dtype=np.float64)

        if getattr(sol, "status", 0) == 1:
            if getattr(sol, 't_events', None) is not None and len(sol.t_events) > 0 and np.asarray(sol.t_events[0]).size > 0:
                t_event_abs = t_cursor + float(sol.t_events[0][0])
            break

        t_cursor = seg_end_abs

    if len(t_all) == 0 or len(y_all) == 0:
        return np.array([0.0], dtype=np.float64), xb_case.reshape((18, 1)), -1.0

    t_cat = np.concatenate(t_all)
    y_cat = np.concatenate(y_all, axis=1)
    return np.asarray(t_cat, dtype=np.float64), np.asarray(y_cat, dtype=np.float64), float(t_event_abs)

def _integrate_single_case_custom_t_eval(oe_case, ballistic_coefficient_case, x_sun_case, x_moon_case, gst0_case,
                                         t_eval_seconds, start_timestamp=None, event_mode='reentry',
                                         t_eval_prepared=False, profile_stats=None):
    xb_case = _build_initial_state_from_oe(oe_case, x_sun_case, x_moon_case)

    MM = nmax + 1
    p_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    pl_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    sml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    cml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    tmp3_pre_local = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

    if t_eval_prepared:
        t_eval_seconds = np.ascontiguousarray(t_eval_seconds, dtype=np.float64)
    else:
        t_eval_seconds = _prepare_t_eval_seconds(t_eval_seconds)

    if t_eval_seconds.size == 0:
        return np.array([0.0], dtype=np.float64), xb_case.reshape((18, 1)), -1.0

    t_final_req = float(t_eval_seconds[-1])
    t_final_date = _seconds_until_date_cutoff(start_timestamp)
    t_final_case = float(min(t_final_req, t_final_date))

    if t_final_case <= 0.0:
        return np.array([0.0], dtype=np.float64), xb_case.reshape((18, 1)), -1.0

    t_eval_case = t_eval_seconds[t_eval_seconds <= (t_final_case + 1e-9)]
    if t_eval_case.size == 0:
        t_eval_case = np.array([0.0], dtype=np.float64)

    if event_mode == 'reentry':
        events = [Reentry]
    elif event_mode == 'batch_cutoff':
        events = [BatchCutoff]
    else:
        events = None

    t_solve0 = timeit.default_timer() if profile_stats is not None else 0.0
    sol = solve_ivp(Derivs, [0.0, t_final_case], xb_case, events=events, method='DOP853',
                    t_eval=t_eval_case, rtol=1.e-10, atol=1.e-12,
                    args=(p_pre, pl_pre, sml_pre_local, cml_pre_local, tmp3_pre_local,
                          float(ballistic_coefficient_case), float(gst0_case)))
    if profile_stats is not None:
        profile_stats["solve_ivp_s"] = profile_stats.get("solve_ivp_s", 0.0) + (timeit.default_timer() - t_solve0)

    t_event = -1.0
    if events is not None and getattr(sol, 't_events', None) is not None:
        if len(sol.t_events) > 0 and np.asarray(sol.t_events[0]).size > 0:
            t_event = float(sol.t_events[0][0])

    return np.asarray(sol.t, dtype=np.float64), np.asarray(sol.y, dtype=np.float64), t_event

def _prepare_mcmc_cases(model_variant="A"):
    model_variant = str(model_variant).upper().strip()
    if model_variant not in ("A", "B"):
        raise RuntimeError(f"Unsupported mcmc_model_variant: {model_variant}")

    if use_tle_initial_conditions != 1:
        raise RuntimeError("Model A MCMC requires use_tle_initial_conditions = 1.")
    if tle_df_all is None or len(tle_df_all) == 0:
        raise RuntimeError("Full TLE dataframe not available. Ensure tle_df_all is saved after load_all_tle_data.")
    if model_variant == "A" and atm_model != 0:
        raise RuntimeError("Model A MCMC is defined here for atm_model = 0 (USSA76).")
    if k_atm_drag != 1:
        raise RuntimeError("Model A MCMC requires atmospheric drag enabled (k_atm_drag = 1).")

    required_cols = ["sat_id", "timestamp", "sma", "ecc", "inc", "aop", "raan", "mean_anomaly"]
    for c in required_cols:
        if c not in tle_df_all.columns:
            raise RuntimeError(f"Missing required TLE-derived column '{c}' in tle_df_all.")

    df = tle_df_all.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "sma", "ecc", "inc", "aop", "raan", "mean_anomaly"]).copy()

    # Apply global study window
    t0_global = pd.Timestamp(tle_earliest_start_epoch)
    tf_global = pd.Timestamp(simulation_date_cutoff)
    df = df[(df["timestamp"] >= t0_global) & (df["timestamp"] < tf_global)].copy()

    if df.empty:
        raise RuntimeError("No TLE samples remain after applying the study time window.")

    # Normalize sat_id once to avoid repeated astype/filter churn
    df["sat_id_str"] = df["sat_id"].astype(str)

    # Satellite selection
    if mcmc_satellite_ids is not None and len(mcmc_satellite_ids) > 0:
        sat_ids = [str(s) for s in mcmc_satellite_ids]
    else:
        sat_ids_all = sorted(df["sat_id_str"].unique().tolist())
        if mcmc_satellite_limit in (None, 0):
            sat_ids = sat_ids_all
        else:
            sat_ids = sat_ids_all[:int(mcmc_satellite_limit)]

    if len(sat_ids) == 0:
        raise RuntimeError("No satellites selected for Model A MCMC.")

    sat_cases = []
    sat_groups = {sid: grp.copy() for sid, grp in df.groupby("sat_id_str", sort=False)}
    epoch_context_cache = {}
    for sat_idx, sat_id_local in enumerate(sat_ids):
        dfi = sat_groups.get(str(sat_id_local))
        if dfi is None:
            continue
        dfi = dfi.copy()
        if dfi.empty:
            continue

        dfi = dfi.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

        # Optional tail window, useful for faster test runs or late-decay inference
        if mcmc_use_tail_days is not None and float(mcmc_use_tail_days) > 0.0 and len(dfi) > 1:
            t_last = pd.Timestamp(dfi["timestamp"].iloc[-1])
            t_min = t_last - pd.Timedelta(days=float(mcmc_use_tail_days))
            dfi = dfi[dfi["timestamp"] >= t_min].copy().reset_index(drop=True)

        # Downsample by minimum time spacing then cap count
        dfi = _downsample_by_min_spacing(dfi, mcmc_min_obs_spacing_days)
        dfi = _cap_obs_count(dfi, mcmc_max_obs_per_sat)

        if len(dfi) < int(mcmc_min_points):
            print(f"Skipping {sat_id_local}: not enough observations after downsampling ({len(dfi)}).")
            continue

        # Build times relative to first TLE for this satellite
        t_start = pd.Timestamp(dfi["timestamp"].iloc[0])
        t_obs_sec = (pd.to_datetime(dfi["timestamp"]) - t_start).dt.total_seconds().to_numpy(dtype=np.float64)
        sma_obs_km = dfi["sma"].to_numpy(dtype=np.float64)

        # Detrend for robust sigma estimate and outlier clipping
        t_days = t_obs_sec / 86400.0
        if len(t_days) >= 2:
            p = np.polyfit(t_days, sma_obs_km, 1)
            trend = p[0] * t_days + p[1]
            resid = sma_obs_km - trend
        else:
            resid = sma_obs_km - np.mean(sma_obs_km)

        sigma_est = max(_robust_mad_sigma(resid), float(mcmc_sigma_floor_km))
        med_resid = np.median(resid)
        clip_mask = np.abs(resid - med_resid) <= (float(mcmc_outlier_sigma_clip) * sigma_est)

        # Always keep endpoints
        clip_mask[0] = True
        clip_mask[-1] = True

        dfi = dfi.iloc[np.where(clip_mask)[0]].copy().reset_index(drop=True)

        if len(dfi) < int(mcmc_min_points):
            print(f"Skipping {sat_id_local}: too many points clipped, remaining={len(dfi)}.")
            continue

        # Recompute arrays after clipping
        t_start = pd.Timestamp(dfi["timestamp"].iloc[0])
        t_obs_sec = (pd.to_datetime(dfi["timestamp"]) - t_start).dt.total_seconds().to_numpy(dtype=np.float64)
        sma_obs_km = dfi["sma"].to_numpy(dtype=np.float64)

        # Build inference on delta-a to remove initial bias between TLE mean elements and osculating model state
        delta_a_obs_km = sma_obs_km - sma_obs_km[0]
        t_eval_sec_prepared = _prepare_t_eval_seconds(t_obs_sec)

        # Conservative observation noise, estimated from detrended residuals and floored
        t_days = t_obs_sec / 86400.0
        if len(t_days) >= 2:
            p = np.polyfit(t_days, sma_obs_km, 1)
            resid = sma_obs_km - (p[0] * t_days + p[1])
        else:
            resid = sma_obs_km - np.mean(sma_obs_km)
        sigma_a_km = max(_robust_mad_sigma(resid), float(mcmc_sigma_floor_km))

        # Initial condition from first TLE in the retained series
        oe0 = _build_oe_from_tle_row(dfi.iloc[0])

        # Major body and GST context at satellite-specific start date
        date_key = pd.Timestamp(t_start).strftime("%Y-%m-%d")
        epoch_ctx = epoch_context_cache.get(date_key)
        if epoch_ctx is None:
            x_sun_case, x_moon_case, gst0_case, date_key = _prepare_sat_epoch_context(t_start)
            epoch_ctx = (x_sun_case, x_moon_case, gst0_case, date_key)
            epoch_context_cache[date_key] = epoch_ctx
        else:
            x_sun_case, x_moon_case, gst0_case, date_key = epoch_ctx

        sat_cases.append({"sat_index": int(sat_idx), "sat_id": str(sat_id_local), "start_timestamp": str(t_start),
                  "mcmc_model_variant": str(model_variant),
                          "epoch_date_key": str(date_key), "oe0": np.ascontiguousarray(oe0, dtype=np.float64),
                          "t_obs_sec": np.ascontiguousarray(t_obs_sec, dtype=np.float64),
                          "t_eval_sec": np.ascontiguousarray(t_eval_sec_prepared, dtype=np.float64),
                          "sma_obs_km": np.ascontiguousarray(sma_obs_km, dtype=np.float64),
                          "delta_a_obs_km": np.ascontiguousarray(delta_a_obs_km, dtype=np.float64),
                          "sigma_a_km": float(sigma_a_km), "x_sun": np.ascontiguousarray(x_sun_case, dtype=np.float64),
                          "x_moon": np.ascontiguousarray(x_moon_case, dtype=np.float64), "gst0": float(gst0_case),
                          "n_obs": int(len(dfi)), "obs_first": str(pd.Timestamp(dfi["timestamp"].iloc[0])),
                          "obs_last": str(pd.Timestamp(dfi["timestamp"].iloc[-1]))})

    if len(sat_cases) == 0:
        raise RuntimeError("No satellite datasets prepared for Model A MCMC.")
    return sat_cases

def _prepare_model_a_mcmc_cases():
    return _prepare_mcmc_cases(model_variant="A")

def _log_prior_log_beta(log_beta):
    if not np.isfinite(log_beta):
        return -np.inf

    beta = np.exp(log_beta)
    if (beta < float(mcmc_beta_min)) or (beta > float(mcmc_beta_max)) or (not np.isfinite(beta)):
        return -np.inf

    mu = np.log(float(ballistic_coefficient_nominal))
    sig = float(mcmc_prior_log_sigma)
    z = (log_beta - mu) / sig
    return -0.5 * z * z - np.log(sig) - 0.5 * np.log(2.0 * np.pi)

def _loglike_delta_a_gaussian(delta_a_obs_km, delta_a_mod_km, sigma_a_km):
    resid = np.asarray(delta_a_obs_km, dtype=np.float64) - np.asarray(delta_a_mod_km, dtype=np.float64)
    s = float(sigma_a_km)

    if s <= 0.0 or not np.isfinite(s):
        return -np.inf
    if np.any(~np.isfinite(resid)):
        return -np.inf

    n = resid.size
    return -0.5 * np.sum((resid / s) ** 2) - n * np.log(s) - 0.5 * n * np.log(2.0 * np.pi)

def _evaluate_mcmc_logposterior(theta, sat_case, profile_stats=None):
    t_lp0 = timeit.default_timer() if profile_stats is not None else 0.0
    model_variant = str(sat_case.get("mcmc_model_variant", mcmc_model_variant)).upper().strip()
    theta = float(theta)

    if model_variant == "A":
        lp = _log_prior_log_beta(theta)
        if not np.isfinite(lp):
            return -np.inf
        beta = float(np.exp(theta))
    elif model_variant == "B":
        lp = _log_prior_log_area_face(theta)
        if not np.isfinite(lp):
            return -np.inf
        beta = _model_b_beta_from_log_area(theta)
        if not np.isfinite(beta) or beta <= 0.0:
            return -np.inf
    else:
        raise RuntimeError(f"Unsupported sat_case model variant: {model_variant}")

    t_req = sat_case.get("t_eval_sec", sat_case["t_obs_sec"])
    oe0 = sat_case["oe0"]
    x_sun_case = sat_case["x_sun"]
    x_moon_case = sat_case["x_moon"]
    gst0_case = sat_case["gst0"]
    start_ts = sat_case["start_timestamp"]
    if model_variant == "B":
        expected_date_key = str(sat_case.get("epoch_date_key", ""))
        actual_date_key = pd.Timestamp(start_ts).strftime("%Y-%m-%d")
        if expected_date_key and expected_date_key != actual_date_key:
            raise RuntimeError(f"Model B date alignment mismatch for sat {sat_case.get('sat_id', 'unknown')}: "
                               f"epoch_date_key={expected_date_key} start_timestamp_date={actual_date_key}")

    # Integrate only at the TLE observation epochs
    if model_variant == "A":
        t_sol, y_sol, t_event = _integrate_single_case_custom_t_eval(oe0, beta, x_sun_case, x_moon_case, gst0_case,
                                                                     t_req, start_timestamp=start_ts, event_mode='reentry',
                                                                     t_eval_prepared=True, profile_stats=profile_stats)
    else:
        t_sol, y_sol, t_event = _integrate_single_case_custom_t_eval_msis(oe0, beta, x_sun_case, x_moon_case, gst0_case,
                                                                          t_req, start_timestamp=start_ts, event_mode='reentry',
                                                                          t_eval_prepared=True, profile_stats=profile_stats)

    # If reentry occurs before the last observation, the model cannot explain the data under this beta
    if t_sol.size != t_req.size:
        return -np.inf

    x_sat_sol = np.ascontiguousarray(y_sol[12:18, :], dtype=np.float64)
    a_mod_km = _sma_from_state_series(x_sat_sol)
    if np.any(~np.isfinite(a_mod_km)):
        return -np.inf

    delta_a_mod_km = a_mod_km - a_mod_km[0]
    ll = _loglike_delta_a_gaussian(sat_case["delta_a_obs_km"], delta_a_mod_km, sat_case["sigma_a_km"])
    if not np.isfinite(ll):
        return -np.inf

    if profile_stats is not None:
        profile_stats["likelihood_eval_s"] = profile_stats.get("likelihood_eval_s", 0.0) + (timeit.default_timer() - t_lp0)

    return float(lp + ll)

def _evaluate_model_a_logposterior(log_beta, sat_case, profile_stats=None):
    # Backward-compatible wrapper for existing benchmarks/callers.
    sat_case_local = dict(sat_case)
    sat_case_local["mcmc_model_variant"] = "A"
    return _evaluate_mcmc_logposterior(log_beta, sat_case_local, profile_stats=profile_stats)

def _mcmc_chain_worker(job):
    global model_b_cfg
    _warmup_numba_derivs()

    compact_mode = bool("sat_case_idx" in job)
    if compact_mode:
        if _MCMC_SHARED_SAT_CASES is None:
            raise RuntimeError("MCMC shared worker context not initialized")
        sat_case = _MCMC_SHARED_SAT_CASES[int(job["sat_case_idx"])]
        cfg = _MCMC_SHARED_CFG if _MCMC_SHARED_CFG is not None else {}
        chain_id = int(job["chain_id"])
        n_steps_local = int(cfg.get("n_steps", mcmc_steps))
        adapt_steps_local = int(cfg.get("adapt_steps", mcmc_adapt_steps))
        seed_local = int(job["seed"])
        init_log_step_local = float(cfg.get("init_log_step", mcmc_init_log_step))
        target_accept_local = float(cfg.get("target_accept", mcmc_target_accept))
        theta_center = float(cfg.get("theta_center", np.log(float(ballistic_coefficient_nominal))))
        model_variant_local = str(cfg.get("model_variant", sat_case.get("mcmc_model_variant", mcmc_model_variant))).upper().strip()
    else:
        sat_case = job["sat_case"]
        chain_id = int(job["chain_id"])
        n_steps_local = int(job["n_steps"])
        adapt_steps_local = int(job["adapt_steps"])
        seed_local = int(job["seed"])
        init_log_step_local = float(job["init_log_step"])
        target_accept_local = float(job["target_accept"])
        theta_center = float(job.get("theta_center", np.log(float(ballistic_coefficient_nominal))))
        model_variant_local = str(job.get("model_variant", sat_case.get("mcmc_model_variant", mcmc_model_variant))).upper().strip()
        if "model_b_cfg" in job:
            model_b_cfg = dict(job["model_b_cfg"])

    profile_enabled = bool(_MCMC_PROFILE_ENABLED)
    timing_stats = {"likelihood_eval_s": 0.0, "solve_ivp_s": 0.0, "proposal_s": 0.0}
    t_worker0 = timeit.default_timer() if profile_enabled else 0.0

    rng = np.random.default_rng(seed_local)

    # Initialize near nominal, with small jitter
    cur = float(theta_center + 0.15 * rng.normal())

    sat_case_local = dict(sat_case)
    sat_case_local["mcmc_model_variant"] = model_variant_local

    lp_cur = _evaluate_mcmc_logposterior(cur, sat_case_local, profile_stats=timing_stats if profile_enabled else None)

    # Robustify initialization if needed
    tries = 0
    while (not np.isfinite(lp_cur)) and (tries < 30):
        cur = float(theta_center + 0.75 * rng.normal())
        lp_cur = _evaluate_mcmc_logposterior(cur, sat_case_local, profile_stats=timing_stats if profile_enabled else None)
        tries += 1

    if not np.isfinite(lp_cur):
        # Last fallback, search entire prior-support interval.
        lo = float(np.log(max(mcmc_beta_min, 1e-16)))
        hi = float(np.log(mcmc_beta_max))
        if model_variant_local == "B":
            lo = float(np.log(max(float(model_b_cfg["area_face_min_m2"]), 1e-16)))
            hi = float(np.log(float(model_b_cfg["area_face_max_m2"])))
        for cur_try in np.linspace(lo, hi, 121):
            lp_try = _evaluate_mcmc_logposterior(float(cur_try), sat_case_local,
                                                 profile_stats=timing_stats if profile_enabled else None)
            if np.isfinite(lp_try):
                cur = float(cur_try)
                lp_cur = float(lp_try)
                break

    if not np.isfinite(lp_cur):
        return {"sat_id": sat_case["sat_id"], "sat_index": int(sat_case["sat_index"]),
                "chain_id": int(chain_id), "failed": 1, "fail_reason": "no_finite_initial_posterior",
            "mcmc_model_variant": model_variant_local,
                "n_obs": int(sat_case["n_obs"]), "sigma_a_km": float(sat_case["sigma_a_km"]),
                "start_timestamp": sat_case["start_timestamp"], "obs_first": sat_case["obs_first"],
            "obs_last": sat_case["obs_last"],
            "timing_likelihood_eval_s": float(timing_stats["likelihood_eval_s"]),
            "timing_solve_ivp_s": float(timing_stats["solve_ivp_s"]),
            "timing_proposal_s": float(timing_stats["proposal_s"]),
            "timing_worker_total_s": float((timeit.default_timer() - t_worker0) if profile_enabled else 0.0)}

    samples_theta = np.empty(n_steps_local, dtype=np.float64)
    samples_lp = np.empty(n_steps_local, dtype=np.float64)
    accepted = np.zeros(n_steps_local, dtype=np.int8)
    prop_sd_hist = np.empty(n_steps_local, dtype=np.float64)

    log_step = float(np.log(max(init_log_step_local, 1e-6)))

    for k in range(n_steps_local):
        t_prop0 = timeit.default_timer() if profile_enabled else 0.0
        prop = float(cur + np.exp(log_step) * rng.normal())
        lp_prop = _evaluate_mcmc_logposterior(prop, sat_case_local, profile_stats=timing_stats if profile_enabled else None)

        acc = 0
        if np.isfinite(lp_prop):
            log_alpha = lp_prop - lp_cur
            if np.log(rng.random()) < log_alpha:
                cur = prop
                lp_cur = lp_prop
                acc = 1
        if profile_enabled:
            timing_stats["proposal_s"] = timing_stats["proposal_s"] + (timeit.default_timer() - t_prop0)

        # Robbins-Monro style adaptation during warmup only
        if k < adapt_steps_local:
            gamma = min(0.05, 1.0 / np.sqrt(k + 1.0))
            log_step = float(log_step + gamma * (float(acc) - target_accept_local))
            # keep proposal scale bounded
            if log_step < np.log(1e-4):
                log_step = float(np.log(1e-4))
            elif log_step > np.log(2.0):
                log_step = float(np.log(2.0))

        samples_theta[k] = cur
        samples_lp[k] = lp_cur
        accepted[k] = acc
        prop_sd_hist[k] = np.exp(log_step)

    if model_variant_local == "B":
        samples_log_area_face = np.asarray(samples_theta, dtype=np.float64)
        samples_area_face_m2 = np.exp(samples_log_area_face)
        samples_area_eff_m2 = 0.5 * samples_area_face_m2 * float(model_b_cfg.get("attitude_scale", 1.0))
        cd_local = float(model_b_cfg.get("Cd", Cd))
        mass_local = float(model_b_cfg.get("mass_kg", mass))
        samples_beta = (cd_local * samples_area_eff_m2) / mass_local
        samples_log_beta = np.log(np.clip(samples_beta, 1e-300, None))
    else:
        samples_log_beta = np.asarray(samples_theta, dtype=np.float64)
        samples_log_area_face = np.full(n_steps_local, np.nan, dtype=np.float64)
        samples_area_face_m2 = np.full(n_steps_local, np.nan, dtype=np.float64)
        samples_area_eff_m2 = np.full(n_steps_local, np.nan, dtype=np.float64)

    return {"sat_id": sat_case["sat_id"], "sat_index": int(sat_case["sat_index"]), "chain_id": int(chain_id),
            "failed": 0, "fail_reason": "", "samples_log_beta": samples_log_beta, "samples_lp": samples_lp,
            "samples_theta": samples_theta,
            "samples_log_area_face": samples_log_area_face,
            "samples_area_face_m2": samples_area_face_m2,
            "samples_area_eff_m2": samples_area_eff_m2,
            "mcmc_model_variant": model_variant_local,
            "accepted": accepted, "prop_sd_hist": prop_sd_hist, "accept_rate_total": float(np.mean(accepted)),
            "accept_rate_warmup": float(np.mean(accepted[:max(1, adapt_steps_local)])), 
            "final_prop_sd": float(prop_sd_hist[-1]), "n_obs": int(sat_case["n_obs"]),
            "sigma_a_km": float(sat_case["sigma_a_km"]), "start_timestamp": sat_case["start_timestamp"],
            "obs_first": sat_case["obs_first"], "obs_last": sat_case["obs_last"],
            "timing_likelihood_eval_s": float(timing_stats["likelihood_eval_s"]),
            "timing_solve_ivp_s": float(timing_stats["solve_ivp_s"]),
            "timing_proposal_s": float(timing_stats["proposal_s"]),
            "timing_worker_total_s": float((timeit.default_timer() - t_worker0) if profile_enabled else 0.0)}

def _compute_rhat(chains_2d):
    x = np.asarray(chains_2d, dtype=np.float64)
    if x.ndim != 2:
        return np.nan
    m, n = x.shape
    if m < 2 or n < 2:
        return np.nan

    chain_means = np.mean(x, axis=1)
    chain_vars = np.var(x, axis=1, ddof=1)

    W = np.mean(chain_vars)
    B = n * np.var(chain_means, ddof=1)

    if not np.isfinite(W) or W <= 0.0:
        return np.nan

    var_hat = ((n - 1.0) / n) * W + (B / n)
    if var_hat <= 0.0:
        return np.nan

    return float(np.sqrt(var_hat / W))

def _save_model_a_mcmc_plots(outdir, by_sat, summary_df):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping MCMC plots: matplotlib unavailable ({exc})")
        return

    plots_dir = Path(outdir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    eps = 1e-16

    for sat_id_local, res_list in by_sat.items():
        if len(res_list) == 0:
            continue

        res_list = sorted(res_list, key=lambda z: z["chain_id"])

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

        for r in res_list:
            it = np.arange(len(r["samples_log_beta"]))
            beta_series = np.exp(r["samples_log_beta"])
            bc_series = 1.0 / np.clip(beta_series, eps, None)
            ax[0].plot(it, bc_series, linewidth=0.7, alpha=0.8, label=f"chain {r['chain_id']}")
            ax[1].plot(it, r["samples_lp"], linewidth=0.7, alpha=0.8)
            acc_cum = np.cumsum(r["accepted"]) / np.maximum(1, (it + 1))
            ax[2].plot(it, acc_cum, linewidth=0.7, alpha=0.8)

        burn_x = max(0, int(mcmc_burn) - 1)
        for jj in range(3):
            ax[jj].axvline(burn_x, color="k", linestyle="--", linewidth=0.9, alpha=0.6)

        ax[0].set_title(f"BC Trace (sat {sat_id_local})")
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Ballistic Coefficient BC [kg/m^2]")

        ax[1].set_title("Log Posterior Trace")
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("log p")

        ax[2].set_title("Cumulative Acceptance")
        ax[2].set_xlabel("Iteration")
        ax[2].set_ylabel("Acceptance Rate")
        ax[2].set_ylim(0.0, 1.0)

        handles, labels = ax[0].get_legend_handles_labels()
        if len(handles) > 0:
            ax[0].legend(loc="best", fontsize=8)

        fig.tight_layout()
        diag_fn = plots_dir / f"{sat_id_local}_mcmc_diagnostics.png"
        fig.savefig(diag_fn, dpi=180, bbox_inches="tight")
        if int(mcmc_show_plots) == 1:
            plt.show()
        else:
            plt.close(fig)

        post_chains = []
        for r in res_list:
            s = r["samples_log_beta"]
            s_post = s[int(mcmc_burn)::int(max(1, mcmc_thin))]
            if len(s_post) > 0:
                post_chains.append(np.asarray(s_post, dtype=np.float64))

        if len(post_chains) == 0:
            continue

        all_post = np.concatenate(post_chains)
        beta_post = np.exp(all_post)
        bc_post = 1.0 / np.clip(beta_post, eps, None)

        q_beta = np.quantile(beta_post, [0.05, 0.50, 0.95])
        q_bc = np.quantile(bc_post, [0.05, 0.50, 0.95])

        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.5))

        ax2[0].hist(beta_post, bins=50, density=True, alpha=0.75)
        ax2[0].axvline(q_beta[0], linestyle="--", linewidth=1.0)
        ax2[0].axvline(q_beta[1], linestyle="-", linewidth=1.2)
        ax2[0].axvline(q_beta[2], linestyle="--", linewidth=1.0)
        ax2[0].set_title(f"Posterior β (sat {sat_id_local})")
        ax2[0].set_xlabel("β = Cd*A/m [m^2/kg]")
        ax2[0].set_ylabel("Density")

        ax2[1].hist(bc_post, bins=50, density=True, alpha=0.75)
        ax2[1].axvline(q_bc[0], linestyle="--", linewidth=1.0)
        ax2[1].axvline(q_bc[1], linestyle="-", linewidth=1.2)
        ax2[1].axvline(q_bc[2], linestyle="--", linewidth=1.0)
        ax2[1].set_title("Posterior BC (actual ballistic coefficient)")
        ax2[1].set_xlabel("BC = m/(Cd*A) [kg/m^2]")
        ax2[1].set_ylabel("Density")

        fig2.tight_layout()
        post_fn = plots_dir / f"{sat_id_local}_posterior_beta_bc.png"
        fig2.savefig(post_fn, dpi=180, bbox_inches="tight")
        if int(mcmc_show_plots) == 1:
            plt.show()
        else:
            plt.close(fig2)

    if summary_df is not None and len(summary_df) > 0:
        sdf = summary_df.sort_values("sat_id").reset_index(drop=True)
        y = sdf["bc_median_kg_per_m2"].to_numpy(dtype=np.float64)
        y_lo = sdf["bc_q05_kg_per_m2"].to_numpy(dtype=np.float64)
        y_hi = sdf["bc_q95_kg_per_m2"].to_numpy(dtype=np.float64)
        x = np.arange(len(sdf), dtype=np.int64)

        fig3, ax3 = plt.subplots(2, 1, figsize=(max(9, 0.8 * len(sdf) + 4), 8), sharex=True)

        yerr = np.vstack((np.maximum(0.0, y - y_lo), np.maximum(0.0, y_hi - y)))
        ax3[0].errorbar(x, y, yerr=yerr, fmt='o', capsize=3)
        ax3[0].set_ylabel("BC [kg/m^2]")
        ax3[0].set_title("Posterior BC median with 90% interval by satellite")
        ax3[0].grid(True, alpha=0.3)

        ax3[1].bar(x, sdf["mean_accept_rate"].to_numpy(dtype=np.float64), alpha=0.8)
        ax3[1].set_ylim(0.0, 1.0)
        ax3[1].set_ylabel("Mean acceptance")
        ax3[1].set_title("MCMC acceptance rate by satellite")
        ax3[1].grid(True, axis='y', alpha=0.3)

        ax3[1].set_xticks(x)
        ax3[1].set_xticklabels(sdf["sat_id"].astype(str).tolist(), rotation=45, ha='right')
        ax3[1].set_xlabel("Satellite ID")

        fig3.tight_layout()
        summary_plot_fn = plots_dir / "modelA_mcmc_bc_summary.png"
        fig3.savefig(summary_plot_fn, dpi=180, bbox_inches="tight")
        if int(mcmc_show_plots) == 1:
            plt.show()
        else:
            plt.close(fig3)

def _run_model_a_mcmc(compact_worker_context=None):
    t_total_start = timeit.default_timer()
    t_phase = timeit.default_timer()
    model_variant_local = str(mcmc_model_variant).upper().strip()
    sat_cases = _prepare_mcmc_cases(model_variant=model_variant_local)
    prep_sec = timeit.default_timer() - t_phase

    if compact_worker_context is None:
        compact_worker_context = (int(mcmc_use_compact_worker_context) == 1)

    outdir_name = mcmc_output_dirname if model_variant_local == "A" else "mcmc_model_b"
    outdir = Path(fd) / outdir_name
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel {model_variant_local} MCMC mode enabled")
    if model_variant_local == "A":
        print(f"Atmosphere: USSA76 (atm_model={atm_model})")
    else:
        print("Atmosphere: NRLMSIS precomputed grid (local Model B path)")
    print(f"Satellites to infer: {len(sat_cases)}")
    print(f"Chains per satellite: {mcmc_chains_per_sat}")
    print(f"Steps per chain: {mcmc_steps}, burn={mcmc_burn}, thin={mcmc_thin}")
    if model_variant_local == "A":
        print(f"Prior center (nominal beta): {ballistic_coefficient_nominal:.6e} m^2/kg")
        print(f"Prior log-sigma: {mcmc_prior_log_sigma:.3f}")
    else:
        print(f"Prior center (nominal A_face): {_model_b_nominal_face_area_m2():.6e} m^2")
        print(f"Prior log-sigma (area): {float(model_b_cfg.get('logA_sigma', 0.75)):.3f}")
    print(f"Output dir: {outdir}")
    if model_variant_local == "B":
        a_face_nom = _model_b_nominal_face_area_m2()
        a_eff_nom = _model_b_effective_area_m2(a_face_nom)
        beta_nom = _model_b_beta_from_log_area(np.log(a_face_nom))
        print(f"Model B nominal A_face: {a_face_nom:.6f} m^2")
        print(f"Model B nominal A_eff: {a_eff_nom:.6f} m^2")
        print(f"Model B implied nominal beta: {beta_nom:.6e} m^2/kg")
        _ensure_msis_mcmc_cache()
    if int(mcmc_enable_phase_timing) == 1:
        print(f"Timing | case preparation: {prep_sec:.2f} s")

    for sc in sat_cases:
        span_days = (np.max(sc["t_obs_sec"]) - np.min(sc["t_obs_sec"])) / 86400.0
        print(f"  sat={sc['sat_id']} | n_obs={sc['n_obs']} | sigma_a={sc['sigma_a_km']:.3f} km "
              f"| span={span_days:.1f} days | start={sc['start_timestamp']}")

    # Flatten jobs across satellites and chains for full parallel utilization
    # Compact mode keeps payloads minimal and resolves sat_case via per-worker shared context.
    jobs = []
    seed_counter = int(mcmc_seed)
    for sat_case_idx, sc in enumerate(sat_cases):
        for chain_id in range(int(mcmc_chains_per_sat)):
            if compact_worker_context:
                jobs.append({"sat_case_idx": int(sat_case_idx), "chain_id": int(chain_id),
                             "seed": int(seed_counter)})
            else:
                theta_center_job = float(np.log(float(ballistic_coefficient_nominal)))
                if model_variant_local == "B":
                    theta_center_job = float(np.log(_model_b_nominal_face_area_m2()))
                jobs.append({"sat_case": sc, "sat_case_idx": int(sat_case_idx), "chain_id": int(chain_id),
                             "n_steps": int(mcmc_steps), "adapt_steps": int(mcmc_adapt_steps),
                             "seed": int(seed_counter), "init_log_step": float(mcmc_init_log_step),
                             "target_accept": float(mcmc_target_accept),
                             "model_variant": model_variant_local,
                             "theta_center": theta_center_job,
                             "model_b_cfg": dict(model_b_cfg)})
            seed_counter += 1

    cpu_total = os.cpu_count() or 1
    workers = min(int(mcmc_max_workers), cpu_total, len(jobs))
    workers = max(1, workers)
    print(f"Using {workers} worker processes for MCMC chains")
    print(f"Worker context mode: {'compact-shared' if compact_worker_context else 'legacy-payload'}")

    chain_results = []
    _warmup_numba_derivs()
    t_phase = timeit.default_timer()

    _prev_worker_quiet = os.environ.get(_WORKER_QUIET_ENV)
    os.environ[_WORKER_QUIET_ENV] = "1"
    try:
        theta_center = float(np.log(float(ballistic_coefficient_nominal)))
        if model_variant_local == "B":
            theta_center = float(np.log(_model_b_nominal_face_area_m2()))

        mcmc_cfg_shared = {"n_steps": int(mcmc_steps),
                           "adapt_steps": int(mcmc_adapt_steps),
                           "init_log_step": float(mcmc_init_log_step),
                           "target_accept": float(mcmc_target_accept),
                           "theta_center": theta_center,
                           "model_variant": model_variant_local,
                           "model_b_cfg": dict(model_b_cfg)}

        if compact_worker_context:
            compact_jobs = jobs

            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=_mcmc_worker_init,
                                     initargs=(sat_cases, mcmc_cfg_shared, int(mcmc_profile_timing) == 1)) as executor:
                t_submit0 = timeit.default_timer()
                futures = [executor.submit(_mcmc_chain_worker, job) for job in compact_jobs]
                submit_sec = timeit.default_timer() - t_submit0

                n_done = 0
                n_total = len(futures)
                first_result_sec = np.nan
                failed_chain_rows = []
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as exc:
                        n_done += 1
                        failed_chain_rows.append({"sat_id": "unknown", "chain_id": -1, "error": str(exc)})
                        print(f"MCMC progress: {n_done}/{n_total} | chain failed with exception: {exc}")
                        continue

                    n_done += 1
                    if int(res.get("failed", 0)) == 1:
                        failed_chain_rows.append({"sat_id": res.get("sat_id", "unknown"),
                                                  "chain_id": int(res.get("chain_id", -1)),
                                                  "error": str(res.get("fail_reason", "unknown"))})
                        print(f"MCMC progress: {n_done}/{n_total} | sat={res['sat_id']} chain={res['chain_id']} "
                              f"| initialization failed ({res.get('fail_reason', 'unknown')})")
                        continue

                    chain_results.append(res)
                    if not np.isfinite(first_result_sec):
                        first_result_sec = timeit.default_timer() - t_phase
                    progress_every = max(1, int(mcmc_progress_print_every))
                    if (n_done % progress_every == 0) or (n_done == n_total):
                        print(f"MCMC progress: {n_done}/{n_total} | sat={res['sat_id']} chain={res['chain_id']} "
                              f"| acc={res['accept_rate_total']:.3f} | final_sd={res['final_prop_sd']:.4f}")
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=_warmup_numba_derivs) as executor:
                t_submit0 = timeit.default_timer()
                futures = [executor.submit(_mcmc_chain_worker, job) for job in jobs]
                submit_sec = timeit.default_timer() - t_submit0

                n_done = 0
                n_total = len(futures)
                first_result_sec = np.nan
                failed_chain_rows = []
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as exc:
                        n_done += 1
                        failed_chain_rows.append({"sat_id": "unknown", "chain_id": -1, "error": str(exc)})
                        print(f"MCMC progress: {n_done}/{n_total} | chain failed with exception: {exc}")
                        continue

                    n_done += 1
                    if int(res.get("failed", 0)) == 1:
                        failed_chain_rows.append({"sat_id": res.get("sat_id", "unknown"),
                                                  "chain_id": int(res.get("chain_id", -1)),
                                                  "error": str(res.get("fail_reason", "unknown"))})
                        print(f"MCMC progress: {n_done}/{n_total} | sat={res['sat_id']} chain={res['chain_id']} "
                              f"| initialization failed ({res.get('fail_reason', 'unknown')})")
                        continue

                    chain_results.append(res)
                    if not np.isfinite(first_result_sec):
                        first_result_sec = timeit.default_timer() - t_phase
                    progress_every = max(1, int(mcmc_progress_print_every))
                    if (n_done % progress_every == 0) or (n_done == n_total):
                        print(f"MCMC progress: {n_done}/{n_total} | sat={res['sat_id']} chain={res['chain_id']} "
                              f"| acc={res['accept_rate_total']:.3f} | final_sd={res['final_prop_sd']:.4f}")
    finally:
        if _prev_worker_quiet is None:
            os.environ.pop(_WORKER_QUIET_ENV, None)
        else:
            os.environ[_WORKER_QUIET_ENV] = _prev_worker_quiet

    chains_sec = timeit.default_timer() - t_phase
    if int(mcmc_enable_phase_timing) == 1:
        print(f"Timing | executor submit/serialization: {submit_sec:.2f} s")
        if np.isfinite(first_result_sec):
            print(f"Timing | first chain result latency: {first_result_sec:.2f} s")
        print(f"Timing | all chains complete: {chains_sec:.2f} s")
        if int(mcmc_profile_timing) == 1 and len(chain_results) > 0:
            sum_like = float(np.sum([r.get("timing_likelihood_eval_s", 0.0) for r in chain_results]))
            sum_solve = float(np.sum([r.get("timing_solve_ivp_s", 0.0) for r in chain_results]))
            sum_prop = float(np.sum([r.get("timing_proposal_s", 0.0) for r in chain_results]))
            sum_worker = float(np.sum([r.get("timing_worker_total_s", 0.0) for r in chain_results]))
            print(f"Timing | worker aggregate likelihood eval: {sum_like:.2f} s")
            print(f"Timing | worker aggregate solve_ivp: {sum_solve:.2f} s")
            print(f"Timing | worker aggregate proposal/accept: {sum_prop:.2f} s")
            print(f"Timing | worker aggregate total: {sum_worker:.2f} s")

    # Group results by satellite
    by_sat = {}
    for res in chain_results:
        by_sat.setdefault(res["sat_id"], []).append(res)

    summary_rows = []
    t_phase = timeit.default_timer()

    # Save a copy of the observation series used, for reproducibility
    for sc in sat_cases:
        obs_fn = outdir / f"{sc['sat_id']}_observations_used.csv"
        df_obs = pd.DataFrame({"t_obs_sec": sc["t_obs_sec"], "t_obs_days": sc["t_obs_sec"] / 86400.0,
                               "sma_obs_km": sc["sma_obs_km"], "delta_a_obs_km": sc["delta_a_obs_km"]})
        df_obs.to_csv(obs_fn, index=False)

    for sat_id_local, res_list in by_sat.items():
        res_list = sorted(res_list, key=lambda z: z["chain_id"])

        if len(res_list) == 0:
            continue

        # Save full chain traces
        if int(mcmc_save_chain_traces) == 1:
            trace_fn = outdir / f"{sat_id_local}_chain_traces.csv"
            rows = []
            for r in res_list:
                n_local = len(r["samples_log_beta"])
                beta_series = np.exp(r["samples_log_beta"])
                area_face_series = np.asarray(r.get("samples_area_face_m2", np.full(n_local, np.nan)), dtype=np.float64)
                area_eff_series = np.asarray(r.get("samples_area_eff_m2", np.full(n_local, np.nan)), dtype=np.float64)
                for k in range(n_local):
                    rows.append([sat_id_local, r["chain_id"], k, float(r["samples_log_beta"][k]),
                                 float(beta_series[k]), float(r["samples_lp"][k]),
                                 int(r["accepted"][k]), float(r["prop_sd_hist"][k]),
                                 float(area_face_series[k]), float(area_eff_series[k])])
            pd.DataFrame(rows, columns=["sat_id", "chain_id", "iter", "log_beta", "beta_m2_per_kg",
                                        "log_posterior", "accepted", "proposal_sd_logbeta",
                                        "area_face_m2", "area_eff_m2"]).to_csv(trace_fn, index=False)

        # Posterior extraction after burn/thin
        model_variant_sat = str(res_list[0].get("mcmc_model_variant", model_variant_local)).upper().strip()
        post_chains = []
        for r in res_list:
            if model_variant_sat == "B":
                s = np.asarray(r.get("samples_log_area_face", np.array([], dtype=np.float64)), dtype=np.float64)
            else:
                s = np.asarray(r["samples_log_beta"], dtype=np.float64)
            s_post = s[int(mcmc_burn)::int(max(1, mcmc_thin))]
            post_chains.append(np.asarray(s_post, dtype=np.float64))

        # Equalize lengths for R-hat
        min_len = min(len(s) for s in post_chains)
        if min_len < 2:
            rhat = np.nan
            post_mat = None
            all_post = np.concatenate(post_chains)
        else:
            post_mat = np.vstack([s[:min_len] for s in post_chains])
            rhat = _compute_rhat(post_mat)
            all_post = post_mat.reshape(-1)

        if model_variant_sat == "B":
            area_face_all = np.exp(all_post)
            area_eff_all = _model_b_effective_area_m2(1.0) * area_face_all
            cd_local = float(model_b_cfg.get("Cd", Cd))
            mass_local = float(model_b_cfg.get("mass_kg", mass))
            beta_all = (cd_local * area_eff_all) / mass_local
        else:
            beta_all = np.exp(all_post)
            area_face_all = np.full_like(beta_all, np.nan)
            area_eff_all = np.full_like(beta_all, np.nan)

        bc_all = 1.0 / np.clip(beta_all, 1e-16, None)

        q05, q50, q95 = np.quantile(beta_all, [0.05, 0.50, 0.95])
        bc_q05, bc_q50, bc_q95 = np.quantile(bc_all, [0.05, 0.50, 0.95])
        if model_variant_sat == "B":
            af_q05, af_q50, af_q95 = np.quantile(area_face_all, [0.05, 0.50, 0.95])
            ae_q05, ae_q50, ae_q95 = np.quantile(area_eff_all, [0.05, 0.50, 0.95])
        else:
            af_q05 = af_q50 = af_q95 = np.nan
            ae_q05 = ae_q50 = ae_q95 = np.nan

        summary_rows.append({"sat_id": sat_id_local,
                             "mcmc_model_variant": model_variant_sat,
                             "n_chains": int(len(res_list)),
                             "steps": int(mcmc_steps),
                             "burn": int(mcmc_burn),
                             "thin": int(mcmc_thin),
                             "posterior_samples": int(beta_all.size),
                             "beta_mean_m2_per_kg": float(np.mean(beta_all)),
                             "beta_std_m2_per_kg": float(np.std(beta_all, ddof=1)) if beta_all.size > 1 else 0.0,
                             "beta_median_m2_per_kg": float(q50),
                             "beta_q05_m2_per_kg": float(q05),
                             "beta_q95_m2_per_kg": float(q95),
                             "bc_mean_kg_per_m2": float(np.mean(bc_all)),
                             "bc_std_kg_per_m2": float(np.std(bc_all, ddof=1)) if bc_all.size > 1 else 0.0,
                             "bc_median_kg_per_m2": float(bc_q50),
                             "bc_q05_kg_per_m2": float(bc_q05),
                             "bc_q95_kg_per_m2": float(bc_q95),
                             "area_face_mean_m2": float(np.mean(area_face_all)) if np.any(np.isfinite(area_face_all)) else np.nan,
                             "area_face_median_m2": float(af_q50),
                             "area_face_q05_m2": float(af_q05),
                             "area_face_q95_m2": float(af_q95),
                             "area_eff_mean_m2": float(np.mean(area_eff_all)) if np.any(np.isfinite(area_eff_all)) else np.nan,
                             "area_eff_median_m2": float(ae_q50),
                             "area_eff_q05_m2": float(ae_q05),
                             "area_eff_q95_m2": float(ae_q95),
                             "log_beta_mean": float(np.mean(np.log(np.clip(beta_all, 1e-300, None)))),
                             "rhat_log_beta": float(rhat) if np.isfinite(rhat) else np.nan,
                             "mean_accept_rate": float(np.mean([r["accept_rate_total"] for r in res_list])),
                             "mean_final_prop_sd": float(np.mean([r["final_prop_sd"] for r in res_list])),
                             "n_obs": int(res_list[0]["n_obs"]),
                             "sigma_a_km": float(res_list[0]["sigma_a_km"]),
                             "obs_first": res_list[0]["obs_first"],
                             "obs_last": res_list[0]["obs_last"],
                             "start_timestamp": res_list[0]["start_timestamp"]})

        # Save posterior-only samples
        post_fn = outdir / f"{sat_id_local}_posterior_samples.csv"
        pd.DataFrame({
            "beta_m2_per_kg": beta_all,
            "bc_kg_per_m2": bc_all,
            "log_beta": all_post,
            "area_face_m2": area_face_all,
            "area_eff_m2": area_eff_all,
        }).to_csv(post_fn, index=False)

    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows).sort_values("sat_id").reset_index(drop=True)
    else:
        summary_df = pd.DataFrame(columns=["sat_id", "n_chains", "steps", "burn", "thin", "posterior_samples",
                                           "mcmc_model_variant",
                                           "beta_mean_m2_per_kg", "beta_std_m2_per_kg", "beta_median_m2_per_kg",
                                           "beta_q05_m2_per_kg", "beta_q95_m2_per_kg",
                                           "bc_mean_kg_per_m2", "bc_std_kg_per_m2", "bc_median_kg_per_m2",
                                           "bc_q05_kg_per_m2", "bc_q95_kg_per_m2",
                                           "area_face_mean_m2", "area_face_median_m2", "area_face_q05_m2", "area_face_q95_m2",
                                           "area_eff_mean_m2", "area_eff_median_m2", "area_eff_q05_m2", "area_eff_q95_m2",
                                           "log_beta_mean", "rhat_log_beta",
                                           "mean_accept_rate", "mean_final_prop_sd", "n_obs", "sigma_a_km",
                                           "obs_first", "obs_last", "start_timestamp"])
    if model_variant_local == "A":
        legacy_cols = ["sat_id", "n_chains", "steps", "burn", "thin", "posterior_samples",
                       "beta_mean_m2_per_kg", "beta_std_m2_per_kg", "beta_median_m2_per_kg",
                       "beta_q05_m2_per_kg", "beta_q95_m2_per_kg",
                       "bc_mean_kg_per_m2", "bc_std_kg_per_m2", "bc_median_kg_per_m2",
                       "bc_q05_kg_per_m2", "bc_q95_kg_per_m2",
                       "log_beta_mean", "rhat_log_beta",
                       "mean_accept_rate", "mean_final_prop_sd", "n_obs", "sigma_a_km",
                       "obs_first", "obs_last", "start_timestamp"]
        for col in legacy_cols:
            if col not in summary_df.columns:
                summary_df[col] = np.nan
        summary_out_df = summary_df[legacy_cols].copy()
        summary_csv = outdir / "modelA_mcmc_summary.csv"
    else:
        summary_out_df = summary_df
        summary_csv = outdir / "modelB_mcmc_summary.csv"

    summary_out_df.to_csv(summary_csv, index=False)

    output_sec = timeit.default_timer() - t_phase
    total_sec = timeit.default_timer() - t_total_start
    if int(mcmc_enable_phase_timing) == 1:
        print(f"Timing | output + summaries: {output_sec:.2f} s")
        print(f"Timing | total MCMC runtime: {total_sec:.2f} s")

    if int(mcmc_make_plots) == 1:
        t_plot = timeit.default_timer()
        _save_model_a_mcmc_plots(outdir, by_sat, summary_out_df)
        if int(mcmc_enable_phase_timing) == 1:
            print(f"Timing | plot generation: {timeit.default_timer() - t_plot:.2f} s")

    if len(failed_chain_rows) > 0:
        failed_csv = outdir / ("modelA_failed_chains.csv" if model_variant_local == "A" else "modelB_failed_chains.csv")
        pd.DataFrame(failed_chain_rows).to_csv(failed_csv, index=False)
        print(f"Model {model_variant_local} failed-chain log saved to: {failed_csv}")

    print(f"\nModel {model_variant_local} MCMC summary saved to: {summary_csv}")
    print(summary_out_df.to_string(index=False))

    return {"sat_cases": sat_cases, "chain_results": chain_results,
            "summary_df": summary_out_df, "output_dir": str(outdir)}

def main():
    start = timeit.default_timer()

    print(f"\nIntegration process:")
    print(f"TF = {tf / tu_conv:.2f} {unit}")
    print(f"dt = {dt / tu_conv:.2e} {unit}")
    print(f"nt = {nt}")
    print('\nRunning...\n')

    if run_model_a_mcmc == 1:
        results_mcmc = _run_model_a_mcmc()
        model_variant_local = str(mcmc_model_variant).upper().strip()

        stop = timeit.default_timer()
        runtime = stop - start
        if runtime < 60.0:
            print(f"Model {model_variant_local} MCMC runtime = {runtime:.2f} seconds.\n")
        elif runtime < 3600.0:
            print(f"Model {model_variant_local} MCMC runtime = {runtime / 60.0:.2f} minutes.\n")
        else:
            print(f"Model {model_variant_local} MCMC runtime = {runtime / 3600.0:.2f} hours.\n")
        return

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

    if atm_model == 1 and k_atm_drag == 1:
        times, state, reentry_t_abs = _run_with_msis_grids(t_eval_override=t_eval_main)
    else:
        if t_final_main <= 0.0:
            times = np.array([0.0], dtype=np.float64)
            state = Xb_init.reshape((18, 1))
            reentry_t_abs = -1.0
        else:
            solution = solve_ivp(Derivs, [0.0, t_final_main], Xb_init, events=[Reentry], method='DOP853',
                             t_eval=t_eval_main, rtol=1.e-10, atol=1.e-12,
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
        import matplotlib.pyplot as plt
        plt.plot(x_sat[0, :], x_sat[1, :], linewidth=0.5)
        plt.show()

    # Plot the satellite's orbit (3D)
    if plot_sv_3d_sat == 1:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_sat[0, :], x_sat[1, :], x_sat[2, :], linewidth=0.5)
        plt.show()

    # Plot the sun's orbit (3D)
    if plot_sv_3d_sun == 1:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_sun[0, :], x_sun[1, :], x_sun[2, :], linewidth=0.5)
        plt.show()

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
        import matplotlib.pyplot as plt
        plt.plot(time, orb[:, 0], linewidth=0.5)
        plt.xlabel('Time (' + unit + ')')
        plt.ylabel('Semi-major axis (km)')
        plt.title('Semi-major axis vs. Time')
        plt.grid(True)
        plt.show()

    # Plot eccentricity vs. time
    if plot_ecc == 1:
        import matplotlib.pyplot as plt
        plt.plot(time, orb[:, 1], linewidth=0.5)
        plt.xlabel('Time (' + unit + ')')
        plt.ylabel('Eccentricity')
        plt.title('Eccentricity vs. Time')
        plt.grid(True)
        plt.show()

    # Plot perigee and apogee altitudes vs. time
    if plot_rp_ra == 1:
        import matplotlib.pyplot as plt
        plt.plot(time, rp, label='Perigee', linewidth=0.5)
        plt.plot(time, ra, label='Apogee', linewidth=0.5)
        plt.xlabel('Time (' + unit + ')')
        plt.ylabel('Altitude (km)')
        plt.title('Perigee and Apogee Altitudes vs. Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Save the orbital elements of the satellite in a file
    if save_sat_oe == 1:
        filename = f'{fd}/orbital_elements_sat.dat'
        with open(filename, 'w') as f:
            for j in range(n):
                f.write(f"{times[j] / tu_conv:<15.8e} {orb[j][0]:<15.8e} "
                        f"{orb[j][1]:<15.8e} {np.rad2deg(orb[j][2]):<15.8e} "
                        f"{np.rad2deg(orb[j][3]):<15.8e} {np.rad2deg(orb[j][4]):<15.8e} "
                        f"{np.rad2deg(orb[j][5]):<15.8e}\n")

def _run_optimization_benchmarks():
    print("\nRunning optimization benchmarks/regression checks...\n")

    def _safe_rel(a, b):
        den = np.maximum(np.abs(a), 1e-30)
        return np.abs(a - b) / den

    # -----------------------------
    # GM2008 nmax=2,mmax=0 equivalence
    # -----------------------------
    MM = nmax + 1
    P_a = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    Pl_a = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    sml_a = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    cml_a = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    out_fast = np.ascontiguousarray(np.zeros(3, dtype=np.float64))
    out_ref = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

    egm_abs_max = 0.0
    egm_rel_max = 0.0
    for k in range(16):
        xi = np.array([6800.0 + 5.0 * k, -1200.0 + 3.0 * k, 500.0 - 2.0 * k], dtype=np.float64)
        tt = 1234.0 * k
        EGM2008(2, 0, xi, C, S, tt, earth_GM, earth_Re, earth_spin, GST0, P_a, Pl_a, sml_a, cml_a, out_fast)
        _EGM2008_n2_m0_reference_generic(xi, C, S, tt, earth_GM, earth_Re, earth_spin, GST0,
                                         P_a, Pl_a, sml_a, cml_a, out_ref)
        dabs = np.max(np.abs(out_fast - out_ref))
        drel = np.max(_safe_rel(out_fast, out_ref))
        if dabs > egm_abs_max:
            egm_abs_max = float(dabs)
        if drel > egm_rel_max:
            egm_rel_max = float(drel)

    print(f"EGM n2m0 fast-vs-reference | max_abs={egm_abs_max:.6e} max_rel={egm_rel_max:.6e}")

    # -----------------------------
    # Nominal propagation benchmark + regression (legacy payload mode vs optimized mode)
    # -----------------------------
    t_final_bench = min(float(_seconds_until_date_cutoff(epoch)), 2.0 * 86400.0)
    t_eval_bench = _build_t_eval_with_cutoff(t_final_bench)

    def _derivs_nominal_ref(t, f, P, Pl, sml, cml, tmp3, ballistic_coefficient, gst0_case):
        Re = earth_Re

        x_sun = f[0:6]
        r_sun = np.sqrt(x_sun[0] ** 2 + x_sun[1] ** 2 + x_sun[2] ** 2)

        x_moon = f[6:12]
        r_moon = np.sqrt(x_moon[0] ** 2 + x_moon[1] ** 2 + x_moon[2] ** 2)

        x_sat = f[12:18]
        r = np.sqrt(x_sat[0] ** 2 + x_sat[1] ** 2 + x_sat[2] ** 2)

        dxsundt = x_sun[3]
        dysundt = x_sun[4]
        dzsundt = x_sun[5]

        mu_sun_earth = sun_GM + earth_GM
        ddxsundt = -mu_sun_earth * x_sun[0] / r_sun ** 3
        ddysundt = -mu_sun_earth * x_sun[1] / r_sun ** 3
        ddzsundt = -mu_sun_earth * x_sun[2] / r_sun ** 3

        dxmoondt = x_moon[3]
        dymoondt = x_moon[4]
        dzmoondt = x_moon[5]

        AC3b(x_moon[0:3], x_sun[0:3], sun_GM, tmp3)
        ac3b_sun0 = tmp3[0]
        ac3b_sun1 = tmp3[1]
        ac3b_sun2 = tmp3[2]

        J2acc(earth_GM, earth_J2, earth_Re, x_moon[0:3], tmp3)
        acj20 = tmp3[0]
        acj21 = tmp3[1]
        acj22 = tmp3[2]
        mu_moon_earth = moon_GM + earth_GM
        ddxmoondt = -mu_moon_earth * x_moon[0] / r_moon ** 3 + acj20 + ac3b_sun0
        ddymoondt = -mu_moon_earth * x_moon[1] / r_moon ** 3 + acj21 + ac3b_sun1
        ddzmoondt = -mu_moon_earth * x_moon[2] / r_moon ** 3 + acj22 + ac3b_sun2

        axp = 0.0
        ayp = 0.0
        azp = 0.0

        if k_EGM2008 == 1:
            if nmax == 2 and mmax == 0:
                _EGM2008_n2_m0_reference_generic(x_sat[0:3], C, S, t, earth_GM, earth_Re, earth_spin, gst0_case,
                                                 P, Pl, sml, cml, tmp3)
            else:
                EGM2008(nmax, mmax, x_sat[0:3], C, S, t, earth_GM, earth_Re, earth_spin, gst0_case,
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
            atm_drag(x_sat, 1.0, ballistic_coefficient, earth_spin, tmp3)
            axp += tmp3[0]
            ayp += tmp3[1]
            azp += tmp3[2]

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

    xb_case = np.ascontiguousarray(Xb_init, dtype=np.float64)
    MM = nmax + 1
    p_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    pl_pre = np.ascontiguousarray(np.zeros((MM, MM), dtype=np.float64))
    sml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    cml_pre_local = np.ascontiguousarray(np.zeros(MM, dtype=np.float64))
    tmp3_pre_local = np.ascontiguousarray(np.zeros(3, dtype=np.float64))

    _warmup_numba_derivs()

    t0 = timeit.default_timer()
    sol_ref = solve_ivp(_derivs_nominal_ref, [0.0, t_final_bench], xb_case, events=[Reentry], method='DOP853',
                        t_eval=t_eval_bench, rtol=1.e-10, atol=1.e-12,
                        args=(p_pre, pl_pre, sml_pre_local, cml_pre_local, tmp3_pre_local,
                              ballistic_coefficient_nominal, GST0))
    t_ref = timeit.default_timer() - t0

    t0 = timeit.default_timer()
    sol_nom = solve_ivp(Derivs, [0.0, t_final_bench], xb_case, events=[Reentry], method='DOP853',
                        t_eval=t_eval_bench, rtol=1.e-10, atol=1.e-12,
                        args=(p_pre, pl_pre, sml_pre_local, cml_pre_local, tmp3_pre_local,
                              ballistic_coefficient_nominal, GST0))
    t_nom = timeit.default_timer() - t0

    nom_ref_state = np.asarray(sol_ref.y[12:18, -1], dtype=np.float64)
    nom_opt_state = np.asarray(sol_nom.y[12:18, -1], dtype=np.float64)
    nom_state_abs = float(np.max(np.abs(nom_opt_state - nom_ref_state)))
    nom_state_rel = float(np.max(_safe_rel(nom_opt_state, nom_ref_state)))
    te_ref = -1.0
    te_opt = -1.0
    if getattr(sol_ref, 't_events', None) is not None and len(sol_ref.t_events) > 0 and np.asarray(sol_ref.t_events[0]).size > 0:
        te_ref = float(sol_ref.t_events[0][0])
    if getattr(sol_nom, 't_events', None) is not None and len(sol_nom.t_events) > 0 and np.asarray(sol_nom.t_events[0]).size > 0:
        te_opt = float(sol_nom.t_events[0][0])
    nom_event_abs = float(abs(te_opt - te_ref)) if (te_opt >= 0.0 and te_ref >= 0.0) else 0.0

    print(f"Nominal runtime reference-RHS: {t_ref:.3f} s")
    print(f"Nominal runtime optimized-RHS: {t_nom:.3f} s")
    if t_nom > 0.0:
        print(f"Nominal speedup (reference/optimized): {t_ref / t_nom:.3f}x")

    # -----------------------------
    # Small batch benchmark: legacy payload vs compact payload
    # -----------------------------
    ns = 8
    if use_tle_initial_conditions == 1 and tle_oe_cases is not None and tle_oe_cases.shape[0] >= ns:
        oe_cases_b = np.ascontiguousarray(tle_oe_cases[:ns], dtype=np.float64)
        sat_ids_b = list(tle_sat_ids_selected[:ns]) if tle_sat_ids_selected is not None else [f"case_{k:05d}" for k in range(ns)]
        if tle_start_datetimes_selected is not None and len(tle_start_datetimes_selected) >= ns:
            start_ts_b = list(tle_start_datetimes_selected[:ns])
        else:
            start_ts_b = [epoch] * ns
    else:
        oe_cases_b = np.ascontiguousarray(np.tile(oe_sat, (ns, 1)), dtype=np.float64)
        sat_ids_b = [f"case_{k:05d}" for k in range(ns)]
        start_ts_b = [epoch] * ns

    bc_b = _sample_batch_ballistic_coefficients(ns, batch_random_seed)
    workers_b = min(8, _resolve_batch_worker_count())

    t0 = timeit.default_timer()
    batch_legacy = _run_parallel_batch(oe_cases=oe_cases_b, sat_ids=sat_ids_b, start_timestamps=start_ts_b,
                                       ballistic_coefficients=bc_b, workers_override=workers_b,
                                       show_case_progress=False, write_summary=False, write_trajectories=False,
                                       compact_payload=False)
    t_legacy = timeit.default_timer() - t0

    t0 = timeit.default_timer()
    batch_compact = _run_parallel_batch(oe_cases=oe_cases_b, sat_ids=sat_ids_b, start_timestamps=start_ts_b,
                                        ballistic_coefficients=bc_b, workers_override=workers_b,
                                        show_case_progress=False, write_summary=False, write_trajectories=False,
                                        compact_payload=True)
    t_compact = timeit.default_timer() - t0

    batch_state_abs = 0.0
    batch_state_rel = 0.0
    batch_event_abs = 0.0
    for k in range(ns):
        v0 = np.asarray(batch_legacy[k]['state_sat'][:, -1], dtype=np.float64)
        v1 = np.asarray(batch_compact[k]['state_sat'][:, -1], dtype=np.float64)
        dabs = float(np.max(np.abs(v0 - v1)))
        drel = float(np.max(_safe_rel(v0, v1)))
        if dabs > batch_state_abs:
            batch_state_abs = dabs
        if drel > batch_state_rel:
            batch_state_rel = drel

        te0 = float(batch_legacy[k]['t_115_s'])
        te1 = float(batch_compact[k]['t_115_s'])
        dte = abs(te0 - te1)
        if dte > batch_event_abs:
            batch_event_abs = dte

    print(f"Batch runtime legacy-payload : {t_legacy:.3f} s")
    print(f"Batch runtime compact-payload: {t_compact:.3f} s")
    if t_compact > 0.0:
        print(f"Batch speedup (legacy/compact): {t_legacy / t_compact:.3f}x")

    print(f"Nominal final-state diff reference-vs-optimized | max_abs={nom_state_abs:.6e} max_rel={nom_state_rel:.6e}")
    print(f"Nominal event-time diff reference-vs-optimized | max_abs={nom_event_abs:.6e}")
    print(f"Batch final-state diff legacy-vs-compact | max_abs={batch_state_abs:.6e} max_rel={batch_state_rel:.6e}")
    print(f"Batch event-time diff legacy-vs-compact | max_abs={batch_event_abs:.6e}")

    # -----------------------------
    # MCMC reproducibility: legacy payload vs compact shared-context
    # -----------------------------
    try:
        sat_cases_m = _prepare_model_a_mcmc_cases()
        if len(sat_cases_m) > 0:
            sc0 = sat_cases_m[0]
            tiny_steps = 12
            tiny_adapt = 6
            tiny_burn = 6
            tiny_thin = 1
            tiny_seeds = [int(mcmc_seed), int(mcmc_seed) + 1]

            legacy_res = []
            t0 = timeit.default_timer()
            for chain_id, seed in enumerate(tiny_seeds):
                legacy_res.append(_mcmc_chain_worker({"sat_case": sc0, "chain_id": int(chain_id),
                                                      "n_steps": int(tiny_steps), "adapt_steps": int(tiny_adapt),
                                                      "seed": int(seed), "init_log_step": float(mcmc_init_log_step),
                                                      "target_accept": float(mcmc_target_accept)}))
            t_tiny_legacy = timeit.default_timer() - t0

            tiny_cfg = {"n_steps": int(tiny_steps), "adapt_steps": int(tiny_adapt),
                        "init_log_step": float(mcmc_init_log_step),
                        "target_accept": float(mcmc_target_accept),
                        "mu0_log_beta": float(np.log(float(ballistic_coefficient_nominal)))}
            _mcmc_worker_init((sc0,), tiny_cfg, True)

            compact_res = []
            t0 = timeit.default_timer()
            for chain_id, seed in enumerate(tiny_seeds):
                compact_res.append(_mcmc_chain_worker({"sat_case_idx": 0, "chain_id": int(chain_id),
                                                       "seed": int(seed)}))
            t_tiny_compact = timeit.default_timer() - t0

            def _tiny_summary(res_list):
                post = []
                acc = []
                for r in res_list:
                    if int(r.get("failed", 0)) == 1:
                        continue
                    s = np.asarray(r["samples_log_beta"], dtype=np.float64)
                    s_post = s[int(tiny_burn)::int(max(1, tiny_thin))]
                    if s_post.size > 0:
                        post.append(s_post)
                    acc.append(float(r.get("accept_rate_total", np.nan)))

                if len(post) == 0:
                    return {"mean_log_beta": np.nan, "q05": np.nan, "q50": np.nan, "q95": np.nan, "acc_mean": np.nan}

                all_post = np.concatenate(post)
                q05, q50, q95 = np.quantile(all_post, [0.05, 0.50, 0.95])
                return {"mean_log_beta": float(np.mean(all_post)), "q05": float(q05), "q50": float(q50), "q95": float(q95),
                        "acc_mean": float(np.nanmean(np.asarray(acc, dtype=np.float64)))}

            sm_legacy = _tiny_summary(legacy_res)
            sm_compact = _tiny_summary(compact_res)

            diffs_abs = np.array([abs(sm_legacy["mean_log_beta"] - sm_compact["mean_log_beta"]),
                                  abs(sm_legacy["q05"] - sm_compact["q05"]),
                                  abs(sm_legacy["q50"] - sm_compact["q50"]),
                                  abs(sm_legacy["q95"] - sm_compact["q95"]),
                                  abs(sm_legacy["acc_mean"] - sm_compact["acc_mean"])], dtype=np.float64)

            vals_legacy = np.array([sm_legacy["mean_log_beta"], sm_legacy["q05"], sm_legacy["q50"],
                                    sm_legacy["q95"], sm_legacy["acc_mean"]], dtype=np.float64)
            vals_compact = np.array([sm_compact["mean_log_beta"], sm_compact["q05"], sm_compact["q50"],
                                     sm_compact["q95"], sm_compact["acc_mean"]], dtype=np.float64)
            diffs_rel = _safe_rel(vals_compact, vals_legacy)

            max_abs = float(np.nanmax(diffs_abs))
            max_rel = float(np.nanmax(diffs_rel))
            strict_abs_tol = 1e-10
            strict_rel_tol = 1e-8
            status = "PASS" if (max_abs <= strict_abs_tol or max_rel <= strict_rel_tol) else "WARN"

            print(f"Tiny MCMC runtime legacy-payload : {t_tiny_legacy:.3f} s")
            print(f"Tiny MCMC runtime compact-context: {t_tiny_compact:.3f} s")
            if t_tiny_compact > 0.0:
                print(f"Tiny MCMC speedup (legacy/compact): {t_tiny_legacy / t_tiny_compact:.3f}x")
            print(f"Tiny MCMC reproducibility (legacy vs compact) | max_abs={max_abs:.6e} max_rel={max_rel:.6e} | {status}")
            if status != "PASS":
                print("Tiny MCMC reproducibility warning: differences exceed strict threshold.")
        else:
            print("Tiny MCMC reproducibility: skipped (no prepared satellite cases).")
    except Exception as exc:
        print(f"Tiny MCMC reproducibility: skipped ({exc})")

    print("\nBenchmark/regression checks complete.\n")

def _run_mcmc_variant_smoke_checks():
    print("\nRunning MCMC variant smoke checks...\n")

    variant_prev = str(mcmc_model_variant)
    atm_prev = int(atm_model)

    # Model A smoke regression: ensure legacy log-beta outputs still exist.
    sat_cases_a = _prepare_mcmc_cases("A")
    sc_a = dict(sat_cases_a[0])
    sc_a["t_obs_sec"] = np.ascontiguousarray(sc_a["t_obs_sec"][:3], dtype=np.float64)
    sc_a["t_eval_sec"] = np.ascontiguousarray(sc_a["t_eval_sec"][:3], dtype=np.float64)
    sc_a["sma_obs_km"] = np.ascontiguousarray(sc_a["sma_obs_km"][:3], dtype=np.float64)
    sc_a["delta_a_obs_km"] = np.ascontiguousarray(sc_a["delta_a_obs_km"][:3], dtype=np.float64)

    res_a = _mcmc_chain_worker({"sat_case": sc_a, "chain_id": 0, "n_steps": 4, "adapt_steps": 2,
                                "seed": int(mcmc_seed), "init_log_step": float(mcmc_init_log_step),
                                  "target_accept": float(mcmc_target_accept), "model_variant": "A",
                                  "theta_center": float(np.log(float(ballistic_coefficient_nominal)))})
    if int(res_a.get("failed", 0)) == 1:
        raise RuntimeError(f"Model A smoke failed: {res_a.get('fail_reason', 'unknown')}")
    if "samples_log_beta" not in res_a:
        raise RuntimeError("Model A smoke failed: samples_log_beta missing")

    # Model B smoke: MSIS cache loads, area fields are emitted, and no global atm_model switch is required.
    _ensure_msis_mcmc_cache()
    sat_cases_b = _prepare_mcmc_cases("B")
    sc_b = dict(sat_cases_b[0])
    if pd.Timestamp(sc_b["start_timestamp"]).strftime("%Y-%m-%d") != sc_b["epoch_date_key"]:
        raise RuntimeError("Model B date alignment sanity failed before sampling")

    # Use a synthetic short same-day window to keep smoke checks deterministic and fast.
    sma0 = float(sc_b["sma_obs_km"][0])
    sc_b["t_obs_sec"] = np.ascontiguousarray(np.array([0.0, 3600.0, 7200.0], dtype=np.float64))
    sc_b["t_eval_sec"] = np.ascontiguousarray(np.array([0.0, 3600.0, 7200.0], dtype=np.float64))
    sc_b["sma_obs_km"] = np.ascontiguousarray(np.array([sma0, sma0, sma0], dtype=np.float64))
    sc_b["delta_a_obs_km"] = np.ascontiguousarray(np.array([0.0, 0.0, 0.0], dtype=np.float64))

    res_b = _mcmc_chain_worker({"sat_case": sc_b, "chain_id": 0, "n_steps": 3, "adapt_steps": 1,
                                "seed": int(mcmc_seed) + 11, "init_log_step": float(mcmc_init_log_step),
                                "target_accept": float(mcmc_target_accept), "model_variant": "B",
                                "theta_center": float(np.log(_model_b_nominal_face_area_m2()))})
    if int(res_b.get("failed", 0)) == 1:
        raise RuntimeError(f"Model B smoke failed: {res_b.get('fail_reason', 'unknown')}")
    if "samples_area_face_m2" not in res_b or "samples_area_eff_m2" not in res_b:
        raise RuntimeError("Model B smoke failed: area sample fields missing")
    if not np.any(np.isfinite(np.asarray(res_b["samples_area_face_m2"], dtype=np.float64))):
        raise RuntimeError("Model B smoke failed: area_face samples are not finite")

    if int(atm_model) != int(atm_prev):
        raise RuntimeError("Model B smoke failed: global atm_model changed unexpectedly")

    print("MCMC variant smoke checks passed.\n")

if __name__ == '__main__':
    if os.getenv('MCMC_BENCHMARK', '0') == '1':
        _run_optimization_benchmarks()
        if os.getenv('MCMC_SMOKE_TEST', '0') == '1':
            _run_mcmc_variant_smoke_checks()
    else:
        main()