"""Consolidated orbital feature-engineering and utility functions.

Includes orbital element conversions, phase-angle helpers, altitude/shell features,
and dataframe helpers shared across analytics modules.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from constants import (
    LOW_ECCENTRICITY_THRESHOLD,
    MU_EARTH,
    PHASE_SEMANTICS_TRUE_ANOMALY_PROXY,
    PHASE_VARIABLE_TRUE_ANOMALY,
    RADIUS_EARTH,
)

TWO_PI = 2.0 * np.pi

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    njit = None
    _HAS_NUMBA = False

_NUMBA_DISABLED_ENV = {"0", "false", "no", "off"}
_USE_NUMBA = _HAS_NUMBA and str(os.getenv("ORBITAL_FEATURES_USE_NUMBA", "1")).strip().lower() not in _NUMBA_DISABLED_ENV


def _semi_major_axis_vector_numpy(mean_motion_array: np.ndarray) -> np.ndarray:
    mean_motion_rad = mean_motion_array * 2.0 * np.pi / 86400.0
    return (MU_EARTH ** (1.0 / 3.0)) / (mean_motion_rad ** (2.0 / 3.0))


def _specific_angular_momentum_vector_numpy(a_array: np.ndarray, e_array: np.ndarray) -> np.ndarray:
    return np.sqrt(MU_EARTH * a_array * (1.0 - e_array**2))


def _mean_to_true_anomaly_vector_numpy(M_array: np.ndarray, e_array: np.ndarray) -> np.ndarray:
    epsilon = 1e-8
    max_iter = 12
    E = np.where(M_array < np.pi, M_array + 0.5 * e_array, M_array - 0.5 * e_array)

    for _ in range(max_iter):
        residual = E - e_array * np.sin(E) - M_array
        active = np.abs(residual) > epsilon
        if not np.any(active):
            break
        denom = 1.0 - e_array[active] * np.cos(E[active])
        E[active] -= residual[active] / denom

    nu = 2.0 * np.arctan(np.sqrt((1.0 + e_array) / (1.0 - e_array)) * np.tan(E / 2.0))
    return np.degrees(np.mod(nu, 2.0 * np.pi))


if _HAS_NUMBA:

    @njit(cache=True)
    def _semi_major_axis_vector_numba(mean_motion_array):
        out = np.empty(mean_motion_array.size, dtype=np.float64)
        mu_term = MU_EARTH ** (1.0 / 3.0)
        two_pi_over_day = 2.0 * np.pi / 86400.0
        for i in range(mean_motion_array.size):
            mean_motion_rad = mean_motion_array[i] * two_pi_over_day
            out[i] = mu_term / (mean_motion_rad ** (2.0 / 3.0))
        return out


    @njit(cache=True)
    def _specific_angular_momentum_vector_numba(a_array, e_array):
        out = np.empty(a_array.size, dtype=np.float64)
        for i in range(a_array.size):
            out[i] = np.sqrt(MU_EARTH * a_array[i] * (1.0 - e_array[i] * e_array[i]))
        return out


    @njit(cache=True)
    def _mean_to_true_anomaly_vector_numba(M_array, e_array, epsilon, max_iter):
        out = np.empty(M_array.size, dtype=np.float64)
        for i in range(M_array.size):
            m_val = M_array[i]
            e_val = e_array[i]

            if not np.isfinite(m_val) or not np.isfinite(e_val):
                out[i] = np.nan
                continue

            if e_val < 0.0 or e_val >= 1.0:
                out[i] = np.nan
                continue

            if m_val < np.pi:
                E = m_val + 0.5 * e_val
            else:
                E = m_val - 0.5 * e_val

            for _ in range(max_iter):
                residual = E - e_val * np.sin(E) - m_val
                if abs(residual) <= epsilon:
                    break
                denom = 1.0 - e_val * np.cos(E)
                if denom == 0.0:
                    break
                E -= residual / denom

            sqrt_arg = (1.0 + e_val) / (1.0 - e_val)
            if sqrt_arg <= 0.0 or not np.isfinite(sqrt_arg):
                out[i] = np.nan
                continue

            nu = 2.0 * np.arctan(np.sqrt(sqrt_arg) * np.tan(E / 2.0))
            out[i] = np.degrees(np.mod(nu, 2.0 * np.pi))

        return out


def resolve_object_col(df: pd.DataFrame) -> str:
    """Resolve canonical object id column from common alternatives."""
    for col in ("norad_cat_id", "sat_id"):
        if col in df.columns:
            return col
    return "object_id"


def ensure_altitude(df: pd.DataFrame, sma_col: str = "sma", earth_radius_km: float = RADIUS_EARTH) -> pd.DataFrame:
    """Return a copy that contains altitude_km derived from semi-major axis."""
    out = df.copy()
    if "altitude_km" not in out.columns:
        out["altitude_km"] = pd.to_numeric(out[sma_col], errors="coerce") - float(earth_radius_km)
    return out


def semi_major_axis(mean_motion: float) -> float:
    """Calculate semi-major axis (km) from mean motion (rev/day)."""
    mean_motion_rad = mean_motion * 2.0 * np.pi / 86400.0
    return (MU_EARTH ** (1.0 / 3.0)) / (mean_motion_rad ** (2.0 / 3.0))


def semi_major_axis_vector(mean_motion_array: np.ndarray) -> np.ndarray:
    """Vectorized semi-major axis (km) from mean motion array (rev/day)."""
    mm = np.asarray(mean_motion_array, dtype=np.float64)
    if not _USE_NUMBA:
        return _semi_major_axis_vector_numpy(mm)

    if mm.ndim != 1:
        return _semi_major_axis_vector_numpy(mm)

    if not np.all(np.isfinite(mm)):
        return _semi_major_axis_vector_numpy(mm)

    if np.any(mm <= 0.0):
        return _semi_major_axis_vector_numpy(mm)

    return _semi_major_axis_vector_numba(mm)


def specific_angular_momentum(a: float, e: float) -> float:
    """Calculate specific angular momentum (km^2/s) from a and e."""
    return np.sqrt(MU_EARTH * a * (1.0 - e**2))


def specific_angular_momentum_vector(a_array: np.ndarray, e_array: np.ndarray) -> np.ndarray:
    """Vectorized specific angular momentum (km^2/s)."""
    a = np.asarray(a_array, dtype=np.float64)
    e = np.asarray(e_array, dtype=np.float64)
    if not _USE_NUMBA:
        return _specific_angular_momentum_vector_numpy(a, e)

    if a.ndim != 1 or e.ndim != 1:
        return _specific_angular_momentum_vector_numpy(a, e)

    if a.shape != e.shape:
        return _specific_angular_momentum_vector_numpy(a, e)

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(e)):
        return _specific_angular_momentum_vector_numpy(a, e)

    return _specific_angular_momentum_vector_numba(a, e)


def mean_to_true_anomaly(M: float, e: float) -> float:
    """Convert mean anomaly (rad) and eccentricity to true anomaly (deg)."""
    epsilon = 1e-8
    E = M + 0.5 * e if M < np.pi else M - 0.5 * e
    ratio = 1.0

    while abs(ratio) > epsilon:
        ratio = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= ratio

    true_anomaly = 2.0 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E / 2.0))
    return np.degrees(np.mod(true_anomaly, 2.0 * np.pi))


def mean_to_true_anomaly_vector(M_array: np.ndarray, e_array: np.ndarray) -> np.ndarray:
    """Vectorized conversion from mean anomaly (rad) to true anomaly (deg)."""
    M_array = np.asarray(M_array, dtype=np.float64)
    e_array = np.asarray(e_array, dtype=np.float64)

    if not _USE_NUMBA:
        return _mean_to_true_anomaly_vector_numpy(M_array, e_array)

    if M_array.ndim != 1 or e_array.ndim != 1:
        return _mean_to_true_anomaly_vector_numpy(M_array, e_array)

    if M_array.shape != e_array.shape:
        return _mean_to_true_anomaly_vector_numpy(M_array, e_array)

    if not np.all(np.isfinite(M_array)) or not np.all(np.isfinite(e_array)):
        return _mean_to_true_anomaly_vector_numpy(M_array, e_array)

    if np.any(e_array < 0.0) or np.any(e_array >= 1.0):
        return _mean_to_true_anomaly_vector_numpy(M_array, e_array)

    return _mean_to_true_anomaly_vector_numba(M_array, e_array, 1e-8, 12)


def wrap_deg_360(angle_deg):
    arr = np.asarray(angle_deg, dtype=np.float64)
    return np.mod(arr, 360.0)


def wrap_deg_pm180(angle_deg):
    arr = np.asarray(angle_deg, dtype=np.float64)
    wrapped = ((arr + 180.0) % 360.0) - 180.0
    wrapped[wrapped == -180.0] = 180.0
    return wrapped


def wrap_rad_2pi(angle_rad):
    arr = np.asarray(angle_rad, dtype=np.float64)
    return np.mod(arr, TWO_PI)


def wrap_rad_pmpi(angle_rad):
    arr = np.asarray(angle_rad, dtype=np.float64)
    wrapped = ((arr + np.pi) % TWO_PI) - np.pi
    wrapped[wrapped == -np.pi] = np.pi
    return wrapped


def unwrap_deg(angle_deg):
    arr = np.asarray(angle_deg, dtype=np.float64)
    return np.rad2deg(np.unwrap(np.deg2rad(arr)))


def unwrap_rad(angle_rad):
    arr = np.asarray(angle_rad, dtype=np.float64)
    return np.unwrap(arr)


def add_orbital_phase_features(df, include_radians=False, include_unwrapped=False):
    """Add low-e-safe angular phase constructs.

    Scientific semantics:
        Inputs sourced from TLE/GP products are SGP4-compatible mean-element
        quantities. Angle combinations derived here are analysis diagnostics.
        They should be treated as descriptive/proxy features unless a dedicated
        mean-to-osculating reconstruction pipeline is used.

    Added primary degree columns:
        - argument_of_latitude_deg = aop + true_anomaly
        - mean_longitude_deg = raan + aop + mean_anomaly
        - longitude_of_perigee_deg = raan + aop
        - mean_argument_of_latitude_deg = aop + mean_anomaly
    """
    out = df.copy()
    if "true_anomaly" in out.columns:
        true_anomaly_numeric = pd.to_numeric(out["true_anomaly"], errors="coerce")
        if "true_anomaly_kepler_proxy_deg" not in out.columns:
            out["true_anomaly_kepler_proxy_deg"] = true_anomaly_numeric
        else:
            alias_numeric = pd.to_numeric(out["true_anomaly_kepler_proxy_deg"], errors="coerce")
            out["true_anomaly_kepler_proxy_deg"] = alias_numeric.where(alias_numeric.notna(), true_anomaly_numeric)

    argument_of_latitude_deg = np.asarray(out["aop"], dtype=np.float64) + np.asarray(out["true_anomaly"], dtype=np.float64)
    mean_longitude_deg = (
        np.asarray(out["raan"], dtype=np.float64)
        + np.asarray(out["aop"], dtype=np.float64)
        + np.asarray(out["mean_anomaly"], dtype=np.float64)
    )
    longitude_of_perigee_deg = np.asarray(out["raan"], dtype=np.float64) + np.asarray(out["aop"], dtype=np.float64)
    mean_argument_of_latitude_deg = np.asarray(out["aop"], dtype=np.float64) + np.asarray(out["mean_anomaly"], dtype=np.float64)

    out["argument_of_latitude_deg"] = argument_of_latitude_deg
    out["mean_longitude_deg"] = mean_longitude_deg
    out["longitude_of_perigee_deg"] = longitude_of_perigee_deg
    out["mean_argument_of_latitude_deg"] = mean_argument_of_latitude_deg

    for base in [
        "argument_of_latitude_deg",
        "mean_longitude_deg",
        "longitude_of_perigee_deg",
        "mean_argument_of_latitude_deg",
    ]:
        out[f"{base}_wrapped_360"] = wrap_deg_360(out[base])
        out[f"{base}_wrapped_pm180"] = wrap_deg_pm180(out[base])
        if include_unwrapped:
            out[f"{base}_unwrapped"] = unwrap_deg(out[base])

    if include_radians:
        for base in [
            "argument_of_latitude_deg",
            "mean_longitude_deg",
            "longitude_of_perigee_deg",
            "mean_argument_of_latitude_deg",
        ]:
            rad_col = base.replace("_deg", "_rad")
            out[rad_col] = np.deg2rad(np.asarray(out[base], dtype=np.float64))
            out[f"{rad_col}_wrapped_2pi"] = wrap_rad_2pi(out[rad_col])
            out[f"{rad_col}_wrapped_pmpi"] = wrap_rad_pmpi(out[rad_col])
            if include_unwrapped:
                out[f"{rad_col}_unwrapped"] = unwrap_rad(out[rad_col])

    return out


def add_low_eccentricity_flag(df, ecc_col="ecc", threshold=LOW_ECCENTRICITY_THRESHOLD):
    out = df.copy()
    ecc = np.asarray(out[ecc_col], dtype=np.float64)
    out["low_eccentricity"] = np.isfinite(ecc) & (ecc < float(threshold))
    return out


def recommend_phase_variable(requested_variable, low_eccentricity, low_e_choice="mean_longitude_deg"):
    return low_e_choice if bool(low_eccentricity) else requested_variable


def select_phase_series(df, requested_variable="true_anomaly", ecc_col="ecc", ecc_threshold=LOW_ECCENTRICITY_THRESHOLD, low_e_choice="mean_longitude_deg"):
    out = add_orbital_phase_features(df)
    out = add_low_eccentricity_flag(out, ecc_col=ecc_col, threshold=ecc_threshold)

    if requested_variable not in out.columns:
        raise KeyError(f"Requested phase variable '{requested_variable}' not found in DataFrame")
    if low_e_choice not in out.columns:
        raise KeyError(f"Low-e fallback phase variable '{low_e_choice}' not found in DataFrame")

    selected = np.where(out["low_eccentricity"].values, out[low_e_choice].values, out[requested_variable].values)
    out["recommended_phase_variable"] = np.where(out["low_eccentricity"].values, low_e_choice, requested_variable)
    out["selected_phase_deg"] = selected
    if "true_anomaly" in out.columns and "true_anomaly_kepler_proxy_deg" not in out.columns:
        out["true_anomaly_kepler_proxy_deg"] = pd.to_numeric(out["true_anomaly"], errors="coerce")

    out["phase_variable"] = str(requested_variable)
    if str(requested_variable) == PHASE_VARIABLE_TRUE_ANOMALY:
        out["phase_semantics"] = PHASE_SEMANTICS_TRUE_ANOMALY_PROXY
    else:
        out["phase_semantics"] = "user_selected_phase_variable"
    return out


def add_standard_tle_proxy_enrichment(
    df,
    *,
    ecc_threshold=LOW_ECCENTRICITY_THRESHOLD,
    include_radians=True,
    include_unwrapped=True,
    requested_phase_variable="true_anomaly",
    low_e_choice="mean_longitude_deg",
):
    """
    This stage is intended to run before synchronization/dispatch so low-e-safe
    phase coordinates are always available to downstream modules.
    """
    out = add_orbital_phase_features(
        df,
        include_radians=bool(include_radians),
        include_unwrapped=bool(include_unwrapped),
    )
    out = add_low_eccentricity_flag(out, threshold=ecc_threshold)
    out = select_phase_series(
        out,
        requested_variable=requested_phase_variable,
        ecc_threshold=ecc_threshold,
        low_e_choice=low_e_choice,
    )
    return out


def add_altitude_features(df, earth_radius_km=RADIUS_EARTH, sma_col="sma"):
    out = df.copy()
    out["altitude_km"] = np.asarray(out[sma_col], dtype=np.float64) - float(earth_radius_km)
    return out


def add_altitude_regime(df, bins, labels=None, altitude_col="altitude_km"):
    out = df.copy()
    if labels is None:
        labels = [f"regime_{i}" for i in range(len(bins) - 1)]
    out["altitude_regime"] = pd.cut(out[altitude_col], bins=bins, labels=labels, include_lowest=True)
    return out


def assign_candidate_shell_id(
    df,
    shell_definitions=None,
    altitude_col="altitude_km",
    *,
    inc_col="inc",
    use_inclination_refinement=False,
    inclination_tolerance_deg=2.0,
):
    out = df.copy()
    if shell_definitions is None:
        out["candidate_shell_id"] = pd.Series([pd.NA] * len(out), dtype="object")
        out["shell_assignment_basis"] = pd.Series([pd.NA] * len(out), dtype="object")
        return out

    alt = np.asarray(out[altitude_col], dtype=np.float64)
    inc = pd.to_numeric(out[inc_col], errors="coerce").to_numpy(dtype=np.float64) if inc_col in out.columns else np.full(len(out), np.nan)
    candidate = np.array([None] * len(out), dtype=object)
    basis = np.array([None] * len(out), dtype=object)

    for definition in shell_definitions:
        sid = definition.get("id")
        amin = definition.get("min_altitude_km", -np.inf)
        amax = definition.get("max_altitude_km", np.inf)
        mask = np.isfinite(alt) & (alt >= float(amin)) & (alt < float(amax))
        candidate[mask] = sid
        basis[mask] = "altitude_only"

        if bool(use_inclination_refinement):
            inc_ref = definition.get("inclination_deg", None)
            if inc_ref is not None and np.isfinite(float(inc_ref)):
                refined = mask & np.isfinite(inc) & (np.abs(inc - float(inc_ref)) <= float(inclination_tolerance_deg))
                candidate[refined] = sid
                basis[refined] = "altitude_plus_inclination"

    out["candidate_shell_id"] = pd.Series(candidate, dtype="object")
    out["shell_assignment_basis"] = pd.Series(basis, dtype="object")
    return out


def add_shell_proximity_metrics(df, shell_definitions, altitude_col="altitude_km"):
    out = df.copy()
    alt = np.asarray(out[altitude_col], dtype=np.float64)

    distance_matrix = []
    ids = []
    for definition in shell_definitions:
        sid = definition.get("id")
        ref_alt = definition.get("altitude_km")
        if sid is None or ref_alt is None:
            continue
        ids.append(str(sid))
        distance_matrix.append(np.abs(alt - float(ref_alt)))

    if len(distance_matrix) == 0:
        out["nearest_shell_id"] = pd.Series([pd.NA] * len(out), dtype="object")
        out["nearest_shell_distance_km"] = np.nan
        return out

    distances = np.vstack(distance_matrix)
    best_idx = np.argmin(distances, axis=0)
    best_dist = distances[best_idx, np.arange(distances.shape[1])]

    nearest_ids = np.array([ids[i] for i in best_idx], dtype=object)
    out["nearest_shell_id"] = pd.Series(nearest_ids, dtype="object")
    out["nearest_shell_distance_km"] = best_dist

    for i, sid in enumerate(ids):
        out[f"shell_distance_km_{sid}"] = distances[i, :]

    return out
