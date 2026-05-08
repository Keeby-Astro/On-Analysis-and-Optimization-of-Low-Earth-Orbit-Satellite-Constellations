"""Resonance and secular commensurability diagnostics.

This module separates resonance screening from resonant-angle behavior checks.
Residual proximity is not treated as proof of resonant capture.

Convention note for tesseral repeat-ground-track screening:
- ``n`` is orbital mean motion in inertial rad/day from ``sqrt(mu/a^3)``.
- ``theta_dot_earth`` is sidereal Earth rotation rate in rad/day.
- For repeat ratio ``m:l`` (m orbits per l sidereal days), residuals use
    ``R = l*n - m*theta_dot_earth`` and angle proxies use
    ``psi = l*lambda - m*theta_earth`` (or ``l*M - m*theta_earth`` when only
    mean anomaly is available).

References for family definitions and interpretation:
- Alessi et al. (LEO SRP resonances): MNRAS 473(2), 2407.
- Rosengren et al. (MEO lunisolar resonance structure): arXiv:1503.02581.
- Celletti & Gales (inclination-only commensurability 2*dot(omega)+dot(Omega)=0).
- Ely & Howell (tesseral/ground-track commensurability foundations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from orbital_features import resolve_object_col
from secular_perturbations import compute_secular_rates_from_elements

MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.14
J2 = 1.082635854e-3
SECONDS_PER_DAY = 86400.0
SOLAR_MOTION_RAD_DAY = 2.0 * math.pi / 365.2422
EARTH_ROTATION_RAD_DAY = 2.0 * math.pi * 1.00273790935
LUNAR_NODE_REGRESSION_RAD_DAY = -(2.0 * math.pi) / (18.6 * 365.2422)
LOW_ECCENTRICITY_DEFAULT = 1.0e-3


@dataclass(frozen=True)
class ResonanceDefinition:
    """Canonical resonance definition payload.

    coefficients apply to frequency keys among:
    dOmega_dt, domega_dt, lambda_dot_sun, Omega_dot_moon, theta_dot_earth, n.
    """

    name: str
    family: str
    angle_expression_text: str
    frequency_expression_text: str
    coefficients: Dict[str, float]
    angle_coefficients: Dict[str, float]
    required_external_frequencies: Tuple[str, ...]
    required_external_angles: Tuple[str, ...]
    validity_notes: str
    literature_tag: str


SRP_SOLAR_LEO_RESONANCES = [
    {
        "name": "srp_omega_minus_lambda_sun",
        "family": "SRP_SOLAR_LEO",
        "angle_expression_text": "psi = omega - lambda_sun",
        "frequency_expression_text": "R = domega_dt - lambda_dot_sun",
        "coefficients": {"domega_dt": 1.0, "lambda_dot_sun": -1.0},
        "angle_coefficients": {"omega": 1.0, "lambda_sun": -1.0},
        "required_external_frequencies": ["lambda_dot_sun"],
        "required_external_angles": ["lambda_sun"],
        "validity_notes": "Apsidal semantics degrade near circularity; use low-e safeguards.",
        "literature_tag": "Alessi2018",
    },
    {
        "name": "srp_omega_plus_lambda_sun",
        "family": "SRP_SOLAR_LEO",
        "angle_expression_text": "psi = omega + lambda_sun",
        "frequency_expression_text": "R = domega_dt + lambda_dot_sun",
        "coefficients": {"domega_dt": 1.0, "lambda_dot_sun": 1.0},
        "angle_coefficients": {"omega": 1.0, "lambda_sun": 1.0},
        "required_external_frequencies": ["lambda_dot_sun"],
        "required_external_angles": ["lambda_sun"],
        "validity_notes": "Apsidal semantics degrade near circularity; use low-e safeguards.",
        "literature_tag": "Alessi2018",
    },
    {
        "name": "srp_Omega_plus_omega_minus_lambda_sun",
        "family": "SRP_SOLAR_LEO",
        "angle_expression_text": "psi = Omega + omega - lambda_sun",
        "frequency_expression_text": "R = dOmega_dt + domega_dt - lambda_dot_sun",
        "coefficients": {"dOmega_dt": 1.0, "domega_dt": 1.0, "lambda_dot_sun": -1.0},
        "angle_coefficients": {"Omega": 1.0, "omega": 1.0, "lambda_sun": -1.0},
        "required_external_frequencies": ["lambda_dot_sun"],
        "required_external_angles": ["lambda_sun"],
        "validity_notes": "Canonical first-order SRP commensurability proxy for LEO screening.",
        "literature_tag": "Alessi2018",
    },
    {
        "name": "srp_omega_minus_Omega_minus_lambda_sun",
        "family": "SRP_SOLAR_LEO",
        "angle_expression_text": "psi = omega - Omega - lambda_sun",
        "frequency_expression_text": "R = domega_dt - dOmega_dt - lambda_dot_sun",
        "coefficients": {"dOmega_dt": -1.0, "domega_dt": 1.0, "lambda_dot_sun": -1.0},
        "angle_coefficients": {"Omega": -1.0, "omega": 1.0, "lambda_sun": -1.0},
        "required_external_frequencies": ["lambda_dot_sun"],
        "required_external_angles": ["lambda_sun"],
        "validity_notes": "Canonical first-order SRP commensurability proxy for LEO screening.",
        "literature_tag": "Alessi2018",
    },
    {
        "name": "srp_twoomega_plus_Omega_minus_lambda_sun",
        "family": "SRP_SOLAR_LEO",
        "angle_expression_text": "psi = 2*omega + Omega - lambda_sun",
        "frequency_expression_text": "R = 2*domega_dt + dOmega_dt - lambda_dot_sun",
        "coefficients": {"dOmega_dt": 1.0, "domega_dt": 2.0, "lambda_dot_sun": -1.0},
        "angle_coefficients": {"Omega": 1.0, "omega": 2.0, "lambda_sun": -1.0},
        "required_external_frequencies": ["lambda_dot_sun"],
        "required_external_angles": ["lambda_sun"],
        "validity_notes": "Also tracked separately as inclination-only-style commensurability when solar term is omitted.",
        "literature_tag": "Alessi2018",
    },
]

LUNISOLAR_SECULAR_MEO_RESONANCES = [
    {
        "name": "meo_twoomega_plus_Omega_minus_Omega_moon",
        "family": "LUNISOLAR_SECULAR_MEO",
        "angle_expression_text": "psi = 2*omega + Omega - Omega_moon",
        "frequency_expression_text": "R = 2*domega_dt + dOmega_dt - Omega_dot_moon",
        "coefficients": {"domega_dt": 2.0, "dOmega_dt": 1.0, "Omega_dot_moon": -1.0},
        "angle_coefficients": {"omega": 2.0, "Omega": 1.0, "Omega_moon": -1.0},
        "required_external_frequencies": ["Omega_dot_moon"],
        "required_external_angles": ["Omega_moon"],
        "validity_notes": "MEO lunisolar resonance family screening with lunar-node regression term.",
        "literature_tag": "Rosengren2015",
    },
    {
        "name": "meo_omega_plus_Omega_minus_Omega_moon",
        "family": "LUNISOLAR_SECULAR_MEO",
        "angle_expression_text": "psi = omega + Omega - Omega_moon",
        "frequency_expression_text": "R = domega_dt + dOmega_dt - Omega_dot_moon",
        "coefficients": {"domega_dt": 1.0, "dOmega_dt": 1.0, "Omega_dot_moon": -1.0},
        "angle_coefficients": {"omega": 1.0, "Omega": 1.0, "Omega_moon": -1.0},
        "required_external_frequencies": ["Omega_dot_moon"],
        "required_external_angles": ["Omega_moon"],
        "validity_notes": "Low-order lunisolar secular commensurability proxy.",
        "literature_tag": "Rosengren2015",
    },
]

INCLINATION_ONLY_SECULAR_RESONANCES = [
    {
        "name": "inc_two_domega_plus_dOmega_zero",
        "family": "INCLINATION_ONLY_SECULAR",
        "angle_expression_text": "psi = 2*omega + Omega",
        "frequency_expression_text": "R = 2*domega_dt + dOmega_dt",
        "coefficients": {"domega_dt": 2.0, "dOmega_dt": 1.0},
        "angle_coefficients": {"omega": 2.0, "Omega": 1.0},
        "required_external_frequencies": [],
        "required_external_angles": [],
        "validity_notes": "Classical inclination-only secular resonance; apsidal caution at low e.",
        "literature_tag": "CellettiGales",
    },
    {
        "name": "inc_domega_zero",
        "family": "INCLINATION_ONLY_SECULAR",
        "angle_expression_text": "psi = omega",
        "frequency_expression_text": "R = domega_dt",
        "coefficients": {"domega_dt": 1.0},
        "angle_coefficients": {"omega": 1.0},
        "required_external_frequencies": [],
        "required_external_angles": [],
        "validity_notes": "Apsidal freezing proxy; treat low-e cases cautiously.",
        "literature_tag": "CellettiGales",
    },
]

TESSERAL_GROUNDTRACK_RESONANCES = [
    {
        "name": "tesseral_14_1",
        "family": "TESSERAL_GROUNDTRACK",
        "angle_expression_text": "psi = 1*M - 14*theta_earth",
        "frequency_expression_text": "R = 1*n - 14*theta_dot_earth",
        "coefficients": {"n": 1.0, "theta_dot_earth": -14.0},
        "angle_coefficients": {"M": 1.0, "theta_earth": -14.0},
        "required_external_frequencies": ["theta_dot_earth"],
        "required_external_angles": ["theta_earth", "M"],
        "validity_notes": "Ground-track commensurability screening; full tesseral arguments need richer geopotential modeling.",
        "literature_tag": "ElyHowell1997",
    },
    {
        "name": "tesseral_15_1",
        "family": "TESSERAL_GROUNDTRACK",
        "angle_expression_text": "psi = 1*M - 15*theta_earth",
        "frequency_expression_text": "R = 1*n - 15*theta_dot_earth",
        "coefficients": {"n": 1.0, "theta_dot_earth": -15.0},
        "angle_coefficients": {"M": 1.0, "theta_earth": -15.0},
        "required_external_frequencies": ["theta_dot_earth"],
        "required_external_angles": ["theta_earth", "M"],
        "validity_notes": "Ground-track commensurability screening; full tesseral arguments need richer geopotential modeling.",
        "literature_tag": "ElyHowell1997",
    },
]

USER_DEFINED_RESONANCES: List[dict] = []

DEFAULT_SRP_RESONANCES = [
    {"name": "domega_minus_nsun", "k1": 0, "k2": 1, "k3": -1},
    {"name": "domega_plus_nsun", "k1": 0, "k2": 1, "k3": 1},
    {"name": "domega_plus_dOmega_minus_nsun", "k1": 1, "k2": 1, "k3": -1},
    {"name": "domega_minus_dOmega_minus_nsun", "k1": -1, "k2": 1, "k3": -1},
    {"name": "two_domega_plus_dOmega_minus_nsun", "k1": 1, "k2": 2, "k3": -1},
]


def _safe_numeric(df: pd.DataFrame, col: str, fallback: float = np.nan) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    return np.full(len(df), fallback, dtype=np.float64)


def _wrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _true_to_mean_anomaly_deg(true_anomaly_deg: np.ndarray, ecc: np.ndarray) -> np.ndarray:
    f = np.deg2rad(true_anomaly_deg)
    e = np.clip(ecc, 0.0, 0.999999)
    sin_E = np.sqrt(1.0 - e * e) * np.sin(f) / (1.0 + e * np.cos(f))
    cos_E = (e + np.cos(f)) / (1.0 + e * np.cos(f))
    E = np.arctan2(sin_E, cos_E)
    M = E - e * np.sin(E)
    return _wrap_deg(np.rad2deg(M))


def _compute_mean_anomaly_deg(
    df: pd.DataFrame,
    mean_anomaly_col: str = "mean_anomaly",
    true_anomaly_col: str = "true_anomaly",
    ecc_col: str = "ecc",
) -> Tuple[np.ndarray, str]:
    if mean_anomaly_col in df.columns:
        return _safe_numeric(df, mean_anomaly_col), "provided"

    if true_anomaly_col in df.columns and ecc_col in df.columns:
        nu = _safe_numeric(df, true_anomaly_col)
        ecc = _safe_numeric(df, ecc_col)
        valid = np.isfinite(nu) & np.isfinite(ecc)
        out = np.full(len(df), np.nan, dtype=np.float64)
        if np.any(valid):
            out[valid] = _true_to_mean_anomaly_deg(nu[valid], ecc[valid])
        return out, "derived_from_true_anomaly"

    return np.full(len(df), np.nan, dtype=np.float64), "unavailable"


def build_lunar_node_rate_provider(
    ephemeris_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    omega_moon_col: str = "Omega_moon_deg",
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Build an interpolated lunar-node regression rate provider.

    The provider expects the target DataFrame to contain ``timestamp_col``.
    """
    eph = ephemeris_df[[timestamp_col, omega_moon_col]].copy()
    eph[timestamp_col] = pd.to_datetime(eph[timestamp_col], errors="coerce", utc=True)
    eph = eph.dropna(subset=[timestamp_col]).sort_values(timestamp_col, kind="mergesort")
    if eph.empty:
        raise ValueError("ephemeris_df has no valid timestamps")

    t_sec = (eph[timestamp_col] - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype=np.float64)
    omega_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(_safe_numeric(eph, omega_moon_col))))
    rate_rad_day = np.gradient(np.deg2rad(omega_unwrapped), t_sec, edge_order=1) * SECONDS_PER_DAY

    def _provider(df: pd.DataFrame) -> np.ndarray:
        t = pd.to_datetime(df.get(timestamp_col), errors="coerce", utc=True)
        if t is None:
            return np.full(len(df), np.nan, dtype=np.float64)
        t_q = (t - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype=np.float64)
        out = np.interp(t_q, t_sec, rate_rad_day, left=np.nan, right=np.nan)
        return out.astype(np.float64)

    return _provider


def build_lunar_node_angle_provider(
    ephemeris_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    omega_moon_col: str = "Omega_moon_deg",
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Build an interpolated lunar-node angle provider (degrees)."""
    eph = ephemeris_df[[timestamp_col, omega_moon_col]].copy()
    eph[timestamp_col] = pd.to_datetime(eph[timestamp_col], errors="coerce", utc=True)
    eph = eph.dropna(subset=[timestamp_col]).sort_values(timestamp_col, kind="mergesort")
    if eph.empty:
        raise ValueError("ephemeris_df has no valid timestamps")

    t_sec = (eph[timestamp_col] - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype=np.float64)
    omega_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(_safe_numeric(eph, omega_moon_col))))

    def _provider(df: pd.DataFrame) -> np.ndarray:
        t = pd.to_datetime(df.get(timestamp_col), errors="coerce", utc=True)
        if t is None:
            return np.full(len(df), np.nan, dtype=np.float64)
        t_q = (t - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype=np.float64)
        out = np.interp(t_q, t_sec, omega_unwrapped, left=np.nan, right=np.nan)
        return _wrap_deg(out.astype(np.float64))

    return _provider


def _normalize_resonance_definition(definition: dict) -> ResonanceDefinition:
    name = str(definition.get("name", "unnamed"))
    family = str(definition.get("family", "USER_DEFINED"))

    coeffs = dict(definition.get("coefficients", {}))
    if not coeffs and any(k in definition for k in ("k1", "k2", "k3")):
        coeffs = {
            "dOmega_dt": float(definition.get("k1", 0.0)),
            "domega_dt": float(definition.get("k2", 0.0)),
            "lambda_dot_sun": float(definition.get("k3", 0.0)),
        }

    if not coeffs:
        raise ValueError(f"Resonance definition '{name}' has no coefficients.")

    if all(abs(float(v)) == 0.0 for v in coeffs.values()):
        raise ValueError(f"Resonance definition '{name}' has degenerate zero coefficients.")

    angle_coeffs = dict(definition.get("angle_coefficients", {}))
    required_external_frequencies = tuple(definition.get("required_external_frequencies", []))
    required_external_angles = tuple(definition.get("required_external_angles", []))

    return ResonanceDefinition(
        name=name,
        family=family,
        angle_expression_text=str(definition.get("angle_expression_text", "psi = undefined")),
        frequency_expression_text=str(definition.get("frequency_expression_text", "R = undefined")),
        coefficients={k: float(v) for k, v in coeffs.items()},
        angle_coefficients={k: float(v) for k, v in angle_coeffs.items()},
        required_external_frequencies=required_external_frequencies,
        required_external_angles=required_external_angles,
        validity_notes=str(definition.get("validity_notes", "")),
        literature_tag=str(definition.get("literature_tag", "unspecified")),
    )


def get_resonance_registry(
    families: Optional[Sequence[str]] = None,
    include_user_defined: bool = True,
    user_defined_resonances: Optional[Iterable[dict]] = None,
) -> List[dict]:
    """Return normalized resonance registry definitions.

    Families are explicit and separated by dynamical type.
    """
    registries = {
        "SRP_SOLAR_LEO": SRP_SOLAR_LEO_RESONANCES,
        "LUNISOLAR_SECULAR_MEO": LUNISOLAR_SECULAR_MEO_RESONANCES,
        "INCLINATION_ONLY_SECULAR": INCLINATION_ONLY_SECULAR_RESONANCES,
        "TESSERAL_GROUNDTRACK": TESSERAL_GROUNDTRACK_RESONANCES,
    }

    selected = set(families) if families is not None else set(registries.keys())
    out: List[dict] = []
    for family_key in ("SRP_SOLAR_LEO", "LUNISOLAR_SECULAR_MEO", "INCLINATION_ONLY_SECULAR", "TESSERAL_GROUNDTRACK"):
        if family_key not in selected:
            continue
        for definition in registries[family_key]:
            out.append(_normalize_resonance_definition(definition).__dict__.copy())

    if include_user_defined:
        merged_user = list(USER_DEFINED_RESONANCES)
        if user_defined_resonances is not None:
            merged_user.extend(list(user_defined_resonances))
        for definition in merged_user:
            out.append(_normalize_resonance_definition(definition).__dict__.copy())

    return out

def compute_secular_rates(df: pd.DataFrame, sma_col: str = "sma", ecc_col: str = "ecc", inc_col: str = "inc",
                          mu: float = MU_EARTH_KM3_S2, r_earth_km: float = R_EARTH_KM,
                          j2: float = J2, include_n_sun: bool = True) -> pd.DataFrame:
    """Compute J2 secular precession rates.

    Returns a copy with:
    - dOmega_dt_rad_s, dOmega_dt_rad_day
    - domega_dt_rad_s, domega_dt_rad_day
    - n_sun_rad_day (optional)
    """
    # Use a shallow copy to avoid block consolidation spikes on very large panels.
    out = df.copy(deep=False)

    if "node_precession_rate" in out.columns and "perigee_precession_rate" in out.columns:
        dOmega = pd.to_numeric(out["node_precession_rate"], errors="coerce").to_numpy(dtype=np.float64)
        domega = pd.to_numeric(out["perigee_precession_rate"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        dOmega, domega = compute_secular_rates_from_elements(
            pd.to_numeric(out[sma_col], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(out[ecc_col], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(out[inc_col], errors="coerce").to_numpy(dtype=np.float64),
            mu_km3_s2=mu,
            reference_radius_km=r_earth_km,
            j2=j2,
            model="J2_first_order",
            element_semantics="diagnostic_proxy",
            output_units="rad_s",
            return_metadata=False,
        )

    out["dOmega_dt_rad_s"] = dOmega
    out["domega_dt_rad_s"] = domega
    out["dOmega_dt_rad_day"] = dOmega * SECONDS_PER_DAY
    out["domega_dt_rad_day"] = domega * SECONDS_PER_DAY
    if include_n_sun:
        out["n_sun_rad_day"] = SOLAR_MOTION_RAD_DAY

    return out


def compute_resonance_frequencies(
    df: pd.DataFrame,
    sma_col: str = "sma",
    ecc_col: str = "ecc",
    inc_col: str = "inc",
    timestamp_col: str = "timestamp",
    lambda_dot_sun_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    omega_dot_moon_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    theta_dot_earth_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    return_metadata: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, dict]:
    """Compute canonical frequency terms used by resonance families."""
    # Avoid deep copies here; this stage only appends derived columns.
    out = df.copy(deep=False)
    has_precomputed_rates = all(col in out.columns for col in ("dOmega_dt_rad_day", "domega_dt_rad_day"))
    if not has_precomputed_rates:
        out = compute_secular_rates(out, sma_col=sma_col, ecc_col=ecc_col, inc_col=inc_col, include_n_sun=True)
    else:
        out["dOmega_dt_rad_day"] = pd.to_numeric(out["dOmega_dt_rad_day"], errors="coerce")
        out["domega_dt_rad_day"] = pd.to_numeric(out["domega_dt_rad_day"], errors="coerce")
        if "dOmega_dt_rad_s" not in out.columns:
            out["dOmega_dt_rad_s"] = out["dOmega_dt_rad_day"] / SECONDS_PER_DAY
        if "domega_dt_rad_s" not in out.columns:
            out["domega_dt_rad_s"] = out["domega_dt_rad_day"] / SECONDS_PER_DAY
        if "n_sun_rad_day" not in out.columns:
            out["n_sun_rad_day"] = SOLAR_MOTION_RAD_DAY

    sma = _safe_numeric(out, sma_col)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_rad_s = np.sqrt(MU_EARTH_KM3_S2 / np.power(sma, 3.0))
    out["n_rad_day"] = n_rad_s * SECONDS_PER_DAY

    metadata = {
        "lambda_dot_sun_approximate": lambda_dot_sun_provider is None,
        "omega_dot_moon_approximate": omega_dot_moon_provider is None,
        "theta_dot_earth_approximate": theta_dot_earth_provider is None,
        "external_frequency_notes": [],
    }

    if lambda_dot_sun_provider is not None:
        out["lambda_dot_sun_rad_day"] = np.asarray(lambda_dot_sun_provider(out), dtype=np.float64)
    else:
        out["lambda_dot_sun_rad_day"] = SOLAR_MOTION_RAD_DAY
        metadata["external_frequency_notes"].append("lambda_dot_sun uses constant solar mean motion approximation")

    if omega_dot_moon_provider is not None:
        out["Omega_dot_moon_rad_day"] = np.asarray(omega_dot_moon_provider(out), dtype=np.float64)
    else:
        out["Omega_dot_moon_rad_day"] = LUNAR_NODE_REGRESSION_RAD_DAY
        metadata["external_frequency_notes"].append("Omega_dot_moon uses constant lunar-node regression approximation")

    if theta_dot_earth_provider is not None:
        out["theta_dot_earth_rad_day"] = np.asarray(theta_dot_earth_provider(out), dtype=np.float64)
    else:
        out["theta_dot_earth_rad_day"] = EARTH_ROTATION_RAD_DAY
        metadata["external_frequency_notes"].append("theta_dot_earth uses constant Earth rotation rate")

    if return_metadata:
        return out, metadata
    return out


def _approximate_external_angles_from_time(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    # Avoid a deep copy for large dataframes; only derived columns are appended.
    out = df.copy(deep=False)
    if timestamp_col not in out.columns:
        out["lambda_sun_deg"] = np.full(len(out), np.nan, dtype=np.float64)
        out["Omega_moon_deg"] = np.full(len(out), np.nan, dtype=np.float64)
        out["theta_earth_deg"] = np.full(len(out), np.nan, dtype=np.float64)
        return out

    t = pd.to_datetime(out.get(timestamp_col), errors="coerce", utc=True)
    if t is None:
        out["lambda_sun_deg"] = np.full(len(out), np.nan, dtype=np.float64)
        out["Omega_moon_deg"] = np.full(len(out), np.nan, dtype=np.float64)
        out["theta_earth_deg"] = np.full(len(out), np.nan, dtype=np.float64)
        return out

    # Approximate angles for screening-grade diagnostics only.
    days = (t - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds() / SECONDS_PER_DAY
    out["lambda_sun_deg"] = _wrap_deg((280.460 + 0.9856474 * days).to_numpy(dtype=np.float64))
    out["Omega_moon_deg"] = _wrap_deg((125.045 - 0.0529538 * days).to_numpy(dtype=np.float64))
    out["theta_earth_deg"] = _wrap_deg((280.46061837 + 360.98564736629 * days).to_numpy(dtype=np.float64))
    return out


def compute_resonant_angles(
    df: pd.DataFrame,
    resonance_definitions: Iterable[dict] | None = None,
    timestamp_col: str = "timestamp",
    omega_col: str = "aop",
    Omega_col: str = "raan",
    mean_anomaly_col: str = "mean_anomaly",
    true_anomaly_col: str = "true_anomaly",
    ecc_col: str = "ecc",
    lambda_sun_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    omega_moon_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    theta_earth_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    mean_anomaly_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    unwrap: bool = True,
    include_apsidal_warning_text: bool = True,
    low_e_threshold: float = LOW_ECCENTRICITY_DEFAULT,
    return_metadata: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, dict]:
    """Compute resonant angles with wrapped and optional unwrapped outputs.

    Output metadata reports when external ephemerides are approximated or absent.
    """
    out = _approximate_external_angles_from_time(df, timestamp_col=timestamp_col)
    defs = list(resonance_definitions) if resonance_definitions is not None else get_resonance_registry()
    defs_n = [_normalize_resonance_definition(d) for d in defs]

    metadata: Dict[str, dict] = {"angle_sources": {}, "warnings": []}

    omega = _safe_numeric(out, omega_col)
    Omega = _safe_numeric(out, Omega_col)
    M, m_source = _compute_mean_anomaly_deg(
        out,
        mean_anomaly_col=mean_anomaly_col,
        true_anomaly_col=true_anomaly_col,
        ecc_col=ecc_col,
    )
    if mean_anomaly_provider is not None:
        M = np.asarray(mean_anomaly_provider(out), dtype=np.float64)
        m_source = "provider"

    if lambda_sun_provider is not None:
        lam_sun = np.asarray(lambda_sun_provider(out), dtype=np.float64)
    else:
        lam_sun = _safe_numeric(out, "lambda_sun_deg")

    if omega_moon_provider is not None:
        Om_moon = np.asarray(omega_moon_provider(out), dtype=np.float64)
    else:
        Om_moon = _safe_numeric(out, "Omega_moon_deg")

    if theta_earth_provider is not None:
        theta_earth = np.asarray(theta_earth_provider(out), dtype=np.float64)
    else:
        theta_earth = _safe_numeric(out, "theta_earth_deg")

    ecc = _safe_numeric(out, ecc_col)

    source_state = {
        "lambda_sun": "provider" if lambda_sun_provider is not None else ("approximate_from_timestamp" if np.isfinite(lam_sun).any() else "unavailable"),
        "Omega_moon": "provider" if omega_moon_provider is not None else ("approximate_from_timestamp" if np.isfinite(Om_moon).any() else "unavailable"),
        "theta_earth": "provider" if theta_earth_provider is not None else ("approximate_from_timestamp" if np.isfinite(theta_earth).any() else "unavailable"),
        "M": m_source,
    }

    out["mean_anomaly_deg_for_tesseral"] = M
    out["mean_longitude_deg_for_tesseral"] = _wrap_deg(M + omega + Omega)
    out["tesseral_argument_semantics"] = (
        "mean_anomaly_provided"
        if m_source == "provided"
        else ("mean_anomaly_derived_proxy" if m_source == "derived_from_true_anomaly" else "mean_anomaly_unavailable")
    )

    for definition in defs_n:
        coeffs = definition.angle_coefficients
        if not coeffs:
            metadata["warnings"].append(f"{definition.name}: no angle coefficients provided")
            continue

        angle_deg = np.zeros(len(out), dtype=np.float64)
        angle_deg += coeffs.get("omega", 0.0) * omega
        angle_deg += coeffs.get("Omega", 0.0) * Omega
        angle_deg += coeffs.get("lambda_sun", 0.0) * lam_sun
        angle_deg += coeffs.get("Omega_moon", 0.0) * Om_moon
        angle_deg += coeffs.get("theta_earth", 0.0) * theta_earth
        angle_deg += coeffs.get("M", 0.0) * M

        wrapped = _wrap_deg(angle_deg)
        out[f"psi_{definition.name}_deg_wrapped"] = wrapped

        if unwrap:
            out[f"psi_{definition.name}_deg_unwrapped"] = np.rad2deg(np.unwrap(np.deg2rad(wrapped)))

        uses_apsidal = abs(coeffs.get("omega", 0.0)) > 0.0
        valid = np.ones(len(out), dtype=bool)
        if uses_apsidal:
            valid = np.isfinite(ecc) & (ecc > float(low_e_threshold))
        out[f"angle_validity_flag_{definition.name}"] = valid
        if include_apsidal_warning_text:
            warning = np.array([""] * len(out), dtype=object)
            if uses_apsidal:
                warning[~valid] = "apsidal angle semantics are delicate near e~0"
            out[f"apsidal_semantics_warning_{definition.name}"] = warning

        metadata["angle_sources"][definition.name] = {
            "external_angles": {key: source_state.get(key, "direct_or_missing") for key in definition.required_external_angles},
            "approximate_ephemerides": any(source_state.get(key) == "approximate_from_timestamp" for key in definition.required_external_angles),
        }

    if return_metadata:
        return out, metadata
    return out


def classify_resonant_angle_behavior(
    angle_deg_series: Sequence[float],
    min_samples: int = 24,
    libration_span_deg: float = 240.0,
) -> str:
    """Conservative circulation/libration screening for one angle time series."""
    arr = pd.to_numeric(pd.Series(angle_deg_series), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < int(min_samples):
        return "insufficient_data"

    wrapped = _wrap_deg(arr)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(wrapped)))
    span_wrapped = float(np.nanmax(wrapped) - np.nanmin(wrapped))
    drift = float(abs(unwrapped[-1] - unwrapped[0]))

    if drift > 540.0 and span_wrapped > 260.0:
        return "circulation"
    if span_wrapped <= float(libration_span_deg) and drift < 360.0:
        return "libration_candidate"
    return "transition_or_mixed"


def evaluate_resonance_proximity(df: pd.DataFrame, resonance_definitions: Iterable[dict] | None = None,
                                 tolerance_rad_day: float = 1.0e-3,
                                 dOmega_col: str = "dOmega_dt_rad_day",
                                 domega_col: str = "domega_dt_rad_day",
                                 n_sun_col: str = "n_sun_rad_day",
                                 ecc_col: str = "ecc",
                                 include_angle_diagnostics: bool = True,
                                 include_unwrapped_angles: bool = True,
                                 include_apsidal_warning_text: bool = True,
                                 low_e_threshold: float = LOW_ECCENTRICITY_DEFAULT,
                                 object_col: Optional[str] = None,
                                 timestamp_col: str = "timestamp",
                                 lambda_sun_angle_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
                                 omega_moon_angle_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
                                 theta_earth_angle_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
                                 mean_anomaly_angle_provider: Optional[Callable[[pd.DataFrame], np.ndarray]] = None) -> pd.DataFrame:
    """Evaluate proximity to secular commensurability conditions.

    The result includes per-definition residual columns and best-match columns.
    """
    out, freq_meta = compute_resonance_frequencies(df, return_metadata=True)

    defs = list(resonance_definitions) if resonance_definitions is not None else get_resonance_registry()
    defs_n = [_normalize_resonance_definition(d) for d in defs]
    if not defs:
        raise ValueError("resonance_definitions must not be empty")
    uses_apsidal_by_name = {
        d.name: bool(abs(float(d.angle_coefficients.get("omega", 0.0))) > 0.0)
        for d in defs_n
    }

    dOmega = pd.to_numeric(out[dOmega_col], errors="coerce").to_numpy(dtype=np.float64)
    domega = pd.to_numeric(out[domega_col], errors="coerce").to_numpy(dtype=np.float64)
    n_sun = _safe_numeric(out, n_sun_col, fallback=SOLAR_MOTION_RAD_DAY)
    n = _safe_numeric(out, "n_rad_day")
    theta_dot_earth = _safe_numeric(out, "theta_dot_earth_rad_day")
    Omega_dot_moon = _safe_numeric(out, "Omega_dot_moon_rad_day")

    freq_map = {
        "dOmega_dt": dOmega,
        "domega_dt": domega,
        "lambda_dot_sun": n_sun,
        "n": n,
        "theta_dot_earth": theta_dot_earth,
        "Omega_dot_moon": Omega_dot_moon,
    }

    residuals = []
    names = []
    families = []
    for definition in defs_n:
        name = definition.name
        val = np.zeros(len(out), dtype=np.float64)
        for key, coef in definition.coefficients.items():
            val = val + float(coef) * freq_map.get(key, np.full(len(out), np.nan, dtype=np.float64))
        out[f"residual_{name}_rad_day"] = val
        out[f"near_{name}"] = np.isfinite(val) & (np.abs(val) <= float(tolerance_rad_day))
        residuals.append(np.abs(val))
        names.append(name)
        families.append(definition.family)

    stack = np.vstack(residuals)
    finite_mask = np.isfinite(stack)
    finite_any = finite_mask.any(axis=0)
    stack_for_argmin = np.where(finite_mask, stack, np.inf)
    idx = np.argmin(stack_for_argmin, axis=0)
    finite_any = np.isfinite(stack).any(axis=0)

    best_name = np.full(stack.shape[1], "none", dtype=object)
    best_family = np.full(stack.shape[1], "none", dtype=object)
    best_abs = np.full(stack.shape[1], np.nan, dtype=np.float64)
    if np.any(finite_any):
        col_idx = np.where(finite_any)[0]
        best_abs[col_idx] = stack[idx[col_idx], col_idx]
        best_name[col_idx] = np.array([names[i] for i in idx[col_idx]], dtype=object)
        best_family[col_idx] = np.array([families[i] for i in idx[col_idx]], dtype=object)

    best_abs = np.where(finite_any, best_abs, np.nan)

    out["best_resonance_name"] = pd.Series(best_name, dtype="object")
    out["best_resonance_family"] = pd.Series(best_family, dtype="object")
    out["best_resonance_abs_residual_rad_day"] = best_abs
    out["is_resonance_proximate"] = np.isfinite(best_abs) & (best_abs <= float(tolerance_rad_day))
    out["resonance_tolerance_rad_day"] = float(tolerance_rad_day)
    out["resonance_capture_warning"] = "proximity_not_capture_requires_angle_libration"
    out["resonance_proximity_only"] = True
    out["capture_not_proven"] = True
    out["external_frequency_approximate"] = bool(
        freq_meta["lambda_dot_sun_approximate"] or freq_meta["omega_dot_moon_approximate"] or freq_meta["theta_dot_earth_approximate"]
    )

    if include_angle_diagnostics:
        out, angle_meta = compute_resonant_angles(
            out,
            resonance_definitions=[d.__dict__ for d in defs_n],
            timestamp_col=timestamp_col,
            unwrap=bool(include_unwrapped_angles),
            include_apsidal_warning_text=bool(include_apsidal_warning_text),
            lambda_sun_provider=lambda_sun_angle_provider,
            omega_moon_provider=omega_moon_angle_provider,
            theta_earth_provider=theta_earth_angle_provider,
            mean_anomaly_provider=mean_anomaly_angle_provider,
            return_metadata=True,
        )

        if object_col is None:
            try:
                resolved_object_col = resolve_object_col(out)
                object_col = resolved_object_col if resolved_object_col in out.columns else None
            except Exception:
                object_col = None
        elif object_col not in out.columns:
            object_col = None

        if object_col is not None and timestamp_col in out.columns:
            behavior_map: Dict[Tuple[str, str], str] = {}
            for definition in defs_n:
                angle_col = f"psi_{definition.name}_deg_wrapped"
                if angle_col not in out.columns:
                    continue
                work = out[[object_col, timestamp_col, angle_col]].copy()
                work[object_col] = work[object_col].astype(str)
                work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
                work = work.dropna(subset=[timestamp_col]).sort_values([object_col, timestamp_col], kind="mergesort")
                for obj_id, grp in work.groupby(object_col, sort=False):
                    behavior_map[(str(obj_id), definition.name)] = classify_resonant_angle_behavior(grp[angle_col].to_numpy(dtype=np.float64))

            out["best_resonance_angle_behavior"] = "insufficient_data"
            obj_values = out[object_col].astype(str).to_numpy(dtype=object)
            best_values = out["best_resonance_name"].astype(str).to_numpy(dtype=object)
            vals = []
            for obj_id, res_name in zip(obj_values, best_values):
                vals.append(behavior_map.get((str(obj_id), str(res_name)), "insufficient_data"))
            out["best_resonance_angle_behavior"] = pd.Series(vals, dtype="object")
            out["angle_based_resonance_candidate"] = out["best_resonance_angle_behavior"].eq("libration_candidate") & out["is_resonance_proximate"].astype(bool)
        else:
            out["best_resonance_angle_behavior"] = "insufficient_data"
            out["angle_based_resonance_candidate"] = False

        out["angle_metadata_summary"] = str(angle_meta)

    if "angle_based_resonance_candidate" not in out.columns:
        out["angle_based_resonance_candidate"] = False

    best_name_series = out["best_resonance_name"].astype(str)
    best_uses_apsides = best_name_series.map(lambda name: uses_apsidal_by_name.get(name, False)).fillna(False).astype(bool)
    ecc = _safe_numeric(out, ecc_col)
    low_e_flag = np.isfinite(ecc) & (ecc <= float(low_e_threshold))
    out["apsidal_semantics_low_e_flag"] = best_uses_apsides.to_numpy(dtype=bool) & low_e_flag
    out["resonance_capture_candidate"] = out["is_resonance_proximate"].astype(bool) & out["angle_based_resonance_candidate"].astype(bool)
    out["resonance_capture_confidence"] = np.where(
        out["resonance_capture_candidate"].astype(bool) & ~out["apsidal_semantics_low_e_flag"].astype(bool),
        "screening_candidate",
        np.where(
            out["resonance_capture_candidate"].astype(bool),
            "low_confidence_low_e_apsidal",
            "not_candidate",
        ),
    )
    out["resonance_summary_semantics"] = "residual_and_angle_screening_only_capture_not_proven"

    return out


def estimate_resonance_width_proxy(
    df: pd.DataFrame,
    residual_col: str = "best_resonance_abs_residual_rad_day",
) -> pd.Series:
    """Compatibility helper for width proxy estimation.

    Use ``estimate_resonance_width_proxy_detailed`` for family-scaled and local models.
    """
    residual = pd.to_numeric(df.get(residual_col), errors="coerce")
    with np.errstate(invalid="ignore", divide="ignore"):
        width = 1.0 / (1.0 + residual.abs())
    return width.rename("resonance_width_proxy")


def estimate_resonance_width_proxy_detailed(
    df: pd.DataFrame,
    residual_col: str = "best_resonance_abs_residual_rad_day",
    family_col: str = "best_resonance_family",
    object_col: Optional[str] = None,
    timestamp_col: str = "timestamp",
    method: str = "family_scaled",
    rolling_window: int = 12,
    family_scales: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Return a richer proxy with method and confidence metadata.

    This remains a heuristic product and does not estimate true dynamical capture width.
    """
    if family_scales is None:
        family_scales = {
            "SRP_SOLAR_LEO": 1.0,
            "LUNISOLAR_SECULAR_MEO": 0.75,
            "INCLINATION_ONLY_SECULAR": 0.9,
            "TESSERAL_GROUNDTRACK": 0.6,
            "none": 0.5,
        }

    out = df.copy()
    residual = pd.to_numeric(out.get(residual_col), errors="coerce").abs()
    fam = out.get(family_col, "none")
    if not isinstance(fam, pd.Series):
        fam = pd.Series(["none"] * len(out), index=out.index)
    fam = fam.fillna("none").astype(str)

    if method not in {"reciprocal", "family_scaled", "rolling_local"}:
        raise ValueError("method must be one of: reciprocal, family_scaled, rolling_local")

    with np.errstate(invalid="ignore", divide="ignore"):
        base = 1.0 / (1.0 + residual)

    if method == "reciprocal":
        out["resonance_width_proxy"] = base
        out["resonance_width_proxy_confidence"] = np.where(np.isfinite(residual), 0.4, np.nan)
        out["resonance_width_proxy_method"] = "reciprocal"
        return out

    scale_arr = fam.map(lambda x: float(family_scales.get(x, family_scales.get("none", 0.5))))
    if method == "family_scaled":
        out["resonance_width_proxy"] = base * scale_arr
        out["resonance_width_proxy_confidence"] = np.where(np.isfinite(residual), 0.6, np.nan)
        out["resonance_width_proxy_method"] = "family_scaled"
        return out

    # rolling_local
    if object_col is None:
        try:
            object_col = resolve_object_col(out)
        except Exception:
            object_col = None

    proxy = np.full(len(out), np.nan, dtype=np.float64)
    conf = np.full(len(out), np.nan, dtype=np.float64)
    if object_col is None or timestamp_col not in out.columns:
        out["resonance_width_proxy"] = base * scale_arr
        out["resonance_width_proxy_confidence"] = np.where(np.isfinite(residual), 0.5, np.nan)
        out["resonance_width_proxy_method"] = "rolling_local_fallback"
        return out

    work = out[[object_col, timestamp_col, residual_col, family_col]].copy()
    work[object_col] = work[object_col].astype(str)
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[residual_col] = pd.to_numeric(work[residual_col], errors="coerce")
    work = work.sort_values([object_col, timestamp_col], kind="mergesort")

    for obj_id, grp in work.groupby(object_col, sort=False):
        ridx = grp.index.to_numpy(dtype=np.int64)
        r = grp[residual_col].abs().to_numpy(dtype=np.float64)
        rolling = pd.Series(r).rolling(window=max(3, int(rolling_window)), min_periods=3).median().to_numpy(dtype=np.float64)
        grad = np.gradient(np.where(np.isfinite(rolling), rolling, np.nan), edge_order=1)
        grad_abs = np.abs(grad)
        with np.errstate(invalid="ignore", divide="ignore"):
            local = 1.0 / (1.0 + np.where(np.isfinite(rolling), rolling, r) + grad_abs)
        fam_obj = grp[family_col].fillna("none").astype(str).map(lambda x: float(family_scales.get(x, family_scales.get("none", 0.5)))).to_numpy(dtype=np.float64)
        proxy[ridx] = local * fam_obj
        conf[ridx] = np.where(np.isfinite(local), 0.75, np.nan)

    out["resonance_width_proxy"] = proxy
    out["resonance_width_proxy_confidence"] = conf
    out["resonance_width_proxy_method"] = "rolling_local"
    return out


def map_resonance_proximity_over_ai_grid(df: pd.DataFrame, a_col: str = "sma", i_col: str = "inc",
                                         prox_col: str = "best_resonance_abs_residual_rad_day",
                                         prox_flag_col: str = "is_resonance_proximate",
                                         family_col: str = "best_resonance_family",
                                         a_bins: int = 40, i_bins: int = 36,
                                         min_count: int = 5) -> pd.DataFrame:
    """Aggregate resonance proximity over (a, i) bins."""
    cols = [a_col, i_col, prox_col]
    if prox_flag_col in df.columns:
        cols.append(prox_flag_col)
    if family_col in df.columns:
        cols.append(family_col)
    work = df[cols].copy()
    work[a_col] = pd.to_numeric(work[a_col], errors="coerce")
    work[i_col] = pd.to_numeric(work[i_col], errors="coerce")
    work[prox_col] = pd.to_numeric(work[prox_col], errors="coerce")
    if prox_flag_col not in work.columns:
        work[prox_flag_col] = np.isfinite(work[prox_col])
    work[prox_flag_col] = pd.Series(work[prox_flag_col]).fillna(False).astype(bool)
    if family_col not in work.columns:
        work[family_col] = "none"
    work[family_col] = work[family_col].fillna("none").astype(str)
    work = work.dropna(subset=[a_col, i_col])

    if work.empty:
        return pd.DataFrame(
            columns=[
                "a_bin", "i_bin", "count", "median_abs_residual_rad_day", "p25_abs_residual_rad_day",
                "p75_abs_residual_rad_day", "proximate_fraction", "dominant_family"
            ]
        )

    work["a_bin"] = pd.cut(work[a_col], bins=a_bins, include_lowest=True)
    work["i_bin"] = pd.cut(work[i_col], bins=i_bins, include_lowest=True)
    g = work.groupby(["a_bin", "i_bin"], observed=False)

    def _nan_percentile(x: pd.Series, p: float) -> float:
        arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.nanpercentile(arr, p))

    summary = g.agg(
        count=(prox_col, "size"),
        median_abs_residual_rad_day=(prox_col, "median"),
        p25_abs_residual_rad_day=(prox_col, lambda x: _nan_percentile(x, 25.0)),
        p75_abs_residual_rad_day=(prox_col, lambda x: _nan_percentile(x, 75.0)),
        n_proximate=(prox_flag_col, "sum"),
    ).reset_index()

    with np.errstate(invalid="ignore", divide="ignore"):
        summary["proximate_fraction"] = summary["n_proximate"] / summary["count"].replace(0, np.nan)

    fam_counts = (
        work.groupby(["a_bin", "i_bin", family_col], observed=False)
        .size()
        .reset_index(name="family_count")
        .sort_values(["a_bin", "i_bin", "family_count", family_col], ascending=[True, True, False, True])
    )
    dominant = fam_counts.drop_duplicates(["a_bin", "i_bin"], keep="first")[ ["a_bin", "i_bin", family_col] ]
    dominant = dominant.rename(columns={family_col: "dominant_family"})
    summary = summary.merge(dominant, on=["a_bin", "i_bin"], how="left")

    valid = summary["count"] >= int(min_count)
    for col in ["median_abs_residual_rad_day", "p25_abs_residual_rad_day", "p75_abs_residual_rad_day", "proximate_fraction"]:
        summary.loc[~valid, col] = np.nan
    summary.loc[~valid, "dominant_family"] = "insufficient_count"
    return summary


def summarize_resonant_objects(df: pd.DataFrame, object_col: str | None = None,
                               prox_flag_col: str = "is_resonance_proximate",
                               timestamp_col: str = "timestamp",
                               fraction_mode: str = "record",
                               time_window_freq: str = "7D",
                               deduplicate_epochs: bool = True) -> pd.DataFrame:
    """Summarize resonance proximity by object."""
    if object_col is None:
        object_col = resolve_object_col(df)

    work = df.copy()
    work[object_col] = work[object_col].astype(str)
    work[prox_flag_col] = pd.Series(work.get(prox_flag_col, False)).fillna(False).astype(bool)
    capture_col_present = "resonance_capture_candidate" in work.columns
    if capture_col_present:
        work["resonance_capture_candidate"] = pd.Series(work["resonance_capture_candidate"]).fillna(False).astype(bool)
    if timestamp_col in work.columns:
        work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")

    if fraction_mode not in {"record", "unique_epoch", "time_window"}:
        raise ValueError("fraction_mode must be one of: record, unique_epoch, time_window")

    if fraction_mode == "record":
        out = work.groupby(object_col, sort=True).agg(
            n_records=(prox_flag_col, "size"),
            n_proximate=(prox_flag_col, "sum"),
            min_abs_residual_rad_day=("best_resonance_abs_residual_rad_day", "min"),
            median_abs_residual_rad_day=("best_resonance_abs_residual_rad_day", "median"),
        ).reset_index().rename(columns={object_col: "object_id"})

        if capture_col_present:
            cap = work.groupby(object_col, sort=True).agg(
                n_capture_candidates=("resonance_capture_candidate", "sum"),
            ).reset_index().rename(columns={object_col: "object_id"})
            out = out.merge(cap, on="object_id", how="left")
            out["capture_candidate_fraction"] = out["n_capture_candidates"] / out["n_records"].replace(0, np.nan)

        out["proximate_fraction"] = out["n_proximate"] / out["n_records"].replace(0, np.nan)
        out["fraction_mode"] = "record"
        out["resonance_proximity_only"] = True
        out["capture_not_proven"] = True
        return out

    if timestamp_col not in work.columns:
        raise KeyError(f"timestamp column '{timestamp_col}' is required for fraction_mode={fraction_mode}")

    work = work.dropna(subset=[timestamp_col])
    if deduplicate_epochs:
        work = work.sort_values([object_col, timestamp_col], kind="mergesort")
        work = work.drop_duplicates(subset=[object_col, timestamp_col], keep="last")

    if fraction_mode == "unique_epoch":
        out = work.groupby(object_col, sort=True).agg(
            n_records=(timestamp_col, "size"),
            n_proximate=(prox_flag_col, "sum"),
            min_abs_residual_rad_day=("best_resonance_abs_residual_rad_day", "min"),
            median_abs_residual_rad_day=("best_resonance_abs_residual_rad_day", "median"),
        ).reset_index().rename(columns={object_col: "object_id"})
        if capture_col_present:
            cap = work.groupby(object_col, sort=True).agg(
                n_capture_candidates=("resonance_capture_candidate", "sum"),
            ).reset_index().rename(columns={object_col: "object_id"})
            out = out.merge(cap, on="object_id", how="left")
            out["capture_candidate_fraction"] = out["n_capture_candidates"] / out["n_records"].replace(0, np.nan)
        out["proximate_fraction"] = out["n_proximate"] / out["n_records"].replace(0, np.nan)
        out["fraction_mode"] = "unique_epoch"
        out["resonance_proximity_only"] = True
        out["capture_not_proven"] = True
        return out

    # time-window mode
    pieces = []
    for obj_id, grp in work.groupby(object_col, sort=True):
        g2 = grp.set_index(timestamp_col).sort_index()
        window_total = g2[prox_flag_col].resample(time_window_freq).size()
        window_prox = g2[prox_flag_col].resample(time_window_freq).max().fillna(False).astype(bool)
        pieces.append(
            {
                "object_id": str(obj_id),
                "n_records": int(window_total.size),
                "n_proximate": int(window_prox.sum()),
                "n_capture_candidates": int(g2["resonance_capture_candidate"].astype(bool).sum()) if capture_col_present else 0,
                "min_abs_residual_rad_day": float(pd.to_numeric(grp["best_resonance_abs_residual_rad_day"], errors="coerce").min()),
                "median_abs_residual_rad_day": float(pd.to_numeric(grp["best_resonance_abs_residual_rad_day"], errors="coerce").median()),
            }
        )

    out = pd.DataFrame(pieces)
    if out.empty:
        out = pd.DataFrame(columns=["object_id", "n_records", "n_proximate", "min_abs_residual_rad_day", "median_abs_residual_rad_day"])
    out["proximate_fraction"] = out["n_proximate"] / out["n_records"].replace(0, np.nan)
    if "n_capture_candidates" in out.columns:
        out["capture_candidate_fraction"] = out["n_capture_candidates"] / out["n_records"].replace(0, np.nan)
    out["fraction_mode"] = "time_window"
    out["resonance_proximity_only"] = True
    out["capture_not_proven"] = True
    return out


def screen_tesseral_commensurability(
    df: pd.DataFrame,
    candidate_ratios: Sequence[Tuple[int, int]] = ((14, 1), (15, 1), (29, 2)),
    include_arguments: bool = True,
    mean_anomaly_col: str = "mean_anomaly",
    true_anomaly_col: str = "true_anomaly",
    omega_col: str = "aop",
    Omega_col: str = "raan",
    ecc_col: str = "ecc",
) -> pd.DataFrame:
    """Return tesseral repeat-ratio residuals and optional argument time series.

    Arguments use a mean-longitude construction: lambda = M + omega + Omega.
    Candidate tuple ``(m, l)`` means ``m:l`` (m orbits in l sidereal days),
    with residual ``R = l*n - m*theta_dot_earth``.
    """
    out = compute_resonance_frequencies(df)
    n = _safe_numeric(out, "n_rad_day")
    theta = _safe_numeric(out, "theta_dot_earth_rad_day", fallback=EARTH_ROTATION_RAD_DAY)
    out = _approximate_external_angles_from_time(out, timestamp_col="timestamp")

    M, m_source = _compute_mean_anomaly_deg(
        out,
        mean_anomaly_col=mean_anomaly_col,
        true_anomaly_col=true_anomaly_col,
        ecc_col=ecc_col,
    )
    omega = _safe_numeric(out, omega_col)
    Omega = _safe_numeric(out, Omega_col)
    lambda_mean = _wrap_deg(M + omega + Omega)

    out["mean_anomaly_deg_for_tesseral"] = M
    out["mean_longitude_deg_for_tesseral"] = lambda_mean
    out["tesseral_argument_semantics"] = (
        "mean_anomaly_provided"
        if m_source == "provided"
        else ("mean_anomaly_derived_proxy" if m_source == "derived_from_true_anomaly" else "mean_anomaly_unavailable")
    )

    theta_angle = _safe_numeric(out, "theta_earth_deg")
    for m, l in candidate_ratios:
        out[f"tesseral_residual_{m}_{l}_rad_day"] = (float(l) * n) - (float(m) * theta)
        if include_arguments:
            out[f"tesseral_argument_{m}_{l}_deg_wrapped"] = _wrap_deg((float(l) * lambda_mean) - (float(m) * theta_angle))
            out[f"tesseral_argument_{m}_{l}_deg_unwrapped"] = np.rad2deg(
                np.unwrap(np.deg2rad(out[f"tesseral_argument_{m}_{l}_deg_wrapped"].to_numpy(dtype=np.float64)))
            )
    return out


def plot_resonance_proximity_ai_map(ai_summary_df: pd.DataFrame, value_col: str = "median_abs_residual_rad_day",
                                    title: str = "Resonance Proximity in (a, i)"):
    """Plot a simple heatmap of resonance proximity over (a,i) bins."""
    if ai_summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f"{title} (no data)")
        return fig

    pivot = ai_summary_df.pivot(index="i_bin", columns="a_bin", values=value_col)
    if pivot.shape[1] > 5:
        pivot = pivot.iloc[:, :5]
    grid = pivot.to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("semi-major axis bins")
    ax.set_ylabel("inclination bins")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    return fig
