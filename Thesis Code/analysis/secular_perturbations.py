"""Secular perturbations and precession helpers.

This module centralizes first-order J2 secular-rate analytics used across the
Research pipeline and keeps mean-vs-proxy element semantics explicit.

Scientific references:
- Vallado et al., Revisiting Spacetrack Report No. 3 (SGP4/TLE semantics)
- Brouwer/Kozai averaged secular theory (first-order zonal secular rates)
- DSST literature for higher-order averaged formulations (future extensions)
- CelesTrak GP semantics: https://celestrak.org/NORAD/documentation/gp-data-formats.php
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional
import os
import warnings

import numpy as np
import pandas as pd

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    njit = None
    _HAS_NUMBA = False

from constants import J2_EARTH, MU_EARTH, RADIUS_EARTH, SECONDS_PER_DAY

SOLAR_MOTION_RAD_DAY = 2.0 * np.pi / 365.2422
J3_EARTH = -2.532153e-6
J4_EARTH = -1.61962159137e-6
MU_SUN = 1.32712440018e11
MU_MOON = 4902.800066
SUN_A_KM = 149_597_870.7
MOON_A_KM = 384_400.0
SUN_ECC = 0.0167086
MOON_ECC = 0.0549

_NUMBA_DISABLED_ENV = {"0", "false", "no", "off"}
_USE_NUMBA = _HAS_NUMBA and str(os.getenv("SECULAR_PERTURBATIONS_USE_NUMBA", "1")).strip().lower() not in _NUMBA_DISABLED_ENV


if _HAS_NUMBA:

    @njit(cache=True)
    def _j2_first_order_rates_numba(a, e, inc_deg, mu_km3_s2, reference_radius_km, j2):
        node = np.empty(a.size, dtype=np.float64)
        apsidal = np.empty(a.size, dtype=np.float64)
        for i in range(a.size):
            node[i] = np.nan
            apsidal[i] = np.nan

            ai = a[i]
            ei = e[i]
            ii = inc_deg[i]
            if not np.isfinite(ai) or not np.isfinite(ei) or not np.isfinite(ii):
                continue
            if ai <= 0.0 or ei < 0.0 or ei >= 1.0:
                continue

            iv = ii * (np.pi / 180.0)
            n = np.sqrt(mu_km3_s2 / (ai * ai * ai))
            p = ai * (1.0 - ei * ei)
            ratio = reference_radius_km / p
            factor = j2 * n * (ratio * ratio)
            c = np.cos(iv)
            c2 = c * c

            node[i] = -1.5 * factor * c
            apsidal[i] = 0.75 * factor * (5.0 * c2 - 1.0)

        return node, apsidal


    @njit(cache=True)
    def _third_body_secular_rates_numba(a, e, inc_deg, mu_km3_s2, mu_sun, mu_moon, a_sun_km, a_moon_km, e_sun, e_moon):
        node = np.empty(a.size, dtype=np.float64)
        apsidal = np.empty(a.size, dtype=np.float64)
        one_minus_esun = (1.0 - e_sun * e_sun)
        one_minus_emoon = (1.0 - e_moon * e_moon)
        esun_term = one_minus_esun ** 1.5
        emoon_term = one_minus_emoon ** 1.5

        for i in range(a.size):
            node[i] = np.nan
            apsidal[i] = np.nan

            ai = a[i]
            ei = e[i]
            ii = inc_deg[i]
            if not np.isfinite(ai) or not np.isfinite(ei) or not np.isfinite(ii):
                continue
            if ai <= 0.0 or ei < 0.0 or ei >= 1.0:
                continue

            iv = ii * (np.pi / 180.0)
            n = np.sqrt(mu_km3_s2 / (ai * ai * ai))
            cos_i = np.cos(iv)
            cos2 = cos_i * cos_i

            sun_scale = (mu_sun / mu_km3_s2) * ((ai / a_sun_km) ** 3.0) / esun_term
            moon_scale = (mu_moon / mu_km3_s2) * ((ai / a_moon_km) ** 3.0) / emoon_term
            scale = 0.75 * n * (sun_scale + moon_scale)

            node[i] = -scale * cos_i
            apsidal[i] = scale * (5.0 * cos2 - 1.0)

        return node, apsidal

SUPPORTED_SECULAR_MODELS = {
    "J2_first_order": {
        "implemented": True,
        "category": "analytic",
        "assumption": "Brouwer/Kozai first-order averaged J2 secular rates",
    },
    "third_body_secular": {
        "implemented": True,
        "category": "analytic",
        "assumption": "Doubly averaged Sun/Moon quadrupole surrogate secular rates (screening-level)",
    },
    "empirical_from_history": {
        "implemented": True,
        "category": "empirical",
        "assumption": "Rates inferred from historical angular series",
    },
}


def get_supported_secular_models() -> Dict[str, Dict[str, Any]]:
    """Return copy of model capability registry for discovery/UI plumbing."""
    return {k: dict(v) for k, v in SUPPORTED_SECULAR_MODELS.items()}


@dataclass(frozen=True)
class _SeriesFit:
    slope_per_second: float
    intercept: float


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _valid_orbital_mask(a: np.ndarray, e: np.ndarray, inc_deg: np.ndarray) -> np.ndarray:
    mask = np.isfinite(a) & np.isfinite(e) & np.isfinite(inc_deg)
    mask &= a > 0.0
    mask &= e >= 0.0
    mask &= e < 1.0
    return mask


def _unit_scale(output_units: str) -> float:
    unit = str(output_units).lower()
    if unit == "rad_s":
        return 1.0
    if unit == "rad_day":
        return SECONDS_PER_DAY
    if unit == "deg_day":
        return SECONDS_PER_DAY * (180.0 / np.pi)
    raise ValueError("output_units must be one of {'rad_s', 'rad_day', 'deg_day'}")


def p_from_ae(semi_major_axis_km: Any, eccentricity: Any) -> np.ndarray:
    """Return semilatus rectum $p=a(1-e^2)$ in km."""
    a = _as_float_array(semi_major_axis_km)
    e = _as_float_array(eccentricity)
    if a.shape != e.shape:
        raise ValueError("semi_major_axis_km and eccentricity must have the same shape")
    return a * (1.0 - np.square(e))


def critical_inclination_deg() -> np.ndarray:
    r"""Return critical inclinations (deg) where $5\cos^2(i)-1=0$."""
    i1 = np.rad2deg(np.arccos(np.sqrt(1.0 / 5.0)))
    i2 = 180.0 - i1
    return np.array([i1, i2], dtype=np.float64)


def is_near_critical_inclination(inclination_deg: Any, tolerance_deg: float = 0.25) -> np.ndarray:
    """Check if inclinations are near either critical inclination branch."""
    inc = _as_float_array(inclination_deg)
    crit = critical_inclination_deg()
    return (np.abs(inc - crit[0]) <= float(tolerance_deg)) | (np.abs(inc - crit[1]) <= float(tolerance_deg))


def sun_sync_required_inclination_deg(
    semi_major_axis_km: float,
    eccentricity: float = 0.0,
    *,
    mu_km3_s2: float = MU_EARTH,
    reference_radius_km: float = RADIUS_EARTH,
    j2: float = J2_EARTH,
    target_node_rate_rad_day: float = SOLAR_MOTION_RAD_DAY,
) -> float:
    r"""Solve inclination for target nodal rate under first-order J2.

    Sign convention is explicit: positive rates are increasing RAAN.

    Default behavior solves for the conventional Earth sun-synchronous
    *retrograde* branch by targeting
    $\dot{\Omega} \approx +n_{sun}$ (eastward inertial nodal precession,
    typically $i > 90^\circ$ for LEO).
    """
    a = float(semi_major_axis_km)
    e = float(eccentricity)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("semi_major_axis_km must be finite and > 0")
    if not np.isfinite(e) or e < 0.0 or e >= 1.0:
        raise ValueError("eccentricity must satisfy 0 <= e < 1")

    n = np.sqrt(float(mu_km3_s2) / (a ** 3.0))
    p = a * (1.0 - e * e)
    coeff = 1.5 * float(j2) * n * (float(reference_radius_km) / p) ** 2.0
    target_rad_s = float(target_node_rate_rad_day) / SECONDS_PER_DAY
    cos_i = -target_rad_s / coeff
    if abs(cos_i) > 1.0:
        raise ValueError("No real inclination solution for the requested sun-sync rate at this (a,e)")
    return float(np.rad2deg(np.arccos(cos_i)))


def _j2_first_order_rates(
    a: np.ndarray,
    e: np.ndarray,
    inc_deg: np.ndarray,
    *,
    mu_km3_s2: float,
    reference_radius_km: float,
    j2: float,
) -> tuple[np.ndarray, np.ndarray]:
    if _USE_NUMBA and a.ndim == 1 and e.ndim == 1 and inc_deg.ndim == 1 and a.shape == e.shape and e.shape == inc_deg.shape:
        return _j2_first_order_rates_numba(
            np.asarray(a, dtype=np.float64),
            np.asarray(e, dtype=np.float64),
            np.asarray(inc_deg, dtype=np.float64),
            float(mu_km3_s2),
            float(reference_radius_km),
            float(j2),
        )

    node = np.full(a.shape, np.nan, dtype=np.float64)
    apsidal = np.full(a.shape, np.nan, dtype=np.float64)

    valid = _valid_orbital_mask(a, e, inc_deg)
    if np.any(valid):
        av = a[valid]
        ev = e[valid]
        iv = np.deg2rad(inc_deg[valid])

        n = np.sqrt(float(mu_km3_s2) / np.power(av, 3.0))
        p = av * (1.0 - np.square(ev))
        factor = float(j2) * n * np.square(float(reference_radius_km) / p)

        node[valid] = -1.5 * factor * np.cos(iv)
        apsidal[valid] = 0.75 * factor * (5.0 * np.square(np.cos(iv)) - 1.0)

    return node, apsidal


def _third_body_secular_rates(
    a: np.ndarray,
    e: np.ndarray,
    inc_deg: np.ndarray,
    *,
    mu_km3_s2: float,
    mu_sun: float,
    mu_moon: float,
    a_sun_km: float,
    a_moon_km: float,
    e_sun: float,
    e_moon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Doubly averaged quadrupole surrogate rates for Sun+Moon screening diagnostics."""
    if _USE_NUMBA and a.ndim == 1 and e.ndim == 1 and inc_deg.ndim == 1 and a.shape == e.shape and e.shape == inc_deg.shape:
        return _third_body_secular_rates_numba(
            np.asarray(a, dtype=np.float64),
            np.asarray(e, dtype=np.float64),
            np.asarray(inc_deg, dtype=np.float64),
            float(mu_km3_s2),
            float(mu_sun),
            float(mu_moon),
            float(a_sun_km),
            float(a_moon_km),
            float(e_sun),
            float(e_moon),
        )

    node = np.full(a.shape, np.nan, dtype=np.float64)
    aps = np.full(a.shape, np.nan, dtype=np.float64)

    valid = _valid_orbital_mask(a, e, inc_deg)
    if np.any(valid):
        av = a[valid]
        iv = np.deg2rad(inc_deg[valid])
        n = np.sqrt(float(mu_km3_s2) / np.power(av, 3.0))
        cos_i = np.cos(iv)
        cos2 = np.square(cos_i)

        sun_scale = (float(mu_sun) / float(mu_km3_s2)) * np.power(av / float(a_sun_km), 3.0) / np.power(1.0 - float(e_sun) ** 2, 1.5)
        moon_scale = (float(mu_moon) / float(mu_km3_s2)) * np.power(av / float(a_moon_km), 3.0) / np.power(1.0 - float(e_moon) ** 2, 1.5)
        scale = 0.75 * n * (sun_scale + moon_scale)

        node[valid] = -scale * cos_i
        aps[valid] = scale * (5.0 * cos2 - 1.0)

    return node, aps


def compute_secular_rates_from_elements(
    semi_major_axis_km: Any,
    eccentricity: Any,
    inclination_deg: Any,
    *,
    mu_km3_s2: float = MU_EARTH,
    reference_radius_km: float = RADIUS_EARTH,
    j2: float = J2_EARTH,
    model: str = "J2_first_order",
    element_semantics: str = "diagnostic_proxy",
    averaging_theory: Optional[str] = None,
    ecc_threshold_for_apsides: float = 1e-3,
    output_units: str = "rad_s",
    return_metadata: bool = False,
    critical_inclination_tolerance: float = 1.0e-3,
    sun_sync_target_rate_rad_day: Optional[float] = None,
    allow_placeholder_metadata: bool = False,
    history_times: Optional[Any] = None,
    history_raan: Optional[Any] = None,
    history_argp: Optional[Any] = None,
    history_angle_units: str = "deg",
    j3: float = J3_EARTH,
    j4: float = J4_EARTH,
) -> Any:
    """Compute secular nodal and apsidal rates from orbital elements.

    Core implemented model:
    - `J2_first_order` with
      n = sqrt(mu/a^3), p = a(1-e^2),
      dOmega/dt = -(3/2) J2 n (Re/p)^2 cos(i),
      domega/dt = (3/4) J2 n (Re/p)^2 (5 cos^2(i) - 1).

    Semantic note:
    - When inputs are TLE/GP-derived elements, rates are best treated as
      diagnostic proxies unless a theory-consistent averaged element pipeline is
      explicitly used.
    """
    a = _as_float_array(semi_major_axis_km)
    e = _as_float_array(eccentricity)
    inc = _as_float_array(inclination_deg)
    if not (a.shape == e.shape == inc.shape):
        raise ValueError("semi_major_axis_km, eccentricity, and inclination_deg must have the same shape")

    model_key = str(model)
    if model_key not in SUPPORTED_SECULAR_MODELS:
        raise ValueError(f"Unsupported model '{model_key}'. Supported: {sorted(SUPPORTED_SECULAR_MODELS)}")
    if not SUPPORTED_SECULAR_MODELS[model_key]["implemented"]:
        available = [k for k, v in SUPPORTED_SECULAR_MODELS.items() if bool(v.get("implemented"))]
        raise ValueError(
            f"Model '{model_key}' is registered but disabled; use one of {available}."
        )

    if model_key == "J2_first_order":
        node_rad_s, apsidal_rad_s = _j2_first_order_rates(
            a,
            e,
            inc,
            mu_km3_s2=float(mu_km3_s2),
            reference_radius_km=float(reference_radius_km),
            j2=float(j2),
        )
        frozen_e = None
    elif model_key == "third_body_secular":
        warnings.warn(
            "third_body_secular is a screening surrogate for trend/explanatory diagnostics; "
            "use J2_first_order as the preferred baseline for LEO Starlink analyses.",
            RuntimeWarning,
            stacklevel=2,
        )
        node_rad_s, apsidal_rad_s = _third_body_secular_rates(
            a,
            e,
            inc,
            mu_km3_s2=float(mu_km3_s2),
            mu_sun=float(MU_SUN),
            mu_moon=float(MU_MOON),
            a_sun_km=float(SUN_A_KM),
            a_moon_km=float(MOON_A_KM),
            e_sun=float(SUN_ECC),
            e_moon=float(MOON_ECC),
        )
        frozen_e = None
    elif model_key == "empirical_from_history":
        if history_times is None or history_raan is None or history_argp is None:
            raise ValueError(
                "model='empirical_from_history' requires history_times, history_raan, and history_argp"
            )
        node_fit = estimate_empirical_secular_rate_from_time_series(
            history_times,
            history_raan,
            angle_units=history_angle_units,
            wrapped=True,
            auto_robust=True,
            return_details=True,
        )
        apsidal_fit = estimate_empirical_secular_rate_from_time_series(
            history_times,
            history_argp,
            angle_units=history_angle_units,
            wrapped=True,
            auto_robust=True,
            return_details=True,
        )
        node_rad_s = np.full(a.shape, float(node_fit["slope_rad_s"]), dtype=np.float64)
        apsidal_rad_s = np.full(a.shape, float(apsidal_fit["slope_rad_s"]), dtype=np.float64)
        frozen_e = None
    else:
        raise ValueError(f"Unhandled implemented model path: {model_key}")

    valid = _valid_orbital_mask(a, e, inc)
    apsidal_rate_valid = valid & (e >= float(ecc_threshold_for_apsides))
    crit_expr = np.full(a.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        cos_i = np.cos(np.deg2rad(inc[valid]))
        crit_expr[valid] = np.abs(5.0 * np.square(cos_i) - 1.0)
    critical_flag = valid & (crit_expr <= float(critical_inclination_tolerance))

    scale = _unit_scale(output_units)
    node_out = node_rad_s * scale
    apsidal_out = apsidal_rad_s * scale

    if not return_metadata:
        return node_out, apsidal_out

    sun_sync_error = None
    if sun_sync_target_rate_rad_day is not None:
        node_day = node_rad_s * SECONDS_PER_DAY
        sun_sync_error = node_day - float(sun_sync_target_rate_rad_day)

    assumptions_by_model = {
        "J2_first_order": (
            "First-order J2 secular rates. Suitable for averaged mean elements; "
            "when applied to TLE/GP-derived proxies treat as diagnostic."
        ),
        "third_body_secular": (
            "Doubly averaged Sun/Moon quadrupole surrogate secular rates. "
            "Use for screening only; not a high-fidelity replacement for full third-body modeling."
        ),
        "empirical_from_history": (
            "Empirical drift estimated directly from historical angular series; "
            "quality depends on sampling coverage and trend stationarity."
        ),
    }
    assumptions_text = assumptions_by_model.get(
        model_key,
        "Analytical secular-rate model used for diagnostics; validate against force-model propagation when needed.",
    )

    return {
        "node_rate": node_out,
        "apsidal_rate": apsidal_out,
        "metadata": {
            "model_used": model_key,
            "element_semantics": str(element_semantics),
            "averaging_theory": averaging_theory,
            "apsidal_rate_valid": apsidal_rate_valid,
            "critical_inclination_flag": critical_flag,
            "sun_sync_rate_error": sun_sync_error,
            "sun_sync_rate_error_abs": np.abs(sun_sync_error) if sun_sync_error is not None else None,
            "placeholder_model": False,
            "frozen_eccentricity_estimate": frozen_e,
            "assumptions": assumptions_text,
            "screening_surrogate": bool(model_key == "third_body_secular"),
            "preferred_leo_baseline": "J2_first_order",
        },
    }


def compute_model_comparison_rates(
    semi_major_axis_km: Any,
    eccentricity: Any,
    inclination_deg: Any,
    *,
    models: Optional[Iterable[str]] = None,
    output_units: str = "deg_day",
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Compatibility helper returning J2-only rates for legacy call paths."""
    requested = list(models) if models is not None else ["J2_first_order"]
    model_list = [name for name in requested if str(name) == "J2_first_order"]
    if not model_list:
        model_list = ["J2_first_order"]
    out: Dict[str, Dict[str, Any]] = {}
    for name in model_list:
        payload = compute_secular_rates_from_elements(
            semi_major_axis_km,
            eccentricity,
            inclination_deg,
            model=name,
            output_units=output_units,
            return_metadata=True,
            allow_placeholder_metadata=True,
            **kwargs,
        )
        out[name] = {
            "node_rate": payload["node_rate"],
            "apsidal_rate": payload["apsidal_rate"],
            "metadata": payload["metadata"],
        }
    return out


def j2_secular_rate_summary(
    semi_major_axis_km: Any,
    eccentricity: Any,
    inclination_deg: Any,
    *,
    output_units: str = "rad_day",
    element_semantics: str = "diagnostic_proxy",
    averaging_theory: Optional[str] = None,
    ecc_threshold_for_apsides: float = 1e-3,
    sun_sync_target_rate_rad_day: Optional[float] = None,
) -> Dict[str, Any]:
    """Return NaN-safe summary statistics and metadata for J2 rates."""
    payload = compute_secular_rates_from_elements(
        semi_major_axis_km,
        eccentricity,
        inclination_deg,
        output_units=output_units,
        return_metadata=True,
        element_semantics=element_semantics,
        averaging_theory=averaging_theory,
        ecc_threshold_for_apsides=ecc_threshold_for_apsides,
        sun_sync_target_rate_rad_day=sun_sync_target_rate_rad_day,
    )
    node = payload["node_rate"]
    apsidal = payload["apsidal_rate"]
    meta = payload["metadata"]

    return {
        "node_rate": node,
        "apsidal_rate": apsidal,
        "node_rate_mean": float(np.nanmean(node)) if np.isfinite(node).any() else np.nan,
        "apsidal_rate_mean": float(np.nanmean(apsidal)) if np.isfinite(apsidal).any() else np.nan,
        "node_rate_median": float(np.nanmedian(node)) if np.isfinite(node).any() else np.nan,
        "apsidal_rate_median": float(np.nanmedian(apsidal)) if np.isfinite(apsidal).any() else np.nan,
        "metadata": meta,
    }


def _fit_ols(x: np.ndarray, y: np.ndarray) -> _SeriesFit:
    coeff = np.polyfit(x, y, deg=1)
    return _SeriesFit(slope_per_second=float(coeff[0]), intercept=float(coeff[1]))


def _fit_huber_irls(x: np.ndarray, y: np.ndarray, max_iter: int = 25, c: float = 1.345) -> _SeriesFit:
    X = np.column_stack((x, np.ones_like(x)))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        resid = y - (X @ beta)
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad if mad > 0.0 else np.std(resid)
        if not np.isfinite(scale) or scale <= 0.0:
            break
        u = resid / (scale * c)
        w = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.abs(u))
        WX = X * w[:, None]
        beta_new = np.linalg.lstsq(WX, y * w, rcond=None)[0]
        if np.allclose(beta_new, beta, rtol=1e-9, atol=1e-12):
            beta = beta_new
            break
        beta = beta_new
    return _SeriesFit(slope_per_second=float(beta[0]), intercept=float(beta[1]))


def _coerce_time_seconds(times: Any) -> np.ndarray:
    arr = np.asarray(times)
    if arr.ndim != 1:
        raise ValueError("times must be one-dimensional")

    if np.issubdtype(arr.dtype, np.number):
        out = arr.astype(np.float64)
        out = out - out[0]
        return out

    ts = pd.to_datetime(arr, errors="coerce", utc=True)
    if ts.isna().all():
        raise ValueError("times could not be parsed as numeric or datetime values")
    delta = ts - ts[0]
    return np.asarray(delta.total_seconds(), dtype=np.float64)


def estimate_empirical_secular_rate_from_time_series(
    times: Any,
    angle_history: Any,
    *,
    angle_units: str = "rad",
    wrapped: bool = True,
    auto_robust: bool = True,
    outlier_sigma: float = 4.0,
    return_details: bool = True,
) -> Dict[str, Any] | float:
    """Estimate secular angular drift using unwrap + trend fitting.

    The default policy is automatic robust fitting when outlier contamination is
    detected from an initial OLS residual check.
    """
    t = _coerce_time_seconds(times)
    ang = _as_float_array(angle_history)
    if t.shape != ang.shape:
        raise ValueError("times and angle_history must have the same shape")

    finite = np.isfinite(t) & np.isfinite(ang)
    t = t[finite]
    ang = ang[finite]
    if len(t) < 3:
        raise ValueError("At least 3 finite samples are required")

    if str(angle_units).lower() == "deg":
        ang = np.deg2rad(ang)
    elif str(angle_units).lower() != "rad":
        raise ValueError("angle_units must be 'rad' or 'deg'")

    if wrapped:
        ang = np.unwrap(ang)

    ols = _fit_ols(t, ang)
    yhat = ols.slope_per_second * t + ols.intercept
    resid = ang - yhat

    mad = np.median(np.abs(resid - np.median(resid)))
    robust_scale = 1.4826 * mad if mad > 0.0 else np.nanstd(resid)
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        outlier_fraction = 0.0
    else:
        outlier_fraction = float(np.mean(np.abs(resid) > float(outlier_sigma) * robust_scale))

    method = "ols"
    fit = ols
    if auto_robust and outlier_fraction > 0.01:
        fit = _fit_huber_irls(t, ang)
        method = "huber_irls"

    slope_rad_s = fit.slope_per_second
    if not return_details:
        return slope_rad_s

    return {
        "slope_rad_s": float(slope_rad_s),
        "slope_rad_day": float(slope_rad_s * SECONDS_PER_DAY),
        "slope_deg_day": float(slope_rad_s * SECONDS_PER_DAY * (180.0 / np.pi)),
        "intercept_rad": float(fit.intercept),
        "method_used": method,
        "outlier_fraction": outlier_fraction,
        "n_samples": int(len(t)),
    }


def validate_secular_rates_with_force_model(
    *,
    times: Iterable[Any],
    initial_state: Any,
    propagate_states: Callable[[Any, Iterable[Any]], np.ndarray],
    states_to_elements: Callable[[np.ndarray], pd.DataFrame],
    analytic_reference: Optional[Dict[str, Any]] = None,
    ecc_threshold_for_apsides: float = 1e-3,
) -> Dict[str, Any]:
    """Optional scaffold to compare force-model trends against analytic J2 rates.

    This function intentionally does not replace analytic J2 formulas. It
    provides a comparison harness for special-perturbations propagation (for
    example, optional EGM2008 acceleration models).
    """
    states = propagate_states(initial_state, times)
    elem_df = states_to_elements(states)

    required_cols = {"sma", "ecc", "inc", "raan", "argp"}
    missing = required_cols - set(elem_df.columns)
    if missing:
        raise KeyError(f"states_to_elements output missing required columns: {sorted(missing)}")

    times_arr = np.asarray(list(times))
    emp_node = estimate_empirical_secular_rate_from_time_series(
        times_arr,
        elem_df["raan"].to_numpy(dtype=np.float64),
        angle_units="deg",
        wrapped=True,
        auto_robust=True,
        return_details=True,
    )
    emp_apsidal = estimate_empirical_secular_rate_from_time_series(
        times_arr,
        elem_df["argp"].to_numpy(dtype=np.float64),
        angle_units="deg",
        wrapped=True,
        auto_robust=True,
        return_details=True,
    )

    if analytic_reference is None:
        analytic = compute_secular_rates_from_elements(
            np.nanmedian(pd.to_numeric(elem_df["sma"], errors="coerce")),
            np.nanmedian(pd.to_numeric(elem_df["ecc"], errors="coerce")),
            np.nanmedian(pd.to_numeric(elem_df["inc"], errors="coerce")),
            output_units="rad_s",
            return_metadata=True,
            ecc_threshold_for_apsides=ecc_threshold_for_apsides,
        )
        analytic_node_rad_s = float(np.ravel(analytic["node_rate"])[0])
        analytic_apsidal_rad_s = float(np.ravel(analytic["apsidal_rate"])[0])
        analytic_meta = analytic["metadata"]
    else:
        analytic_node_rad_s = float(analytic_reference["node_rate_rad_s"])
        analytic_apsidal_rad_s = float(analytic_reference["apsidal_rate_rad_s"])
        analytic_meta = analytic_reference.get("metadata", {})

    return {
        "empirical": {
            "node": emp_node,
            "apsidal": emp_apsidal,
        },
        "analytic": {
            "node_rate_rad_s": analytic_node_rad_s,
            "apsidal_rate_rad_s": analytic_apsidal_rad_s,
            "metadata": analytic_meta,
        },
        "comparison": {
            "node_rate_error_rad_s": float(emp_node["slope_rad_s"] - analytic_node_rad_s),
            "apsidal_rate_error_rad_s": float(emp_apsidal["slope_rad_s"] - analytic_apsidal_rad_s),
        },
        "assumptions": (
            "Force-model validation compares osculating-history empirical drifts to analytic averaged rates; "
            "differences include averaging and force-model fidelity effects."
        ),
    }


# Consolidated from calculate_precession_rates.py
def calculate_precession_rates(semi_major_axes, eccentricities, inclinations, J2, r_E, chunk_size=10000):
    """
    Calculate first-order J2 secular nodal and apsidal precession rates.

    This legacy wrapper preserves the historical public signature and returns
    rates in rad/s. Internally it routes to the shared secular core.

    Parameters:
        semi_major_axes (np.array): The semi-major axes of the satellites in km.
        eccentricities (np.array): The eccentricities of the satellites.
        inclinations (np.array): The inclinations of the satellites in degrees.
        J2 (float): The J2 perturbation constant.
        r_E (float): The radius of the central body in km.
        chunk_size (int): Deprecated and unused; retained for backward
            compatibility.

    Returns:
        (node_precession_rates, perigee_precession_rates) (tuple): A tuple containing the node and perigee precession rates.

        node_precession_rates (np.array): The node precession rates of the satellites in rad/s.
        perigee_precession_rates (np.array): The perigee precession rates of the satellites in rad/s.
    """
    # Keep chunk_size in the signature for backward compatibility.
    _ = chunk_size

    a = np.asarray(semi_major_axes, dtype=np.float64)
    e = np.asarray(eccentricities, dtype=np.float64)
    inc_rad = np.deg2rad(np.asarray(inclinations, dtype=np.float64))

    if not (a.shape == e.shape == inc_rad.shape):
        raise ValueError("semi_major_axes, eccentricities, and inclinations must have the same shape")

    node_precession_rates, perigee_precession_rates = compute_secular_rates_from_elements(a, e, np.rad2deg(inc_rad), mu_km3_s2=398600.4418,
                                                                                          reference_radius_km=float(r_E), j2=float(J2),
                                                                                          model="J2_first_order",
                                                                                          element_semantics="diagnostic_proxy",
                                                                                          output_units="rad_s", return_metadata=False)

    return node_precession_rates, perigee_precession_rates

