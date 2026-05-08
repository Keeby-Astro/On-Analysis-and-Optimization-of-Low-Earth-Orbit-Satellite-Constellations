"""
Reduced-Order Orbital Dynamics for Mean-Element Propagation
===========================================================

This module provides the orbital mechanics primitives used by the Starlink
Stage A / Stage B Hall-thruster inference pipeline.  Every function here
operates on **mean Keplerian elements** derived from TLE-like catalog data,
*not* osculating elements from a full Cowell integrator.

Longitudinal Observable — Definition
-------------------------------------
The "mean longitude" used in this project is:

    λ_mean  =  Ω + ω + M          (all in radians)

where Ω = RAAN, ω = argument of perigee, M = mean anomaly, as extracted
from the public TLE / GP catalog.  This is wrapped to [0, 2π] for segment
endpoints and to (-π, π] for inter-epoch deltas.

This quantity is **not** the osculating true longitude.  It is a convenient
along-track proxy that captures secular drift from J2 and manoeuvre-induced
SMA changes, and is identifiable from consecutive TLE mean elements.

Approximation Regime
--------------------
All secular rates assume:
    - first-order J2 (no J3, J4, …),
    - mean elements (no short-period oscillations),
    - near-circular orbits (e ≲ 0.01 for Starlink Gen 1).

Provenance
----------
J2 secular rate formulas follow Brouwer (1959) / Vallado (2013) Ch. 9.

Constants
---------
All orbital constants are declared once here and re-exported.
Units are explicit in variable names or docstrings.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch

# ── Physical constants ───────────────────────────────────────────────────────
MU_EARTH_KM3_S2: float = 398600.4418
"""Earth gravitational parameter  [km³ s⁻²]  (WGS-84)."""

R_EARTH_KM: float = 6378.1366
"""Earth mean equatorial radius  [km]  (WGS-84)."""

J2_EARTH: float = 1.0826359e-3
"""Earth J2 zonal harmonic  [–]  (WGS-84)."""

G0_M_S2: float = 9.80665
"""Standard gravitational acceleration  [m s⁻²]  (ISO 80000-3)."""

TWOPI: float = 2.0 * math.pi

K_BOLTZMANN_J_K: float = 1.380649e-23
"""Boltzmann constant  [J K⁻¹]  (exact, 2019 SI)."""

KR_MASS_KG: float = 83.798 * 1.66053906660e-27
"""Krypton atomic mass  [kg] (83.798 u)."""

COLLOCATION_TAU_POINTS = (0.25, 0.50, 0.75)

RUN_SCHEMA_VERSION = "2026.04-stage-ab-r3"
"""Pipeline schema tag, bumped whenever the forward model changes."""


# ── Angle wrapping ───────────────────────────────────────────────────────────
def wrap_to_pi(angle_rad: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Wrap angle to  (-π, π].  Works element-wise on arrays."""
    return (np.asarray(angle_rad) + np.pi) % TWOPI - np.pi


def wrap_to_2pi(angle_rad: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Wrap angle to  [0, 2π).  Works element-wise on arrays."""
    return np.asarray(angle_rad) % TWOPI


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Gradient-safe wrap to (-π, π] via atan2(sin, cos)."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def angle_residual(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Signed angular residual in (-π, π] avoiding branch cuts."""
    return torch.atan2(torch.sin(pred - obs), torch.cos(pred - obs))


def deg2rad(x) -> np.ndarray:
    return np.deg2rad(np.asarray(x, dtype=np.float64))


# ── Mean motion ──────────────────────────────────────────────────────────────
def mean_motion_rad_s(a_km: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Keplerian mean motion  n = √(μ/a³)  [rad s⁻¹]."""
    a_km = np.asarray(a_km, dtype=np.float64)
    return np.sqrt(MU_EARTH_KM3_S2 / np.maximum(a_km, 1.0) ** 3)


# ── J2 secular rates (NumPy / float) ────────────────────────────────────────
# Provenance:  Vallado (2013) Eq. 9-41 .. 9-43,  Brouwer (1959).

def raan_rate_j2_rad_s(
    a_km: Union[np.ndarray, float],
    e: Union[np.ndarray, float],
    inc_rad: Union[np.ndarray, float],
) -> Union[np.ndarray, float]:
    r"""First-order J2 secular rate of RAAN:

    .. math::
        \dot{\Omega}_{J2} = -\frac{3}{2}\,J_2\,
            \left(\frac{R_E}{a}\right)^2\,
            \frac{n\,\cos i}{(1-e^2)^2}

    Returns  [rad s⁻¹].
    """
    a_km = np.asarray(a_km, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    inc_rad = np.asarray(inc_rad, dtype=np.float64)
    n = mean_motion_rad_s(a_km)
    eta2 = np.maximum((1.0 - e ** 2), 1.0e-12) ** 2
    return -1.5 * J2_EARTH * (R_EARTH_KM / np.maximum(a_km, 1.0)) ** 2 * n * np.cos(inc_rad) / eta2


def omega_dot_j2_rad_s(
    a_km: Union[np.ndarray, float],
    e: Union[np.ndarray, float],
    inc_rad: Union[np.ndarray, float],
) -> Union[np.ndarray, float]:
    r"""First-order J2 secular rate of argument of perigee ω:

    .. math::
        \dot{\omega}_{J2} = +\frac{3}{2}\,J_2\,
            \left(\frac{R_E}{a}\right)^2\,
            \frac{n\,(2 - \tfrac{5}{2}\sin^2 i)}{(1-e^2)^2}

    For Starlink (i ≈ 53°), sin²i ≈ 0.64, so ω̇ > 0 (prograde apsidal advance).

    Returns  [rad s⁻¹].
    """
    a_km = np.asarray(a_km, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    inc_rad = np.asarray(inc_rad, dtype=np.float64)
    n = mean_motion_rad_s(a_km)
    eta2 = np.maximum((1.0 - e ** 2), 1.0e-12) ** 2
    return 1.5 * J2_EARTH * (R_EARTH_KM / np.maximum(a_km, 1.0)) ** 2 * n * (2.0 - 2.5 * np.sin(inc_rad) ** 2) / eta2


def M_dot_j2_rad_s(
    a_km: Union[np.ndarray, float],
    e: Union[np.ndarray, float],
    inc_rad: Union[np.ndarray, float],
) -> Union[np.ndarray, float]:
    r"""First-order J2-perturbed mean anomaly rate:

    .. math::
        \dot{M}_{J2} = n\,\left[1 + \frac{3}{2}\,J_2\,
            \left(\frac{R_E}{a}\right)^2\,
            \frac{1 - \tfrac{3}{2}\sin^2 i}{(1-e^2)^{3/2}}\right]

    Returns  [rad s⁻¹].
    """
    a_km = np.asarray(a_km, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    inc_rad = np.asarray(inc_rad, dtype=np.float64)
    n = mean_motion_rad_s(a_km)
    eta32 = np.maximum((1.0 - e ** 2), 1.0e-12) ** 1.5
    return n * (1.0 + 1.5 * J2_EARTH * (R_EARTH_KM / np.maximum(a_km, 1.0)) ** 2 * (1.0 - 1.5 * np.sin(inc_rad) ** 2) / eta32)


def lambda_dot_j2_rad_s(
    a_km: Union[np.ndarray, float],
    e: Union[np.ndarray, float],
    inc_rad: Union[np.ndarray, float],
) -> Union[np.ndarray, float]:
    r"""Secular mean-longitude rate  λ̇ = Ω̇ + ω̇ + Ṁ  under first-order J2.

    This is the total secular drift of λ_mean = Ω + ω + M as used
    throughout the pipeline.

    Returns  [rad s⁻¹].
    """
    return (
        raan_rate_j2_rad_s(a_km, e, inc_rad)
        + omega_dot_j2_rad_s(a_km, e, inc_rad)
        + M_dot_j2_rad_s(a_km, e, inc_rad)
    )


# ── J2 secular rates (Torch, differentiable) ────────────────────────────────
# Mirrors the NumPy versions above for use inside the Stage A forward pass
# and Stage B simulator where autograd is required.

def raan_rate_j2_torch(
    a_km: torch.Tensor,
    e: torch.Tensor,
    inc_rad: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """Differentiable J2 RAAN rate [rad s⁻¹]."""
    n = torch.sqrt(mu / (a_km ** 3))
    eta2 = torch.clamp(1.0 - e ** 2, min=1.0e-6) ** 2
    return -1.5 * J2_EARTH * (R_EARTH_KM ** 2) * n * torch.cos(inc_rad) / ((a_km ** 2) * eta2)


def omega_dot_j2_torch(
    a_km: torch.Tensor,
    e: torch.Tensor,
    inc_rad: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """Differentiable J2 argument-of-perigee rate [rad s⁻¹].
    Provenance: Vallado (2013) Eq. 9-42."""
    n = torch.sqrt(mu / (a_km ** 3))
    eta2 = torch.clamp(1.0 - e ** 2, min=1.0e-6) ** 2
    return 1.5 * J2_EARTH * (R_EARTH_KM ** 2) * n * (2.0 - 2.5 * torch.sin(inc_rad) ** 2) / ((a_km ** 2) * eta2)


def M_dot_j2_torch(
    a_km: torch.Tensor,
    e: torch.Tensor,
    inc_rad: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """Differentiable J2-perturbed mean-anomaly rate [rad s⁻¹].
    Provenance: Vallado (2013) Eq. 9-43."""
    n = torch.sqrt(mu / (a_km ** 3))
    eta32 = torch.clamp(1.0 - e ** 2, min=1.0e-6) ** 1.5
    return n * (1.0 + 1.5 * J2_EARTH * (R_EARTH_KM ** 2) * (1.0 - 1.5 * torch.sin(inc_rad) ** 2) / ((a_km ** 2) * eta32))


def lambda_dot_j2_torch(
    a_km: torch.Tensor,
    e: torch.Tensor,
    inc_rad: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    r"""Differentiable secular mean-longitude rate λ̇ = Ω̇ + ω̇ + Ṁ.

    This is the primary reduced-model propagation kernel for the
    longitudinal observable used in Stage A and Stage B.

    Parameters
    ----------
    a_km : semi-major axis [km]
    e    : eccentricity [–]
    inc_rad : inclination [rad]
    mu   : gravitational parameter tensor (same device / dtype)

    Returns
    -------
    λ̇  [rad s⁻¹]
    """
    return (
        raan_rate_j2_torch(a_km, e, inc_rad, mu)
        + omega_dot_j2_torch(a_km, e, inc_rad, mu)
        + M_dot_j2_torch(a_km, e, inc_rad, mu)
    )
