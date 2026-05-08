"""Consolidated orbital mechanics utilities.

Core references:
- Curtis: universal variables and LVLH kinematics
- Vallado: foundational astrodynamics conventions
- Herrick / Shepperd: universal-variable propagation and state transition context
- Clohessy-Wiltshire and Yamanaka-Ankersen: relative-motion models

Frame notes:
- All two-body routines here assume a single inertial frame and do not perform
  frame transforms.
- SGP4 states from external modules are typically TEME and must be reconciled
  before mixing with non-TEME states.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    njit = None
    _HAS_NUMBA = False

from constants import MU_EARTH

_STUMPFF_SERIES_THRESHOLD = 1.0e-8
_EPS = 1.0e-12
_NUMBA_DISABLED_ENV = {"0", "false", "no", "off"}
_USE_NUMBA = _HAS_NUMBA and str(os.getenv("ORBITAL_MECHANICS_USE_NUMBA", "1")).strip().lower() not in _NUMBA_DISABLED_ENV


if _HAS_NUMBA:

    @njit(cache=True)
    def _stumpC_numba(z):
        abs_z = abs(z)
        if abs_z < _STUMPFF_SERIES_THRESHOLD:
            return 0.5 - z / 24.0 + (z**2) / 720.0 - (z**3) / 40320.0 + (z**4) / 3628800.0
        if z > 0.0:
            root = np.sqrt(z)
            return (1.0 - np.cos(root)) / z
        root = np.sqrt(-z)
        return (np.cosh(root) - 1.0) / (-z)


    @njit(cache=True)
    def _stumpS_numba(z):
        abs_z = abs(z)
        if abs_z < _STUMPFF_SERIES_THRESHOLD:
            return 1.0 / 6.0 - z / 120.0 + (z**2) / 5040.0 - (z**3) / 362880.0 + (z**4) / 39916800.0
        if z > 0.0:
            root = np.sqrt(z)
            return (root - np.sin(root)) / (root**3)
        root = np.sqrt(-z)
        return (np.sinh(root) - root) / (root**3)


def _wrap_to_2pi(angle_rad: float) -> float:
    return float(np.mod(angle_rad, 2.0 * np.pi))


def _angle_between(u_vec: np.ndarray, v_vec: np.ndarray, eps: float = _EPS) -> float:
    u_norm = float(np.linalg.norm(u_vec))
    v_norm = float(np.linalg.norm(v_vec))
    if u_norm <= eps or v_norm <= eps:
        return 0.0
    cos_val = float(np.dot(u_vec, v_vec) / (u_norm * v_norm))
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def stumpC(z: float) -> float:
    """Evaluate Stumpff C(z) robustly near z=0.

    Uses a short series expansion near zero to reduce subtractive cancellation.
    """
    z = float(z)
    if _USE_NUMBA and np.isfinite(z):
        return float(_stumpC_numba(z))

    abs_z = abs(z)
    if abs_z < _STUMPFF_SERIES_THRESHOLD:
        return float(0.5 - z / 24.0 + (z**2) / 720.0 - (z**3) / 40320.0 + (z**4) / 3628800.0)
    if z > 0.0:
        root = float(np.sqrt(z))
        return float((1.0 - np.cos(root)) / z)
    root = float(np.sqrt(-z))
    return float((np.cosh(root) - 1.0) / (-z))


def stumpS(z: float) -> float:
    """Evaluate Stumpff S(z) robustly near z=0.

    Uses a short series expansion near zero to reduce subtractive cancellation.
    """
    z = float(z)
    if _USE_NUMBA and np.isfinite(z):
        return float(_stumpS_numba(z))

    abs_z = abs(z)
    if abs_z < _STUMPFF_SERIES_THRESHOLD:
        return float(1.0 / 6.0 - z / 120.0 + (z**2) / 5040.0 - (z**3) / 362880.0 + (z**4) / 39916800.0)
    if z > 0.0:
        root = float(np.sqrt(z))
        return float((root - np.sin(root)) / (root**3))
    root = float(np.sqrt(-z))
    return float((np.sinh(root) - root) / (root**3))


def f_and_g(x: float, t: float, ro: float, a: float, mu: float = MU_EARTH) -> tuple[float, float]:
    """Compute Lagrange f and g coefficients for universal-variable propagation.

    Here ``a`` is ``alpha = 1 / a_orbit`` in universal-variable notation.
    """
    if ro <= 0.0:
        raise ValueError("ro must be positive")
    z = float(a) * float(x) ** 2
    f = 1.0 - (x**2 / ro) * stumpC(z)
    g = float(t) - (x**3 / np.sqrt(mu)) * stumpS(z)
    return float(f), float(g)


def fDot_and_gDot(x: float, r: float, ro: float, a: float, mu: float = MU_EARTH) -> tuple[float, float]:
    """Compute time derivatives of Lagrange f and g coefficients."""
    if r <= 0.0 or ro <= 0.0:
        raise ValueError("r and ro must be positive")
    z = float(a) * float(x) ** 2
    fdot = (np.sqrt(mu) / (r * ro)) * (z * stumpS(z) - 1.0) * x
    gdot = 1.0 - (x**2 / r) * stumpC(z)
    return float(fdot), float(gdot)


@dataclass(frozen=True)
class KeplerUDiagnostics:
    x: float
    converged: bool
    iterations: int
    final_residual: float
    final_step: float
    reason: str


def _kepler_U_core(
    dt: float,
    ro: float,
    vro: float,
    a: float,
    mu: float,
    tol: float = 1.0e-10,
    max_iter: int = 100,
    derivative_floor: float = 1.0e-14,
) -> KeplerUDiagnostics:
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if ro <= 0.0:
        raise ValueError("ro must be positive")

    sqrt_mu = float(np.sqrt(mu))
    dt = float(dt)
    a = float(a)
    vro = float(vro)

    if abs(a) > _EPS:
        x = sqrt_mu * abs(a) * dt
    else:
        # Parabolic-like initialization; keeps x finite near alpha -> 0.
        x = sqrt_mu * dt / ro

    converged = False
    reason = "max_iter"
    ratio = np.nan
    residual = np.nan
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration
        z = a * x**2
        c_val = stumpC(z)
        s_val = stumpS(z)
        residual = (
            (ro * vro / sqrt_mu) * x**2 * c_val
            + (1.0 - a * ro) * x**3 * s_val
            + ro * x
            - sqrt_mu * dt
        )
        dfdx = (
            (ro * vro / sqrt_mu) * x * (1.0 - z * s_val)
            + (1.0 - a * ro) * x**2 * c_val
            + ro
        )

        if not np.isfinite(residual) or not np.isfinite(dfdx):
            reason = "non_finite"
            break

        if abs(dfdx) < derivative_floor:
            reason = "small_derivative"
            break

        ratio = residual / dfdx
        x -= ratio

        if not np.isfinite(x):
            reason = "non_finite_state"
            break

        if abs(ratio) <= tol:
            converged = True
            reason = "converged"
            break

    return KeplerUDiagnostics(
        x=float(x),
        converged=bool(converged),
        iterations=int(iterations),
        final_residual=float(residual) if np.isfinite(residual) else np.nan,
        final_step=float(ratio) if np.isfinite(ratio) else np.nan,
        reason=reason,
    )


def kepler_U(dt: float, ro: float, vro: float, a: float, mu: float = MU_EARTH) -> float:
    """Solve universal Kepler equation via Newton iterations.

    Returns only ``x`` for backward compatibility.
    """
    return _kepler_U_core(dt, ro, vro, a, mu).x


def kepler_U_diagnostics(
    dt: float,
    ro: float,
    vro: float,
    a: float,
    mu: float = MU_EARTH,
    tol: float = 1.0e-10,
    max_iter: int = 100,
) -> dict[str, float | int | bool | str]:
    """Solve universal Kepler equation and return diagnostics metadata."""
    out = _kepler_U_core(dt, ro, vro, a, mu, tol=tol, max_iter=max_iter)
    return {
        "x": out.x,
        "converged": out.converged,
        "iterations": out.iterations,
        "final_residual": out.final_residual,
        "final_step": out.final_step,
        "reason": out.reason,
    }


def rv_from_r0v0(R0: np.ndarray, V0: np.ndarray, t: float, mu: float = MU_EARTH) -> tuple[np.ndarray, np.ndarray]:
    """Propagate state from initial position/velocity with two-body universal variables."""
    r0_vec = np.asarray(R0, dtype=np.float64)
    v0_vec = np.asarray(V0, dtype=np.float64)

    if r0_vec.shape != (3,) or v0_vec.shape != (3,):
        raise ValueError("R0 and V0 must each be 3-vectors")
    if mu <= 0.0:
        raise ValueError("mu must be positive")

    r0 = float(np.linalg.norm(r0_vec))
    v0 = float(np.linalg.norm(v0_vec))
    if r0 <= 0.0:
        raise ValueError("|R0| must be positive")

    vr0 = float(np.dot(r0_vec, v0_vec) / r0)
    alpha = float(2.0 / r0 - v0**2 / mu)

    x = kepler_U(float(t), r0, vr0, alpha, mu)
    f_val, g_val = f_and_g(x, float(t), r0, alpha, mu)
    r_vec = f_val * r0_vec + g_val * v0_vec
    r = float(np.linalg.norm(r_vec))

    fdot, gdot = fDot_and_gDot(x, r, r0, alpha, mu)
    v_vec = fdot * r0_vec + gdot * v0_vec
    return r_vec, v_vec


def rv_from_r0v0_diagnostics(
    R0: np.ndarray,
    V0: np.ndarray,
    t: float,
    mu: float = MU_EARTH,
    tol: float = 1.0e-10,
    max_iter: int = 100,
) -> dict[str, object]:
    """Propagate state and return convergence/provenance diagnostics."""
    r0_vec = np.asarray(R0, dtype=np.float64)
    v0_vec = np.asarray(V0, dtype=np.float64)
    r0 = float(np.linalg.norm(r0_vec))
    v0 = float(np.linalg.norm(v0_vec))
    vr0 = float(np.dot(r0_vec, v0_vec) / r0)
    alpha = float(2.0 / r0 - v0**2 / mu)

    kdiag = _kepler_U_core(float(t), r0, vr0, alpha, mu, tol=tol, max_iter=max_iter)
    f_val, g_val = f_and_g(kdiag.x, float(t), r0, alpha, mu)
    r_vec = f_val * r0_vec + g_val * v0_vec
    r = float(np.linalg.norm(r_vec))
    fdot, gdot = fDot_and_gDot(kdiag.x, r, r0, alpha, mu)
    v_vec = fdot * r0_vec + gdot * v0_vec

    return {
        "R": r_vec,
        "V": v_vec,
        "alpha": alpha,
        "r0": r0,
        "v0": v0,
        "vr0": vr0,
        "kepler": {
            "x": kdiag.x,
            "converged": kdiag.converged,
            "iterations": kdiag.iterations,
            "final_residual": kdiag.final_residual,
            "final_step": kdiag.final_step,
            "reason": kdiag.reason,
        },
    }


def rv_from_r0v0_many(
    R0: np.ndarray,
    V0: np.ndarray,
    times: np.ndarray,
    mu: float = MU_EARTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a state for many times using the same two-body model.

    Returns arrays with shape (N, 3).
    """
    t_arr = np.asarray(times, dtype=np.float64)
    if t_arr.ndim != 1:
        raise ValueError("times must be a 1D array")

    r_hist = np.empty((t_arr.size, 3), dtype=np.float64)
    v_hist = np.empty((t_arr.size, 3), dtype=np.float64)
    for idx, dt in enumerate(t_arr):
        r_hist[idx], v_hist[idx] = rv_from_r0v0(R0, V0, float(dt), mu)
    return r_hist, v_hist


def sv_from_coe(coe: tuple[float, float, float, float, float, float], mu: float = MU_EARTH) -> tuple[np.ndarray, np.ndarray]:
    """Compute inertial state from classical orbital elements.

    Input order: (h, e, raan, incl, argp, true_anomaly), with angles in radians.

    Notes on singular regimes:
    - Near e -> 0, argument of periapsis is ill-conditioned.
    - Near i -> 0, RAAN is ill-conditioned.
    The returned Cartesian state is still well-defined when values are finite.
    """
    if mu <= 0.0:
        raise ValueError("mu must be positive")

    h, e, raan, incl, argp, true_anomaly = [float(val) for val in coe]
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be finite and positive")
    if not np.isfinite(e) or e < 0.0:
        raise ValueError("eccentricity e must be finite and nonnegative")

    p = (h**2) / mu
    cos_nu = float(np.cos(true_anomaly))
    sin_nu = float(np.sin(true_anomaly))
    denom = 1.0 + e * cos_nu
    if abs(denom) <= _EPS:
        raise ValueError("singular conic geometry: 1 + e*cos(true_anomaly) is too small")

    r = p / denom
    rf_vec = np.array([r * cos_nu, r * sin_nu, 0.0], dtype=np.float64)
    factor = mu / h
    vf_vec = np.array([-factor * sin_nu, factor * (e + cos_nu), 0.0], dtype=np.float64)

    cos_w = float(np.cos(argp))
    sin_w = float(np.sin(argp))
    cos_o = float(np.cos(raan))
    sin_o = float(np.sin(raan))
    cos_i = float(np.cos(incl))
    sin_i = float(np.sin(incl))

    rot = np.array(
        [
            [cos_o * cos_w - sin_o * sin_w * cos_i, -cos_o * sin_w - sin_o * cos_w * cos_i, sin_o * sin_i],
            [sin_o * cos_w + cos_o * sin_w * cos_i, -sin_o * sin_w + cos_o * cos_w * cos_i, -cos_o * sin_i],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ],
        dtype=np.float64,
    )

    r_vec = rot @ rf_vec
    v_vec = rot @ vf_vec
    return r_vec, v_vec


def specific_energy(r_vec: np.ndarray, v_vec: np.ndarray, mu: float = MU_EARTH) -> float:
    """Return specific orbital energy, km^2/s^2."""
    r_norm = float(np.linalg.norm(r_vec))
    if r_norm <= 0.0:
        raise ValueError("|r| must be positive")
    return float(0.5 * np.dot(v_vec, v_vec) - mu / r_norm)


def specific_angular_momentum_vector(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Return specific angular momentum vector, km^2/s."""
    return np.cross(np.asarray(r_vec, dtype=np.float64), np.asarray(v_vec, dtype=np.float64))


def eccentricity_vector(r_vec: np.ndarray, v_vec: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """Return eccentricity vector in the same inertial frame as r/v."""
    r_vec = np.asarray(r_vec, dtype=np.float64)
    v_vec = np.asarray(v_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r_vec))
    if r_norm <= 0.0:
        raise ValueError("|r| must be positive")
    h_vec = specific_angular_momentum_vector(r_vec, v_vec)
    return np.cross(v_vec, h_vec) / mu - r_vec / r_norm


def coe_from_sv(r_vec: np.ndarray, v_vec: np.ndarray, mu: float = MU_EARTH) -> tuple[float, float, float, float, float, float]:
    """Compute classical orbital elements from an inertial state.

    Returns: (h, e, raan, incl, argp, true_anomaly), angles in radians.
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    v_vec = np.asarray(v_vec, dtype=np.float64)
    if r_vec.shape != (3,) or v_vec.shape != (3,):
        raise ValueError("r_vec and v_vec must be 3-vectors")

    r_norm = float(np.linalg.norm(r_vec))
    v_norm = float(np.linalg.norm(v_vec))
    if r_norm <= 0.0:
        raise ValueError("|r| must be positive")

    h_vec = specific_angular_momentum_vector(r_vec, v_vec)
    h = float(np.linalg.norm(h_vec))
    if h <= _EPS:
        raise ValueError("specific angular momentum is too small for COE conversion")

    k_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_vec = np.cross(k_hat, h_vec)
    n = float(np.linalg.norm(n_vec))

    e_vec = eccentricity_vector(r_vec, v_vec, mu)
    e = float(np.linalg.norm(e_vec))

    incl = float(np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0)))

    if n > _EPS:
        raan = float(np.arctan2(n_vec[1], n_vec[0]))
    else:
        raan = 0.0
    raan = _wrap_to_2pi(raan)

    if e > 1.0e-10 and n > _EPS:
        argp = _angle_between(n_vec, e_vec)
        if e_vec[2] < 0.0:
            argp = 2.0 * np.pi - argp
    else:
        argp = 0.0
    argp = _wrap_to_2pi(argp)

    if e > 1.0e-10:
        nu = _angle_between(e_vec, r_vec)
        if float(np.dot(r_vec, v_vec)) < 0.0:
            nu = 2.0 * np.pi - nu
    elif n > _EPS:
        # Circular inclined case: use argument of latitude.
        nu = _angle_between(n_vec, r_vec)
        if r_vec[2] < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        # Circular equatorial case: use true longitude.
        nu = float(np.arctan2(r_vec[1], r_vec[0]))
    nu = _wrap_to_2pi(float(nu))

    return h, e, raan, incl, argp, nu


def _lvlh_basis(r_chief: np.ndarray, v_chief: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_chief = np.asarray(r_chief, dtype=np.float64)
    v_chief = np.asarray(v_chief, dtype=np.float64)

    r_norm = float(np.linalg.norm(r_chief))
    if r_norm <= 0.0:
        raise ValueError("chief position norm must be positive")

    h_chief = np.cross(r_chief, v_chief)
    h_norm = float(np.linalg.norm(h_chief))
    if h_norm <= _EPS:
        raise ValueError("chief angular momentum norm is too small for LVLH basis")

    e_r = r_chief / r_norm
    e_h = h_chief / h_norm
    e_theta = np.cross(e_h, e_r)
    dcm_i_to_lvlh = np.vstack((e_r, e_theta, e_h))
    return e_r, e_theta, e_h, dcm_i_to_lvlh


def rva_relative(
    rA: np.ndarray,
    vA: np.ndarray,
    rB: np.ndarray,
    vB: np.ndarray,
    mu: float = MU_EARTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute deputy-relative state in chief LVLH coordinates.

    This is exact chief-centered nonlinear LVLH kinematics (not HCW).
    Includes transport-rate terms via omega and omega_dot.
    """
    e_r, e_theta, e_h, dcm_i_to_lvlh = _lvlh_basis(rA, vA)
    _ = e_theta  # explicit for readability in basis return wrapper.

    rA = np.asarray(rA, dtype=np.float64)
    vA = np.asarray(vA, dtype=np.float64)
    rB = np.asarray(rB, dtype=np.float64)
    vB = np.asarray(vB, dtype=np.float64)

    hA = np.cross(rA, vA)
    rA_norm = float(np.linalg.norm(rA))
    omega = hA / (rA_norm**2)
    omega_dot = -2.0 * float(np.dot(rA, vA)) / (rA_norm**2) * omega

    aA = -mu * rA / (np.linalg.norm(rA) ** 3)
    aB = -mu * rB / (np.linalg.norm(rB) ** 3)

    rho_i = rB - rA
    rho_dot_i = vB - vA
    rho_dot_lvlh_i = rho_dot_i - np.cross(omega, rho_i)
    rho_ddot_lvlh_i = (
        (aB - aA)
        - np.cross(omega_dot, rho_i)
        - np.cross(omega, np.cross(omega, rho_i))
        - 2.0 * np.cross(omega, rho_dot_lvlh_i)
    )

    rho_lvlh = dcm_i_to_lvlh @ rho_i
    rho_dot_lvlh = dcm_i_to_lvlh @ rho_dot_lvlh_i
    rho_ddot_lvlh = dcm_i_to_lvlh @ rho_ddot_lvlh_i
    return rho_lvlh, rho_dot_lvlh, rho_ddot_lvlh


def rva_relative_with_basis(
    rA: np.ndarray,
    vA: np.ndarray,
    rB: np.ndarray,
    vB: np.ndarray,
    mu: float = MU_EARTH,
) -> dict[str, object]:
    """Compute relative LVLH kinematics and return chief basis metadata."""
    e_r, e_theta, e_h, dcm_i_to_lvlh = _lvlh_basis(rA, vA)
    rho_lvlh, rho_dot_lvlh, rho_ddot_lvlh = rva_relative(rA, vA, rB, vB, mu)

    hA = np.cross(np.asarray(rA, dtype=np.float64), np.asarray(vA, dtype=np.float64))
    rA_norm = float(np.linalg.norm(rA))
    omega = hA / (rA_norm**2)

    return {
        "r_rel_lvlh": rho_lvlh,
        "v_rel_lvlh": rho_dot_lvlh,
        "a_rel_lvlh": rho_ddot_lvlh,
        "basis": {
            "e_r": e_r,
            "e_theta": e_theta,
            "e_h": e_h,
            "dcm_i_to_lvlh": dcm_i_to_lvlh,
        },
        "omega_lvlh_inertial": omega,
        "chief_frame_label": "chief_centered_lvlh",
    }


def _sorted_time_array(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(times, dtype=np.float64)
    if t_arr.ndim != 1:
        raise ValueError("times must be 1D")
    if t_arr.size == 0:
        raise ValueError("times must be non-empty")
    if np.any(~np.isfinite(t_arr)):
        raise ValueError("times must be finite")
    order = np.argsort(t_arr)
    return t_arr[order], order


def relative_motion_exact_lvlh(
    chief_r0: np.ndarray,
    chief_v0: np.ndarray,
    deputy_r0: np.ndarray,
    deputy_v0: np.ndarray,
    times: np.ndarray,
    mu: float = MU_EARTH,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | dict[str, object]:
    """Exact nonlinear chief-centered LVLH relative motion using two-body propagation."""
    t_sorted, order = _sorted_time_array(times)
    rho_hist = np.empty((t_sorted.size, 3), dtype=np.float64)
    rhodot_hist = np.empty((t_sorted.size, 3), dtype=np.float64)
    rhoddot_hist = np.empty((t_sorted.size, 3), dtype=np.float64)

    for idx, dt in enumerate(t_sorted):
        r_c, v_c = rv_from_r0v0(chief_r0, chief_v0, float(dt), mu)
        r_d, v_d = rv_from_r0v0(deputy_r0, deputy_v0, float(dt), mu)
        rho, rhodot, rhoddot = rva_relative(r_c, v_c, r_d, v_d, mu)
        rho_hist[idx] = rho
        rhodot_hist[idx] = rhodot
        rhoddot_hist[idx] = rhoddot

    inv = np.argsort(order)
    rho_hist = rho_hist[inv]
    rhodot_hist = rhodot_hist[inv]
    rhoddot_hist = rhoddot_hist[inv]

    if not return_diagnostics:
        return rho_hist, rhodot_hist, rhoddot_hist
    return {
        "r_rel_lvlh": rho_hist,
        "v_rel_lvlh": rhodot_hist,
        "a_rel_lvlh": rhoddot_hist,
        "model": "exact_lvlh",
        "assumptions": {
            "chief_centered_nonlinear": True,
            "two_body": True,
            "perturbations_included": False,
        },
    }


def relative_motion_hcw(
    chief_r0: np.ndarray,
    chief_v0: np.ndarray,
    deputy_r0: np.ndarray,
    deputy_v0: np.ndarray,
    times: np.ndarray,
    mu: float = MU_EARTH,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | dict[str, object]:
    """HCW circular-chief linearized relative motion model in LVLH.

    Assumptions:
    - chief orbit is approximately circular
    - deputy separation is small relative to chief orbital radius
    - two-body central gravity, no perturbations
    """
    t_sorted, order = _sorted_time_array(times)
    rho0, rhodot0, _ = rva_relative(chief_r0, chief_v0, deputy_r0, deputy_v0, mu)

    rc_norm = float(np.linalg.norm(chief_r0))
    if rc_norm <= 0.0:
        raise ValueError("chief_r0 norm must be positive")
    n = float(np.sqrt(mu / (rc_norm**3)))

    x0, y0, z0 = rho0
    xdot0, ydot0, zdot0 = rhodot0

    rho_hist = np.empty((t_sorted.size, 3), dtype=np.float64)
    rhodot_hist = np.empty((t_sorted.size, 3), dtype=np.float64)
    rhoddot_hist = np.empty((t_sorted.size, 3), dtype=np.float64)

    for idx, t in enumerate(t_sorted):
        nt = n * float(t)
        s_nt = float(np.sin(nt))
        c_nt = float(np.cos(nt))

        x = (4.0 - 3.0 * c_nt) * x0 + (s_nt / n) * xdot0 + (2.0 / n) * (1.0 - c_nt) * ydot0
        y = (
            6.0 * (s_nt - nt) * x0
            + y0
            - (2.0 / n) * (1.0 - c_nt) * xdot0
            + (1.0 / n) * (4.0 * s_nt - 3.0 * nt) * ydot0
        )
        z = z0 * c_nt + (s_nt / n) * zdot0

        xdot = 3.0 * n * s_nt * x0 + c_nt * xdot0 + 2.0 * s_nt * ydot0
        ydot = 6.0 * n * (c_nt - 1.0) * x0 - 2.0 * s_nt * xdot0 + (4.0 * c_nt - 3.0) * ydot0
        zdot = -n * s_nt * z0 + c_nt * zdot0

        xddot = 2.0 * n * ydot + 3.0 * n**2 * x
        yddot = -2.0 * n * xdot
        zddot = -(n**2) * z

        rho_hist[idx] = np.array([x, y, z], dtype=np.float64)
        rhodot_hist[idx] = np.array([xdot, ydot, zdot], dtype=np.float64)
        rhoddot_hist[idx] = np.array([xddot, yddot, zddot], dtype=np.float64)

    inv = np.argsort(order)
    rho_hist = rho_hist[inv]
    rhodot_hist = rhodot_hist[inv]
    rhoddot_hist = rhoddot_hist[inv]

    if not return_diagnostics:
        return rho_hist, rhodot_hist, rhoddot_hist
    return {
        "r_rel_lvlh": rho_hist,
        "v_rel_lvlh": rhodot_hist,
        "a_rel_lvlh": rhoddot_hist,
        "model": "hcw",
        "assumptions": {
            "chief_near_circular": True,
            "small_separation": True,
            "two_body": True,
            "perturbations_included": False,
        },
        "mean_motion_rad_s": n,
    }


def _gravity_gradient_matrix(r_vec: np.ndarray, mu: float) -> np.ndarray:
    r = np.asarray(r_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 0.0:
        raise ValueError("|r| must be positive")
    identity = np.eye(3, dtype=np.float64)
    outer_rr = np.outer(r, r)
    return -mu * (identity / (r_norm**3) - 3.0 * outer_rr / (r_norm**5))


def _rk4_step_variational(
    y: np.ndarray,
    dt: float,
    chief_state_at: callable,
    t0: float,
    mu: float,
) -> np.ndarray:
    def dyn(t_now: float, state: np.ndarray) -> np.ndarray:
        r_chief, _ = chief_state_at(t_now)
        a_mat = _gravity_gradient_matrix(r_chief, mu)
        dr = state[3:]
        dv = a_mat @ state[:3]
        return np.hstack((dr, dv))

    k1 = dyn(t0, y)
    k2 = dyn(t0 + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = dyn(t0 + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = dyn(t0 + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_step_state_transition(
    phi: np.ndarray,
    dt: float,
    chief_state_at: callable,
    t0: float,
    mu: float,
) -> np.ndarray:
    def dyn(t_now: float, phi_now: np.ndarray) -> np.ndarray:
        r_chief, _ = chief_state_at(t_now)
        a_mat = _gravity_gradient_matrix(r_chief, mu)
        sys = np.block(
            [
                [np.zeros((3, 3), dtype=np.float64), np.eye(3, dtype=np.float64)],
                [a_mat, np.zeros((3, 3), dtype=np.float64)],
            ]
        )
        return sys @ phi_now

    k1 = dyn(t0, phi)
    k2 = dyn(t0 + 0.5 * dt, phi + 0.5 * dt * k1)
    k3 = dyn(t0 + 0.5 * dt, phi + 0.5 * dt * k2)
    k4 = dyn(t0 + dt, phi + dt * k3)
    return phi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def relative_motion_yamanaka_ankersen(
    chief_r0: np.ndarray,
    chief_v0: np.ndarray,
    deputy_r0: np.ndarray,
    deputy_v0: np.ndarray,
    times: np.ndarray,
    mu: float = MU_EARTH,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | dict[str, object]:
    """Elliptical-chief linearized relative motion via TH/YA-style variational dynamics.

    This implementation is a practical linearized elliptical-chief STM interface:
    - chief is propagated with two-body dynamics
    - relative perturbation is propagated via a numerical 6x6 STM
    - output is reported in chief LVLH coordinates

    It is not the closed-form YA analytical STM expression, but follows the
    same linearized elliptical-chief intent and is valid for small separation.
    """
    t_sorted, order = _sorted_time_array(times)

    def chief_state_at(dt: float) -> tuple[np.ndarray, np.ndarray]:
        return rv_from_r0v0(chief_r0, chief_v0, float(dt), mu)

    delta_r0 = np.asarray(deputy_r0, dtype=np.float64) - np.asarray(chief_r0, dtype=np.float64)
    delta_v0 = np.asarray(deputy_v0, dtype=np.float64) - np.asarray(chief_v0, dtype=np.float64)
    y0 = np.hstack((delta_r0, delta_v0)).astype(np.float64)
    phi = np.eye(6, dtype=np.float64)

    rho_hist = np.empty((t_sorted.size, 3), dtype=np.float64)
    rhodot_hist = np.empty((t_sorted.size, 3), dtype=np.float64)
    rhoddot_hist = np.empty((t_sorted.size, 3), dtype=np.float64)

    prev_t = 0.0
    for idx, t_now in enumerate(t_sorted):
        dt = float(t_now - prev_t)
        if dt < 0.0:
            raise ValueError("times must be monotonically nondecreasing")
        if dt > 0.0:
            phi = _rk4_step_state_transition(phi, dt, chief_state_at, prev_t, mu)
        prev_t = float(t_now)

        y = phi @ y0

        r_chief, v_chief = chief_state_at(float(t_now))
        r_dep_approx = r_chief + y[:3]
        v_dep_approx = v_chief + y[3:]
        rho, rhodot, rhoddot = rva_relative(r_chief, v_chief, r_dep_approx, v_dep_approx, mu)
        rho_hist[idx] = rho
        rhodot_hist[idx] = rhodot
        rhoddot_hist[idx] = rhoddot

    inv = np.argsort(order)
    rho_hist = rho_hist[inv]
    rhodot_hist = rhodot_hist[inv]
    rhoddot_hist = rhoddot_hist[inv]

    if not return_diagnostics:
        return rho_hist, rhodot_hist, rhoddot_hist
    return {
        "r_rel_lvlh": rho_hist,
        "v_rel_lvlh": rhodot_hist,
        "a_rel_lvlh": rhoddot_hist,
        "model": "yamanaka_ankersen",
        "assumptions": {
            "elliptical_chief_linearized": True,
            "small_separation": True,
            "two_body": True,
            "perturbations_included": False,
            "implementation": "numerical_state_transition_matrix_th_ya_style",
        },
        "stm_based": True,
    }


def relative_motion_roe_linearized(
    rho_lvlh: np.ndarray,
    rhodot_lvlh: np.ndarray,
    semi_major_axis_km: float,
    mean_motion_rad_s: float,
) -> dict[str, float]:
    """Lightweight ROE scaffolding for formation-geometry extensions.

    This is an intentionally lightweight linearized mapping scaffold for
    near-circular formations; it is not a full Schaub-consistent ROE pipeline.
    """
    rho = np.asarray(rho_lvlh, dtype=np.float64)
    rhodot = np.asarray(rhodot_lvlh, dtype=np.float64)
    a = float(semi_major_axis_km)
    n = float(mean_motion_rad_s)
    if a <= 0.0 or n <= 0.0:
        raise ValueError("semi_major_axis_km and mean_motion_rad_s must be positive")

    da = 2.0 * rhodot[1] / n
    dlam = rho[1] / a + 2.0 * rhodot[0] / (n * a)
    dex = rho[0] / a
    dey = rhodot[0] / (n * a)
    dix = rho[2] / a
    diy = rhodot[2] / (n * a)

    return {
        "delta_a_km": float(da),
        "delta_lambda_rad": float(dlam),
        "delta_ex": float(dex),
        "delta_ey": float(dey),
        "delta_ix": float(dix),
        "delta_iy": float(diy),
        "scaffold_only": True,
    }


__all__ = [
    "stumpC",
    "stumpS",
    "f_and_g",
    "fDot_and_gDot",
    "kepler_U",
    "kepler_U_diagnostics",
    "rv_from_r0v0",
    "rv_from_r0v0_diagnostics",
    "rv_from_r0v0_many",
    "sv_from_coe",
    "coe_from_sv",
    "specific_energy",
    "specific_angular_momentum_vector",
    "eccentricity_vector",
    "rva_relative",
    "rva_relative_with_basis",
    "relative_motion_exact_lvlh",
    "relative_motion_hcw",
    "relative_motion_yamanaka_ankersen",
    "relative_motion_roe_linearized",
]