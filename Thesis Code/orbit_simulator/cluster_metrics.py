"""Cluster-level metrics for Chapter 7 optimization.

Converts raw simulator output dictionaries into the five local objective
terms (J_a, J_lambda, J_m, J_d, J_s), reduced-coordinate envelopes,
and inter-cluster coupling penalties.

Reduced orbital coordinates used throughout:
    z_i(t) = [a_i(t),  Omega_i(t),  lambda_i(t)]
where lambda_i = Omega_i + omega_i + M_i  (mean longitude, rad).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from optimization_config import ObjectiveWeights, ClusterPolicy

# Earth constants (must match simulator)
_EARTH_GM = 398600.4418  # km^3/s^2
_EARTH_RE = 6378.1366    # km


# ======================================================================
# Orbital time-series extraction
# ======================================================================

def _xyz2orb_py(mu: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pure-Python xyz→orbital-elements (6,).

    Returns [a, e, i, omega, Omega, M] in km / radians.
    Avoids importing the Numba version so this module stays lightweight.
    """
    rv = r
    vv = v
    r_mag = np.linalg.norm(rv)
    v_mag = np.linalg.norm(vv)

    h = np.cross(rv, vv)
    h_mag = np.linalg.norm(h)

    # node vector
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    n_mag = np.linalg.norm(n)

    # eccentricity vector
    e_vec = ((v_mag**2 - mu / r_mag) * rv - np.dot(rv, vv) * vv) / mu
    ecc = np.linalg.norm(e_vec)

    # specific orbital energy
    energy = v_mag**2 / 2.0 - mu / r_mag
    if abs(ecc - 1.0) > 1e-12:
        a = -mu / (2.0 * energy)
    else:
        a = np.inf

    # inclination
    inc = np.arccos(np.clip(h[2] / max(h_mag, 1e-30), -1.0, 1.0))

    # RAAN
    if n_mag > 1e-12:
        Omega = np.arccos(np.clip(n[0] / n_mag, -1.0, 1.0))
        if n[1] < 0.0:
            Omega = 2.0 * np.pi - Omega
    else:
        Omega = 0.0

    # argument of perigee
    if n_mag > 1e-12 and ecc > 1e-12:
        omega = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * ecc), -1.0, 1.0))
        if e_vec[2] < 0.0:
            omega = 2.0 * np.pi - omega
    else:
        omega = 0.0

    # true anomaly
    if ecc > 1e-12:
        nu = np.arccos(np.clip(np.dot(e_vec, rv) / (ecc * r_mag), -1.0, 1.0))
        if np.dot(rv, vv) < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        nu = 0.0

    # eccentric anomaly → mean anomaly
    E = 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * np.sin(nu / 2.0),
                          np.sqrt(1.0 + ecc) * np.cos(nu / 2.0))
    M = E - ecc * np.sin(E)
    if M < 0.0:
        M += 2.0 * np.pi

    return np.array([a, ecc, inc, omega, Omega, M], dtype=np.float64)


def extract_orbital_timeseries(
    result: dict,
    checkpoint_stride: int = 1,
    earth_gm: float = _EARTH_GM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract reduced orbital coordinates at checkpoints.

    Parameters
    ----------
    result : dict
        Output from ``_run_one_batch_case`` / ``run_batch_cases``.
        Must contain ``times`` and ``state_sat`` (6×N).
    checkpoint_stride : int
        Sample every N-th time step.
    earth_gm : float
        Central body GM.

    Returns
    -------
    times_ck : ndarray (M,)
        Checkpoint times in seconds.
    z_ck : ndarray (M, 3)
        Reduced coordinates [a, Omega, lambda] at each checkpoint.
        lambda = Omega + omega + M   (mean longitude, rad).
    """
    times = result["times"]
    state_sat = result["state_sat"]  # (6, N)

    idxs = np.arange(0, times.size, checkpoint_stride)
    if idxs[-1] != times.size - 1:
        idxs = np.append(idxs, times.size - 1)

    M = idxs.size
    times_ck = times[idxs]
    z_ck = np.zeros((M, 3), dtype=np.float64)

    for j, idx in enumerate(idxs):
        r = state_sat[0:3, idx]
        v = state_sat[3:6, idx]
        oe = _xyz2orb_py(earth_gm, r, v)
        a = oe[0]
        Omega = oe[4]  # RAAN
        omega = oe[3]  # argument of perigee
        Ma = oe[5]     # mean anomaly
        lam = (Omega + omega + Ma) % (2.0 * np.pi)
        z_ck[j] = [a, Omega, lam]

    return times_ck, z_ck


def extract_full_orbital_elements(
    result: dict,
    checkpoint_stride: int = 1,
    earth_gm: float = _EARTH_GM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract full 6-element orbital-element series at checkpoints.

    Parameters
    ----------
    result : dict
        Simulator output with ``times`` and ``state_sat`` (6×N).
    checkpoint_stride : int
        Sample every N-th time step.
    earth_gm : float
        Central body GM.

    Returns
    -------
    times_ck : ndarray (M,)
        Checkpoint times in seconds.
    oe_ck : ndarray (M, 6)
        [a(km), ecc, inc(rad), omega(rad), Omega(rad), M(rad)]
    """
    times = result["times"]
    state_sat = result["state_sat"]  # (6, N)

    idxs = np.arange(0, times.size, checkpoint_stride)
    if idxs[-1] != times.size - 1:
        idxs = np.append(idxs, times.size - 1)

    K = idxs.size
    times_ck = times[idxs]
    oe_ck = np.zeros((K, 6), dtype=np.float64)

    for j, idx in enumerate(idxs):
        r = state_sat[0:3, idx]
        v = state_sat[3:6, idx]
        oe_ck[j] = _xyz2orb_py(earth_gm, r, v)

    return times_ck, oe_ck


# ======================================================================
# Individual objective terms
# ======================================================================

def compute_sma_tracking_loss(
    z_series_list: List[np.ndarray],
    a_ref: float,
    sigma_a: float = 5.0,
) -> float:
    """J_a: Mean squared SMA tracking error across representatives and time.

    Parameters
    ----------
    z_series_list : list of ndarray (M, 3)
        Reduced coordinates per representative.
    a_ref : float
        Reference (target) SMA for this cluster (km).
    sigma_a : float
        Normalisation denominator (km).
    """
    if not z_series_list:
        return 0.0
    total = 0.0
    count = 0
    for z in z_series_list:
        da = z[:, 0] - a_ref
        total += float(np.sum((da / sigma_a) ** 2))
        count += z.shape[0]
    return total / max(count, 1)


def compute_phase_coherence_loss(
    z_series_list: List[np.ndarray],
    lambda_ref: float,
    sigma_lambda: float = 0.05,
) -> float:
    """J_lambda: Mean squared mean-longitude drift across reps and time.

    Uses circular (angular) difference to handle wrapping.
    """
    if not z_series_list:
        return 0.0
    total = 0.0
    count = 0
    for z in z_series_list:
        dlam = np.arctan2(np.sin(z[:, 2] - lambda_ref),
                          np.cos(z[:, 2] - lambda_ref))
        total += float(np.sum((dlam / sigma_lambda) ** 2))
        count += z.shape[0]
    return total / max(count, 1)


def compute_propellant_loss(results: List[dict]) -> float:
    """J_m: Mean propellant usage across representatives (kg)."""
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        dm = r.get("initial_mass_kg", 0.0) - r.get("final_mass_kg", 0.0)
        total += max(dm, 0.0)
    return total / len(results)


def compute_disposal_penalty(
    results: List[dict],
    z_series_list: List[np.ndarray],
    h_min_required: float = 120.0,
    target_disposal_time_s: float = 0.0,
) -> float:
    """J_d: Disposal / end-of-life penalty.

    Uses the final altitude.  If the satellite terminated at the cutoff
    altitude, we check how far above ``h_min_required`` the cluster ended.
    """
    if not results:
        return 0.0
    earth_Re = _EARTH_RE
    total = 0.0
    for r in results:
        # Final altitude
        fx, fy, fz = r.get("final_x_km", 0.0), r.get("final_y_km", 0.0), r.get("final_z_km", 0.0)
        h_final = np.sqrt(fx**2 + fy**2 + fz**2) - earth_Re

        # Penalty: if satellite is still above h_min at end of sim
        shortfall = h_min_required - h_final
        # We penalise when disposal is NOT achieved (satellite stuck too high
        # or too low to meet timing targets).  For now, a simple quadratic on
        # the shortfall:
        if target_disposal_time_s > 0.0:
            t_term = r.get("t_115_s", -1.0)
            if t_term < 0.0:
                # Never reached disposal altitude — penalise
                total += (h_final - h_min_required) ** 2
            else:
                total += max(0.0, t_term - target_disposal_time_s) ** 2 * 1e-10
        else:
            # No timing target — penalise if satellite never reaches disposal
            if r.get("terminated_at_115km", 0) == 0:
                # Still orbiting — small penalty proportional to excess altitude
                excess = max(0.0, h_final - h_min_required)
                total += excess ** 2 * 1e-4

    return total / max(len(results), 1)


def compute_cluster_spread(
    z_series_list: List[np.ndarray],
) -> float:
    """J_s: Mean trace of cluster covariance in reduced coordinates.

    Sigma_g(t_m) is computed over the representative pack at each
    checkpoint.  The penalty is the time-averaged trace across all
    checkpoints.
    """
    if len(z_series_list) <= 1:
        return 0.0

    # Use shortest series (a rep may terminate early on decay)
    M = min(z.shape[0] for z in z_series_list)
    R = len(z_series_list)

    total_trace = 0.0
    for m in range(M):
        z_stack = np.array([z_series_list[r][m] for r in range(R)])  # (R, 3)
        # Handle angular wrapping for Omega and lambda (cols 1, 2)
        for col in [1, 2]:
            angles = z_stack[:, col]
            ref = angles[0]
            z_stack[:, col] = np.arctan2(np.sin(angles - ref),
                                         np.cos(angles - ref)) + ref
        cov = np.cov(z_stack.T)  # (3, 3)
        total_trace += float(np.trace(cov))

    return total_trace / max(M, 1)


# ======================================================================
# Combined objective
# ======================================================================

@dataclass
class ClusterObjectiveResult:
    """Structured result from a cluster objective evaluation."""
    J_total: float = 0.0
    J_a: float = 0.0
    J_lambda: float = 0.0
    J_m: float = 0.0
    J_d: float = 0.0
    J_s: float = 0.0
    n_representatives: int = 0
    n_checkpoints: int = 0

    def to_dict(self) -> dict:
        return {
            "J_total": self.J_total,
            "J_a": self.J_a,
            "J_lambda": self.J_lambda,
            "J_m": self.J_m,
            "J_d": self.J_d,
            "J_s": self.J_s,
            "n_representatives": self.n_representatives,
            "n_checkpoints": self.n_checkpoints,
        }


def compute_cluster_objective_terms(
    results: List[dict],
    weights: ObjectiveWeights,
    a_ref: float,
    lambda_ref: float = 0.0,
    checkpoint_stride: int = 100,
) -> ClusterObjectiveResult:
    """Compute the full local cluster objective from simulator results.

    Parameters
    ----------
    results : list[dict]
        Simulator output dicts (must include ``times``, ``state_sat``).
    weights : ObjectiveWeights
        Scalarisation weights and normalisation parameters.
    a_ref : float
        Reference SMA for this cluster (km).
    lambda_ref : float
        Reference mean longitude (rad).
    checkpoint_stride : int
        Sample every N-th time step for metric computation.

    Returns
    -------
    ClusterObjectiveResult
    """
    # Extract reduced orbital time-series per representative
    z_list: list[np.ndarray] = []
    times_list: list[np.ndarray] = []
    valid_results: list[dict] = []

    for r in results:
        if "times" not in r or "state_sat" not in r:
            continue
        if r["times"].size < 2:
            continue
        t_ck, z_ck = extract_orbital_timeseries(r, checkpoint_stride=checkpoint_stride)
        z_list.append(z_ck)
        times_list.append(t_ck)
        valid_results.append(r)

    if not z_list:
        return ClusterObjectiveResult()

    # Compute individual terms
    J_a = compute_sma_tracking_loss(z_list, a_ref, weights.sigma_a)
    J_lam = compute_phase_coherence_loss(z_list, lambda_ref, weights.sigma_lambda)
    J_m = compute_propellant_loss(valid_results)
    J_d = compute_disposal_penalty(valid_results, z_list,
                                    h_min_required=weights.h_min_required,
                                    target_disposal_time_s=weights.target_disposal_time_s)
    J_s = compute_cluster_spread(z_list)

    # Weighted total
    J_total = (weights.w_a * J_a
               + weights.w_lambda * J_lam
               + weights.w_m * J_m
               + weights.w_d * J_d
               + weights.w_s * J_s)

    n_ck = z_list[0].shape[0] if z_list else 0
    return ClusterObjectiveResult(
        J_total=J_total,
        J_a=J_a,
        J_lambda=J_lam,
        J_m=J_m,
        J_d=J_d,
        J_s=J_s,
        n_representatives=len(valid_results),
        n_checkpoints=n_ck,
    )


# ======================================================================
# Cluster envelope (for inter-cluster coupling)
# ======================================================================

@dataclass
class ClusterEnvelope:
    """Reduced-coordinate Gaussian envelope summary at checkpoints."""
    cluster_id: int = 0
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    mu: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))        # (M, 3)
    Sigma: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))  # (M, 3, 3)


def compute_cluster_envelope(
    z_series_list: List[np.ndarray],
    times_ck: np.ndarray,
    cluster_id: int = 0,
) -> ClusterEnvelope:
    """Compute mean and covariance envelope in reduced coordinates.

    Parameters
    ----------
    z_series_list : list of ndarray (M, 3)
        Reduced coordinates per representative at shared checkpoints.
    times_ck : ndarray (M,)
        Checkpoint times.
    cluster_id : int
        For labelling.

    Returns
    -------
    ClusterEnvelope
    """
    if not z_series_list:
        return ClusterEnvelope(cluster_id=cluster_id)

    # Use shortest series (a rep may terminate early on decay)
    M = min(z.shape[0] for z in z_series_list)
    R = len(z_series_list)

    mu = np.zeros((M, 3), dtype=np.float64)
    Sigma = np.zeros((M, 3, 3), dtype=np.float64)

    times_ck = times_ck[:M]

    for m in range(M):
        z_stack = np.array([z_series_list[r][m] for r in range(R)])
        # Handle angular wrapping
        for col in [1, 2]:
            angles = z_stack[:, col]
            ref = angles[0]
            z_stack[:, col] = np.arctan2(np.sin(angles - ref),
                                         np.cos(angles - ref)) + ref
        mu[m] = np.mean(z_stack, axis=0)
        if R > 1:
            Sigma[m] = np.cov(z_stack.T)
        else:
            # Regularization for single-representative envelopes.
            # Scale to coordinate units: SMA~km, Omega/lambda~rad.
            # Use representative uncertainty: ~5 km SMA, ~0.05 rad angles.
            Sigma[m] = np.diag([25.0, 0.0025, 0.0025])

    return ClusterEnvelope(
        cluster_id=cluster_id,
        times=times_ck,
        mu=mu,
        Sigma=Sigma,
    )


# ======================================================================
# Inter-cluster coupling penalty
# ======================================================================

def compute_intercluster_penalty(
    env_g: ClusterEnvelope,
    env_h: ClusterEnvelope,
    R_max: float = 0.01,
    eps: float = 1.0,
) -> float:
    """Psi_gh: overlap-risk proxy between two cluster envelopes.

    R_gh(t) = exp(-0.5 * (mu_g - mu_h)^T (Sigma_g + Sigma_h + eps*D)^{-1} (mu_g - mu_h))
    Psi_gh  = (1/M) sum_m max(0, R_gh(t_m) - R_max)^2

    eps scales a diagonal regularization matrix D = diag(1 km^2, 1e-4 rad^2, 1e-4 rad^2)
    to prevent singular covariance sums.
    """
    if env_g.mu.shape[0] == 0 or env_h.mu.shape[0] == 0:
        return 0.0

    # Align checkpoints to the shorter envelope
    M = min(env_g.mu.shape[0], env_h.mu.shape[0])

    total = 0.0
    # Regularization in coordinate-appropriate units
    D_reg = eps * np.diag([1.0, 1e-4, 1e-4])

    for m in range(M):
        diff = env_g.mu[m] - env_h.mu[m]
        # Handle angular wrapping for Omega and lambda
        for col in [1, 2]:
            diff[col] = np.arctan2(np.sin(diff[col]), np.cos(diff[col]))

        Sigma_sum = env_g.Sigma[m] + env_h.Sigma[m] + D_reg
        try:
            Sigma_inv = np.linalg.inv(Sigma_sum)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma_sum)

        maha_sq = float(diff @ Sigma_inv @ diff)
        R_gh = np.exp(-0.5 * maha_sq)

        excess = max(0.0, R_gh - R_max)
        total += excess ** 2

    return total / max(M, 1)
