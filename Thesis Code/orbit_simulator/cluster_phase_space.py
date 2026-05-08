"""RAAN-vs-phase toroidal regularization for Chapter 7 optimization.

Implements:
    - Wrapped angular arithmetic on the torus (Omega, psi)
    - Configurable phase variable: mean anomaly, argument of latitude, mean longitude
    - Ideal torus lattice generation with inter-plane phasing
    - Linear-sum-assignment slot fitting
    - Circular gap-uniformity losses
    - Drift-coherence loss from propagated time series
    - Full torus regularizer: J_torus = w_slot*J_slot + w_gap*J_gap + w_drift*J_drift

All angles in this module are in DEGREES unless explicitly stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar


# ======================================================================
# Angular utility functions
# ======================================================================

def wrap_to_360(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [0, 360)."""
    return np.mod(x, 360.0)


def wrap_to_pm180(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-180, +180)."""
    return np.mod(x + 180.0, 360.0) - 180.0


def circular_distance_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Signed circular distance a - b wrapped to [-180, +180)."""
    return wrap_to_pm180(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def unwrap_angle_series_deg(theta: np.ndarray) -> np.ndarray:
    """Unwrap a degree-valued angle series for drift estimation.

    Converts to radians, unwraps, converts back.
    """
    return np.rad2deg(np.unwrap(np.deg2rad(theta)))


# ======================================================================
# Phase variable computation
# ======================================================================

_SUPPORTED_MODES = ("raan_mean_anomaly", "raan_argument_of_latitude", "raan_mean_longitude")


def compute_phase_variable_deg(
    oe_rad: np.ndarray,
    mode: str = "raan_mean_anomaly",
) -> float:
    """Compute phase variable psi in degrees from orbital elements.

    Parameters
    ----------
    oe_rad : ndarray (6,)
        [a(km), ecc, inc(rad), omega(rad), Omega(rad), M(rad)]
    mode : str
        One of ``raan_mean_anomaly``, ``raan_argument_of_latitude``,
        ``raan_mean_longitude``.

    Returns
    -------
    psi_deg : float
        Phase variable in [0, 360).
    """
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"Unsupported phase mode '{mode}'. Use one of {_SUPPORTED_MODES}")

    omega = oe_rad[3]  # argument of perigee (rad)
    M = oe_rad[5]      # mean anomaly (rad)
    Omega = oe_rad[4]  # RAAN (rad)

    if mode == "raan_mean_anomaly":
        psi_rad = M
    elif mode == "raan_argument_of_latitude":
        # u = omega + f; approximate f from M for near-circular orbits
        ecc = oe_rad[1]
        f = _mean_to_true_anomaly(M, ecc)
        psi_rad = omega + f
    elif mode == "raan_mean_longitude":
        psi_rad = Omega + omega + M
    else:
        psi_rad = M

    return float(wrap_to_360(np.rad2deg(psi_rad)))


def compute_raan_deg(oe_rad: np.ndarray) -> float:
    """Extract RAAN in degrees from orbital elements [a,e,i,w,Omega,M] (rad)."""
    return float(wrap_to_360(np.rad2deg(oe_rad[4])))


def _mean_to_true_anomaly(M: float, ecc: float, tol: float = 1e-12) -> float:
    """Solve Kepler's equation M → E → f (radians)."""
    if ecc < 1e-14:
        return M
    # Newton-Raphson for E
    E = M
    for _ in range(30):
        dE = (E - ecc * np.sin(E) - M) / (1.0 - ecc * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    f = 2.0 * np.arctan2(np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
                          np.sqrt(1.0 - ecc) * np.cos(E / 2.0))
    return f


def extract_raan_phase_timeseries_deg(
    result: dict,
    mode: str = "raan_mean_anomaly",
    checkpoint_stride: int = 1,
    earth_gm: float = 398600.4418,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract RAAN and phase time series from a simulator result dict.

    Parameters
    ----------
    result : dict
        Must contain ``times`` (M,) and ``state_sat`` (6, M).
    mode : str
        Phase variable mode.
    checkpoint_stride : int
        Sample every N-th step.
    earth_gm : float
        Central body GM.

    Returns
    -------
    times_days : ndarray (K,)
        Checkpoint times in days.
    raan_deg : ndarray (K,)
        RAAN at each checkpoint (degrees).
    phase_deg : ndarray (K,)
        Phase variable at each checkpoint (degrees).
    """
    from cluster_metrics import _xyz2orb_py

    times = result["times"]
    state = result["state_sat"]

    idxs = np.arange(0, times.size, checkpoint_stride)
    if idxs[-1] != times.size - 1:
        idxs = np.append(idxs, times.size - 1)

    K = idxs.size
    raan_deg = np.empty(K, dtype=np.float64)
    phase_deg = np.empty(K, dtype=np.float64)

    for j, idx in enumerate(idxs):
        oe = _xyz2orb_py(earth_gm, state[0:3, idx], state[3:6, idx])
        raan_deg[j] = compute_raan_deg(oe)
        phase_deg[j] = compute_phase_variable_deg(oe, mode=mode)

    times_days = times[idxs] / 86400.0
    return times_days, raan_deg, phase_deg


# ======================================================================
# Ideal torus lattice
# ======================================================================

@dataclass
class TorusLattice:
    """Target torus lattice for a shell or cluster family."""
    n_planes: int = 1
    slots_per_plane: int = 1
    omega0_deg: float = 0.0
    psi0_deg: float = 0.0
    eta: float = 0.0            # inter-plane phasing coefficient
    target_raan_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    target_phase_deg: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_slots(self) -> int:
        return self.n_planes * self.slots_per_plane

    def to_dict(self) -> dict:
        return {
            "n_planes": self.n_planes,
            "slots_per_plane": self.slots_per_plane,
            "omega0_deg": self.omega0_deg,
            "psi0_deg": self.psi0_deg,
            "eta": self.eta,
            "delta_omega_deg": 360.0 / max(self.n_planes, 1),
            "delta_psi_deg": 360.0 / max(self.slots_per_plane, 1),
        }


def build_torus_target_slots(
    n_planes: int,
    slots_per_plane: int,
    omega0_deg: float = 0.0,
    psi0_deg: float = 0.0,
    eta: float = 0.0,
    sparse_mask: Optional[np.ndarray] = None,
) -> TorusLattice:
    """Build ideal torus lattice (Omega*, psi*) for all slots.

    Parameters
    ----------
    n_planes : int
        Number of orbital planes.
    slots_per_plane : int
        Number of slots per plane.
    omega0_deg : float
        RAAN offset (reference for plane 0).
    psi0_deg : float
        Phase offset (reference for slot 0 in plane 0).
    eta : float
        Inter-plane phasing coefficient.
    sparse_mask : ndarray (n_planes, slots_per_plane) bool, optional
        True = occupied slot. None → full occupancy.

    Returns
    -------
    TorusLattice
    """
    delta_omega = 360.0 / max(n_planes, 1)
    delta_psi = 360.0 / max(slots_per_plane, 1)

    raan_list = []
    phase_list = []
    for p in range(n_planes):
        for s in range(slots_per_plane):
            if sparse_mask is not None and not sparse_mask[p, s]:
                continue
            raan_slot = wrap_to_360(omega0_deg + p * delta_omega)
            psi_slot = wrap_to_360(psi0_deg + s * delta_psi + eta * p * delta_psi)
            raan_list.append(raan_slot)
            phase_list.append(psi_slot)

    return TorusLattice(
        n_planes=n_planes,
        slots_per_plane=slots_per_plane,
        omega0_deg=omega0_deg,
        psi0_deg=psi0_deg,
        eta=eta,
        target_raan_deg=np.array(raan_list, dtype=np.float64),
        target_phase_deg=np.array(phase_list, dtype=np.float64),
    )


# ======================================================================
# Assignment-based torus fit
# ======================================================================

def build_assignment_cost_matrix(
    raan_deg: np.ndarray,
    phase_deg: np.ndarray,
    target_raan_deg: np.ndarray,
    target_phase_deg: np.ndarray,
    w_raan: float = 1.0,
    w_phase: float = 1.0,
) -> np.ndarray:
    """Build cost matrix C[i, j] for satellite i → slot j assignment.

    C_ij = w_raan * d_angle(Omega_i, Omega*_j)^2
         + w_phase * d_angle(psi_i, psi*_j)^2

    Parameters
    ----------
    raan_deg, phase_deg : ndarray (N,)
        Actual satellite RAAN and phase angles (degrees).
    target_raan_deg, target_phase_deg : ndarray (S,)
        Target slot positions (degrees).
    w_raan, w_phase : float
        Relative weights.

    Returns
    -------
    cost : ndarray (N, S)
    """
    N = raan_deg.shape[0]
    S = target_raan_deg.shape[0]
    cost = np.empty((N, S), dtype=np.float64)

    for i in range(N):
        dr = circular_distance_deg(raan_deg[i], target_raan_deg)
        dp = circular_distance_deg(phase_deg[i], target_phase_deg)
        cost[i, :] = w_raan * dr ** 2 + w_phase * dp ** 2

    return cost


def solve_slot_assignment(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve optimal satellite-to-slot assignment via Hungarian algorithm.

    Handles rectangular matrices (N sats vs S slots where N != S)
    by padding the smaller dimension.

    Parameters
    ----------
    cost_matrix : ndarray (N, S)

    Returns
    -------
    row_ind : ndarray — satellite indices
    col_ind : ndarray — assigned slot indices
    total_cost : float — sum of assigned costs
    """
    N, S = cost_matrix.shape
    if N == S:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    elif N < S:
        # Fewer sats than slots: pad rows with large cost
        padded = np.full((S, S), 1e12, dtype=np.float64)
        padded[:N, :] = cost_matrix
        r, c = linear_sum_assignment(padded)
        mask = r < N
        row_ind, col_ind = r[mask], c[mask]
    else:
        # More sats than slots: pad cols with large cost
        padded = np.full((N, N), 1e12, dtype=np.float64)
        padded[:, :S] = cost_matrix
        r, c = linear_sum_assignment(padded)
        mask = c < S
        row_ind, col_ind = r[mask], c[mask]

    total_cost = float(cost_matrix[row_ind, col_ind].sum())
    return row_ind, col_ind, total_cost


def compute_slot_fit_loss(
    raan_deg: np.ndarray,
    phase_deg: np.ndarray,
    lattice: TorusLattice,
    w_raan: float = 1.0,
    w_phase: float = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute J_slot via optimal assignment.

    Returns
    -------
    J_slot : float
        Mean assignment cost per satellite.
    row_ind, col_ind : assignment mapping.
    """
    N = raan_deg.shape[0]
    if N == 0 or lattice.n_slots == 0:
        return 0.0, np.array([], dtype=int), np.array([], dtype=int)

    C = build_assignment_cost_matrix(
        raan_deg, phase_deg,
        lattice.target_raan_deg, lattice.target_phase_deg,
        w_raan, w_phase,
    )
    row_ind, col_ind, total = solve_slot_assignment(C)
    J_slot = total / max(len(row_ind), 1)
    return J_slot, row_ind, col_ind


# ======================================================================
# Gap-uniformity losses
# ======================================================================

def compute_circular_gap_loss_deg(angles_deg: np.ndarray) -> float:
    """Gap-uniformity loss for sorted circular angles.

    J_gap = (1/n) * sum_k ((gap_k - gap_target) / gap_target)^2
    where gap_target = 360 / n.

    Returns 0.0 for n <= 1.
    """
    n = angles_deg.shape[0]
    if n <= 1:
        return 0.0

    sorted_a = np.sort(wrap_to_360(angles_deg))
    gaps = np.diff(sorted_a)
    # Circular wrap-around gap
    wrap_gap = 360.0 - sorted_a[-1] + sorted_a[0]
    gaps = np.append(gaps, wrap_gap)

    gap_target = 360.0 / n
    return float(np.mean(((gaps - gap_target) / gap_target) ** 2))


def compute_gap_losses(
    raan_deg: np.ndarray,
    phase_deg: np.ndarray,
    lattice: TorusLattice,
    col_ind: np.ndarray,
    alpha_gap_raan: float = 1.0,
    alpha_gap_phase: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute RAAN gap loss + within-plane phase gap losses.

    Parameters
    ----------
    raan_deg, phase_deg : ndarray (N,)
        Actual angles.
    lattice : TorusLattice
        Target lattice (used for plane identification via col_ind).
    col_ind : ndarray (N_assigned,)
        Slot indices from assignment.
    alpha_gap_raan, alpha_gap_phase : float
        Weighting of RAAN vs phase gap terms.

    Returns
    -------
    J_gap_raan, J_gap_phase, J_gap_total
    """
    # Global RAAN gap loss
    J_gap_raan = compute_circular_gap_loss_deg(raan_deg)

    # Per-plane phase gap loss
    spp = max(lattice.slots_per_plane, 1)
    plane_indices = col_ind // spp  # which plane each assigned sat belongs to
    unique_planes = np.unique(plane_indices)

    phase_gap_sum = 0.0
    n_planes_seen = 0
    for p in unique_planes:
        mask = plane_indices == p
        if mask.sum() <= 1:
            continue
        phase_gap_sum += compute_circular_gap_loss_deg(phase_deg[mask])
        n_planes_seen += 1

    J_gap_phase = phase_gap_sum / max(n_planes_seen, 1)
    J_gap_total = alpha_gap_raan * J_gap_raan + alpha_gap_phase * J_gap_phase
    return J_gap_raan, J_gap_phase, J_gap_total


# ======================================================================
# Drift-coherence loss
# ======================================================================

def estimate_drift_deg_per_day(
    angle_series_deg: np.ndarray,
    time_days: np.ndarray,
) -> float:
    """Estimate mean angular drift rate from unwrapped endpoint difference.

    Returns
    -------
    drift : float
        deg/day  (positive = prograde)
    """
    if time_days.size < 2:
        return 0.0
    unwrapped = unwrap_angle_series_deg(angle_series_deg)
    dt = time_days[-1] - time_days[0]
    if abs(dt) < 1e-12:
        return 0.0
    return float((unwrapped[-1] - unwrapped[0]) / dt)


def compute_drift_loss(
    raan_series_list: List[np.ndarray],
    phase_series_list: List[np.ndarray],
    times_days_list: List[np.ndarray],
    kappa_phase_drift: float = 1.0,
) -> float:
    """Drift-coherence loss: variance of drift rates across representatives.

    J_drift = var(Omega_dot_i) + kappa * var(psi_dot_i)
    """
    if len(raan_series_list) < 2:
        return 0.0

    raan_drifts = np.array([
        estimate_drift_deg_per_day(r, t)
        for r, t in zip(raan_series_list, times_days_list)
    ])
    phase_drifts = np.array([
        estimate_drift_deg_per_day(p, t)
        for p, t in zip(phase_series_list, times_days_list)
    ])

    return float(np.var(raan_drifts) + kappa_phase_drift * np.var(phase_drifts))


# ======================================================================
# Torus lattice fitting
# ======================================================================

def fit_torus_lattice(
    raan_deg: np.ndarray,
    phase_deg: np.ndarray,
    n_planes: int,
    slots_per_plane: int,
    w_raan: float = 1.0,
    w_phase: float = 1.0,
    fit_eta: bool = True,
    eta_candidates: Optional[np.ndarray] = None,
) -> TorusLattice:
    """Fit torus lattice parameters (omega0, psi0, eta) to observed positions.

    Layer A: fixed n_planes and slots_per_plane, optimize offsets + eta.

    Parameters
    ----------
    raan_deg, phase_deg : ndarray (N,)
        Observed satellite RAAN and phase (degrees).
    n_planes, slots_per_plane : int
        Lattice dimensions.
    w_raan, w_phase : float
        Assignment cost weights.
    fit_eta : bool
        Whether to optimize inter-plane phasing (True) or use eta=0.
    eta_candidates : ndarray, optional
        Discrete eta values to search. Default: np.linspace(0, 1, 21).

    Returns
    -------
    TorusLattice
        Best-fit lattice.
    """
    if eta_candidates is None:
        eta_candidates = np.linspace(0.0, 1.0, 21)

    delta_omega = 360.0 / max(n_planes, 1)
    delta_psi = 360.0 / max(slots_per_plane, 1)

    best_cost = np.inf
    best_lattice = None

    etas_to_try = eta_candidates if fit_eta else np.array([0.0])

    # Brute-force grid over omega0 and eta; psi0 optimized by scalar search
    omega0_grid = np.linspace(0.0, delta_omega, max(n_planes * 6, 72), endpoint=False)

    for eta_val in etas_to_try:
        for omega0 in omega0_grid:
            # For each (omega0, eta), find best psi0 by scalar optimization
            def _cost_psi0(psi0):
                lat = build_torus_target_slots(n_planes, slots_per_plane,
                                               omega0, psi0, eta_val)
                C = build_assignment_cost_matrix(
                    raan_deg, phase_deg,
                    lat.target_raan_deg, lat.target_phase_deg,
                    w_raan, w_phase,
                )
                _, _, total = solve_slot_assignment(C)
                return total

            res = minimize_scalar(_cost_psi0, bounds=(0.0, delta_psi),
                                  method="bounded",
                                  options={"xatol": 0.5, "maxiter": 30})
            if res.fun < best_cost:
                best_cost = res.fun
                best_lattice = build_torus_target_slots(
                    n_planes, slots_per_plane,
                    omega0, float(res.x), eta_val,
                )

    if best_lattice is None:
        best_lattice = build_torus_target_slots(n_planes, slots_per_plane)

    return best_lattice


# ======================================================================
# Full torus regularizer
# ======================================================================

@dataclass
class TorusRegularizerResult:
    """Structured result from the torus regularizer."""
    J_slot: float = 0.0
    J_gap_raan: float = 0.0
    J_gap_phase: float = 0.0
    J_gap_total: float = 0.0
    J_drift: float = 0.0
    J_torus: float = 0.0
    lattice: Optional[TorusLattice] = None
    row_ind: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    col_ind: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def to_dict(self) -> dict:
        d = {
            "J_slot": self.J_slot,
            "J_gap_raan": self.J_gap_raan,
            "J_gap_phase": self.J_gap_phase,
            "J_gap_total": self.J_gap_total,
            "J_drift": self.J_drift,
            "J_torus": self.J_torus,
        }
        if self.lattice is not None:
            d.update({
                "fitted_omega0_deg": self.lattice.omega0_deg,
                "fitted_psi0_deg": self.lattice.psi0_deg,
                "fitted_delta_omega_deg": 360.0 / max(self.lattice.n_planes, 1),
                "fitted_delta_psi_deg": 360.0 / max(self.lattice.slots_per_plane, 1),
                "fitted_eta": self.lattice.eta,
            })
        return d


def compute_raan_phase_regularizer(
    results: List[dict],
    mode: str = "raan_mean_anomaly",
    n_planes: int = 1,
    slots_per_plane: int = 1,
    w_slot: float = 1.0,
    w_gap: float = 0.25,
    w_drift: float = 0.25,
    w_raan: float = 1.0,
    w_phase: float = 1.0,
    alpha_gap_raan: float = 1.0,
    alpha_gap_phase: float = 1.0,
    kappa_phase_drift: float = 1.0,
    fit_eta: bool = True,
    checkpoint_stride: int = 100,
    lattice_override: Optional[TorusLattice] = None,
) -> TorusRegularizerResult:
    """Compute the full RAAN-phase torus regularizer from simulator outputs.

    Parameters
    ----------
    results : list[dict]
        Propagation results with ``times`` and ``state_sat``.
    mode : str
        Phase variable mode.
    n_planes, slots_per_plane : int
        Lattice geometry.
    w_slot, w_gap, w_drift : float
        Sub-term weights inside J_torus.
    w_raan, w_phase : float
        Assignment cost weights.
    alpha_gap_raan, alpha_gap_phase : float
        Gap-loss sub-weights.
    kappa_phase_drift : float
        Relative weight of phase drift variance vs RAAN drift variance.
    fit_eta : bool
        Whether to fit inter-plane phasing.
    checkpoint_stride : int
        Sub-sample stride for time series.
    lattice_override : TorusLattice, optional
        Pre-fitted lattice (skip fitting step).

    Returns
    -------
    TorusRegularizerResult
    """
    # Extract snapshot RAAN and phase for each representative
    raan_list = []
    phase_list = []
    raan_series_list: list[np.ndarray] = []
    phase_series_list: list[np.ndarray] = []
    times_days_list: list[np.ndarray] = []

    for r in results:
        if "times" not in r or "state_sat" not in r or r["times"].size < 2:
            continue
        t_days, raan_ts, phase_ts = extract_raan_phase_timeseries_deg(
            r, mode=mode, checkpoint_stride=checkpoint_stride,
        )
        # Use initial-epoch snapshot for assignment
        raan_list.append(raan_ts[0])
        phase_list.append(phase_ts[0])
        raan_series_list.append(raan_ts)
        phase_series_list.append(phase_ts)
        times_days_list.append(t_days)

    if not raan_list:
        return TorusRegularizerResult()

    raan_arr = np.array(raan_list, dtype=np.float64)
    phase_arr = np.array(phase_list, dtype=np.float64)

    # Fit or use provided lattice
    if lattice_override is not None:
        lattice = lattice_override
    else:
        lattice = fit_torus_lattice(
            raan_arr, phase_arr,
            n_planes, slots_per_plane,
            w_raan, w_phase, fit_eta,
        )

    # Slot-fit loss
    J_slot, row_ind, col_ind = compute_slot_fit_loss(
        raan_arr, phase_arr, lattice, w_raan, w_phase,
    )

    # Gap losses
    J_gap_raan, J_gap_phase, J_gap_total = compute_gap_losses(
        raan_arr, phase_arr, lattice, col_ind, alpha_gap_raan, alpha_gap_phase,
    )

    # Drift loss
    J_drift = compute_drift_loss(
        raan_series_list, phase_series_list, times_days_list, kappa_phase_drift,
    )

    # Combined torus regularizer
    J_torus = w_slot * J_slot + w_gap * J_gap_total + w_drift * J_drift

    return TorusRegularizerResult(
        J_slot=J_slot,
        J_gap_raan=J_gap_raan,
        J_gap_phase=J_gap_phase,
        J_gap_total=J_gap_total,
        J_drift=J_drift,
        J_torus=J_torus,
        lattice=lattice,
        row_ind=row_ind,
        col_ind=col_ind,
    )


# ======================================================================
# Shell-level torus summary (for stitcher)
# ======================================================================

@dataclass
class ShellTorusSummary:
    """Per-candidate torus summary for stitching coordination."""
    cluster_id: int = 0
    shell_id: int = 0
    omega0_deg: float = 0.0
    psi0_deg: float = 0.0
    delta_omega_deg: float = 0.0
    delta_psi_deg: float = 0.0
    eta: float = 0.0
    J_torus: float = 0.0
    raan_residual_rms_deg: float = 0.0
    phase_residual_rms_deg: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "shell_id": self.shell_id,
            "omega0_deg": self.omega0_deg,
            "psi0_deg": self.psi0_deg,
            "delta_omega_deg": self.delta_omega_deg,
            "delta_psi_deg": self.delta_psi_deg,
            "eta": self.eta,
            "J_torus": self.J_torus,
            "raan_residual_rms_deg": self.raan_residual_rms_deg,
            "phase_residual_rms_deg": self.phase_residual_rms_deg,
        }


def build_shell_torus_summary(
    cluster_id: int,
    torus_result: TorusRegularizerResult,
    raan_deg: np.ndarray,
    phase_deg: np.ndarray,
    shell_id: int = 0,
) -> ShellTorusSummary:
    """Build a summary for the stitcher from a regularizer result."""
    lat = torus_result.lattice
    if lat is None:
        return ShellTorusSummary(cluster_id=cluster_id, shell_id=shell_id)

    # Compute RMS residuals from assignment
    raan_resid = 0.0
    phase_resid = 0.0
    n = len(torus_result.row_ind)
    if n > 0:
        ri = torus_result.row_ind
        ci = torus_result.col_ind
        dr = circular_distance_deg(raan_deg[ri], lat.target_raan_deg[ci])
        dp = circular_distance_deg(phase_deg[ri], lat.target_phase_deg[ci])
        raan_resid = float(np.sqrt(np.mean(dr ** 2)))
        phase_resid = float(np.sqrt(np.mean(dp ** 2)))

    return ShellTorusSummary(
        cluster_id=cluster_id,
        shell_id=shell_id,
        omega0_deg=lat.omega0_deg,
        psi0_deg=lat.psi0_deg,
        delta_omega_deg=360.0 / max(lat.n_planes, 1),
        delta_psi_deg=360.0 / max(lat.slots_per_plane, 1),
        eta=lat.eta,
        J_torus=torus_result.J_torus,
        raan_residual_rms_deg=raan_resid,
        phase_residual_rms_deg=phase_resid,
    )


# ======================================================================
# Pairwise torus consistency penalty (for stitcher)
# ======================================================================

def compute_torus_consistency_penalty(
    summary_g: ShellTorusSummary,
    summary_h: ShellTorusSummary,
    beta_raan: float = 1.0,
    beta_phase: float = 1.0,
    beta_eta: float = 1.0,
) -> float:
    """Pairwise torus consistency penalty between shell-adjacent clusters.

    Psi_phase(g, h) = beta_raan  * d_angle(omega0_g, omega0_h)^2
                    + beta_phase * d_angle(psi0_g,   psi0_h)^2
                    + beta_eta   * (eta_g - eta_h)^2

    Only meaningful for clusters in the same shell/family.
    """
    d_omega0 = circular_distance_deg(summary_g.omega0_deg, summary_h.omega0_deg)
    d_psi0 = circular_distance_deg(summary_g.psi0_deg, summary_h.psi0_deg)
    d_eta = summary_g.eta - summary_h.eta

    return (beta_raan * d_omega0 ** 2
            + beta_phase * d_psi0 ** 2
            + beta_eta * d_eta ** 2)
