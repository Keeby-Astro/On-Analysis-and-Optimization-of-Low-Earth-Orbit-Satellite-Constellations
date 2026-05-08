"""
Trajectory-Matching Forward Model and Loss
===========================================

Implements the SMA trajectory-level forward propagation and loss functions
for the ``fit_mode="trajectory_matching"`` path in Stage A.

Physics model
-------------
Same Gauss VOP reduced-order SMA propagation as the segment-endpoint mode:

    a(t) = a₀ + (2·a_net / n₀) · t

where n₀ = √(μ/a₀³) and a_net = thrust_accel − drag.

The key difference from segment-endpoint fitting is that the loss compares
the predicted SMA at **every TLE observation time** along each arc, not just
at the endpoint.  RAAN and λ remain endpoint-only (secondary targets).

Multiple shooting
-----------------
For long arcs (> ``max_subarc_days``), the arc is split into sub-arcs at
nearest TLE observation boundaries.  Each sub-arc is propagated independently
from its observed (TLE) initial SMA.  A continuity penalty encourages the
predicted endpoint of sub-arc k to match the observed start of sub-arc k+1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from reduced_dynamics import (
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    G0_M_S2,
    raan_rate_j2_torch,
    lambda_dot_j2_torch,
    wrap_angle,
    angle_residual,
)


# ── USSA76 atmosphere model (differentiable PyTorch) ─────────────────────────

# Table from USSA1976 — altitude breakpoints [km], base density [kg/m³],
# scale height [km].
_USSA76_ALT = torch.tensor([
    0.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
    110.0, 120.0, 130.0, 140.0, 150.0, 180.0, 200.0, 250.0, 300.0,
    350.0, 400.0, 450.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
], dtype=torch.float64)

_USSA76_RHO = torch.tensor([
    1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4,
    8.283e-5, 1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8,
    8.152e-9, 3.831e-9, 2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11,
    1.916e-11, 7.014e-12, 2.803e-12, 1.184e-12, 5.215e-13, 1.137e-13,
    3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15,
], dtype=torch.float64)

_USSA76_H = torch.tensor([
    7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661, 5.927, 5.533,
    5.703, 6.782, 9.973, 13.243, 16.322, 21.652, 27.974, 34.934,
    43.342, 49.755, 54.513, 58.019, 60.980, 65.654, 76.377, 100.587,
    147.203, 208.020,
], dtype=torch.float64)


def ussa76_density(alt_km: torch.Tensor) -> torch.Tensor:
    """Differentiable USSA76 atmospheric density [kg/m³].

    Parameters
    ----------
    alt_km : Altitude above sea level [km], any shape.

    Returns
    -------
    rho : Atmospheric density [kg/m³], same shape as input.
    """
    device = alt_km.device
    dtype = alt_km.dtype

    h = _USSA76_ALT.to(device=device, dtype=dtype)
    rho_base = _USSA76_RHO.to(device=device, dtype=dtype)
    H = _USSA76_H.to(device=device, dtype=dtype)

    # Clamp altitude to [0, 1000] — same as USSA76_optimized.py
    z = torch.clamp(alt_km, min=0.0, max=1000.0)

    # searchsorted gives the insertion index; subtract 1 for bracket start.
    # h has 28 elements, scale heights H has 27 elements.
    idx = torch.searchsorted(h, z, right=True) - 1
    idx = torch.clamp(idx, min=0, max=len(H) - 1)

    rho0 = rho_base[idx]
    h0 = h[idx]
    Hscale = H[idx]

    return rho0 * torch.exp(-(z - h0) / Hscale)


def ussa76_drag_accel_kmps2(
    a_km: torch.Tensor,
    inv_ballistic_coeff: float = 0.0334,
    mu: float = MU_EARTH_KM3_S2,
) -> torch.Tensor:
    """Orbit-averaged drag deceleration [km/s²] from USSA76 at given SMA.

    Assumes near-circular orbit: altitude ≈ a − R_Earth, v ≈ √(μ/a).

    drag_accel = inv_BC · ρ(h) · v²

    where inv_BC = Cd·A / (2·m) [m²/kg] is the inverse ballistic coefficient.

    Parameters
    ----------
    a_km : Semi-major axis [km], any shape.
    inv_ballistic_coeff : Cd·A/(2·m) in [m²/kg].  Default 0.0334 (Starlink flat plate).
    mu : Gravitational parameter [km³/s²].

    Returns
    -------
    drag_accel : Drag deceleration [km/s²], same shape.  Always ≥ 0.
    """
    alt_km = a_km - R_EARTH_KM
    rho_kg_m3 = ussa76_density(alt_km)          # kg/m³
    v_km_s = torch.sqrt(torch.tensor(mu, device=a_km.device, dtype=a_km.dtype) / a_km)

    # Units:  inv_BC [m²/kg] · ρ [kg/m³] · v² [km²/s²]
    # = [1/m] · [km²/s²] = [km²/(m·s²)]
    # Multiply by 1e3 to convert m→km:  result in [km/s²]
    drag = inv_ballistic_coeff * rho_kg_m3 * v_km_s ** 2 * 1.0e3

    return torch.clamp(drag, min=0.0)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TrajectoryConfig:
    """Configuration for trajectory-matching forward model and loss."""
    # Loss weights
    lambda_path: float = 5.0           # SMA path (primary)
    lambda_endpoint_a: float = 1.0     # SMA endpoint
    lambda_endpoint_raan: float = 0.0  # RAAN endpoint
    lambda_endpoint_lam: float = 0.0   # lambda endpoint
    lambda_continuity: float = 0.1     # multiple-shooting continuity
    # Multiple shooting
    max_subarc_days: float = 30.0
    # Arc weighting
    arc_weight_mode: str = "sqrt_inv_n_obs"   # "uniform" | "inv_n_obs" | "sqrt_inv_n_obs"
    # Robust loss (mirrors TrainConfig defaults)
    robust_loss: str = "mse"
    huber_delta: float = 1.0
    obs_scale_a_km: float = 3.0
    obs_scale_angle_rad: float = 0.02
    student_t_dof: float = 4.0
    student_t_scale: float = 1.0
    # Atmosphere-based drag
    use_atmosphere_drag: bool = True    # Replace constant drag with USSA76-based
    inv_ballistic_coeff: float = 0.0334  # Cd·A/(2·m) [m²/kg] for Starlink flat plate
    # Non-linear propagation (RK4 ODE integration with altitude-dependent drag)
    nonlinear_propagation: bool = False  # Use RK4 instead of linear a(t) = a0 + rate*t
    rk4_step_hours: float = 6.0         # RK4 integration step size [hours]


# ── Forward model ────────────────────────────────────────────────────────────

def compute_accel_net(
    phase_sign: torch.Tensor,          # (B,)
    thrust_N: torch.Tensor,            # (B,)
    duty_effective: torch.Tensor,      # (B,)
    mass_kg: torch.Tensor,             # scalar or (B,)
    drag_kmps2: torch.Tensor,          # (B,)
    shell_drag_comp_fraction: torch.Tensor,  # (B,)
    direction_strength: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """Compute net acceleration [km/s²] for SMA propagation.

    For operational shell (phase_sign ≈ 0):
        accel_net = shell_drag_comp_fraction * duty * thrust/mass/1000 - drag
        When fraction ≈ 1.0 and thrust ≈ drag*mass*1000/duty, net ≈ 0 (drag-balanced).

    For orbit raise (phase_sign = +1):
        accel_net = +softplus(dir) * duty * thrust/mass/1000 - drag

    For disposal (phase_sign = -1):
        accel_net = -softplus(dir) * duty * thrust/mass/1000 - drag

    Returns: (B,) tensor of net SMA acceleration [km/s²].
    """
    direction = torch.where(
        phase_sign.abs() < 0.5,
        # operational shell: prograde drag make-up modulated by learned fraction
        shell_drag_comp_fraction,
        # orbit raise / disposal: signed direction with minimum 0.25
        # Use softplus instead of abs to avoid zero gradient at initialization
        phase_sign * torch.clamp(
            torch.nn.functional.softplus(direction_strength) + 0.25,
            min=0.25, max=1.0,
        ),
    )
    thrust_accel_kmps2 = direction * duty_effective * (thrust_N / mass_kg) / 1000.0
    return thrust_accel_kmps2 - drag_kmps2


def trajectory_sma_forward(
    a0_km: torch.Tensor,               # (B,)
    dt_array: torch.Tensor,            # (B, max_obs)
    mask: torch.Tensor,                # (B, max_obs)
    accel_net_kmps2: torch.Tensor,     # (B,)
    mu: float = MU_EARTH_KM3_S2,
) -> torch.Tensor:
    """Propagate SMA along trajectory using Gauss VOP reduced model.

    a_pred(t_i) = a0 + (2 · accel_net / n0) · t_i

    Parameters
    ----------
    a0_km : Initial SMA [km], shape (B,)
    dt_array : Time offsets from arc start [s], shape (B, max_obs)
    mask : Binary mask (1 where valid), shape (B, max_obs)
    accel_net_kmps2 : Net acceleration [km/s²], shape (B,)
    mu : Gravitational parameter [km³/s²]

    Returns
    -------
    a_pred_km : Predicted SMA at each time [km], shape (B, max_obs)
    """
    a0_safe = torch.clamp(a0_km, min=R_EARTH_KM + 120.0)
    mu_t = torch.tensor(mu, device=a0_km.device, dtype=a0_km.dtype)
    n0 = torch.sqrt(mu_t / (a0_safe ** 3))  # (B,)

    # Expand for broadcasting: (B,1) * (B, max_obs) → (B, max_obs)
    sma_rate = 2.0 * accel_net_kmps2 / n0  # (B,) [km/s]
    a_pred = a0_safe.unsqueeze(1) + sma_rate.unsqueeze(1) * dt_array  # (B, max_obs)

    # Clamp predictions to physical range, masked
    a_pred = torch.clamp(a_pred, min=R_EARTH_KM + 100.0) * mask

    return a_pred


def _sma_deriv(
    a_km: torch.Tensor,
    thrust_accel_kmps2: torch.Tensor,
    drag_scale: torch.Tensor,
    inv_BC: float,
    mu: float,
    a_min: float,
) -> torch.Tensor:
    """Compute da/dt [km/s] given current SMA.

    da/dt = 2 * (thrust_accel - drag_scale * ussa76_drag(a)) / n(a)

    where n(a) = sqrt(mu / a^3).
    The drag_scale absorbs the learned per-phase and per-satellite drag factors.
    """
    a_safe = torch.clamp(a_km, min=a_min)
    mu_t = torch.tensor(mu, device=a_km.device, dtype=a_km.dtype)
    n = torch.sqrt(mu_t / (a_safe ** 3))
    drag = ussa76_drag_accel_kmps2(a_safe, inv_ballistic_coeff=inv_BC, mu=mu)
    accel_net = thrust_accel_kmps2 - drag_scale * drag
    return 2.0 * accel_net / n


def trajectory_sma_forward_nonlinear(
    a0_km: torch.Tensor,               # (B,)
    dt_array: torch.Tensor,            # (B, max_obs)
    mask: torch.Tensor,                # (B, max_obs)
    thrust_accel_kmps2: torch.Tensor,  # (B,) — signed thrust acceleration
    drag_scale: torch.Tensor,          # (B,) — learned drag multiplicative factor
    inv_ballistic_coeff: float = 0.0334,
    rk4_step_s: float = 21600.0,       # 6 hours default
    mu: float = MU_EARTH_KM3_S2,
) -> torch.Tensor:
    """Propagate SMA using RK4 ODE integration with altitude-dependent drag.

    Integrates  da/dt = 2 * (thrust_accel - drag_scale * ussa76_drag(a)) / n(a)
    where ussa76_drag(a) is re-evaluated at the current altitude each step
    and n(a) = sqrt(mu/a^3) is the current mean motion.

    This replaces the linear model a(t) = a0 + const*t with a proper
    numerical ODE integration that captures:
    - Exponential drag increase as altitude decreases
    - Changing mean motion as SMA evolves
    - Non-linear feedback between drag and orbit decay

    Parameters
    ----------
    a0_km : Initial SMA [km], shape (B,)
    dt_array : Time offsets from arc start [s], shape (B, max_obs)
    mask : Binary mask (1 where valid), shape (B, max_obs)
    thrust_accel_kmps2 : Signed thrust acceleration [km/s²], shape (B,)
        This is the thrust-only component with phase/direction applied.
    drag_scale : Learned drag scaling factor, shape (B,)
        Multiplied by USSA76 drag at each integration step.
    inv_ballistic_coeff : Cd·A/(2·m) [m²/kg]
    rk4_step_s : Integration step size [seconds]
    mu : Gravitational parameter [km³/s²]

    Returns
    -------
    a_pred_km : Predicted SMA at each observation time [km], shape (B, max_obs)
    """
    B, max_obs = dt_array.shape
    device = a0_km.device
    dtype = a0_km.dtype
    a_min = R_EARTH_KM + 100.0

    a0_safe = torch.clamp(a0_km, min=R_EARTH_KM + 120.0)

    # Find max time in the batch to determine number of integration steps
    dt_max = dt_array.max().item()
    if dt_max <= 0.0:
        return a0_safe.unsqueeze(1).expand(B, max_obs) * mask

    n_steps = max(int(math.ceil(dt_max / rk4_step_s)), 1)
    h = dt_max / n_steps  # actual step size [s]

    h_t = torch.tensor(h, device=device, dtype=dtype)

    # Integrate all arcs in parallel using list accumulation (autograd-safe)
    a_steps = [a0_safe]  # list of (B,) tensors

    for i in range(n_steps):
        a_i = a_steps[i]

        # RK4 stages
        k1 = _sma_deriv(a_i, thrust_accel_kmps2, drag_scale, inv_ballistic_coeff, mu, a_min)
        k2 = _sma_deriv(
            torch.clamp(a_i + 0.5 * h_t * k1, min=a_min),
            thrust_accel_kmps2, drag_scale, inv_ballistic_coeff, mu, a_min,
        )
        k3 = _sma_deriv(
            torch.clamp(a_i + 0.5 * h_t * k2, min=a_min),
            thrust_accel_kmps2, drag_scale, inv_ballistic_coeff, mu, a_min,
        )
        k4 = _sma_deriv(
            torch.clamp(a_i + h_t * k3, min=a_min),
            thrust_accel_kmps2, drag_scale, inv_ballistic_coeff, mu, a_min,
        )

        a_next = torch.clamp(
            a_i + (h_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            min=a_min,
        )
        a_steps.append(a_next)

    # Stack into (B, n_steps+1)
    a_history = torch.stack(a_steps, dim=1)

    # Interpolate from RK4 grid to observation times
    # Normalize observation times to [0, n_steps] index space
    t_idx = dt_array * (float(n_steps) / max(dt_max, 1e-8))  # (B, max_obs)
    t_idx = torch.clamp(t_idx, min=0.0, max=float(n_steps))

    # Floor/ceil indices for linear interpolation
    idx_lo = t_idx.long().clamp(min=0, max=n_steps - 1)
    idx_hi = (idx_lo + 1).clamp(max=n_steps)
    frac = t_idx - idx_lo.to(dtype=dtype)

    # Gather from a_history
    a_lo = a_history.gather(1, idx_lo)
    a_hi = a_history.gather(1, idx_hi)
    a_pred = a_lo + frac * (a_hi - a_lo)

    a_pred = torch.clamp(a_pred, min=a_min) * mask
    return a_pred


def trajectory_sma_dispatch(
    a0_km: torch.Tensor,
    dt_array: torch.Tensor,
    mask: torch.Tensor,
    accel_net_kmps2: torch.Tensor,
    traj_cfg: TrajectoryConfig,
    thrust_accel_kmps2: Optional[torch.Tensor] = None,
    drag_scale: Optional[torch.Tensor] = None,
    mu: float = MU_EARTH_KM3_S2,
) -> torch.Tensor:
    """Dispatch to linear or non-linear SMA forward model based on config.

    For non-linear mode, ``thrust_accel_kmps2`` and ``drag_scale`` must be provided.
    For linear mode, only ``accel_net_kmps2`` is used.
    """
    if traj_cfg.nonlinear_propagation and traj_cfg.use_atmosphere_drag:
        if thrust_accel_kmps2 is None or drag_scale is None:
            raise ValueError(
                "nonlinear_propagation requires thrust_accel_kmps2 and drag_scale"
            )
        return trajectory_sma_forward_nonlinear(
            a0_km, dt_array, mask, thrust_accel_kmps2, drag_scale,
            inv_ballistic_coeff=traj_cfg.inv_ballistic_coeff,
            rk4_step_s=traj_cfg.rk4_step_hours * 3600.0,
            mu=mu,
        )
    return trajectory_sma_forward(a0_km, dt_array, mask, accel_net_kmps2, mu=mu)


def trajectory_endpoint_raan(
    raan0_rad: torch.Tensor,           # (B,)
    a0_km: torch.Tensor,               # (B,)
    a_final_pred_km: torch.Tensor,     # (B,)
    e0: torch.Tensor,                  # (B,)
    inc0_rad: torch.Tensor,            # (B,)
    dt_final_s: torch.Tensor,          # (B,)
    mu: float = MU_EARTH_KM3_S2,
) -> torch.Tensor:
    """Predict RAAN at arc endpoint via J2 secular rate.

    Uses midpoint SMA for rate evaluation (trapezoidal average).

    Returns: raan_final_pred [rad], shape (B,).
    """
    mu_t = torch.tensor(mu, device=a0_km.device, dtype=a0_km.dtype)
    a_mid = torch.clamp(0.5 * (a0_km + a_final_pred_km), min=R_EARTH_KM + 120.0)
    raan_dot = raan_rate_j2_torch(a_mid, e0, inc0_rad, mu_t)
    return raan0_rad + raan_dot * dt_final_s


def trajectory_endpoint_lambda(
    lam0_rad: torch.Tensor,            # (B,)
    a0_km: torch.Tensor,               # (B,)
    a_final_pred_km: torch.Tensor,     # (B,)
    e0: torch.Tensor,                  # (B,)
    inc0_rad: torch.Tensor,            # (B,)
    dt_final_s: torch.Tensor,          # (B,)
    mu: float = MU_EARTH_KM3_S2,
) -> torch.Tensor:
    """Predict mean longitude at arc endpoint via J2 secular rate.

    Uses trapezoidal average over [a0, a_mid] for better accuracy.

    Returns: lam_final_pred [rad], shape (B,).
    """
    mu_t = torch.tensor(mu, device=a0_km.device, dtype=a0_km.dtype)
    a0_safe = torch.clamp(a0_km, min=R_EARTH_KM + 120.0)
    a_mid = torch.clamp(0.5 * (a0_safe + a_final_pred_km), min=R_EARTH_KM + 120.0)
    lam_dot_0 = lambda_dot_j2_torch(a0_safe, e0, inc0_rad, mu_t)
    lam_dot_1 = lambda_dot_j2_torch(a_mid, e0, inc0_rad, mu_t)
    lam_dot_avg = 0.5 * (lam_dot_0 + lam_dot_1)
    return lam0_rad + wrap_angle(lam_dot_avg * dt_final_s)


# ── Robust loss (self-contained to avoid circular import) ────────────────────

def _robust_loss(
    residual: torch.Tensor,
    weight: Optional[torch.Tensor],
    mode: str,
    scale: float,
    huber_delta: float,
    student_t_dof: float = 4.0,
    student_t_scale: float = 1.0,
) -> torch.Tensor:
    """Weighted robust loss matching the Stage A convention."""
    resid = residual.reshape(-1)
    finite = torch.isfinite(resid)
    if weight is None:
        w = torch.ones_like(resid)
    else:
        w = weight.reshape(-1)
        finite = finite & torch.isfinite(w)
    if not bool(torch.any(finite)):
        return torch.zeros((), dtype=resid.dtype, device=resid.device)

    resid = resid[finite] / max(float(scale), 1.0e-8)
    w = w[finite]

    robust_mode = str(mode).strip().lower()
    if robust_mode == "huber":
        delta = max(float(huber_delta), 1.0e-6)
        abs_r = torch.abs(resid)
        quad = torch.minimum(abs_r, torch.full_like(abs_r, delta))
        lin = abs_r - quad
        per_item = 0.5 * quad ** 2 + delta * lin
    elif robust_mode == "pseudo_huber":
        delta = max(float(huber_delta), 1.0e-6)
        per_item = (delta ** 2) * (torch.sqrt(1.0 + (resid / delta) ** 2) - 1.0)
    elif robust_mode == "student_t":
        dof = max(float(student_t_dof), 1.01)
        t_scale = max(float(student_t_scale), 1.0e-8)
        z = resid / t_scale
        per_item = 0.5 * (dof + 1.0) * torch.log1p((z ** 2) / dof) + math.log(t_scale)
    else:
        per_item = resid ** 2

    return torch.sum(per_item * w) / torch.clamp(torch.sum(w), min=1.0e-6)


# ── Multiple shooting ───────────────────────────────────────────────────────

def split_arcs_for_multiple_shooting(
    dt_s: torch.Tensor,                # (B, max_obs)
    mask: torch.Tensor,                # (B, max_obs)
    a_obs_km: torch.Tensor,            # (B, max_obs)
    max_subarc_s: float,
) -> List[List[Tuple[int, int]]]:
    """Compute sub-arc boundaries for multiple shooting.

    For each arc in the batch, returns a list of (start_idx, end_idx) pairs
    indicating observation index ranges for each sub-arc.

    If an arc is shorter than ``max_subarc_s``, it gets a single sub-arc
    spanning all observations.

    Parameters
    ----------
    dt_s : Time array (B, max_obs)
    mask : Validity mask (B, max_obs)
    a_obs_km : SMA observations (B, max_obs) — used for shooting node values
    max_subarc_s : Maximum sub-arc duration in seconds

    Returns
    -------
    List of length B, each element is a list of (start_idx, end_idx) tuples.
    """
    B = dt_s.shape[0]
    dt_np = dt_s.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    batch_subarcs = []

    for b in range(B):
        valid_count = int(mask_np[b].sum())
        if valid_count == 0:
            batch_subarcs.append([(0, 0)])
            continue

        times = dt_np[b, :valid_count]
        total_duration = times[-1] - times[0]

        if total_duration <= max_subarc_s:
            batch_subarcs.append([(0, valid_count - 1)])
            continue

        # Split at boundaries every max_subarc_s
        subarcs = []
        start_idx = 0
        while start_idx < valid_count - 1:
            t_start = times[start_idx]
            # Find the last index where dt <= t_start + max_subarc_s
            end_idx = start_idx
            for j in range(start_idx + 1, valid_count):
                if times[j] - t_start <= max_subarc_s:
                    end_idx = j
                else:
                    break
            # Ensure at least 2 observations per subarc
            end_idx = max(end_idx, min(start_idx + 1, valid_count - 1))
            subarcs.append((start_idx, end_idx))
            start_idx = end_idx  # Next subarc starts where this one ends

        # Merge last subarc if it's too short (only 1 obs)
        if len(subarcs) > 1 and subarcs[-1][0] == subarcs[-1][1]:
            last = subarcs.pop()
            prev = subarcs.pop()
            subarcs.append((prev[0], last[1]))

        batch_subarcs.append(subarcs)

    return batch_subarcs


def compute_continuity_loss(
    a_pred_km: torch.Tensor,           # (B, max_obs)
    a_obs_km: torch.Tensor,            # (B, max_obs)
    batch_subarcs: List[List[Tuple[int, int]]],
    scale: float = 5.0,
) -> torch.Tensor:
    """Continuity penalty: predicted SMA at end of sub-arc k should match
    observed SMA at start of sub-arc k+1.

    For shooting nodes fixed to TLE observations, this measures how well the
    constant-acceleration model within each sub-arc connects to the next TLE
    observation.
    """
    device = a_pred_km.device
    total = torch.zeros((), dtype=a_pred_km.dtype, device=device)
    count = 0

    for b, subarcs in enumerate(batch_subarcs):
        if len(subarcs) <= 1:
            continue
        for k in range(len(subarcs) - 1):
            end_idx = subarcs[k][1]
            next_start_idx = subarcs[k + 1][0]
            # Predicted SMA at end of sub-arc k
            a_pred_end = a_pred_km[b, end_idx]
            # Observed SMA at start of sub-arc k+1 (shooting node)
            a_obs_start_next = a_obs_km[b, next_start_idx]
            residual = (a_pred_end - a_obs_start_next) / max(scale, 1.0e-8)
            total = total + residual ** 2
            count += 1

    if count > 0:
        total = total / float(count)
    return total


# ── Full trajectory loss ─────────────────────────────────────────────────────

def trajectory_forward_and_loss(
    model,
    batch: Dict[str, torch.Tensor],
    traj_cfg: TrajectoryConfig,
    epoch: int = 1,
    curriculum_kinematics_epochs: int = 20,
    curriculum_physics_ramp_start: int = 40,
    curriculum_physics_ramp_epochs: int = 20,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Full trajectory-matching forward pass and loss computation.

    Uses the *same* StageAModel parameter structure as segment-endpoint mode,
    but evaluates predicted SMA at every TLE observation time rather than just
    endpoints.

    Parameters
    ----------
    model : StageAModel instance
    batch : Dict from ArcDataset / collate_arcs
    traj_cfg : TrajectoryConfig with loss weights & settings
    epoch : Current training epoch (for curriculum scheduling)

    Returns
    -------
    loss : Scalar total loss
    metrics : Dict of per-term loss values
    """
    p = model.constrained_parameters()

    # ── Extract batch ────────────────────────────────────────────────────
    phase_idx = batch["phase_idx"]          # (B,)
    sat_idx = batch["sat_idx"]              # (B,)
    a0_km = batch["a0_km"]                  # (B,)
    e0 = batch["e0"]                        # (B,)
    inc0_rad = batch["inc0_rad"]            # (B,)
    raan0_rad = batch["raan0_rad"]          # (B,)
    lam0_rad = batch["lam0_rad"]            # (B,)
    dt_array = batch["dt_s"]                # (B, max_obs)
    a_obs_km = batch["a_obs_km"]            # (B, max_obs)
    obs_mask = batch["mask"]                # (B, max_obs)
    n_obs = batch["n_obs"]                  # (B,)
    raan_final_obs = batch["raan_final_rad"]  # (B,)
    lam_final_obs = batch["lam_final_rad"]    # (B,)

    phase_sign = model.phase_signs[phase_idx]  # (B,)

    # ── Phase/satellite-indexed parameters ───────────────────────────────
    thrust_phase = p["thrust_N"][phase_idx] * p["sat_thrust_scale"][sat_idx]
    duty_phase = p["duty"][phase_idx]
    phase_power_cap = p["phase_power_cap_W"][phase_idx]
    phase_eta_total = p["phase_eta_total"][phase_idx]

    # Power cap scaling
    power_nominal_W = thrust_phase * G0_M_S2 * p["isp_s"] / (2.0 * phase_eta_total)
    if model.cfg.use_power_cap:
        power_scale = torch.clamp(phase_power_cap / torch.clamp(power_nominal_W, min=1.0), max=1.0)
    else:
        power_scale = torch.ones_like(duty_phase)
    duty_effective = torch.clamp(duty_phase * power_scale, min=1.0e-4, max=model.cfg.thermal_duty_cap)

    if model.cfg.use_drag:
        if traj_cfg.use_atmosphere_drag:
            # Altitude-dependent drag from USSA76 atmosphere model.
            # Base drag at arc start altitude, scaled by learned per-phase and
            # per-satellite multiplicative factors.
            drag_base = ussa76_drag_accel_kmps2(
                a0_km, inv_ballistic_coeff=traj_cfg.inv_ballistic_coeff,
            )
            drag_phase = drag_base * p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
        else:
            drag_phase = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
    else:
        drag_phase = torch.zeros_like(duty_effective)

    # ── Net acceleration ─────────────────────────────────────────────────
    # Piecewise thrust schedule (midpoint scale) applies if configured
    if model.cfg.use_piecewise_thrust_schedule:
        ramp = p["phase_ramp_fraction"][phase_idx]
        ramp_scale = 1.0 - 0.25 * ramp
        thrust_schedule_scale = torch.clamp(
            0.5 * (1.0 + p["phase_midpoint_scale"][phase_idx]), min=0.4, max=2.0
        )
        thrust_effective = thrust_phase * ramp_scale * thrust_schedule_scale
    else:
        thrust_effective = thrust_phase

    accel_net = compute_accel_net(
        phase_sign=phase_sign,
        thrust_N=thrust_effective,
        duty_effective=duty_effective,
        mass_kg=p["mass_kg"],
        drag_kmps2=drag_phase,
        shell_drag_comp_fraction=p["shell_drag_comp_fraction"][phase_idx],
        direction_strength=p["phase_direction_strength"][phase_idx],
    )

    # ── SMA trajectory forward ───────────────────────────────────────────
    if traj_cfg.nonlinear_propagation and traj_cfg.use_atmosphere_drag:
        # Non-linear RK4 mode: separate thrust acceleration from drag.
        # Drag is re-evaluated at each integration step's altitude.
        # Compute thrust-only acceleration (same direction logic as compute_accel_net)
        direction = torch.where(
            phase_sign.abs() < 0.5,
            p["shell_drag_comp_fraction"][phase_idx],
            phase_sign * torch.clamp(
                torch.nn.functional.softplus(p["phase_direction_strength"][phase_idx]) + 0.25,
                min=0.25, max=1.0,
            ),
        )
        thrust_accel_only = direction * duty_effective * (thrust_effective / p["mass_kg"]) / 1000.0
        # Drag scale: learned per-phase × per-satellite multiplicative factor
        drag_scale = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
        a_pred = trajectory_sma_forward_nonlinear(
            a0_km, dt_array, obs_mask, thrust_accel_only, drag_scale,
            inv_ballistic_coeff=traj_cfg.inv_ballistic_coeff,
            rk4_step_s=traj_cfg.rk4_step_hours * 3600.0,
        )
    else:
        a_pred = trajectory_sma_forward(a0_km, dt_array, obs_mask, accel_net)  # (B, max_obs)

    # ── Get final observation index per arc ───────────────────────────────
    # n_obs gives count; final index = n_obs - 1, clamped to valid range
    max_obs_dim = dt_array.shape[1]
    final_idx = torch.clamp(n_obs - 1, min=0, max=max_obs_dim - 1)  # (B,)
    # Gather final predicted SMA
    a_final_pred = a_pred.gather(1, final_idx.unsqueeze(1)).squeeze(1)  # (B,)
    dt_final = dt_array.gather(1, final_idx.unsqueeze(1)).squeeze(1)  # (B,)
    a_final_obs = a_obs_km.gather(1, final_idx.unsqueeze(1)).squeeze(1)  # (B,)

    # ── RAAN/lambda endpoint predictions ─────────────────────────────────
    raan_final_pred = trajectory_endpoint_raan(
        raan0_rad, a0_km, a_final_pred, e0, inc0_rad, dt_final,
    )
    lam_final_pred = trajectory_endpoint_lambda(
        lam0_rad, a0_km, a_final_pred, e0, inc0_rad, dt_final,
    )

    # ── Per-observation weights ──────────────────────────────────────────
    # obs_mask already zeros out padded positions
    arc_weight_mode = traj_cfg.arc_weight_mode
    if arc_weight_mode == "inv_n_obs":
        # Weight each observation by 1/n_obs so long arcs don't dominate
        n_obs_f = n_obs.to(dtype=torch.float32).clamp(min=1.0)
        per_obs_weight = obs_mask / n_obs_f.unsqueeze(1)
    elif arc_weight_mode == "sqrt_inv_n_obs":
        n_obs_f = n_obs.to(dtype=torch.float32).clamp(min=1.0)
        per_obs_weight = obs_mask / torch.sqrt(n_obs_f).unsqueeze(1)
    else:  # "uniform"
        per_obs_weight = obs_mask

    # ── SMA path loss (primary) ──────────────────────────────────────────
    path_residual = (a_pred - a_obs_km) * obs_mask  # zero where masked
    loss_path = _robust_loss(
        path_residual,
        per_obs_weight,
        traj_cfg.robust_loss,
        traj_cfg.obs_scale_a_km,
        traj_cfg.huber_delta,
        traj_cfg.student_t_dof,
        traj_cfg.student_t_scale,
    )

    # ── SMA endpoint loss (secondary) ────────────────────────────────────
    loss_endpoint_a = _robust_loss(
        a_final_pred - a_final_obs,
        None,
        traj_cfg.robust_loss,
        traj_cfg.obs_scale_a_km,
        traj_cfg.huber_delta,
        traj_cfg.student_t_dof,
        traj_cfg.student_t_scale,
    )

    # ── RAAN endpoint loss (secondary) ───────────────────────────────────
    loss_endpoint_raan = _robust_loss(
        angle_residual(raan_final_pred, raan_final_obs),
        None,
        traj_cfg.robust_loss,
        traj_cfg.obs_scale_angle_rad,
        traj_cfg.huber_delta,
        traj_cfg.student_t_dof,
        traj_cfg.student_t_scale,
    )

    # ── Lambda endpoint loss (tertiary) ──────────────────────────────────
    loss_endpoint_lam = _robust_loss(
        angle_residual(lam_final_pred, lam_final_obs),
        None,
        traj_cfg.robust_loss,
        traj_cfg.obs_scale_angle_rad,
        traj_cfg.huber_delta,
        traj_cfg.student_t_dof,
        traj_cfg.student_t_scale,
    )

    # ── Multiple shooting continuity loss ────────────────────────────────
    max_subarc_s = traj_cfg.max_subarc_days * 86400.0
    batch_subarcs = split_arcs_for_multiple_shooting(
        dt_array, obs_mask, a_obs_km, max_subarc_s,
    )
    loss_continuity = compute_continuity_loss(a_pred, a_obs_km, batch_subarcs)

    # ── Curriculum: kinematics-only → endpoint → full ────────────────────
    # During early epochs, only train SMA path + endpoint.
    # After curriculum_physics_ramp_start, ramp in RAAN/lam/continuity.
    if epoch <= curriculum_kinematics_epochs:
        raan_scale = 0.0
        lam_scale = 0.0
        continuity_scale = 0.0
    elif epoch <= curriculum_physics_ramp_start:
        raan_scale = 0.5
        lam_scale = 0.0
        continuity_scale = 0.5
    elif curriculum_physics_ramp_epochs > 0 and epoch <= curriculum_physics_ramp_start + curriculum_physics_ramp_epochs:
        frac = float(epoch - curriculum_physics_ramp_start) / float(curriculum_physics_ramp_epochs)
        raan_scale = 0.5 + 0.5 * frac
        lam_scale = frac
        continuity_scale = 0.5 + 0.5 * frac
    else:
        raan_scale = 1.0
        lam_scale = 1.0
        continuity_scale = 1.0

    # ── Total loss ───────────────────────────────────────────────────────
    total = (
        traj_cfg.lambda_path * loss_path
        + traj_cfg.lambda_endpoint_a * loss_endpoint_a
        + raan_scale * traj_cfg.lambda_endpoint_raan * loss_endpoint_raan
        + lam_scale * traj_cfg.lambda_endpoint_lam * loss_endpoint_lam
        + continuity_scale * traj_cfg.lambda_continuity * loss_continuity
    )

    if not bool(torch.isfinite(total)):
        raise FloatingPointError(
            f"Trajectory loss became non-finite: path={float(loss_path.detach().cpu())}, "
            f"endpoint_a={float(loss_endpoint_a.detach().cpu())}, "
            f"raan={float(loss_endpoint_raan.detach().cpu())}, "
            f"lam={float(loss_endpoint_lam.detach().cpu())}, "
            f"continuity={float(loss_continuity.detach().cpu())}"
        )

    metrics = {
        "loss_total": float(total.detach().cpu()),
        "loss_path": float(loss_path.detach().cpu()),
        "loss_endpoint_a": float(loss_endpoint_a.detach().cpu()),
        "loss_endpoint_raan": float(loss_endpoint_raan.detach().cpu()),
        "loss_endpoint_lam": float(loss_endpoint_lam.detach().cpu()),
        "loss_continuity": float(loss_continuity.detach().cpu()),
        "raan_scale": raan_scale,
        "lam_scale": lam_scale,
        "a_pred_mean_km": float(a_pred[obs_mask > 0].mean().detach().cpu()) if obs_mask.sum() > 0 else 0.0,
        "a_obs_mean_km": float(a_obs_km[obs_mask > 0].mean().detach().cpu()) if obs_mask.sum() > 0 else 0.0,
        "path_rmse_km": float(
            torch.sqrt((path_residual ** 2).sum() / torch.clamp(obs_mask.sum(), min=1.0)).detach().cpu()
        ),
    }
    return total, metrics
