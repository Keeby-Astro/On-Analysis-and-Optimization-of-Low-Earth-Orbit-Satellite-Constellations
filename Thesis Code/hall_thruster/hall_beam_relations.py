"""
Krypton Hall-Thruster Beam Relations
=====================================

Self-consistent closure relations for a Krypton-fuelled Hall-effect
thruster, used to derive thrust and specific impulse from beam voltage
(Vb), beam current (Ib), and efficiency decomposition factors.

Physical Basis
--------------
The two primary beam relations used in this pipeline are:

    T  = C_T · γ · I_b · √V_b      [mN]
    Isp = C_I · γ · η_m · √V_b     [s]

where  C_T = 1.32  and  C_I = 154.8  are the Krypton-specific factor
coefficients derived from the one-dimensional beam approximation for an
ion of mass m_Kr = 83.798 u.

**Provenance:**  Goebel & Katz, *Fundamentals of Electric Propulsion:
Ion and Hall Thrusters* (2008), §7–8.

C_T derivation:
    C_T = (2 m_Kr e)^{1/2} × 1e3   ≈ 1.32   when T in mN, Ib in A, Vb in V

C_I derivation:
    C_I = (2 e / m_Kr)^{1/2} / g0   ≈ 154.8  when Isp in s, Vb in V

These coefficients are NOT free parameters — they follow directly from
m_Kr and the elementary charge.  γ (current utilisation) and η_m (mass
utilisation) ARE inferred parameters that absorb multiply-charged species,
neutral leakage, and facility effects.

Efficiency Decomposition
------------------------
Following Brown & Gallimore convention:

    η_total = η_b · η_v · η_m · η_o

    η_b : beam/anode current efficiency  I_b / I_d
    η_v : beam voltage utilisation        V_b / V_d
    η_m : mass utilisation efficiency     (= propellant utilisation)
    η_o : other (facility, divergence, multiply-charged)

Implementation Notes
--------------------
All Torch functions are autograd-safe and broadcastable for use inside
the Stage A forward model and Stage B simulator.
"""
from __future__ import annotations

import torch

# ── Krypton beam constants ───────────────────────────────────────────────────
C_T_KR: float = 1.32
"""Krypton thrust coefficient  [mN / (A · V^{1/2})]  —  (2 m_Kr e)^{1/2} × 1e3."""

C_I_KR: float = 154.8
"""Krypton Isp coefficient  [s / V^{1/2}]  —  (2 e / m_Kr)^{1/2} / g0."""


def thrust_kr_mN(
    gamma: torch.Tensor,
    ib_A: torch.Tensor,
    vb_V: torch.Tensor,
) -> torch.Tensor:
    r"""Beam thrust  T = C_T · γ · I_b · √V_b   [mN].

    Parameters
    ----------
    gamma  : current utilisation factor  (≈ I_b / I_d)
    ib_A   : beam current  [A]
    vb_V   : beam voltage  [V], clamped internally to ≥ 1 V
    """
    return C_T_KR * gamma * ib_A * torch.sqrt(torch.clamp(vb_V, min=1.0))


def isp_kr_s(
    gamma: torch.Tensor,
    eta_m: torch.Tensor,
    vb_V: torch.Tensor,
) -> torch.Tensor:
    r"""Beam specific impulse  Isp = C_I · γ · η_m · √V_b   [s].

    Parameters
    ----------
    gamma  : current utilisation factor
    eta_m  : mass utilisation efficiency
    vb_V   : beam voltage  [V], clamped internally to ≥ 1 V
    """
    return C_I_KR * gamma * eta_m * torch.sqrt(torch.clamp(vb_V, min=1.0))


def beam_exhaust_velocity_m_s(
    gamma: torch.Tensor,
    eta_m: torch.Tensor,
    vb_V: torch.Tensor,
    g0: float = 9.80665,
) -> torch.Tensor:
    """Effective exhaust velocity  v_e = Isp · g0  [m s⁻¹]."""
    return isp_kr_s(gamma, eta_m, vb_V) * g0


def electrical_power_W(
    ib_A: torch.Tensor,
    vb_V: torch.Tensor,
    eta_b: torch.Tensor,
    eta_v: torch.Tensor,
) -> torch.Tensor:
    """Discharge power  P_d = V_d · I_d ≈ V_b · I_b / (η_b · η_v)  [W]."""
    eff = torch.clamp(eta_b * eta_v, min=1.0e-4)
    return vb_V * ib_A / eff
