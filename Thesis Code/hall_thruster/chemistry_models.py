"""
Chemistry / Ionisation Models for a Krypton Hall Thruster
=========================================================

This module provides **pluggable** closure models for the plasma-chemistry
block used in the Stage A physics penalty and the Stage B simulator.

Model Hierarchy
---------------
``closure_mode``  (set in StageAConfig / StageBConfig):

    ``"legacy_surrogate"``
        The original crude surrogate.  Te is a linear proxy of Vb and νa;
        the Voronov-form ionisation cross-section σ_iv is point-evaluated,
        and the ionisation length λ_i is compared to a fixed reference
        chamber length.

        **Limitation:**  this is NOT a validated physics model.  It is a
        plausible *reduced-order surrogate* used only to regularise the
        beam-parameter space.  The loss penalty it produces is labelled
        ``chemistry_penalty``, not ``chemistry_loss``, to make the
        distinction explicit.

    ``"tabulated"`` *(future placeholder)*
        Load Te(Vb, ṁ, B) from a pre-computed 0-D Boltzmann or PIC
        lookup table.  Not yet implemented; code will raise
        ``NotImplementedError``.

Status Flags
------------
Every model returns a ``ChemistryResult`` dataclass so that downstream
code can branch on ``is_surrogate`` to label plots and loss terms
correctly.

Provenance
----------
σ_iv(Te) Voronov-form for Krypton: Wetzel et al. (1987), refit by
Lotz (1967) form with coefficients from the NIST Electron-Impact
Ionisation Database.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional

import torch

from reduced_dynamics import K_BOLTZMANN_J_K, KR_MASS_KG


# ── Enum / result ────────────────────────────────────────────────────────────
class ClosureMode(enum.Enum):
    LEGACY_SURROGATE = "legacy_surrogate"
    TABULATED = "tabulated"


@dataclass
class ChemistryResult:
    """Bundled outputs of the chemistry block."""
    te_eV: torch.Tensor
    sigma_iv_m3_s: torch.Tensor
    neutral_density_m3: torch.Tensor
    electron_density_m3: torch.Tensor
    lambda_i_m: torch.Tensor
    ionization_ratio: torch.Tensor
    is_surrogate: bool = True  # True → outputs are surrogate, not validated physics


# ── Defaults (from original code) ───────────────────────────────────────────
_DEFAULT_ELECTRON_TEMP_BASE_eV: float = 4.0
_DEFAULT_ELECTRON_TEMP_GAIN_PER_V: float = 0.015
_DEFAULT_ELECTRON_TEMP_GAIN_NUA: float = 8.0
_DEFAULT_PRESSURE_BASE_PA: float = 1.0e-3
_DEFAULT_PRESSURE_GAIN_PA_PER_KG_S: float = 1.0e4
_DEFAULT_NEUTRAL_TEMP_K: float = 500.0
_DEFAULT_IONIZATION_LENGTH_M: float = 0.03


def legacy_surrogate_chemistry(
    vb_V: torch.Tensor,
    nu_a: torch.Tensor,
    mdot_latent_kg_s: torch.Tensor,
    *,
    electron_temp_base_eV: float = _DEFAULT_ELECTRON_TEMP_BASE_eV,
    electron_temp_gain_per_V: float = _DEFAULT_ELECTRON_TEMP_GAIN_PER_V,
    electron_temp_gain_nua: float = _DEFAULT_ELECTRON_TEMP_GAIN_NUA,
    pressure_base_pa: float = _DEFAULT_PRESSURE_BASE_PA,
    pressure_gain_pa_per_kg_s: float = _DEFAULT_PRESSURE_GAIN_PA_PER_KG_S,
    neutral_temp_K: float = _DEFAULT_NEUTRAL_TEMP_K,
    ionization_length_m: float = _DEFAULT_IONIZATION_LENGTH_M,
) -> ChemistryResult:
    r"""Legacy surrogate chemistry block.

    **This is NOT validated physics.**  It is a reduced-order surrogate
    whose sole purpose is to add a soft feasibility regulariser to the
    Stage A loss, preventing the optimiser from wandering into obviously
    unphysical beam-parameter corners.

    Electron temperature model (linear surrogate):
        Te [eV] ≈ base + gain_V · Vb + gain_νa · νa

    Ionisation cross-section (Voronov-form for Kr):
        σ_iv ≈ 5×10⁻¹⁵ · Te^{1.25} · exp(−9.7 / Te)   [m³ s⁻¹]

    Mean-free-path ratio:
        ionization_ratio = λ_i / L_ref

    where L_ref = 0.03 m (representative chamber length).
    """
    te_eV = torch.clamp(
        electron_temp_base_eV
        + electron_temp_gain_per_V * vb_V
        + electron_temp_gain_nua * nu_a,
        min=0.5,
        max=120.0,
    )
    sigma_iv_kr = 5.0e-15 * torch.pow(te_eV, 1.25) * torch.exp(-9.7 / torch.clamp(te_eV, min=0.5))

    pressure_pa = torch.clamp(
        pressure_base_pa + pressure_gain_pa_per_kg_s * mdot_latent_kg_s,
        min=1.0e-7,
        max=50.0,
    )
    neutral_temp_t = torch.clamp(
        torch.full_like(pressure_pa, neutral_temp_K),
        min=100.0,
        max=6000.0,
    )
    neutral_density_m3 = pressure_pa / torch.clamp(
        torch.full_like(pressure_pa, K_BOLTZMANN_J_K) * neutral_temp_t,
        min=1.0e-16,
    )
    electron_density_m3 = torch.clamp(nu_a * neutral_density_m3, min=1.0e8)
    neutral_speed_m_s = torch.sqrt(
        torch.clamp(
            torch.full_like(pressure_pa, 8.0 * K_BOLTZMANN_J_K / math.pi)
            * neutral_temp_t / KR_MASS_KG,
            min=1.0,
        )
    )
    lambda_i_m = neutral_speed_m_s / torch.clamp(electron_density_m3 * sigma_iv_kr, min=1.0e-16)
    ionization_ratio = lambda_i_m / max(1.0e-6, ionization_length_m)

    return ChemistryResult(
        te_eV=te_eV,
        sigma_iv_m3_s=sigma_iv_kr,
        neutral_density_m3=neutral_density_m3,
        electron_density_m3=electron_density_m3,
        lambda_i_m=lambda_i_m,
        ionization_ratio=ionization_ratio,
        is_surrogate=True,
    )


def tabulated_chemistry(
    vb_V: torch.Tensor,
    nu_a: torch.Tensor,
    mdot_latent_kg_s: torch.Tensor,
    **_kwargs,
) -> ChemistryResult:
    """Placeholder for a pre-computed Te(Vb, ṁ, B) lookup table.
    Not yet implemented."""
    raise NotImplementedError(
        "Tabulated chemistry closure is not yet available. "
        "Use closure_mode='legacy_surrogate'."
    )


def compute_chemistry(
    closure_mode: ClosureMode,
    vb_V: torch.Tensor,
    nu_a: torch.Tensor,
    mdot_latent_kg_s: torch.Tensor,
    **kwargs,
) -> ChemistryResult:
    """Dispatch to the selected chemistry closure model."""
    if closure_mode is ClosureMode.LEGACY_SURROGATE:
        return legacy_surrogate_chemistry(vb_V, nu_a, mdot_latent_kg_s, **kwargs)
    elif closure_mode is ClosureMode.TABULATED:
        return tabulated_chemistry(vb_V, nu_a, mdot_latent_kg_s, **kwargs)
    else:
        raise ValueError(f"Unknown closure_mode: {closure_mode!r}")
