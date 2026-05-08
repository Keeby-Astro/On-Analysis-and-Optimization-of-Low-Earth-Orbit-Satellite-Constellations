"""Configuration dataclasses for Chapter 7 constellation-level optimization.

Defines the policy parameterization, objective weights, fidelity levels,
uncertainty scenarios, and top-level study configuration used by the
hierarchical cluster optimizer.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple


# ======================================================================
# Cluster policy vector  (theta_g)
# ======================================================================

@dataclass
class ClusterPolicy:
    """Shared policy vector applied to every satellite in a cluster.

    Units follow SI / orbital-mechanics conventions:
        delta_a                 km
        delta_Omega             rad
        delta_lambda            rad
        tau_keep                seconds
        deadband_a              km
        deadband_lambda         rad
        reserve_prop_frac       dimensionless  [0, 1]
        disposal_altitude_trigger  km (altitude above Earth surface)
    """
    delta_a: float = 0.0
    delta_Omega: float = 0.0
    delta_lambda: float = 0.0
    tau_keep: float = 86400.0       # 1 day default
    deadband_a: float = 5.0         # km
    deadband_lambda: float = 0.01   # ~0.57 deg
    reserve_prop_frac: float = 0.10
    disposal_altitude_trigger: float = 300.0  # km

    def to_array(self) -> list[float]:
        return [self.delta_a, self.delta_Omega, self.delta_lambda,
                self.tau_keep, self.deadband_a, self.deadband_lambda,
                self.reserve_prop_frac, self.disposal_altitude_trigger]

    @classmethod
    def from_array(cls, arr) -> "ClusterPolicy":
        return cls(
            delta_a=float(arr[0]),
            delta_Omega=float(arr[1]),
            delta_lambda=float(arr[2]),
            tau_keep=float(arr[3]),
            deadband_a=float(arr[4]),
            deadband_lambda=float(arr[5]),
            reserve_prop_frac=float(arr[6]),
            disposal_altitude_trigger=float(arr[7]),
        )

    def stable_hash(self) -> str:
        """Deterministic short hash for caching."""
        raw = json.dumps(self.to_array(), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


# ======================================================================
# Policy search bounds
# ======================================================================

@dataclass
class PolicyBounds:
    """Min / max bounds for each policy field (used by Optuna sampler)."""
    delta_a:       Tuple[float, float] = (-10.0, 10.0)          # km
    delta_Omega:   Tuple[float, float] = (-0.05, 0.05)          # rad (~±2.9 deg)
    delta_lambda:  Tuple[float, float] = (-0.10, 0.10)          # rad (~±5.7 deg)
    tau_keep:      Tuple[float, float] = (3600.0, 604800.0)     # 1 h … 7 d
    deadband_a:    Tuple[float, float] = (0.5, 20.0)            # km
    deadband_lambda: Tuple[float, float] = (0.001, 0.10)        # rad
    reserve_prop_frac: Tuple[float, float] = (0.0, 0.40)
    disposal_altitude_trigger: Tuple[float, float] = (200.0, 350.0)  # km


# ======================================================================
# Objective weights and normalisation
# ======================================================================

@dataclass
class ObjectiveWeights:
    """Scalarisation weights and normalisation for the local cluster objective.

    J_g = w_a * J_a + w_lambda * J_lambda + w_m * J_m + w_d * J_d + w_s * J_s
    """
    w_a: float = 1.0
    w_lambda: float = 1.0
    w_m: float = 0.5
    w_d: float = 2.0
    w_s: float = 0.5

    # Normalisation denominators (user-tunable)
    sigma_a: float = 5.0           # km
    sigma_lambda: float = 0.05     # rad

    # Reference altitude for disposal penalty
    h_min_required: float = 120.0  # km (25-year guideline proxy)
    target_disposal_time_s: float = 0.0  # optional; 0 disables

    # Risk-adjustment coefficient  (robust_mode = "risk_adjusted")
    kappa: float = 1.0


# ======================================================================
# Fidelity levels
# ======================================================================

@dataclass
class FidelityConfig:
    """Controls how expensive each evaluation is.

    level : int
        0  proxy screen only (no propagation)
        1  medoid-only, short horizon, coarse checkpoints
        2  representative-pack, moderate horizon
        3  full labeled cluster, long horizon
    """
    level: int = 2
    horizon_fraction: float = 1.0      # fraction of tf to propagate
    checkpoint_stride: int = 100       # extract every N-th time step
    representative_mode: str = "medoid+boundary"   # medoid_only | medoid+boundary | full_members
    propagate: bool = True             # False → proxy screen only
    solver_rtol: float = 1.e-10        # integrator relative tolerance
    solver_atol: float = 1.e-12        # integrator absolute tolerance
    output_stride: int = 1             # thin t_eval grid (1=full, 100=every 100th)

    @classmethod
    def fidelity_0(cls) -> "FidelityConfig":
        return cls(level=0, horizon_fraction=0.0, checkpoint_stride=1,
                   representative_mode="medoid_only", propagate=False)

    @classmethod
    def fidelity_1(cls) -> "FidelityConfig":
        return cls(level=1, horizon_fraction=0.25, checkpoint_stride=200,
                   representative_mode="medoid_only", propagate=True,
                   solver_rtol=1.e-8, solver_atol=1.e-10,
                   output_stride=100)

    @classmethod
    def fidelity_2(cls) -> "FidelityConfig":
        return cls(level=2, horizon_fraction=0.5, checkpoint_stride=100,
                   representative_mode="medoid+boundary", propagate=True,
                   solver_rtol=1.e-9, solver_atol=1.e-11,
                   output_stride=50)

    @classmethod
    def fidelity_3(cls) -> "FidelityConfig":
        return cls(level=3, horizon_fraction=1.0, checkpoint_stride=50,
                   representative_mode="full_members", propagate=True,
                   output_stride=1)


# ======================================================================
# Uncertainty scenarios
# ======================================================================

@dataclass
class UncertaintyScenario:
    """One realisation of environmental / model uncertainty.

    These are drawn once per study (common random numbers) and reused
    across all policy evaluations so that comparisons are fair.
    """
    beta_scale: float = 1.0       # multiplicative factor on ballistic coeff
    thrust_eff_scale: float = 1.0 # multiplicative factor on T_eff_N
    epoch_shift_days: float = 0.0 # shift start epoch (for sensitivity)
    label: str = "nominal"


def build_default_uncertainty_scenarios(
    n_beta: int = 3,
    beta_spread: float = 0.15,
    n_thrust: int = 1,
    thrust_spread: float = 0.0,
    seed: int = 42,
) -> List[UncertaintyScenario]:
    """Build a small fixed set of uncertainty realisations.

    Default: 3 beta scenarios (nominal, +15%, -15%), no thrust variation.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    scenarios: list[UncertaintyScenario] = []
    # Nominal
    scenarios.append(UncertaintyScenario(label="nominal"))

    if n_beta > 1:
        betas = rng.normal(1.0, beta_spread, size=n_beta - 1)
        betas = np.clip(betas, 0.5, 2.0)
        for k, b in enumerate(betas):
            scenarios.append(UncertaintyScenario(
                beta_scale=float(b),
                label=f"beta_{k+1}",
            ))

    if n_thrust > 1 and thrust_spread > 0:
        thrusts = rng.normal(1.0, thrust_spread, size=n_thrust - 1)
        thrusts = np.clip(thrusts, 0.5, 1.5)
        for k, t in enumerate(thrusts):
            scenarios.append(UncertaintyScenario(
                thrust_eff_scale=float(t),
                label=f"thrust_{k+1}",
            ))

    return scenarios


# ======================================================================
# Phase-space (RAAN-vs-phase) regularization
# ======================================================================

@dataclass
class PhaseSpaceConfig:
    """Configuration for the toroidal RAAN-vs-phase regularizer.

    When ``enabled=False`` the regularizer is skipped entirely and all
    downstream code behaves identically to the baseline without any
    phase-space awareness.
    """
    enabled: bool = False
    mode: str = "raan_mean_anomaly"   # raan_mean_anomaly | raan_argument_of_latitude | raan_mean_longitude

    # Torus regularizer sub-term weights  (inside J_torus)
    w_torus_total: float = 1.0        # multiplier on J_torus when added to cluster objective
    w_slot: float = 1.0
    w_gap: float = 0.25
    w_drift: float = 0.25

    # Assignment cost sub-weights  (Omega vs psi relative importance)
    w_raan: float = 1.0
    w_phase: float = 1.0

    # Gap-loss sub-weights
    alpha_gap_raan: float = 1.0
    alpha_gap_phase: float = 1.0

    # Drift-coherence
    kappa_phase_drift: float = 1.0

    # Lattice fitting
    fit_eta: bool = True
    eta_candidates: Optional[Sequence[float]] = None

    # Plane/slot geometry  (None → infer from cluster membership count)
    infer_plane_count: bool = True
    infer_slots_per_plane: bool = True
    n_planes: Optional[int] = None
    slots_per_plane: Optional[int] = None

    # Stitcher torus consistency penalty
    gamma_phase: float = 1.0         # coupling weight for pairwise Psi_phase
    beta_raan: float = 1.0           # sub-weight: omega0 offset
    beta_phase: float = 1.0          # sub-weight: psi0 offset
    beta_eta: float = 1.0            # sub-weight: eta difference

    # Time series stride for drift computation
    checkpoint_stride: int = 100


# ======================================================================
# Top-level study configuration
# ======================================================================

@dataclass
class OptimizationStudyConfig:
    """Master configuration for one constellation optimization study."""

    # Cluster selection
    cluster_ids: Optional[List[int]] = None         # None → all non-noise clusters
    max_clusters: Optional[int] = None               # None → no limit

    # Fidelity ladder used during local search
    local_search_fidelity: FidelityConfig = field(default_factory=FidelityConfig.fidelity_1)
    verification_fidelity: FidelityConfig = field(default_factory=FidelityConfig.fidelity_1)

    # Local search hyperparameters
    n_local_trials: int = 5
    n_candidates_per_cluster: int = 3               # top-K kept per cluster
    policy_bounds: PolicyBounds = field(default_factory=PolicyBounds)
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    # Uncertainty
    n_uncertainty_scenarios: int = 1
    uncertainty_scenarios: Optional[List[UncertaintyScenario]] = None
    robust_mode: str = "mean"  # "mean" | "risk_adjusted"

    # Global stitching
    stitching_method: str = "beam"  # "beam" | "dp" | "enumerate"
    stitching_beam_width: int = 10
    stitching_gamma: float = 1.0
    stitching_R_max: float = 0.01

    # Adjacency graph thresholds
    adjacency_sma_threshold_km: float = 50.0
    adjacency_raan_threshold_deg: float = 30.0

    # Representative packs
    n_boundary_representatives: int = 0

    # Phase-space regularizer
    phase_space: PhaseSpaceConfig = field(default_factory=PhaseSpaceConfig)

    # Execution
    random_seed: int = 42
    max_parallel_workers: Optional[int] = None      # None → use module default
    output_dir: str = "optimization_outputs"

    # TLE data
    tle_data_folders: Optional[List[str]] = None     # None → use module default
    tle_satellite_limit: int = 0                     # 0 → all
    tle_earliest_start_epoch: str = "2019-10-01"
    simulation_date_cutoff: str = "2035-01-01"

    # Cluster CSV paths
    cluster_assignments_csv: Optional[str] = None    # None → use module default
    cluster_stats_csv: Optional[str] = None

    def copy(self) -> "OptimizationStudyConfig":
        return copy.deepcopy(self)
