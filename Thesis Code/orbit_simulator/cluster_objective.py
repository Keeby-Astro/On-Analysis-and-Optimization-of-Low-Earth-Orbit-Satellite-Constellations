"""Cluster objective evaluator for Chapter 7 optimization.

Evaluates a single cluster policy at a specified fidelity level by:
    1. Building simulation cases from the representative pack + policy
    2. Optionally expanding across uncertainty scenarios
    3. Calling the batch propagation API
    4. Computing the cluster objective terms and envelope

Fidelity 0 is a pure proxy screen (no propagation).
Fidelities 1-3 run progressively more complete propagations.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from optimization_config import (
    ClusterPolicy,
    FidelityConfig,
    ObjectiveWeights,
    PhaseSpaceConfig,
    PolicyBounds,
    UncertaintyScenario,
)
from cluster_representatives import RepresentativePack
from cluster_case_builder import (
    build_cluster_cases,
    build_cluster_uncertainty_cases,
    group_results_by_scenario,
)
from cluster_metrics import (
    ClusterObjectiveResult,
    ClusterEnvelope,
    compute_cluster_objective_terms,
    compute_cluster_envelope,
    extract_orbital_timeseries,
)
from cluster_phase_space import (
    TorusRegularizerResult,
    ShellTorusSummary,
    compute_raan_phase_regularizer,
    build_shell_torus_summary,
    extract_raan_phase_timeseries_deg,
)


# ======================================================================
# Evaluation result container
# ======================================================================

@dataclass
class ClusterEvaluation:
    """Full evaluation result for one cluster policy at one fidelity."""
    cluster_id: int = 0
    policy: Optional[ClusterPolicy] = None
    fidelity_level: int = 0
    cost_scalar: float = float("inf")
    cost_terms: Optional[ClusterObjectiveResult] = None
    envelope: Optional[ClusterEnvelope] = None
    n_propagations: int = 0
    feasible: bool = True
    infeasibility_reason: str = ""

    # Per-scenario costs (for robust aggregation)
    scenario_costs: List[float] = field(default_factory=list)

    # Phase-space torus regularizer
    torus_result: Optional[TorusRegularizerResult] = None
    torus_summary: Optional[ShellTorusSummary] = None

    def to_summary_dict(self) -> dict:
        d = {
            "cluster_id": self.cluster_id,
            "fidelity_level": self.fidelity_level,
            "cost_scalar": self.cost_scalar,
            "n_propagations": self.n_propagations,
            "feasible": self.feasible,
        }
        if self.cost_terms is not None:
            d.update(self.cost_terms.to_dict())
        if self.policy is not None:
            d["policy_hash"] = self.policy.stable_hash()
            pa = self.policy.to_array()
            for i, name in enumerate(["delta_a", "delta_Omega", "delta_lambda",
                                       "tau_keep", "deadband_a", "deadband_lambda",
                                       "reserve_prop_frac", "disposal_altitude_trigger"]):
                d[f"policy_{name}"] = pa[i]
        if self.torus_result is not None:
            d.update(self.torus_result.to_dict())
        return d


# ======================================================================
# Evaluation cache (in-process only)
# ======================================================================

_eval_cache: Dict[str, ClusterEvaluation] = {}


def _cache_key(cluster_id: int, policy: ClusterPolicy,
               fidelity: int, seed: int) -> str:
    raw = json.dumps({
        "cid": int(cluster_id),
        "p": [float(x) for x in policy.to_array()],
        "f": int(fidelity),
        "s": int(seed),
    }, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def clear_eval_cache():
    """Clear the in-process evaluation cache."""
    _eval_cache.clear()


# ======================================================================
# Fidelity-0 proxy screen
# ======================================================================

def _proxy_screen(
    policy: ClusterPolicy,
    bounds: Optional[PolicyBounds] = None,
    a_ref: float = 6928.0,
) -> tuple[bool, str]:
    """Quick feasibility check without propagation.

    Returns (feasible, reason_string).
    """
    if bounds is None:
        bounds = PolicyBounds()

    checks = [
        ("delta_a", policy.delta_a, bounds.delta_a),
        ("delta_Omega", policy.delta_Omega, bounds.delta_Omega),
        ("delta_lambda", policy.delta_lambda, bounds.delta_lambda),
        ("tau_keep", policy.tau_keep, bounds.tau_keep),
        ("deadband_a", policy.deadband_a, bounds.deadband_a),
        ("deadband_lambda", policy.deadband_lambda, bounds.deadband_lambda),
        ("reserve_prop_frac", policy.reserve_prop_frac, bounds.reserve_prop_frac),
        ("disposal_altitude_trigger", policy.disposal_altitude_trigger,
         bounds.disposal_altitude_trigger),
    ]
    for name, val, (lo, hi) in checks:
        if val < lo or val > hi:
            return False, f"{name}={val:.6g} outside [{lo}, {hi}]"

    # Check resulting SMA is sane
    earth_Re = 6378.1366
    if a_ref + policy.delta_a < earth_Re + 120.0:
        return False, f"a_ref+delta_a={a_ref + policy.delta_a:.1f} below minimum safe altitude"

    if policy.reserve_prop_frac > 0.99:
        return False, "reserve_prop_frac too high (>0.99)"

    return True, ""


def _build_runtime_policy_meta_map(
    cluster_id: int,
    policy: ClusterPolicy,
    sat_ids: List[str],
    oe_cases: np.ndarray,
    representative_count: int,
) -> dict[str, dict]:
    """Build per-satellite control metadata for the live simulator path."""
    meta_map: dict[str, dict] = {}
    for idx, sat_id in enumerate(sat_ids):
        target_lambda = float((oe_cases[idx, 3] + oe_cases[idx, 4] + oe_cases[idx, 5]) % (2.0 * np.pi))
        meta_map[str(sat_id)] = {
            "cluster_id": int(cluster_id),
            "cluster_weight_active": int(representative_count),
            "cluster_weight_global": int(representative_count),
            "pooled_role": "optimizer_case",
            "is_noise": False,
            "representative_sat_id": str(sat_id),
            "color_hex": "",
            "cluster_policy_applied": True,
            "cluster_policy_source": "cluster_objective",
            "policy_delta_a_km": float(policy.delta_a),
            "policy_delta_Omega_rad": float(policy.delta_Omega),
            "policy_delta_lambda_rad": float(policy.delta_lambda),
            "policy_tau_keep_s": float(policy.tau_keep),
            "policy_deadband_a_km": float(policy.deadband_a),
            "policy_deadband_lambda_rad": float(policy.deadband_lambda),
            "policy_reserve_prop_frac": float(policy.reserve_prop_frac),
            "policy_disposal_altitude_trigger_km": float(policy.disposal_altitude_trigger),
            "target_a_km": float(oe_cases[idx, 0]),
            "target_raan_rad": float(oe_cases[idx, 4]),
            "target_mean_anomaly_rad": float(oe_cases[idx, 5]),
            "target_lambda_rad": target_lambda,
        }
    return meta_map


# ======================================================================
# Main evaluator
# ======================================================================

def evaluate_cluster_policy(
    cluster_id: int,
    policy: ClusterPolicy,
    pack: RepresentativePack,
    tle_latest_df: pd.DataFrame,
    fidelity: FidelityConfig,
    weights: ObjectiveWeights,
    *,
    a_ref: float = 6928.0,
    lambda_ref: float = 0.0,
    uncertainty_scenarios: Optional[List[UncertaintyScenario]] = None,
    robust_mode: str = "mean",
    seed: int = 42,
    nominal_ballistic_coef: float = 0.0334,
    policy_bounds: Optional[PolicyBounds] = None,
    workers_override: Optional[int] = None,
    use_cache: bool = True,
    phase_space_config: Optional[PhaseSpaceConfig] = None,
) -> ClusterEvaluation:
    """Evaluate one cluster policy at a specified fidelity level.

    Parameters
    ----------
    cluster_id : int
        Global cluster ID.
    policy : ClusterPolicy
        Policy vector to evaluate.
    pack : RepresentativePack
        Representative members for this cluster.
    tle_latest_df : DataFrame
        Source TLE data.
    fidelity : FidelityConfig
        Controls propagation depth.
    weights : ObjectiveWeights
        Objective scalarisation weights.
    a_ref : float
        Reference SMA for this cluster (km).
    lambda_ref : float
        Reference mean longitude (rad).
    uncertainty_scenarios : list[UncertaintyScenario] | None
        If provided, expand cases and aggregate robustly.
    robust_mode : str
        ``"mean"`` or ``"risk_adjusted"`` (mean + kappa * std).
    seed : int
        For deterministic scenario expansion.
    nominal_ballistic_coef : float
        Default Cd*A/m (m^2/kg).
    policy_bounds : PolicyBounds | None
        For Fidelity-0 proxy screen.
    workers_override : int | None
        Max parallel propagation workers.
    use_cache : bool
        Check/populate the in-process evaluation cache.

    Returns
    -------
    ClusterEvaluation
    """
    # ---- Cache lookup ----
    ck = _cache_key(cluster_id, policy, fidelity.level, seed)
    if use_cache and ck in _eval_cache:
        return _eval_cache[ck]

    # ---- Fidelity 0: proxy screen ----
    if fidelity.level == 0 or not fidelity.propagate:
        feasible, reason = _proxy_screen(policy, policy_bounds, a_ref)
        ev = ClusterEvaluation(
            cluster_id=cluster_id,
            policy=policy,
            fidelity_level=0,
            cost_scalar=0.0 if feasible else float("inf"),
            feasible=feasible,
            infeasibility_reason=reason,
            n_propagations=0,
        )
        if use_cache:
            _eval_cache[ck] = ev
        return ev

    # ---- Build base cases ----
    oe_cases, sat_ids, start_timestamps, ballistic_coeffs = build_cluster_cases(
        pack, policy, tle_latest_df, fidelity,
        nominal_ballistic_coef=nominal_ballistic_coef,
    )

    # ---- Uncertainty expansion ----
    scenarios = uncertainty_scenarios or [UncertaintyScenario(label="nominal")]
    n_scen = len(scenarios)

    if n_scen > 1:
        oe_cases, sat_ids, start_timestamps, ballistic_coeffs, scen_idx = \
            build_cluster_uncertainty_cases(
                oe_cases, sat_ids, start_timestamps, ballistic_coeffs,
                scenarios, seed=seed,
            )
    else:
        scen_idx = np.zeros(oe_cases.shape[0], dtype=np.int32)

    # ---- Compute propagation time cap from horizon_fraction ----
    max_prop_s = None
    if fidelity.horizon_fraction < 1.0:
        # tf (seconds) is the max propagation duration from the simulator globals.
        # Import lazily to avoid circular imports.
        from constellation_simulator_optimized_thrust import tf as _tf_global
        max_prop_s = fidelity.horizon_fraction * _tf_global

    # ---- Run propagation via batch API ----
    # Import here to avoid circular import at module level
    from constellation_simulator_optimized_thrust import run_batch_cases

    runtime_policy_meta = _build_runtime_policy_meta_map(
        cluster_id,
        policy,
        sat_ids,
        oe_cases,
        len(pack.representative_sat_ids),
    )

    results = run_batch_cases(
        oe_cases=oe_cases,
        sat_ids=sat_ids,
        start_timestamps=start_timestamps,
        ballistic_coefficients=ballistic_coeffs,
        cluster_meta_map=runtime_policy_meta,
        write_outputs=False,
        return_trajectories=True,
        return_mass_series=True,
        workers_override=workers_override,
        solver_rtol=fidelity.solver_rtol,
        solver_atol=fidelity.solver_atol,
        max_prop_time_s=max_prop_s,
        output_stride=fidelity.output_stride,
    )

    n_propagations = len(results)

    # ---- Compute per-scenario objectives ----
    if n_scen > 1:
        scenario_groups = group_results_by_scenario(results, scen_idx, n_scen)
    else:
        scenario_groups = [results]

    scenario_costs: list[float] = []
    scenario_terms: list[ClusterObjectiveResult] = []

    for group in scenario_groups:
        obj = compute_cluster_objective_terms(
            group, weights, a_ref, lambda_ref,
            checkpoint_stride=fidelity.checkpoint_stride,
        )
        scenario_costs.append(obj.J_total)
        scenario_terms.append(obj)

    # ---- Aggregate across scenarios ----
    sc_arr = np.array(scenario_costs, dtype=np.float64)
    if robust_mode == "risk_adjusted" and len(sc_arr) > 1:
        cost_scalar = float(np.mean(sc_arr) + weights.kappa * np.std(sc_arr))
    else:
        cost_scalar = float(np.mean(sc_arr))

    # Use the nominal (first) scenario for the primary terms
    primary_terms = scenario_terms[0] if scenario_terms else ClusterObjectiveResult()

    # ---- Compute envelope (from nominal scenario) ----
    envelope = None
    if scenario_groups and scenario_groups[0]:
        z_list = []
        t_ck = None
        for r in scenario_groups[0]:
            if "times" in r and "state_sat" in r and r["times"].size >= 2:
                tc, zc = extract_orbital_timeseries(r, checkpoint_stride=fidelity.checkpoint_stride)
                z_list.append(zc)
                if t_ck is None:
                    t_ck = tc
        if z_list and t_ck is not None:
            envelope = compute_cluster_envelope(z_list, t_ck, cluster_id=cluster_id)

    # ---- Phase-space torus regularizer (optional) ----
    torus_result = None
    torus_summary = None
    if phase_space_config is not None and phase_space_config.enabled:
        psc = phase_space_config
        # Determine lattice geometry
        n_pl = psc.n_planes if psc.n_planes is not None else max(1, len(scenario_groups[0]) if scenario_groups else 1)
        spp = psc.slots_per_plane if psc.slots_per_plane is not None else max(1, len(scenario_groups[0]) if scenario_groups else 1)

        # Use nominal scenario results for torus computation
        nominal_results = scenario_groups[0] if scenario_groups else results
        torus_result = compute_raan_phase_regularizer(
            nominal_results,
            mode=psc.mode,
            n_planes=n_pl,
            slots_per_plane=spp,
            w_slot=psc.w_slot,
            w_gap=psc.w_gap,
            w_drift=psc.w_drift,
            w_raan=psc.w_raan,
            w_phase=psc.w_phase,
            alpha_gap_raan=psc.alpha_gap_raan,
            alpha_gap_phase=psc.alpha_gap_phase,
            kappa_phase_drift=psc.kappa_phase_drift,
            fit_eta=psc.fit_eta,
            checkpoint_stride=psc.checkpoint_stride,
        )
        cost_scalar += psc.w_torus_total * torus_result.J_torus

        # Build torus summary for stitcher coordination
        if torus_result.lattice is not None and len(nominal_results) > 0:
            raan_snap = []
            phase_snap = []
            for r in nominal_results:
                if "times" in r and "state_sat" in r and r["times"].size >= 2:
                    _, raan_ts, phase_ts = extract_raan_phase_timeseries_deg(
                        r, mode=psc.mode, checkpoint_stride=max(r["times"].size, 1),
                    )
                    raan_snap.append(raan_ts[0])
                    phase_snap.append(phase_ts[0])
            if raan_snap:
                torus_summary = build_shell_torus_summary(
                    cluster_id, torus_result,
                    np.array(raan_snap), np.array(phase_snap),
                )

    ev = ClusterEvaluation(
        cluster_id=cluster_id,
        policy=policy,
        fidelity_level=fidelity.level,
        cost_scalar=cost_scalar,
        cost_terms=primary_terms,
        envelope=envelope,
        n_propagations=n_propagations,
        feasible=np.isfinite(cost_scalar),
        scenario_costs=scenario_costs,
        torus_result=torus_result,
        torus_summary=torus_summary,
    )

    if use_cache:
        _eval_cache[ck] = ev

    return ev
