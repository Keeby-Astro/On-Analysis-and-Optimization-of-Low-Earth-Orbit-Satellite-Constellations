"""Local cluster policy optimizer for Chapter 7.

Runs a black-box search over the cluster policy space using Optuna TPE.
Returns the top-K distinct candidate policies per cluster sorted by cost.

The search backend is deliberately isolated behind a simple interface
so it can later be swapped for BoTorch TuRBO, multi-fidelity BO, or
Ray Tune without changing the outer orchestration.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

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
from cluster_objective import ClusterEvaluation, evaluate_cluster_policy, clear_eval_cache


# ======================================================================
# Optuna-based local search
# ======================================================================

def optimize_cluster_policy(
    cluster_id: int,
    pack: RepresentativePack,
    tle_latest_df: pd.DataFrame,
    *,
    bounds: PolicyBounds = PolicyBounds(),
    weights: ObjectiveWeights = ObjectiveWeights(),
    fidelity: FidelityConfig = FidelityConfig.fidelity_1(),
    n_trials: int = 50,
    n_top_k: int = 5,
    seed: int = 42,
    a_ref: float = 6928.0,
    lambda_ref: float = 0.0,
    uncertainty_scenarios: Optional[List[UncertaintyScenario]] = None,
    robust_mode: str = "mean",
    nominal_ballistic_coef: float = 0.0334,
    workers_override: Optional[int] = None,
    verbose: bool = True,
    phase_space_config: Optional[PhaseSpaceConfig] = None,
) -> List[Tuple[ClusterPolicy, ClusterEvaluation]]:
    """Run local optimization for a single cluster.

    Parameters
    ----------
    cluster_id : int
        Global cluster ID.
    pack : RepresentativePack
        Representative set for this cluster.
    tle_latest_df : DataFrame
        Source TLE data.
    bounds : PolicyBounds
        Search space bounds.
    weights : ObjectiveWeights
        Objective weights.
    fidelity : FidelityConfig
        Propagation fidelity for evaluations.
    n_trials : int
        Number of Optuna trials.
    n_top_k : int
        Return the best K distinct candidates.
    seed : int
        Random seed for reproducibility.
    a_ref : float
        Reference SMA (km).
    lambda_ref : float
        Reference mean longitude (rad).
    uncertainty_scenarios : list | None
        Uncertainty realisations.
    robust_mode : str
        ``"mean"`` or ``"risk_adjusted"``.
    nominal_ballistic_coef : float
        Default Cd*A/m.
    workers_override : int | None
        Max parallel workers per batch.
    verbose : bool
        Print progress.

    Returns
    -------
    tuple[list[tuple[ClusterPolicy, ClusterEvaluation]], list[tuple[ClusterPolicy, ClusterEvaluation]]]
        (top_k, all_evaluations) — top-K unique sorted + full trial history.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "Optuna is required for local cluster optimization. "
            "Install it with: pip install optuna"
        ) from e

    # Suppress Optuna trial logs unless verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Collect all evaluations for top-K extraction
    all_evaluations: list[tuple[ClusterPolicy, ClusterEvaluation]] = []

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: sample policy → evaluate → return cost."""
        policy = ClusterPolicy(
            delta_a=trial.suggest_float(
                "delta_a", bounds.delta_a[0], bounds.delta_a[1]),
            delta_Omega=trial.suggest_float(
                "delta_Omega", bounds.delta_Omega[0], bounds.delta_Omega[1]),
            delta_lambda=trial.suggest_float(
                "delta_lambda", bounds.delta_lambda[0], bounds.delta_lambda[1]),
            tau_keep=trial.suggest_float(
                "tau_keep", bounds.tau_keep[0], bounds.tau_keep[1], log=True),
            deadband_a=trial.suggest_float(
                "deadband_a", bounds.deadband_a[0], bounds.deadband_a[1]),
            deadband_lambda=trial.suggest_float(
                "deadband_lambda", bounds.deadband_lambda[0],
                bounds.deadband_lambda[1]),
            reserve_prop_frac=trial.suggest_float(
                "reserve_prop_frac", bounds.reserve_prop_frac[0],
                bounds.reserve_prop_frac[1]),
            disposal_altitude_trigger=trial.suggest_float(
                "disposal_altitude_trigger",
                bounds.disposal_altitude_trigger[0],
                bounds.disposal_altitude_trigger[1]),
        )

        ev = evaluate_cluster_policy(
            cluster_id=cluster_id,
            policy=policy,
            pack=pack,
            tle_latest_df=tle_latest_df,
            fidelity=fidelity,
            weights=weights,
            a_ref=a_ref,
            lambda_ref=lambda_ref,
            uncertainty_scenarios=uncertainty_scenarios,
            robust_mode=robust_mode,
            seed=seed,
            nominal_ballistic_coef=nominal_ballistic_coef,
            policy_bounds=bounds,
            workers_override=workers_override,
            use_cache=True,
            phase_space_config=phase_space_config,
        )

        all_evaluations.append((policy, ev))

        if verbose and (len(all_evaluations) % 5 == 0 or len(all_evaluations) == 1):
            print(f"  [Cluster {cluster_id}] Trial {len(all_evaluations)}/{n_trials} "
                  f"| cost={ev.cost_scalar:.6f} "
                  f"| feasible={ev.feasible}")

        if not ev.feasible:
            return float("inf")

        return ev.cost_scalar

    # Create study with TPE sampler (deterministic)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"cluster_{cluster_id}",
    )

    # Always evaluate the zero (nominal) policy first as a baseline
    baseline = ClusterPolicy()
    study.enqueue_trial({
        "delta_a": baseline.delta_a,
        "delta_Omega": baseline.delta_Omega,
        "delta_lambda": baseline.delta_lambda,
        "tau_keep": baseline.tau_keep,
        "deadband_a": baseline.deadband_a,
        "deadband_lambda": baseline.deadband_lambda,
        "reserve_prop_frac": baseline.reserve_prop_frac,
        "disposal_altitude_trigger": baseline.disposal_altitude_trigger,
    })

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # ---- Extract top-K ----
    # Preserve execution order for convergence tracking
    execution_order = list(all_evaluations)  # copy before sort

    # Sort by cost, take unique policies
    all_evaluations.sort(key=lambda x: x[1].cost_scalar)

    seen_hashes: set[str] = set()
    top_k: list[tuple[ClusterPolicy, ClusterEvaluation]] = []
    for policy, ev in all_evaluations:
        h = policy.stable_hash()
        if h not in seen_hashes and ev.feasible:
            seen_hashes.add(h)
            top_k.append((policy, ev))
            if len(top_k) >= n_top_k:
                break

    # If we don't have enough feasible, include infeasible ones
    if len(top_k) < n_top_k:
        for policy, ev in all_evaluations:
            h = policy.stable_hash()
            if h not in seen_hashes:
                seen_hashes.add(h)
                top_k.append((policy, ev))
                if len(top_k) >= n_top_k:
                    break

    if verbose:
        best_cost = top_k[0][1].cost_scalar if top_k else float("inf")
        print(f"  [Cluster {cluster_id}] Optimization done: "
              f"{len(top_k)} candidates, best cost={best_cost:.6f}")

    return top_k, execution_order


def candidates_to_dataframe(
    candidates_by_cluster: Dict[int, List[Tuple[ClusterPolicy, ClusterEvaluation]]],
) -> pd.DataFrame:
    """Flatten all cluster candidate results into a summary DataFrame."""
    rows: list[dict] = []
    for cid, candidates in sorted(candidates_by_cluster.items()):
        for rank, (policy, ev) in enumerate(candidates):
            d = ev.to_summary_dict()
            d["candidate_rank"] = rank
            rows.append(d)
    return pd.DataFrame(rows)
