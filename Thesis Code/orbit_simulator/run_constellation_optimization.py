#!/usr/bin/env python3
"""Chapter 7 — Constellation-Level Optimization under Environmental Uncertainty.

Top-level orchestration script that:
    1. Loads TLE data and cluster assignments
    2. Builds representative packs per cluster
    3. Runs local policy optimization per cluster (Optuna TPE)
    4. Stitches local candidates into a global constellation design
    5. Verifies the elite stitched solution at higher fidelity
    6. Saves study outputs and lightweight diagnostics

Usage (from orbit_simulator/):
    python run_constellation_optimization.py

All knobs are in the ``STUDY_CONFIG`` block below.  The script is
designed to be runnable on a single workstation.
"""

from __future__ import annotations

import json
import os
import sys
import timeit
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Chapter 7 modules ──
from optimization_config import (
    ClusterPolicy,
    FidelityConfig,
    ObjectiveWeights,
    OptimizationStudyConfig,
    PhaseSpaceConfig,
    PolicyBounds,
    UncertaintyScenario,
    build_default_uncertainty_scenarios,
)
from cluster_representatives import (
    RepresentativePack,
    build_cluster_representatives,
    load_cluster_assignments,
    load_cluster_stats,
    representatives_to_dataframe,
)
from cluster_objective import (
    ClusterEvaluation,
    clear_eval_cache,
    evaluate_cluster_policy,
)
from cluster_local_search import (
    candidates_to_dataframe,
    optimize_cluster_policy,
)
from cluster_stitcher import (
    build_cluster_adjacency_graph,
    stitch_cluster_policies,
    stitching_result_to_dataframe,
    StitchingResult,
)

# ── Existing simulator helpers ──
from load_all_tle_data import load_all_tle_data

# ======================================================================
# STUDY CONFIGURATION  (edit these knobs for your study)
# ======================================================================

STUDY_CONFIG = OptimizationStudyConfig(
    # Cluster selection — None = all non-noise clusters, or specify IDs
    cluster_ids=None,
    max_clusters=None,  # all 93 non-noise clusters (~2.5 min/cluster → ~4 h)

    # Fidelity ladder  (measured on 2-sat benchmark)
    #   ~10 s/trial with 2 reps in parallel on fidelity_1 settings
    #   Verification at fidelity_2: ~25 s/sat, 50% horizon
    local_search_fidelity=FidelityConfig(
        level=1, horizon_fraction=0.25, checkpoint_stride=100,
        representative_mode="medoid+boundary", propagate=True,
        solver_rtol=1.e-8, solver_atol=1.e-10, output_stride=100),
    verification_fidelity=FidelityConfig.fidelity_2(),

    # Local search — 93 clusters × 10 trials × ~10 s/trial ≈ 155 min
    #   + verification: 93 clusters × 3 reps × ~25 s ≈ 38 min
    #   Total ≈ ~3.5 h  (threaded reps cut per-trial time ~2×)
    n_local_trials=10,
    n_candidates_per_cluster=5,
    policy_bounds=PolicyBounds(),
    # Rebalanced objective weights:
    #   sigma_a=50 km and sigma_lambda=0.5 rad normalise geometry terms
    #   to ~10, comparable to J_m (~3 kg) and J_d (~20).
    #   Increased w_m and w_s to give mass/spread meaningful influence.
    objective_weights=ObjectiveWeights(
        w_a=1.0, w_lambda=1.0, w_m=5.0, w_d=1.0, w_s=2.0,
        sigma_a=50.0, sigma_lambda=0.5,
    ),

    # Uncertainty — single nominal scenario for speed (no beta expansion)
    n_uncertainty_scenarios=1,
    robust_mode="mean",

    # Stitching — gamma increased to make coupling nontrivial
    stitching_method="beam",
    stitching_beam_width=10,
    stitching_gamma=10.0,
    stitching_R_max=0.01,
    adjacency_sma_threshold_km=50.0,
    adjacency_raan_threshold_deg=30.0,

    # Representatives — medoid + 2 boundary members for spread / coupling
    n_boundary_representatives=2,

    # Phase-space (RAAN-vs-phase) torus regularizer
    phase_space=PhaseSpaceConfig(enabled=True),

    # Execution
    random_seed=42,
    max_parallel_workers=12,
    output_dir="optimization_outputs",

    # TLE data — None uses module defaults from the simulator
    tle_data_folders=None,
    tle_satellite_limit=0,
    tle_earliest_start_epoch="2019-10-01",
    simulation_date_cutoff="2035-01-01",

    # Cluster CSVs — None uses module defaults
    cluster_assignments_csv=None,
    cluster_stats_csv=None,
)


# ======================================================================
# Helpers
# ======================================================================

def _resolve_path(p: Optional[str], fallback_name: str, base: Path) -> Path:
    """Resolve a user path or fall back to workspace default."""
    if p is not None:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return base / pp
    return base / "global_analysis" / "tables" / fallback_name


def _load_tle_data(cfg: OptimizationStudyConfig) -> pd.DataFrame:
    """Load and prepare TLE data for the study."""
    base = Path(__file__).resolve().parent.parent
    if cfg.tle_data_folders is not None:
        folders = cfg.tle_data_folders
    else:
        folders = [str(base / "starlink_backup")]

    # Ensure absolute paths
    abs_folders = []
    for f in folders:
        fp = Path(f)
        if not fp.is_absolute():
            fp = base / fp
        abs_folders.append(str(fp))

    tle_df, _ = load_all_tle_data(abs_folders, derived={"sma", "specific_angular_momentum"})
    if tle_df is None or tle_df.empty:
        raise RuntimeError("No TLE data loaded. Check tle_data_folders.")

    # Take latest TLE per satellite
    tle_latest = (
        tle_df.sort_values("timestamp")
        .groupby("sat_id", as_index=False)
        .head(1)
        .sort_values("sat_id")
        .reset_index(drop=True)
    )

    if cfg.tle_satellite_limit > 0:
        tle_latest = tle_latest.iloc[: cfg.tle_satellite_limit].copy()

    print(f"[TLE] Loaded {len(tle_latest)} satellites from {len(abs_folders)} folder(s)")
    return tle_latest


# ======================================================================
# Main study orchestration
# ======================================================================

def run_study(cfg: OptimizationStudyConfig) -> dict:
    """Execute the full constellation optimization study.

    Returns a summary dict suitable for JSON serialisation.
    """
    t0_study = timeit.default_timer()
    base = Path(__file__).resolve().parent.parent

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Chapter 7: Constellation-Level Optimization")
    print("=" * 60)

    tle_latest = _load_tle_data(cfg)

    cluster_csv = _resolve_path(cfg.cluster_assignments_csv,
                                "combined_cluster_labels_global.csv", base)
    stats_csv = _resolve_path(cfg.cluster_stats_csv,
                              "global_cluster_stats.csv", base)

    cluster_assignments = load_cluster_assignments(cluster_csv)
    cluster_stats = load_cluster_stats(stats_csv)

    print(f"[Cluster] Loaded {len(cluster_assignments)} assignments, "
          f"{len(cluster_stats)} cluster stats entries")

    # ----------------------------------------------------------------
    # 2. Build representative packs
    # ----------------------------------------------------------------
    print("\n--- Building representative packs ---")

    packs = build_cluster_representatives(
        tle_latest,
        cluster_assignments,
        mode=cfg.local_search_fidelity.representative_mode,
        n_boundary=cfg.n_boundary_representatives,
        include_beta=True,
        include_h=True,
        cluster_ids=cfg.cluster_ids,
        verbose=True,
    )

    if not packs:
        raise RuntimeError("No cluster representative packs built. Check cluster CSVs.")

    # Limit number of clusters
    if cfg.max_clusters is not None and len(packs) > cfg.max_clusters:
        # Keep clusters with most members
        sorted_cids = sorted(packs.keys(),
                             key=lambda c: packs[c].n_members, reverse=True)
        packs = {c: packs[c] for c in sorted_cids[: cfg.max_clusters]}
        print(f"[Cluster] Trimmed to top {cfg.max_clusters} clusters by size")

    # Save representative packs CSV
    reps_df = representatives_to_dataframe(packs)
    reps_csv = os.path.join(cfg.output_dir, "cluster_representatives.csv")
    reps_df.to_csv(reps_csv, index=False)
    print(f"[Output] {reps_csv}")

    # ----------------------------------------------------------------
    # 3. Build uncertainty scenarios
    # ----------------------------------------------------------------
    if cfg.uncertainty_scenarios is not None:
        scenarios = cfg.uncertainty_scenarios
    else:
        scenarios = build_default_uncertainty_scenarios(
            n_beta=cfg.n_uncertainty_scenarios,
            seed=cfg.random_seed,
        )
    print(f"[Uncertainty] {len(scenarios)} scenarios: "
          f"{[s.label for s in scenarios]}")

    # ----------------------------------------------------------------
    # 4. Local cluster optimization
    # ----------------------------------------------------------------
    print("\n--- Local cluster optimization ---")

    candidate_menus: dict[int, list[tuple[ClusterPolicy, ClusterEvaluation]]] = {}
    trial_histories: dict[int, list[tuple[ClusterPolicy, ClusterEvaluation]]] = {}
    all_candidates_flat: list[dict] = []

    for cid, pack in sorted(packs.items()):
        # Determine reference SMA from cluster stats or pack
        stats = cluster_stats.get(cid, {})
        a_ref = float(stats.get("sma_mean", 6928.0))

        # Determine reference mean longitude from first member
        lambda_ref = 0.0  # default; could be computed from TLE data

        print(f"\n  Cluster {cid} ({pack.n_members} members, "
              f"a_ref={a_ref:.1f} km)")

        clear_eval_cache()

        candidates, all_trials = optimize_cluster_policy(
            cluster_id=cid,
            pack=pack,
            tle_latest_df=tle_latest,
            bounds=cfg.policy_bounds,
            weights=cfg.objective_weights,
            fidelity=cfg.local_search_fidelity,
            n_trials=cfg.n_local_trials,
            n_top_k=cfg.n_candidates_per_cluster,
            seed=cfg.random_seed + cid,
            a_ref=a_ref,
            lambda_ref=lambda_ref,
            uncertainty_scenarios=scenarios if len(scenarios) > 1 else None,
            robust_mode=cfg.robust_mode,
            workers_override=cfg.max_parallel_workers,
            verbose=True,
            phase_space_config=cfg.phase_space if cfg.phase_space.enabled else None,
        )

        candidate_menus[cid] = candidates
        trial_histories[cid] = all_trials

        # Collect flat summary
        for rank, (pol, ev) in enumerate(candidates):
            d = ev.to_summary_dict()
            d["candidate_rank"] = rank
            all_candidates_flat.append(d)

    # Save local candidates CSV
    cands_df = pd.DataFrame(all_candidates_flat)
    cands_csv = os.path.join(cfg.output_dir, "cluster_policy_candidates.csv")
    cands_df.to_csv(cands_csv, index=False)
    print(f"\n[Output] {cands_csv}")

    # Save local objectives CSV (best per cluster)
    obj_rows = []
    for cid, cands in sorted(candidate_menus.items()):
        if cands:
            _, best_ev = cands[0]
            d = best_ev.to_summary_dict()
            d["candidate_rank"] = 0
            obj_rows.append(d)
    obj_df = pd.DataFrame(obj_rows)
    obj_csv = os.path.join(cfg.output_dir, "cluster_local_objectives.csv")
    obj_df.to_csv(obj_csv, index=False)
    print(f"[Output] {obj_csv}")

    # Save trial convergence CSV (full optimization history)
    trial_rows: list[dict] = []
    for cid, trials in sorted(trial_histories.items()):
        best_so_far = float("inf")
        for trial_idx, (pol, ev) in enumerate(trials):
            best_so_far = min(best_so_far, ev.cost_scalar)
            row = {
                "cluster_id": cid,
                "trial": trial_idx,
                "cost": ev.cost_scalar,
                "best_cost_so_far": best_so_far,
                "feasible": ev.feasible,
            }
            if ev.cost_terms is not None:
                row.update(ev.cost_terms.to_dict())
            trial_rows.append(row)
    trial_df = pd.DataFrame(trial_rows)
    trial_csv = os.path.join(cfg.output_dir, "trial_convergence.csv")
    trial_df.to_csv(trial_csv, index=False)
    print(f"[Output] {trial_csv}")

    # ----------------------------------------------------------------
    # 5. Global stitching
    # ----------------------------------------------------------------
    print("\n--- Global stitching ---")

    adjacency = build_cluster_adjacency_graph(
        cluster_stats,
        sma_threshold_km=cfg.adjacency_sma_threshold_km,
        raan_threshold_deg=cfg.adjacency_raan_threshold_deg,
    )
    print(f"[Adjacency] {len(adjacency)} edges among "
          f"{len(candidate_menus)} clusters")
    if adjacency:
        # Log strongest edges
        top_edges = sorted(adjacency.items(), key=lambda x: x[1], reverse=True)[:10]
        for (g, h), w in top_edges:
            print(f"  Edge ({g},{h}): proximity={w:.3f}")
    else:
        print("  WARNING: adjacency graph is empty — no inter-cluster coupling")

    stitch_result = stitch_cluster_policies(
        candidate_menus,
        adjacency,
        gamma=cfg.stitching_gamma,
        R_max=cfg.stitching_R_max,
        method=cfg.stitching_method,
        beam_width=cfg.stitching_beam_width,
        verbose=True,
        phase_space_config=cfg.phase_space if cfg.phase_space.enabled else None,
    )
    print(f"[Stitch] J_global = {stitch_result.total_cost:.2f} "
          f"= J_local({stitch_result.local_cost:.2f}) "
          f"+ gamma*J_coupling({stitch_result.coupling_cost:.2f})"
          f"+ J_phase_coupling({stitch_result.phase_coupling_cost:.2f})")
    # Log which candidate index was selected per cluster
    non_zero_selections = sum(1 for k in stitch_result.selected.values() if k > 0)
    print(f"[Stitch] {non_zero_selections}/{len(stitch_result.selected)} "
          f"clusters selected candidate != 0")

    # Save stitching summary CSV
    stitch_df = stitching_result_to_dataframe(stitch_result)
    stitch_csv = os.path.join(cfg.output_dir, "cluster_stitching_summary.csv")
    stitch_df.to_csv(stitch_csv, index=False)
    print(f"[Output] {stitch_csv}")

    # Save cluster overview CSV (cluster stats + optimization result merged)
    overview_rows: list[dict] = []
    for cid in sorted(packs.keys()):
        row: dict = {"cluster_id": cid, "n_members": packs[cid].n_members,
                      "medoid_sat_id": packs[cid].medoid_sat_id}
        st = cluster_stats.get(cid, {})
        for k in ("sma_mean", "sma_std", "inc_mean", "raan_mean", "ecc_mean"):
            row[k] = st.get(k)
        ev = stitch_result.selected_evaluations.get(cid)
        if ev is not None:
            row["cost_scalar"] = ev.cost_scalar
            row["feasible"] = ev.feasible
            if ev.cost_terms:
                row.update({f"obj_{k}": v for k, v in ev.cost_terms.to_dict().items()
                            if k != "n_representatives" and k != "n_checkpoints"})
        pol = stitch_result.selected_policies.get(cid)
        if pol is not None:
            for i, name in enumerate(["delta_a", "delta_Omega", "delta_lambda",
                                       "tau_keep", "deadband_a", "deadband_lambda",
                                       "reserve_prop_frac", "disposal_altitude_trigger"]):
                row[f"policy_{name}"] = pol.to_array()[i]
        overview_rows.append(row)
    overview_df = pd.DataFrame(overview_rows)
    overview_csv = os.path.join(cfg.output_dir, "cluster_overview.csv")
    overview_df.to_csv(overview_csv, index=False)
    print(f"[Output] {overview_csv}")

    # Save human-readable stitched policy table CSV
    policy_table_rows: list[dict] = []
    policy_names = ["delta_a", "delta_Omega", "delta_lambda", "tau_keep",
                    "deadband_a", "deadband_lambda", "reserve_prop_frac",
                    "disposal_altitude_trigger"]
    policy_units = ["km", "rad", "rad", "s", "km", "rad", "", "km"]
    for cid in sorted(stitch_result.selected_policies.keys()):
        pol = stitch_result.selected_policies[cid]
        ev = stitch_result.selected_evaluations.get(cid)
        arr = pol.to_array()
        row = {"cluster_id": cid}
        for name, val, unit in zip(policy_names, arr, policy_units):
            row[f"{name} ({unit})".rstrip(" ()")] = val
        row["total_cost"] = ev.cost_scalar if ev else None
        row["propellant_kg"] = ev.cost_terms.J_m if ev and ev.cost_terms else None
        policy_table_rows.append(row)
    policy_table_df = pd.DataFrame(policy_table_rows)
    policy_table_csv = os.path.join(cfg.output_dir, "stitched_policy_table.csv")
    policy_table_df.to_csv(policy_table_csv, index=False)
    print(f"[Output] {policy_table_csv}")

    # ----------------------------------------------------------------
    # 6. Higher-fidelity verification of elite solution
    # ----------------------------------------------------------------
    print("\n--- Elite verification ---")

    verification_results: dict[int, ClusterEvaluation] = {}
    if stitch_result.feasible and cfg.verification_fidelity.level > cfg.local_search_fidelity.level:
        for cid, pol in sorted(stitch_result.selected_policies.items()):
            pack = packs.get(cid)
            if pack is None:
                continue

            stats = cluster_stats.get(cid, {})
            a_ref = float(stats.get("sma_mean", 6928.0))

            print(f"  Verifying cluster {cid} at fidelity {cfg.verification_fidelity.level}...")

            ev = evaluate_cluster_policy(
                cluster_id=cid,
                policy=pol,
                pack=pack,
                tle_latest_df=tle_latest,
                fidelity=cfg.verification_fidelity,
                weights=cfg.objective_weights,
                a_ref=a_ref,
                uncertainty_scenarios=scenarios if len(scenarios) > 1 else None,
                robust_mode=cfg.robust_mode,
                seed=cfg.random_seed + cid + 10000,
                workers_override=cfg.max_parallel_workers,
                use_cache=False,
                phase_space_config=cfg.phase_space if cfg.phase_space.enabled else None,
            )
            verification_results[cid] = ev
            print(f"    Cluster {cid}: verified cost={ev.cost_scalar:.6f} "
                  f"(search cost={stitch_result.selected_evaluations[cid].cost_scalar:.6f})")
    else:
        print("  Skipping verification (same fidelity or infeasible stitch).")

    # ----------------------------------------------------------------
    # 7. Save global summary
    # ----------------------------------------------------------------
    t_total = timeit.default_timer() - t0_study

    summary = {
        "study_config": {
            "n_clusters": len(packs),
            "cluster_ids": sorted(packs.keys()),
            "n_local_trials": cfg.n_local_trials,
            "n_candidates_per_cluster": cfg.n_candidates_per_cluster,
            "local_fidelity": cfg.local_search_fidelity.level,
            "verification_fidelity": cfg.verification_fidelity.level,
            "n_uncertainty_scenarios": len(scenarios),
            "robust_mode": cfg.robust_mode,
            "stitching_method": cfg.stitching_method,
            "random_seed": cfg.random_seed,
        },
        "stitching": stitch_result.to_summary_dict(),
        "verification": {
            str(cid): {
                "cost_scalar": ev.cost_scalar,
                "feasible": ev.feasible,
                "n_propagations": ev.n_propagations,
            }
            for cid, ev in verification_results.items()
        },
        "runtime_seconds": round(t_total, 2),
        "output_files": [
            reps_csv, cands_csv, obj_csv, trial_csv,
            stitch_csv, overview_csv, policy_table_csv,
        ],
    }

    summary_path = os.path.join(cfg.output_dir, "constellation_optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[Output] {summary_path}")

    # ----------------------------------------------------------------
    # 8. Optional lightweight diagnostic plots
    # ----------------------------------------------------------------
    _plot_study_diagnostics(cfg, stitch_result, verification_results,
                            candidate_menus, cluster_stats, packs,
                            trial_histories)

    # ----------------------------------------------------------------
    # 9. Global combined outputs (report + CSV)
    # ----------------------------------------------------------------
    report_path, global_csv_path = _write_global_outputs(
        cfg, packs, cluster_stats, stitch_result, verification_results,
        trial_histories, t_total,
    )
    summary["output_files"].extend([report_path, global_csv_path])

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    if t_total < 60:
        print(f"\nStudy completed in {t_total:.1f} seconds.")
    elif t_total < 3600:
        print(f"\nStudy completed in {t_total / 60:.1f} minutes.")
    else:
        print(f"\nStudy completed in {t_total / 3600:.2f} hours.")

    return summary


# ======================================================================
# Global combined outputs
# ======================================================================

def _write_global_outputs(
    cfg: OptimizationStudyConfig,
    packs: Dict[int, RepresentativePack],
    stats: Dict[int, dict],
    stitch: StitchingResult,
    verif: Dict[int, ClusterEvaluation],
    trial_histories: Dict[int, list],
    runtime_s: float,
) -> Tuple[str, str]:
    """Write a combined global summary CSV and a Markdown report.

    Returns (report_path, global_csv_path).
    """
    out = cfg.output_dir

    # ---- 1. Global summary CSV (one row per cluster, all key metrics) ----
    rows: list[dict] = []
    cids = sorted(stitch.selected.keys())
    for cid in cids:
        ev = stitch.selected_evaluations.get(cid)
        pack = packs.get(cid)
        st = stats.get(cid, {})
        pol = stitch.selected_policies.get(cid)
        vev = verif.get(cid)
        row: dict = {
            "cluster_id": cid,
            "n_members": pack.n_members if pack else 0,
            "medoid_sat_id": pack.medoid_sat_id if pack else "",
            "sma_mean_km": st.get("sma_mean"),
            "inc_mean_deg": st.get("inc_mean"),
            "raan_mean_deg": st.get("raan_mean"),
            "ecc_mean": st.get("ecc_mean"),
        }
        if ev and ev.cost_terms:
            row["search_cost"] = ev.cost_scalar
            row["J_a"] = ev.cost_terms.J_a
            row["J_lambda"] = ev.cost_terms.J_lambda
            row["J_m"] = ev.cost_terms.J_m
            row["J_d"] = ev.cost_terms.J_d
            row["J_s"] = ev.cost_terms.J_s
            row["feasible"] = ev.feasible
        if vev:
            row["verified_cost"] = vev.cost_scalar
            row["verified_feasible"] = vev.feasible
        if pol:
            arr = pol.to_array()
            for i, name in enumerate([
                "delta_a_km", "delta_Omega_rad", "delta_lambda_rad",
                "tau_keep_s", "deadband_a_km", "deadband_lambda_rad",
                "reserve_prop_frac", "disposal_alt_km",
            ]):
                row[name] = arr[i]
        # Trial convergence stats
        trials = trial_histories.get(cid, [])
        if trials:
            costs = [e.cost_scalar for _, e in trials]
            row["n_trials"] = len(costs)
            row["best_trial_cost"] = min(costs)
            row["worst_trial_cost"] = max(costs)
            row["median_trial_cost"] = float(np.median(costs))
        # Torus info
        if ev and ev.torus_result is not None:
            row["J_torus"] = ev.torus_result.J_torus
        rows.append(row)

    global_df = pd.DataFrame(rows)
    global_csv_path = os.path.join(out, "constellation_global_summary.csv")
    global_df.to_csv(global_csv_path, index=False)
    print(f"[Output] {global_csv_path}")

    # ---- 2. Markdown report ----
    lines: list[str] = []
    lines.append("# Constellation Optimization Report\n")

    # Runtime
    if runtime_s < 3600:
        rt_str = f"{runtime_s / 60:.1f} minutes"
    else:
        rt_str = f"{runtime_s / 3600:.2f} hours"
    lines.append(f"- **Runtime**: {rt_str}")
    lines.append(f"- **Clusters optimized**: {len(cids)}")
    total_sats = sum(packs[c].n_members for c in cids if c in packs)
    lines.append(f"- **Total satellites covered**: {total_sats}")
    lines.append(f"- **Trials per cluster**: {cfg.n_local_trials}")
    lines.append(f"- **Search fidelity**: level {cfg.local_search_fidelity.level}")
    lines.append(f"- **Verification fidelity**: level {cfg.verification_fidelity.level}")
    lines.append(f"- **Stitching method**: {cfg.stitching_method} "
                 f"(gamma={cfg.stitching_gamma})")
    lines.append(f"- **Phase-space regularizer**: "
                 f"{'enabled' if cfg.phase_space.enabled else 'disabled'}")
    lines.append("")

    # Global cost summary
    lines.append("## Global Cost Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| J_global | {stitch.total_cost:.4f} |")
    lines.append(f"| J_local (sum) | {stitch.local_cost:.4f} |")
    lines.append(f"| J_coupling | {stitch.coupling_cost:.4f} |")
    lines.append(f"| J_phase_coupling | {stitch.phase_coupling_cost:.4f} |")
    lines.append(f"| Feasible | {stitch.feasible} |")
    lines.append("")

    # Aggregate objective breakdown
    all_Ja = [ev.cost_terms.J_a for ev in stitch.selected_evaluations.values()
              if ev and ev.cost_terms]
    all_Jl = [ev.cost_terms.J_lambda for ev in stitch.selected_evaluations.values()
              if ev and ev.cost_terms]
    all_Jm = [ev.cost_terms.J_m for ev in stitch.selected_evaluations.values()
              if ev and ev.cost_terms]
    all_Jd = [ev.cost_terms.J_d for ev in stitch.selected_evaluations.values()
              if ev and ev.cost_terms]
    all_Js = [ev.cost_terms.J_s for ev in stitch.selected_evaluations.values()
              if ev and ev.cost_terms]
    if all_Ja:
        lines.append("## Aggregate Objective Breakdown\n")
        lines.append("| Term | Mean | Std | Min | Max |")
        lines.append("|------|------|-----|-----|-----|")
        for name, vals in [("J_a (SMA tracking)", all_Ja),
                           ("J_lambda (phase coherence)", all_Jl),
                           ("J_m (propellant)", all_Jm),
                           ("J_d (disposal)", all_Jd),
                           ("J_s (spread)", all_Js)]:
            a = np.array(vals)
            lines.append(f"| {name} | {a.mean():.4f} | {a.std():.4f} "
                         f"| {a.min():.4f} | {a.max():.4f} |")
        lines.append("")

    # Per-cluster table
    lines.append("## Per-Cluster Results\n")
    header = "| Cluster | Members | SMA (km) | Cost | Propellant (kg) | Feasible |"
    if verif:
        header += " Verified Cost |"
    lines.append(header)
    sep = "|---------|---------|----------|------|-----------------|----------|"
    if verif:
        sep += "---------------|"
    lines.append(sep)

    for cid in cids:
        ev = stitch.selected_evaluations.get(cid)
        pack = packs.get(cid)
        st = stats.get(cid, {})
        vev = verif.get(cid)
        n_mem = pack.n_members if pack else 0
        sma = st.get("sma_mean", 0)
        cost = f"{ev.cost_scalar:.2f}" if ev else "—"
        prop = f"{ev.cost_terms.J_m:.2f}" if ev and ev.cost_terms else "—"
        feas = "Yes" if ev and ev.feasible else "No"
        row_str = f"| {cid} | {n_mem} | {sma:.1f} | {cost} | {prop} | {feas} |"
        if verif:
            if vev:
                row_str += f" {vev.cost_scalar:.2f} |"
            else:
                row_str += " — |"
        lines.append(row_str)
    lines.append("")

    # Convergence summary
    lines.append("## Convergence Summary\n")
    lines.append("| Cluster | Trials | Best Cost | Median Cost | Worst Cost |")
    lines.append("|---------|--------|-----------|-------------|------------|")
    for cid in cids:
        trials = trial_histories.get(cid, [])
        if trials:
            costs = [e.cost_scalar for _, e in trials]
            lines.append(f"| {cid} | {len(costs)} | {min(costs):.2f} "
                         f"| {np.median(costs):.2f} | {max(costs):.2f} |")
    lines.append("")

    # Stitching selection
    non_zero = sum(1 for k in stitch.selected.values() if k > 0)
    lines.append("## Stitching\n")
    lines.append(f"- Clusters selecting non-default candidate: "
                 f"{non_zero}/{len(stitch.selected)}")
    if stitch.coupling_cost > 0:
        lines.append(f"- Inter-cluster coupling cost: {stitch.coupling_cost:.4f}")
    if stitch.phase_coupling_cost > 0:
        lines.append(f"- Phase coupling cost: {stitch.phase_coupling_cost:.4f}")
    lines.append("")

    report_path = os.path.join(out, "constellation_optimization_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Output] {report_path}")

    return report_path, global_csv_path


# ======================================================================
# Diagnostic plots (final solution only)
# ======================================================================

def _plot_study_diagnostics(
    cfg: OptimizationStudyConfig,
    stitch: StitchingResult,
    verif: Dict[int, ClusterEvaluation],
    menus: Dict[int, list],
    stats: Dict[int, dict],
    packs: Dict[int, RepresentativePack],
    trial_histories: Optional[Dict[int, list]] = None,
):
    """Generate diagnostic plots and extra CSVs for the stitched solution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("[Plots] matplotlib not available, skipping diagnostics.")
        return

    if not stitch.feasible:
        print("[Plots] Infeasible stitch — skipping plots.")
        return

    out = cfg.output_dir
    os.makedirs(out, exist_ok=True)

    cids = sorted(stitch.selected.keys())

    # ---- 1. Propellant usage by cluster ----
    fig, ax = plt.subplots(figsize=(8, 5))
    prop_vals = []
    labels = []
    for cid in cids:
        ev = stitch.selected_evaluations.get(cid)
        if ev and ev.cost_terms:
            prop_vals.append(ev.cost_terms.J_m)
        else:
            prop_vals.append(0.0)
        labels.append(str(cid))
    ax.bar(labels, prop_vals, color="#15528e", alpha=0.8)
    ax.set_xlabel("Global Cluster ID")
    ax.set_ylabel("Mean Propellant Usage (kg)")
    ax.set_title("Propellant Usage by Cluster (Stitched Solution)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "propellant_by_cluster.png"), dpi=600)
    plt.close(fig)

    # ---- 2. Local cost breakdown by cluster ----
    fig, ax = plt.subplots(figsize=(10, 5))
    terms = ["J_a", "J_lambda", "J_m", "J_d", "J_s"]
    term_labels = ["SMA tracking", "Phase coherence", "Propellant", "Disposal", "Spread"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x = np.arange(len(cids))
    width = 0.15
    for i, (term, tlbl, col) in enumerate(zip(terms, term_labels, colors)):
        vals = []
        for cid in cids:
            ev = stitch.selected_evaluations.get(cid)
            if ev and ev.cost_terms:
                vals.append(getattr(ev.cost_terms, term, 0.0))
            else:
                vals.append(0.0)
        ax.bar(x + i * width, vals, width, label=tlbl, color=col, alpha=0.8)
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels([str(c) for c in cids], rotation=45)
    ax.set_xlabel("Global Cluster ID")
    ax.set_ylabel("Objective Term Value")
    ax.set_title("Cost Breakdown by Cluster")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cost_breakdown_by_cluster.png"), dpi=600)
    plt.close(fig)

    # ---- 3. Search vs verification cost ----
    if verif:
        fig, ax = plt.subplots(figsize=(7, 5))
        search_costs = []
        verif_costs = []
        vlabels = []
        for cid in cids:
            if cid in verif:
                search_costs.append(stitch.selected_evaluations[cid].cost_scalar)
                verif_costs.append(verif[cid].cost_scalar)
                vlabels.append(str(cid))
        x = np.arange(len(vlabels))
        ax.bar(x - 0.15, search_costs, 0.3, label="Search", color="#15528e", alpha=0.8)
        ax.bar(x + 0.15, verif_costs, 0.3, label="Verification", color="#b25800", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(vlabels, rotation=45)
        ax.set_xlabel("Global Cluster ID")
        ax.set_ylabel("Cost")
        ax.set_title("Search vs Verification Cost")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "search_vs_verification.png"), dpi=600)
        plt.close(fig)

    # ---- 4. Policy summary heatmap ----
    policy_names = ["delta_a", "delta_Omega", "delta_lambda", "tau_keep",
                    "deadband_a", "deadband_lambda", "reserve_prop_frac",
                    "disposal_altitude_trigger"]
    policy_matrix = []
    for cid in cids:
        pol = stitch.selected_policies.get(cid)
        if pol:
            policy_matrix.append(pol.to_array())
        else:
            policy_matrix.append([0.0] * 8)
    if policy_matrix:
        fig, ax = plt.subplots(figsize=(10, max(3, len(cids) * 0.4 + 1)))
        pm = np.array(policy_matrix)
        # Normalise per column for heatmap visibility
        pm_norm = pm.copy()
        for j in range(pm.shape[1]):
            rng_j = pm[:, j].max() - pm[:, j].min()
            if rng_j > 1e-12:
                pm_norm[:, j] = (pm[:, j] - pm[:, j].min()) / rng_j
            else:
                pm_norm[:, j] = 0.5
        im = ax.imshow(pm_norm, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(policy_names)))
        ax.set_xticklabels(policy_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(cids)))
        ax.set_yticklabels([str(c) for c in cids])
        ax.set_xlabel("Policy Parameter")
        ax.set_ylabel("Cluster ID")
        ax.set_title("Stitched Policy Summary (normalised)")
        fig.colorbar(im, ax=ax, shrink=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "policy_summary_heatmap.png"), dpi=600)
        plt.close(fig)

    # ---- 5. Optimization convergence per cluster ----
    if trial_histories:
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(trial_histories), 1))
        for idx, cid in enumerate(sorted(trial_histories.keys())):
            trials = trial_histories[cid]
            costs = [ev.cost_scalar for _, ev in trials]
            best = np.minimum.accumulate(costs)
            ax.plot(range(len(best)), best, label=f"C{cid}",
                    color=cmap(idx % 20), linewidth=1.3, alpha=0.8)
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Best Cost So Far")
        ax.set_title("Optimization Convergence by Cluster")
        ax.legend(fontsize=6, ncol=max(1, len(trial_histories) // 8),
                  loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "optimization_convergence.png"), dpi=600)
        plt.close(fig)

    # ---- 6. Cluster map in SMA vs RAAN space ----
    if stats:
        fig, ax = plt.subplots(figsize=(9, 6))
        sma_vals, raan_vals, cost_vals, size_vals, map_cids = [], [], [], [], []
        for cid in cids:
            st = stats.get(cid, {})
            sma_v = st.get("sma_mean")
            raan_v = st.get("raan_mean")
            if sma_v is None or raan_v is None:
                continue
            ev = stitch.selected_evaluations.get(cid)
            cost_v = ev.cost_scalar if ev else 0.0
            n_mem = packs[cid].n_members if cid in packs else 10
            sma_vals.append(float(sma_v))
            raan_vals.append(float(raan_v))
            cost_vals.append(cost_v)
            size_vals.append(n_mem)
            map_cids.append(cid)
        if sma_vals:
            sizes = np.array(size_vals, dtype=float)
            sizes = 30 + 200 * (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1)
            sc = ax.scatter(raan_vals, sma_vals, c=cost_vals, s=sizes,
                            cmap="RdYlGn_r", edgecolors="k", linewidths=0.5, alpha=0.85)
            for i, cid in enumerate(map_cids):
                ax.annotate(str(cid), (raan_vals[i], sma_vals[i]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7)
            fig.colorbar(sc, ax=ax, label="Total Cost")
            ax.set_xlabel("RAAN (deg)")
            ax.set_ylabel("SMA (km)")
            ax.set_title("Cluster Map: SMA vs RAAN (size ~ n_members, color ~ cost)")
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "cluster_sma_raan_map.png"), dpi=600)
            plt.close(fig)

    # ---- 7. Cost vs cluster size ----
    fig, ax = plt.subplots(figsize=(7, 5))
    csizes, ccosts, clabels = [], [], []
    for cid in cids:
        ev = stitch.selected_evaluations.get(cid)
        if ev is None:
            continue
        csizes.append(packs[cid].n_members if cid in packs else 0)
        ccosts.append(ev.cost_scalar)
        clabels.append(cid)
    if csizes:
        ax.scatter(csizes, ccosts, s=60, color="#15528e", edgecolors="k",
                   linewidths=0.5, alpha=0.8)
        for i, cid in enumerate(clabels):
            ax.annotate(str(cid), (csizes[i], ccosts[i]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Cluster Size (n_members)")
        ax.set_ylabel("Optimized Cost")
        ax.set_title("Cost vs Cluster Size")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "cost_vs_cluster_size.png"), dpi=600)
        plt.close(fig)

    # ---- 8. Objective terms correlation matrix ----
    term_names = ["J_a", "J_lambda", "J_m", "J_d", "J_s"]
    term_matrix = []
    for cid in cids:
        ev = stitch.selected_evaluations.get(cid)
        if ev and ev.cost_terms:
            term_matrix.append([getattr(ev.cost_terms, t, 0.0) for t in term_names])
    if len(term_matrix) >= 3:
        tm = np.array(term_matrix)
        # Compute correlation with safety for constant columns
        corr = np.corrcoef(tm.T)
        corr = np.nan_to_num(corr, nan=0.0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        display_names = ["SMA track", "Phase coh.", "Propellant", "Disposal", "Spread"]
        ax.set_xticks(range(5))
        ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(5))
        ax.set_yticklabels(display_names, fontsize=8)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(corr[i, j]) > 0.5 else "black")
        ax.set_title("Objective Terms Correlation")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "objective_correlation_matrix.png"), dpi=600)
        plt.close(fig)

    # ---- 9. Radar chart: normalized policy comparison ----
    if len(cids) >= 2 and policy_matrix:
        pm = np.array(policy_matrix)
        # Normalize to [0, 1] per parameter
        pm_norm = np.zeros_like(pm)
        for j in range(pm.shape[1]):
            lo, hi = pm[:, j].min(), pm[:, j].max()
            if hi - lo > 1e-12:
                pm_norm[:, j] = (pm[:, j] - lo) / (hi - lo)
            else:
                pm_norm[:, j] = 0.5
        angles = np.linspace(0, 2 * np.pi, len(policy_names), endpoint=False).tolist()
        angles += angles[:1]  # close polygon
        short_names = ["Δa", "ΔΩ", "Δλ", "τ_keep", "db_a", "db_λ", "rsv", "disp_alt"]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(cids), 1))
        for idx, cid in enumerate(cids):
            vals = pm_norm[idx].tolist() + [pm_norm[idx, 0]]
            ax.plot(angles, vals, linewidth=1.5, label=f"C{cid}",
                    color=cmap(idx % 10), alpha=0.8)
            ax.fill(angles, vals, color=cmap(idx % 10), alpha=0.05)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_names, fontsize=8)
        ax.set_title("Policy Comparison (normalised)", pad=20)
        ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        fig.savefig(os.path.join(out, "radar_policy_comparison.png"), dpi=600)
        plt.close(fig)

    # ---- 10. RAAN-phase torus diagnostics (if enabled) ----
    _has_torus = any(
        ev.torus_result is not None
        for ev in stitch.selected_evaluations.values()
    )
    if _has_torus:
        try:
            from cluster_phase_space_plots import (
                plot_raan_phase_shell,
                plot_torus_cost_breakdown,
            )
            # Torus cost breakdown
            torus_results = {
                cid: ev.torus_result
                for cid, ev in stitch.selected_evaluations.items()
                if ev.torus_result is not None
            }
            if torus_results:
                plot_torus_cost_breakdown(
                    list(torus_results.keys()),
                    torus_results,
                    save_path=os.path.join(out, "torus_cost_breakdown.png"),
                )

            # RAAN-phase scatter per cluster (with fitted slots)
            for cid, ev in stitch.selected_evaluations.items():
                if ev.torus_result is not None and ev.torus_result.lattice is not None:
                    lat = ev.torus_result.lattice
                    # Reconstruct snapshot RAAN/phase from assignment indices
                    if lat.target_raan_deg.size > 0:
                        plot_raan_phase_shell(
                            lat.target_raan_deg[ev.torus_result.col_ind]
                            if ev.torus_result.col_ind.size > 0
                            else np.array([]),
                            lat.target_phase_deg[ev.torus_result.col_ind]
                            if ev.torus_result.col_ind.size > 0
                            else np.array([]),
                            lattice=lat,
                            title=f"Cluster {cid}: RAAN vs Phase (fitted lattice)",
                            save_path=os.path.join(out, f"raan_phase_cluster_{cid}.png"),
                        )
        except Exception as exc:
            print(f"[Plots] Phase-space plots failed: {exc}")

    print(f"[Plots] Diagnostic plots saved to {out}/")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    summary = run_study(STUDY_CONFIG)
    print("\nDone.")
