"""Global cluster-policy stitcher for Chapter 7 optimization.

Coordinates locally-optimal cluster candidate menus into a single
constellation-wide design.  The stitcher minimises total local cost
plus pairwise coupling penalties over a sparse inter-cluster
adjacency graph.

Three stitching methods are provided:
    beam        Beam search over candidate menus (default, general)
    dp          DP-like chain sweep along RAAN-ordered clusters
    enumerate   Brute-force for small problems (≤5 clusters × ≤5 cands)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimization_config import ClusterPolicy, PhaseSpaceConfig
from cluster_objective import ClusterEvaluation
from cluster_metrics import ClusterEnvelope, compute_intercluster_penalty
from cluster_phase_space import compute_torus_consistency_penalty


# ======================================================================
# Adjacency graph
# ======================================================================

def build_cluster_adjacency_graph(
    cluster_stats: Dict[int, dict],
    sma_threshold_km: float = 50.0,
    raan_threshold_deg: float = 30.0,
) -> Dict[Tuple[int, int], float]:
    """Build sparse interaction graph from cluster statistics.

    Two clusters are considered neighbours if they are in the same
    shell (SMA within threshold) AND have adjacent RAAN (within
    threshold).  The edge weight is a proximity score in [0, 1].

    Parameters
    ----------
    cluster_stats : dict
        ``{cluster_id: {sma_mean, raan_mean, ...}}``
    sma_threshold_km : float
        Maximum SMA difference (km) to create an edge.
    raan_threshold_deg : float
        Maximum RAAN difference (deg) to create an edge.

    Returns
    -------
    dict[(int,int), float]
        Edges keyed by sorted cluster-ID pairs with proximity weights.
    """
    cids = sorted(cluster_stats.keys())
    edges: dict[tuple[int, int], float] = {}

    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            ci, cj = cids[i], cids[j]
            si, sj = cluster_stats[ci], cluster_stats[cj]

            a_i = float(si.get("sma_mean", 0.0))
            a_j = float(sj.get("sma_mean", 0.0))
            r_i = float(si.get("raan_mean", 0.0))
            r_j = float(sj.get("raan_mean", 0.0))

            da = abs(a_i - a_j)
            # Circular RAAN difference
            dr = abs(r_i - r_j)
            dr = min(dr, 360.0 - dr)

            if da <= sma_threshold_km and dr <= raan_threshold_deg:
                # Proximity: 1 when identical, 0 at threshold boundary
                w_a = 1.0 - da / max(sma_threshold_km, 1e-12)
                w_r = 1.0 - dr / max(raan_threshold_deg, 1e-12)
                proximity = w_a * w_r
                edges[(ci, cj)] = proximity

    return edges


# ======================================================================
# Coupling cost
# ======================================================================

def compute_pairwise_coupling_matrix(
    envelopes: Dict[int, ClusterEnvelope],
    adjacency: Dict[Tuple[int, int], float],
    R_max: float = 0.01,
) -> Dict[Tuple[int, int], float]:
    """Pre-compute Psi_gh for all edges in the adjacency graph.

    Returns a dict ``{(g,h): psi_value}`` aligned with the adjacency graph.
    """
    coupling: dict[tuple[int, int], float] = {}
    for (g, h), prox in adjacency.items():
        env_g = envelopes.get(g)
        env_h = envelopes.get(h)
        if env_g is not None and env_h is not None:
            psi = compute_intercluster_penalty(env_g, env_h, R_max=R_max)
            coupling[(g, h)] = psi * prox  # weight by proximity
        else:
            coupling[(g, h)] = 0.0
    return coupling


# ======================================================================
# Stitching result
# ======================================================================

@dataclass
class StitchingResult:
    """Output of the global stitcher."""
    selected: Dict[int, int] = field(default_factory=dict)            # cluster_id → candidate index
    selected_policies: Dict[int, ClusterPolicy] = field(default_factory=dict)
    selected_evaluations: Dict[int, ClusterEvaluation] = field(default_factory=dict)
    total_cost: float = float("inf")
    local_cost: float = 0.0
    coupling_cost: float = 0.0
    phase_coupling_cost: float = 0.0
    method: str = ""
    feasible: bool = True

    def to_summary_dict(self) -> dict:
        return {
            "method": self.method,
            "total_cost": self.total_cost,
            "local_cost": self.local_cost,
            "coupling_cost": self.coupling_cost,
            "phase_coupling_cost": self.phase_coupling_cost,
            "n_clusters": len(self.selected),
            "feasible": self.feasible,
            "selected_candidates": {str(k): v for k, v in self.selected.items()},
        }


# ======================================================================
# Stitching methods
# ======================================================================

def _evaluate_combination(
    selection: Dict[int, int],
    menus: Dict[int, List[Tuple[ClusterPolicy, ClusterEvaluation]]],
    adjacency: Dict[Tuple[int, int], float],
    gamma: float,
    R_max: float,
    phase_space_config: Optional[PhaseSpaceConfig] = None,
) -> Tuple[float, float, float, float]:
    """Compute total cost = sum local + gamma * sum coupling + gamma_phase * sum Psi_phase.

    Returns (total, local_cost, coupling_cost, phase_coupling_cost).
    """
    local_cost = 0.0
    envelopes: dict[int, ClusterEnvelope] = {}

    for cid, k in selection.items():
        _, ev = menus[cid][k]
        local_cost += ev.cost_scalar
        if ev.envelope is not None:
            envelopes[cid] = ev.envelope

    coupling_cost = 0.0
    for (g, h), prox in adjacency.items():
        kg = selection.get(g)
        kh = selection.get(h)
        if kg is None or kh is None:
            continue
        env_g = envelopes.get(g)
        env_h = envelopes.get(h)
        if env_g is not None and env_h is not None:
            psi = compute_intercluster_penalty(env_g, env_h, R_max=R_max)
            coupling_cost += prox * psi

    # Phase-space torus consistency penalty
    phase_coupling_cost = 0.0
    if phase_space_config is not None and phase_space_config.enabled:
        gamma_phase = phase_space_config.gamma_phase
        for (g, h), prox in adjacency.items():
            kg = selection.get(g)
            kh = selection.get(h)
            if kg is None or kh is None:
                continue
            _, ev_g = menus[g][kg]
            _, ev_h = menus[h][kh]
            if ev_g.torus_summary is not None and ev_h.torus_summary is not None:
                psi_phase = compute_torus_consistency_penalty(
                    ev_g.torus_summary, ev_h.torus_summary,
                    beta_raan=phase_space_config.beta_raan,
                    beta_phase=phase_space_config.beta_phase,
                    beta_eta=phase_space_config.beta_eta,
                )
                phase_coupling_cost += prox * psi_phase
        phase_coupling_cost *= gamma_phase

    total = local_cost + gamma * coupling_cost + phase_coupling_cost
    return total, local_cost, coupling_cost, phase_coupling_cost


# ---- Enumerate (brute-force) ----

def _stitch_enumerate(
    menus: Dict[int, List[Tuple[ClusterPolicy, ClusterEvaluation]]],
    adjacency: Dict[Tuple[int, int], float],
    gamma: float,
    R_max: float,
    phase_space_config: Optional[PhaseSpaceConfig] = None,
) -> StitchingResult:
    """Brute-force exhaustive search (small problems only)."""
    cids = sorted(menus.keys())
    candidate_counts = [range(len(menus[c])) for c in cids]

    best_sel: dict[int, int] = {c: 0 for c in cids}
    best_total = float("inf")
    best_local = 0.0
    best_coupling = 0.0
    best_phase_coupling = 0.0

    for combo in itertools.product(*candidate_counts):
        sel = {cids[i]: int(combo[i]) for i in range(len(cids))}
        total, loc, coup, pcoup = _evaluate_combination(sel, menus, adjacency, gamma, R_max, phase_space_config)
        if total < best_total:
            best_total = total
            best_local = loc
            best_coupling = coup
            best_phase_coupling = pcoup
            best_sel = sel

    return _build_result(best_sel, menus, best_total, best_local, best_coupling, "enumerate", best_phase_coupling)


# ---- Beam search ----

def _stitch_beam(
    menus: Dict[int, List[Tuple[ClusterPolicy, ClusterEvaluation]]],
    adjacency: Dict[Tuple[int, int], float],
    gamma: float,
    R_max: float,
    beam_width: int = 10,
    phase_space_config: Optional[PhaseSpaceConfig] = None,
) -> StitchingResult:
    """Beam search over cluster candidate menus.

    Process clusters in order of mean RAAN (ascending).  At each step,
    expand each beam entry with all candidates of the next cluster, then
    prune to beam_width best.
    """
    cids = sorted(menus.keys())
    if not cids:
        return StitchingResult(method="beam", feasible=False)

    # Sort by RAAN if we have evaluations with envelopes; otherwise by ID
    def _raan_key(cid: int) -> float:
        for _, ev in menus[cid]:
            if ev.envelope is not None and ev.envelope.mu.shape[0] > 0:
                return float(ev.envelope.mu[0, 1])  # initial RAAN
        return float(cid)
    cids.sort(key=_raan_key)

    # Beam: list of (partial_selection, cumulative_cost)
    beam: list[tuple[dict[int, int], float]] = []

    # Initialise with first cluster's candidates
    first_cid = cids[0]
    for k in range(len(menus[first_cid])):
        _, ev = menus[first_cid][k]
        beam.append(({first_cid: k}, ev.cost_scalar))

    # Expand for each subsequent cluster
    for step_idx in range(1, len(cids)):
        next_cid = cids[step_idx]
        new_beam: list[tuple[dict[int, int], float]] = []

        for sel, cum_cost in beam:
            for k in range(len(menus[next_cid])):
                new_sel = dict(sel)
                new_sel[next_cid] = k
                _, ev_k = menus[next_cid][k]
                added_cost = ev_k.cost_scalar

                # Add coupling penalty with already-selected neighbours
                coupling_add = 0.0
                if ev_k.envelope is not None:
                    for prev_cid, prev_k in sel.items():
                        edge = (min(prev_cid, next_cid), max(prev_cid, next_cid))
                        prox = adjacency.get(edge, 0.0)
                        if prox > 0.0:
                            _, prev_ev = menus[prev_cid][prev_k]
                            if prev_ev.envelope is not None:
                                psi = compute_intercluster_penalty(
                                    prev_ev.envelope, ev_k.envelope, R_max=R_max)
                                coupling_add += gamma * prox * psi

                new_beam.append((new_sel, cum_cost + added_cost + coupling_add))

        # Prune to beam_width
        new_beam.sort(key=lambda x: x[1])
        beam = new_beam[:beam_width]

    if not beam:
        return StitchingResult(method="beam", feasible=False)

    best_sel, _ = beam[0]
    total, loc, coup, pcoup = _evaluate_combination(best_sel, menus, adjacency, gamma, R_max, phase_space_config)
    return _build_result(best_sel, menus, total, loc, coup, "beam", pcoup)


# ---- DP chain ----

def _stitch_dp(
    menus: Dict[int, List[Tuple[ClusterPolicy, ClusterEvaluation]]],
    adjacency: Dict[Tuple[int, int], float],
    gamma: float,
    R_max: float,
    n_budget_levels: int = 20,
    phase_space_config: Optional[PhaseSpaceConfig] = None,
) -> StitchingResult:
    """DP-like chain stitcher for RAAN-ordered clusters.

    Approximate Bellman sweep along a chain with a discretised
    propellant budget dimension.

    F_j(b, k_j) = C_j(k_j) + min_{k_{j-1}} [F_{j-1}(b-b_j, k_{j-1})
                   + gamma * Psi_{j-1,j}(k_{j-1}, k_j)]

    Since exact budget tracking is problem-specific, we use a simplified
    version without budget: the DP just tracks (cost, backpointer).
    Budget extension is noted but not forced.
    """
    cids = sorted(menus.keys())
    if not cids:
        return StitchingResult(method="dp", feasible=False)

    # Sort by RAAN (same logic as beam)
    def _raan_key(cid: int) -> float:
        for _, ev in menus[cid]:
            if ev.envelope is not None and ev.envelope.mu.shape[0] > 0:
                return float(ev.envelope.mu[0, 1])
        return float(cid)
    cids.sort(key=_raan_key)

    n_clusters = len(cids)

    # cost_table[j][k] = best total cost ending at cluster j with candidate k
    # back_table[j][k] = (prev_k) backpointer
    cost_table: list[list[float]] = []
    back_table: list[list[int]] = []

    # Initialise j=0
    first_cid = cids[0]
    n_k0 = len(menus[first_cid])
    cost_table.append([menus[first_cid][k][1].cost_scalar for k in range(n_k0)])
    back_table.append([-1] * n_k0)

    for j in range(1, n_clusters):
        cid_j = cids[j]
        cid_prev = cids[j - 1]
        n_kj = len(menus[cid_j])
        n_kprev = len(menus[cid_prev])

        edge = (min(cid_prev, cid_j), max(cid_prev, cid_j))
        prox = adjacency.get(edge, 0.0)

        costs_j: list[float] = []
        backs_j: list[int] = []

        for k in range(n_kj):
            _, ev_k = menus[cid_j][k]
            best_prev_cost = float("inf")
            best_prev_k = 0

            for kp in range(n_kprev):
                coupling = 0.0
                if prox > 0.0:
                    _, ev_kp = menus[cid_prev][kp]
                    if ev_kp.envelope is not None and ev_k.envelope is not None:
                        psi = compute_intercluster_penalty(
                            ev_kp.envelope, ev_k.envelope, R_max=R_max)
                        coupling = gamma * prox * psi

                candidate_cost = cost_table[j - 1][kp] + coupling
                if candidate_cost < best_prev_cost:
                    best_prev_cost = candidate_cost
                    best_prev_k = kp

            costs_j.append(best_prev_cost + ev_k.cost_scalar)
            backs_j.append(best_prev_k)

        cost_table.append(costs_j)
        back_table.append(backs_j)

    # Backtrack from the best terminal state
    last_costs = cost_table[-1]
    best_k_last = int(np.argmin(last_costs))

    selection: dict[int, int] = {}
    k_trace = best_k_last
    for j in range(n_clusters - 1, -1, -1):
        selection[cids[j]] = k_trace
        k_trace = back_table[j][k_trace]

    total, loc, coup, pcoup = _evaluate_combination(selection, menus, adjacency, gamma, R_max, phase_space_config)
    return _build_result(selection, menus, total, loc, coup, "dp", pcoup)


# ======================================================================
# Helpers
# ======================================================================

def _build_result(
    selection: Dict[int, int],
    menus: Dict[int, List[Tuple[ClusterPolicy, ClusterEvaluation]]],
    total: float,
    local: float,
    coupling: float,
    method: str,
    phase_coupling: float = 0.0,
) -> StitchingResult:
    sel_policies = {}
    sel_evals = {}
    for cid, k in selection.items():
        pol, ev = menus[cid][k]
        sel_policies[cid] = pol
        sel_evals[cid] = ev

    return StitchingResult(
        selected=selection,
        selected_policies=sel_policies,
        selected_evaluations=sel_evals,
        total_cost=total,
        local_cost=local,
        coupling_cost=coupling,
        phase_coupling_cost=phase_coupling,
        method=method,
        feasible=np.isfinite(total),
    )


# ======================================================================
# Public entry point
# ======================================================================

def stitch_cluster_policies(
    candidate_menus: Dict[int, List[Tuple[ClusterPolicy, ClusterEvaluation]]],
    adjacency: Dict[Tuple[int, int], float],
    *,
    gamma: float = 1.0,
    R_max: float = 0.01,
    method: str = "beam",
    beam_width: int = 10,
    verbose: bool = True,
    phase_space_config: Optional[PhaseSpaceConfig] = None,
) -> StitchingResult:
    """Select one candidate per cluster to minimise global cost.

    Parameters
    ----------
    candidate_menus : dict[int, list[(policy, evaluation)]]
        Per-cluster candidate lists from local optimizers.
    adjacency : dict[(int,int), float]
        Sparse interaction graph.
    gamma : float
        Coupling penalty weight.
    R_max : float
        Overlap risk threshold.
    method : str
        ``"beam"`` | ``"dp"`` | ``"enumerate"``.
    beam_width : int
        Beam width (beam method only).
    verbose : bool
        Print summary.

    Returns
    -------
    StitchingResult
    """
    if not candidate_menus:
        return StitchingResult(method=method, feasible=False)

    # Total number of combinations for complexity check
    total_combos = 1
    for cid in candidate_menus:
        total_combos *= len(candidate_menus[cid])

    if method == "enumerate" or (method == "beam" and total_combos <= 1000):
        if total_combos <= 100000:
            result = _stitch_enumerate(candidate_menus, adjacency, gamma, R_max, phase_space_config)
        else:
            result = _stitch_beam(candidate_menus, adjacency, gamma, R_max, beam_width, phase_space_config)
    elif method == "dp":
        result = _stitch_dp(candidate_menus, adjacency, gamma, R_max, phase_space_config=phase_space_config)
    else:
        result = _stitch_beam(candidate_menus, adjacency, gamma, R_max, beam_width, phase_space_config)

    if verbose:
        print(f"\n[Stitcher] Method={result.method} | "
              f"Total cost={result.total_cost:.6f} | "
              f"Local={result.local_cost:.6f} | "
              f"Coupling={result.coupling_cost:.6f} | "
              f"Phase coupling={result.phase_coupling_cost:.6f} | "
              f"Clusters={len(result.selected)}")

    return result


def stitching_result_to_dataframe(result: StitchingResult) -> pd.DataFrame:
    """Flatten a stitching result into a row-per-cluster DataFrame."""
    rows: list[dict] = []
    for cid in sorted(result.selected.keys()):
        k = result.selected[cid]
        ev = result.selected_evaluations.get(cid)
        pol = result.selected_policies.get(cid)
        d = {
            "global_cluster_id": cid,
            "candidate_index": k,
            "cost_scalar": ev.cost_scalar if ev else None,
            "fidelity_level": ev.fidelity_level if ev else None,
        }
        if pol is not None:
            pa = pol.to_array()
            for i, name in enumerate(["delta_a", "delta_Omega", "delta_lambda",
                                       "tau_keep", "deadband_a", "deadband_lambda",
                                       "reserve_prop_frac", "disposal_altitude_trigger"]):
                d[f"policy_{name}"] = pa[i]
        if ev and ev.cost_terms:
            d.update(ev.cost_terms.to_dict())
        rows.append(d)
    return pd.DataFrame(rows)
