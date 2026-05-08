"""Cluster representative-pack selection for Chapter 7 optimization.

Extends the existing medoid-only logic with boundary representatives that
capture within-cluster orbital-element spread.  Three modes are supported:

    medoid_only        – single centroid-closest member per cluster
    medoid+boundary    – medoid plus configurable boundary members
    full_members       – all members (for verification fidelity)

The 9-D orbital feature basis is reused from the main simulator and can
optionally be extended with ballistic coefficient and specific angular
momentum.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Re-use helpers from the main simulator without importing module-level state
# ---------------------------------------------------------------------------

def _normalize_sat_id(sat_id_value: str) -> str:
    """Canonical filename key: lowercase, basename, _decay→.txt, ensure .txt."""
    s = str(sat_id_value).strip().lower()
    s = os.path.basename(s)
    if s.endswith("_decay.txt"):
        s = s.replace("_decay.txt", ".txt")
    elif s.endswith("_decay"):
        s = s.replace("_decay", ".txt")
    elif not s.endswith(".txt"):
        s = s + ".txt"
    return s


def load_cluster_assignments(csv_path: str | Path) -> Dict[str, int]:
    """Load cluster assignment CSV → {sat_key: global_cluster_id}.

    Majority-vote deduplication when a satellite appears in multiple rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cluster assignments CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["sat_id", "global_cluster_id"])
    if df.empty:
        return {}

    df["sat_key"] = df["sat_id"].map(_normalize_sat_id)
    df["global_cluster_id"] = (
        pd.to_numeric(df["global_cluster_id"], errors="coerce")
        .fillna(0)
        .astype(np.int64)
    )

    counts = (
        df.groupby(["sat_key", "global_cluster_id"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    chosen = (
        counts.sort_values(["sat_key", "count", "global_cluster_id"],
                           ascending=[True, False, True])
        .drop_duplicates(subset=["sat_key"], keep="first")
    )
    return {str(row.sat_key): int(row.global_cluster_id)
            for row in chosen.itertuples(index=False)}


def load_cluster_stats(csv_path: str | Path) -> Dict[int, dict]:
    """Load global cluster stats CSV → {cluster_id: stats_dict}."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    out: dict[int, dict] = {}
    for row in df.itertuples(index=False):
        cid = int(getattr(row, "global_cluster_id", 0))
        out[cid] = {c: getattr(row, c, None) for c in df.columns}
    return out


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    *,
    include_beta: bool = False,
    include_h: bool = False,
) -> np.ndarray:
    """Build an (N, D) orbital feature matrix from a TLE DataFrame.

    Base features (D=9):
        sma, ecc, inc, sin(aop), cos(aop), sin(raan), cos(raan),
        sin(mean_anomaly), cos(mean_anomaly)

    Extended features:
        beta (ballistic_coefficient column, D+1)
        h    (specific_angular_momentum column, D+1)

    Returns a contiguous float64 array with NaN-safe imputation.
    """
    sma = pd.to_numeric(df["sma"], errors="coerce").to_numpy(np.float64)
    ecc = pd.to_numeric(df["ecc"], errors="coerce").to_numpy(np.float64)
    inc = pd.to_numeric(df["inc"], errors="coerce").to_numpy(np.float64)

    aop = np.deg2rad(pd.to_numeric(df["aop"], errors="coerce").to_numpy(np.float64))
    raan = np.deg2rad(pd.to_numeric(df["raan"], errors="coerce").to_numpy(np.float64))
    ma = np.deg2rad(pd.to_numeric(df["mean_anomaly"], errors="coerce").to_numpy(np.float64))

    cols = [sma, ecc, inc,
            np.sin(aop), np.cos(aop),
            np.sin(raan), np.cos(raan),
            np.sin(ma), np.cos(ma)]

    if include_beta and "ballistic_coefficient" in df.columns:
        beta = pd.to_numeric(df["ballistic_coefficient"], errors="coerce").to_numpy(np.float64)
        cols.append(beta)
    elif include_beta:
        cols.append(np.full(len(df), np.nan, dtype=np.float64))

    if include_h and "specific_angular_momentum" in df.columns:
        h = pd.to_numeric(df["specific_angular_momentum"], errors="coerce").to_numpy(np.float64)
        cols.append(h)
    elif include_h:
        cols.append(np.full(len(df), np.nan, dtype=np.float64))

    features = np.column_stack(cols)

    # NaN-safe column-wise imputation (mean of valid values, else 0)
    for j in range(features.shape[1]):
        col = features[:, j]
        mask = ~np.isfinite(col)
        if mask.any():
            valid = col[~mask]
            fill = float(np.mean(valid)) if valid.size > 0 else 0.0
            col[mask] = fill

    return np.ascontiguousarray(features, dtype=np.float64)


def _normalize_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise columns. Returns (normalised, means, stds)."""
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    stds[stds < 1e-12] = 1.0
    normed = (features - means) / stds
    return normed, means, stds


# ---------------------------------------------------------------------------
# Medoid and boundary selection
# ---------------------------------------------------------------------------

def _pick_medoid(features: np.ndarray) -> int:
    """Index of the member closest to the centroid (L2 distance)."""
    if features.shape[0] <= 1:
        return 0
    centroid = np.mean(features, axis=0)
    diff = features - centroid
    dists_sq = np.sum(diff * diff, axis=1)
    return int(np.argmin(dists_sq))


def select_boundary_members(
    features_normed: np.ndarray,
    medoid_local_idx: int,
    n_boundary: int = 6,
) -> List[int]:
    """Select boundary representatives by distance from medoid in normalised space.

    Strategy (ordered):
        1. farthest point from medoid
        2. nearest to min-SMA edge  (lowest a)
        3. nearest to max-SMA edge  (highest a)
        4. nearest to min-beta edge (col 9, if available)
        5. nearest to max-beta edge
        6. nearest to extreme phase coordinate (col 7 = sin(M))
        7+. remaining: farthest unused from medoid

    Returns local indices into the *features_normed* matrix.
    """
    n = features_normed.shape[0]
    if n <= 1:
        return []

    medoid_vec = features_normed[medoid_local_idx]
    dists = np.sqrt(np.sum((features_normed - medoid_vec) ** 2, axis=1))
    dists[medoid_local_idx] = -1.0  # exclude medoid itself

    selected: list[int] = []

    def _pick_and_add(idx: int):
        if idx not in selected and idx != medoid_local_idx:
            selected.append(idx)

    # 1. farthest from medoid
    _pick_and_add(int(np.argmax(dists)))

    # 2-3. SMA extremes (column 0 in normalised space)
    sma_col = features_normed[:, 0].copy()
    sma_col[medoid_local_idx] = np.nan
    valid_mask = np.isfinite(sma_col)
    if valid_mask.any():
        _pick_and_add(int(np.nanargmin(sma_col)))
        _pick_and_add(int(np.nanargmax(sma_col)))

    # 4-5. Beta extremes (column 9, if feature matrix has ≥10 columns)
    if features_normed.shape[1] >= 10:
        beta_col = features_normed[:, 9].copy()
        beta_col[medoid_local_idx] = np.nan
        if np.isfinite(beta_col).any():
            _pick_and_add(int(np.nanargmin(beta_col)))
            _pick_and_add(int(np.nanargmax(beta_col)))

    # 6. Phase extreme (sin(M), column 7)
    if features_normed.shape[1] >= 8:
        phase_col = features_normed[:, 7].copy()
        phase_col[medoid_local_idx] = np.nan
        if np.isfinite(phase_col).any():
            _pick_and_add(int(np.nanargmax(np.abs(phase_col))))

    # Fill remaining slots with farthest unused from medoid
    if len(selected) < n_boundary:
        order = np.argsort(-dists)  # descending distance
        for idx in order:
            idx = int(idx)
            if idx != medoid_local_idx and idx not in selected:
                selected.append(idx)
                if len(selected) >= n_boundary:
                    break

    return selected[:n_boundary]


# ---------------------------------------------------------------------------
# Representative pack dataclass
# ---------------------------------------------------------------------------

@dataclass
class RepresentativePack:
    """One cluster's representative set for optimization."""
    cluster_id: int
    medoid_sat_id: str
    medoid_row_idx: int                        # index into the cluster sub-DataFrame
    boundary_sat_ids: List[str] = field(default_factory=list)
    boundary_row_idxs: List[int] = field(default_factory=list)
    all_member_sat_ids: List[str] = field(default_factory=list)
    all_member_row_idxs: List[int] = field(default_factory=list)
    n_members: int = 0
    feature_matrix: Optional[np.ndarray] = None  # (n_members, D)

    @property
    def representative_sat_ids(self) -> List[str]:
        """Medoid + boundary sat IDs (deduped, order-preserved)."""
        seen: set[str] = set()
        out: list[str] = []
        for sid in [self.medoid_sat_id] + self.boundary_sat_ids:
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
        return out

    @property
    def representative_row_idxs(self) -> List[int]:
        """Medoid + boundary row indices (deduped)."""
        seen: set[int] = set()
        out: list[int] = []
        for idx in [self.medoid_row_idx] + self.boundary_row_idxs:
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_cluster_representatives(
    tle_latest_df: pd.DataFrame,
    cluster_assignments: Dict[str, int],
    *,
    mode: str = "medoid+boundary",
    n_boundary: int = 6,
    include_beta: bool = False,
    include_h: bool = False,
    cluster_ids: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[int, RepresentativePack]:
    """Build representative packs for all (or selected) clusters.

    Parameters
    ----------
    tle_latest_df : DataFrame
        One row per satellite with columns: sat_id, sma, ecc, inc, aop,
        raan, mean_anomaly.  Optionally: ballistic_coefficient,
        specific_angular_momentum.
    cluster_assignments : dict
        ``{normalised_sat_key: global_cluster_id}``
    mode : str
        ``medoid_only``, ``medoid+boundary``, or ``full_members``.
    n_boundary : int
        Number of boundary representatives per cluster.
    include_beta : bool
        Include ballistic coefficient in feature space.
    include_h : bool
        Include specific angular momentum in feature space.
    cluster_ids : list[int] | None
        Restrict to these cluster IDs.  None → all non-noise (>0).
    verbose : bool
        Print summary info.

    Returns
    -------
    dict[int, RepresentativePack]
        Keyed by global_cluster_id.
    """
    work = tle_latest_df.copy()
    work["sat_id"] = work["sat_id"].astype(str)
    work["sat_key"] = work["sat_id"].map(_normalize_sat_id)
    work["cluster_id"] = work["sat_key"].map(
        lambda k: int(cluster_assignments.get(str(k), 0))
    )

    # Filter to requested cluster IDs
    if cluster_ids is not None:
        work = work[work["cluster_id"].isin(cluster_ids)].copy()
    else:
        work = work[work["cluster_id"] > 0].copy()  # exclude noise

    if work.empty:
        if verbose:
            print("[Representatives] No satellites matched the cluster filter.")
        return {}

    # Build global feature matrix (for boundary selection in normalised space)
    features_full = build_feature_matrix(
        work, include_beta=include_beta, include_h=include_h,
    )
    features_normed, _, _ = _normalize_features(features_full)

    packs: dict[int, RepresentativePack] = {}
    unique_clusters = sorted(work["cluster_id"].unique())

    for cid in unique_clusters:
        mask = work["cluster_id"] == cid
        sub = work[mask].reset_index(drop=True)
        sub_idxs = np.where(mask.values)[0]  # indices into `work`
        n = len(sub)

        sub_features = features_normed[sub_idxs]

        medoid_local = _pick_medoid(sub_features)
        medoid_sid = str(sub.iloc[medoid_local]["sat_id"])

        boundary_local: list[int] = []
        boundary_sids: list[str] = []

        if mode in ("medoid+boundary", "full_members") and n > 1:
            boundary_local = select_boundary_members(
                sub_features, medoid_local, n_boundary=min(n_boundary, n - 1),
            )
            boundary_sids = [str(sub.iloc[bl]["sat_id"]) for bl in boundary_local]

        pack = RepresentativePack(
            cluster_id=cid,
            medoid_sat_id=medoid_sid,
            medoid_row_idx=int(sub_idxs[medoid_local]),
            boundary_sat_ids=boundary_sids,
            boundary_row_idxs=[int(sub_idxs[bl]) for bl in boundary_local],
            all_member_sat_ids=sub["sat_id"].tolist(),
            all_member_row_idxs=[int(x) for x in sub_idxs],
            n_members=n,
            feature_matrix=features_full[sub_idxs] if mode == "full_members" else None,
        )
        packs[cid] = pack

    if verbose:
        total_reps = sum(len(p.representative_sat_ids) for p in packs.values())
        total_members = sum(p.n_members for p in packs.values())
        print(f"[Representatives] Built {len(packs)} cluster packs: "
              f"{total_reps} representatives from {total_members} members "
              f"(mode={mode}, n_boundary={n_boundary})")

    return packs


def representatives_to_dataframe(
    packs: Dict[int, RepresentativePack],
) -> pd.DataFrame:
    """Flatten representative packs to a summary DataFrame for CSV export."""
    rows: list[dict] = []
    for cid, pack in sorted(packs.items()):
        for sid in pack.representative_sat_ids:
            role = "medoid" if sid == pack.medoid_sat_id else "boundary"
            rows.append({
                "global_cluster_id": cid,
                "sat_id": sid,
                "role": role,
                "n_cluster_members": pack.n_members,
            })
    return pd.DataFrame(rows)
