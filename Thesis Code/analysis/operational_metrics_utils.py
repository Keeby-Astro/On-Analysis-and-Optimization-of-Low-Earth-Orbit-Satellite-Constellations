"""Shared helpers for operational analytics.

Scope labels used across shell/disposal/sustainability/risk products:
- descriptive
- proxy
- compliance-oriented
- conjunction-input-readiness

Notes:
- These helpers support deterministic, duplicate-safe conditioning for archive panels.
- Outputs remain descriptive/proxy unless explicitly flagged as compliance-oriented.
- This utility layer does not implement covariance-aware conjunction probability.

Reference context:
- IADC mitigation guidelines (Revision 2)
- FCC 5-year LEO disposal rule and compliance guide updates
- NASA CARA Appendix N boundary for Pc (covariance-aware conjunction assessment)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from orbital_features import ensure_altitude, resolve_object_col

SCOPE_DESCRIPTIVE = "descriptive"
SCOPE_PROXY = "proxy"
SCOPE_COMPLIANCE = "compliance-oriented"
SCOPE_CONJUNCTION_INPUT_READINESS = "conjunction-input-readiness"
SCOPE_CONJUNCTION_PLACEHOLDER = SCOPE_CONJUNCTION_INPUT_READINESS


@dataclass(frozen=True)
class ScopeLabels:
    descriptive: str = SCOPE_DESCRIPTIVE
    proxy: str = SCOPE_PROXY
    compliance_oriented: str = SCOPE_COMPLIANCE
    conjunction_placeholder: str = SCOPE_CONJUNCTION_INPUT_READINESS


def get_scope_labels() -> ScopeLabels:
    return ScopeLabels()


def resolve_object_column(work: pd.DataFrame, object_col: str | None = None) -> str:
    """Resolve object identity column with explicit fallback behavior."""
    if object_col is not None:
        if object_col not in work.columns:
            raise KeyError(f"Requested object column '{object_col}' not present.")
        return object_col
    return resolve_object_col(work)


def prepare_time_binned_panel(
    df: pd.DataFrame,
    time_freq: str,
    object_col: str | None = None,
    *,
    sort_by_time_only: bool = False,
    ensure_altitude_features: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Standardize timestamps/object IDs and add deterministic time bins."""
    if ensure_altitude_features:
        work = ensure_altitude(df).copy()
    else:
        work = df.copy()
    resolved_object_col = resolve_object_column(work, object_col=object_col)

    work[resolved_object_col] = work[resolved_object_col].astype(str)
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])

    if sort_by_time_only:
        work = work.sort_values(["timestamp"], kind="mergesort")
    else:
        work = work.sort_values([resolved_object_col, "timestamp"], kind="mergesort")

    work["time_bin"] = work["timestamp"].dt.floor(time_freq)
    return work, resolved_object_col


def apply_duplicate_safe_conditioning(
    work: pd.DataFrame,
    subset: Iterable[str],
    *,
    keep: str = "last",
) -> pd.DataFrame:
    """Drop duplicate rows deterministically using a stable sort and key subset."""
    valid_subset = [col for col in subset if col in work.columns]
    if not valid_subset:
        return work.copy()

    ordered = work.sort_values(valid_subset, kind="mergesort")
    return ordered.drop_duplicates(subset=valid_subset, keep=keep).reset_index(drop=True)


def resolve_shell_identity(
    work: pd.DataFrame,
    *,
    shell_col: str = "candidate_shell_id",
    altitude_bins: list[float] | None = None,
    output_col: str = "shell_or_bin",
) -> pd.DataFrame:
    """Resolve shell identity with explicit altitude-bin fallback labeling."""
    if altitude_bins is None:
        altitude_bins = [300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500]

    out = work.copy()
    out["altitude_bin"] = pd.cut(
        pd.to_numeric(out["altitude_km"], errors="coerce"),
        bins=altitude_bins,
        include_lowest=True,
    ).astype(str)

    if shell_col in out.columns:
        shell_series = out[shell_col].astype("object")
    else:
        shell_series = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")

    out[output_col] = shell_series.astype(str)
    missing_shell = shell_series.isna() | (out[output_col] == "<NA>") | (out[output_col] == "nan")
    out.loc[missing_shell, output_col] = out.loc[missing_shell, "altitude_bin"]

    out["shell_identity_source"] = np.where(missing_shell, "altitude_bin_fallback", "candidate_shell_id")
    out["shell_is_altitude_fallback"] = missing_shell.astype(bool)
    if "shell_assignment_basis" not in out.columns:
        out["shell_assignment_basis"] = np.where(missing_shell, "altitude_bin_fallback", "candidate_shell_id")
    else:
        basis = out["shell_assignment_basis"].astype("object")
        basis = basis.where(~missing_shell, "altitude_bin_fallback")
        out["shell_assignment_basis"] = basis
    return out


def get_compliance_horizons_years() -> dict[str, int]:
    """Return standard disposal compliance horizons"""
    return {"five_year": 5, "twenty_five_year": 25}


def add_compliance_horizon_columns(
    df: pd.DataFrame,
    *,
    reference_col: str,
    prefix: str = "estimated",
    compliance_horizons_years: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Add 5-year and 25-year horizon dates from a reference timestamp column."""
    out = df.copy()
    horizons = get_compliance_horizons_years()
    if compliance_horizons_years is not None:
        try:
            vals = sorted({int(v) for v in compliance_horizons_years if int(v) > 0})
        except Exception:
            vals = []
        if vals:
            horizons = {
                'five_year': vals[0],
                'twenty_five_year': vals[1] if len(vals) > 1 else vals[0],
            }
    ref = pd.to_datetime(out.get(reference_col), errors="coerce")

    out[f"{prefix}_five_year_horizon"] = ref + pd.DateOffset(years=horizons["five_year"])
    out[f"{prefix}_twenty_five_year_horizon"] = ref + pd.DateOffset(years=horizons["twenty_five_year"])
    return out


def safe_ratio(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(num, den, out=np.full_like(num, np.nan), where=np.isfinite(den) & (den != 0.0))
    return out


def add_occupancy_normalization(
    shell_time_df: pd.DataFrame,
    *,
    n_records_col: str = "n_records",
    n_objects_col: str = "n_objects",
) -> pd.DataFrame:
    """Add occupancy normalization proxies by time bin (record and object shares)."""
    out = shell_time_df.copy()
    if out.empty:
        out["record_share_in_time_bin"] = np.nan
        out["object_share_in_time_bin"] = np.nan
        return out

    total_records = out.groupby("time_bin", observed=False)[n_records_col].transform("sum")
    total_objects = out.groupby("time_bin", observed=False)[n_objects_col].transform("sum")

    out["record_share_in_time_bin"] = safe_ratio(out[n_records_col], total_records)
    out["object_share_in_time_bin"] = safe_ratio(out[n_objects_col], total_objects)
    return out


def stable_time_rank(work: pd.DataFrame, key_col: str = "timestamp") -> np.ndarray:
    """Create a stable numeric time rank for slope/centroid trend calculations."""
    ts = pd.to_datetime(work[key_col], errors="coerce")
    return ts.astype("int64").to_numpy(dtype=np.float64)
