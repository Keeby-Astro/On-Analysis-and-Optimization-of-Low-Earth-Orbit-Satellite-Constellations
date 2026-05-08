"""RAAN-vs-phase diagnostic plots for Chapter 7 optimization.

Provides four plotting helpers that can be called standalone or from
_plot_study_diagnostics in run_constellation_optimization.py.

All functions follow the same pattern:
    plot_*(data, ..., ax=None) -> Figure | None
If ``ax`` is given, draws onto that axes; otherwise creates a new figure.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from cluster_phase_space import (
    TorusLattice,
    TorusRegularizerResult,
    ShellTorusSummary,
    circular_distance_deg,
    wrap_to_360,
)


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ======================================================================
# 1.  RAAN-phase scatter with target slots
# ======================================================================

def plot_raan_phase_shell(
    raan_deg: np.ndarray,
    phase_deg: np.ndarray,
    lattice: Optional[TorusLattice] = None,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "RAAN vs Phase (shell)",
    ax=None,
    save_path: Optional[str] = None,
):
    """Scatter plot of RAAN vs phase for all satellites in a shell.

    Parameters
    ----------
    raan_deg, phase_deg : ndarray (N,)
        Observed satellite positions.
    lattice : TorusLattice, optional
        If given, overlay target slot grid.
    cluster_labels : ndarray (N,) int, optional
        Colour points by cluster membership.
    title : str
    ax : matplotlib Axes, optional
    save_path : str, optional
        If given, save figure to this path.
    """
    plt = _ensure_matplotlib()
    if plt is None:
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Main scatter (colour by cluster if available)
    if cluster_labels is not None:
        unique = np.unique(cluster_labels)
        cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
        for i, lab in enumerate(unique):
            mask = cluster_labels == lab
            ax.scatter(raan_deg[mask], phase_deg[mask], s=12, alpha=0.6,
                       color=cmap(i % 20), label=f"C{lab}", edgecolors="none")
        ax.legend(fontsize=6, ncol=max(1, len(unique) // 8),
                  loc="upper right", markerscale=1.5)
    else:
        ax.scatter(raan_deg, phase_deg, s=12, alpha=0.5, color="#15528e",
                   edgecolors="none")

    # Overlay target lattice
    if lattice is not None and lattice.target_raan_deg.size > 0:
        ax.scatter(lattice.target_raan_deg, lattice.target_phase_deg,
                   s=80, marker="x", linewidths=1.5, color="red",
                   zorder=5, label="Target slots")
        ax.legend(fontsize=6)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel("RAAN (deg)")
    ax.set_ylabel("Phase variable (deg)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
    return fig


# ======================================================================
# 2.  Assignment residual map
# ======================================================================

def plot_raan_phase_assignment_residuals(
    raan_deg: np.ndarray,
    phase_deg: np.ndarray,
    lattice: TorusLattice,
    row_ind: np.ndarray,
    col_ind: np.ndarray,
    title: str = "Slot Assignment Residuals",
    ax=None,
    save_path: Optional[str] = None,
):
    """Draw arrows from each satellite to its assigned slot."""
    plt = _ensure_matplotlib()
    if plt is None:
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(raan_deg, phase_deg, s=15, alpha=0.4, color="#15528e",
               edgecolors="none", label="Sats")
    ax.scatter(lattice.target_raan_deg, lattice.target_phase_deg,
               s=60, marker="x", linewidths=1.2, color="red",
               zorder=5, label="Slots")

    for ri, ci in zip(row_ind, col_ind):
        dr = circular_distance_deg(lattice.target_raan_deg[ci], raan_deg[ri])
        dp = circular_distance_deg(lattice.target_phase_deg[ci], phase_deg[ri])
        ax.annotate("", xy=(raan_deg[ri] + dr, phase_deg[ri] + dp),
                     xytext=(raan_deg[ri], phase_deg[ri]),
                     arrowprops=dict(arrowstyle="->", color="gray",
                                     alpha=0.5, lw=0.7))

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel("RAAN (deg)")
    ax.set_ylabel("Phase (deg)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.4)

    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
    return fig


# ======================================================================
# 3.  Drift map (RAAN-dot and phase-dot)
# ======================================================================

def plot_raan_phase_drift_map(
    raan_drift_rates: np.ndarray,
    phase_drift_rates: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    title: str = "RAAN & Phase Drift Rates",
    ax=None,
    save_path: Optional[str] = None,
):
    """Scatter of RAAN-dot vs phase-dot for each representative."""
    plt = _ensure_matplotlib()
    if plt is None:
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    if cluster_ids is not None:
        unique = np.unique(cluster_ids)
        cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
        for i, lab in enumerate(unique):
            mask = cluster_ids == lab
            ax.scatter(raan_drift_rates[mask], phase_drift_rates[mask],
                       s=30, alpha=0.7, color=cmap(i % 20),
                       edgecolors="k", linewidths=0.3, label=f"C{lab}")
        ax.legend(fontsize=6, ncol=max(1, len(unique) // 8))
    else:
        ax.scatter(raan_drift_rates, phase_drift_rates,
                   s=30, alpha=0.7, color="#15528e",
                   edgecolors="k", linewidths=0.3)

    ax.set_xlabel("RAAN drift rate (deg/day)")
    ax.set_ylabel("Phase drift rate (deg/day)")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
    return fig


# ======================================================================
# 4.  J_torus breakdown by cluster
# ======================================================================

def plot_torus_cost_breakdown(
    cluster_ids: List[int],
    torus_results: Dict[int, TorusRegularizerResult],
    title: str = "Torus Regularizer Breakdown",
    ax=None,
    save_path: Optional[str] = None,
):
    """Stacked bar chart of J_slot / J_gap / J_drift per cluster."""
    plt = _ensure_matplotlib()
    if plt is None:
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    cids = sorted(cluster_ids)
    j_slot = [torus_results[c].J_slot if c in torus_results else 0.0 for c in cids]
    j_gap = [torus_results[c].J_gap_total if c in torus_results else 0.0 for c in cids]
    j_drift = [torus_results[c].J_drift if c in torus_results else 0.0 for c in cids]

    x = np.arange(len(cids))
    w = 0.6
    ax.bar(x, j_slot, w, label="J_slot", color="#1f77b4", alpha=0.8)
    ax.bar(x, j_gap, w, bottom=j_slot, label="J_gap", color="#ff7f0e", alpha=0.8)
    bot2 = np.array(j_slot) + np.array(j_gap)
    ax.bar(x, j_drift, w, bottom=bot2, label="J_drift", color="#2ca02c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cids], rotation=45)
    ax.set_xlabel("Global Cluster ID")
    ax.set_ylabel("Torus Regularizer Component")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
    return fig
