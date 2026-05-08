"""
Trajectory-Matching Diagnostics and Plots

Provides validation reports and visualisations for the trajectory-matching Stage A mode. 
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

PHASE_DISPLAY_NAMES = {
    "operational_shell": "Operational Shell",
    "disposal_lowering": "Disposal Lowering",
    "insertion_or_orbit_raise": "Insertion / Orbit Raise",
}

TRAJECTORY_LOSS_LABELS = {
    "loss_total": "Total",
    "loss_path": "Path",
    "loss_endpoint_a": "Endpoint a",
    "loss_endpoint_raan": "Endpoint RAAN",
    "loss_endpoint_lam": r"Endpoint $\lambda$",
    "loss_continuity": "Continuity",
}


def _phase_display_name(phase: str) -> str:
    return PHASE_DISPLAY_NAMES.get(str(phase), str(phase))


# Validation report
def trajectory_validation_report(
    arcs,
    model,
    traj_cfg,
    device: torch.device,
    out_json: Path,
) -> Dict[str, Any]:
    """Run validation on all arcs and save a JSON report.

    Parameters
    ----------
    arcs : List[ArcRecord]
    model : StageAModel (eval mode, on device)
    traj_cfg : TrajectoryConfig
    device : torch.device
    out_json : Path to write the JSON report

    Returns
    -------
    report : Dict with per-phase and aggregate metrics.
    """
    from arc_building import ArcDataset, collate_arcs
    from trajectory_matching import (
        compute_accel_net,
        trajectory_sma_forward,
        trajectory_sma_dispatch,
        trajectory_endpoint_raan,
        trajectory_endpoint_lambda,
        ussa76_drag_accel_kmps2,
    )
    from reduced_dynamics import MU_EARTH_KM3_S2, R_EARTH_KM, G0_M_S2, angle_residual

    ds = ArcDataset(arcs)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_arcs)

    model.eval()
    all_path_resid = []
    all_raan_resid = []
    all_lam_resid = []
    all_phases = []
    all_sat_ids = []
    all_n_obs = []

    with torch.no_grad():
        p = model.constrained_parameters()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            phase_idx = batch["phase_idx"]
            sat_idx = batch["sat_idx"]
            phase_sign = model.phase_signs[phase_idx]

            thrust_phase = p["thrust_N"][phase_idx] * p["sat_thrust_scale"][sat_idx]
            duty_phase = p["duty"][phase_idx]
            phase_power_cap = p["phase_power_cap_W"][phase_idx]
            phase_eta_total = p["phase_eta_total"][phase_idx]
            power_nominal_W = thrust_phase * G0_M_S2 * p["isp_s"] / (2.0 * phase_eta_total)
            if model.cfg.use_power_cap:
                power_scale = torch.clamp(phase_power_cap / torch.clamp(power_nominal_W, min=1.0), max=1.0)
            else:
                power_scale = torch.ones_like(duty_phase)
            duty_effective = torch.clamp(duty_phase * power_scale, min=1.0e-4, max=model.cfg.thermal_duty_cap)
            if model.cfg.use_drag:
                if traj_cfg.use_atmosphere_drag:
                    drag_base = ussa76_drag_accel_kmps2(
                        batch["a0_km"],
                        inv_ballistic_coeff=traj_cfg.inv_ballistic_coeff,
                    )
                    drag_phase = drag_base * p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
                else:
                    drag_phase = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
            else:
                drag_phase = torch.zeros_like(duty_effective)

            if model.cfg.use_piecewise_thrust_schedule:
                ramp = p["phase_ramp_fraction"][phase_idx]
                ramp_scale = 1.0 - 0.25 * ramp
                tss = torch.clamp(0.5 * (1.0 + p["phase_midpoint_scale"][phase_idx]), min=0.4, max=2.0)
                thrust_eff = thrust_phase * ramp_scale * tss
            else:
                thrust_eff = thrust_phase

            accel_net = compute_accel_net(
                phase_sign, thrust_eff, duty_effective, p["mass_kg"], drag_phase,
                p["shell_drag_comp_fraction"][phase_idx],
                p["phase_direction_strength"][phase_idx],
            )

            # Compute thrust-only and drag_scale for non-linear dispatch
            _direction = torch.where(
                phase_sign.abs() < 0.5,
                p["shell_drag_comp_fraction"][phase_idx],
                phase_sign * torch.clamp(
                    torch.nn.functional.softplus(p["phase_direction_strength"][phase_idx]) + 0.25,
                    min=0.25, max=1.0,
                ),
            )
            _thrust_accel = _direction * duty_effective * (thrust_eff / p["mass_kg"]) / 1000.0
            _drag_scale = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]

            a_pred = trajectory_sma_dispatch(
                batch["a0_km"], batch["dt_s"], batch["mask"], accel_net,
                traj_cfg, thrust_accel_kmps2=_thrust_accel, drag_scale=_drag_scale,
            )
            path_resid = (a_pred - batch["a_obs_km"]) * batch["mask"]

            max_obs_dim = batch["dt_s"].shape[1]
            final_idx = torch.clamp(batch["n_obs"] - 1, min=0, max=max_obs_dim - 1)
            a_final_pred = a_pred.gather(1, final_idx.unsqueeze(1)).squeeze(1)
            dt_final = batch["dt_s"].gather(1, final_idx.unsqueeze(1)).squeeze(1)

            raan_pred = trajectory_endpoint_raan(
                batch["raan0_rad"], batch["a0_km"], a_final_pred,
                batch["e0"], batch["inc0_rad"], dt_final,
            )
            lam_pred = trajectory_endpoint_lambda(
                batch["lam0_rad"], batch["a0_km"], a_final_pred,
                batch["e0"], batch["inc0_rad"], dt_final,
            )
            raan_resid = angle_residual(raan_pred, batch["raan_final_rad"])
            lam_resid = angle_residual(lam_pred, batch["lam_final_rad"])

            # Collect per-arc RMSEs
            B = path_resid.shape[0]
            for b in range(B):
                nobs = int(batch["n_obs"][b].item())
                if nobs > 0:
                    pr = path_resid[b, :nobs].cpu().numpy()
                    all_path_resid.append(float(np.sqrt(np.mean(pr ** 2))))
                    all_raan_resid.append(float(raan_resid[b].cpu()))
                    all_lam_resid.append(float(lam_resid[b].cpu()))
                    all_phases.append(arcs[len(all_path_resid) - 1].phase if len(all_path_resid) - 1 < len(arcs) else "unknown")
                    all_n_obs.append(nobs)

    # Build report
    path_rmse_arr = np.array(all_path_resid)
    raan_arr = np.abs(np.array(all_raan_resid))
    lam_arr = np.abs(np.array(all_lam_resid))
    phases_arr = np.array(all_phases)

    report: Dict[str, Any] = {
        "n_arcs": len(all_path_resid),
        "aggregate": {
            "path_rmse_km_mean": float(np.mean(path_rmse_arr)) if len(path_rmse_arr) > 0 else None,
            "path_rmse_km_median": float(np.median(path_rmse_arr)) if len(path_rmse_arr) > 0 else None,
            "path_rmse_km_p90": float(np.percentile(path_rmse_arr, 90)) if len(path_rmse_arr) > 0 else None,
            "raan_mae_rad": float(np.mean(raan_arr)) if len(raan_arr) > 0 else None,
            "lam_mae_rad": float(np.mean(lam_arr)) if len(lam_arr) > 0 else None,
        },
        "per_phase": {},
    }

    for phase in sorted(set(all_phases)):
        mask = phases_arr == phase
        report["per_phase"][phase] = {
            "n_arcs": int(mask.sum()),
            "path_rmse_km_mean": float(np.mean(path_rmse_arr[mask])),
            "path_rmse_km_median": float(np.median(path_rmse_arr[mask])),
            "raan_mae_rad": float(np.mean(raan_arr[mask])),
            "lam_mae_rad": float(np.mean(lam_arr[mask])),
        }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Trajectory validation report: %s", str(out_json))
    return report


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_trajectory_fits(
    arcs,
    model,
    device: torch.device,
    out_dir: Path,
    max_plots: int = 12,
    traj_cfg=None,
) -> List[str]:
    """Plot observed vs predicted SMA for individual arcs.

    Selects worst-fit arcs (by path RMSE) and plots each.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping trajectory plots.")
        return []

    from arc_building import ArcDataset, collate_arcs
    from trajectory_matching import (
        compute_accel_net,
        trajectory_sma_forward,
        trajectory_sma_dispatch,
        ussa76_drag_accel_kmps2,
    )
    from reduced_dynamics import G0_M_S2

    ds = ArcDataset(arcs)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=collate_arcs)

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        p = model.constrained_parameters()
        phase_idx = batch["phase_idx"]
        sat_idx = batch["sat_idx"]
        phase_sign = model.phase_signs[phase_idx]

        thrust_phase = p["thrust_N"][phase_idx] * p["sat_thrust_scale"][sat_idx]
        duty_phase = p["duty"][phase_idx]
        phase_eta_total = p["phase_eta_total"][phase_idx]
        power_nominal_W = thrust_phase * G0_M_S2 * p["isp_s"] / (2.0 * phase_eta_total)
        phase_power_cap = p["phase_power_cap_W"][phase_idx]
        if model.cfg.use_power_cap:
            power_scale = torch.clamp(phase_power_cap / torch.clamp(power_nominal_W, min=1.0), max=1.0)
        else:
            power_scale = torch.ones_like(duty_phase)
        duty_effective = torch.clamp(duty_phase * power_scale, min=1.0e-4, max=model.cfg.thermal_duty_cap)
        if model.cfg.use_drag:
            _use_atm = getattr(model.cfg, 'use_atmosphere_drag', False)
            if _use_atm:
                drag_base = ussa76_drag_accel_kmps2(
                    batch["a0_km"],
                    inv_ballistic_coeff=getattr(model.cfg, 'inv_ballistic_coeff', 0.0334),
                )
                drag_phase = drag_base * p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
            else:
                drag_phase = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
        else:
            drag_phase = torch.zeros_like(duty_effective)

        if model.cfg.use_piecewise_thrust_schedule:
            ramp = p["phase_ramp_fraction"][phase_idx]
            ramp_scale = 1.0 - 0.25 * ramp
            tss = torch.clamp(0.5 * (1.0 + p["phase_midpoint_scale"][phase_idx]), min=0.4, max=2.0)
            thrust_eff = thrust_phase * ramp_scale * tss
        else:
            thrust_eff = thrust_phase

        accel_net = compute_accel_net(
            phase_sign, thrust_eff, duty_effective, p["mass_kg"], drag_phase,
            p["shell_drag_comp_fraction"][phase_idx],
            p["phase_direction_strength"][phase_idx],
        )

        if traj_cfg is not None:
            _direction = torch.where(
                phase_sign.abs() < 0.5,
                p["shell_drag_comp_fraction"][phase_idx],
                phase_sign * torch.clamp(
                    torch.nn.functional.softplus(p["phase_direction_strength"][phase_idx]) + 0.25,
                    min=0.25, max=1.0,
                ),
            )
            _thrust_accel = _direction * duty_effective * (thrust_eff / p["mass_kg"]) / 1000.0
            _drag_scale = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
            a_pred = trajectory_sma_dispatch(
                batch["a0_km"], batch["dt_s"], batch["mask"], accel_net,
                traj_cfg, thrust_accel_kmps2=_thrust_accel, drag_scale=_drag_scale,
            )
        else:
            a_pred = trajectory_sma_forward(batch["a0_km"], batch["dt_s"], batch["mask"], accel_net)

    # Move to CPU
    dt_np = batch["dt_s"].cpu().numpy()
    a_obs_np = batch["a_obs_km"].cpu().numpy()
    a_pred_np = a_pred.cpu().numpy()
    mask_np = batch["mask"].cpu().numpy()
    n_obs_np = batch["n_obs"].cpu().numpy()

    # Compute per-arc RMSE and sort
    rmses = []
    for b in range(len(arcs)):
        nobs = int(n_obs_np[b])
        if nobs > 0:
            resid = a_pred_np[b, :nobs] - a_obs_np[b, :nobs]
            rmses.append(float(np.sqrt(np.mean(resid ** 2))))
        else:
            rmses.append(0.0)

    # Sort by RMSE descending (worst first)
    order = np.argsort(rmses)[::-1]
    n_plots = min(max_plots, len(arcs))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    for rank in range(n_plots):
        idx = int(order[rank])
        arc = arcs[idx]
        nobs = int(n_obs_np[idx])
        if nobs < 2:
            continue

        dt_days = dt_np[idx, :nobs] / 86400.0
        a_obs = a_obs_np[idx, :nobs]
        a_pred_arc = a_pred_np[idx, :nobs]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(dt_days, a_obs, "o", markersize=3, color="C0", label="TLE observed", alpha=0.7)
        ax1.plot(dt_days, a_pred_arc, "-", coloSr="C1", linewidth=1.5, label="Model predicted")
        ax1.set_ylabel("SMA [km]")
        ax1.set_title(f"{arc.sat_id} — {arc.phase} (RMSE={rmses[idx]:.2f} km)")
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        resid = a_pred_arc - a_obs
        ax2.bar(dt_days, resid, width=np.median(np.diff(dt_days)) * 0.8 if len(dt_days) > 1 else 0.5,
                color="C2", alpha=0.6)
        ax2.axhline(0, color="k", linewidth=0.5)
        ax2.set_ylabel("Residual [km]")
        ax2.set_xlabel("Time from arc start [days]")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f"trajectory_fit_{rank:02d}_{arc.sat_id}_{arc.phase}.png"
        fig.savefig(out_dir / fname, dpi=600, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(fname)

    # Grid overview plot
    if n_plots >= 4:
        n_cols = min(4, n_plots)
        n_rows = (min(n_plots, 16) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.atleast_2d(axes)
        for r in range(n_rows):
            for c in range(n_cols):
                flat = r * n_cols + c
                ax = axes[r, c]
                if flat >= n_plots:
                    ax.set_visible(False)
                    continue
                idx = int(order[flat])
                nobs = int(n_obs_np[idx])
                if nobs < 2:
                    ax.set_visible(False)
                    continue
                dt_days = dt_np[idx, :nobs] / 86400.0
                ax.plot(dt_days, a_obs_np[idx, :nobs], ".", markersize=2, color="C0")
                ax.plot(dt_days, a_pred_np[idx, :nobs], "-", linewidth=1, color="C1")
                ax.set_title(f"{arcs[idx].sat_id[:12]} RMSE={rmses[idx]:.1f}", fontsize=7)
                ax.tick_params(labelsize=6)
        fig.suptitle("Trajectory fits (worst arcs)", fontsize=10)
        fig.tight_layout()
        fname = "trajectory_fit_grid.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(fname)

    logger.info("Generated %d trajectory fit plots in %s", len(plot_paths), str(out_dir))
    return plot_paths


def plot_trajectory_residual_analysis(
    arcs,
    model,
    device: torch.device,
    out_dir: Path,
    traj_cfg=None,
) -> List[str]:
    """Histogram of per-arc path RMSE by phase, and residual vs duration scatter."""
    if not MATPLOTLIB_AVAILABLE:
        return []

    from arc_building import ArcDataset, collate_arcs
    from trajectory_matching import compute_accel_net, trajectory_sma_forward, trajectory_sma_dispatch, ussa76_drag_accel_kmps2
    from reduced_dynamics import G0_M_S2

    ds = ArcDataset(arcs)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=collate_arcs)

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        p = model.constrained_parameters()
        phase_idx = batch["phase_idx"]
        sat_idx = batch["sat_idx"]
        phase_sign = model.phase_signs[phase_idx]

        thrust_phase = p["thrust_N"][phase_idx] * p["sat_thrust_scale"][sat_idx]
        duty_phase = p["duty"][phase_idx]
        phase_eta_total = p["phase_eta_total"][phase_idx]
        power_nominal_W = thrust_phase * G0_M_S2 * p["isp_s"] / (2.0 * phase_eta_total)
        phase_power_cap = p["phase_power_cap_W"][phase_idx]
        if model.cfg.use_power_cap:
            power_scale = torch.clamp(phase_power_cap / torch.clamp(power_nominal_W, min=1.0), max=1.0)
        else:
            power_scale = torch.ones_like(duty_phase)
        duty_effective = torch.clamp(duty_phase * power_scale, min=1.0e-4, max=model.cfg.thermal_duty_cap)
        if model.cfg.use_drag:
            _use_atm = getattr(model.cfg, 'use_atmosphere_drag', False)
            if _use_atm:
                drag_base = ussa76_drag_accel_kmps2(
                    batch["a0_km"],
                    inv_ballistic_coeff=getattr(model.cfg, 'inv_ballistic_coeff', 0.0334),
                )
                drag_phase = drag_base * p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
            else:
                drag_phase = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
        else:
            drag_phase = torch.zeros_like(duty_effective)

        if model.cfg.use_piecewise_thrust_schedule:
            ramp = p["phase_ramp_fraction"][phase_idx]
            ramp_scale = 1.0 - 0.25 * ramp
            tss = torch.clamp(0.5 * (1.0 + p["phase_midpoint_scale"][phase_idx]), min=0.4, max=2.0)
            thrust_eff = thrust_phase * ramp_scale * tss
        else:
            thrust_eff = thrust_phase

        accel_net = compute_accel_net(
            phase_sign, thrust_eff, duty_effective, p["mass_kg"], drag_phase,
            p["shell_drag_comp_fraction"][phase_idx],
            p["phase_direction_strength"][phase_idx],
        )

        if traj_cfg is not None:
            _direction = torch.where(
                phase_sign.abs() < 0.5,
                p["shell_drag_comp_fraction"][phase_idx],
                phase_sign * torch.clamp(
                    torch.nn.functional.softplus(p["phase_direction_strength"][phase_idx]) + 0.25,
                    min=0.25, max=1.0,
                ),
            )
            _thrust_accel = _direction * duty_effective * (thrust_eff / p["mass_kg"]) / 1000.0
            _drag_scale = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
            a_pred = trajectory_sma_dispatch(
                batch["a0_km"], batch["dt_s"], batch["mask"], accel_net,
                traj_cfg, thrust_accel_kmps2=_thrust_accel, drag_scale=_drag_scale,
            )
        else:
            a_pred = trajectory_sma_forward(batch["a0_km"], batch["dt_s"], batch["mask"], accel_net)

    n_obs_np = batch["n_obs"].cpu().numpy()
    a_obs_np = batch["a_obs_km"].cpu().numpy()
    a_pred_np = a_pred.cpu().numpy()
    duration_np = batch["duration_s"].cpu().numpy()

    rmses = []
    phases = []
    durations = []
    for b in range(len(arcs)):
        nobs = int(n_obs_np[b])
        if nobs > 0:
            resid = a_pred_np[b, :nobs] - a_obs_np[b, :nobs]
            rmses.append(float(np.sqrt(np.mean(resid ** 2))))
            phases.append(arcs[b].phase)
            durations.append(float(duration_np[b]) / 86400.0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    # ── Histogram by phase ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.75))
    unique_phases = sorted(set(phases))
    for i, phase in enumerate(unique_phases):
        vals = [r for r, p in zip(rmses, phases) if p == phase]
        ax.hist(vals, bins=30, alpha=0.5, label=f"{_phase_display_name(phase)} (n={len(vals)})")
    ax.set_xlabel("Path RMSE (km)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "trajectory_rmse_by_phase.png"
    fig.savefig(out_dir / fname, dpi=600, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(fname)

    # ── Scatter: RMSE vs duration ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.75))
    colors = {"insertion_or_orbit_raise": "C0", "operational_shell": "C1", "disposal_lowering": "C2"}
    for phase in unique_phases:
        mask = [p == phase for p in phases]
        r = [rmses[i] for i in range(len(mask)) if mask[i]]
        d = [durations[i] for i in range(len(mask)) if mask[i]]
        ax.scatter(d, r, s=10, alpha=0.5, label=_phase_display_name(phase), color=colors.get(phase, "grey"))
    ax.set_xlabel("Arc Duration (days)")
    ax.set_ylabel("Path RMSE (km)")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "trajectory_rmse_vs_duration.png"
    fig.savefig(out_dir / fname, dpi=600, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(fname)

    logger.info("Generated %d residual analysis plots in %s", len(plot_paths), str(out_dir))
    return plot_paths


# ── Training history plots ───────────────────────────────────────────────────

def plot_trajectory_training_history(
    history_csv: Path,
    out_dir: Path,
) -> List[str]:
    """Plot trajectory-mode loss components and RMSE over epochs."""
    if not MATPLOTLIB_AVAILABLE:
        return []

    history_csv = Path(history_csv)
    if not history_csv.exists():
        logger.warning("History CSV not found: %s", str(history_csv))
        return []

    df = pd.read_csv(history_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    # ── 1) Loss components vs epoch ──────────────────────────────────────
    loss_cols = [c for c in df.columns if c.startswith("loss_") and c != "loss_total"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.75))
    ax1, ax2 = axes

    if "loss_total" in df.columns:
        ax1.plot(df["epoch"], df["loss_total"], "k-", linewidth=2, label=TRAJECTORY_LOSS_LABELS["loss_total"])
    for col in loss_cols:
        if col in df.columns:
            ax1.plot(df["epoch"], df[col], linewidth=1, alpha=0.8, label=TRAJECTORY_LOSS_LABELS.get(col, col.replace("loss_", "")))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Trajectory Loss Components")
    ax1.legend(fontsize=12, ncols=2, loc="upper center", bbox_to_anchor=(0.58, 0.74))
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    if "path_rmse_km" in df.columns:
        ax2.plot(df["epoch"], df["path_rmse_km"], "C0-", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Path RMSE (km)")
        ax2.set_title("SMA Path RMSE per Epoch")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = "trajectory_training_history.png"
    fig.savefig(out_dir / fname, dpi=600, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(fname)

    # ── 2) Individual loss terms ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.5))
    all_loss_cols = ["loss_path", "loss_endpoint_a", "loss_endpoint_raan",
                     "loss_endpoint_lam", "loss_continuity", "path_rmse_km"]
    titles = ["SMA Path", "SMA Endpoint", "RAAN Endpoint",
              "Lambda Endpoint", "Continuity", "Path RMSE [km]"]
    for ax, col, title in zip(axes.flat, all_loss_cols, titles):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], linewidth=1.5)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    fname = "trajectory_loss_breakdown.png"
    fig.savefig(out_dir / fname, dpi=600, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(fname)

    logger.info("Generated %d training history plots in %s", len(plot_paths), str(out_dir))
    return plot_paths


# ── Parameter evolution plots ────────────────────────────────────────────────

def plot_trajectory_parameter_evolution(
    trace_csv: Path,
    out_dir: Path,
) -> List[str]:
    """Plot per-phase parameter evolution (thrust, duty, drag, shell_drag_comp) over training."""
    if not MATPLOTLIB_AVAILABLE:
        return []

    trace_csv = Path(trace_csv)
    if not trace_csv.exists():
        logger.warning("Trace CSV not found: %s", str(trace_csv))
        return []

    df = pd.read_csv(trace_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    # Detect phase names from column pattern "thrust_N__<phase>"
    thrust_cols = [c for c in df.columns if c.startswith("thrust_N__")]
    phase_names = [c.replace("thrust_N__", "") for c in thrust_cols]
    if not phase_names:
        return plot_paths

    # ── Global parameters ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.5))
    globals_cols = ["mass_kg", "dry_mass_kg", "isp_s", "eta_total",
                    "sat_thrust_scale_mean", "sat_drag_scale_mean"]
    for ax, col in zip(axes.flat, globals_cols):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], linewidth=1.5)
        ax.set_title(col, fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    fname = "trajectory_global_params.png"
    fig.savefig(out_dir / fname, dpi=600, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(fname)

    # ── Per-phase parameters ─────────────────────────────────────────────
    param_types = [
        ("thrust_N", "Thrust"),
        ("duty", "Duty"),
        ("drag_kmps2", "Drag"),
        ("shell_drag_comp", "Shell Drag"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, (ptype, title) in zip(axes.flat, param_types):
        for phase in phase_names:
            col = f"{ptype}__{phase}"
            if col in df.columns:
                label = _phase_display_name(phase)
                ax.plot(df["epoch"], df[col], linewidth=1.2, label=label)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=12, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    fname = "trajectory_phase_params.png"
    fig.savefig(out_dir / fname, dpi=600, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(fname)

    logger.info("Generated %d parameter evolution plots in %s", len(plot_paths), str(out_dir))
    return plot_paths


# ── Per-satellite RMSE box plot ──────────────────────────────────────────────

def plot_trajectory_per_satellite_rmse(
    arcs,
    model,
    device: torch.device,
    out_dir: Path,
    traj_cfg=None,
) -> List[str]:
    """Box plot of path RMSE grouped by satellite."""
    if not MATPLOTLIB_AVAILABLE:
        return []

    from arc_building import ArcDataset, collate_arcs
    from trajectory_matching import compute_accel_net, trajectory_sma_forward, trajectory_sma_dispatch, ussa76_drag_accel_kmps2
    from reduced_dynamics import G0_M_S2

    ds = ArcDataset(arcs)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=collate_arcs)

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        p = model.constrained_parameters()
        phase_idx = batch["phase_idx"]
        sat_idx = batch["sat_idx"]
        phase_sign = model.phase_signs[phase_idx]

        thrust_phase = p["thrust_N"][phase_idx] * p["sat_thrust_scale"][sat_idx]
        duty_phase = p["duty"][phase_idx]
        phase_eta_total = p["phase_eta_total"][phase_idx]
        power_nominal_W = thrust_phase * G0_M_S2 * p["isp_s"] / (2.0 * phase_eta_total)
        phase_power_cap = p["phase_power_cap_W"][phase_idx]
        if model.cfg.use_power_cap:
            power_scale = torch.clamp(phase_power_cap / torch.clamp(power_nominal_W, min=1.0), max=1.0)
        else:
            power_scale = torch.ones_like(duty_phase)
        duty_effective = torch.clamp(duty_phase * power_scale, min=1.0e-4, max=model.cfg.thermal_duty_cap)
        if model.cfg.use_drag:
            _use_atm = getattr(model.cfg, 'use_atmosphere_drag', False)
            if _use_atm:
                drag_base = ussa76_drag_accel_kmps2(
                    batch["a0_km"],
                    inv_ballistic_coeff=getattr(model.cfg, 'inv_ballistic_coeff', 0.0334),
                )
                drag_phase = drag_base * p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
            else:
                drag_phase = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
        else:
            drag_phase = torch.zeros_like(duty_effective)

        if model.cfg.use_piecewise_thrust_schedule:
            ramp = p["phase_ramp_fraction"][phase_idx]
            ramp_scale = 1.0 - 0.25 * ramp
            tss = torch.clamp(0.5 * (1.0 + p["phase_midpoint_scale"][phase_idx]), min=0.4, max=2.0)
            thrust_eff = thrust_phase * ramp_scale * tss
        else:
            thrust_eff = thrust_phase

        accel_net = compute_accel_net(
            phase_sign, thrust_eff, duty_effective, p["mass_kg"], drag_phase,
            p["shell_drag_comp_fraction"][phase_idx],
            p["phase_direction_strength"][phase_idx],
        )

        if traj_cfg is not None:
            _direction = torch.where(
                phase_sign.abs() < 0.5,
                p["shell_drag_comp_fraction"][phase_idx],
                phase_sign * torch.clamp(
                    torch.nn.functional.softplus(p["phase_direction_strength"][phase_idx]) + 0.25,
                    min=0.25, max=1.0,
                ),
            )
            _thrust_accel = _direction * duty_effective * (thrust_eff / p["mass_kg"]) / 1000.0
            _drag_scale = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
            a_pred = trajectory_sma_dispatch(
                batch["a0_km"], batch["dt_s"], batch["mask"], accel_net,
                traj_cfg, thrust_accel_kmps2=_thrust_accel, drag_scale=_drag_scale,
            )
        else:
            a_pred = trajectory_sma_forward(batch["a0_km"], batch["dt_s"], batch["mask"], accel_net)

    n_obs_np = batch["n_obs"].cpu().numpy()
    a_obs_np = batch["a_obs_km"].cpu().numpy()
    a_pred_np = a_pred.cpu().numpy()

    # Collect per-arc RMSE keyed by satellite
    sat_rmses: Dict[str, List[float]] = {}
    for b in range(len(arcs)):
        nobs = int(n_obs_np[b])
        if nobs > 0:
            resid = a_pred_np[b, :nobs] - a_obs_np[b, :nobs]
            rmse = float(np.sqrt(np.mean(resid ** 2)))
            sid = arcs[b].sat_id
            sat_rmses.setdefault(sid, []).append(rmse)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    sorted_sats = sorted(sat_rmses.keys(), key=lambda s: np.median(sat_rmses[s]), reverse=True)
    if len(sorted_sats) > 30:
        sorted_sats = sorted_sats[:30]

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_sats) * 0.5), 5))
    data = [sat_rmses[s] for s in sorted_sats]
    labels = [s[:12] for s in sorted_sats]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("C0")
        patch.set_alpha(0.4)
    ax.set_ylabel("Path RMSE [km]")
    ax.set_title("Per-satellite trajectory RMSE")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = "trajectory_per_satellite_rmse.png"
    fig.savefig(out_dir / fname, dpi=600, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(fname)

    logger.info("Generated per-satellite RMSE plot in %s", str(out_dir))
    return plot_paths
