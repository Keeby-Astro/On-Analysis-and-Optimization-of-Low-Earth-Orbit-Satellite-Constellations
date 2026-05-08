"""Regenerate selected Stage A diagnostic plots from saved run artifacts.

This script reloads CSV/JSON/checkpoint outputs from an existing run directory
and rewrites the requested plot images without rerunning Stage A training or
Stage B inference. The trajectory RMSE-by-phase plot requires rebuilding
diagnostic arcs from TLE files and reloading the saved checkpoint, but it does
not retrain the model.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from cycler import cycler

from arc_building import ArcBuildConfig, build_arcs_from_tles_and_intervals
from starlink_stage_ab_starter import (
    PlotConfig,
    SmoothingConfig,
    generate_data_quality_plots,
    generate_stage_a_fit_plots,
    generate_stage_a_parameter_plots,
    generate_stage_a_training_plots,
    generate_timing_sensitivity_plots,
    load_stage_a_model_from_checkpoint,
    load_tle_data_with_progress,
    preprocess_tle_dataframe,
    select_tle_files,
)
from trajectory_diagnostics import (
    plot_trajectory_parameter_evolution,
    plot_trajectory_residual_analysis,
    plot_trajectory_training_history,
)
from trajectory_matching import TrajectoryConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLORS = [
    "#15528e", "#b25800", "#1e701e", "#951c1c", "#673284",
    "#623c34", "#9e5387", "#585858", "#848417", "#108590",
    "#798ba2", "#b28254", "#6a9c60", "#b26a68", "#8a7b94",
    "#896d67", "#ac7f93", "#8b8b8b", "#999962", "#6f989f",
]


def apply_plot_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (10.0, 7.5),
        "xtick.direction": "in", "xtick.labelsize": 14, "xtick.major.size": 3,
        "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.5,
        "xtick.minor.visible": True, "xtick.top": True,
        "ytick.direction": "in", "ytick.labelsize": 14, "ytick.major.size": 3,
        "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.5,
        "ytick.minor.visible": True, "ytick.right": True,
        "axes.linewidth": 0.5, "grid.linewidth": 0.5, "lines.linewidth": 1.0,
        "legend.fontsize": 14, "legend.frameon": False,
        "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "dejavuserif",
        "font.size": 12, "axes.labelsize": 16, "axes.titlesize": 18,
        "axes.grid": True, "grid.linestyle": "--", "grid.color": "0.5",
        "lines.markersize": 8, "axes.spines.top": True, "axes.spines.right": True,
    })
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLORS)


def _resolve_path(path: Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    return obj if isinstance(obj, dict) else {}


def _plot_dir(run_dir: Path, plot_root: Optional[Path], name: str) -> Path:
    root = plot_root if plot_root is not None else run_dir / "plots"
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _append_paths(paths: List[str], written: List[str]) -> None:
    for path in paths:
        path_str = str(path)
        if path_str not in written:
            written.append(path_str)


def _build_trajectory_config(stage_a_cfg: Dict[str, Any]) -> TrajectoryConfig:
    return TrajectoryConfig(
        lambda_path=float(stage_a_cfg.get("lambda_path", 5.0)),
        lambda_endpoint_a=float(stage_a_cfg.get("lambda_endpoint_a", 1.0)),
        lambda_endpoint_raan=float(stage_a_cfg.get("lambda_endpoint_raan", 0.0)),
        lambda_endpoint_lam=float(stage_a_cfg.get("lambda_endpoint_lam", 0.0)),
        lambda_continuity=float(stage_a_cfg.get("lambda_continuity", 0.1)),
        max_subarc_days=float(stage_a_cfg.get("max_subarc_days", 30.0)),
        arc_weight_mode=str(stage_a_cfg.get("arc_weight_mode", "sqrt_inv_n_obs")),
        robust_loss=str(stage_a_cfg.get("robust_loss", "mse")),
        huber_delta=float(stage_a_cfg.get("huber_delta", 1.0)),
        use_atmosphere_drag=bool(stage_a_cfg.get("use_atmosphere_drag", True)),
        inv_ballistic_coeff=float(stage_a_cfg.get("inv_ballistic_coeff", 0.0334)),
        nonlinear_propagation=bool(stage_a_cfg.get("nonlinear_propagation", True)),
        rk4_step_hours=float(stage_a_cfg.get("rk4_step_hours", 12.0)),
    )


def regenerate_artifact_plots(run_dir: Path, plot_root: Optional[Path], plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []

    segments_csv = run_dir / "segments" / "segments.csv"
    if segments_csv.exists():
        seg_df = pd.read_csv(segments_csv)
        _append_paths(
            generate_data_quality_plots(seg_df, _plot_dir(run_dir, plot_root, "data_quality"), plot_cfg),
            written,
        )
    else:
        logging.warning("Skipping data-quality plots; missing %s", segments_csv)

    history_csv = run_dir / "checkpoints" / "stage_a" / "stage_a_history.csv"
    if history_csv.exists():
        history_df = pd.read_csv(history_csv)
        train_dir = _plot_dir(run_dir, plot_root, "train")
        _append_paths(generate_stage_a_training_plots(history_df, train_dir, plot_cfg), written)
        _append_paths([str(train_dir / name) for name in plot_trajectory_training_history(history_csv, train_dir)], written)
    else:
        logging.warning("Skipping training plots; missing %s", history_csv)

    predictions_csv = run_dir / "tables" / "stage_a_predictions.csv"
    if predictions_csv.exists():
        pred_df = pd.read_csv(predictions_csv)
        _append_paths(generate_stage_a_fit_plots(pred_df, _plot_dir(run_dir, plot_root, "fit"), plot_cfg), written)
    else:
        logging.warning("Skipping observed-vs-predicted plot; missing %s", predictions_csv)

    trace_csv = run_dir / "checkpoints" / "stage_a" / "stage_a_parameter_trace.csv"
    summary_json = run_dir / "checkpoints" / "stage_a" / "stage_a_parameter_summary.json"
    if trace_csv.exists():
        trace_df = pd.read_csv(trace_csv)
        summary = _load_json(summary_json) if summary_json.exists() else {}
        param_dir = _plot_dir(run_dir, plot_root, "parameters")
        _append_paths(generate_stage_a_parameter_plots(trace_df, summary, param_dir, plot_cfg), written)
        _append_paths([str(param_dir / name) for name in plot_trajectory_parameter_evolution(trace_csv, param_dir)], written)
    else:
        logging.warning("Skipping parameter plots; missing %s", trace_csv)

    timing_csv = run_dir / "tables" / "timing_sensitivity.csv"
    if timing_csv.exists():
        timing_df = pd.read_csv(timing_csv)
        _append_paths(generate_timing_sensitivity_plots(timing_df, _plot_dir(run_dir, plot_root, "sensitivity"), plot_cfg), written)
    else:
        logging.warning("Skipping timing sensitivity plot; missing %s", timing_csv)

    return written


def regenerate_trajectory_rmse(run_dir: Path, plot_root: Optional[Path], config: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    stage_a_cfg = dict(config.get("stage_a", {}))
    run_cfg = dict(config.get("run", {}))
    paths_cfg = dict(config.get("paths", {}))

    tle_dir = _resolve_path(Path(paths_cfg.get("tle_dir", "starlink_backup")))
    max_sats = args.max_sats if args.max_sats is not None else int(run_cfg.get("max_sats", 500))
    selected_files = select_tle_files(tle_dir, max_sats)

    tle_df, _ = load_tle_data_with_progress(
        tle_dir=tle_dir,
        only_files=selected_files,
        derived_cols=("sma",),
        workers=args.tle_workers,
        chunk_size=int(args.tle_chunk_size or run_cfg.get("tle_chunk_size") or 128),
        progress_every_files=int(args.tle_progress_files or run_cfg.get("tle_progress_files") or 100),
    )
    smoothing_cfg = SmoothingConfig(**dict(config.get("smoothing", {})))
    tle_df = preprocess_tle_dataframe(tle_df, smoothing_cfg)

    intervals_raw = str(stage_a_cfg.get("intervals_csv") or "")
    intervals_csv = _resolve_path(Path(intervals_raw)) if intervals_raw else PROJECT_ROOT / "full_exports" / "maneuver_phase_intervals_gen1_full.csv"
    intervals_df = pd.read_csv(intervals_csv)

    arc_cfg = ArcBuildConfig(
        min_obs=int(stage_a_cfg.get("min_arc_obs", 5)),
        min_duration_s=6.0 * 3600.0,
        max_duration_s=120.0 * 86400.0,
    )
    arcs = build_arcs_from_tles_and_intervals(tle_df, intervals_df, arc_cfg)
    if not arcs:
        logging.warning("Skipping trajectory RMSE plots; no diagnostic arcs were rebuilt.")
        return []

    checkpoint = run_dir / "checkpoints" / "stage_a" / "stage_a_checkpoint.pt"
    model, _, _, _ = load_stage_a_model_from_checkpoint(checkpoint, device=args.device)
    traj_cfg = _build_trajectory_config(stage_a_cfg)
    fit_dir = _plot_dir(run_dir, plot_root, "fit")
    names = plot_trajectory_residual_analysis(
        arcs=arcs,
        model=model,
        device=torch.device(args.device),
        out_dir=fit_dir,
        traj_cfg=traj_cfg,
    )
    return [str(fit_dir / name) for name in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate selected Stage A plots from an existing outputs/latest run."
    )
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/latest"), help="Existing run directory to read.")
    parser.add_argument(
        "--plot-root",
        type=Path,
        default=None,
        help="Optional output plot root. Defaults to <run-dir>/plots and overwrites those PNGs.",
    )
    parser.add_argument("--dpi", type=int, default=None, help="PNG DPI. Defaults to resolved_config plotting.dpi or 600.")
    parser.add_argument("--save-pdf", action="store_true", help="Also save PDF copies beside PNGs.")
    parser.add_argument(
        "--skip-trajectory-rmse",
        action="store_true",
        help="Skip trajectory_rmse_by_phase.png; avoids reloading TLEs and rebuilding diagnostic arcs.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for checkpoint reload used by trajectory RMSE plots.")
    parser.add_argument("--max-sats", type=int, default=None, help="Override number of TLE satellite files used for arc rebuild.")
    parser.add_argument("--tle-workers", type=int, default=1, help="TLE loading workers for trajectory RMSE rebuild.")
    parser.add_argument("--tle-chunk-size", type=int, default=None, help="Files per TLE worker chunk.")
    parser.add_argument("--tle-progress-files", type=int, default=None, help="TLE progress print cadence in files.")
    parser.add_argument("--verbose", action="store_true", help="Print more progress logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    apply_plot_style()

    run_dir = _resolve_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    plot_root = _resolve_path(args.plot_root) if args.plot_root is not None else None

    config_path = run_dir / "resolved_config.json"
    config = _load_json(config_path) if config_path.exists() else {}
    plotting_cfg = dict(config.get("plotting", {}))
    dpi = int(args.dpi if args.dpi is not None else plotting_cfg.get("dpi", 600))
    plot_cfg = PlotConfig(enabled=True, save_pdf=bool(args.save_pdf), dpi=dpi)

    written = regenerate_artifact_plots(run_dir, plot_root, plot_cfg)
    if not args.skip_trajectory_rmse:
        _append_paths(regenerate_trajectory_rmse(run_dir, plot_root, config, args), written)

    print("Regenerated plots:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()