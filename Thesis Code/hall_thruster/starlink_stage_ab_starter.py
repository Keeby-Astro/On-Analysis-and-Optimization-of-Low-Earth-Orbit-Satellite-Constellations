from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.func import jacfwd, jacrev, vmap
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Accelerator defaults
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

if MATPLOTLIB_AVAILABLE:
    # UPDATE FIGURE SETTINGS & DEFINE CUSTOM PALETTE
    plt.rcParams.update({'figure.figsize': (10.0, 7.5),
                         'xtick.direction': 'in', 'xtick.labelsize': 14, 'xtick.major.size': 3,
                         'xtick.major.width': 0.5, 'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5,
                         'xtick.minor.visible': True, 'xtick.top': True,
                         'ytick.direction': 'in', 'ytick.labelsize': 14, 'ytick.major.size': 3,
                         'ytick.major.width': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
                         'ytick.minor.visible': True, 'ytick.right': True,
                         'axes.linewidth': 0.5, 'grid.linewidth': 0.5, 'lines.linewidth': 1.0,
                         'legend.fontsize': 14, 'legend.frameon': False,
                         'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'dejavuserif',
                         'font.size': 12, 'axes.labelsize': 16, 'axes.titlesize': 18,
                         'axes.grid': True, 'grid.linestyle': '--', 'grid.color': '0.5',
                         'lines.markersize': 8, 'axes.spines.top': True, 'axes.spines.right': True})

    # Define the custom 20-color palette (darkened colors)
    colors = ['#15528e', '#b25800', '#1e701e', '#951c1c', '#673284',
              '#623c34', '#9e5387', '#585858', '#848417', '#108590',
              '#798ba2', '#b28254', '#6a9c60', '#b26a68', '#8a7b94',
              '#896d67', '#ac7f93', '#8b8b8b', '#999962', '#6f989f']

# Project physics modules
from reduced_dynamics import (
    MU_EARTH_KM3_S2 as _RD_MU,
    R_EARTH_KM as _RD_RE,
    J2_EARTH as _RD_J2,
    G0_M_S2 as _RD_G0,
    TWOPI as _RD_TWOPI,
    K_BOLTZMANN_J_K as _RD_KB,
    KR_MASS_KG as _RD_MKRKG,
    RUN_SCHEMA_VERSION as _RD_SCHEMA,
    COLLOCATION_TAU_POINTS as _RD_COLLTAU,
    wrap_to_pi as _rd_wrap_to_pi,
    wrap_to_2pi as _rd_wrap_to_2pi,
    wrap_angle as _rd_wrap_angle,
    angle_residual as _rd_angle_residual,
    deg2rad as _rd_deg2rad,
    mean_motion_rad_s as _rd_mean_motion_rad_s,
    raan_rate_j2_rad_s as _rd_raan_rate_j2_rad_s,
    omega_dot_j2_rad_s,
    M_dot_j2_rad_s,
    lambda_dot_j2_rad_s,
    raan_rate_j2_torch,
    omega_dot_j2_torch,
    M_dot_j2_torch,
    lambda_dot_j2_torch,
)
from hall_beam_relations import (
    C_T_KR,
    C_I_KR,
    thrust_kr_mN as _hb_thrust_kr_mN,
    isp_kr_s as _hb_isp_kr_s,
    beam_exhaust_velocity_m_s,
    electrical_power_W,
)
from chemistry_models import (
    ClosureMode,
    ChemistryResult,
    legacy_surrogate_chemistry,
    compute_chemistry,
)
from arc_building import (
    ArcRecord,
    ArcBuildConfig,
    ArcDataset,
    build_arcs_from_tles_and_intervals,
    collate_arcs as collate_arcs_fn,
    save_arcs_to_parquet,
    load_arcs_from_parquet,
)
from trajectory_matching import (
    TrajectoryConfig,
    compute_accel_net as traj_compute_accel_net,
    trajectory_sma_forward,
    trajectory_forward_and_loss,
    ussa76_density,
    ussa76_drag_accel_kmps2,
)
from trajectory_diagnostics import (
    trajectory_validation_report,
    plot_trajectory_fits,
    plot_trajectory_residual_analysis,
    plot_trajectory_training_history,
    plot_trajectory_parameter_evolution,
    plot_trajectory_per_satellite_rmse,
)


# -----------------------------------------------------------------------------
# Dynamic import helper for reusing existing project modules without requiring
# the exact same filenames in every environment.
# -----------------------------------------------------------------------------
def _import_local_module(module_name: str, candidate_filenames: Sequence[str]):
    try:
        return __import__(module_name)
    except Exception:
        pass

    import importlib.util
    import sys

    this_dir = Path(__file__).resolve().parent
    for name in candidate_filenames:
        candidate = this_dir / name
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    raise ImportError(f"Could not import {module_name}. Tried: {candidate_filenames}")


_LOAD_TLE = _import_local_module(
    "load_all_tle_data",
    ["load_all_tle_data.py", "load_all_tle_data(1).py"],
)
load_all_tle_data = _LOAD_TLE.load_all_tle_data


# -----------------------------------------------------------------------------
# Constants  (re-exported from reduced_dynamics for backward compatibility)
# -----------------------------------------------------------------------------
MU_EARTH_KM3_S2 = _RD_MU
R_EARTH_KM = _RD_RE
J2_EARTH = _RD_J2
G0_M_S2 = _RD_G0
TWOPI = _RD_TWOPI
K_BOLTZMANN_J_K = _RD_KB
KR_MASS_KG = _RD_MKRKG
RUN_SCHEMA_VERSION = "2026.04-stage-ab-r3"  # bumped: J2-secular λ, shell reparam, closure_mode
COLLOCATION_TAU_POINTS = _RD_COLLTAU


# -----------------------------------------------------------------------------
# Project configuration (CLI > env > config file > defaults)
# -----------------------------------------------------------------------------
ENV_PREFIX = "STAGE_AB_"


@dataclass
class PathsConfig:
    tle_dir: str = "starlink_backup"
    labels_csv: str = "full_exports"
    segments_csv: str = "segments/segments.csv"
    output_root: str = "outputs"
    config_file: str = ""


@dataclass
class SmoothingConfig:
    enabled: bool = True
    rolling_window: int = 5
    mad_z_thresh: float = 6.0
    trim_edge_points: int = 0
    interpolate_common_time_base: bool = False
    progress_every_sats: int = 250
    auto_disable_large_run: bool = False
    auto_disable_row_threshold: int = 50_000_000


@dataclass
class SplitConfig:
    by_satellite: bool = True
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42


@dataclass
class ForceModelConfig:
    mode: str = "full_reduced_default"
    use_j2: bool = True
    use_drag: bool = True
    use_power_cap: bool = True
    use_timing_bias: bool = True


@dataclass
class StageAConfig:
    enabled: bool = True
    resume: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3.0e-2
    lr_schedule: str = "cosine"  # "cosine" | "none"
    lr_min_factor: float = 0.01  # min LR = lr * lr_min_factor
    early_stopping_patience: int = 10  # stop if no improvement for N epochs; 0 = disabled
    weight_decay: float = 1.0e-5
    seed: int = 42
    compile_model: bool = False
    dry_mass_kg: float = 150.0
    mass_init_kg: float = 250.0
    isp_init_s: float = 1500.0
    eta_init: float = 0.48
    thrust_init_N: float = 0.070
    drag_init_kmps2: float = 1.0e-10
    lambda_a: float = 1.0
    lambda_da: float = 2.0
    lambda_raan: float = 5.0
    lambda_lam: float = 0.1  # cautious: enabled now that J2 secular rates are correct
    lambda_a_end: float = 0.25
    lambda_rate: float = 1.0
    lambda_prior: float = 1.0
    robust_loss: str = "mse"
    huber_delta: float = 1.0
    thermal_duty_cap: float = 0.85
    max_grad_norm: float = 1.0
    use_secondary_obs: bool = False
    enable_satellite_random_effects: bool = True
    synthetic_noise_std_a_km: float = 0.5
    synthetic_noise_std_angle_rad: float = 2.0e-3
    robust_student_t_dof: float = 4.0
    robust_student_t_scale: float = 1.0
    phase_loss_weight_power: float = 0.50
    obs_weight_a: float = 1.0
    obs_weight_da: float = 1.0
    obs_weight_raan: float = 1.0
    obs_weight_lam: float = 1.0
    obs_weight_rate: float = 1.0
    obs_weight_da_rate: float = 1.0
    obs_weight_draan_rate: float = 1.0
    obs_weight_dlam_rate: float = 0.0  # disabled until lambda wrapping fix verified
    obs_weight_collocation: float = 0.5
    obs_scale_a_km: float = 3.0
    obs_scale_angle_rad: float = 0.02
    obs_scale_da_rate_km_day: float = 0.5
    obs_scale_angle_rate_rad_day: float = 5.0e-3
    segment_tolerance_hours: float = 1.0
    collocation_enabled: bool = True
    collocation_taus: Tuple[float, float, float] = COLLOCATION_TAU_POINTS
    collocation_tolerance_hours: float = 6.0
    use_piecewise_thrust_schedule: bool = True
    piecewise_midpoint_scale_init: float = 1.0
    lambda_hall: float = 1.0e-3
    lambda_chemistry: float = 1.0e-4
    lambda_feasibility: float = 1.0
    curriculum_kinematics_epochs: int = 5
    curriculum_collocation_epochs: int = 20
    curriculum_physics_ramp_epochs: int = 15
    duration_weight_enabled: bool = True
    duration_weight_power: float = 0.25
    vd_init_V: float = 320.0
    vc_init_V: float = 25.0
    vb_init_V: float = 295.0
    ib_init_A: float = 3.0
    eta_b_init: float = 0.85
    eta_v_init: float = 0.90
    eta_m_init: float = 0.75
    eta_o_init: float = 0.82
    gamma_init: float = 1.0
    nu_a_init: float = 0.25
    mdot_a_init_kg_s: float = 4.5e-6
    mdot_c_init_kg_s: float = 8.0e-7
    pressure_base_pa: float = 4.5e-3
    pressure_gain_pa_per_kg_s: float = 350.0
    neutral_temp_K: float = 900.0
    electron_temp_base_eV: float = 4.0
    electron_temp_gain_per_V: float = 0.015
    electron_temp_gain_nua: float = 8.0
    ionization_length_m: float = 0.03
    ionization_ratio_min: float = 0.25
    ionization_ratio_max: float = 3.0
    # New r3 fields──
    closure_mode: str = "legacy_surrogate"  # "legacy_surrogate" | "tabulated"
    shell_drag_comp_fraction_init: float = 1.0  # ≈1 means drag-compensated; <1 net decay; >1 slight raise
    # Trajectory-matching r4 fields ────────────────────────────────────
    fit_mode: str = "trajectory_matching"  # "segment_endpoint" | "trajectory_matching"
    intervals_csv: str = ""  # path to maneuver phase intervals CSV
    max_arc_obs: int = 200
    min_arc_obs: int = 5
    max_subarc_days: float = 30.0
    lambda_continuity: float = 0.1
    lambda_path: float = 5.0
    lambda_endpoint_a: float = 1.0
    lambda_endpoint_raan: float = 0.0
    lambda_endpoint_lam: float = 0.0
    arc_weight_mode: str = "sqrt_inv_n_obs"  # "uniform" | "inv_n_obs" | "sqrt_inv_n_obs"
    # Atmosphere-based drag ────────────────────────────────────────────
    use_atmosphere_drag: bool = True  # Replace constant drag with USSA76 altitude-dependent model
    inv_ballistic_coeff: float = 0.0334  # Cd·A/(2·m) [m²/kg] for Starlink flat plate
    # Non-linear propagation ───────────────────────────────────────────
    nonlinear_propagation: bool = True   # RK4 ODE integration with altitude-dependent drag
    rk4_step_hours: float = 12.0         # RK4 integration step size [hours]


@dataclass
class StageBConfig:
    enabled: bool = True
    method: str = "snpe"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    density_estimator: str = "maf"
    num_simulations: int = 3000
    max_segments: int = 128
    max_phase_parameters: int = 6
    num_posterior_samples: int = 2000
    ppc_samples: int = 256
    normalize_observation: bool = True
    include_initial_conditions: bool = True
    include_phase_context: bool = True
    include_rate_features: bool = True
    normalization_eps: float = 1.0e-6
    run_sbc: bool = False
    sbc_draws: int = 64
    sbc_posterior_samples: int = 256
    calibration_subset_segments: int = 128
    effective_mode: bool = True  # reduced Tier-1: mass + per-phase thrust/drag only
    anchor_from_stage_a: bool = True  # use fitted Stage A summary instead of init defaults
    mixed_precision: bool = True  # Use torch.amp autocast for faster GPU simulations


@dataclass
class ValidationConfig:
    enabled: bool = True
    device: str = "cpu"
    run_loso: bool = False
    loso_max_satellites: int = 5
    run_timing_sensitivity: bool = True
    timing_shift_hours: float = 6.0
    timing_shift_samples: int = 8
    run_force_model_ablations: bool = True
    run_synthetic_recovery: bool = True
    synthetic_refit_epochs: int = 300


@dataclass
class PlotConfig:
    enabled: bool = True
    save_pdf: bool = False
    dpi: int = 600
    data_quality: bool = True
    training: bool = True
    fit: bool = True
    parameters: bool = True
    stage_b: bool = True
    sensitivity: bool = True
    ablations: bool = True
    synthetic: bool = True


@dataclass
class RunControlConfig:
    mode: str = "default"
    max_sats: Optional[int] = None
    rebuild_segments: bool = False
    skip_stage_b: bool = False
    tle_workers: Optional[int] = None
    tle_chunk_size: int = 128
    tle_progress_files: int = 100


@dataclass
class ProjectConfig:
    paths: PathsConfig
    smoothing: SmoothingConfig
    split: SplitConfig
    force_model: ForceModelConfig
    stage_a: StageAConfig
    stage_b: StageBConfig
    validation: ValidationConfig
    plotting: PlotConfig
    run: RunControlConfig


def _default_project_config() -> ProjectConfig:
    return ProjectConfig(
        paths=PathsConfig(),
        smoothing=SmoothingConfig(),
        split=SplitConfig(),
        force_model=ForceModelConfig(),
        stage_a=StageAConfig(),
        stage_b=StageBConfig(),
        validation=ValidationConfig(),
        plotting=PlotConfig(),
        run=RunControlConfig(),
    )


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _coerce_env_value(raw: str) -> Any:
    v = raw.strip()
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in v or "e" in low:
            return float(v)
        return int(v)
    except Exception:
        return v


def _load_env_overrides(prefix: str = ENV_PREFIX) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        key = k[len(prefix):].lower().replace("__", ".")
        parts = [p for p in key.split(".") if p]
        if not parts:
            continue
        ref = out
        for p in parts[:-1]:
            ref = ref.setdefault(p, {})
        ref[parts[-1]] = _coerce_env_value(v)
    return out


def _load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config file must contain a top-level object: {path}")
    return obj


def _dict_to_project_config(cfg_dict: Dict[str, Any]) -> ProjectConfig:
    d = dict(cfg_dict)
    return ProjectConfig(
        paths=PathsConfig(**dict(d.get("paths", {}))),
        smoothing=SmoothingConfig(**dict(d.get("smoothing", {}))),
        split=SplitConfig(**dict(d.get("split", {}))),
        force_model=ForceModelConfig(**dict(d.get("force_model", {}))),
        stage_a=StageAConfig(**dict(d.get("stage_a", {}))),
        stage_b=StageBConfig(**dict(d.get("stage_b", {}))),
        validation=ValidationConfig(**dict(d.get("validation", {}))),
        plotting=PlotConfig(**dict(d.get("plotting", {}))),
        run=RunControlConfig(**dict(d.get("run", {}))),
    )


def resolve_project_config(cli_overrides: Optional[Dict[str, Any]] = None) -> ProjectConfig:
    default_cfg = _default_project_config()
    merged = asdict(default_cfg)

    script_dir = Path(__file__).resolve().parent
    default_config_file = script_dir / "starlink_stage_ab_config.json"
    file_cfg = _load_json_config(default_config_file)
    _deep_update(merged, file_cfg)

    env_cfg = _load_env_overrides()
    _deep_update(merged, env_cfg)

    if cli_overrides:
        _deep_update(merged, cli_overrides)

    cfg = _dict_to_project_config(merged)
    if cfg.paths.config_file:
        explicit_path = Path(cfg.paths.config_file)
        explicit_cfg = _load_json_config(explicit_path)
        merged2 = asdict(cfg)
        _deep_update(merged2, explicit_cfg)
        if cli_overrides:
            _deep_update(merged2, cli_overrides)
        cfg = _dict_to_project_config(merged2)
    return cfg


@dataclass
class RunPaths:
    root: Path
    latest: Path
    checkpoints: Path
    plots: Path
    plots_data_quality: Path
    plots_train: Path
    plots_fit: Path
    plots_parameters: Path
    plots_stage_b: Path
    plots_sensitivity: Path
    plots_ablations: Path
    plots_synthetic: Path
    tables: Path
    reports: Path
    logs: Path


def prepare_run_paths(output_root: Path) -> RunPaths:
    root = output_root.resolve()
    latest = root / "latest"
    checkpoints = latest / "checkpoints"
    plots = latest / "plots"
    tables = latest / "tables"
    reports = latest / "reports"
    logs = latest / "logs"

    if latest.exists():
        shutil.rmtree(latest)

    for p in [
        checkpoints,
        plots / "data_quality",
        plots / "train",
        plots / "fit",
        plots / "parameters",
        plots / "stage_b",
        plots / "sensitivity",
        plots / "ablations",
        plots / "synthetic",
        tables,
        reports,
        logs,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        root=root,
        latest=latest,
        checkpoints=checkpoints,
        plots=plots,
        plots_data_quality=plots / "data_quality",
        plots_train=plots / "train",
        plots_fit=plots / "fit",
        plots_parameters=plots / "parameters",
        plots_stage_b=plots / "stage_b",
        plots_sensitivity=plots / "sensitivity",
        plots_ablations=plots / "ablations",
        plots_synthetic=plots / "synthetic",
        tables=tables,
        reports=reports,
        logs=logs,
    )


def setup_logging(log_file: Path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_default_tle_dir(raw_tle_dir: str) -> Path:
    cwd = Path.cwd()
    candidates = [
        cwd / raw_tle_dir,
        cwd / "starlink_backup",
        cwd / "starlink_decay",
        cwd / "extra" / "TLE",
    ]
    pick = _first_existing(candidates)
    if pick is None:
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"No default TLE directory found. Tried:\n{tried}")
    return pick


def resolve_default_labels_path(raw_labels: str) -> Path:
    cwd = Path.cwd()
    as_path = cwd / raw_labels
    candidates = [
        as_path,
        cwd / "full_exports",
        cwd / "full_exports" / "maneuver_phase_intervals_gen1_full.csv",
        cwd / "full_exports" / "maneuver_phase_labels_gen1_full.csv",
    ]
    pick = _first_existing(candidates)
    if pick is None:
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"No default labels source found. Tried:\n{tried}")
    return pick


def resolve_default_segments_path(raw_segments: str, run_paths: RunPaths) -> Path:
    cwd = Path.cwd()
    candidate = cwd / raw_segments
    if candidate.exists():
        return candidate
    return run_paths.latest / "segments" / "segments.csv"


def save_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


# -----------------------------------------------------------------------------
# Utility functions  (thin wrappers delegating to reduced_dynamics)
# -----------------------------------------------------------------------------
wrap_to_pi = _rd_wrap_to_pi
wrap_to_2pi = _rd_wrap_to_2pi
wrap_angle = _rd_wrap_angle
angle_residual = _rd_angle_residual
deg2rad = _rd_deg2rad
mean_motion_rad_s = _rd_mean_motion_rad_s
raan_rate_j2_rad_s = _rd_raan_rate_j2_rad_s


# Canonical phase state names expected in label data.
KNOWN_PHASE_STATES = (
    "insertion_or_orbit_raise",
    "operational_shell",
    "disposal_lowering",
)

PHASE_DISPLAY_NAMES = {
    "operational_shell": "Operational Shell",
    "disposal_lowering": "Disposal Lowering",
    "insertion_or_orbit_raise": "Insertion / Orbit Raise",
}


def phase_sign_from_name(phase_name: str) -> float:
    """Return thrust direction sign for a manoeuvre phase.

    +1  = prograde (orbit raise, station-keeping drag make-up)
    -1  = retrograde (deorbit / disposal lowering)
     0  = no net secular thrust (operational shell, pure drag)
    """
    p = str(phase_name).strip().lower()
    retro_keywords = ["deorbit", "lower", "disposal", "retro", "drop", "descent"]
    if any(k in p for k in retro_keywords):
        return -1.0
    stationkeep_keywords = ["operational", "shell", "station", "maintain"]
    if any(k in p for k in stationkeep_keywords):
        return 0.0  # drag make-up only; learned drag term drives SMA change
    return 1.0


def find_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def canonical_sat_id(s: str) -> str:
    x = str(s).strip().lower()
    if x.endswith(".txt"):
        x = x[:-4]
    if x.endswith("_decay"):
        x = x[:-6]
    return x


def build_mean_longitude_rad(df: pd.DataFrame) -> np.ndarray:
    return wrap_to_2pi(
        deg2rad(df["raan"].to_numpy())
        + deg2rad(df["aop"].to_numpy())
        + deg2rad(df["mean_anomaly"].to_numpy())
    )


def nearest_row_by_time(group: pd.DataFrame, target_ts: pd.Timestamp, tolerance_s: float) -> Optional[pd.Series]:
    dt_s = np.abs((group["timestamp"] - target_ts).dt.total_seconds().to_numpy())
    if dt_s.size == 0:
        return None
    idx = int(np.argmin(dt_s))
    if dt_s[idx] > tolerance_s:
        return None
    return group.iloc[idx]


def nearest_index_by_time_ns(time_ns_sorted: np.ndarray, target_ns: int, tolerance_ns: int) -> Optional[int]:
    if time_ns_sorted.size == 0:
        return None

    idx = int(np.searchsorted(time_ns_sorted, int(target_ns)))
    candidates: List[int] = []
    if idx < int(time_ns_sorted.size):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None

    best = min(candidates, key=lambda i: abs(int(time_ns_sorted[i]) - int(target_ns)))
    if abs(int(time_ns_sorted[best]) - int(target_ns)) > int(tolerance_ns):
        return None
    return int(best)


def robust_mad_zscore(x: pd.Series) -> pd.Series:
    med = float(np.nanmedian(x.to_numpy(dtype=np.float64)))
    abs_dev = np.abs(x.to_numpy(dtype=np.float64) - med)
    mad = float(np.nanmedian(abs_dev))
    if mad <= 1.0e-12:
        return pd.Series(np.zeros(len(x), dtype=np.float64), index=x.index)
    return pd.Series(0.6745 * (x - med) / mad, index=x.index)


def preprocess_tle_dataframe(tle_df: pd.DataFrame, cfg: SmoothingConfig) -> pd.DataFrame:
    t0 = time.perf_counter()
    tle = tle_df.copy()
    tle["timestamp"] = pd.to_datetime(tle["timestamp"], utc=True).dt.tz_convert(None)
    tle["sat_id"] = tle["sat_id"].map(canonical_sat_id)
    tle = tle.sort_values(["sat_id", "timestamp"]).reset_index(drop=True)

    if bool(cfg.auto_disable_large_run) and len(tle) >= int(cfg.auto_disable_row_threshold):
        _log_info_or_print(
            f"TLE preprocessing fast-path: rows={len(tle)} exceeds threshold={int(cfg.auto_disable_row_threshold)}, smoothing disabled"
        )
        tle["sma_proc"] = tle["sma"]
        tle["raan_proc"] = tle["raan"]
        tle["mean_anomaly_proc"] = tle["mean_anomaly"]
        tle["tle_outlier_flag"] = False
        _log_info_or_print(f"TLE preprocessing complete (fast-path) in {time.perf_counter() - t0:.1f}s")
        return tle

    if not cfg.enabled:
        _log_info_or_print("TLE preprocessing disabled by config; using raw orbital columns")
        tle["sma_proc"] = tle["sma"]
        tle["raan_proc"] = tle["raan"]
        tle["mean_anomaly_proc"] = tle["mean_anomaly"]
        tle["tle_outlier_flag"] = False
        _log_info_or_print(f"TLE preprocessing complete (disabled mode) in {time.perf_counter() - t0:.1f}s")
        return tle

    proc_list: List[pd.DataFrame] = []
    w = max(1, int(cfg.rolling_window))
    zthr = float(cfg.mad_z_thresh)
    trim_n = max(0, int(cfg.trim_edge_points))

    def _circular_rolling_median(series: pd.Series, window: int) -> np.ndarray:
        """Fast rolling median for angular quantities via numpy unwrap."""
        vals = series.to_numpy(dtype=np.float64)
        # np.unwrap removes 2-pi jumps so standard rolling median works.
        unwrapped = np.unwrap(vals)
        med = pd.Series(unwrapped).rolling(window=window, center=True, min_periods=1).median().to_numpy()
        # Re-wrap to [-pi, pi].
        return np.arctan2(np.sin(med), np.cos(med))

    total_sats = int(tle["sat_id"].nunique())
    total_rows_processed = 0
    for sat_idx, (_, g) in enumerate(tle.groupby("sat_id", sort=False), start=1):
        gg = g.copy()
        gg["sma_smooth"] = gg["sma"].rolling(window=w, center=True, min_periods=1).median()
        gg["raan_smooth"] = _circular_rolling_median(gg["raan"], w)
        gg["ma_smooth"] = _circular_rolling_median(gg["mean_anomaly"], w)

        z = robust_mad_zscore(gg["sma"] - gg["sma_smooth"])
        gg["tle_outlier_flag"] = z.abs() > zthr

        # Replace outliers with smooth estimate for reduced-order fitting.
        gg["sma_proc"] = np.where(gg["tle_outlier_flag"], gg["sma_smooth"], gg["sma"])
        gg["raan_proc"] = np.where(gg["tle_outlier_flag"], gg["raan_smooth"], gg["raan"])
        gg["mean_anomaly_proc"] = np.where(gg["tle_outlier_flag"], gg["ma_smooth"], gg["mean_anomaly"])

        if trim_n > 0 and len(gg) > 2 * trim_n:
            gg = gg.iloc[trim_n:-trim_n].copy()

        proc_list.append(gg)
        total_rows_processed += int(len(g))
        if int(cfg.progress_every_sats) > 0 and sat_idx % int(cfg.progress_every_sats) == 0:
            elapsed = time.perf_counter() - t0
            _log_info_or_print(
                f"TLE preprocessing progress: sats={sat_idx}/{total_sats} rows_seen={total_rows_processed} elapsed={elapsed:.1f}s"
            )

    out = pd.concat(proc_list, axis=0).sort_values(["sat_id", "timestamp"]).reset_index(drop=True)
    _log_info_or_print(f"TLE preprocessing complete in {time.perf_counter() - t0:.1f}s")
    return out


def enrich_segment_dataframe(seg_df: pd.DataFrame) -> pd.DataFrame:
    out = seg_df.copy()
    dt_days = np.clip(out["dt_s"].to_numpy(dtype=np.float64) / 86400.0, 1.0e-6, np.inf)
    out["da_rate_km_day"] = out["da_obs_km"].to_numpy(dtype=np.float64) / dt_days
    out["draan_rate_rad_day"] = out["draan_obs_rad"].to_numpy(dtype=np.float64) / dt_days
    out["dlam_rate_rad_day"] = out["dlam_obs_rad"].to_numpy(dtype=np.float64) / dt_days
    out["segment_quality_flag"] = (
        np.isfinite(out["da_rate_km_day"].to_numpy())
        & np.isfinite(out["draan_rate_rad_day"].to_numpy())
        & np.isfinite(out["dlam_rate_rad_day"].to_numpy())
    )
    return out


CORE_STAGE_A_COLS = [
    "a0_km",
    "a1_km",
    "e0",
    "inc0_rad",
    "raan0_rad",
    "lam0_rad",
    "da_obs_km",
    "draan_obs_rad",
    "dlam_obs_rad",
    "dt_s",
    "da_rate_km_day",
    "draan_rate_rad_day",
    "dlam_rate_rad_day",
]


def sanitize_segments_for_training(seg_df: pd.DataFrame) -> pd.DataFrame:
    out = seg_df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    missing = [col for col in CORE_STAGE_A_COLS if col not in out.columns]
    if missing:
        raise KeyError(f"Segments dataframe is missing required Stage A columns: {missing}")

    before_rows = len(out)
    finite_core = np.isfinite(out[CORE_STAGE_A_COLS].to_numpy(dtype=np.float64)).all(axis=1)
    out = out.loc[finite_core].copy()
    dropped_rows = before_rows - len(out)
    _log_info_or_print(f"Dropped {dropped_rows} non-finite segment rows before Stage A/B processing.")

    for tau in COLLOCATION_TAU_POINTS:
        tau_label = int(round(100.0 * float(tau)))
        a_col = f"a_tau{tau_label}_km"
        raan_col = f"raan_tau{tau_label}_rad"
        lam_col = f"lam_tau{tau_label}_rad"
        mask_col = f"has_tau{tau_label}"
        if a_col in out.columns and raan_col in out.columns and lam_col in out.columns:
            finite_tau = np.isfinite(
                out[[a_col, raan_col, lam_col]].to_numpy(dtype=np.float64)
            ).all(axis=1).astype(np.float32)
            if mask_col in out.columns:
                mask_values = out[mask_col].to_numpy(dtype=np.float32) * finite_tau
            else:
                mask_values = finite_tau
            out[mask_col] = np.nan_to_num(mask_values, nan=0.0, posinf=0.0, neginf=0.0)
            mask_off = out[mask_col].to_numpy(dtype=np.float32) <= 0.5
            # Zero masked collocation targets so downstream losses never touch stale NaN values.
            out.loc[mask_off, [a_col, raan_col, lam_col]] = 0.0

    if out.empty:
        raise RuntimeError("No finite maneuver segments remain after Stage A/B sanitization.")
    return out.reset_index(drop=True)


def split_segments_dataframe(seg_df: pd.DataFrame, split_cfg: SplitConfig) -> Dict[str, pd.DataFrame]:
    df = seg_df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return {"train": df.copy(), "val": df.copy(), "test": df.copy()}

    rng = np.random.default_rng(int(split_cfg.seed))
    train_frac = float(split_cfg.train_fraction)
    val_frac = float(split_cfg.val_fraction)
    test_frac = float(split_cfg.test_fraction)
    denom = max(train_frac + val_frac + test_frac, 1.0e-9)
    train_frac /= denom
    val_frac /= denom

    if bool(split_cfg.by_satellite) and "sat_id" in df.columns:
        sats = df["sat_id"].astype(str).unique().tolist()
        rng.shuffle(sats)
        n_train_sat = int(round(len(sats) * train_frac))
        n_val_sat = int(round(len(sats) * val_frac))
        train_sats = set(sats[:n_train_sat])
        val_sats = set(sats[n_train_sat:n_train_sat + n_val_sat])
        test_sats = set(sats[n_train_sat + n_val_sat:])
        train_df = df[df["sat_id"].astype(str).isin(train_sats)].copy()
        val_df = df[df["sat_id"].astype(str).isin(val_sats)].copy()
        test_df = df[df["sat_id"].astype(str).isin(test_sats)].copy()
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        test_df = df.iloc[test_idx].copy()

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def _compute_fit_metrics_from_predictions(pred_df: pd.DataFrame) -> Dict[str, float]:
    if pred_df.empty:
        return {
            "num_segments": 0,
            "rmse_a1_km": float("nan"),
            "rmse_da_km": float("nan"),
            "rmse_draan_rad": float("nan"),
            "rmse_dlambda_rad": float("nan"),
        }
    return {
        "num_segments": int(len(pred_df)),
        "rmse_a1_km": float(np.sqrt(np.nanmean(np.square(pred_df["a1_resid_km"].to_numpy(dtype=np.float64))))),
        "rmse_da_km": float(np.sqrt(np.nanmean(np.square(pred_df["da_resid_km"].to_numpy(dtype=np.float64))))),
        "rmse_draan_rad": float(np.sqrt(np.nanmean(np.square(pred_df["draan_resid_rad"].to_numpy(dtype=np.float64))))),
        "rmse_dlambda_rad": float(np.sqrt(np.nanmean(np.square(pred_df["dlam_resid_rad"].to_numpy(dtype=np.float64))))),
    }


def summarize_fit_metrics_by_phase(pred_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if pred_df.empty or "phase" not in pred_df.columns:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for phase, g in pred_df.groupby("phase"):
        out[str(phase)] = _compute_fit_metrics_from_predictions(g)
    return out


def inv_softplus_scalar(x: float) -> float:
    x = float(x)
    if x <= 0.0:
        return -50.0
    if x > 20.0:
        # softplus(z) ~= z in this regime; avoid overflow in expm1(x).
        return x
    return math.log(math.expm1(x))


def resolve_labels_csv_path(labels_csv_path: Path) -> Path:
    labels_csv_path = Path(labels_csv_path)
    if labels_csv_path.is_file():
        return labels_csv_path
    if not labels_csv_path.exists():
        raise FileNotFoundError(f"labels path does not exist: {labels_csv_path}")
    if not labels_csv_path.is_dir():
        raise ValueError(f"labels path must be a CSV file or a directory: {labels_csv_path}")

    csv_candidates = sorted(labels_csv_path.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV files found under labels directory: {labels_csv_path}")

    # Prefer interval-style labels when a directory is provided.
    preferred_patterns = [
        "*maneuver_phase_intervals*.csv",
        "*phase_intervals*.csv",
        "*interval*.csv",
        "*labels*.csv",
    ]
    for pattern in preferred_patterns:
        matches = sorted(labels_csv_path.glob(pattern))
        if matches:
            return matches[0]
    return csv_candidates[0]


def select_tle_files(tle_dir: Path, max_sats: Optional[int]) -> Optional[List[str]]:
    if max_sats is None:
        return None
    max_sats = int(max_sats)
    if max_sats <= 0:
        return None
    if not tle_dir.exists() or not tle_dir.is_dir():
        raise FileNotFoundError(f"TLE directory does not exist or is not a directory: {tle_dir}")

    tle_files = sorted([p.name for p in tle_dir.glob("*.txt")])
    if not tle_files:
        raise FileNotFoundError(f"No .txt TLE files found in {tle_dir}")
    return tle_files[:max_sats]


def resolve_output_csv_path(out_csv_path: Path) -> Path:
    out_csv_path = Path(out_csv_path)
    if out_csv_path.suffix.lower() == ".csv":
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        return out_csv_path

    if out_csv_path.exists() and out_csv_path.is_file():
        raise FileExistsError(
            f"Output path exists as a file: {out_csv_path}. "
            "If you want a folder output, rename/remove this file or pass --out-csv <folder>/segments.csv."
        )

    out_csv_path.mkdir(parents=True, exist_ok=True)
    return out_csv_path / "segments.csv"


def _log_info_or_print(message: str):
    if logging.getLogger().handlers:
        logging.info(message)
    else:
        print(message)


def _chunk_list(items: Sequence[str], chunk_size: int) -> List[List[str]]:
    chunk_size = max(1, int(chunk_size))
    return [list(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]


def _resolve_tle_workers(requested_workers: Optional[int], num_files: int) -> int:
    cpu_count = max(1, int(os.cpu_count() or 1))
    if requested_workers is None:
        # Auto-mode: avoid process overhead for tiny batches, otherwise use up to 12 cores.
        if int(num_files) < 64:
            return 1
        return max(1, min(12, cpu_count - 1))

    workers = int(requested_workers)
    if workers <= 1:
        return 1
    return max(1, min(workers, cpu_count))


def _load_tle_chunk_worker(tle_dir: str, chunk_files: Sequence[str], derived_cols: Sequence[str]) -> pd.DataFrame:
    df_chunk, _ = load_all_tle_data([tle_dir], only_files=list(chunk_files), derived=set(derived_cols))
    return df_chunk


def load_tle_data_with_progress(
    tle_dir: Path,
    only_files: Optional[Sequence[str]],
    derived_cols: Sequence[str],
    workers: Optional[int],
    chunk_size: int,
    progress_every_files: int,
) -> Tuple[pd.DataFrame, List[str]]:
    tle_dir = Path(tle_dir)
    if only_files is None:
        file_list = sorted([p.name for p in tle_dir.glob("*.txt")])
    else:
        file_list = sorted([str(x) for x in only_files])

    if not file_list:
        raise FileNotFoundError(f"No TLE .txt files selected under {tle_dir}")

    t0 = time.perf_counter()
    resolved_workers = _resolve_tle_workers(workers, len(file_list))
    _log_info_or_print(
        f"TLE load plan: files={len(file_list)} workers={resolved_workers} chunk_size={int(chunk_size)}"
    )

    if resolved_workers <= 1:
        df, filenames = load_all_tle_data([str(tle_dir)], only_files=file_list, derived=set(derived_cols))
        elapsed = time.perf_counter() - t0
        rows_per_s = (len(df) / elapsed) if elapsed > 0 else float("inf")
        _log_info_or_print(
            f"TLE load complete (sequential): rows={len(df)} files={len(file_list)} elapsed={elapsed:.1f}s rows_per_s={rows_per_s:.1f}"
        )
        return df, filenames

    chunks = _chunk_list(file_list, int(chunk_size))
    total_chunks = len(chunks)
    completed_files = 0
    completed_chunks = 0
    completed_rows = 0
    chunk_dfs: List[pd.DataFrame] = []

    try:
        with ProcessPoolExecutor(max_workers=resolved_workers) as pool:
            futures = {
                pool.submit(_load_tle_chunk_worker, str(tle_dir), chunk, tuple(derived_cols)): len(chunk)
                for chunk in chunks
            }
            for fut in as_completed(futures):
                n_files = int(futures[fut])
                df_chunk = fut.result()
                chunk_dfs.append(df_chunk)
                completed_rows += int(len(df_chunk))

                completed_files += n_files
                completed_chunks += 1

                should_report = (
                    int(progress_every_files) > 0
                    and (
                        completed_files % int(progress_every_files) == 0
                        or completed_files >= len(file_list)
                        or completed_chunks == total_chunks
                    )
                )
                if should_report:
                    elapsed = time.perf_counter() - t0
                    _log_info_or_print(
                        "TLE load progress: "
                        f"files={completed_files}/{len(file_list)} "
                        f"chunks={completed_chunks}/{total_chunks} "
                        f"rows={completed_rows} elapsed={elapsed:.1f}s"
                    )
    except Exception as exc:
        _log_info_or_print(f"Parallel TLE load failed ({exc}); falling back to sequential mode.")
        df, filenames = load_all_tle_data([str(tle_dir)], only_files=file_list, derived=set(derived_cols))
        elapsed = time.perf_counter() - t0
        _log_info_or_print(
            f"TLE load complete (fallback sequential): rows={len(df)} files={len(file_list)} elapsed={elapsed:.1f}s"
        )
        return df, filenames

    if chunk_dfs:
        df = pd.concat(chunk_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    elapsed = time.perf_counter() - t0
    rows_per_s = (len(df) / elapsed) if elapsed > 0 else float("inf")
    _log_info_or_print(
        f"TLE load complete (parallel): rows={len(df)} files={len(file_list)} elapsed={elapsed:.1f}s rows_per_s={rows_per_s:.1f}"
    )
    return df, file_list


# -----------------------------------------------------------------------------
# Segment-building
# -----------------------------------------------------------------------------
@dataclass
class SegmentBuilderConfig:
    tolerance_seconds: float = 1.0 * 3600.0  # 1 h; was 6 h
    min_duration_seconds: float = 6.0 * 3600.0
    max_duration_seconds: float = 45.0 * 86400.0
    progress_every_labels: int = 5000
    collocation_enabled: bool = True
    collocation_taus: Tuple[float, float, float] = COLLOCATION_TAU_POINTS


def build_segments_from_tles_and_labels(
    tle_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    cfg: SegmentBuilderConfig,
) -> pd.DataFrame:
    t_start = time.perf_counter()
    tle = tle_df.copy()
    tle["timestamp"] = pd.to_datetime(tle["timestamp"], utc=True).dt.tz_convert(None)
    tle["sat_id"] = tle["sat_id"].map(canonical_sat_id)
    if "sma_proc" not in tle.columns:
        tle["sma_proc"] = tle["sma"]
    if "raan_proc" not in tle.columns:
        tle["raan_proc"] = tle["raan"]
    if "mean_anomaly_proc" not in tle.columns:
        tle["mean_anomaly_proc"] = tle["mean_anomaly"]
    tle["raan"] = tle["raan_proc"]
    tle["mean_anomaly"] = tle["mean_anomaly_proc"]
    tle = tle.sort_values(["sat_id", "timestamp"]).reset_index(drop=True)
    tle["mean_longitude_rad"] = build_mean_longitude_rad(tle)
    tle["inc_rad"] = deg2rad(tle["inc"].to_numpy())
    tle["raan_rad"] = deg2rad(tle["raan"].to_numpy())

    sat_col = find_first_existing(labels_df, ["sat_id", "norad_cat_id", "object_id"])
    phase_col = find_first_existing(labels_df, ["phase", "label", "maneuver_label", "phase_state"])
    start_col = find_first_existing(labels_df, ["start_timestamp", "start_time", "start", "t0", "start_dt", "phase_start"])
    end_col = find_first_existing(labels_df, ["end_timestamp", "end_time", "end", "t1", "end_dt", "phase_end"])

    labels = labels_df.copy()
    labels["sat_id"] = labels[sat_col].map(canonical_sat_id)
    labels["phase"] = labels[phase_col].astype(str)
    labels["start_ts"] = pd.to_datetime(labels[start_col], utc=True).dt.tz_convert(None)
    labels["end_ts"] = pd.to_datetime(labels[end_col], utc=True).dt.tz_convert(None)
    labels = labels.sort_values(["sat_id", "start_ts"]).reset_index(drop=True)

    # Build sat_id -> TLE subgroup once to avoid repeatedly scanning the full table.
    tle_groups: Dict[str, pd.DataFrame] = {
        str(sid): grp.reset_index(drop=True)
        for sid, grp in tle.groupby("sat_id", sort=False)
    }

    records: List[Dict[str, float | int | str]] = []
    total_labels = int(len(labels))
    processed_labels = 0
    tolerance_ns = int(float(cfg.tolerance_seconds) * 1.0e9)
    tau_specs = [
        (float(tau), int(round(100.0 * float(tau))))
        for tau in cfg.collocation_taus
        if 0.0 < float(tau) < 1.0
    ]

    _log_info_or_print(
        f"Segment build start: labels={total_labels} tolerance_h={float(cfg.tolerance_seconds) / 3600.0:.2f}"
    )

    for sat_id, group_labels in labels.groupby("sat_id"):
        group_tle = tle_groups.get(str(sat_id))
        if group_tle is None or group_tle.empty:
            processed_labels += int(len(group_labels))
            continue

        ts_ns = group_tle["timestamp"].astype("datetime64[ns]").to_numpy(dtype=np.int64)
        sma_proc = group_tle["sma_proc"].to_numpy(dtype=np.float64)
        ecc = group_tle["ecc"].to_numpy(dtype=np.float64)
        inc_rad = group_tle["inc_rad"].to_numpy(dtype=np.float64)
        raan_rad = group_tle["raan_rad"].to_numpy(dtype=np.float64)
        lam_rad = group_tle["mean_longitude_rad"].to_numpy(dtype=np.float64)

        if "drag_term" in group_tle.columns:
            bstar = group_tle["drag_term"].to_numpy(dtype=np.float64)
        else:
            bstar = np.full(len(group_tle), np.nan, dtype=np.float64)
        if "ballistic_coefficient" in group_tle.columns:
            cdam = group_tle["ballistic_coefficient"].to_numpy(dtype=np.float64)
        else:
            cdam = np.full(len(group_tle), np.nan, dtype=np.float64)

        for row in group_labels.itertuples(index=False):
            processed_labels += 1
            if int(cfg.progress_every_labels) > 0 and processed_labels % int(cfg.progress_every_labels) == 0:
                elapsed = time.perf_counter() - t_start
                _log_info_or_print(
                    f"Segment build progress: labels={processed_labels}/{total_labels} segments={len(records)} elapsed={elapsed:.1f}s"
                )

            start_ts = row.start_ts
            end_ts = row.end_ts
            dt_s = float((end_ts - start_ts).total_seconds())
            if not np.isfinite(dt_s):
                continue
            if dt_s < cfg.min_duration_seconds or dt_s > cfg.max_duration_seconds:
                continue

            idx0 = nearest_index_by_time_ns(ts_ns, int(start_ts.value), tolerance_ns)
            idx1 = nearest_index_by_time_ns(ts_ns, int(end_ts.value), tolerance_ns)
            if idx0 is None or idx1 is None:
                continue

            lam0 = float(lam_rad[idx0])
            lam1 = float(lam_rad[idx1])
            draan = float(wrap_to_pi(float(raan_rad[idx1]) - float(raan_rad[idx0])))
            dlam = float(wrap_to_pi(lam1 - lam0))

            colloc_record: Dict[str, float | int] = {}
            if bool(cfg.collocation_enabled):
                start_ns = int(start_ts.value)
                for tau, tau_label in tau_specs:
                    t_tau_ns = int(start_ns + tau * dt_s * 1.0e9)
                    idx_tau = nearest_index_by_time_ns(ts_ns, t_tau_ns, tolerance_ns)
                    key_mask = f"has_tau{tau_label}"
                    key_a = f"a_tau{tau_label}_km"
                    key_raan = f"raan_tau{tau_label}_rad"
                    key_lam = f"lam_tau{tau_label}_rad"
                    if idx_tau is None:
                        colloc_record[key_mask] = 0
                        colloc_record[key_a] = float("nan")
                        colloc_record[key_raan] = float("nan")
                        colloc_record[key_lam] = float("nan")
                    else:
                        colloc_record[key_mask] = 1
                        colloc_record[key_a] = float(sma_proc[idx_tau])
                        colloc_record[key_raan] = float(raan_rad[idx_tau])
                        colloc_record[key_lam] = float(lam_rad[idx_tau])

            records.append(
                {
                    "sat_id": sat_id,
                    "phase": str(row.phase),
                    "phase_sign": phase_sign_from_name(str(row.phase)),
                    "start_timestamp": start_ts.isoformat(),
                    "end_timestamp": end_ts.isoformat(),
                    "dt_s": dt_s,
                    "a0_km": float(sma_proc[idx0]),
                    "a1_km": float(sma_proc[idx1]),
                    "e0": float(ecc[idx0]),
                    "e1": float(ecc[idx1]),
                    "inc0_rad": float(inc_rad[idx0]),
                    "inc1_rad": float(inc_rad[idx1]),
                    "raan0_rad": float(raan_rad[idx0]),
                    "raan1_rad": float(raan_rad[idx1]),
                    "lam0_rad": lam0,
                    "lam1_rad": lam1,
                    "da_obs_km": float(sma_proc[idx1] - sma_proc[idx0]),
                    "draan_obs_rad": draan,
                    "dlam_obs_rad": dlam,
                    "bstar0": float(bstar[idx0]),
                    "cdam0": float(cdam[idx0]),
                    **colloc_record,
                }
            )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise RuntimeError("No valid maneuver segments were created. Check label timestamps and matching tolerance.")
    elapsed = time.perf_counter() - t_start
    _log_info_or_print(
        f"Segment build complete: labels={processed_labels}/{total_labels} segments={len(out)} elapsed={elapsed:.1f}s"
    )
    return enrich_segment_dataframe(out)


# -----------------------------------------------------------------------------
# Torch dataset
# -----------------------------------------------------------------------------
class SegmentDataset(Dataset):
    def __init__(self, seg_df: pd.DataFrame):
        self.df = seg_df.reset_index(drop=True).copy()

        if "da_rate_km_day" not in self.df.columns or "draan_rate_rad_day" not in self.df.columns or "dlam_rate_rad_day" not in self.df.columns:
            dt_days = np.clip(self.df["dt_s"].to_numpy(dtype=np.float64) / 86400.0, 1.0e-6, np.inf)
            if "da_rate_km_day" not in self.df.columns:
                self.df["da_rate_km_day"] = self.df["da_obs_km"].to_numpy(dtype=np.float64) / dt_days
            if "draan_rate_rad_day" not in self.df.columns:
                self.df["draan_rate_rad_day"] = self.df["draan_obs_rad"].to_numpy(dtype=np.float64) / dt_days
            if "dlam_rate_rad_day" not in self.df.columns:
                self.df["dlam_rate_rad_day"] = self.df["dlam_obs_rad"].to_numpy(dtype=np.float64) / dt_days

        sat_ids = sorted(self.df["sat_id"].astype(str).unique().tolist())
        phases = sorted(self.df["phase"].astype(str).unique().tolist())
        self.sat_to_idx = {s: i for i, s in enumerate(sat_ids)}
        self.phase_to_idx = {p: i for i, p in enumerate(phases)}

        self.df["sat_idx"] = self.df["sat_id"].map(self.sat_to_idx).astype(int)
        self.df["phase_idx"] = self.df["phase"].map(self.phase_to_idx).astype(int)

        self.tensor_dict = {
            "a0_km": torch.tensor(self.df["a0_km"].to_numpy(), dtype=torch.float32),
            "a1_km": torch.tensor(self.df["a1_km"].to_numpy(), dtype=torch.float32),
            "e0": torch.tensor(self.df["e0"].to_numpy(), dtype=torch.float32),
            "inc0_rad": torch.tensor(self.df["inc0_rad"].to_numpy(), dtype=torch.float32),
            "raan0_rad": torch.tensor(self.df["raan0_rad"].to_numpy(), dtype=torch.float32),
            "raan1_rad": torch.tensor(self.df["raan1_rad"].to_numpy(), dtype=torch.float32),
            "lam0_rad": torch.tensor(self.df["lam0_rad"].to_numpy(), dtype=torch.float32),
            "lam1_rad": torch.tensor(self.df["lam1_rad"].to_numpy(), dtype=torch.float32),
            "dt_s": torch.tensor(self.df["dt_s"].to_numpy(), dtype=torch.float32),
            "phase_sign": torch.tensor(self.df["phase_sign"].to_numpy(), dtype=torch.float32),
            "sat_idx": torch.tensor(self.df["sat_idx"].to_numpy(), dtype=torch.long),
            "phase_idx": torch.tensor(self.df["phase_idx"].to_numpy(), dtype=torch.long),
            "target_a1_km": torch.tensor(self.df["a1_km"].to_numpy(), dtype=torch.float32),
            "target_da_km": torch.tensor(self.df["da_obs_km"].to_numpy(), dtype=torch.float32),
            "target_draan_rad": torch.tensor(self.df["draan_obs_rad"].to_numpy(), dtype=torch.float32),
            "target_dlam_rad": torch.tensor(self.df["dlam_obs_rad"].to_numpy(), dtype=torch.float32),
            "target_da_rate_km_day": torch.tensor(self.df["da_rate_km_day"].to_numpy(), dtype=torch.float32),
            "target_draan_rate_rad_day": torch.tensor(self.df["draan_rate_rad_day"].to_numpy(), dtype=torch.float32),
            "target_dlam_rate_rad_day": torch.tensor(self.df["dlam_rate_rad_day"].to_numpy(), dtype=torch.float32),
        }

        for _, tau_label in [(t, int(round(100.0 * float(t)))) for t in COLLOCATION_TAU_POINTS]:
            a_col = f"a_tau{tau_label}_km"
            raan_col = f"raan_tau{tau_label}_rad"
            lam_col = f"lam_tau{tau_label}_rad"
            mask_col = f"has_tau{tau_label}"
            if a_col in self.df.columns and raan_col in self.df.columns and lam_col in self.df.columns:
                self.tensor_dict[f"target_a_tau{tau_label}_km"] = torch.tensor(
                    self.df[a_col].to_numpy(dtype=np.float32), dtype=torch.float32
                )
                self.tensor_dict[f"target_raan_tau{tau_label}_rad"] = torch.tensor(
                    self.df[raan_col].to_numpy(dtype=np.float32), dtype=torch.float32
                )
                self.tensor_dict[f"target_lam_tau{tau_label}_rad"] = torch.tensor(
                    self.df[lam_col].to_numpy(dtype=np.float32), dtype=torch.float32
                )
                finite_tau = np.isfinite(
                    self.df[[a_col, raan_col, lam_col]].to_numpy(dtype=np.float64)
                ).all(axis=1).astype(np.float32)
                if mask_col in self.df.columns:
                    mask_np = self.df[mask_col].to_numpy(dtype=np.float32) * finite_tau
                else:
                    mask_np = finite_tau
                mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=0.0, neginf=0.0)
                self.tensor_dict[f"mask_tau{tau_label}"] = torch.tensor(mask_np, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.tensor_dict.items()}


# -----------------------------------------------------------------------------
# Stage A model
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 100
    batch_size: int = 256
    lr: float = 3.0e-2
    lr_schedule: str = "cosine"  # "cosine" | "none"
    lr_min_factor: float = 0.01  # min LR = lr * lr_min_factor
    early_stopping_patience: int = 10  # stop if no improvement for N epochs; 0 = disabled
    weight_decay: float = 1.0e-5
    seed: int = 42
    compile_model: bool = False
    dry_mass_kg: float = 150.0
    mass_init_kg: float = 250.0
    isp_init_s: float = 1500.0
    eta_init: float = 0.48
    thrust_init_N: float = 0.070
    drag_init_kmps2: float = 1.0e-10
    phase_power_cap_init_W: float = 3500.0
    phase_ramp_fraction_init: float = 0.25
    phase_time_offset_init_s: float = 0.0
    sat_drag_scale_init: float = 1.0
    sat_thrust_scale_init: float = 1.0
    sat_time_bias_init_s: float = 0.0
    util_mass_init: float = 0.85
    util_current_init: float = 0.85
    util_voltage_init: float = 0.85
    divergence_eff_init: float = 0.90
    transport_proxy_init: float = 0.50
    shielding_weight_init: float = 0.10
    lifetime_weight_init: float = 0.10
    thermal_duty_cap: float = 0.85
    use_j2: bool = True
    use_drag: bool = True
    use_power_cap: bool = True
    use_timing_bias: bool = True
    lambda_a: float = 1.0
    lambda_da: float = 2.0
    lambda_raan: float = 5.0
    lambda_lam: float = 0.1  # cautious: enabled now that J2 secular rates are correct
    lambda_a_end: float = 0.5
    lambda_rate: float = 0.25
    lambda_prior: float = 1.0
    robust_loss: str = "mse"
    huber_delta: float = 1.0
    robust_student_t_dof: float = 4.0
    robust_student_t_scale: float = 1.0
    phase_loss_weight_power: float = 0.50
    obs_weight_a: float = 1.0
    obs_weight_da: float = 1.0
    obs_weight_raan: float = 1.0
    obs_weight_lam: float = 1.0
    obs_weight_rate: float = 1.0
    obs_weight_da_rate: float = 1.0
    obs_weight_draan_rate: float = 1.0
    obs_weight_dlam_rate: float = 0.0  # disabled until lambda wrapping fix verified
    obs_weight_collocation: float = 0.5
    obs_scale_a_km: float = 3.0
    obs_scale_angle_rad: float = 0.02
    obs_scale_da_rate_km_day: float = 0.5
    obs_scale_angle_rate_rad_day: float = 5.0e-3
    collocation_enabled: bool = True
    collocation_taus: Tuple[float, float, float] = COLLOCATION_TAU_POINTS
    collocation_tolerance_hours: float = 6.0
    use_piecewise_thrust_schedule: bool = True
    piecewise_midpoint_scale_init: float = 1.0
    vd_init_V: float = 320.0
    vc_init_V: float = 25.0
    vb_init_V: float = 295.0
    ib_init_A: float = 3.0
    eta_b_init: float = 0.85
    eta_v_init: float = 0.90
    eta_m_init: float = 0.75
    eta_o_init: float = 0.82
    gamma_init: float = 1.0
    nu_a_init: float = 0.25
    mdot_a_init_kg_s: float = 4.5e-6
    mdot_c_init_kg_s: float = 8.0e-7
    pressure_base_pa: float = 4.5e-3
    pressure_gain_pa_per_kg_s: float = 350.0
    neutral_temp_K: float = 900.0
    electron_temp_base_eV: float = 4.0
    electron_temp_gain_per_V: float = 0.015
    electron_temp_gain_nua: float = 8.0
    ionization_length_m: float = 0.03
    ionization_ratio_min: float = 0.25
    ionization_ratio_max: float = 3.0
    max_grad_norm: float = 1.0
    lambda_hall: float = 1.0e-3
    lambda_chemistry: float = 1.0e-4
    lambda_feasibility: float = 1.0
    curriculum_kinematics_epochs: int = 5
    curriculum_collocation_epochs: int = 20
    curriculum_physics_ramp_epochs: int = 15
    duration_weight_enabled: bool = True
    duration_weight_power: float = 0.25
    # New r3 fields──
    closure_mode: str = "legacy_surrogate"  # "legacy_surrogate" | "tabulated"
    shell_drag_comp_fraction_init: float = 1.0  # ≈1 means drag-compensated
    # Trajectory-matching r4 fields ────────────────────────────────────
    fit_mode: str = "segment_endpoint"  # "segment_endpoint" | "trajectory_matching"
    intervals_csv: str = ""
    max_arc_obs: int = 200
    min_arc_obs: int = 5
    max_subarc_days: float = 30.0
    lambda_continuity: float = 0.1
    lambda_path: float = 5.0
    lambda_endpoint_a: float = 1.0
    lambda_endpoint_raan: float = 0.0
    lambda_endpoint_lam: float = 0.0
    arc_weight_mode: str = "sqrt_inv_n_obs"
    # Atmosphere-based drag ────────────────────────────────────────────
    use_atmosphere_drag: bool = True  # Replace constant drag with USSA76 altitude-dependent model
    inv_ballistic_coeff: float = 0.0334  # Cd·A/(2·m) [m²/kg] for Starlink flat plate
    # Non-linear propagation ───────────────────────────────────────────
    nonlinear_propagation: bool = True   # RK4 ODE integration with altitude-dependent drag
    rk4_step_hours: float = 12.0         # RK4 integration step size [hours]
    # Mixed-precision training ──────────────────────────────────────────
    mixed_precision: bool = True         # Use torch.amp autocast + GradScaler for faster GPU training


def _logit_scalar(x: float, eps: float = 1.0e-6) -> float:
    x = min(max(float(x), eps), 1.0 - eps)
    return math.log(x / (1.0 - x))


class StageAModel(nn.Module):
    """Physics-informed orbit-manoeuvre model.

    Parameter architecture
    ----------------------
    **Global** (shared across all phase states):
        mass_kg, dry_mass_kg, isp_s, eta_total, util_mass, util_current,
        util_voltage, divergence_eff, transport_proxy, shielding_weight,
        lifetime_weight.
    **Per-phase** (one value per phase state — insertion_or_orbit_raise,
        operational_shell, disposal_lowering):
        thrust_N, duty, drag_kmps2, power_cap_W, ramp_fraction,
        time_offset_s, direction_strength, midpoint_scale,
        vd_V, vc_V, vb_V, ib_A, eta_b, eta_v, eta_m, eta_o,
        gamma, nu_a, mdot_a_kg_s, mdot_c_kg_s.
    **Per-satellite random effects**:
        thrust_scale, drag_scale, time_bias_s.
    """
    def __init__(self, nsat: int, nphase: int, phase_signs: Sequence[float], cfg: TrainConfig):
        super().__init__()
        self.nsat = nsat
        self.nphase = nphase
        self.cfg = cfg

        self.raw_mass_kg = nn.Parameter(torch.tensor(float(cfg.mass_init_kg)))
        self.raw_dry_mass_kg = nn.Parameter(torch.tensor(float(cfg.dry_mass_kg)))
        self.raw_isp_s = nn.Parameter(torch.tensor(float(cfg.isp_init_s)))
        self.raw_eta = nn.Parameter(torch.tensor(float(_logit_scalar(cfg.eta_init))))

        self.raw_util_mass = nn.Parameter(torch.tensor(float(_logit_scalar(cfg.util_mass_init))))
        self.raw_util_current = nn.Parameter(torch.tensor(float(_logit_scalar(cfg.util_current_init))))
        self.raw_util_voltage = nn.Parameter(torch.tensor(float(_logit_scalar(cfg.util_voltage_init))))
        self.raw_divergence_eff = nn.Parameter(torch.tensor(float(_logit_scalar(cfg.divergence_eff_init))))
        self.raw_transport_proxy = nn.Parameter(torch.tensor(float(_logit_scalar(cfg.transport_proxy_init))))
        self.raw_shielding_weight = nn.Parameter(torch.tensor(float(cfg.shielding_weight_init)))
        self.raw_lifetime_weight = nn.Parameter(torch.tensor(float(cfg.lifetime_weight_init)))

        thrust_raw_init = inv_softplus_scalar(float(cfg.thrust_init_N))
        drag_raw_init = inv_softplus_scalar(float(cfg.drag_init_kmps2))

        self.raw_phase_thrust_N = nn.Parameter(torch.full((nphase,), thrust_raw_init))
        self.raw_phase_duty_logit = nn.Parameter(torch.zeros(nphase))
        self.raw_phase_drag_kmps2 = nn.Parameter(torch.full((nphase,), drag_raw_init))
        self.raw_phase_power_cap_W = nn.Parameter(torch.full((nphase,), inv_softplus_scalar(float(cfg.phase_power_cap_init_W))))
        self.raw_phase_ramp_logit = nn.Parameter(torch.full((nphase,), _logit_scalar(cfg.phase_ramp_fraction_init)))
        self.raw_phase_time_offset_s = nn.Parameter(torch.full((nphase,), float(cfg.phase_time_offset_init_s)))
        self.raw_phase_direction_logit = nn.Parameter(torch.zeros(nphase))
        self.raw_phase_midpoint_log_scale = nn.Parameter(
            torch.full((nphase,), math.log(max(1.0e-3, float(cfg.piecewise_midpoint_scale_init))))
        )

        self.raw_phase_vd_V = nn.Parameter(torch.full((nphase,), inv_softplus_scalar(float(cfg.vd_init_V))))
        self.raw_phase_vc_V = nn.Parameter(torch.full((nphase,), inv_softplus_scalar(float(cfg.vc_init_V))))
        self.raw_phase_vb_V = nn.Parameter(torch.full((nphase,), inv_softplus_scalar(float(cfg.vb_init_V))))
        self.raw_phase_ib_A = nn.Parameter(torch.full((nphase,), inv_softplus_scalar(float(cfg.ib_init_A))))
        self.raw_phase_eta_b = nn.Parameter(torch.full((nphase,), _logit_scalar(cfg.eta_b_init)))
        self.raw_phase_eta_v = nn.Parameter(torch.full((nphase,), _logit_scalar(cfg.eta_v_init)))
        self.raw_phase_eta_m = nn.Parameter(torch.full((nphase,), _logit_scalar(cfg.eta_m_init)))
        self.raw_phase_eta_o = nn.Parameter(torch.full((nphase,), _logit_scalar(cfg.eta_o_init)))
        self.raw_phase_gamma = nn.Parameter(torch.full((nphase,), inv_softplus_scalar(float(cfg.gamma_init))))
        self.raw_phase_nu_a = nn.Parameter(torch.full((nphase,), _logit_scalar(cfg.nu_a_init)))
        self.raw_phase_mdot_a_kg_s = nn.Parameter(torch.full((nphase,), inv_softplus_scalar(float(cfg.mdot_a_init_kg_s))))
        self.raw_phase_mdot_c_kg_s = nn.Parameter(torch.full((nphase,), inv_softplus_scalar(float(cfg.mdot_c_init_kg_s))))

        # Shell drag-compensation fraction (per phase): ≈1 means drag-compensated station-keeping
        self.raw_shell_drag_comp_fraction = nn.Parameter(
            torch.full((nphase,), inv_softplus_scalar(float(cfg.shell_drag_comp_fraction_init)))
        )

        sat_thrust_raw = math.log(max(1.0e-6, float(cfg.sat_thrust_scale_init)))
        sat_drag_raw = math.log(max(1.0e-6, float(cfg.sat_drag_scale_init)))
        self.raw_sat_thrust_log_scale = nn.Parameter(torch.full((nsat,), sat_thrust_raw))
        self.raw_sat_drag_log_scale = nn.Parameter(torch.full((nsat,), sat_drag_raw))
        self.raw_sat_time_bias_s = nn.Parameter(torch.full((nsat,), float(cfg.sat_time_bias_init_s)))

        self.register_buffer("phase_signs", torch.tensor(phase_signs, dtype=torch.float32))

    def constrained_parameters(self) -> Dict[str, torch.Tensor]:
        dry_mass_kg = torch.clamp(torch.nn.functional.softplus(self.raw_dry_mass_kg), min=80.0, max=350.0)
        mass_kg = torch.clamp(torch.nn.functional.softplus(self.raw_mass_kg), min=dry_mass_kg + 1.0)
        mass_kg = torch.clamp(mass_kg, max=400.0)
        isp_s = torch.clamp(torch.nn.functional.softplus(self.raw_isp_s), min=500.0, max=3500.0)
        eta_total_global = torch.sigmoid(self.raw_eta).clamp(1.0e-4, 0.95)

        util_mass = torch.sigmoid(self.raw_util_mass).clamp(0.05, 0.999)
        util_current = torch.sigmoid(self.raw_util_current).clamp(0.05, 0.999)
        util_voltage = torch.sigmoid(self.raw_util_voltage).clamp(0.05, 0.999)
        divergence_eff = torch.sigmoid(self.raw_divergence_eff).clamp(0.05, 0.999)
        transport_proxy = torch.sigmoid(self.raw_transport_proxy).clamp(1.0e-3, 0.999)
        shielding_weight = torch.nn.functional.softplus(self.raw_shielding_weight)
        lifetime_weight = torch.nn.functional.softplus(self.raw_lifetime_weight)

        thrust_N = torch.nn.functional.softplus(self.raw_phase_thrust_N) + 1.0e-7
        duty = torch.sigmoid(self.raw_phase_duty_logit).clamp(1.0e-4, 0.999)
        drag_kmps2 = torch.nn.functional.softplus(self.raw_phase_drag_kmps2) + 1.0e-14
        phase_power_cap_W = torch.nn.functional.softplus(self.raw_phase_power_cap_W) + 1.0
        phase_ramp_fraction = torch.sigmoid(self.raw_phase_ramp_logit).clamp(0.0, 1.0)
        phase_time_offset_s = torch.clamp(self.raw_phase_time_offset_s, min=-24.0 * 3600.0, max=24.0 * 3600.0)
        phase_direction_strength = torch.tanh(self.raw_phase_direction_logit).clamp(-1.0, 1.0)
        # Clamp log-scales before exponentiation to avoid overflow and exploding gradients.
        phase_midpoint_log_scale = torch.clamp(
            self.raw_phase_midpoint_log_scale,
            min=math.log(0.25),
            max=math.log(4.0),
        )
        phase_midpoint_scale = torch.exp(phase_midpoint_log_scale)

        phase_vd_V = torch.clamp(torch.nn.functional.softplus(self.raw_phase_vd_V), min=120.0, max=600.0)
        phase_vc_V = torch.clamp(torch.nn.functional.softplus(self.raw_phase_vc_V), min=5.0, max=150.0)
        phase_vb_direct_V = torch.clamp(torch.nn.functional.softplus(self.raw_phase_vb_V), min=20.0, max=550.0)
        phase_vb_from_diff_V = torch.clamp(phase_vd_V - phase_vc_V, min=5.0, max=550.0)
        phase_vb_effective_V = torch.clamp(0.5 * (phase_vb_direct_V + phase_vb_from_diff_V), min=5.0, max=550.0)
        phase_ib_A = torch.clamp(torch.nn.functional.softplus(self.raw_phase_ib_A), min=0.10, max=25.0)
        phase_eta_b = torch.sigmoid(self.raw_phase_eta_b).clamp(0.20, 0.99)
        phase_eta_v = torch.sigmoid(self.raw_phase_eta_v).clamp(0.20, 0.99)
        phase_eta_m = torch.sigmoid(self.raw_phase_eta_m).clamp(0.10, 0.99)
        phase_eta_o = torch.sigmoid(self.raw_phase_eta_o).clamp(0.10, 0.99)
        phase_gamma = torch.clamp(torch.nn.functional.softplus(self.raw_phase_gamma), min=0.35, max=2.5)
        phase_nu_a = torch.sigmoid(self.raw_phase_nu_a).clamp(1.0e-4, 0.999)
        phase_mdot_a_kg_s = torch.nn.functional.softplus(self.raw_phase_mdot_a_kg_s) + 1.0e-10
        phase_mdot_c_kg_s = torch.nn.functional.softplus(self.raw_phase_mdot_c_kg_s) + 1.0e-10

        phase_eta_factorized = torch.clamp(
            (phase_gamma ** 2) * phase_eta_b * phase_eta_v * phase_eta_m * phase_eta_o,
            min=1.0e-4,
            max=0.98,
        )
        phase_eta_total = torch.clamp(0.5 * (phase_eta_factorized + eta_total_global), min=1.0e-4, max=0.98)

        # Shell drag-compensation fraction: ≈1.0 for station-keeping
        shell_drag_comp_fraction = torch.clamp(
            torch.nn.functional.softplus(self.raw_shell_drag_comp_fraction), min=0.5, max=1.5
        )

        sat_thrust_scale = torch.exp(torch.clamp(self.raw_sat_thrust_log_scale, min=-4.0, max=4.0))
        sat_drag_scale = torch.exp(torch.clamp(self.raw_sat_drag_log_scale, min=-6.0, max=6.0))
        sat_time_bias_s = torch.clamp(self.raw_sat_time_bias_s, min=-24.0 * 3600.0, max=24.0 * 3600.0)

        return {
            "mass_kg": mass_kg,
            "dry_mass_kg": dry_mass_kg,
            "isp_s": isp_s,
            "eta_total": eta_total_global,
            "util_mass": util_mass,
            "util_current": util_current,
            "util_voltage": util_voltage,
            "divergence_eff": divergence_eff,
            "transport_proxy": transport_proxy,
            "shielding_weight": shielding_weight,
            "lifetime_weight": lifetime_weight,
            "thrust_N": thrust_N,
            "duty": duty,
            "drag_kmps2": drag_kmps2,
            "phase_power_cap_W": phase_power_cap_W,
            "phase_ramp_fraction": phase_ramp_fraction,
            "phase_time_offset_s": phase_time_offset_s,
            "phase_direction_strength": phase_direction_strength,
            "phase_midpoint_scale": phase_midpoint_scale,
            "phase_vd_V": phase_vd_V,
            "phase_vc_V": phase_vc_V,
            "phase_vb_direct_V": phase_vb_direct_V,
            "phase_vb_from_diff_V": phase_vb_from_diff_V,
            "phase_vb_effective_V": phase_vb_effective_V,
            "phase_ib_A": phase_ib_A,
            "phase_eta_b": phase_eta_b,
            "phase_eta_v": phase_eta_v,
            "phase_eta_m": phase_eta_m,
            "phase_eta_o": phase_eta_o,
            "phase_gamma": phase_gamma,
            "phase_nu_a": phase_nu_a,
            "phase_mdot_a_kg_s": phase_mdot_a_kg_s,
            "phase_mdot_c_kg_s": phase_mdot_c_kg_s,
            "phase_eta_factorized": phase_eta_factorized,
            "phase_eta_total": phase_eta_total,
            "shell_drag_comp_fraction": shell_drag_comp_fraction,
            "sat_thrust_scale": sat_thrust_scale,
            "sat_drag_scale": sat_drag_scale,
            "sat_time_bias_s": sat_time_bias_s,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p = self.constrained_parameters()

        phase_idx = batch["phase_idx"]
        sat_idx = batch["sat_idx"]
        a0_km = batch["a0_km"]
        e0 = batch["e0"]
        inc0_rad = batch["inc0_rad"]
        dt_s = batch["dt_s"]
        raan0 = batch["raan0_rad"]
        lam0 = batch["lam0_rad"]

        phase_sign = self.phase_signs[phase_idx]
        # sign=+1 (orbit raise): prograde; sign=-1 (disposal): retrograde;
        # sign=0  (operational shell): prograde drag make-up, modulated by
        #         shell_drag_comp_fraction (≈1.0 for station-keeping).
        dir_strength = p["phase_direction_strength"][phase_idx]
        direction = torch.where(
            phase_sign.abs() < 0.5,
            # operational_shell → prograde drag make-up scaled by learned fraction
            p["shell_drag_comp_fraction"][phase_idx],
            phase_sign * torch.clamp(
                torch.nn.functional.softplus(dir_strength) + 0.25,
                min=0.25, max=1.0,
            ),
        )

        ramp = p["phase_ramp_fraction"][phase_idx]
        ramp_scale = 1.0 - 0.25 * ramp

        thrust_nominal_phase = p["thrust_N"][phase_idx] * p["sat_thrust_scale"][sat_idx]
        if self.cfg.use_piecewise_thrust_schedule:
            thrust_schedule_scale = torch.clamp(0.5 * (1.0 + p["phase_midpoint_scale"][phase_idx]), min=0.4, max=2.0)
        else:
            thrust_schedule_scale = torch.ones_like(thrust_nominal_phase)
        thrust_phase = thrust_nominal_phase * ramp_scale * thrust_schedule_scale
        duty_phase = p["duty"][phase_idx]
        phase_power_cap = p["phase_power_cap_W"][phase_idx]
        phase_eta_total = p["phase_eta_total"][phase_idx]

        power_nominal_W = thrust_phase * G0_M_S2 * p["isp_s"] / (2.0 * phase_eta_total)
        if self.cfg.use_power_cap:
            power_scale = torch.clamp(phase_power_cap / torch.clamp(power_nominal_W, min=1.0), max=1.0)
        else:
            power_scale = torch.ones_like(duty_phase)

        duty_effective = torch.clamp(duty_phase * power_scale, min=1.0e-4, max=self.cfg.thermal_duty_cap)

        if self.cfg.use_drag:
            drag_phase = p["drag_kmps2"][phase_idx] * p["sat_drag_scale"][sat_idx]
        else:
            drag_phase = torch.zeros_like(duty_effective)

        if self.cfg.use_timing_bias:
            dt_effective_s = torch.clamp(
                dt_s + p["phase_time_offset_s"][phase_idx] + p["sat_time_bias_s"][sat_idx],
                min=300.0,
            )
        else:
            dt_effective_s = dt_s

        thrust_accel_kmps2 = direction * duty_effective * (thrust_phase / p["mass_kg"]) / 1000.0
        accel_net_kmps2 = thrust_accel_kmps2 - drag_phase

        a0_safe = torch.clamp(a0_km, min=R_EARTH_KM + 120.0)
        mu_t = torch.tensor(MU_EARTH_KM3_S2, device=a0_km.device, dtype=a0_km.dtype)
        n0 = torch.sqrt(mu_t / (a0_safe ** 3))
        a1_pred = a0_safe + dt_effective_s * (2.0 * accel_net_kmps2 / n0)
        a_mid = 0.5 * (a0_km + a1_pred)
        a_mid = torch.clamp(a_mid, min=R_EARTH_KM + 120.0)

        # RAAN via J2 secular rate (Vallado 2013 Eq. 9-41) ────────────
        if self.cfg.use_j2:
            raan_dot = raan_rate_j2_torch(a_mid, e0, inc0_rad, mu_t)
        else:
            raan_dot = torch.zeros_like(a_mid)
        draan_pred = raan_dot * dt_effective_s
        raan1_pred = raan0 + draan_pred

        # Lambda via documented J2 secular model ───────────────────────
        # λ̇ = Ω̇_J2 + ω̇_J2 + Ṁ_J2  (Brouwer 1959 / Vallado 2013 Ch.9)
        # Trapezoidal average over [a0, a_mid] for better accuracy on
        # manoeuvring arcs where SMA changes significantly.
        if self.cfg.use_j2:
            lam_dot_0 = lambda_dot_j2_torch(a0_safe, e0, inc0_rad, mu_t)
            lam_dot_1 = lambda_dot_j2_torch(a_mid, e0, inc0_rad, mu_t)
            lam_dot_mid = 0.5 * (lam_dot_0 + lam_dot_1)
        else:
            lam_dot_mid = torch.sqrt(mu_t / (a_mid ** 3))
        dlam_pred = wrap_angle(lam_dot_mid * dt_effective_s)
        lam1_pred = lam0 + dlam_pred

        mdot_thrust_kg_s = torch.clamp(
            thrust_phase / (G0_M_S2 * torch.clamp(p["isp_s"], min=150.0) * torch.clamp(p["util_mass"], min=1.0e-3)),
            min=0.0,
        )
        mdot_latent_kg_s = (p["phase_mdot_a_kg_s"][phase_idx] + p["phase_mdot_c_kg_s"][phase_idx]) * duty_effective
        mdot_used_kg_s = 0.5 * (mdot_thrust_kg_s + mdot_latent_kg_s)
        propellant_used_kg = mdot_used_kg_s * dt_effective_s
        mass_end_kg = torch.clamp(p["mass_kg"] - propellant_used_kg, min=p["dry_mass_kg"])
        power_in_W = power_nominal_W * p["util_current"] * p["util_voltage"] * p["divergence_eff"]

        vb_phase = p["phase_vb_effective_V"][phase_idx]
        ib_phase = p["phase_ib_A"][phase_idx]
        gamma_phase = p["phase_gamma"][phase_idx]
        eta_m_phase = p["phase_eta_m"][phase_idx]

        # Krypton beam relations (Goebel & Katz 2008) ─────────────────
        thrust_kr_mN_val = _hb_thrust_kr_mN(gamma_phase, ib_phase, vb_phase)
        thrust_kr_N = thrust_kr_mN_val * 1.0e-3
        isp_kr_s = _hb_isp_kr_s(gamma_phase, eta_m_phase, vb_phase)

        # Chemistry / ionisation closure (surrogate) ───────────────────
        _chem = legacy_surrogate_chemistry(
            vb_phase,
            p["phase_nu_a"][phase_idx],
            mdot_latent_kg_s,
            electron_temp_base_eV=self.cfg.electron_temp_base_eV,
            electron_temp_gain_per_V=self.cfg.electron_temp_gain_per_V,
            electron_temp_gain_nua=self.cfg.electron_temp_gain_nua,
            pressure_base_pa=self.cfg.pressure_base_pa,
            pressure_gain_pa_per_kg_s=self.cfg.pressure_gain_pa_per_kg_s,
            neutral_temp_K=self.cfg.neutral_temp_K,
            ionization_length_m=self.cfg.ionization_length_m,
        )
        te_eV = _chem.te_eV
        sigma_iv_kr = _chem.sigma_iv_m3_s
        neutral_density_m3 = _chem.neutral_density_m3
        electron_density_m3 = _chem.electron_density_m3
        lambda_i_m = _chem.lambda_i_m
        ionization_ratio = _chem.ionization_ratio

        dt_days = torch.clamp(dt_effective_s / 86400.0, min=1.0e-6)
        da_pred_km = a1_pred - a0_safe
        da_rate_km_day = da_pred_km / dt_days
        draan_rate_rad_day = draan_pred / dt_days
        dlam_rate_rad_day = dlam_pred / dt_days  # dlam_pred is already wrapped

        collocation_out: Dict[str, torch.Tensor] = {}
        for tau in self.cfg.collocation_taus:
            tau_f = float(tau)
            tau_label = int(round(100.0 * tau_f))
            dt_tau_s = dt_effective_s * tau_f
            a_tau = a0_safe + dt_tau_s * (2.0 * accel_net_kmps2 / n0)
            a_tau = torch.clamp(a_tau, min=R_EARTH_KM + 120.0)
            a_tau_mid = torch.clamp(0.5 * (a0_safe + a_tau), min=R_EARTH_KM + 120.0)
            if self.cfg.use_j2:
                raan_dot_tau = raan_rate_j2_torch(a_tau_mid, e0, inc0_rad, mu_t)
            else:
                raan_dot_tau = torch.zeros_like(a_tau)
            draan_tau = raan_dot_tau * dt_tau_s
            # Lambda collocation: same J2 secular model as main propagation
            if self.cfg.use_j2:
                lam_dot_tau = lambda_dot_j2_torch(a_tau_mid, e0, inc0_rad, mu_t)
            else:
                lam_dot_tau = torch.sqrt(mu_t / (a_tau_mid ** 3))
            dlam_tau = wrap_angle(lam_dot_tau * dt_tau_s)
            collocation_out[f"a_tau{tau_label}_pred_km"] = a_tau
            collocation_out[f"raan_tau{tau_label}_pred_rad"] = raan0 + draan_tau
            collocation_out[f"lam_tau{tau_label}_pred_rad"] = lam0 + dlam_tau

        return {
            "a1_pred": a1_pred,
            "raan1_pred": raan1_pred,
            "lam1_pred": lam1_pred,
            "da_pred_km": da_pred_km,
            "da_rate_pred_km_day": da_rate_km_day,
            "draan_rate_pred_rad_day": draan_rate_rad_day,
            "dlam_rate_pred_rad_day": dlam_rate_rad_day,
            "draan_pred": draan_pred,
            "dlam_pred": dlam_pred,
            "mass_kg": p["mass_kg"],
            "dry_mass_kg": p["dry_mass_kg"],
            "mass_end_kg": mass_end_kg,
            "isp_s": p["isp_s"],
            "eta_total": p["eta_total"],
            "eta_total_phase": phase_eta_total,
            "eta_factorized_phase": p["phase_eta_factorized"][phase_idx],
            "util_mass": p["util_mass"],
            "util_current": p["util_current"],
            "util_voltage": p["util_voltage"],
            "divergence_eff": p["divergence_eff"],
            "transport_proxy": p["transport_proxy"],
            "shielding_weight": p["shielding_weight"],
            "lifetime_weight": p["lifetime_weight"],
            "thrust_N": thrust_phase,
            "thrust_nominal_N": thrust_nominal_phase,
            "duty": duty_phase,
            "duty_effective": duty_effective,
            "drag_kmps2": drag_phase,
            "phase_power_cap_W": phase_power_cap,
            "dt_effective_s": dt_effective_s,
            "direction": direction,
            "power_nominal_W": power_nominal_W,
            "power_in_W": power_in_W,
            "thrust_schedule_scale": thrust_schedule_scale,
            "mdot_thrust_kg_s": mdot_thrust_kg_s,
            "mdot_latent_kg_s": mdot_latent_kg_s,
            "mdot_used_kg_s": mdot_used_kg_s,
            "propellant_used_kg": propellant_used_kg,
            "phase_vd_V": p["phase_vd_V"][phase_idx],
            "phase_vc_V": p["phase_vc_V"][phase_idx],
            "phase_vb_direct_V": p["phase_vb_direct_V"][phase_idx],
            "phase_vb_from_diff_V": p["phase_vb_from_diff_V"][phase_idx],
            "phase_vb_effective_V": vb_phase,
            "phase_ib_A": ib_phase,
            "phase_eta_b": p["phase_eta_b"][phase_idx],
            "phase_eta_v": p["phase_eta_v"][phase_idx],
            "phase_eta_m": eta_m_phase,
            "phase_eta_o": p["phase_eta_o"][phase_idx],
            "phase_gamma": gamma_phase,
            "phase_nu_a": p["phase_nu_a"][phase_idx],
            "phase_mdot_a_kg_s": p["phase_mdot_a_kg_s"][phase_idx],
            "phase_mdot_c_kg_s": p["phase_mdot_c_kg_s"][phase_idx],
            "shell_drag_comp_fraction": p["shell_drag_comp_fraction"][phase_idx],
            "thrust_kr_N": thrust_kr_N,
            "isp_kr_s": isp_kr_s,
            "te_eV": te_eV,
            "sigma_iv_kr_m3_s": sigma_iv_kr,
            "neutral_density_m3": neutral_density_m3,
            "electron_density_m3": electron_density_m3,
            "lambda_i_m": lambda_i_m,
            "ionization_ratio": ionization_ratio,
            "chemistry_is_surrogate": True,  # flag for downstream labelling
            **collocation_out,
        }


# angle_residual is now imported from reduced_dynamics


def physics_box_penalty(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.relu(lo - x).pow(2).mean() + torch.relu(x - hi).pow(2).mean()


def safe_mse(x: torch.Tensor, scale: float = 1.0, clip: float = 1.0e6) -> torch.Tensor:
    # Use a scaled, clipped quadratic so large physical residuals do not explode the loss.
    if x.numel() == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    scale_value = max(float(scale), 1.0e-8)
    clip_value = float(clip)
    z = x / scale_value
    z = torch.nan_to_num(z, nan=0.0, posinf=clip_value, neginf=-clip_value)
    z = torch.clamp(z, min=-clip_value, max=clip_value)
    return torch.mean(z * z)


def _phase_sample_weights(phase_idx: torch.Tensor, power: float) -> torch.Tensor:
    phase_idx = phase_idx.to(torch.long)
    if phase_idx.numel() == 0:
        return torch.ones_like(phase_idx, dtype=torch.float32)
    counts = torch.bincount(phase_idx, minlength=int(torch.max(phase_idx).item()) + 1).to(torch.float32)
    weights = 1.0 / torch.clamp(counts[phase_idx], min=1.0).pow(float(power))
    return weights / torch.clamp(weights.mean(), min=1.0e-6)


def _duration_sample_weights(dt_s: torch.Tensor, power: float) -> torch.Tensor:
    """Upweight longer segments that carry stronger maneuver signatures."""
    if dt_s.numel() == 0:
        return torch.ones_like(dt_s, dtype=torch.float32)
    dt_days = torch.clamp(dt_s / 86400.0, min=1.0e-6)
    weights = dt_days.pow(float(power))
    return weights / torch.clamp(weights.mean(), min=1.0e-6)


def _robust_loss_from_residual(
    residual: torch.Tensor,
    sample_weight: Optional[torch.Tensor],
    robust_loss: str,
    scale: float,
    huber_delta: float,
    student_t_dof: float,
    student_t_scale: float,
) -> torch.Tensor:
    resid = residual.reshape(-1)
    finite = torch.isfinite(resid)
    if sample_weight is None:
        w = torch.ones_like(resid)
    else:
        w = sample_weight.reshape(-1)
        finite = finite & torch.isfinite(w)
    if not bool(torch.any(finite)):
        return torch.zeros((), dtype=resid.dtype, device=resid.device)

    resid = resid[finite] / max(float(scale), 1.0e-8)
    w = w[finite]

    robust_mode = str(robust_loss).strip().lower()
    if robust_mode == "huber":
        delta = max(float(huber_delta), 1.0e-6)
        abs_r = torch.abs(resid)
        quad = torch.minimum(abs_r, torch.full_like(abs_r, delta))
        lin = abs_r - quad
        per_item = 0.5 * quad ** 2 + delta * lin
    elif robust_mode == "pseudo_huber":
        delta = max(float(huber_delta), 1.0e-6)
        per_item = (delta ** 2) * (torch.sqrt(1.0 + (resid / delta) ** 2) - 1.0)
    elif robust_mode == "student_t":
        dof = max(float(student_t_dof), 1.01)
        t_scale = max(float(student_t_scale), 1.0e-8)
        z = resid / t_scale
        per_item = 0.5 * (dof + 1.0) * torch.log1p((z ** 2) / dof) + math.log(t_scale)
    else:
        per_item = resid ** 2

    return torch.sum(per_item * w) / torch.clamp(torch.sum(w), min=1.0e-6)


def _physics_residual_block(out: Dict[str, torch.Tensor], cfg: TrainConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    hall_eff_resid = out["eta_total_phase"] - out["eta_factorized_phase"]
    beam_voltage_resid = out["phase_vb_direct_V"] - out["phase_vb_from_diff_V"]
    power_resid = out["power_in_W"] - (out["thrust_N"] * G0_M_S2 * out["isp_s"] / (2.0 * torch.clamp(out["eta_total_phase"], min=1.0e-6)))
    krypton_thrust_resid = out["thrust_N"] - out["thrust_kr_N"]
    krypton_isp_resid = out["isp_s"] - out["isp_kr_s"]
    mdot_consistency_resid = out["mdot_thrust_kg_s"] - out["mdot_latent_kg_s"]

    ratio = out["ionization_ratio"]
    ratio_min = float(cfg.ionization_ratio_min)
    ratio_max = float(cfg.ionization_ratio_max)
    ionization_feasibility = torch.relu(ratio_min - ratio) + torch.relu(ratio - ratio_max)

    mass_feas = torch.relu(out["dry_mass_kg"] - out["mass_end_kg"])
    power_cap_feas = torch.relu(out["power_nominal_W"] - out["phase_power_cap_W"])
    duty_feas = torch.relu(out["duty_effective"] - float(cfg.thermal_duty_cap))

    hall_loss = (
        safe_mse(hall_eff_resid, scale=0.1)
        + 1.0e-2 * safe_mse(beam_voltage_resid, scale=100.0)
        + 1.0e-2 * safe_mse(power_resid, scale=1000.0)
        + safe_mse(krypton_thrust_resid, scale=0.05)
        + 1.0e-2 * safe_mse(krypton_isp_resid, scale=1000.0)
        + safe_mse(mdot_consistency_resid, scale=1.0e-5)
    )
    chemistry_loss = safe_mse(ionization_feasibility, scale=1.0, clip=100.0)
    feasibility_loss = (
        safe_mse(mass_feas, scale=10.0)
        + safe_mse(power_cap_feas, scale=1000.0)
        + safe_mse(duty_feas, scale=0.1)
    )

    metrics = {
        "res_hall_eff": float(torch.mean(torch.abs(hall_eff_resid)).detach().cpu()),
        "res_beam_voltage_V": float(torch.mean(torch.abs(beam_voltage_resid)).detach().cpu()),
        "res_power_W": float(torch.mean(torch.abs(power_resid)).detach().cpu()),
        "res_kr_thrust_N": float(torch.mean(torch.abs(krypton_thrust_resid)).detach().cpu()),
        "res_kr_isp_s": float(torch.mean(torch.abs(krypton_isp_resid)).detach().cpu()),
        "res_mdot_kg_s": float(torch.mean(torch.abs(mdot_consistency_resid)).detach().cpu()),
        "res_ionization_ratio": float(torch.mean(torch.abs(ionization_feasibility)).detach().cpu()),
        "res_mass_feas": float(torch.mean(mass_feas).detach().cpu()),
        "res_power_cap_feas": float(torch.mean(power_cap_feas).detach().cpu()),
        "res_duty_feas": float(torch.mean(duty_feas).detach().cpu()),
    }
    return hall_loss, chemistry_loss, feasibility_loss, metrics


def stage_a_loss(model: StageAModel, batch: Dict[str, torch.Tensor], cfg: TrainConfig, epoch: int = 1) -> Tuple[torch.Tensor, Dict[str, float]]:
    out = model(batch)
    for name, value in out.items():
        if torch.is_tensor(value):
            bad_count = int((~torch.isfinite(value)).sum().item())
            if bad_count > 0:
                raise FloatingPointError(f"Stage A forward produced non-finite tensor '{name}' with {bad_count} bad entries.")

    sample_w = _phase_sample_weights(batch["phase_idx"], power=float(cfg.phase_loss_weight_power))
    if bool(getattr(cfg, 'duration_weight_enabled', False)):
        dur_w = _duration_sample_weights(batch["dt_s"], power=float(getattr(cfg, 'duration_weight_power', 0.25)))
        sample_w = sample_w * dur_w
        sample_w = sample_w / torch.clamp(sample_w.mean(), min=1.0e-6)
    loss_a = _robust_loss_from_residual(
        out["a1_pred"] - batch["target_a1_km"],
        sample_w,
        cfg.robust_loss,
        cfg.obs_scale_a_km,
        cfg.huber_delta,
        cfg.robust_student_t_dof,
        cfg.robust_student_t_scale,
    )
    loss_da = _robust_loss_from_residual(
        out["da_pred_km"] - batch["target_da_km"],
        sample_w,
        cfg.robust_loss,
        cfg.obs_scale_a_km,
        cfg.huber_delta,
        cfg.robust_student_t_dof,
        cfg.robust_student_t_scale,
    )
    loss_raan = _robust_loss_from_residual(
        angle_residual(batch["raan0_rad"] + out["draan_pred"], batch["raan0_rad"] + batch["target_draan_rad"]),
        sample_w,
        cfg.robust_loss,
        cfg.obs_scale_angle_rad,
        cfg.huber_delta,
        cfg.robust_student_t_dof,
        cfg.robust_student_t_scale,
    )
    loss_lam = _robust_loss_from_residual(
        angle_residual(batch["lam0_rad"] + out["dlam_pred"], batch["lam0_rad"] + batch["target_dlam_rad"]),
        sample_w,
        cfg.robust_loss,
        cfg.obs_scale_angle_rad,
        cfg.huber_delta,
        cfg.robust_student_t_dof,
        cfg.robust_student_t_scale,
    )
    loss_da_rate = _robust_loss_from_residual(
        out["da_rate_pred_km_day"] - batch["target_da_rate_km_day"],
        sample_w,
        cfg.robust_loss,
        cfg.obs_scale_da_rate_km_day,
        cfg.huber_delta,
        cfg.robust_student_t_dof,
        cfg.robust_student_t_scale,
    )
    loss_draan_rate = _robust_loss_from_residual(
        out["draan_rate_pred_rad_day"] - batch["target_draan_rate_rad_day"],
        sample_w,
        cfg.robust_loss,
        cfg.obs_scale_angle_rate_rad_day,
        cfg.huber_delta,
        cfg.robust_student_t_dof,
        cfg.robust_student_t_scale,
    )
    loss_dlam_rate = _robust_loss_from_residual(
        out["dlam_rate_pred_rad_day"] - batch["target_dlam_rate_rad_day"],
        sample_w,
        cfg.robust_loss,
        cfg.obs_scale_angle_rate_rad_day,
        cfg.huber_delta,
        cfg.robust_student_t_dof,
        cfg.robust_student_t_scale,
    )
    loss_a_end = _robust_loss_from_residual(
        out["a1_pred"] - (batch["a0_km"] + batch["target_da_km"]),
        sample_w,
        cfg.robust_loss,
        cfg.obs_scale_a_km,
        cfg.huber_delta,
        cfg.robust_student_t_dof,
        cfg.robust_student_t_scale,
    )

    loss_collocation = torch.zeros((), dtype=out["a1_pred"].dtype, device=out["a1_pred"].device)
    if bool(cfg.collocation_enabled):
        colloc_count = 0
        for tau in cfg.collocation_taus:
            tau_label = int(round(100.0 * float(tau)))
            pred_a_key = f"a_tau{tau_label}_pred_km"
            pred_raan_key = f"raan_tau{tau_label}_pred_rad"
            pred_lam_key = f"lam_tau{tau_label}_pred_rad"
            tgt_a_key = f"target_a_tau{tau_label}_km"
            tgt_raan_key = f"target_raan_tau{tau_label}_rad"
            tgt_lam_key = f"target_lam_tau{tau_label}_rad"
            mask_key = f"mask_tau{tau_label}"
            if (
                pred_a_key in out
                and pred_raan_key in out
                and pred_lam_key in out
                and tgt_a_key in batch
                and tgt_raan_key in batch
                and tgt_lam_key in batch
            ):
                if mask_key in batch:
                    m = batch[mask_key].to(dtype=sample_w.dtype)
                else:
                    m = torch.isfinite(batch[tgt_a_key]).to(dtype=sample_w.dtype)
                w_tau = sample_w * m
                loss_collocation = loss_collocation + _robust_loss_from_residual(
                    out[pred_a_key] - batch[tgt_a_key],
                    w_tau,
                    cfg.robust_loss,
                    cfg.obs_scale_a_km,
                    cfg.huber_delta,
                    cfg.robust_student_t_dof,
                    cfg.robust_student_t_scale,
                )
                loss_collocation = loss_collocation + _robust_loss_from_residual(
                    angle_residual(out[pred_raan_key], batch[tgt_raan_key]),
                    w_tau,
                    cfg.robust_loss,
                    cfg.obs_scale_angle_rad,
                    cfg.huber_delta,
                    cfg.robust_student_t_dof,
                    cfg.robust_student_t_scale,
                )
                loss_collocation = loss_collocation + _robust_loss_from_residual(
                    angle_residual(out[pred_lam_key], batch[tgt_lam_key]),
                    w_tau,
                    cfg.robust_loss,
                    cfg.obs_scale_angle_rad,
                    cfg.huber_delta,
                    cfg.robust_student_t_dof,
                    cfg.robust_student_t_scale,
                )
                colloc_count += 3
        if colloc_count > 0:
            loss_collocation = loss_collocation / float(colloc_count)

    utility_penalty = (out["util_mass"] - cfg.util_mass_init) ** 2
    utility_penalty = utility_penalty + (out["util_current"] - cfg.util_current_init) ** 2
    utility_penalty = utility_penalty + (out["util_voltage"] - cfg.util_voltage_init) ** 2
    utility_penalty = utility_penalty + (out["divergence_eff"] - cfg.divergence_eff_init) ** 2
    utility_penalty = utility_penalty + (out["transport_proxy"] - cfg.transport_proxy_init) ** 2
    utility_penalty = torch.mean(utility_penalty)

    power_target = torch.full_like(out["power_in_W"], float(cfg.phase_power_cap_init_W))
    loss_power_target = safe_mse(out["power_in_W"] - power_target, scale=1000.0)

    hall_residual, chemistry_penalty, feasibility_residual, physics_metrics = _physics_residual_block(out, cfg)

    prior_penalty = 0.0 * loss_a
    prior_penalty = prior_penalty + physics_box_penalty(out["mass_kg"], 180.0, 320.0)
    prior_penalty = prior_penalty + physics_box_penalty(out["isp_s"], 900.0, 2500.0)
    prior_penalty = prior_penalty + physics_box_penalty(out["eta_total_phase"], 0.20, 0.90)
    prior_penalty = prior_penalty + physics_box_penalty(out["thrust_N"], 0.01, 0.20)
    prior_penalty = prior_penalty + physics_box_penalty(out["power_in_W"], 100.0, 4000.0)

    # Curriculum scheduling: ramp weights based on training epoch.
    kin_epochs = int(getattr(cfg, 'curriculum_kinematics_epochs', 0))
    colloc_epochs = int(getattr(cfg, 'curriculum_collocation_epochs', 0))
    physics_ramp_epochs = int(getattr(cfg, 'curriculum_physics_ramp_epochs', 0))
    # Phase 1: kinematics-only (epochs 1..kin_epochs)
    # Phase 2: add collocation + prior (epochs kin_epochs+1..colloc_epochs)
    # Phase 3: ramp in Hall/chemistry over physics_ramp_epochs
    curriculum_collocation_scale = 1.0
    curriculum_hall_scale = 1.0
    curriculum_chemistry_scale = 1.0
    if kin_epochs > 0 or colloc_epochs > 0:
        if epoch <= kin_epochs:
            curriculum_collocation_scale = 0.0
            curriculum_hall_scale = 0.0
            curriculum_chemistry_scale = 0.0
        elif epoch <= colloc_epochs:
            curriculum_collocation_scale = 1.0
            curriculum_hall_scale = 0.0
            curriculum_chemistry_scale = 0.0
        elif physics_ramp_epochs > 0 and epoch <= colloc_epochs + physics_ramp_epochs:
            ramp_frac = float(epoch - colloc_epochs) / float(physics_ramp_epochs)
            curriculum_hall_scale = ramp_frac
            curriculum_chemistry_scale = ramp_frac

    # Per-channel rate weights (replaces single obs_weight_rate for all three)
    w_da_rate = float(getattr(cfg, 'obs_weight_da_rate', cfg.obs_weight_rate))
    w_draan_rate = float(getattr(cfg, 'obs_weight_draan_rate', cfg.obs_weight_rate))
    w_dlam_rate = float(getattr(cfg, 'obs_weight_dlam_rate', cfg.obs_weight_rate))

    total = (
        cfg.obs_weight_a * cfg.lambda_a * loss_a
        + cfg.obs_weight_da * cfg.lambda_da * loss_da
        + cfg.obs_weight_raan * cfg.lambda_raan * loss_raan
        + cfg.obs_weight_lam * cfg.lambda_lam * loss_lam
        + cfg.lambda_rate * (w_da_rate * loss_da_rate + w_draan_rate * loss_draan_rate + w_dlam_rate * loss_dlam_rate)
        + cfg.lambda_a_end * loss_a_end
        + curriculum_collocation_scale * cfg.obs_weight_collocation * loss_collocation
        + cfg.lambda_prior * (prior_penalty + utility_penalty + loss_power_target)
        + curriculum_hall_scale * cfg.lambda_hall * hall_residual
        + curriculum_chemistry_scale * cfg.lambda_chemistry * chemistry_penalty
        + cfg.lambda_feasibility * feasibility_residual
    )
    metrics = {
        "loss_total": float(total.detach().cpu()),
        "loss_a": float(loss_a.detach().cpu()),
        "loss_da": float(loss_da.detach().cpu()),
        "loss_raan": float(loss_raan.detach().cpu()),
        "loss_lam": float(loss_lam.detach().cpu()),
        "loss_da_rate": float(loss_da_rate.detach().cpu()),
        "loss_draan_rate": float(loss_draan_rate.detach().cpu()),
        "loss_dlam_rate": float(loss_dlam_rate.detach().cpu()),
        "loss_a_end": float(loss_a_end.detach().cpu()),
        "loss_collocation": float(loss_collocation.detach().cpu()),
        "loss_power_target": float(loss_power_target.detach().cpu()),
        "loss_utility": float(utility_penalty.detach().cpu()),
        "loss_hall": float(hall_residual.detach().cpu()),
        "loss_chemistry_penalty": float(chemistry_penalty.detach().cpu()),
        "loss_feasibility": float(feasibility_residual.detach().cpu()),
        "loss_prior": float(prior_penalty.detach().cpu()),
    }
    metrics.update(physics_metrics)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Stage A loss became non-finite after loss assembly.")
    return total, metrics


def load_stage_a_model_from_checkpoint(checkpoint_path: Path, device: str | torch.device) -> Tuple[StageAModel, Dict[str, int], Dict[str, int], Dict[str, Any]]:
    device_obj = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device_obj)

    sat_to_idx = dict(ckpt["sat_to_idx"])
    phase_to_idx = dict(ckpt["phase_to_idx"])
    phase_names_sorted = sorted(phase_to_idx, key=lambda x: phase_to_idx[x])
    phase_signs = ckpt.get("phase_signs", [phase_sign_from_name(p) for p in phase_names_sorted])

    cfg = TrainConfig()
    for k, v in dict(ckpt.get("train_config", {})).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.device = str(device_obj)

    model = StageAModel(len(sat_to_idx), len(phase_to_idx), phase_signs, cfg).to(device_obj)
    state_dict = ckpt["model_state"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Handle checkpoints saved from torch.compile wrappers or older parameter sets.
        stripped_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                stripped_state[k[len("_orig_mod."):]] = v
            else:
                stripped_state[k] = v
        model.load_state_dict(stripped_state, strict=False)
    model.eval()
    return model, sat_to_idx, phase_to_idx, ckpt


def build_stage_a_batch_from_segments(
    seg_df: pd.DataFrame,
    sat_to_idx: Dict[str, int],
    phase_to_idx: Dict[str, int],
    device: str | torch.device,
    allow_unknown_sat: bool = False,
) -> Dict[str, torch.Tensor]:
    df = seg_df.copy()

    if "da_rate_km_day" not in df.columns or "draan_rate_rad_day" not in df.columns or "dlam_rate_rad_day" not in df.columns:
        dt_days = np.clip(df["dt_s"].to_numpy(dtype=np.float64) / 86400.0, 1.0e-6, np.inf)
        if "da_rate_km_day" not in df.columns:
            df["da_rate_km_day"] = df["da_obs_km"].to_numpy(dtype=np.float64) / dt_days
        if "draan_rate_rad_day" not in df.columns:
            df["draan_rate_rad_day"] = df["draan_obs_rad"].to_numpy(dtype=np.float64) / dt_days
        if "dlam_rate_rad_day" not in df.columns:
            df["dlam_rate_rad_day"] = df["dlam_obs_rad"].to_numpy(dtype=np.float64) / dt_days

    sat_idx = df["sat_id"].astype(str).map(sat_to_idx)
    if sat_idx.isna().any():
        if bool(allow_unknown_sat):
            sat_idx = sat_idx.fillna(0)
        else:
            missing = sorted(df.loc[sat_idx.isna(), "sat_id"].astype(str).unique().tolist())
            raise KeyError(f"Segments include sat_ids not present in checkpoint mapping: {missing[:10]}")

    phase_idx = df["phase"].astype(str).map(phase_to_idx)
    if phase_idx.isna().any():
        missing = sorted(df.loc[phase_idx.isna(), "phase"].astype(str).unique().tolist())
        raise KeyError(f"Segments include phases not present in checkpoint mapping: {missing}")

    if "phase_sign" not in df.columns:
        df["phase_sign"] = df["phase"].map(phase_sign_from_name)

    device_obj = torch.device(device)
    batch = {
        "a0_km": torch.tensor(df["a0_km"].to_numpy(), dtype=torch.float32, device=device_obj),
        "a1_km": torch.tensor(df["a1_km"].to_numpy(), dtype=torch.float32, device=device_obj),
        "e0": torch.tensor(df["e0"].to_numpy(), dtype=torch.float32, device=device_obj),
        "inc0_rad": torch.tensor(df["inc0_rad"].to_numpy(), dtype=torch.float32, device=device_obj),
        "raan0_rad": torch.tensor(df["raan0_rad"].to_numpy(), dtype=torch.float32, device=device_obj),
        "lam0_rad": torch.tensor(df["lam0_rad"].to_numpy(), dtype=torch.float32, device=device_obj),
        "dt_s": torch.tensor(df["dt_s"].to_numpy(), dtype=torch.float32, device=device_obj),
        "phase_sign": torch.tensor(df["phase_sign"].to_numpy(), dtype=torch.float32, device=device_obj),
        "sat_idx": torch.tensor(sat_idx.to_numpy(), dtype=torch.long, device=device_obj),
        "phase_idx": torch.tensor(phase_idx.to_numpy(), dtype=torch.long, device=device_obj),
        "target_a1_km": torch.tensor(df["a1_km"].to_numpy(), dtype=torch.float32, device=device_obj),
        "target_da_km": torch.tensor(df["da_obs_km"].to_numpy(), dtype=torch.float32, device=device_obj),
        "target_draan_rad": torch.tensor(df["draan_obs_rad"].to_numpy(), dtype=torch.float32, device=device_obj),
        "target_dlam_rad": torch.tensor(df["dlam_obs_rad"].to_numpy(), dtype=torch.float32, device=device_obj),
        "target_da_rate_km_day": torch.tensor(df["da_rate_km_day"].to_numpy(), dtype=torch.float32, device=device_obj),
        "target_draan_rate_rad_day": torch.tensor(df["draan_rate_rad_day"].to_numpy(), dtype=torch.float32, device=device_obj),
        "target_dlam_rate_rad_day": torch.tensor(df["dlam_rate_rad_day"].to_numpy(), dtype=torch.float32, device=device_obj),
    }

    for _, tau_label in [(t, int(round(100.0 * float(t)))) for t in COLLOCATION_TAU_POINTS]:
        a_col = f"a_tau{tau_label}_km"
        raan_col = f"raan_tau{tau_label}_rad"
        lam_col = f"lam_tau{tau_label}_rad"
        mask_col = f"has_tau{tau_label}"
        if a_col in df.columns and raan_col in df.columns and lam_col in df.columns:
            batch[f"target_a_tau{tau_label}_km"] = torch.tensor(
                df[a_col].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_obj
            )
            batch[f"target_raan_tau{tau_label}_rad"] = torch.tensor(
                df[raan_col].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_obj
            )
            batch[f"target_lam_tau{tau_label}_rad"] = torch.tensor(
                df[lam_col].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_obj
            )
            finite_tau = np.isfinite(
                df[[a_col, raan_col, lam_col]].to_numpy(dtype=np.float64)
            ).all(axis=1).astype(np.float32)
            if mask_col in df.columns:
                mask_np = df[mask_col].to_numpy(dtype=np.float32) * finite_tau
            else:
                mask_np = finite_tau
            mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=0.0, neginf=0.0)
            batch[f"mask_tau{tau_label}"] = torch.tensor(mask_np, dtype=torch.float32, device=device_obj)
    return batch


def run_stage_a_validation(
    segments_csv: Path,
    checkpoint_path: Path,
    out_json: Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    seg_df = sanitize_segments_for_training(_ensure_segment_rate_columns(pd.read_csv(segments_csv)))
    model, sat_to_idx, phase_to_idx, _ = load_stage_a_model_from_checkpoint(checkpoint_path, device=device)
    batch = build_stage_a_batch_from_segments(seg_df, sat_to_idx, phase_to_idx, device=device)

    with torch.no_grad():
        out = model(batch)

    da_pred = out["a1_pred"] - batch["a0_km"]
    phase_sign = batch["phase_sign"]
    prograde_mask = phase_sign > 0
    retrograde_mask = phase_sign < 0

    prograde_mean_da = float(da_pred[prograde_mask].mean().detach().cpu()) if bool(prograde_mask.any()) else float("nan")
    retrograde_mean_da = float(da_pred[retrograde_mask].mean().detach().cpu()) if bool(retrograde_mask.any()) else float("nan")
    prograde_pos_frac = float((da_pred[prograde_mask] > 0.0).float().mean().detach().cpu()) if bool(prograde_mask.any()) else float("nan")
    retrograde_neg_frac = float((da_pred[retrograde_mask] < 0.0).float().mean().detach().cpu()) if bool(retrograde_mask.any()) else float("nan")

    mass_monotonic_ok = bool(torch.all(out["mass_end_kg"] <= out["mass_kg"] + 1.0e-6).detach().cpu())
    power = out["power_in_W"]
    power_in_range_frac = float(((power >= 100.0) & (power <= 4000.0)).float().mean().detach().cpu())

    raan_resid = angle_residual(
        batch["raan0_rad"] + out["draan_pred"],
        batch["raan0_rad"] + batch["target_draan_rad"],
    )
    lam_resid = angle_residual(
        batch["lam0_rad"] + out["dlam_pred"],
        batch["lam0_rad"] + batch["target_dlam_rad"],
    )

    max_abs_raan_resid = float(torch.max(torch.abs(raan_resid)).detach().cpu())
    max_abs_lam_resid = float(torch.max(torch.abs(lam_resid)).detach().cpu())

    rmse_a_km = float(torch.sqrt(torch.mean((out["a1_pred"] - batch["target_a1_km"]) ** 2)).detach().cpu())
    rmse_draan_rad = float(torch.sqrt(torch.mean(raan_resid ** 2)).detach().cpu())
    rmse_dlam_rad = float(torch.sqrt(torch.mean(lam_resid ** 2)).detach().cpu())

    collocation_metrics: Dict[str, Dict[str, float]] = {}
    for tau in COLLOCATION_TAU_POINTS:
        tau_label = int(round(100.0 * float(tau)))
        pred_a_key = f"a_tau{tau_label}_pred_km"
        pred_raan_key = f"raan_tau{tau_label}_pred_rad"
        pred_lam_key = f"lam_tau{tau_label}_pred_rad"
        tgt_a_key = f"target_a_tau{tau_label}_km"
        tgt_raan_key = f"target_raan_tau{tau_label}_rad"
        tgt_lam_key = f"target_lam_tau{tau_label}_rad"
        mask_key = f"mask_tau{tau_label}"
        if pred_a_key in out and tgt_a_key in batch and tgt_raan_key in batch and tgt_lam_key in batch:
            mask = torch.ones_like(batch[tgt_a_key], dtype=torch.bool)
            if mask_key in batch:
                mask = mask & (batch[mask_key] > 0.5)
            mask = mask & torch.isfinite(batch[tgt_a_key]) & torch.isfinite(batch[tgt_raan_key]) & torch.isfinite(batch[tgt_lam_key])
            if bool(torch.any(mask)):
                a_rmse = torch.sqrt(torch.mean((out[pred_a_key][mask] - batch[tgt_a_key][mask]) ** 2))
                raan_rmse = torch.sqrt(torch.mean(angle_residual(out[pred_raan_key][mask], batch[tgt_raan_key][mask]) ** 2))
                lam_rmse = torch.sqrt(torch.mean(angle_residual(out[pred_lam_key][mask], batch[tgt_lam_key][mask]) ** 2))
                collocation_metrics[f"tau_{tau_label}"] = {
                    "rmse_a_km": float(a_rmse.detach().cpu()),
                    "rmse_raan_rad": float(raan_rmse.detach().cpu()),
                    "rmse_lambda_rad": float(lam_rmse.detach().cpu()),
                    "num_points": int(mask.sum().detach().cpu()),
                }

    phase_idx_np = batch["phase_idx"].detach().cpu().numpy().astype(np.int64)
    idx_to_phase = {v: k for k, v in phase_to_idx.items()}
    per_phase_metrics: Dict[str, Dict[str, float]] = {}
    for pidx in np.unique(phase_idx_np):
        m = phase_idx_np == int(pidx)
        if not np.any(m):
            continue
        phase_name = str(idx_to_phase.get(int(pidx), f"phase_{int(pidx)}"))
        m_t = torch.from_numpy(m).to(device=batch["a0_km"].device)
        phase_rmse_a = torch.sqrt(torch.mean((out["a1_pred"][m_t] - batch["target_a1_km"][m_t]) ** 2))
        phase_rmse_draan = torch.sqrt(
            torch.mean(
                angle_residual(
                    batch["raan0_rad"][m_t] + out["draan_pred"][m_t],
                    batch["raan0_rad"][m_t] + batch["target_draan_rad"][m_t],
                ) ** 2
            )
        )
        phase_rmse_dlam = torch.sqrt(
            torch.mean(
                angle_residual(
                    batch["lam0_rad"][m_t] + out["dlam_pred"][m_t],
                    batch["lam0_rad"][m_t] + batch["target_dlam_rad"][m_t],
                ) ** 2
            )
        )
        per_phase_metrics[phase_name] = {
            "num_segments": int(np.sum(m)),
            "rmse_a_km": float(phase_rmse_a.detach().cpu()),
            "rmse_draan_rad": float(phase_rmse_draan.detach().cpu()),
            "rmse_dlambda_rad": float(phase_rmse_dlam.detach().cpu()),
            "mean_power_W": float(torch.mean(out["power_in_W"][m_t]).detach().cpu()),
            "mean_ionization_ratio": float(torch.mean(out["ionization_ratio"][m_t]).detach().cpu()),
        }

    checks = [
        {
            "name": "raise_segments_da_positive",
            "pass": bool(np.isfinite(prograde_mean_da) and prograde_mean_da > 0.0 and prograde_pos_frac >= 0.50),
            "metrics": {
                "mean_da_km": prograde_mean_da,
                "positive_fraction": prograde_pos_frac,
            },
        },
        {
            "name": "deorbit_segments_da_negative",
            "pass": bool(np.isfinite(retrograde_mean_da) and retrograde_mean_da < 0.0 and retrograde_neg_frac >= 0.50),
            "metrics": {
                "mean_da_km": retrograde_mean_da,
                "negative_fraction": retrograde_neg_frac,
            },
        },
        {
            "name": "mass_monotonic_decrease",
            "pass": mass_monotonic_ok,
            "metrics": {},
        },
        {
            "name": "power_in_low_kW_regime",
            "pass": bool(power_in_range_frac >= 0.80),
            "metrics": {
                "fraction_in_100_4000W": power_in_range_frac,
            },
        },
        {
            "name": "angle_residual_wrap_sane",
            "pass": bool(max_abs_raan_resid <= math.pi + 1.0e-6 and max_abs_lam_resid <= math.pi + 1.0e-6),
            "metrics": {
                "max_abs_raan_resid_rad": max_abs_raan_resid,
                "max_abs_lambda_resid_rad": max_abs_lam_resid,
            },
        },
    ]

    report = {
        "segments_csv": str(Path(segments_csv).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "device": str(device),
        "num_segments": int(len(seg_df)),
        "fit_metrics": {
            "rmse_a_km": rmse_a_km,
            "rmse_draan_rad": rmse_draan_rad,
            "rmse_dlambda_rad": rmse_dlam_rad,
        },
        "collocation_metrics": collocation_metrics,
        "per_phase_metrics": per_phase_metrics,
        "hall_summary": {
            "mean_vd_V": float(torch.mean(out["phase_vd_V"]).detach().cpu()),
            "mean_vc_V": float(torch.mean(out["phase_vc_V"]).detach().cpu()),
            "mean_vb_V": float(torch.mean(out["phase_vb_effective_V"]).detach().cpu()),
            "mean_ib_A": float(torch.mean(out["phase_ib_A"]).detach().cpu()),
            "mean_eta_factorized": float(torch.mean(out["eta_factorized_phase"]).detach().cpu()),
            "mean_gamma": float(torch.mean(out["phase_gamma"]).detach().cpu()),
            "mean_mdot_a_kg_s": float(torch.mean(out["phase_mdot_a_kg_s"]).detach().cpu()),
            "mean_mdot_c_kg_s": float(torch.mean(out["phase_mdot_c_kg_s"]).detach().cpu()),
        },
        "checks": checks,
        "all_checks_pass": bool(all(c["pass"] for c in checks)),
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def evaluate_stage_a_metrics_on_dataframe(
    seg_df: pd.DataFrame,
    model: StageAModel,
    sat_to_idx: Dict[str, int],
    phase_to_idx: Dict[str, int],
    device: str,
    allow_unknown_sat: bool = False,
) -> Dict[str, float]:
    batch = build_stage_a_batch_from_segments(
        seg_df,
        sat_to_idx,
        phase_to_idx,
        device=device,
        allow_unknown_sat=allow_unknown_sat,
    )
    with torch.no_grad():
        out = model(batch)

    raan_resid = angle_residual(
        batch["raan0_rad"] + out["draan_pred"],
        batch["raan0_rad"] + batch["target_draan_rad"],
    )
    lam_resid = angle_residual(
        batch["lam0_rad"] + out["dlam_pred"],
        batch["lam0_rad"] + batch["target_dlam_rad"],
    )
    return {
        "num_segments": int(len(seg_df)),
        "rmse_a_km": float(torch.sqrt(torch.mean((out["a1_pred"] - batch["target_a1_km"]) ** 2)).detach().cpu()),
        "rmse_draan_rad": float(torch.sqrt(torch.mean(raan_resid ** 2)).detach().cpu()),
        "rmse_dlambda_rad": float(torch.sqrt(torch.mean(lam_resid ** 2)).detach().cpu()),
        "mean_power_W": float(torch.mean(out["power_in_W"]).detach().cpu()),
        "mean_ionization_ratio": float(torch.mean(out["ionization_ratio"]).detach().cpu()),
    }


def run_timing_sensitivity_analysis(
    segments_csv: Path,
    checkpoint_path: Path,
    out_csv: Path,
    shift_hours: float,
    n_samples: int,
    device: str = "cpu",
) -> pd.DataFrame:
    seg_df = sanitize_segments_for_training(_ensure_segment_rate_columns(pd.read_csv(segments_csv)))
    model, sat_to_idx, phase_to_idx, _ = load_stage_a_model_from_checkpoint(checkpoint_path, device=device)

    shifts = np.linspace(-float(shift_hours), float(shift_hours), max(2, int(n_samples)))
    rows: List[Dict[str, float]] = []
    for sh in shifts:
        df_shift = seg_df.copy()
        df_shift["dt_s"] = np.clip(df_shift["dt_s"].to_numpy(dtype=np.float64) + float(sh) * 3600.0, 300.0, np.inf)
        df_shift = enrich_segment_dataframe(df_shift)
        metrics = evaluate_stage_a_metrics_on_dataframe(df_shift, model, sat_to_idx, phase_to_idx, device=device)
        metrics["timing_shift_hours"] = float(sh)
        rows.append(metrics)

    out_df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def run_force_model_sensitivity_analysis(
    segments_csv: Path,
    checkpoint_path: Path,
    out_csv: Path,
    device: str = "cpu",
) -> pd.DataFrame:
    """Frozen-model toggle sensitivity: evaluate a *trained* model with
    individual force-model terms switched off.

    **This is NOT a retrained ablation.**  The model was fitted with all
    terms active; switching them off post-hoc only measures the *sensitivity*
    of the forward-model output to each term, not the effect on the posterior.

    A true ablation study would retrain from scratch with each term disabled,
    which is far more expensive and is left as a future extension.
    """
    seg_df = sanitize_segments_for_training(_ensure_segment_rate_columns(pd.read_csv(segments_csv)))
    model, sat_to_idx, phase_to_idx, _ = load_stage_a_model_from_checkpoint(checkpoint_path, device=device)

    variants = [
        ("full", True, True, True, True),
        ("no_j2", False, True, True, True),
        ("no_drag", True, False, True, True),
        ("no_power_cap", True, True, False, True),
        ("no_timing_bias", True, True, True, False),
    ]

    rows: List[Dict[str, float]] = []
    old_flags = (model.cfg.use_j2, model.cfg.use_drag, model.cfg.use_power_cap, model.cfg.use_timing_bias)
    try:
        for name, use_j2, use_drag, use_power_cap, use_timing_bias in variants:
            model.cfg.use_j2 = bool(use_j2)
            model.cfg.use_drag = bool(use_drag)
            model.cfg.use_power_cap = bool(use_power_cap)
            model.cfg.use_timing_bias = bool(use_timing_bias)
            m = evaluate_stage_a_metrics_on_dataframe(seg_df, model, sat_to_idx, phase_to_idx, device=device)
            m["variant"] = str(name)
            rows.append(m)
    finally:
        model.cfg.use_j2, model.cfg.use_drag, model.cfg.use_power_cap, model.cfg.use_timing_bias = old_flags

    out_df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def run_synthetic_recovery_refit(
    segments_csv: Path,
    checkpoint_path: Path,
    outdir: Path,
    refit_epochs: int,
    noise_std_a_km: float,
    noise_std_angle_rad: float,
    device: str = "cpu",
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = [42]
    seeds = list(seeds)

    seg_df = sanitize_segments_for_training(_ensure_segment_rate_columns(pd.read_csv(segments_csv)))
    if len(seg_df) == 0:
        raise RuntimeError("Synthetic recovery requires non-empty segments.")

    base_model, sat_to_idx, phase_to_idx, ckpt = load_stage_a_model_from_checkpoint(checkpoint_path, device=device)
    subset_n = int(min(128, len(seg_df)))
    use_df = seg_df.iloc[:subset_n].copy().reset_index(drop=True)
    batch = build_stage_a_batch_from_segments(use_df, sat_to_idx, phase_to_idx, device=device)

    with torch.no_grad():
        out = base_model(batch)

    base_summary_path = Path(checkpoint_path).parent / "stage_a_parameter_summary.json"
    base_summary = _json_load(base_summary_path) if base_summary_path.exists() else {}

    a1 = out["a1_pred"].detach().cpu().numpy()
    draan = out["draan_pred"].detach().cpu().numpy()
    dlam = out["dlam_pred"].detach().cpu().numpy()

    all_seed_reports: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = outdir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(int(seed))
        synth_df = use_df.copy()

        synth_df["a1_km"] = a1 + rng.normal(0.0, float(noise_std_a_km), size=subset_n)
        synth_df["da_obs_km"] = synth_df["a1_km"].to_numpy(dtype=np.float64) - synth_df["a0_km"].to_numpy(dtype=np.float64)
        synth_df["draan_obs_rad"] = draan + rng.normal(0.0, float(noise_std_angle_rad), size=subset_n)
        synth_df["dlam_obs_rad"] = dlam + rng.normal(0.0, float(noise_std_angle_rad), size=subset_n)
        synth_df = enrich_segment_dataframe(synth_df)

        for tau in COLLOCATION_TAU_POINTS:
            tau_label = int(round(100.0 * float(tau)))
            pred_a = out.get(f"a_tau{tau_label}_pred_km")
            pred_raan = out.get(f"raan_tau{tau_label}_pred_rad")
            pred_lam = out.get(f"lam_tau{tau_label}_pred_rad")
            if pred_a is not None:
                synth_df[f"a_tau{tau_label}_km"] = pred_a.detach().cpu().numpy() + rng.normal(0.0, float(noise_std_a_km), size=subset_n)
                synth_df[f"has_tau{tau_label}"] = 1
            if pred_raan is not None:
                synth_df[f"raan_tau{tau_label}_rad"] = pred_raan.detach().cpu().numpy() + rng.normal(0.0, float(noise_std_angle_rad), size=subset_n)
            if pred_lam is not None:
                synth_df[f"lam_tau{tau_label}_rad"] = pred_lam.detach().cpu().numpy() + rng.normal(0.0, float(noise_std_angle_rad), size=subset_n)

        synthetic_csv = seed_dir / "synthetic_segments.csv"
        synth_df.to_csv(synthetic_csv, index=False)

        refit_cfg = TrainConfig()
        for k, v in dict(ckpt.get("train_config", {})).items():
            if hasattr(refit_cfg, k):
                setattr(refit_cfg, k, v)
        refit_cfg.device = str(device)
        refit_cfg.epochs = int(max(10, min(int(refit_epochs), 400)))
        refit_cfg.compile_model = False

        refit_dir = seed_dir / "synthetic_refit"
        refit_ckpt, _ = train_stage_a(synthetic_csv, refit_dir, refit_cfg)
        refit_summary_path = refit_dir / "stage_a_parameter_summary.json"
        refit_summary = _json_load(refit_summary_path) if refit_summary_path.exists() else {}

        recovered = {
            "seed": int(seed),
            "mass_kg_true": float(base_summary.get("mass_kg", np.nan)),
            "mass_kg_recovered": float(refit_summary.get("mass_kg", np.nan)),
            "isp_s_true": float(base_summary.get("isp_s", np.nan)),
            "isp_s_recovered": float(refit_summary.get("isp_s", np.nan)),
            "eta_total_true": float(base_summary.get("eta_total", np.nan)),
            "eta_total_recovered": float(refit_summary.get("eta_total", np.nan)),
        }
        base_phases = base_summary.get("phase_parameters", {}) if isinstance(base_summary, dict) else {}
        refit_phases = refit_summary.get("phase_parameters", {}) if isinstance(refit_summary, dict) else {}
        phase_recovery: Dict[str, Dict[str, float]] = {}
        for phase in base_phases:
            bp = base_phases[phase]
            rp = refit_phases.get(phase, {})
            phase_recovery[str(phase)] = {
                "thrust_N_true": float(bp.get("thrust_N", np.nan)),
                "thrust_N_recovered": float(rp.get("thrust_N", np.nan)),
                "drag_kmps2_true": float(bp.get("drag_kmps2", np.nan)),
                "drag_kmps2_recovered": float(rp.get("drag_kmps2", np.nan)),
            }
        seed_report = {
            "seed": int(seed),
            "synthetic_segments_csv": str(synthetic_csv),
            "refit_checkpoint": str(refit_ckpt),
            "num_segments": int(len(synth_df)),
            "epochs_used": int(refit_cfg.epochs),
            "recovery": recovered,
            "phase_recovery": phase_recovery,
        }
        all_seed_reports.append(seed_report)

    report = {
        "num_seeds": len(seeds),
        "seeds": seeds,
        "seed_reports": all_seed_reports,
    }
    with (outdir / "synthetic_recovery_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def run_loso_cross_validation(
    segments_csv: Path,
    checkpoint_path: Path,
    outdir: Path,
    max_satellites: int,
    device: str = "cpu",
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seg_df = sanitize_segments_for_training(_ensure_segment_rate_columns(pd.read_csv(segments_csv)))
    if "sat_id" not in seg_df.columns:
        raise RuntimeError("LOSO requires sat_id column in segments.")

    sats = sorted(seg_df["sat_id"].astype(str).unique().tolist())[: max(1, int(max_satellites))]
    base_ckpt = torch.load(checkpoint_path, map_location="cpu")

    rows: List[Dict[str, float | str]] = []
    for sat in sats:
        test_df = seg_df[seg_df["sat_id"].astype(str) == sat].copy().reset_index(drop=True)
        train_df = seg_df[seg_df["sat_id"].astype(str) != sat].copy().reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue

        sat_dir = outdir / f"holdout_{sat}"
        sat_dir.mkdir(parents=True, exist_ok=True)
        train_csv = sat_dir / "train_segments.csv"
        train_df.to_csv(train_csv, index=False)

        cfg = TrainConfig()
        for k, v in dict(base_ckpt.get("train_config", {})).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.device = str(device)
        cfg.epochs = int(max(12, min(80, int(cfg.epochs * 0.1))))
        cfg.compile_model = False

        ckpt_path, _ = train_stage_a(train_csv, sat_dir / "refit", cfg)
        model, sat_to_idx, phase_to_idx, _ = load_stage_a_model_from_checkpoint(ckpt_path, device=device)
        metrics = evaluate_stage_a_metrics_on_dataframe(
            test_df,
            model,
            sat_to_idx,
            phase_to_idx,
            device=device,
            allow_unknown_sat=True,
        )
        row: Dict[str, float | str] = {
            "held_out_sat_id": str(sat),
            "num_train_segments": int(len(train_df)),
            "num_test_segments": int(len(test_df)),
        }
        row.update(metrics)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_csv = outdir / "loso_metrics.csv"
    out_df.to_csv(out_csv, index=False)

    summary = {
        "num_satellites": int(len(rows)),
        "loso_metrics_csv": str(out_csv),
        "rmse_a_km_mean": float(out_df["rmse_a_km"].mean()) if "rmse_a_km" in out_df.columns and not out_df.empty else float("nan"),
        "rmse_draan_rad_mean": float(out_df["rmse_draan_rad"].mean()) if "rmse_draan_rad" in out_df.columns and not out_df.empty else float("nan"),
        "rmse_dlambda_rad_mean": float(out_df["rmse_dlambda_rad"].mean()) if "rmse_dlambda_rad" in out_df.columns and not out_df.empty else float("nan"),
    }
    with (outdir / "loso_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def collate_dict(batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch_list[0].keys():
        out[k] = torch.stack([b[k] for b in batch_list], dim=0)
    return out


def train_stage_a(segments_csv: Path, outdir: Path, cfg: TrainConfig) -> Tuple[Path, pd.DataFrame]:
    seed_everything(cfg.seed)
    outdir.mkdir(parents=True, exist_ok=True)

    seg_df = sanitize_segments_for_training(_ensure_segment_rate_columns(pd.read_csv(segments_csv)))
    ds = SegmentDataset(seg_df)
    device = torch.device(cfg.device)

    loader_workers = 2 if device.type == "cuda" else 0
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_dict,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(loader_workers > 0),
    )

    phase_signs = [phase_sign_from_name(p) for p in ds.phase_to_idx.keys()]
    model = StageAModel(len(ds.sat_to_idx), len(ds.phase_to_idx), phase_signs, cfg).to(device)

    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            # Prefer a backend that does not require Triton.
            model = torch.compile(model, backend="aot_eager")
        except Exception as exc:
            print(f"torch.compile unavailable ({exc}); continuing without compile.")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # LR scheduler
    _lr_sched_mode = str(getattr(cfg, 'lr_schedule', 'none')).strip().lower()
    if _lr_sched_mode == 'cosine':
        _eta_min = cfg.lr * float(getattr(cfg, 'lr_min_factor', 0.01))
        scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = (
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=_eta_min)
        )
    else:
        scheduler = None

    # Mixed-precision setup
    _use_amp = bool(cfg.mixed_precision) and device.type == "cuda"
    _amp_dtype = torch.float16 if _use_amp else None
    scaler = torch.amp.GradScaler(device="cuda", enabled=_use_amp)

    history: List[Dict[str, float]] = []
    param_trace: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        metrics_epoch = []
        for batch in loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype, enabled=_use_amp):
                loss, metrics = stage_a_loss(model, batch, cfg, epoch=epoch)
            if not bool(torch.isfinite(loss)):
                dt_min = float(batch["dt_s"].min().detach().cpu())
                dt_max = float(batch["dt_s"].max().detach().cpu())
                raise FloatingPointError(
                    f"Non-finite Stage A loss before backward: loss={float(loss.detach().cpu())}, "
                    f"dt_s_min={dt_min}, dt_s_max={dt_max}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.max_grad_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                raise FloatingPointError(f"Non-finite gradients encountered during Stage A training: {exc}") from exc
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    f"Non-finite gradient norm encountered during Stage A training: {float(grad_norm.detach().cpu())}"
                )
            scaler.step(opt)
            scaler.update()
            metrics_epoch.append(metrics)

        epoch_metrics = {k: float(np.mean([m[k] for m in metrics_epoch])) for k in metrics_epoch[0].keys()}
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)

        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        with torch.no_grad():
            p = base_model.constrained_parameters()
            trace_row: Dict[str, float] = {
                "epoch": float(epoch),
                "mass_kg": float(p["mass_kg"].detach().cpu()),
                "dry_mass_kg": float(p["dry_mass_kg"].detach().cpu()),
                "isp_s": float(p["isp_s"].detach().cpu()),
                "eta_total": float(p["eta_total"].detach().cpu()),
                "util_mass": float(p["util_mass"].detach().cpu()),
                "util_current": float(p["util_current"].detach().cpu()),
                "util_voltage": float(p["util_voltage"].detach().cpu()),
                "divergence_eff": float(p["divergence_eff"].detach().cpu()),
                "transport_proxy": float(p["transport_proxy"].detach().cpu()),
                "shielding_weight": float(p["shielding_weight"].detach().cpu()),
                "lifetime_weight": float(p["lifetime_weight"].detach().cpu()),
                "sat_thrust_scale_mean": float(p["sat_thrust_scale"].mean().detach().cpu()),
                "sat_drag_scale_mean": float(p["sat_drag_scale"].mean().detach().cpu()),
                "sat_time_bias_s_mean": float(p["sat_time_bias_s"].mean().detach().cpu()),
            }
            for phase, idx in ds.phase_to_idx.items():
                trace_row[f"thrust_N__{phase}"] = float(p["thrust_N"][idx].detach().cpu())
                trace_row[f"duty__{phase}"] = float(p["duty"][idx].detach().cpu())
                trace_row[f"drag_kmps2__{phase}"] = float(p["drag_kmps2"][idx].detach().cpu())
                trace_row[f"vd_V__{phase}"] = float(p["phase_vd_V"][idx].detach().cpu())
                trace_row[f"vc_V__{phase}"] = float(p["phase_vc_V"][idx].detach().cpu())
                trace_row[f"vb_V__{phase}"] = float(p["phase_vb_effective_V"][idx].detach().cpu())
                trace_row[f"ib_A__{phase}"] = float(p["phase_ib_A"][idx].detach().cpu())
                trace_row[f"eta_b__{phase}"] = float(p["phase_eta_b"][idx].detach().cpu())
                trace_row[f"eta_v__{phase}"] = float(p["phase_eta_v"][idx].detach().cpu())
                trace_row[f"eta_m__{phase}"] = float(p["phase_eta_m"][idx].detach().cpu())
                trace_row[f"eta_o__{phase}"] = float(p["phase_eta_o"][idx].detach().cpu())
                trace_row[f"gamma__{phase}"] = float(p["phase_gamma"][idx].detach().cpu())
                trace_row[f"nu_a__{phase}"] = float(p["phase_nu_a"][idx].detach().cpu())
                trace_row[f"mdot_a_kg_s__{phase}"] = float(p["phase_mdot_a_kg_s"][idx].detach().cpu())
                trace_row[f"mdot_c_kg_s__{phase}"] = float(p["phase_mdot_c_kg_s"][idx].detach().cpu())
            param_trace.append(trace_row)

        if epoch == 1 or epoch % 2 == 0 or epoch == cfg.epochs:
            print(
                f"epoch={epoch:04d} "
                f"total={epoch_metrics['loss_total']:.6e} "
                f"a={epoch_metrics['loss_a']:.6e} "
                f"da={epoch_metrics['loss_da']:.6e} "
                f"raan={epoch_metrics['loss_raan']:.6e} "
                f"lam={epoch_metrics['loss_lam']:.6e} "
                f"prior={epoch_metrics['loss_prior']:.6e}"
            )
        if scheduler is not None:
            scheduler.step()

    ckpt_path = outdir / "stage_a_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "train_config": asdict(cfg),
            "sat_to_idx": ds.sat_to_idx,
            "phase_to_idx": ds.phase_to_idx,
            "phase_signs": phase_signs,
        },
        ckpt_path,
    )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(outdir / "stage_a_history.csv", index=False)
    trace_df = pd.DataFrame(param_trace)
    trace_df.to_csv(outdir / "stage_a_parameter_trace.csv", index=False)

    # Save a parameter summary.
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    with torch.no_grad():
        p = base_model.constrained_parameters()
        summary = {
            "mass_kg": float(p["mass_kg"].cpu()),
            "dry_mass_kg": float(p["dry_mass_kg"].cpu()),
            "isp_s": float(p["isp_s"].cpu()),
            "eta_total": float(p["eta_total"].cpu()),
            "util_mass": float(p["util_mass"].cpu()),
            "util_current": float(p["util_current"].cpu()),
            "util_voltage": float(p["util_voltage"].cpu()),
            "divergence_eff": float(p["divergence_eff"].cpu()),
            "transport_proxy": float(p["transport_proxy"].cpu()),
            "shielding_weight": float(p["shielding_weight"].cpu()),
            "lifetime_weight": float(p["lifetime_weight"].cpu()),
            "phase_parameters": {
                phase: {
                    "thrust_N": float(p["thrust_N"][idx].cpu()),
                    "duty": float(p["duty"][idx].cpu()),
                    "drag_kmps2": float(p["drag_kmps2"][idx].cpu()),
                    "power_cap_W": float(p["phase_power_cap_W"][idx].cpu()),
                    "ramp_fraction": float(p["phase_ramp_fraction"][idx].cpu()),
                    "time_offset_s": float(p["phase_time_offset_s"][idx].cpu()),
                    "direction_strength": float(p["phase_direction_strength"][idx].cpu()),
                    "vd_V": float(p["phase_vd_V"][idx].cpu()),
                    "vc_V": float(p["phase_vc_V"][idx].cpu()),
                    "vb_direct_V": float(p["phase_vb_direct_V"][idx].cpu()),
                    "vb_from_diff_V": float(p["phase_vb_from_diff_V"][idx].cpu()),
                    "vb_effective_V": float(p["phase_vb_effective_V"][idx].cpu()),
                    "ib_A": float(p["phase_ib_A"][idx].cpu()),
                    "eta_b": float(p["phase_eta_b"][idx].cpu()),
                    "eta_v": float(p["phase_eta_v"][idx].cpu()),
                    "eta_m": float(p["phase_eta_m"][idx].cpu()),
                    "eta_o": float(p["phase_eta_o"][idx].cpu()),
                    "eta_factorized": float(p["phase_eta_factorized"][idx].cpu()),
                    "eta_total_phase": float(p["phase_eta_total"][idx].cpu()),
                    "gamma": float(p["phase_gamma"][idx].cpu()),
                    "nu_a": float(p["phase_nu_a"][idx].cpu()),
                    "mdot_a_kg_s": float(p["phase_mdot_a_kg_s"][idx].cpu()),
                    "mdot_c_kg_s": float(p["phase_mdot_c_kg_s"][idx].cpu()),
                    "sign": float(phase_signs[idx]),
                }
                for phase, idx in ds.phase_to_idx.items()
            },
            "satellite_scale_summary": {
                "thrust_scale_mean": float(p["sat_thrust_scale"].mean().cpu()),
                "thrust_scale_std": float(p["sat_thrust_scale"].std(unbiased=False).cpu()),
                "drag_scale_mean": float(p["sat_drag_scale"].mean().cpu()),
                "drag_scale_std": float(p["sat_drag_scale"].std(unbiased=False).cpu()),
                "time_bias_s_mean": float(p["sat_time_bias_s"].mean().cpu()),
                "time_bias_s_std": float(p["sat_time_bias_s"].std(unbiased=False).cpu()),
            },
        }
    with (outdir / "stage_a_parameter_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return ckpt_path, hist_df


# Trajectory-matching training loop ────────────────────────────────────────

def train_stage_a_trajectory(
    arcs: List[ArcRecord],
    outdir: Path,
    cfg: TrainConfig,
) -> Tuple[Path, pd.DataFrame]:
    """Train Stage A model using trajectory-matching mode.

    Same model + optimizer infrastructure as ``train_stage_a``, but uses
    ``ArcDataset`` and ``trajectory_forward_and_loss`` instead of segment
    endpoints.
    """
    seed_everything(cfg.seed)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = ArcDataset(arcs, max_obs=int(cfg.max_arc_obs))
    device = torch.device(cfg.device)

    loader_workers = 2 if device.type == "cuda" else 0
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_arcs_fn,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(loader_workers > 0),
    )

    phase_signs_list = [phase_sign_from_name(p) for p in ds.phase_to_idx.keys()]
    model = StageAModel(len(ds.sat_to_idx), len(ds.phase_to_idx), phase_signs_list, cfg).to(device)

    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, backend="aot_eager")
        except Exception as exc:
            print(f"torch.compile unavailable ({exc}); continuing without compile.")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    _lr_sched_mode = str(getattr(cfg, 'lr_schedule', 'none')).strip().lower()
    if _lr_sched_mode == 'cosine':
        _eta_min = cfg.lr * float(getattr(cfg, 'lr_min_factor', 0.01))
        scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = (
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=_eta_min)
        )
    else:
        scheduler = None

    # Mixed-precision setup
    _use_amp = bool(cfg.mixed_precision) and device.type == "cuda"
    _amp_dtype = torch.float16 if _use_amp else None
    scaler = torch.amp.GradScaler(device="cuda", enabled=_use_amp)

    traj_cfg = TrajectoryConfig(
        lambda_path=float(cfg.lambda_path),
        lambda_endpoint_a=float(cfg.lambda_endpoint_a),
        lambda_endpoint_raan=float(cfg.lambda_endpoint_raan),
        lambda_endpoint_lam=float(cfg.lambda_endpoint_lam),
        lambda_continuity=float(cfg.lambda_continuity),
        max_subarc_days=float(cfg.max_subarc_days),
        arc_weight_mode=str(cfg.arc_weight_mode),
        robust_loss=cfg.robust_loss,
        huber_delta=cfg.huber_delta,
        obs_scale_a_km=cfg.obs_scale_a_km,
        obs_scale_angle_rad=cfg.obs_scale_angle_rad,
        student_t_dof=cfg.robust_student_t_dof,
        student_t_scale=cfg.robust_student_t_scale,
        use_atmosphere_drag=bool(cfg.use_atmosphere_drag),
        inv_ballistic_coeff=float(cfg.inv_ballistic_coeff),
        nonlinear_propagation=bool(getattr(cfg, 'nonlinear_propagation', True)),
        rk4_step_hours=float(getattr(cfg, 'rk4_step_hours', 12.0)),
    )

    history: List[Dict[str, float]] = []
    param_trace: List[Dict[str, float]] = []

    # Early stopping state
    _es_patience = int(getattr(cfg, 'early_stopping_patience', 10))
    _es_best_rmse = float('inf')
    _es_best_epoch = 0
    _es_best_state: Optional[Dict[str, Any]] = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        metrics_epoch = []
        for batch in loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype, enabled=_use_amp):
                loss, metrics = trajectory_forward_and_loss(
                    model, batch, traj_cfg, epoch=epoch,
                    curriculum_kinematics_epochs=cfg.curriculum_kinematics_epochs,
                    curriculum_physics_ramp_start=cfg.curriculum_collocation_epochs,
                    curriculum_physics_ramp_epochs=cfg.curriculum_physics_ramp_epochs,
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite trajectory loss at epoch {epoch}: {float(loss.detach().cpu())}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                raise FloatingPointError(f"Non-finite gradients during trajectory training: {exc}") from exc
            scaler.step(opt)
            scaler.update()
            metrics_epoch.append(metrics)

        epoch_metrics = {k: float(np.mean([m[k] for m in metrics_epoch])) for k in metrics_epoch[0].keys()}
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)

        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        with torch.no_grad():
            p = base_model.constrained_parameters()
            trace_row: Dict[str, float] = {
                "epoch": float(epoch),
                "mass_kg": float(p["mass_kg"].detach().cpu()),
                "dry_mass_kg": float(p["dry_mass_kg"].detach().cpu()),
                "isp_s": float(p["isp_s"].detach().cpu()),
                "eta_total": float(p["eta_total"].detach().cpu()),
                "sat_thrust_scale_mean": float(p["sat_thrust_scale"].mean().detach().cpu()),
                "sat_drag_scale_mean": float(p["sat_drag_scale"].mean().detach().cpu()),
            }
            for phase, idx in ds.phase_to_idx.items():
                trace_row[f"thrust_N__{phase}"] = float(p["thrust_N"][idx].detach().cpu())
                trace_row[f"duty__{phase}"] = float(p["duty"][idx].detach().cpu())
                trace_row[f"drag_kmps2__{phase}"] = float(p["drag_kmps2"][idx].detach().cpu())
                trace_row[f"shell_drag_comp__{phase}"] = float(p["shell_drag_comp_fraction"][idx].detach().cpu())
            param_trace.append(trace_row)

        if epoch == 1 or epoch % 2 == 0 or epoch == cfg.epochs:
            print(
                f"epoch={epoch:04d} "
                f"total={epoch_metrics['loss_total']:.6e} "
                f"path={epoch_metrics['loss_path']:.6e} "
                f"ep_a={epoch_metrics['loss_endpoint_a']:.6e} "
                f"raan={epoch_metrics['loss_endpoint_raan']:.6e} "
                f"lam={epoch_metrics['loss_endpoint_lam']:.6e} "
                f"cont={epoch_metrics['loss_continuity']:.6e} "
                f"rmse={epoch_metrics['path_rmse_km']:.2f}"
            )
        if scheduler is not None:
            scheduler.step()

        # Early stopping check
        if _es_patience > 0:
            _cur_rmse = epoch_metrics['path_rmse_km']
            if _cur_rmse < _es_best_rmse:
                _es_best_rmse = _cur_rmse
                _es_best_epoch = epoch
                _es_best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif epoch - _es_best_epoch >= _es_patience:
                print(
                    f"Early stopping at epoch {epoch}: best path_rmse_km={_es_best_rmse:.4f} "
                    f"at epoch {_es_best_epoch} (patience={_es_patience})"
                )
                model.load_state_dict(_es_best_state)
                break

    ckpt_path = outdir / "stage_a_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "train_config": asdict(cfg),
            "sat_to_idx": ds.sat_to_idx,
            "phase_to_idx": ds.phase_to_idx,
            "phase_signs": phase_signs_list,
            "fit_mode": "trajectory_matching",
        },
        ckpt_path,
    )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(outdir / "stage_a_history.csv", index=False)
    trace_df = pd.DataFrame(param_trace)
    trace_df.to_csv(outdir / "stage_a_parameter_trace.csv", index=False)

    # Parameter summary
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    with torch.no_grad():
        p = base_model.constrained_parameters()
        summary = {
            "fit_mode": "trajectory_matching",
            "mass_kg": float(p["mass_kg"].cpu()),
            "dry_mass_kg": float(p["dry_mass_kg"].cpu()),
            "isp_s": float(p["isp_s"].cpu()),
            "eta_total": float(p["eta_total"].cpu()),
            "n_arcs": len(arcs),
            "n_satellites": len(ds.sat_to_idx),
            "n_phases": len(ds.phase_to_idx),
            "phase_parameters": {
                phase: {
                    "thrust_N": float(p["thrust_N"][idx].cpu()),
                    "duty": float(p["duty"][idx].cpu()),
                    "drag_kmps2": float(p["drag_kmps2"][idx].cpu()),
                    "shell_drag_comp_fraction": float(p["shell_drag_comp_fraction"][idx].cpu()),
                    "sign": float(phase_signs_list[idx]),
                }
                for phase, idx in ds.phase_to_idx.items()
            },
        }
    with (outdir / "stage_a_parameter_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return ckpt_path, hist_df


# -----------------------------------------------------------------------------
# Stage B: SNPE scaffold
# -----------------------------------------------------------------------------
def _ensure_segment_rate_columns(seg_df: pd.DataFrame) -> pd.DataFrame:
    out = seg_df.copy()
    if "da_rate_km_day" not in out.columns or "draan_rate_rad_day" not in out.columns or "dlam_rate_rad_day" not in out.columns:
        dt_days = np.clip(out["dt_s"].to_numpy(dtype=np.float64) / 86400.0, 1.0e-6, np.inf)
        if "da_rate_km_day" not in out.columns:
            out["da_rate_km_day"] = out["da_obs_km"].to_numpy(dtype=np.float64) / dt_days
        if "draan_rate_rad_day" not in out.columns:
            out["draan_rate_rad_day"] = out["draan_obs_rad"].to_numpy(dtype=np.float64) / dt_days
        if "dlam_rate_rad_day" not in out.columns:
            out["dlam_rate_rad_day"] = out["dlam_obs_rad"].to_numpy(dtype=np.float64) / dt_days
    if "phase_sign" not in out.columns:
        out["phase_sign"] = out["phase"].map(phase_sign_from_name)
    out["dt_days"] = np.clip(out["dt_s"].to_numpy(dtype=np.float64) / 86400.0, 1.0e-6, np.inf)
    return out


def _select_stage_b_subset(seg_df: pd.DataFrame, max_segments: int, seed: int = 42) -> pd.DataFrame:
    n = int(min(len(seg_df), max(1, int(max_segments))))
    if n >= len(seg_df):
        return seg_df.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    if "phase" not in seg_df.columns:
        idx = np.arange(len(seg_df))
        rng.shuffle(idx)
        return seg_df.iloc[idx[:n]].reset_index(drop=True).copy()

    picks: List[int] = []
    counts = seg_df["phase"].astype(str).value_counts()
    for phase, c in counts.items():
        phase_idx = np.where(seg_df["phase"].astype(str).to_numpy() == phase)[0]
        k = int(round(n * (float(c) / float(len(seg_df)))))
        if k <= 0:
            continue
        rng.shuffle(phase_idx)
        picks.extend(phase_idx[: min(k, len(phase_idx))].tolist())
    if len(picks) < n:
        remaining = [i for i in range(len(seg_df)) if i not in set(picks)]
        rng.shuffle(remaining)
        picks.extend(remaining[: n - len(picks)])
    picks = sorted(picks[:n])
    return seg_df.iloc[picks].reset_index(drop=True).copy()


def _stage_b_observation_feature_columns(
    include_initial_conditions: bool = True,
    include_phase_context: bool = True,
    include_rate_features: bool = True,
    drop_parameter_invariant_context: bool = True,
) -> Tuple[List[str], List[str]]:
    feature_cols = ["da_obs_km", "draan_obs_rad", "dlam_obs_rad"]
    if include_rate_features:
        feature_cols += ["da_rate_km_day", "draan_rate_rad_day", "dlam_rate_rad_day"]

    context_cols: List[str] = []
    if include_initial_conditions:
        context_cols += ["a0_km", "e0", "inc0_rad", "dt_days"]
    if include_phase_context:
        context_cols += ["phase_sign"]

    if not drop_parameter_invariant_context:
        feature_cols += context_cols
    return feature_cols, context_cols


def build_stage_b_observation_summary(
    seg_df: pd.DataFrame,
    max_segments: int = 128,
    include_initial_conditions: bool = True,
    include_phase_context: bool = True,
    include_rate_features: bool = True,
    drop_parameter_invariant_context: bool = True,
) -> Tuple[np.ndarray, List[str], pd.DataFrame, List[str]]:
    use_df = _ensure_segment_rate_columns(seg_df)
    use_df = _select_stage_b_subset(use_df, max_segments=max_segments)

    feature_cols, context_cols = _stage_b_observation_feature_columns(
        include_initial_conditions=include_initial_conditions,
        include_phase_context=include_phase_context,
        include_rate_features=include_rate_features,
        drop_parameter_invariant_context=drop_parameter_invariant_context,
    )

    obs_matrix = use_df[feature_cols].to_numpy(dtype=np.float32)
    excluded_context_cols = context_cols if drop_parameter_invariant_context else []
    return obs_matrix.reshape(-1), feature_cols, use_df, excluded_context_cols


def _normalization_stats_from_summary(flat_obs: np.ndarray, num_segments: int, num_features: int, eps: float) -> Dict[str, List[float]]:
    mat = flat_obs.reshape(num_segments, num_features)
    finite_rows = np.isfinite(mat).all(axis=1)
    if not bool(np.all(finite_rows)):
        bad_rows = int((~finite_rows).sum())
        raise RuntimeError(
            f"Stage B normalization summary contains non-finite values in {bad_rows} observation rows."
        )
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0)
    std = np.where((~np.isfinite(std)) | (std < eps), 1.0, std)
    return {
        "mean": mean.astype(np.float64).tolist(),
        "std": std.astype(np.float64).tolist(),
    }


def _apply_summary_normalization(flat_obs: np.ndarray, num_segments: int, stats: Optional[Dict[str, List[float]]]) -> np.ndarray:
    if stats is None:
        return flat_obs.astype(np.float32)
    mat = flat_obs.reshape(num_segments, -1)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((mat - mean) / np.clip(std, 1.0e-8, np.inf)).astype(np.float32).reshape(-1)


def _resolve_runtime_device(device: str | torch.device | None) -> torch.device:
    requested = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(requested)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device '{device_obj}' for Stage B, but CUDA is not available.")
    return device_obj


def _load_stage_b_phase_anchors(
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
    phase_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    train_cfg = dict(checkpoint.get("train_config", {}))

    defaults = {
        "thrust_N": float(train_cfg.get("thrust_init_N", 0.07)),
        "duty": 0.80,
        "drag_kmps2": float(train_cfg.get("drag_init_kmps2", 1.0e-10)),
        "power_cap_W": float(train_cfg.get("phase_power_cap_init_W", 3500.0)),
        "vd_V": float(train_cfg.get("vd_init_V", 320.0)),
        "vc_V": float(train_cfg.get("vc_init_V", 25.0)),
        "vb_V": float(train_cfg.get("vb_init_V", 295.0)),
        "ib_A": float(train_cfg.get("ib_init_A", 3.0)),
        "eta_b": float(train_cfg.get("eta_b_init", 0.85)),
        "eta_v": float(train_cfg.get("eta_v_init", 0.90)),
        "eta_m": float(train_cfg.get("eta_m_init", 0.75)),
        "eta_o": float(train_cfg.get("eta_o_init", 0.82)),
        "gamma": float(train_cfg.get("gamma_init", 1.0)),
        "mdot_a_kg_s": float(train_cfg.get("mdot_a_init_kg_s", 4.5e-6)),
        "mdot_c_kg_s": float(train_cfg.get("mdot_c_init_kg_s", 8.0e-7)),
        "nu_a": float(train_cfg.get("nu_a_init", 0.25)),
    }

    anchors = {str(p): dict(defaults) for p in phase_names}
    summary_path = Path(checkpoint_path).parent / "stage_a_parameter_summary.json"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary_obj = json.load(f)
            phase_params = dict(summary_obj.get("phase_parameters", {}))
            for phase in phase_names:
                if phase in phase_params and isinstance(phase_params[phase], dict):
                    pp = phase_params[phase]
                    anchors[phase]["thrust_N"] = float(pp.get("thrust_N", anchors[phase]["thrust_N"]))
                    anchors[phase]["duty"] = float(pp.get("duty", anchors[phase]["duty"]))
                    anchors[phase]["drag_kmps2"] = float(pp.get("drag_kmps2", anchors[phase]["drag_kmps2"]))
                    anchors[phase]["power_cap_W"] = float(pp.get("power_cap_W", anchors[phase]["power_cap_W"]))
                    anchors[phase]["vd_V"] = float(pp.get("vd_V", anchors[phase]["vd_V"]))
                    anchors[phase]["vc_V"] = float(pp.get("vc_V", anchors[phase]["vc_V"]))
                    anchors[phase]["vb_V"] = float(pp.get("vb_effective_V", anchors[phase]["vb_V"]))
                    anchors[phase]["ib_A"] = float(pp.get("ib_A", anchors[phase]["ib_A"]))
                    anchors[phase]["eta_b"] = float(pp.get("eta_b", anchors[phase]["eta_b"]))
                    anchors[phase]["eta_v"] = float(pp.get("eta_v", anchors[phase]["eta_v"]))
                    anchors[phase]["eta_m"] = float(pp.get("eta_m", anchors[phase]["eta_m"]))
                    anchors[phase]["eta_o"] = float(pp.get("eta_o", anchors[phase]["eta_o"]))
                    anchors[phase]["gamma"] = float(pp.get("gamma", anchors[phase]["gamma"]))
                    anchors[phase]["mdot_a_kg_s"] = float(pp.get("mdot_a_kg_s", anchors[phase]["mdot_a_kg_s"]))
                    anchors[phase]["mdot_c_kg_s"] = float(pp.get("mdot_c_kg_s", anchors[phase]["mdot_c_kg_s"]))
                    anchors[phase]["nu_a"] = float(pp.get("nu_a", anchors[phase]["nu_a"]))
        except Exception:
            pass
    return anchors


def make_stage_b_simulator(
    seg_df: pd.DataFrame,
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
    max_phase_parameters: int = 6,
    include_initial_conditions: bool = True,
    include_phase_context: bool = True,
    include_rate_features: bool = True,
    drop_parameter_invariant_context: bool = True,
    normalization_stats: Optional[Dict[str, List[float]]] = None,
    device: str | torch.device | None = None,
    effective_mode: bool = True,
    anchor_from_stage_a: bool = True,
):
    device_obj = _resolve_runtime_device(device)
    use_df = _ensure_segment_rate_columns(seg_df).reset_index(drop=True)
    phase_names = sorted(use_df["phase"].astype(str).unique().tolist())
    anchors_by_phase = _load_stage_b_phase_anchors(checkpoint, checkpoint_path, phase_names)

    phase_counts = use_df["phase"].astype(str).value_counts()
    active_phases = phase_counts.head(max(1, int(max_phase_parameters))).index.tolist()
    active_phase_set = set(active_phases)

    # Effective mode: reduced Tier-1 parameter set (mass + per-phase thrust/drag only)
    if effective_mode:
        phase_param_specs: List[Tuple[str, float, float]] = [
            ("thrust_N", 5.0e-3, 0.25),
            ("drag_kmps2", 1.0e-13, 1.0e-7),
        ]
    else:
        phase_param_specs: List[Tuple[str, float, float]] = [
            ("thrust_N", 5.0e-3, 0.25),
            ("duty", 1.0e-3, 0.999),
            ("drag_kmps2", 1.0e-13, 1.0e-7),
            ("power_cap_W", 100.0, 8000.0),
            ("vd_V", 120.0, 600.0),
            ("vc_V", 5.0, 150.0),
            ("vb_V", 5.0, 550.0),
            ("ib_A", 0.1, 25.0),
            ("eta_b", 0.1, 0.99),
            ("eta_v", 0.1, 0.99),
            ("eta_m", 0.1, 0.99),
            ("eta_o", 0.1, 0.99),
            ("gamma", 0.35, 2.5),
            ("mdot_a_kg_s", 1.0e-10, 5.0e-4),
            ("mdot_c_kg_s", 1.0e-10, 5.0e-4),
            ("nu_a", 1.0e-4, 0.999),
        ]

    # In effective mode, remove eta_total_global from the free set (not identifiable).
    if effective_mode:
        global_specs: List[Tuple[str, float, float]] = [
            ("mass_kg", 120.0, 400.0),
            ("isp_s", 700.0, 3500.0),
        ]
    else:
        global_specs: List[Tuple[str, float, float]] = [
            ("mass_kg", 120.0, 400.0),
            ("isp_s", 700.0, 3500.0),
            ("eta_total_global", 0.10, 0.95),
        ]

    train_cfg = dict(checkpoint.get("train_config", {}))
    # Build global anchors from fitted Stage A summary when available.
    global_anchors = {
        "mass_kg": float(train_cfg.get("mass_init_kg", 250.0)),
        "isp_s": float(train_cfg.get("isp_init_s", 1500.0)),
        "eta_total_global": float(train_cfg.get("eta_init", 0.48)),
    }
    if anchor_from_stage_a:
        summary_path = Path(checkpoint_path).parent / "stage_a_parameter_summary.json"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    summary_obj = json.load(f)
                if isinstance(summary_obj, dict):
                    global_anchors["mass_kg"] = float(summary_obj.get("mass_kg", global_anchors["mass_kg"]))
                    global_anchors["isp_s"] = float(summary_obj.get("isp_s", global_anchors["isp_s"]))
                    global_anchors["eta_total_global"] = float(summary_obj.get("eta_total", global_anchors["eta_total_global"]))
            except Exception:
                pass

    phase_col = use_df["phase"].astype(str).to_numpy()
    # Cache simulator context on the target device so repeated SBI calls avoid host->device traffic.
    phase_sign = torch.tensor(use_df["phase_sign"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_obj)
    a0 = torch.tensor(use_df["a0_km"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_obj)
    e0 = torch.tensor(use_df["e0"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_obj)
    inc0 = torch.tensor(use_df["inc0_rad"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_obj)
    dt_s = torch.tensor(use_df["dt_s"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device_obj)

    phase_indices: Dict[str, torch.Tensor] = {}
    for p in active_phases:
        idx = np.where(phase_col == p)[0]
        phase_indices[p] = torch.tensor(idx, dtype=torch.long, device=device_obj)

    # Populate anchor_per_segment with ALL closure variable keys (not just
    # the free phase_param_specs), so the effective-mode simulator can read
    # frozen closure values even when they are not part of the prior.
    all_closure_keys = [
        "thrust_N", "duty", "drag_kmps2", "power_cap_W",
        "vd_V", "vc_V", "vb_V", "ib_A",
        "eta_b", "eta_v", "eta_m", "eta_o",
        "gamma", "mdot_a_kg_s", "mdot_c_kg_s", "nu_a",
    ]
    anchor_per_segment: Dict[str, torch.Tensor] = {}
    for pname in all_closure_keys:
        anchor_per_segment[pname] = torch.tensor(
            [anchors_by_phase[str(p)].get(pname, 0.0) for p in phase_col],
            dtype=torch.float32,
            device=device_obj,
        )

    norm_mean_t: Optional[torch.Tensor] = None
    norm_std_t: Optional[torch.Tensor] = None
    if normalization_stats is not None:
        norm_mean_t = torch.tensor(normalization_stats["mean"], dtype=torch.float32, device=device_obj)
        norm_std_t = torch.tensor(normalization_stats["std"], dtype=torch.float32, device=device_obj)

    mu_earth_t = torch.tensor(MU_EARTH_KM3_S2, dtype=torch.float32, device=device_obj)
    kb_times_temp = K_BOLTZMANN_J_K * 900.0
    neutral_speed_coeff = 8.0 * K_BOLTZMANN_J_K * 900.0 / (math.pi * KR_MASS_KG)

    feature_names, context_feature_names = _stage_b_observation_feature_columns(
        include_initial_conditions=include_initial_conditions,
        include_phase_context=include_phase_context,
        include_rate_features=include_rate_features,
        drop_parameter_invariant_context=drop_parameter_invariant_context,
    )
    num_segments = int(len(use_df))
    num_features = int(len(feature_names))
    obs_dim = int(num_segments * num_features)

    param_names: List[str] = []
    low: List[float] = []
    high: List[float] = []
    anchor: List[float] = []
    for name, lo, hi in global_specs:
        param_names.append(name)
        low.append(float(lo))
        high.append(float(hi))
        anchor.append(float(global_anchors[name]))
    for phase in active_phases:
        for pname, lo, hi in phase_param_specs:
            param_names.append(f"{pname}__{phase}")
            low.append(float(lo))
            high.append(float(hi))
            anchor.append(float(anchors_by_phase[phase][pname]))

    def _per_segment_param(theta: torch.Tensor, sampled: Dict[str, Dict[str, torch.Tensor]], pname: str) -> torch.Tensor:
        out = anchor_per_segment[pname].to(dtype=theta.dtype).clone()
        for phase in active_phases:
            idx = phase_indices[phase]
            if idx.numel() > 0:
                out[idx] = sampled[pname][phase]
        return out

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        theta = theta.reshape(-1).to(device=device_obj, dtype=torch.float32)
        expected_dim = len(global_specs) + len(active_phases) * len(phase_param_specs)
        if int(theta.numel()) != int(expected_dim):
            raise ValueError(f"Expected theta dimension {expected_dim}, got {int(theta.numel())}")

        cursor = 0
        mass_kg = torch.clamp(theta[cursor], min=1.0)
        cursor += 1
        isp_s = torch.clamp(theta[cursor], min=200.0)
        cursor += 1
        if not effective_mode:
            eta_total_global = torch.clamp(theta[cursor], min=1.0e-4, max=0.99)
            cursor += 1
        else:
            # In effective mode, eta_total_global is frozen at anchor value.
            eta_total_global = torch.tensor(float(global_anchors.get("eta_total_global", 0.48)), dtype=theta.dtype, device=device_obj)

        sampled: Dict[str, Dict[str, torch.Tensor]] = {pname: {} for pname, _, _ in phase_param_specs}
        for phase in active_phases:
            for pname, lo, hi in phase_param_specs:
                sampled[pname][phase] = torch.clamp(theta[cursor], min=float(lo), max=float(hi))
                cursor += 1

        thrust_N = _per_segment_param(theta, sampled, "thrust_N")
        drag_kmps2 = _per_segment_param(theta, sampled, "drag_kmps2")

        if effective_mode:
            # In effective mode, closure variables frozen at anchor values.
            duty = anchor_per_segment["duty"].to(dtype=theta.dtype)
            power_cap_W = anchor_per_segment["power_cap_W"].to(dtype=theta.dtype)
            vd_V = anchor_per_segment["vd_V"].to(dtype=theta.dtype)
            vc_V = anchor_per_segment["vc_V"].to(dtype=theta.dtype)
            vb_direct_V = anchor_per_segment["vb_V"].to(dtype=theta.dtype)
            ib_A = anchor_per_segment["ib_A"].to(dtype=theta.dtype)
            eta_b = anchor_per_segment["eta_b"].to(dtype=theta.dtype)
            eta_v = anchor_per_segment["eta_v"].to(dtype=theta.dtype)
            eta_m = anchor_per_segment["eta_m"].to(dtype=theta.dtype)
            eta_o = anchor_per_segment["eta_o"].to(dtype=theta.dtype)
            gamma = anchor_per_segment["gamma"].to(dtype=theta.dtype)
            mdot_a_kg_s = anchor_per_segment["mdot_a_kg_s"].to(dtype=theta.dtype)
            mdot_c_kg_s = anchor_per_segment["mdot_c_kg_s"].to(dtype=theta.dtype)
            nu_a = anchor_per_segment["nu_a"].to(dtype=theta.dtype)
        else:
            duty = _per_segment_param(theta, sampled, "duty")
            power_cap_W = _per_segment_param(theta, sampled, "power_cap_W")
            vd_V = _per_segment_param(theta, sampled, "vd_V")
            vc_V = _per_segment_param(theta, sampled, "vc_V")
            vb_direct_V = _per_segment_param(theta, sampled, "vb_V")
            ib_A = _per_segment_param(theta, sampled, "ib_A")
            eta_b = _per_segment_param(theta, sampled, "eta_b")
            eta_v = _per_segment_param(theta, sampled, "eta_v")
            eta_m = _per_segment_param(theta, sampled, "eta_m")
            eta_o = _per_segment_param(theta, sampled, "eta_o")
            gamma = _per_segment_param(theta, sampled, "gamma")
            mdot_a_kg_s = _per_segment_param(theta, sampled, "mdot_a_kg_s")
            mdot_c_kg_s = _per_segment_param(theta, sampled, "mdot_c_kg_s")
            nu_a = _per_segment_param(theta, sampled, "nu_a")

        vb_from_diff_V = torch.clamp(vd_V - vc_V, min=5.0, max=550.0)
        vb_eff_V = torch.clamp(0.5 * (vb_direct_V + vb_from_diff_V), min=5.0, max=550.0)
        eta_factorized = torch.clamp((gamma ** 2) * eta_b * eta_v * eta_m * eta_o, min=1.0e-4, max=0.98)
        eta_total_phase = torch.clamp(0.5 * (eta_factorized + eta_total_global), min=1.0e-4, max=0.98)

        power_nominal_W = thrust_N * G0_M_S2 * isp_s / (2.0 * eta_total_phase)
        power_scale = torch.clamp(power_cap_W / torch.clamp(power_nominal_W, min=1.0), max=1.0)
        duty_effective = torch.clamp(duty * power_scale, min=1.0e-4, max=0.95)

        phase_sign_t = phase_sign.to(dtype=theta.dtype)
        a0_t = a0.to(dtype=theta.dtype)
        e0_t = e0.to(dtype=theta.dtype)
        inc0_t = inc0.to(dtype=theta.dtype)
        dt_s_t = dt_s.to(dtype=theta.dtype)

        accel_kmps2 = phase_sign_t * duty_effective * (thrust_N / mass_kg) / 1000.0 - drag_kmps2
        a0_safe = torch.clamp(a0_t, min=R_EARTH_KM + 120.0)
        n0 = torch.sqrt(mu_earth_t.to(dtype=theta.dtype) / (a0_safe ** 3))
        a1 = a0_safe + dt_s_t * (2.0 * accel_kmps2 / n0)
        a_mid = torch.clamp(0.5 * (a0_safe + a1), min=R_EARTH_KM + 120.0)

        # J2 secular rates (Stage B uses same model as Stage A) ────────
        mu_typed = mu_earth_t.to(dtype=theta.dtype)
        draan = raan_rate_j2_torch(a_mid, e0_t, inc0_t, mu_typed) * dt_s_t
        # λ̇ = Ω̇_J2 + ω̇_J2 + Ṁ_J2  (Brouwer 1959 / Vallado 2013 Ch.9)
        lam_dot_0 = lambda_dot_j2_torch(a0_safe, e0_t, inc0_t, mu_typed)
        lam_dot_1 = lambda_dot_j2_torch(a_mid, e0_t, inc0_t, mu_typed)
        dlam = wrap_angle(0.5 * (lam_dot_0 + lam_dot_1) * dt_s_t)

        dt_days = torch.clamp(dt_s_t / 86400.0, min=1.0e-6)
        da = a1 - a0_safe
        da_rate = da / dt_days
        draan_rate = draan / dt_days
        dlam_rate = dlam / dt_days

        mdot_total_kg_s = (mdot_a_kg_s + mdot_c_kg_s) * duty_effective
        pressure_pa = torch.clamp(4.5e-3 + 350.0 * mdot_total_kg_s, min=1.0e-7, max=50.0)
        te_eV = torch.clamp(4.0 + 0.015 * vb_eff_V + 8.0 * nu_a, min=0.5, max=120.0)
        sigma_iv_kr = 5.0e-15 * torch.pow(te_eV, 1.25) * torch.exp(-9.7 / torch.clamp(te_eV, min=0.5))
        neutral_density = pressure_pa / torch.clamp(torch.full_like(pressure_pa, kb_times_temp), min=1.0e-16)
        electron_density = torch.clamp(nu_a * neutral_density, min=1.0e8)
        neutral_speed = torch.sqrt(torch.clamp(torch.full_like(pressure_pa, neutral_speed_coeff), min=1.0))
        lambda_i = neutral_speed / torch.clamp(electron_density * sigma_iv_kr, min=1.0e-16)
        ionization_ratio = lambda_i / 0.03
        da = da + 0.0 * torch.mean(torch.relu(0.25 - ionization_ratio) + torch.relu(ionization_ratio - 3.0))

        feature_tensors = {
            "da_obs_km": da,
            "draan_obs_rad": draan,
            "dlam_obs_rad": dlam,
            "da_rate_km_day": da_rate,
            "draan_rate_rad_day": draan_rate,
            "dlam_rate_rad_day": dlam_rate,
            "a0_km": a0_safe,
            "e0": e0_t,
            "inc0_rad": inc0_t,
            "dt_days": dt_days,
            "phase_sign": phase_sign_t,
        }

        feat_mat = torch.stack([feature_tensors[name] for name in feature_names], dim=1)
        if norm_mean_t is not None and norm_std_t is not None:
            feat_mat = (feat_mat - norm_mean_t.to(dtype=theta.dtype).unsqueeze(0)) / torch.clamp(
                norm_std_t.to(dtype=theta.dtype).unsqueeze(0),
                min=1.0e-8,
            )

        y = feat_mat.reshape(1, -1)
        if (not torch.isfinite(y).all()) or y.numel() != obs_dim:
            return torch.full((1, obs_dim), 1.0e9, dtype=theta.dtype, device=device_obj)
        return y

    prior_info = {
        "phase_names_sorted": sorted(phase_names),
        "active_phases": active_phases,
        "param_names": param_names,
        "feature_names": feature_names,
        "excluded_parameter_invariant_context_features": context_feature_names if drop_parameter_invariant_context else [],
        "num_segments": num_segments,
        "num_features": num_features,
        "obs_dim": obs_dim,
        "low": low,
        "high": high,
        "anchor": anchor,
        "device": str(device_obj),
    }
    return simulator, prior_info, use_df


def summarize_posterior_samples(samples_np: np.ndarray, param_names: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_samples": int(samples_np.shape[0]),
        "num_parameters": int(samples_np.shape[1]),
        "parameters": {},
    }
    q05, q50, q95 = np.quantile(samples_np, [0.05, 0.50, 0.95], axis=0)
    for idx, name in enumerate(param_names):
        vals = samples_np[:, idx]
        summary["parameters"][name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "q05": float(q05[idx]),
            "median": float(q50[idx]),
            "q95": float(q95[idx]),
        }
    return summary


def posterior_predictive_check(
    simulator,
    posterior_samples: torch.Tensor,
    x_o: torch.Tensor,
    num_segments: int,
    num_features: int,
    feature_names: Sequence[str],
    n_samples: int = 256,
    device: str | torch.device | None = None,
) -> Dict[str, Any]:
    device_obj = _resolve_runtime_device(device) if device is not None else posterior_samples.device
    ns = int(min(int(n_samples), int(posterior_samples.shape[0])))
    theta_eval = posterior_samples[:ns].to(device=device_obj, dtype=torch.float32)
    x_o_vec = x_o.reshape(-1).to(device=device_obj, dtype=torch.float32)

    def sim_flat(theta_1d: torch.Tensor) -> torch.Tensor:
        return simulator(theta_1d).reshape(-1)

    try:
        x_sim = vmap(sim_flat)(theta_eval)
    except Exception:
        x_sim = torch.stack([sim_flat(theta_eval[i]) for i in range(theta_eval.shape[0])], dim=0)

    valid = torch.isfinite(x_sim).all(dim=1)
    x_sim_valid = x_sim[valid]
    if x_sim_valid.numel() == 0:
        return {
            "num_requested": ns,
            "num_valid": 0,
            "warning": "All posterior predictive simulations were invalid.",
        }

    x_mean = x_sim_valid.mean(dim=0)
    resid = x_mean - x_o_vec
    resid_2d = resid.reshape(num_segments, num_features)

    report: Dict[str, Any] = {
        "num_requested": ns,
        "num_valid": int(x_sim_valid.shape[0]),
        "num_features": int(num_features),
    }
    for i, fname in enumerate(feature_names):
        safe_name = str(fname).replace(" ", "_")
        report[f"rmse__{safe_name}"] = float(torch.sqrt(torch.mean(resid_2d[:, i] ** 2)).detach().cpu())
        report[f"mae__{safe_name}"] = float(torch.mean(torch.abs(resid_2d[:, i])).detach().cpu())
    return report


def run_sbc_diagnostics(
    posterior,
    prior,
    simulator,
    param_names: Sequence[str],
    draws: int,
    posterior_samples: int,
) -> Dict[str, Any]:
    ranks = np.zeros((int(draws), len(param_names)), dtype=np.int64)
    valid_draws = 0
    for d in range(int(draws)):
        theta_true = prior.sample((1,)).reshape(-1)
        x_true = simulator(theta_true).reshape(-1)
        if not bool(torch.isfinite(x_true).all()):
            continue
        try:
            post_x = posterior.set_default_x(x_true)
            samples_x = post_x.sample((int(posterior_samples),)).detach().cpu()
        except Exception:
            continue
        theta_true_np = theta_true.detach().cpu().numpy()
        samples_np = samples_x.numpy()
        ranks[valid_draws, :] = np.sum(samples_np < theta_true_np.reshape(1, -1), axis=0)
        valid_draws += 1

    if valid_draws == 0:
        return {
            "num_draws_requested": int(draws),
            "num_draws_valid": 0,
            "warning": "No valid SBC draws were produced.",
        }

    ranks = ranks[:valid_draws]
    expected_rank_mean = 0.5 * float(posterior_samples)
    rank_mean = np.mean(ranks, axis=0)
    return {
        "num_draws_requested": int(draws),
        "num_draws_valid": int(valid_draws),
        "posterior_samples_per_draw": int(posterior_samples),
        "param_names": list(param_names),
        "ranks": ranks.tolist(),
        "rank_mean": rank_mean.astype(np.float64).tolist(),
        "rank_mean_abs_error": np.abs(rank_mean - expected_rank_mean).astype(np.float64).tolist(),
    }


def run_stage_b_snpe(
    segments_csv: Path,
    checkpoint_path: Path,
    outdir: Path,
    num_simulations: int = 4000,
    max_segments: int = 128,
    max_phase_parameters: int = 6,
    num_posterior_samples: int = 2000,
    ppc_samples: int = 256,
    density_estimator: str = "maf",
    normalize_observation: bool = True,
    include_initial_conditions: bool = True,
    include_phase_context: bool = True,
    include_rate_features: bool = True,
    drop_parameter_invariant_context: bool = True,
    normalization_eps: float = 1.0e-6,
    calibration_subset_segments: int = 128,
    run_sbc: bool = False,
    sbc_draws: int = 64,
    sbc_posterior_samples: int = 256,
    device: str | torch.device | None = None,
    effective_mode: bool = True,
    anchor_from_stage_a: bool = True,
    mixed_precision: bool = False,
):
    try:
        from sbi.inference import SNPE, simulate_for_sbi
        from sbi.utils import BoxUniform
    except Exception as exc:
        raise RuntimeError(
            "Stage B requires the `sbi` package. Install with `pip install sbi`."
        ) from exc

    device_obj = _resolve_runtime_device(device)
    outdir.mkdir(parents=True, exist_ok=True)
    seg_df = sanitize_segments_for_training(_ensure_segment_rate_columns(pd.read_csv(segments_csv)))
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    subset_n = int(min(len(seg_df), int(max_segments), int(calibration_subset_segments)))
    subset_df = _select_stage_b_subset(seg_df, max_segments=max(1, subset_n))
    x_o_raw, feature_names, subset_df, excluded_context_features = build_stage_b_observation_summary(
        subset_df,
        max_segments=max(1, subset_n),
        include_initial_conditions=include_initial_conditions,
        include_phase_context=include_phase_context,
        include_rate_features=include_rate_features,
        drop_parameter_invariant_context=drop_parameter_invariant_context,
    )
    num_segments = int(len(subset_df))
    num_features = int(len(feature_names))
    if excluded_context_features:
        logging.info(
            "Stage B excluded parameter-invariant context features from the SBI summary: %s",
            ", ".join(excluded_context_features),
        )
    if not bool(np.isfinite(x_o_raw).all()):
        bad_count = int(np.size(x_o_raw) - np.count_nonzero(np.isfinite(x_o_raw)))
        raise RuntimeError(f"Stage B observed summary contains {bad_count} non-finite values before normalization.")
    norm_stats: Optional[Dict[str, List[float]]] = None
    if bool(normalize_observation):
        norm_stats = _normalization_stats_from_summary(
            x_o_raw,
            num_segments=num_segments,
            num_features=num_features,
            eps=float(normalization_eps),
        )
    x_o_vec = _apply_summary_normalization(x_o_raw, num_segments=num_segments, stats=norm_stats)
    if not bool(np.isfinite(x_o_vec).all()):
        bad_count = int(np.size(x_o_vec) - np.count_nonzero(np.isfinite(x_o_vec)))
        raise RuntimeError(f"Stage B normalized observed summary contains {bad_count} non-finite values.")

    simulator, prior_info, _ = make_stage_b_simulator(
        subset_df,
        ckpt,
        checkpoint_path=checkpoint_path,
        max_phase_parameters=max_phase_parameters,
        include_initial_conditions=include_initial_conditions,
        include_phase_context=include_phase_context,
        include_rate_features=include_rate_features,
        drop_parameter_invariant_context=drop_parameter_invariant_context,
        normalization_stats=norm_stats,
        device=device_obj,
        effective_mode=effective_mode,
        anchor_from_stage_a=anchor_from_stage_a,
    )

    low = torch.tensor(prior_info["low"], dtype=torch.float32, device=device_obj)
    high = torch.tensor(prior_info["high"], dtype=torch.float32, device=device_obj)
    prior = BoxUniform(low=low, high=high)
    x_o = torch.tensor(x_o_vec, dtype=torch.float32, device=device_obj)

    test_theta = prior.sample((2,))
    test_x0 = simulator(test_theta[0]).reshape(-1)
    test_x1 = simulator(test_theta[1]).reshape(-1)
    if test_x0.numel() != test_x1.numel() or test_x0.numel() != prior_info["obs_dim"]:
        raise RuntimeError(
            f"Simulator output dimension is not fixed. Expected {prior_info['obs_dim']}, "
            f"got {test_x0.numel()} and {test_x1.numel()}."
        )

    # Mixed-precision: wrap simulator with autocast for faster GPU simulations
    _use_amp_b = bool(mixed_precision) and device_obj.type == "cuda"
    if _use_amp_b:
        _raw_simulator = simulator
        def simulator(theta: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                return _raw_simulator(theta)

    theta, x = simulate_for_sbi(simulator, prior, num_simulations=int(num_simulations))
    if x.ndim == 1:
        if x.numel() % theta.shape[0] != 0:
            raise RuntimeError(
                f"Unexpected simulator output size: {x.numel()} for theta batch {theta.shape[0]}"
            )
        x = x.reshape(theta.shape[0], -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    theta = theta.to(device=device_obj, dtype=torch.float32)
    x = x.to(device=device_obj, dtype=torch.float32)

    if x.shape[1] != prior_info["obs_dim"]:
        raise RuntimeError(f"Unexpected SBI observation dim: {x.shape[1]}, expected {prior_info['obs_dim']}")

    finite_mask = torch.isfinite(theta).all(dim=1) & torch.isfinite(x).all(dim=1)
    finite_mask = finite_mask & (torch.abs(x) < 1.0e8).all(dim=1)
    dropped = int((~finite_mask).sum().item())
    if dropped > 0:
        print(f"Dropping {dropped} invalid simulations before SNPE training.")
    theta = theta[finite_mask]
    x = x[finite_mask]
    if theta.shape[0] < max(200, int(0.25 * int(num_simulations))):
        raise RuntimeError(
            f"Too few valid simulations after filtering: {theta.shape[0]} / {num_simulations}"
        )

    in_prior = ((theta >= low.unsqueeze(0)) & (theta <= high.unsqueeze(0))).all(dim=1)
    if not bool(torch.all(in_prior)):
        raise RuntimeError("Simulated theta values left prior support unexpectedly.")

    density_estimator_name = str(density_estimator)
    try:
        inference = SNPE(prior=prior, density_estimator=density_estimator_name, device=str(device_obj))
    except TypeError:
        inference = SNPE(prior=prior, density_estimator=density_estimator_name)
    density_estimator_fit = inference.append_simulations(theta, x).train()
    try:
        posterior = inference.build_posterior(density_estimator_fit, sample_with="mcmc").set_default_x(x_o)
    except TypeError:
        posterior = inference.build_posterior(density_estimator_fit).set_default_x(x_o)

    samples = posterior.sample((int(num_posterior_samples),))
    samples_np = samples.detach().cpu().numpy()
    np.save(outdir / "stage_b_posterior_samples.npy", samples_np)
    if samples_np.ndim == 2 and samples_np.shape[1] >= 2:
        corr = np.corrcoef(samples_np.T)
        np.save(outdir / "stage_b_posterior_correlation.npy", corr.astype(np.float64))

    posterior_summary = summarize_posterior_samples(samples_np, prior_info["param_names"])
    with (outdir / "stage_b_posterior_summary.json").open("w", encoding="utf-8") as f:
        json.dump(posterior_summary, f, indent=2)

    ppc = posterior_predictive_check(
        simulator=simulator,
        posterior_samples=samples,
        x_o=x_o,
        num_segments=int(prior_info["num_segments"]),
        num_features=int(prior_info["num_features"]),
        feature_names=list(prior_info["feature_names"]),
        n_samples=int(ppc_samples),
        device=device_obj,
    )
    with (outdir / "stage_b_posterior_predictive_check.json").open("w", encoding="utf-8") as f:
        json.dump(ppc, f, indent=2)

    if bool(run_sbc):
        sbc = run_sbc_diagnostics(
            posterior=posterior,
            prior=prior,
            simulator=simulator,
            param_names=prior_info["param_names"],
            draws=int(sbc_draws),
            posterior_samples=int(sbc_posterior_samples),
        )
        with (outdir / "stage_b_sbc_summary.json").open("w", encoding="utf-8") as f:
            json.dump(sbc, f, indent=2)

    torch.save(
        {
            "density_estimator_state_dict": density_estimator_fit.state_dict(),
            "prior_info": prior_info,
            "x_o": x_o.detach().cpu(),
            "normalization_stats": norm_stats,
        },
        outdir / "stage_b_density_estimator.pt",
    )

    prior_info["num_simulations_requested"] = int(num_simulations)
    prior_info["num_simulations_used"] = int(theta.shape[0])
    prior_info["num_posterior_samples"] = int(num_posterior_samples)
    prior_info["ppc_samples"] = int(ppc_samples)
    prior_info["density_estimator"] = density_estimator_name
    prior_info["normalize_observation"] = bool(normalize_observation)
    prior_info["drop_parameter_invariant_context"] = bool(drop_parameter_invariant_context)
    prior_info["normalization_eps"] = float(normalization_eps)
    prior_info["calibration_subset_segments"] = int(subset_n)
    prior_info["run_sbc"] = bool(run_sbc)
    prior_info["device"] = str(device_obj)
    prior_info["schema_version"] = RUN_SCHEMA_VERSION
    with (outdir / "stage_b_prior_info.json").open("w", encoding="utf-8") as f:
        json.dump(prior_info, f, indent=2)

    if norm_stats is not None:
        with (outdir / "stage_b_normalization_stats.json").open("w", encoding="utf-8") as f:
            json.dump(norm_stats, f, indent=2)

    with (outdir / "stage_b_observation_features.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "num_features": int(num_features),
                "num_segments": int(num_segments),
                "drop_parameter_invariant_context": bool(drop_parameter_invariant_context),
                "excluded_parameter_invariant_context_features": excluded_context_features,
            },
            f,
            indent=2,
        )

    print(f"Saved Stage B posterior samples to {(outdir / 'stage_b_posterior_samples.npy').resolve()}")


def run_identifiability_diagnostics(
    segments_csv: Path,
    checkpoint_path: Path,
    outdir: Path,
    max_segments: int = 128,
    max_phase_parameters: int = 6,
    device: str | torch.device | None = None,
    effective_mode: bool = True,
    anchor_from_stage_a: bool = True,
    drop_parameter_invariant_context: bool = True,
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device_obj = _resolve_runtime_device(device)
    seg_df = sanitize_segments_for_training(_ensure_segment_rate_columns(pd.read_csv(segments_csv)))
    subset_df = _select_stage_b_subset(seg_df, max_segments=max_segments)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    simulator, prior_info, _ = make_stage_b_simulator(
        subset_df,
        ckpt,
        checkpoint_path=checkpoint_path,
        max_phase_parameters=max_phase_parameters,
        include_initial_conditions=True,
        include_phase_context=True,
        include_rate_features=True,
        drop_parameter_invariant_context=drop_parameter_invariant_context,
        normalization_stats=None,
        device=device_obj,
        effective_mode=effective_mode,
        anchor_from_stage_a=anchor_from_stage_a,
    )

    theta0 = torch.tensor(prior_info.get("anchor", []), dtype=torch.float64, device=device_obj)
    if theta0.numel() == 0:
        raise RuntimeError("Identifiability diagnostics could not build an anchor parameter vector.")

    def sim_flat(theta: torch.Tensor) -> torch.Tensor:
        return simulator(theta.float()).reshape(-1).double()

    try:
        j_mat = jacrev(sim_flat)(theta0)
    except Exception:
        j_mat = jacfwd(sim_flat)(theta0)

    if j_mat.ndim != 2:
        j_mat = j_mat.reshape(int(j_mat.shape[0]), -1)
    j_np = j_mat.detach().cpu().numpy().astype(np.float64)
    fisher = j_np.T @ j_np
    # Symmetrize to eliminate numerical asymmetry before eigendecomposition.
    fisher = 0.5 * (fisher + fisher.T)
    svals = np.linalg.svd(j_np, full_matrices=False, compute_uv=False)
    svals = np.asarray(svals, dtype=np.float64)

    max_sv = float(np.max(svals)) if svals.size > 0 else 0.0
    tol = max(1.0e-12, max_sv * 1.0e-6)
    rank = int(np.sum(svals > tol)) if svals.size > 0 else 0
    cond_number = float((svals[0] / max(svals[-1], 1.0e-12)) if svals.size > 1 else 1.0)

    fisher_diag = np.diag(fisher).astype(np.float64)
    sens_norm = fisher_diag / max(float(np.max(fisher_diag)) if fisher_diag.size > 0 else 1.0, 1.0e-12)

    param_names = list(prior_info.get("param_names", [f"theta_{i}" for i in range(theta0.numel())]))
    if len(param_names) != int(theta0.numel()):
        param_names = [f"theta_{i}" for i in range(theta0.numel())]

    warnings: List[str] = []
    if rank < int(theta0.numel()):
        warnings.append(
            f"Rank deficiency detected: rank={rank} < num_params={int(theta0.numel())}."
        )
    if cond_number > 1.0e8:
        warnings.append(f"High Jacobian condition number: {cond_number:.3e}")

    weak_params = [param_names[i] for i, v in enumerate(sens_norm.tolist()) if float(v) < 1.0e-4]
    if weak_params:
        warnings.append(f"Weakly identifiable parameters: {weak_params[:24]}")

    phase_summary: Dict[str, Dict[str, float]] = {}
    for phase in prior_info.get("active_phases", []):
        idxs = [i for i, n in enumerate(param_names) if n.endswith(f"__{phase}")]
        if not idxs:
            continue
        sub_fisher = fisher[np.ix_(idxs, idxs)]
        sub_fisher = 0.5 * (sub_fisher + sub_fisher.T)
        eigvals = np.linalg.eigvalsh(sub_fisher)
        eigvals = np.asarray(eigvals, dtype=np.float64)
        min_e = float(np.min(eigvals)) if eigvals.size > 0 else 0.0
        max_e = float(np.max(eigvals)) if eigvals.size > 0 else 0.0
        phase_summary[str(phase)] = {
            "num_params": int(len(idxs)),
            "min_fisher_eig": min_e,
            "max_fisher_eig": max_e,
            "condition": float(max_e / max(min_e, 1.0e-12)) if max_e > 0 else float("inf"),
        }

    sv_df = pd.DataFrame({
        "singular_value": svals,
        "index": np.arange(len(svals), dtype=np.int64),
    })
    sv_csv = outdir / "identifiability_singular_values.csv"
    sv_df.to_csv(sv_csv, index=False)

    sens_df = pd.DataFrame({
        "parameter": param_names,
        "fisher_diag": fisher_diag,
        "normalized_sensitivity": sens_norm,
    })
    sens_csv = outdir / "identifiability_parameter_sensitivity.csv"
    sens_df.to_csv(sens_csv, index=False)

    fisher_npy = outdir / "identifiability_fisher_matrix.npy"
    np.save(fisher_npy, fisher.astype(np.float64))

    # Global Fisher eigenvalues (symmetric, so use eigvalsh).
    fisher_eigvals = np.linalg.eigvalsh(fisher)
    fisher_eigvals = np.sort(fisher_eigvals)[::-1]  # descending

    report = {
        "schema_version": RUN_SCHEMA_VERSION,
        "num_segments": int(prior_info.get("num_segments", 0)),
        "num_features": int(prior_info.get("num_features", 0)),
        "num_params": int(theta0.numel()),
        "rank": int(rank),
        "condition_number": cond_number,
        "drop_parameter_invariant_context": bool(drop_parameter_invariant_context),
        "excluded_parameter_invariant_context_features": list(prior_info.get("excluded_parameter_invariant_context_features", [])),
        "sv_max": max_sv,
        "sv_min": float(np.min(svals)) if svals.size > 0 else 0.0,
        "fisher_eig_max": float(fisher_eigvals[0]) if fisher_eigvals.size > 0 else 0.0,
        "fisher_eig_min": float(fisher_eigvals[-1]) if fisher_eigvals.size > 0 else 0.0,
        "warnings": warnings,
        "phase_summary": phase_summary,
        "outputs": {
            "singular_values_csv": str(sv_csv),
            "parameter_sensitivity_csv": str(sens_csv),
            "fisher_matrix_npy": str(fisher_npy),
        },
    }
    with (outdir / "identifiability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


# -----------------------------------------------------------------------------
# Diagnostics and plotting
# -----------------------------------------------------------------------------
def _save_figure(fig, out_png: Path, dpi: int, save_pdf: bool):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return {}
    return obj


def collect_stage_a_predictions(segments_csv: Path, checkpoint_path: Path, device: str = "cpu") -> pd.DataFrame:
    seg_df = sanitize_segments_for_training(_ensure_segment_rate_columns(pd.read_csv(segments_csv)))
    model, sat_to_idx, phase_to_idx, _ = load_stage_a_model_from_checkpoint(checkpoint_path, device=device)
    batch = build_stage_a_batch_from_segments(seg_df, sat_to_idx, phase_to_idx, device=device)

    with torch.no_grad():
        out = model(batch)

    pred_df = seg_df.copy()
    pred_df["a1_pred_km"] = out["a1_pred"].detach().cpu().numpy()
    pred_df["da_pred_km"] = out["da_pred_km"].detach().cpu().numpy()
    pred_df["draan_pred_rad"] = out["draan_pred"].detach().cpu().numpy()
    pred_df["dlam_pred_rad"] = out["dlam_pred"].detach().cpu().numpy()
    pred_df["da_rate_pred_km_day"] = out["da_rate_pred_km_day"].detach().cpu().numpy()
    pred_df["draan_rate_pred_rad_day"] = out["draan_rate_pred_rad_day"].detach().cpu().numpy()
    pred_df["dlam_rate_pred_rad_day"] = out["dlam_rate_pred_rad_day"].detach().cpu().numpy()
    pred_df["thrust_pred_N"] = out["thrust_N"].detach().cpu().numpy()
    pred_df["thrust_kr_N"] = out["thrust_kr_N"].detach().cpu().numpy()
    pred_df["isp_pred_s"] = out["isp_s"].detach().cpu().numpy()
    pred_df["isp_kr_s"] = out["isp_kr_s"].detach().cpu().numpy()
    pred_df["vd_V"] = out["phase_vd_V"].detach().cpu().numpy()
    pred_df["vc_V"] = out["phase_vc_V"].detach().cpu().numpy()
    pred_df["vb_V"] = out["phase_vb_effective_V"].detach().cpu().numpy()
    pred_df["ib_A"] = out["phase_ib_A"].detach().cpu().numpy()
    pred_df["eta_total_phase"] = out["eta_total_phase"].detach().cpu().numpy()
    pred_df["eta_factorized_phase"] = out["eta_factorized_phase"].detach().cpu().numpy()
    pred_df["eta_b"] = out["phase_eta_b"].detach().cpu().numpy()
    pred_df["eta_v"] = out["phase_eta_v"].detach().cpu().numpy()
    pred_df["eta_m"] = out["phase_eta_m"].detach().cpu().numpy()
    pred_df["eta_o"] = out["phase_eta_o"].detach().cpu().numpy()
    pred_df["gamma"] = out["phase_gamma"].detach().cpu().numpy()
    pred_df["nu_a"] = out["phase_nu_a"].detach().cpu().numpy()
    pred_df["mdot_a_kg_s"] = out["phase_mdot_a_kg_s"].detach().cpu().numpy()
    pred_df["mdot_c_kg_s"] = out["phase_mdot_c_kg_s"].detach().cpu().numpy()
    pred_df["power_in_W"] = out["power_in_W"].detach().cpu().numpy()
    pred_df["power_nominal_W"] = out["power_nominal_W"].detach().cpu().numpy()
    pred_df["mass_end_kg"] = out["mass_end_kg"].detach().cpu().numpy()
    pred_df["te_eV"] = out["te_eV"].detach().cpu().numpy()
    pred_df["sigma_iv_kr_m3_s"] = out["sigma_iv_kr_m3_s"].detach().cpu().numpy()
    pred_df["lambda_i_m"] = out["lambda_i_m"].detach().cpu().numpy()
    pred_df["ionization_ratio"] = out["ionization_ratio"].detach().cpu().numpy()

    pred_df["a1_resid_km"] = pred_df["a1_pred_km"] - pred_df["a1_km"]
    pred_df["da_resid_km"] = pred_df["da_pred_km"] - pred_df["da_obs_km"]
    pred_df["draan_resid_rad"] = np.arctan2(
        np.sin(pred_df["draan_pred_rad"] - pred_df["draan_obs_rad"]),
        np.cos(pred_df["draan_pred_rad"] - pred_df["draan_obs_rad"]),
    )
    pred_df["dlam_resid_rad"] = np.arctan2(
        np.sin(pred_df["dlam_pred_rad"] - pred_df["dlam_obs_rad"]),
        np.cos(pred_df["dlam_pred_rad"] - pred_df["dlam_obs_rad"]),
    )

    for tau in COLLOCATION_TAU_POINTS:
        tau_label = int(round(100.0 * float(tau)))
        pred_a_key = f"a_tau{tau_label}_pred_km"
        pred_raan_key = f"raan_tau{tau_label}_pred_rad"
        pred_lam_key = f"lam_tau{tau_label}_pred_rad"
        tgt_a_col = f"a_tau{tau_label}_km"
        tgt_raan_col = f"raan_tau{tau_label}_rad"
        tgt_lam_col = f"lam_tau{tau_label}_rad"
        if pred_a_key in out:
            pred_df[pred_a_key] = out[pred_a_key].detach().cpu().numpy()
        if pred_raan_key in out:
            pred_df[pred_raan_key] = out[pred_raan_key].detach().cpu().numpy()
        if pred_lam_key in out:
            pred_df[pred_lam_key] = out[pred_lam_key].detach().cpu().numpy()
        if tgt_a_col in pred_df.columns and pred_a_key in pred_df.columns:
            pred_df[f"a_tau{tau_label}_resid_km"] = pred_df[pred_a_key] - pred_df[tgt_a_col]
        if tgt_raan_col in pred_df.columns and pred_raan_key in pred_df.columns:
            pred_df[f"raan_tau{tau_label}_resid_rad"] = np.arctan2(
                np.sin(pred_df[pred_raan_key] - pred_df[tgt_raan_col]),
                np.cos(pred_df[pred_raan_key] - pred_df[tgt_raan_col]),
            )
        if tgt_lam_col in pred_df.columns and pred_lam_key in pred_df.columns:
            pred_df[f"lam_tau{tau_label}_resid_rad"] = np.arctan2(
                np.sin(pred_df[pred_lam_key] - pred_df[tgt_lam_col]),
                np.cos(pred_df[pred_lam_key] - pred_df[tgt_lam_col]),
            )
    return pred_df


def generate_data_quality_plots(seg_df: pd.DataFrame, out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or (not bool(plot_cfg.data_quality)):
        return written
    if seg_df.empty:
        return written

    dt_hours = seg_df["dt_s"].to_numpy(dtype=np.float64) / 3600.0
    da_km = seg_df["da_obs_km"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(dt_hours[np.isfinite(dt_hours)], bins=40, color="#3b82f6", alpha=0.85)
    axes[0].set_title("Segment Duration")
    axes[0].set_xlabel("Hours")
    axes[0].set_ylabel("Count")

    axes[1].hist(da_km[np.isfinite(da_km)], bins=40, color="#f97316", alpha=0.85)
    axes[1].set_title(r"Observed $\Delta$ a")
    axes[1].set_xlabel(r"$\Delta$ a (km)")
    axes[1].set_ylabel("Count")

    p1 = out_dir / "segment_duration_and_da_hist.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))

    if "phase" in seg_df.columns:
        phase_counts = seg_df["phase"].astype(str).value_counts().head(24)
        phase_labels = [PHASE_DISPLAY_NAMES.get(str(phase), str(phase)) for phase in phase_counts.index]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(phase_labels, phase_counts.to_numpy(), color="#22c55e", alpha=0.85)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=0)
        p2 = out_dir / "phase_counts.png"
        _save_figure(fig, p2, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
        written.append(str(p2))

    return written


def generate_stage_a_training_plots(history_df: pd.DataFrame, out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or (not bool(plot_cfg.training)):
        return written
    if history_df.empty or "epoch" not in history_df.columns:
        return written

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["loss_total", "loss_a", "loss_da", "loss_raan", "loss_lam", "loss_prior"]:
        if col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[col], label=col, linewidth=1.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best", ncol=2, fontsize=12)
    p1 = out_dir / "loss_curves.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))

    fig, ax = plt.subplots(figsize=(10, 5))
    if "loss_total" in history_df.columns:
        ax.plot(history_df["epoch"], np.log10(np.maximum(history_df["loss_total"].to_numpy(dtype=np.float64), 1.0e-12)), color="#ef4444")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log10(Loss)")
    p2 = out_dir / "loss_total_log10.png"
    _save_figure(fig, p2, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p2))

    physics_cols = [c for c in ["loss_hall", "loss_chemistry_penalty", "loss_feasibility", "loss_collocation"] if c in history_df.columns]
    if physics_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in physics_cols:
            ax.plot(history_df["epoch"], history_df[col], label=col, linewidth=1.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        p3 = out_dir / "physics_residual_losses.png"
        _save_figure(fig, p3, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
        written.append(str(p3))

    # Per-term log-scale panels for all loss components.
    all_loss_cols = [c for c in history_df.columns if c.startswith("loss_") and c != "loss_total"]
    if len(all_loss_cols) >= 2:
        n_cols = min(4, len(all_loss_cols))
        n_rows = int(np.ceil(len(all_loss_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), squeeze=False)
        for idx, col in enumerate(all_loss_cols):
            ax = axes[idx // n_cols, idx % n_cols]
            vals = history_df[col].to_numpy(dtype=np.float64)
            vals = np.maximum(vals, 1.0e-16)
            ax.semilogy(history_df["epoch"], vals, linewidth=1.4, color="#2563eb")
            ax.set_title(col, fontsize=16)
            ax.set_xlabel("Epoch", fontsize=12)
        # Hide unused axes.
        for idx in range(len(all_loss_cols), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)
        fig.tight_layout()
        p4 = out_dir / "loss_per_term_log.png"
        _save_figure(fig, p4, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
        written.append(str(p4))

    return written


def generate_stage_a_fit_plots(pred_df: pd.DataFrame, out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or (not bool(plot_cfg.fit)):
        return written
    if pred_df.empty:
        return written

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Use wrapped angle differences for angular channels so both axes share the same convention.
    pairs = [
        ("a1_km", "a1_pred_km", "a1", False),
        ("draan_obs_rad", "draan_pred_rad", r"$\Delta$ RAAN", True),
        ("dlam_obs_rad", "dlam_pred_rad", r"$\Delta$ $\lambda$", True),
    ]
    for ax, (obs_col, pred_col, label, is_angle) in zip(axes, pairs):
        x = pred_df[obs_col].to_numpy(dtype=np.float64)
        y = pred_df[pred_col].to_numpy(dtype=np.float64)
        if is_angle:
            x = wrap_to_pi(x)
            y = wrap_to_pi(y)
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], s=12, alpha=0.5, color="#2563eb")
        if np.any(good):
            lo = float(np.nanmin(np.concatenate([x[good], y[good]])))
            hi = float(np.nanmax(np.concatenate([x[good], y[good]])))
            ax.plot([lo, hi], [lo, hi], color="#ef4444", linestyle="--", linewidth=1.2)
        ax.set_title(f"Observed vs Predicted: {label}")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
    p1 = out_dir / "observed_vs_predicted.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    resid_specs = [
        ("a1_resid_km", "Residual a1", "#f97316"),
        ("draan_resid_rad", r"Residual $\Delta$ RAAN", "#22c55e"),
        ("dlam_resid_rad", r"Residual $\Delta$ $\lambda$", "#8b5cf6"),
    ]
    for ax, (col, title, color) in zip(axes, resid_specs):
        vals = pred_df[col].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=40, alpha=0.85, color=color)
        ax.set_title(title)
        ax.set_ylabel("Count")
    p2 = out_dir / "residual_histograms.png"
    _save_figure(fig, p2, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p2))
    return written


def generate_robust_residual_plots(pred_df: pd.DataFrame, out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or pred_df.empty:
        return written

    resid_cols = [c for c in ["a1_resid_km", "da_resid_km", "draan_resid_rad", "dlam_resid_rad"] if c in pred_df.columns]
    if not resid_cols:
        return written

    fig, axes = plt.subplots(1, len(resid_cols), figsize=(4.2 * len(resid_cols), 4.0))
    if len(resid_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, resid_cols):
        vals = pred_df[col].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=40, alpha=0.85, color="#0ea5e9")
        ax.set_title(f"Robust Residual: {col}")
    p1 = out_dir / "robust_residual_histograms.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))
    return written


def generate_segment_internal_overlay_plots(pred_df: pd.DataFrame, out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or pred_df.empty:
        return written

    tau_labels = [int(round(100.0 * float(t))) for t in COLLOCATION_TAU_POINTS]
    needed = [
        f"a_tau{tau_labels[0]}_pred_km",
        f"a_tau{tau_labels[1]}_pred_km",
        f"a_tau{tau_labels[2]}_pred_km",
    ]
    if not all(c in pred_df.columns for c in needed):
        return written

    sample_df = pred_df.head(min(24, len(pred_df))).copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    tau_axis = [0.0] + [float(t) for t in COLLOCATION_TAU_POINTS] + [1.0]
    for _, row in sample_df.iterrows():
        obs_track = [row.get("a0_km", np.nan)]
        pred_track = [row.get("a0_km", np.nan)]
        for tau_label in tau_labels:
            # Restore masked (zero-filled) collocation targets to NaN for plotting.
            mask_val = row.get(f"has_tau{tau_label}", 1)
            if mask_val is not None and float(mask_val) < 0.5:
                obs_track.append(np.nan)
            else:
                obs_track.append(row.get(f"a_tau{tau_label}_km", np.nan))
            pred_track.append(row.get(f"a_tau{tau_label}_pred_km", np.nan))
        obs_track.append(row.get("a1_km", np.nan))
        pred_track.append(row.get("a1_pred_km", np.nan))
        if np.all(np.isfinite(obs_track)):
            ax.plot(tau_axis, obs_track, color="#94a3b8", alpha=0.35, linewidth=1.0)
        if np.all(np.isfinite(pred_track)):
            ax.plot(tau_axis, pred_track, color="#ef4444", alpha=0.45, linewidth=1.0)
    ax.set_title("Segment-Internal a(tau) Overlays (observed gray, predicted red)")
    ax.set_xlabel("normalized segment time tau")
    ax.set_ylabel("a (km)")
    p1 = out_dir / "segment_internal_a_overlays.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))
    return written


def generate_chemistry_closure_plots(pred_df: pd.DataFrame, out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or pred_df.empty:
        return written
    needed = ["te_eV", "sigma_iv_kr_m3_s", "ionization_ratio"]
    if not all(c in pred_df.columns for c in needed):
        return written

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = pred_df["te_eV"].to_numpy(dtype=np.float64)
    y = pred_df["sigma_iv_kr_m3_s"].to_numpy(dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    axes[0].scatter(x[good], y[good], s=10, alpha=0.4, color="#2563eb")
    axes[0].set_xlabel("T_e (eV)")
    axes[0].set_ylabel("<sigma_i v>_Kr (m^3/s)")
    axes[0].set_title("Krypton Ionization (Surrogate — NOT validated physics)")

    r = pred_df["ionization_ratio"].to_numpy(dtype=np.float64)
    r = r[np.isfinite(r)]
    axes[1].hist(r, bins=40, color="#22c55e", alpha=0.85)
    axes[1].axvline(0.25, color="#ef4444", linestyle="--", linewidth=1.2)
    axes[1].axvline(3.0, color="#ef4444", linestyle="--", linewidth=1.2)
    axes[1].set_title("Ionization-Length Ratio λ_i / L (Surrogate)")
    axes[1].set_xlabel("ratio")
    p1 = out_dir / "chemistry_surrogate_diagnostics.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))
    return written


def generate_identifiability_plots(ident_dir: Path, out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or (not bool(plot_cfg.sensitivity)):
        return written

    sv_csv = ident_dir / "identifiability_singular_values.csv"
    fisher_npy = ident_dir / "identifiability_fisher_matrix.npy"
    if not sv_csv.exists():
        return written

    sv_df = pd.read_csv(sv_csv)
    if "singular_value" in sv_df.columns and (not sv_df.empty):
        vals = sv_df["singular_value"].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.semilogy(np.arange(len(vals)), np.maximum(vals, 1.0e-16), color="#ef4444", linewidth=1.8)
            ax.set_title("Identifiability Jacobian Singular Values")
            ax.set_xlabel("index")
            ax.set_ylabel("singular value (log)")
            p1 = out_dir / "identifiability_singular_values.png"
            _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
            written.append(str(p1))

    if fisher_npy.exists():
        fisher = np.load(fisher_npy)
        if fisher.ndim == 2 and fisher.shape[0] == fisher.shape[1]:
            corr = np.zeros_like(fisher, dtype=np.float64)
            d = np.sqrt(np.clip(np.diag(fisher), 1.0e-16, np.inf))
            for i in range(fisher.shape[0]):
                for j in range(fisher.shape[1]):
                    corr[i, j] = fisher[i, j] / (d[i] * d[j])
            corr = np.clip(corr, -1.0, 1.0)
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            ax.set_title("Fisher-Style Parameter Correlation")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            p2 = out_dir / "identifiability_correlation_heatmap.png"
            _save_figure(fig, p2, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
            written.append(str(p2))
    return written


def generate_timing_sensitivity_plots(df: pd.DataFrame, out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or (not bool(plot_cfg.sensitivity)) or df.empty:
        return written
    if "timing_shift_hours" not in df.columns:
        return written

    fig, ax = plt.subplots(figsize=(10, 3.75))
    ax.plot(df["timing_shift_hours"], df["rmse_a_km"], marker="o", label="RMSE a (km)")
    if "rmse_draan_rad" in df.columns:
        ax.plot(df["timing_shift_hours"], df["rmse_draan_rad"], marker="s", label=r"RMSE $\Delta$ RAAN (rad)")
    if "rmse_dlambda_rad" in df.columns:
        ax.plot(df["timing_shift_hours"], df["rmse_dlambda_rad"], marker="^", label=r"RMSE $\Delta$ $\lambda$ (rad)")
    ax.set_xlabel("Timing Shift (hours)")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best")
    p1 = out_dir / "timing_sensitivity_rmse.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))
    return written


def generate_ablation_plots(df: pd.DataFrame, out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or (not bool(plot_cfg.ablations)) or df.empty:
        return written
    if "variant" not in df.columns:
        return written

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["variant"].astype(str).tolist(), df["rmse_a_km"].to_numpy(dtype=np.float64), color="#0ea5e9", alpha=0.85)
    ax.set_title("Force-Model Ablation: RMSE a")
    ax.set_ylabel("rmse_a_km")
    ax.tick_params(axis="x", labelrotation=45)
    p1 = out_dir / "force_model_ablation_rmse_a.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))
    return written


def generate_synthetic_recovery_plots(report: Dict[str, Any], out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or (not bool(plot_cfg.synthetic)):
        return written
    rec = dict(report.get("recovery", {})) if isinstance(report, dict) else {}
    keys = [
        ("mass_kg", rec.get("mass_kg_true", np.nan), rec.get("mass_kg_recovered", np.nan)),
        ("isp_s", rec.get("isp_s_true", np.nan), rec.get("isp_s_recovered", np.nan)),
        ("eta_total", rec.get("eta_total_true", np.nan), rec.get("eta_total_recovered", np.nan)),
    ]
    keys = [k for k in keys if np.isfinite(k[1]) and np.isfinite(k[2])]
    if not keys:
        return written

    fig, ax = plt.subplots(figsize=(6, 6))
    true_vals = np.array([k[1] for k in keys], dtype=np.float64)
    rec_vals = np.array([k[2] for k in keys], dtype=np.float64)
    ax.scatter(true_vals, rec_vals, color="#ef4444", s=40)
    lo = float(np.min(np.concatenate([true_vals, rec_vals])))
    hi = float(np.max(np.concatenate([true_vals, rec_vals])))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#334155", linewidth=1.2)
    for name, tv, rv in keys:
        ax.annotate(name, (tv, rv), textcoords="offset points", xytext=(4, 4), fontsize=12)
    ax.set_xlabel("true")
    ax.set_ylabel("recovered")
    ax.set_title("Synthetic Recovery: Truth vs Recovered")
    p1 = out_dir / "synthetic_truth_vs_recovered.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))
    return written


def generate_stage_a_parameter_plots(trace_df: pd.DataFrame, summary: Dict[str, Any], out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or (not bool(plot_cfg.parameters)):
        return written
    if trace_df.empty:
        return written

    fig, ax = plt.subplots(figsize=(11, 5))
    for col in [
        "mass_kg",
        "isp_s",
        "eta_total",
        "util_mass",
        "util_current",
        "util_voltage",
        "divergence_eff",
    ]:
        if col in trace_df.columns:
            ax.plot(trace_df["epoch"], trace_df[col], label=col, linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.legend(loc="best", ncol=3, fontsize=12)
    p1 = out_dir / "global_parameter_evolution.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))

    thrust_cols = [c for c in trace_df.columns if c.startswith("thrust_N__")]
    if thrust_cols:
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in thrust_cols[:12]:
            ax.plot(trace_df["epoch"], trace_df[col], linewidth=1.1, alpha=0.85, label=col.replace("thrust_N__", ""))
        ax.set_xlabel("epoch")
        ax.set_ylabel("thrust N")
        ax.legend(loc="best", ncol=3, fontsize=12)
        p2 = out_dir / "phase_thrust_evolution.png"
        _save_figure(fig, p2, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
        written.append(str(p2))

    phase_params = summary.get("phase_parameters", {}) if isinstance(summary, dict) else {}
    if isinstance(phase_params, dict) and phase_params:
        phases = list(phase_params.keys())[:24]
        phase_labels = [PHASE_DISPLAY_NAMES.get(str(phase), str(phase)) for phase in phases]
        thrust_vals = [float(phase_params[p].get("thrust_N", np.nan)) for p in phases]
        duty_vals = [float(phase_params[p].get("duty", np.nan)) for p in phases]

        fig, ax1 = plt.subplots(figsize=(12, 5))
        x = np.arange(len(phases))
        ax1.bar(x - 0.2, thrust_vals, width=0.4, color="#3b82f6", alpha=0.85, label="Thrust (N)")
        ax1.set_ylabel("Thrust (N)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(phase_labels, rotation=0, ha="center")

        ax2 = ax1.twinx()
        ax2.bar(x + 0.2, duty_vals, width=0.4, color="#22c55e", alpha=0.65, label="Duty")
        ax2.set_ylabel("Duty")
        p3 = out_dir / "final_phase_parameters.png"
        _save_figure(fig, p3, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
        written.append(str(p3))

        vd_vals = [float(phase_params[p].get("vd_V", np.nan)) for p in phases]
        vc_vals = [float(phase_params[p].get("vc_V", np.nan)) for p in phases]
        vb_vals = [float(phase_params[p].get("vb_effective_V", np.nan)) for p in phases]
        ib_vals = [float(phase_params[p].get("ib_A", np.nan)) for p in phases]
        gamma_vals = [float(phase_params[p].get("gamma", np.nan)) for p in phases]
        nu_vals = [float(phase_params[p].get("nu_a", np.nan)) for p in phases]
        mdot_a_vals = [float(phase_params[p].get("mdot_a_kg_s", np.nan)) for p in phases]
        mdot_c_vals = [float(phase_params[p].get("mdot_c_kg_s", np.nan)) for p in phases]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        x = np.arange(len(phases))
        axes[0, 0].plot(x, vd_vals, label="Vd", color="#2563eb")
        axes[0, 0].plot(x, vc_vals, label="Vc", color="#f97316")
        axes[0, 0].plot(x, vb_vals, label="Vb", color="#16a34a")
        axes[0, 0].set_title("Discharge/Cathode/Beam Voltage by Phase")
        axes[0, 0].legend(fontsize=12)

        axes[0, 1].plot(x, ib_vals, label="Ib (A)", color="#7c3aed")
        axes[0, 1].plot(x, gamma_vals, label="gamma", color="#dc2626")
        axes[0, 1].plot(x, nu_vals, label="nu_a", color="#0891b2")
        axes[0, 1].set_title("Current and Transport by Phase")
        axes[0, 1].legend(fontsize=12)

        axes[1, 0].plot(x, mdot_a_vals, label="mdot_a", color="#0ea5e9")
        axes[1, 0].plot(x, mdot_c_vals, label="mdot_c", color="#22c55e")
        axes[1, 0].set_title("Mass Flow by Phase")
        axes[1, 0].legend(fontsize=12)

        axes[1, 1].bar(x, [float(phase_params[p].get("eta_b", np.nan)) for p in phases], alpha=0.65, label="eta_b")
        axes[1, 1].bar(x, [float(phase_params[p].get("eta_v", np.nan)) for p in phases], alpha=0.50, label="eta_v")
        axes[1, 1].bar(x, [float(phase_params[p].get("eta_m", np.nan)) for p in phases], alpha=0.50, label="eta_m")
        axes[1, 1].bar(x, [float(phase_params[p].get("eta_o", np.nan)) for p in phases], alpha=0.50, label="eta_o")
        axes[1, 1].set_title("Efficiency Factors by Phase")
        axes[1, 1].legend(fontsize=12)

        for ax in axes.ravel():
            ax.set_xticks(x)
            ax.set_xticklabels(phases, rotation=45, ha="right", fontsize=12)

        p4 = out_dir / "hall_latents_by_phase.png"
        _save_figure(fig, p4, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
        written.append(str(p4))

    return written


def generate_stage_b_plots(stage_b_dir: Path, out_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    written: List[str] = []
    if (not MATPLOTLIB_AVAILABLE) or (not bool(plot_cfg.enabled)) or (not bool(plot_cfg.stage_b)):
        return written

    samples_path = stage_b_dir / "stage_b_posterior_samples.npy"
    prior_info_path = stage_b_dir / "stage_b_prior_info.json"
    ppc_path = stage_b_dir / "stage_b_posterior_predictive_check.json"
    if (not samples_path.exists()) or (not prior_info_path.exists()):
        return written

    samples = np.load(samples_path)
    prior_info = _json_load(prior_info_path)
    param_names = prior_info.get("param_names", [f"theta_{i}" for i in range(samples.shape[1])])
    if not isinstance(param_names, list) or len(param_names) != samples.shape[1]:
        param_names = [f"theta_{i}" for i in range(samples.shape[1])]

    n_show = min(12, samples.shape[1])
    fig, axes = plt.subplots(nrows=n_show, ncols=1, figsize=(11, 2.2 * n_show))
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        vals = samples[:, i]
        ax.hist(vals[np.isfinite(vals)], bins=40, color="#7c3aed", alpha=0.85)
        ax.set_title(f"Posterior: {param_names[i]}")
    p1 = out_dir / "posterior_histograms_top12.png"
    _save_figure(fig, p1, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
    written.append(str(p1))

    if samples.shape[1] >= 4:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        pair_idx = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        for ax, (i, j) in zip(axes.ravel(), pair_idx):
            ax.scatter(samples[:, i], samples[:, j], s=8, alpha=0.3, color="#2563eb")
            ax.set_xlabel(param_names[i])
            ax.set_ylabel(param_names[j])
        fig.suptitle("Posterior Pairwise Projections (first 4 params)")
        p2 = out_dir / "posterior_pairwise_first4.png"
        _save_figure(fig, p2, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
        written.append(str(p2))

    if samples.shape[1] >= 2:
        corr = np.corrcoef(samples.T)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title("Posterior Parameter Correlation")
        step = max(1, int(len(param_names) / 20))
        ticks = list(range(0, len(param_names), step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([param_names[t] for t in ticks], rotation=45, fontsize=12)
        ax.set_yticklabels([param_names[t] for t in ticks], fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        p_corr = out_dir / "posterior_correlation_heatmap.png"
        _save_figure(fig, p_corr, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
        written.append(str(p_corr))

    if ppc_path.exists():
        ppc = _json_load(ppc_path)
        names = [k for k in ppc.keys() if k.startswith("rmse__") or k.startswith("mae__")]
        if names:
            vals = [float(ppc[k]) for k in names]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(names, vals, color="#f97316", alpha=0.85)
            ax.set_title("Stage B Posterior Predictive Metrics")
            ax.tick_params(axis="x", labelrotation=45)
            p3 = out_dir / "ppc_metrics.png"
            _save_figure(fig, p3, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
            written.append(str(p3))

    sbc_path = stage_b_dir / "stage_b_sbc_summary.json"
    if sbc_path.exists():
        sbc = _json_load(sbc_path)
        ranks = sbc.get("ranks", []) if isinstance(sbc, dict) else []
        if isinstance(ranks, list) and len(ranks) > 0:
            ranks_np = np.asarray(ranks, dtype=np.float64)
            if ranks_np.ndim == 2 and ranks_np.shape[1] > 0:
                n_show_rank = min(12, ranks_np.shape[1])
                fig, axes = plt.subplots(n_show_rank, 1, figsize=(10, 2.0 * n_show_rank))
                if n_show_rank == 1:
                    axes = [axes]
                for i in range(n_show_rank):
                    ax = axes[i]
                    ax.hist(ranks_np[:, i], bins=20, color="#10b981", alpha=0.85)
                    ax.set_title(f"SBC Rank: {param_names[i] if i < len(param_names) else f'param_{i}'}")
                p_sbc = out_dir / "sbc_rank_histograms_top12.png"
                _save_figure(fig, p_sbc, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
                written.append(str(p_sbc))

    # Prior vs Posterior overlay ────────────────────────────────────
    prior_low = prior_info.get("prior_low", [])
    prior_high = prior_info.get("prior_high", [])
    if isinstance(prior_low, list) and isinstance(prior_high, list) and len(prior_low) == samples.shape[1]:
        n_overlay = min(12, samples.shape[1])
        fig, axes = plt.subplots(nrows=n_overlay, ncols=1, figsize=(11, 2.4 * n_overlay))
        if n_overlay == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            lo, hi = float(prior_low[i]), float(prior_high[i])
            vals = samples[:, i]
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=40, density=True, color="#7c3aed", alpha=0.75, label="Posterior")
            if hi > lo:
                xx = np.linspace(lo, hi, 200)
                ax.plot(xx, np.ones_like(xx) / (hi - lo), color="#ef4444", linewidth=2, linestyle="--", label="Prior")
            ax.set_title(param_names[i], fontsize=16)
            ax.legend(fontsize=12)
        fig.suptitle("Prior vs Posterior", fontsize=12)
        fig.tight_layout()
        p_pvp = out_dir / "prior_vs_posterior_top12.png"
        _save_figure(fig, p_pvp, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
        written.append(str(p_pvp))

    # Posterior summary table ───────────────────────────────────────
    summary_path = stage_b_dir / "stage_b_posterior_summary.json"
    if summary_path.exists():
        summ = _json_load(summary_path)
        fig, ax = plt.subplots(figsize=(12, max(3, 0.35 * min(20, samples.shape[1]))))
        ax.axis("off")
        pnames_show = param_names[:20]
        table_data = []
        for pn in pnames_show:
            entry = summ.get(pn, {})
            table_data.append([
                pn[:30],
                f"{entry.get('mean', float('nan')):.4g}",
                f"{entry.get('std', float('nan')):.4g}",
                f"{entry.get('q05', entry.get('q25', float('nan'))):.4g}",
                f"{entry.get('q50', float('nan')):.4g}",
                f"{entry.get('q95', entry.get('q75', float('nan'))):.4g}",
            ])
        if table_data:
            tbl = ax.table(cellText=table_data, colLabels=["Parameter", "Mean", "Std", "Q5/Q25", "Median", "Q95/Q75"],
                           loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1.0, 1.3)
            ax.set_title("Stage B Posterior Summary", fontsize=11, pad=10)
            p_tbl = out_dir / "posterior_summary_table.png"
            _save_figure(fig, p_tbl, dpi=plot_cfg.dpi, save_pdf=bool(plot_cfg.save_pdf))
            written.append(str(p_tbl))

    return written


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def project_to_train_config(project_cfg: ProjectConfig) -> TrainConfig:
    defaults = TrainConfig()
    device_str = str(project_cfg.stage_a.device).lower()
    compile_model = bool(project_cfg.stage_a.compile_model) and device_str.startswith("cuda")

    return TrainConfig(
        device=project_cfg.stage_a.device,
        epochs=int(project_cfg.stage_a.epochs),
        batch_size=int(project_cfg.stage_a.batch_size),
        lr=float(project_cfg.stage_a.lr),
        lr_schedule=str(getattr(project_cfg.stage_a, 'lr_schedule', 'cosine')),
        lr_min_factor=float(getattr(project_cfg.stage_a, 'lr_min_factor', 0.01)),
        weight_decay=float(project_cfg.stage_a.weight_decay),
        seed=int(project_cfg.stage_a.seed),
        compile_model=compile_model,
        dry_mass_kg=float(project_cfg.stage_a.dry_mass_kg),
        mass_init_kg=float(project_cfg.stage_a.mass_init_kg),
        isp_init_s=float(project_cfg.stage_a.isp_init_s),
        eta_init=float(project_cfg.stage_a.eta_init),
        thrust_init_N=float(project_cfg.stage_a.thrust_init_N),
        drag_init_kmps2=float(project_cfg.stage_a.drag_init_kmps2),
        thermal_duty_cap=float(project_cfg.stage_a.thermal_duty_cap),
        max_grad_norm=float(project_cfg.stage_a.max_grad_norm),
        use_j2=bool(project_cfg.force_model.use_j2),
        use_drag=bool(project_cfg.force_model.use_drag),
        use_power_cap=bool(project_cfg.force_model.use_power_cap),
        use_timing_bias=bool(project_cfg.force_model.use_timing_bias),
        lambda_a=float(project_cfg.stage_a.lambda_a),
        lambda_da=float(project_cfg.stage_a.lambda_da),
        lambda_raan=float(project_cfg.stage_a.lambda_raan),
        lambda_lam=float(project_cfg.stage_a.lambda_lam),
        lambda_a_end=float(project_cfg.stage_a.lambda_a_end),
        lambda_rate=float(project_cfg.stage_a.lambda_rate),
        lambda_prior=float(project_cfg.stage_a.lambda_prior),
        lambda_hall=float(project_cfg.stage_a.lambda_hall),
        lambda_chemistry=float(project_cfg.stage_a.lambda_chemistry),
        lambda_feasibility=float(project_cfg.stage_a.lambda_feasibility),
        robust_loss=str(project_cfg.stage_a.robust_loss),
        huber_delta=float(project_cfg.stage_a.huber_delta),
        robust_student_t_dof=float(project_cfg.stage_a.robust_student_t_dof),
        robust_student_t_scale=float(project_cfg.stage_a.robust_student_t_scale),
        phase_loss_weight_power=float(project_cfg.stage_a.phase_loss_weight_power),
        obs_weight_a=float(project_cfg.stage_a.obs_weight_a),
        obs_weight_da=float(project_cfg.stage_a.obs_weight_da),
        obs_weight_raan=float(project_cfg.stage_a.obs_weight_raan),
        obs_weight_lam=float(project_cfg.stage_a.obs_weight_lam),
        obs_weight_rate=float(project_cfg.stage_a.obs_weight_rate),
        obs_weight_da_rate=float(project_cfg.stage_a.obs_weight_da_rate),
        obs_weight_draan_rate=float(project_cfg.stage_a.obs_weight_draan_rate),
        obs_weight_dlam_rate=float(project_cfg.stage_a.obs_weight_dlam_rate),
        obs_weight_collocation=float(project_cfg.stage_a.obs_weight_collocation),
        obs_scale_a_km=float(project_cfg.stage_a.obs_scale_a_km),
        obs_scale_angle_rad=float(project_cfg.stage_a.obs_scale_angle_rad),
        obs_scale_da_rate_km_day=float(project_cfg.stage_a.obs_scale_da_rate_km_day),
        obs_scale_angle_rate_rad_day=float(project_cfg.stage_a.obs_scale_angle_rate_rad_day),
        collocation_enabled=bool(project_cfg.stage_a.collocation_enabled),
        collocation_taus=tuple(float(x) for x in project_cfg.stage_a.collocation_taus),
        collocation_tolerance_hours=float(project_cfg.stage_a.collocation_tolerance_hours),
        use_piecewise_thrust_schedule=bool(project_cfg.stage_a.use_piecewise_thrust_schedule),
        piecewise_midpoint_scale_init=float(project_cfg.stage_a.piecewise_midpoint_scale_init),
        vd_init_V=float(project_cfg.stage_a.vd_init_V),
        vc_init_V=float(project_cfg.stage_a.vc_init_V),
        vb_init_V=float(project_cfg.stage_a.vb_init_V),
        ib_init_A=float(project_cfg.stage_a.ib_init_A),
        eta_b_init=float(project_cfg.stage_a.eta_b_init),
        eta_v_init=float(project_cfg.stage_a.eta_v_init),
        eta_m_init=float(project_cfg.stage_a.eta_m_init),
        eta_o_init=float(project_cfg.stage_a.eta_o_init),
        gamma_init=float(project_cfg.stage_a.gamma_init),
        nu_a_init=float(project_cfg.stage_a.nu_a_init),
        mdot_a_init_kg_s=float(project_cfg.stage_a.mdot_a_init_kg_s),
        mdot_c_init_kg_s=float(project_cfg.stage_a.mdot_c_init_kg_s),
        pressure_base_pa=float(project_cfg.stage_a.pressure_base_pa),
        pressure_gain_pa_per_kg_s=float(project_cfg.stage_a.pressure_gain_pa_per_kg_s),
        neutral_temp_K=float(project_cfg.stage_a.neutral_temp_K),
        electron_temp_base_eV=float(project_cfg.stage_a.electron_temp_base_eV),
        electron_temp_gain_per_V=float(project_cfg.stage_a.electron_temp_gain_per_V),
        electron_temp_gain_nua=float(project_cfg.stage_a.electron_temp_gain_nua),
        ionization_length_m=float(project_cfg.stage_a.ionization_length_m),
        ionization_ratio_min=float(project_cfg.stage_a.ionization_ratio_min),
        ionization_ratio_max=float(project_cfg.stage_a.ionization_ratio_max),
        curriculum_kinematics_epochs=int(project_cfg.stage_a.curriculum_kinematics_epochs),
        curriculum_collocation_epochs=int(project_cfg.stage_a.curriculum_collocation_epochs),
        curriculum_physics_ramp_epochs=int(project_cfg.stage_a.curriculum_physics_ramp_epochs),
        duration_weight_enabled=bool(project_cfg.stage_a.duration_weight_enabled),
        duration_weight_power=float(project_cfg.stage_a.duration_weight_power),
        closure_mode=str(getattr(project_cfg.stage_a, "closure_mode", "legacy_surrogate")),
        shell_drag_comp_fraction_init=float(getattr(project_cfg.stage_a, "shell_drag_comp_fraction_init", 1.0)),
        phase_power_cap_init_W=float(defaults.phase_power_cap_init_W),
        phase_ramp_fraction_init=float(defaults.phase_ramp_fraction_init),
        phase_time_offset_init_s=float(defaults.phase_time_offset_init_s),
        sat_drag_scale_init=float(defaults.sat_drag_scale_init),
        sat_thrust_scale_init=float(defaults.sat_thrust_scale_init),
        sat_time_bias_init_s=float(defaults.sat_time_bias_init_s),
        util_mass_init=float(defaults.util_mass_init),
        util_current_init=float(defaults.util_current_init),
        util_voltage_init=float(defaults.util_voltage_init),
        divergence_eff_init=float(defaults.divergence_eff_init),
        transport_proxy_init=float(defaults.transport_proxy_init),
        shielding_weight_init=float(defaults.shielding_weight_init),
        lifetime_weight_init=float(defaults.lifetime_weight_init),
        fit_mode=str(getattr(project_cfg.stage_a, "fit_mode", "trajectory_matching")),
        intervals_csv=str(getattr(project_cfg.stage_a, "intervals_csv", "")),
        max_arc_obs=int(getattr(project_cfg.stage_a, "max_arc_obs", 200)),
        min_arc_obs=int(getattr(project_cfg.stage_a, "min_arc_obs", 5)),
        max_subarc_days=float(getattr(project_cfg.stage_a, "max_subarc_days", 30.0)),
        lambda_continuity=float(getattr(project_cfg.stage_a, "lambda_continuity", 0.1)),
        lambda_path=float(getattr(project_cfg.stage_a, "lambda_path", 5.0)),
        lambda_endpoint_a=float(getattr(project_cfg.stage_a, "lambda_endpoint_a", 1.0)),
        lambda_endpoint_raan=float(getattr(project_cfg.stage_a, "lambda_endpoint_raan", 0.0)),
        lambda_endpoint_lam=float(getattr(project_cfg.stage_a, "lambda_endpoint_lam", 0.0)),
        arc_weight_mode=str(getattr(project_cfg.stage_a, "arc_weight_mode", "sqrt_inv_n_obs")),
        use_atmosphere_drag=bool(getattr(project_cfg.stage_a, "use_atmosphere_drag", True)),
        inv_ballistic_coeff=float(getattr(project_cfg.stage_a, "inv_ballistic_coeff", 0.0334)),
        nonlinear_propagation=bool(getattr(project_cfg.stage_a, "nonlinear_propagation", True)),
        rk4_step_hours=float(getattr(project_cfg.stage_a, "rk4_step_hours", 12.0)),
        early_stopping_patience=int(getattr(project_cfg.stage_a, "early_stopping_patience", 10)),
    )


def run_default_pipeline(project_cfg: ProjectConfig) -> Dict[str, Any]:
    pipeline_start = time.perf_counter()
    output_root = Path(project_cfg.paths.output_root)
    run_paths = prepare_run_paths(output_root)
    setup_logging(run_paths.logs / "run.log")

    save_json(run_paths.latest / "resolved_config.json", asdict(project_cfg))
    logging.info("Starting default Stage A/B pipeline")
    logging.info("Run root: %s", str(run_paths.latest))
    logging.info(
        "Hardware: cpu_count=%s cuda_available=%s cuda_device_count=%s",
        str(os.cpu_count()),
        str(torch.cuda.is_available()),
        str(torch.cuda.device_count() if torch.cuda.is_available() else 0),
    )
    if torch.cuda.is_available():
        try:
            logging.info("CUDA device[0]: %s", torch.cuda.get_device_name(0))
        except Exception:
            pass

    stage_t0 = time.perf_counter()
    tle_dir = resolve_default_tle_dir(project_cfg.paths.tle_dir)
    labels_source = resolve_default_labels_path(project_cfg.paths.labels_csv)
    labels_csv = resolve_labels_csv_path(labels_source)
    logging.info("Resolved input paths in %.2fs", time.perf_counter() - stage_t0)
    logging.info("TLE directory: %s", str(tle_dir))
    logging.info("Labels CSV: %s", str(labels_csv))

    segments_path_raw = Path(project_cfg.paths.segments_csv)
    if segments_path_raw.is_absolute():
        segments_csv = resolve_output_csv_path(segments_path_raw)
    else:
        segments_csv = resolve_output_csv_path(run_paths.latest / segments_path_raw)
    logging.info("Segments output path: %s", str(segments_csv))

    diagnostics_manifest: Dict[str, Any] = {
        "plots": {},
        "tables": {},
        "reports": {},
    }

    tle_df: Optional[pd.DataFrame] = None  # Will be set if TLEs are loaded

    if bool(project_cfg.run.rebuild_segments) or (not segments_csv.exists()):
        selected_tle_files = select_tle_files(tle_dir, project_cfg.run.max_sats)
        if selected_tle_files is not None:
            logging.info("Using %d TLE files from %s", len(selected_tle_files), str(tle_dir))
        else:
            logging.info("Using all available TLE files from %s", str(tle_dir))

        load_t0 = time.perf_counter()
        tle_df, _ = load_tle_data_with_progress(
            tle_dir=tle_dir,
            only_files=selected_tle_files,
            derived_cols=("sma",),
            workers=project_cfg.run.tle_workers,
            chunk_size=int(project_cfg.run.tle_chunk_size),
            progress_every_files=int(project_cfg.run.tle_progress_files),
        )
        logging.info("Raw TLE load completed in %.2fs with %d rows", time.perf_counter() - load_t0, len(tle_df))

        preprocess_t0 = time.perf_counter()
        tle_df = preprocess_tle_dataframe(tle_df, project_cfg.smoothing)
        logging.info("TLE preprocessing completed in %.2fs", time.perf_counter() - preprocess_t0)

        labels_t0 = time.perf_counter()
        labels_df = pd.read_csv(labels_csv)
        logging.info("Labels read completed in %.2fs with %d rows", time.perf_counter() - labels_t0, len(labels_df))

        segment_build_t0 = time.perf_counter()
        seg_cfg = SegmentBuilderConfig(
            tolerance_seconds=float(project_cfg.stage_a.segment_tolerance_hours) * 3600.0,
            collocation_enabled=bool(project_cfg.stage_a.collocation_enabled),
            collocation_taus=tuple(float(x) for x in project_cfg.stage_a.collocation_taus),
        )
        seg_df = build_segments_from_tles_and_labels(tle_df, labels_df, seg_cfg)
        seg_df.to_csv(segments_csv, index=False)
        logging.info(
            "Built and wrote %d segments to %s in %.2fs",
            len(seg_df),
            str(segments_csv),
            time.perf_counter() - segment_build_t0,
        )
    else:
        seg_df = sanitize_segments_for_training(_ensure_segment_rate_columns(pd.read_csv(segments_csv)))
        logging.info("Using existing segments: %s (%d rows)", str(segments_csv), len(seg_df))

    save_json(
        run_paths.tables / "segment_summary.json",
        {
            "num_segments": int(len(seg_df)),
            "num_satellites": int(seg_df["sat_id"].nunique()) if "sat_id" in seg_df.columns else 0,
            "num_phases": int(seg_df["phase"].nunique()) if "phase" in seg_df.columns else 0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )

    try:
        data_quality_plots = generate_data_quality_plots(seg_df, run_paths.plots_data_quality, project_cfg.plotting)
        diagnostics_manifest["plots"]["data_quality"] = data_quality_plots
    except Exception as exc:
        logging.exception("Data-quality plotting failed: %s", str(exc))

    stage_a_outdir = run_paths.checkpoints / "stage_a"
    if bool(project_cfg.stage_a.enabled):
        train_cfg = project_to_train_config(project_cfg)
        logging.info("Stage A config: device=%s epochs=%d batch_size=%d lr=%g fit_mode=%s", train_cfg.device, train_cfg.epochs, train_cfg.batch_size, train_cfg.lr, train_cfg.fit_mode)
        stage_a_train_t0 = time.perf_counter()

        if train_cfg.fit_mode == "trajectory_matching":
            # Trajectory-matching path ──────────────────────────────
            intervals_csv_path = train_cfg.intervals_csv.strip()
            if not intervals_csv_path:
                # Auto-resolve from full_exports/
                _default_intervals = Path(project_cfg.paths.output_root).parent / "full_exports" / "maneuver_phase_intervals_gen1_full.csv"
                if not _default_intervals.exists():
                    _default_intervals = Path("full_exports") / "maneuver_phase_intervals_gen1_full.csv"
                intervals_csv_path = str(_default_intervals)
            logging.info("Intervals CSV: %s", intervals_csv_path)

            # Need tle_df for arc building; if segments were rebuilt we already have it.
            # Otherwise reload.
            if tle_df is None:
                _tle_dir = resolve_default_tle_dir(project_cfg.paths.tle_dir)
                _sel = select_tle_files(_tle_dir, project_cfg.run.max_sats)
                tle_df, _ = load_tle_data_with_progress(
                    tle_dir=_tle_dir,
                    only_files=_sel,
                    derived_cols=("sma",),
                    workers=project_cfg.run.tle_workers,
                    chunk_size=int(project_cfg.run.tle_chunk_size),
                    progress_every_files=int(project_cfg.run.tle_progress_files),
                )
                tle_df = preprocess_tle_dataframe(tle_df, project_cfg.smoothing)

            intervals_df = pd.read_csv(intervals_csv_path)
            logging.info("Loaded %d phase intervals", len(intervals_df))

            arc_cfg = ArcBuildConfig(
                min_obs=int(train_cfg.min_arc_obs),
                min_duration_s=6 * 3600.0,
                max_duration_s=120 * 86400.0,
            )
            arcs = build_arcs_from_tles_and_intervals(tle_df, intervals_df, arc_cfg)
            logging.info("Built %d arcs from %d intervals", len(arcs), len(intervals_df))
            if len(arcs) == 0:
                raise RuntimeError("No valid arcs built — check intervals CSV and TLE coverage")

            traj_cfg = TrajectoryConfig(
                lambda_path=float(train_cfg.lambda_path),
                lambda_endpoint_a=float(train_cfg.lambda_endpoint_a),
                lambda_endpoint_raan=float(train_cfg.lambda_endpoint_raan),
                lambda_endpoint_lam=float(train_cfg.lambda_endpoint_lam),
                lambda_continuity=float(train_cfg.lambda_continuity),
                max_subarc_days=float(train_cfg.max_subarc_days),
                arc_weight_mode=str(train_cfg.arc_weight_mode),
                robust_loss=train_cfg.robust_loss,
                huber_delta=train_cfg.huber_delta,
                use_atmosphere_drag=bool(train_cfg.use_atmosphere_drag),
                inv_ballistic_coeff=float(train_cfg.inv_ballistic_coeff),
                nonlinear_propagation=bool(getattr(train_cfg, 'nonlinear_propagation', True)),
                rk4_step_hours=float(getattr(train_cfg, 'rk4_step_hours', 12.0)),
            )

            ckpt_path, hist_df = train_stage_a_trajectory(arcs, stage_a_outdir, train_cfg)

            # Run trajectory-specific diagnostics
            try:
                ckpt_data = torch.load(ckpt_path, map_location=train_cfg.device, weights_only=False)
                diag_model = StageAModel(
                    len(ckpt_data["sat_to_idx"]),
                    len(ckpt_data["phase_to_idx"]),
                    ckpt_data["phase_signs"],
                    train_cfg,
                ).to(train_cfg.device)
                diag_model.load_state_dict(ckpt_data["model_state"])
                diag_model.eval()
                traj_report = trajectory_validation_report(
                    arcs=arcs,
                    model=diag_model,
                    traj_cfg=traj_cfg,
                    device=torch.device(train_cfg.device),
                    out_json=run_paths.reports / "trajectory_validation_report.json",
                )
                diagnostics_manifest["reports"]["trajectory_validation"] = str(run_paths.reports / "trajectory_validation_report.json")
                logging.info("Trajectory validation: path_rmse_km=%.2f", traj_report.get("aggregate", {}).get("path_rmse_km_mean", float("nan")))
            except Exception as exc:
                logging.exception("Trajectory validation report failed: %s", str(exc))

            try:
                plot_trajectory_fits(
                    arcs=arcs,
                    model=diag_model,
                    device=torch.device(train_cfg.device),
                    out_dir=run_paths.plots_fit,
                    traj_cfg=traj_cfg,
                )
                plot_trajectory_residual_analysis(
                    arcs=arcs,
                    model=diag_model,
                    device=torch.device(train_cfg.device),
                    out_dir=run_paths.plots_fit,
                    traj_cfg=traj_cfg,
                )
                plot_trajectory_per_satellite_rmse(
                    arcs=arcs,
                    model=diag_model,
                    device=torch.device(train_cfg.device),
                    out_dir=run_paths.plots_fit,
                    traj_cfg=traj_cfg,
                )
            except Exception as exc:
                logging.exception("Trajectory plotting failed: %s", str(exc))

            try:
                plot_trajectory_training_history(
                    history_csv=stage_a_outdir / "stage_a_history.csv",
                    out_dir=run_paths.plots_train,
                )
                plot_trajectory_parameter_evolution(
                    trace_csv=stage_a_outdir / "stage_a_parameter_trace.csv",
                    out_dir=run_paths.plots_parameters,
                )
            except Exception as exc:
                logging.exception("Trajectory training/parameter plotting failed: %s", str(exc))

        else:
            # Segment-endpoint path (default) ───────────────────────
            ckpt_path, hist_df = train_stage_a(segments_csv, stage_a_outdir, train_cfg)

        logging.info("Stage A checkpoint: %s", str(ckpt_path))
        logging.info("Stage A training completed in %.2fs", time.perf_counter() - stage_a_train_t0)
    else:
        ckpt_path = stage_a_outdir / "stage_a_checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Stage A disabled but checkpoint not found at expected path: {ckpt_path}"
            )
        hist_path = stage_a_outdir / "stage_a_history.csv"
        hist_df = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()

    try:
        training_plots = generate_stage_a_training_plots(hist_df, run_paths.plots_train, project_cfg.plotting)
        diagnostics_manifest["plots"]["training"] = training_plots
    except Exception as exc:
        logging.exception("Training plotting failed: %s", str(exc))

    validation_report_path = run_paths.reports / "stage_a_validation_report.json"
    if bool(project_cfg.validation.enabled):
        validation_t0 = time.perf_counter()
        report = run_stage_a_validation(
            segments_csv=segments_csv,
            checkpoint_path=ckpt_path,
            out_json=validation_report_path,
            device=str(project_cfg.validation.device),
        )
        diagnostics_manifest["reports"]["stage_a_validation"] = str(validation_report_path)
        logging.info("Stage A validation all_checks_pass=%s", str(report.get("all_checks_pass")))
        logging.info("Stage A validation completed in %.2fs", time.perf_counter() - validation_t0)

    if bool(project_cfg.validation.run_timing_sensitivity):
        try:
            timing_csv = run_paths.tables / "timing_sensitivity.csv"
            timing_df = run_timing_sensitivity_analysis(
                segments_csv=segments_csv,
                checkpoint_path=ckpt_path,
                out_csv=timing_csv,
                shift_hours=float(project_cfg.validation.timing_shift_hours),
                n_samples=int(project_cfg.validation.timing_shift_samples),
                device=str(project_cfg.validation.device),
            )
            diagnostics_manifest["tables"]["timing_sensitivity"] = str(timing_csv)
            timing_plots = generate_timing_sensitivity_plots(timing_df, run_paths.plots_sensitivity, project_cfg.plotting)
            diagnostics_manifest["plots"]["timing_sensitivity"] = timing_plots
        except Exception as exc:
            logging.exception("Timing sensitivity analysis failed: %s", str(exc))

    if bool(project_cfg.validation.run_force_model_ablations):
        try:
            abl_csv = run_paths.tables / "force_model_sensitivity.csv"
            abl_df = run_force_model_sensitivity_analysis(
                segments_csv=segments_csv,
                checkpoint_path=ckpt_path,
                out_csv=abl_csv,
                device=str(project_cfg.validation.device),
            )
            diagnostics_manifest["tables"]["force_model_sensitivity"] = str(abl_csv)
            abl_plots = generate_ablation_plots(abl_df, run_paths.plots_ablations, project_cfg.plotting)
            diagnostics_manifest["plots"]["sensitivity"] = abl_plots
        except Exception as exc:
            logging.exception("Force model sensitivity analysis failed: %s", str(exc))

    if bool(project_cfg.validation.run_synthetic_recovery):
        try:
            synth_dir = run_paths.reports / "synthetic_recovery"
            synth_report = run_synthetic_recovery_refit(
                segments_csv=segments_csv,
                checkpoint_path=ckpt_path,
                outdir=synth_dir,
                refit_epochs=int(project_cfg.validation.synthetic_refit_epochs),
                noise_std_a_km=float(project_cfg.stage_a.synthetic_noise_std_a_km),
                noise_std_angle_rad=float(project_cfg.stage_a.synthetic_noise_std_angle_rad),
                device=str(project_cfg.validation.device),
            )
            synth_report_path = synth_dir / "synthetic_recovery_report.json"
            diagnostics_manifest["reports"]["synthetic_recovery"] = str(synth_report_path)
            synth_plots = generate_synthetic_recovery_plots(synth_report, run_paths.plots_synthetic, project_cfg.plotting)
            diagnostics_manifest["plots"]["synthetic_recovery"] = synth_plots
        except Exception as exc:
            logging.exception("Synthetic recovery failed: %s", str(exc))

    if bool(project_cfg.validation.run_loso):
        try:
            loso_dir = run_paths.reports / "loso"
            loso_report = run_loso_cross_validation(
                segments_csv=segments_csv,
                checkpoint_path=ckpt_path,
                outdir=loso_dir,
                max_satellites=int(project_cfg.validation.loso_max_satellites),
                device=str(project_cfg.validation.device),
            )
            diagnostics_manifest["reports"]["loso"] = str(loso_dir / "loso_summary.json")
            diagnostics_manifest["tables"]["loso_metrics"] = str(loso_dir / "loso_metrics.csv")
            logging.info("LOSO completed for %s satellites", str(loso_report.get("num_satellites")))
        except Exception as exc:
            logging.exception("LOSO validation failed: %s", str(exc))

    stage_a_prediction_table = run_paths.tables / "stage_a_predictions.csv"
    stage_a_fit_report = run_paths.reports / "stage_a_fit_diagnostics.json"
    try:
        pred_df = collect_stage_a_predictions(
            segments_csv=segments_csv,
            checkpoint_path=ckpt_path,
            device=str(project_cfg.validation.device),
        )
        pred_df.to_csv(stage_a_prediction_table, index=False)
        diagnostics_manifest["tables"]["stage_a_predictions"] = str(stage_a_prediction_table)

        split_pred = split_segments_dataframe(pred_df, project_cfg.split)
        fit_report = _compute_fit_metrics_from_predictions(pred_df)
        fit_report["split_metrics"] = {
            split_name: _compute_fit_metrics_from_predictions(split_df)
            for split_name, split_df in split_pred.items()
        }
        fit_report["split_phase_metrics"] = {
            split_name: summarize_fit_metrics_by_phase(split_df)
            for split_name, split_df in split_pred.items()
        }
        save_json(stage_a_fit_report, fit_report)
        diagnostics_manifest["reports"]["stage_a_fit"] = str(stage_a_fit_report)

        fit_plots = generate_stage_a_fit_plots(pred_df, run_paths.plots_fit, project_cfg.plotting)
        diagnostics_manifest["plots"]["fit"] = fit_plots

        robust_plots = generate_robust_residual_plots(pred_df, run_paths.plots_fit, project_cfg.plotting)
        diagnostics_manifest["plots"]["robust_residuals"] = robust_plots

        overlay_plots = generate_segment_internal_overlay_plots(pred_df, run_paths.plots_fit, project_cfg.plotting)
        diagnostics_manifest["plots"]["segment_internal_overlays"] = overlay_plots

        chemistry_plots = generate_chemistry_closure_plots(pred_df, run_paths.plots_fit, project_cfg.plotting)
        diagnostics_manifest["plots"]["chemistry"] = chemistry_plots
    except Exception as exc:
        logging.exception("Stage A fit diagnostics failed: %s", str(exc))

    try:
        summary_path = stage_a_outdir / "stage_a_parameter_summary.json"
        trace_path = stage_a_outdir / "stage_a_parameter_trace.csv"
        summary_obj = _json_load(summary_path) if summary_path.exists() else {}
        trace_df = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()
        parameter_plots = generate_stage_a_parameter_plots(trace_df, summary_obj, run_paths.plots_parameters, project_cfg.plotting)
        diagnostics_manifest["plots"]["parameters"] = parameter_plots
        diagnostics_manifest["reports"]["stage_a_parameter_summary"] = str(summary_path)

        if not hist_df.empty:
            physics_summary = {
                "final_epoch": int(hist_df["epoch"].iloc[-1]) if "epoch" in hist_df.columns else int(len(hist_df)),
                "loss_hall": float(hist_df["loss_hall"].iloc[-1]) if "loss_hall" in hist_df.columns else float("nan"),
                "loss_chemistry_penalty": float(hist_df["loss_chemistry_penalty"].iloc[-1]) if "loss_chemistry_penalty" in hist_df.columns else float("nan"),
                "loss_feasibility": float(hist_df["loss_feasibility"].iloc[-1]) if "loss_feasibility" in hist_df.columns else float("nan"),
                "loss_collocation": float(hist_df["loss_collocation"].iloc[-1]) if "loss_collocation" in hist_df.columns else float("nan"),
            }
            physics_summary_path = run_paths.reports / "stage_a_physics_residual_summary.json"
            save_json(physics_summary_path, physics_summary)
            diagnostics_manifest["reports"]["stage_a_physics_residual_summary"] = str(physics_summary_path)
    except Exception as exc:
        logging.exception("Stage A parameter plotting failed: %s", str(exc))

    ident_report_path = run_paths.reports / "identifiability_summary.json"
    try:
        ident_report = run_identifiability_diagnostics(
            segments_csv=segments_csv,
            checkpoint_path=ckpt_path,
            outdir=run_paths.reports,
            max_segments=int(project_cfg.stage_b.max_segments),
            max_phase_parameters=int(project_cfg.stage_b.max_phase_parameters),
            device=str(project_cfg.stage_b.device),
            effective_mode=bool(project_cfg.stage_b.effective_mode),
            anchor_from_stage_a=bool(project_cfg.stage_b.anchor_from_stage_a),
        )
        diagnostics_manifest["reports"]["identifiability"] = str(ident_report_path)
        diagnostics_manifest["tables"]["identifiability_singular_values"] = str(run_paths.reports / "identifiability_singular_values.csv")
        diagnostics_manifest["tables"]["identifiability_parameter_sensitivity"] = str(run_paths.reports / "identifiability_parameter_sensitivity.csv")

        ident_plots = generate_identifiability_plots(run_paths.reports, run_paths.plots_sensitivity, project_cfg.plotting)
        diagnostics_manifest["plots"]["identifiability"] = ident_plots
        logging.info("Identifiability diagnostics rank=%s cond=%.3e", str(ident_report.get("rank")), float(ident_report.get("condition_number", float("nan"))))
    except Exception as exc:
        logging.exception("Identifiability diagnostics failed: %s", str(exc))

    stage_b_ran = False
    stage_b_outdir = run_paths.checkpoints / "stage_b"
    if bool(project_cfg.stage_b.enabled) and (not bool(project_cfg.run.skip_stage_b)):
        stage_b_t0 = time.perf_counter()
        run_stage_b_snpe(
            segments_csv=segments_csv,
            checkpoint_path=ckpt_path,
            outdir=stage_b_outdir,
            num_simulations=int(project_cfg.stage_b.num_simulations),
            max_segments=int(project_cfg.stage_b.max_segments),
            max_phase_parameters=int(project_cfg.stage_b.max_phase_parameters),
            num_posterior_samples=int(project_cfg.stage_b.num_posterior_samples),
            ppc_samples=int(project_cfg.stage_b.ppc_samples),
            density_estimator=str(project_cfg.stage_b.density_estimator),
            normalize_observation=bool(project_cfg.stage_b.normalize_observation),
            include_initial_conditions=bool(project_cfg.stage_b.include_initial_conditions),
            include_phase_context=bool(project_cfg.stage_b.include_phase_context),
            include_rate_features=bool(project_cfg.stage_b.include_rate_features),
            normalization_eps=float(project_cfg.stage_b.normalization_eps),
            calibration_subset_segments=int(project_cfg.stage_b.calibration_subset_segments),
            run_sbc=bool(project_cfg.stage_b.run_sbc),
            sbc_draws=int(project_cfg.stage_b.sbc_draws),
            sbc_posterior_samples=int(project_cfg.stage_b.sbc_posterior_samples),
            device=str(project_cfg.stage_b.device),
            effective_mode=bool(project_cfg.stage_b.effective_mode),
            anchor_from_stage_a=bool(project_cfg.stage_b.anchor_from_stage_a),
            mixed_precision=bool(project_cfg.stage_b.mixed_precision),
        )
        stage_b_ran = True
        logging.info("Stage B completed in %.2fs", time.perf_counter() - stage_b_t0)

    if stage_b_ran:
        try:
            stage_b_plots = generate_stage_b_plots(stage_b_outdir, run_paths.plots_stage_b, project_cfg.plotting)
            diagnostics_manifest["plots"]["stage_b"] = stage_b_plots
            diagnostics_manifest["reports"]["stage_b_summary"] = str(stage_b_outdir / "stage_b_posterior_summary.json")
            diagnostics_manifest["reports"]["stage_b_ppc"] = str(stage_b_outdir / "stage_b_posterior_predictive_check.json")
            if (stage_b_outdir / "stage_b_sbc_summary.json").exists():
                diagnostics_manifest["reports"]["stage_b_sbc"] = str(stage_b_outdir / "stage_b_sbc_summary.json")
            if (stage_b_outdir / "stage_b_observation_features.json").exists():
                diagnostics_manifest["reports"]["stage_b_observation_features"] = str(stage_b_outdir / "stage_b_observation_features.json")
            if (stage_b_outdir / "stage_b_posterior_correlation.npy").exists():
                diagnostics_manifest["tables"]["stage_b_posterior_correlation"] = str(stage_b_outdir / "stage_b_posterior_correlation.npy")
        except Exception as exc:
            logging.exception("Stage B plotting failed: %s", str(exc))

    diagnostics_manifest_path = run_paths.reports / "diagnostics_manifest.json"
    save_json(diagnostics_manifest_path, diagnostics_manifest)

    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_root": str(run_paths.latest),
        "segments_csv": str(segments_csv),
        "stage_a_checkpoint": str(ckpt_path),
        "stage_a_validation_report": str(validation_report_path),
        "stage_a_parameter_summary": str(stage_a_outdir / "stage_a_parameter_summary.json"),
        "stage_a_physics_residual_summary": str(run_paths.reports / "stage_a_physics_residual_summary.json"),
        "identifiability_report": str(ident_report_path),
        "diagnostics_manifest": str(diagnostics_manifest_path),
        "stage_b_ran": bool(stage_b_ran),
        "stage_b_posterior_summary": str(stage_b_outdir / "stage_b_posterior_summary.json") if stage_b_ran else "",
        "stage_b_ppc": str(stage_b_outdir / "stage_b_posterior_predictive_check.json") if stage_b_ran else "",
        "elapsed_seconds": float(time.perf_counter() - pipeline_start),
        "limitations": [
            "Reduced mean-element model; no full Cowell propagation in training loop.",
            "Krypton closure uses compact surrogate coefficients rather than full plasma PDE solve.",
            "Stage B calibration uses reduced summary vectors and subset controls for tractability.",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    save_json(run_paths.reports / "run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gen 1 Starlink Stage A/B reduced-order tool. If no subcommand is provided, runs default Stage A+B flow."
    )

    # Global/default-mode overrides.
    p.add_argument("--config-file", type=Path, default=None)
    p.add_argument("--tle-dir", type=Path, default=None)
    p.add_argument("--labels-csv", type=Path, default=None)
    p.add_argument("--segments-csv", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--max-sats", type=int, default=500)
    p.add_argument("--tle-workers", type=int, default=None)
    p.add_argument("--tle-chunk-size", type=int, default=None)
    p.add_argument("--tle-progress-files", type=int, default=None)
    p.add_argument("--rebuild-segments", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--stage-b-device", type=str, default=None)
    p.add_argument("--run-stage-b", action="store_true")
    p.add_argument("--skip-stage-b", action="store_true")
    p.add_argument("--fit-mode", type=str, default=None, choices=["segment_endpoint", "trajectory_matching"],
                   help="Stage A fitting mode: segment_endpoint (default) or trajectory_matching")
    p.add_argument("--intervals-csv", type=Path, default=None,
                   help="Phase intervals CSV path (required for trajectory_matching, auto-resolved if not given)")
    p.add_argument("--max-subarc-days", type=float, default=None,
                   help="Max subarc length for multiple-shooting continuity (default 30)")
    p.add_argument("--min-arc-obs", type=int, default=None,
                   help="Minimum TLE observations per arc (default 5)")

    sub = p.add_subparsers(dest="cmd", required=False)

    p_seg = sub.add_parser("build-segments", help="Build maneuver segments from TLEs and labels")
    p_seg.add_argument("--tle-dir", required=True, type=Path)
    p_seg.add_argument("--labels-csv", required=True, type=Path, help="Labels CSV file or folder containing a labels CSV")
    p_seg.add_argument("--out-csv", required=True, type=Path, help="Output CSV path, or a directory path to write segments.csv")
    p_seg.add_argument("--max-sats", type=int, default=500, help="Optional max number of TLE satellite files to use")
    p_seg.add_argument("--tle-workers", type=int, default=None, help="CPU worker processes for TLE loading (auto when omitted)")
    p_seg.add_argument("--tle-chunk-size", type=int, default=128, help="Files per worker task chunk during parallel TLE load")
    p_seg.add_argument("--tle-progress-files", type=int, default=100, help="Progress print cadence in files during parallel TLE load")
    p_seg.add_argument("--tolerance-hours", type=float, default=6.0)
    p_seg.add_argument("--min-duration-hours", type=float, default=6.0)
    p_seg.add_argument("--max-duration-days", type=float, default=45.0)

    p_train = sub.add_parser("train-stage-a", help="Train differentiable reduced-order Stage A model")
    p_train.add_argument("--segments-csv", required=True, type=Path)
    p_train.add_argument("--outdir", required=True, type=Path)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=3e-3)
    p_train.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_train.add_argument("--no-compile", action="store_true")
    p_train.add_argument("--skip-validation", action="store_true")
    p_train.add_argument("--validation-device", type=str, default="cpu")

    p_validate = sub.add_parser("validate-stage-a", help="Validate a trained Stage A checkpoint on segment data")
    p_validate.add_argument("--segments-csv", required=True, type=Path)
    p_validate.add_argument("--checkpoint", required=True, type=Path)
    p_validate.add_argument("--out-json", type=Path, default=None)
    p_validate.add_argument("--device", type=str, default="cpu")

    p_sbi = sub.add_parser("run-stage-b", help="Run Stage B SNPE on top of fixed segment context")
    p_sbi.add_argument("--segments-csv", required=True, type=Path)
    p_sbi.add_argument("--checkpoint", required=True, type=Path)
    p_sbi.add_argument("--outdir", required=True, type=Path)
    p_sbi.add_argument("--num-simulations", type=int, default=4000)
    p_sbi.add_argument("--max-segments", type=int, default=128)
    p_sbi.add_argument("--max-phase-parameters", type=int, default=6)
    p_sbi.add_argument("--num-posterior-samples", type=int, default=2000)
    p_sbi.add_argument("--ppc-samples", type=int, default=256)
    p_sbi.add_argument("--density-estimator", type=str, default="maf")
    p_sbi.add_argument("--disable-normalize-observation", action="store_true")
    p_sbi.add_argument("--disable-initial-conditions", action="store_true")
    p_sbi.add_argument("--disable-phase-context", action="store_true")
    p_sbi.add_argument("--disable-rate-features", action="store_true")
    p_sbi.add_argument("--normalization-eps", type=float, default=1.0e-6)
    p_sbi.add_argument("--calibration-subset-segments", type=int, default=128)
    p_sbi.add_argument("--run-sbc", action="store_true")
    p_sbi.add_argument("--sbc-draws", type=int, default=64)
    p_sbi.add_argument("--sbc-posterior-samples", type=int, default=256)
    p_sbi.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_sbi.add_argument("--no-effective-mode", action="store_true", help="Disable effective-mode (use full Hall closure params)")
    p_sbi.add_argument("--no-anchor-from-stage-a", action="store_true", help="Disable anchoring globals from Stage A fit")
    p_sbi.add_argument("--mixed-precision", action="store_true", help="Use torch.amp autocast for faster GPU simulations")

    return p.parse_args()


def main():
    args = parse_args()

    if args.cmd is None:
        cli_overrides: Dict[str, Any] = {}

        def set_override(path: Sequence[str], value: Any):
            if value is None:
                return
            ref: Dict[str, Any] = cli_overrides
            for key in path[:-1]:
                ref = ref.setdefault(key, {})
            ref[path[-1]] = value

        set_override(["paths", "config_file"], str(args.config_file) if args.config_file is not None else None)
        set_override(["paths", "tle_dir"], str(args.tle_dir) if args.tle_dir is not None else None)
        set_override(["paths", "labels_csv"], str(args.labels_csv) if args.labels_csv is not None else None)
        set_override(["paths", "segments_csv"], str(args.segments_csv) if args.segments_csv is not None else None)
        set_override(["paths", "output_root"], str(args.output_root) if args.output_root is not None else None)

        set_override(["run", "max_sats"], int(args.max_sats) if args.max_sats is not None else None)
        set_override(["run", "tle_workers"], int(args.tle_workers) if args.tle_workers is not None else None)
        set_override(["run", "tle_chunk_size"], int(args.tle_chunk_size) if args.tle_chunk_size is not None else None)
        set_override(["run", "tle_progress_files"], int(args.tle_progress_files) if args.tle_progress_files is not None else None)
        if bool(args.rebuild_segments):
            set_override(["run", "rebuild_segments"], True)

        set_override(["stage_a", "epochs"], int(args.epochs) if args.epochs is not None else None)
        set_override(["stage_a", "batch_size"], int(args.batch_size) if args.batch_size is not None else None)
        set_override(["stage_a", "lr"], float(args.lr) if args.lr is not None else None)
        set_override(["stage_a", "device"], str(args.device) if args.device is not None else None)
        set_override(["stage_b", "device"], str(args.stage_b_device) if args.stage_b_device is not None else None)

        if bool(args.run_stage_b):
            set_override(["stage_b", "enabled"], True)
            set_override(["run", "skip_stage_b"], False)
        if bool(args.skip_stage_b):
            set_override(["run", "skip_stage_b"], True)

        if args.fit_mode is not None:
            set_override(["stage_a", "fit_mode"], str(args.fit_mode))
        if args.intervals_csv is not None:
            set_override(["stage_a", "intervals_csv"], str(args.intervals_csv))
        if args.max_subarc_days is not None:
            set_override(["stage_a", "max_subarc_days"], float(args.max_subarc_days))
        if args.min_arc_obs is not None:
            set_override(["stage_a", "min_arc_obs"], int(args.min_arc_obs))

        project_cfg = resolve_project_config(cli_overrides=cli_overrides)
        summary = run_default_pipeline(project_cfg)
        print("Default pipeline completed.")
        print(f"Run root: {summary['run_root']}")
        print(f"Segments: {summary['segments_csv']}")
        print(f"Stage A checkpoint: {summary['stage_a_checkpoint']}")
        print(f"Stage B ran: {summary['stage_b_ran']}")
        return

    if args.cmd == "build-segments":
        t_build = time.perf_counter()
        selected_tle_files = select_tle_files(args.tle_dir, args.max_sats)
        if selected_tle_files is not None:
            print(f"Using {len(selected_tle_files)} satellite TLE files from {args.tle_dir.resolve()}")

        tle_df, _ = load_tle_data_with_progress(
            tle_dir=args.tle_dir,
            only_files=selected_tle_files,
            derived_cols=("sma",),
            workers=args.tle_workers,
            chunk_size=int(args.tle_chunk_size),
            progress_every_files=int(args.tle_progress_files),
        )

        t_pre = time.perf_counter()
        tle_df = preprocess_tle_dataframe(tle_df, SmoothingConfig(enabled=False))
        print(f"Preprocessed TLE rows in {time.perf_counter() - t_pre:.2f}s")

        t_labels = time.perf_counter()
        labels_csv_path = resolve_labels_csv_path(args.labels_csv)
        if labels_csv_path != args.labels_csv:
            print(f"Using labels CSV: {labels_csv_path.resolve()}")
        out_csv_path = resolve_output_csv_path(args.out_csv)
        if out_csv_path != args.out_csv:
            print(f"Writing segments CSV to: {out_csv_path.resolve()}")
        labels_df = pd.read_csv(labels_csv_path)
        print(f"Loaded labels in {time.perf_counter() - t_labels:.2f}s")

        t_seg = time.perf_counter()
        cfg = SegmentBuilderConfig(
            tolerance_seconds=float(args.tolerance_hours) * 3600.0,
            min_duration_seconds=float(args.min_duration_hours) * 3600.0,
            max_duration_seconds=float(args.max_duration_days) * 86400.0,
        )
        seg_df = build_segments_from_tles_and_labels(tle_df, labels_df, cfg)
        seg_df.to_csv(out_csv_path, index=False)
        print(f"Segment build time: {time.perf_counter() - t_seg:.2f}s")
        print(f"Wrote {len(seg_df)} segments to {out_csv_path.resolve()}")
        print(f"Total build-segments elapsed: {time.perf_counter() - t_build:.2f}s")
        return

    if args.cmd == "train-stage-a":
        device_str = str(args.device).lower()
        compile_model = (not bool(args.no_compile)) and device_str.startswith("cuda")
        if not bool(args.no_compile) and not compile_model:
            print("torch.compile disabled on non-CUDA device; continuing without compile.")
        cfg = TrainConfig(
            device=args.device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            compile_model=compile_model,
        )
        ckpt_path, _ = train_stage_a(args.segments_csv, args.outdir, cfg)
        print(f"Saved Stage A checkpoint to {ckpt_path.resolve()}")
        if not bool(args.skip_validation):
            report_path = args.outdir / "stage_a_validation_report.json"
            report = run_stage_a_validation(
                segments_csv=args.segments_csv,
                checkpoint_path=ckpt_path,
                out_json=report_path,
                device=args.validation_device,
            )
            print(f"Saved Stage A validation report to {report_path.resolve()} (all_checks_pass={report['all_checks_pass']})")
        else:
            print("Skipped Stage A validation report (--skip-validation).")
        return

    if args.cmd == "validate-stage-a":
        out_json = args.out_json if args.out_json is not None else (args.checkpoint.parent / "stage_a_validation_report.json")
        report = run_stage_a_validation(
            segments_csv=args.segments_csv,
            checkpoint_path=args.checkpoint,
            out_json=out_json,
            device=args.device,
        )
        print(f"Saved Stage A validation report to {Path(out_json).resolve()} (all_checks_pass={report['all_checks_pass']})")
        return

    if args.cmd == "run-stage-b":
        run_stage_b_snpe(
            args.segments_csv,
            args.checkpoint,
            args.outdir,
            num_simulations=int(args.num_simulations),
            max_segments=int(args.max_segments),
            max_phase_parameters=int(args.max_phase_parameters),
            num_posterior_samples=int(args.num_posterior_samples),
            ppc_samples=int(args.ppc_samples),
            density_estimator=str(args.density_estimator),
            normalize_observation=(not bool(args.disable_normalize_observation)),
            include_initial_conditions=(not bool(args.disable_initial_conditions)),
            include_phase_context=(not bool(args.disable_phase_context)),
            include_rate_features=(not bool(args.disable_rate_features)),
            normalization_eps=float(args.normalization_eps),
            calibration_subset_segments=int(args.calibration_subset_segments),
            run_sbc=bool(args.run_sbc),
            sbc_draws=int(args.sbc_draws),
            sbc_posterior_samples=int(args.sbc_posterior_samples),
            device=args.device,
            effective_mode=(not bool(args.no_effective_mode)),
            anchor_from_stage_a=(not bool(args.no_anchor_from_stage_a)),
            mixed_precision=bool(args.mixed_precision),
        )
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
