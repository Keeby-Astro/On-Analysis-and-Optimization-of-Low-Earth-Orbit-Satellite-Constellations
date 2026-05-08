import os
import sys
import json
import argparse
import random
import platform
import warnings
import itertools
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import seaborn as sns
import umap
import hdbscan
import DBCV_fixed
from numba import njit
from sklearn.manifold import trustworthiness
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from filter_tle_data_by_date import filter_tle_data_by_date
from load_all_tle_data import load_all_tle_data

DEFAULT_GENERATION_SPLIT_PATH = str(Path(__file__).resolve().parents[1] / "starlink_generation_split")

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("HDBSCAN_UMAP_SEED", "42")),
                        help="Random seed (also read from HDBSCAN_UMAP_SEED).")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("HDBSCAN_UMAP_THREADS", "1")),
                        help="Thread count for libs that support it (best reproducibility: 1).")
    parser.add_argument("--outdir", type=str, default=os.environ.get("HDBSCAN_UMAP_OUTDIR", "runs"),
                        help="Output folder for run metadata.")
    parser.add_argument("--run-name", type=str, default=os.environ.get("HDBSCAN_UMAP_RUN_NAME"),
                        help="Optional run name prefix for artifacts.")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Disable writing run metadata JSON.")
    parser.add_argument("--cluster-plots-dir", type=str, default=os.environ.get("HDBSCAN_UMAP_CLUSTER_PLOTS_DIR", "cluster_plots"),
                        help="Folder for exported cluster plot bundles.")
    parser.add_argument("--plot-dpi", type=int, default=int(os.environ.get("HDBSCAN_UMAP_PLOT_DPI", "600")),
                        help="DPI for exported plots (high zoom detail).")
    parser.add_argument("--cluster-counts", type=str, default=os.environ.get("HDBSCAN_UMAP_CLUSTER_COUNTS"),
                        help="Optional comma-separated list of cluster counts to export (e.g., 11,12,17).")
    parser.add_argument("--dataset-choice", type=str, default=os.environ.get("HDBSCAN_UMAP_DATASET_CHOICE"),
                        help="Dataset folder selection: 1=gps_files_corrected, 2=oneweb_files_corrected, 3=starlink_files_corrected.")
    parser.add_argument("--starlink-inc-band", type=str, default=os.environ.get("HDBSCAN_UMAP_STARLINK_INC_BAND"),
                        help="Starlink inclination band option: 1..7 (used only when starlink dataset is selected).")
    parser.add_argument("--starlink-generation", type=str, default=os.environ.get("HDBSCAN_UMAP_STARLINK_GENERATION"),
                        help="Starlink generation selection: gen1, gen2, or all (used only when starlink dataset is selected).")
    parser.add_argument("--generation-split-root", type=str,
                        default=os.environ.get("HDBSCAN_UMAP_GENERATION_SPLIT_ROOT", DEFAULT_GENERATION_SPLIT_PATH),
                        help="Folder containing generation subfolders (gen1/gen2).")
    parser.add_argument("--cluster-labels-dir", type=str,
                        default=os.environ.get("HDBSCAN_UMAP_CLUSTER_LABELS_DIR", "cluster_labels"),
                        help="Folder for exported cluster label/stat CSV files.")
    parser.add_argument("--labels-float-format", type=str,
                        default=os.environ.get("HDBSCAN_UMAP_LABELS_FLOAT_FORMAT", "%.8g"),
                        help="Float format for label/stat CSV exports to reduce file size while keeping full data.")
    parser.add_argument("--pairplot-scientific-notation", action=argparse.BooleanOptionalAction,
                        default=_env_flag("HDBSCAN_UMAP_PAIRPLOT_SCIENTIFIC_NOTATION", False),
                        help="Use scientific notation on orbital-elements pairplot axes. Disabled by default.")
    args, _unknown = parser.parse_known_args()
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if args.plot_dpi < 72:
        raise ValueError("--plot-dpi must be >= 72")
    if args.dataset_choice is not None and args.dataset_choice not in {"1", "2", "3"}:
        raise ValueError("--dataset-choice must be one of: 1, 2, 3")
    if args.starlink_inc_band is not None and args.starlink_inc_band not in {"1", "2", "3", "4", "5", "6", "7"}:
        raise ValueError("--starlink-inc-band must be one of: 1, 2, 3, 4, 5, 6, 7")
    if args.starlink_generation is not None:
        args.starlink_generation = args.starlink_generation.strip().lower()
        if args.starlink_generation not in {"all", "gen1", "gen2"}:
            raise ValueError("--starlink-generation must be one of: all, gen1, gen2")
    if not args.labels_float_format.strip():
        raise ValueError("--labels-float-format must be a non-empty printf-style float format string")
    return args

def _choose_starlink_generation_interactive() -> str:
    print("Select Starlink generation filter:")
    print("  1) gen1")
    print("  2) gen2")
    print("  3) all")
    while True:
        choice = input("Enter 1-3 [default 1]: ").strip()
        if choice == "":
            return "gen1"
        if choice == "1":
            return "gen1"
        if choice == "2":
            return "gen2"
        if choice == "3":
            return "all"
        print("Invalid selection. Please enter 1, 2, or 3.")

def _resolve_generation_only_files(generation_selection: str | None, generation_split_root: str) -> list[str] | None:
    selection = (generation_selection or "all").strip().lower()
    if selection in {"", "all"}:
        return None

    generation_dir = os.path.join(generation_split_root, selection)
    if not os.path.isdir(generation_dir):
        raise FileNotFoundError(f"Generation directory not found: {generation_dir}. "
                                f"Provide --generation-split-root with valid gen1/gen2 folders.")

    allowed = sorted({name
                      for name in os.listdir(generation_dir)
                      if name.lower().endswith(".txt")})
    if not allowed:
        raise FileNotFoundError(f"No .txt files found in generation directory: {generation_dir}")

    return allowed

def _choose_starlink_inclination_band_interactive() -> str:
    print("Select Starlink inclination filter band:")
    print("  1) 42.95 - 43.05")
    print("  2) 52.95 - 53.10")
    print("  3) 53.10 - 53.17")
    print("  4) 53.17 - 53.25")
    print("  5) 52.95 - 53.25")
    print("  6) 69.90 - 70.10")
    print("  7) 97.50 - 97.75")
    while True:
        choice = input("Enter 1-7 [default 7]: ").strip()
        if choice == "":
            return "7"
        if choice in {"1", "2", "3", "4", "5", "6", "7"}:
            return choice
        print("Invalid selection. Please enter an integer from 1 to 7.")

def _filter_starlink_inclination_band(data: pd.DataFrame, folder_paths: list[str],
                                            band_choice: str | None) -> pd.DataFrame:
    is_starlink = any("starlink_files_corrected" in p for p in folder_paths)
    if not is_starlink:
        print("Inclination band filter skipped (dataset is not starlink).\n")
        return data

    band_map = {"1": (42.95, 43.05),
                "2": (52.95, 53.10),
                "3": (53.10, 53.17),
                "4": (53.17, 53.25),
                "5": (52.95, 53.25),
                "6": (69.90, 70.10),
                "7": (97.50, 97.75)}

    if band_choice is None:
        if sys.stdin is not None and sys.stdin.isatty():
            band_choice = _choose_starlink_inclination_band_interactive()
        else:
            band_choice = "7"

    low, high = band_map[band_choice]
    filtered = data[(data["inc"] > low) & (data["inc"] < high)]
    print("Applied Starlink inclination band "
          f"{band_choice}: {low:.2f} - {high:.2f} deg")
    print(f"Number of records after inclination band filtering: {len(filtered)}\n")
    return filtered

def _choose_dataset_interactive() -> str:
    print("Select TLE dataset folder:")
    print("  1) gps_files_corrected")
    print("  2) oneweb_files_corrected")
    print("  3) starlink_files_corrected")
    while True:
        choice = input("Enter 1, 2, or 3 [default 3]: ").strip()
        if choice == "":
            return "3"
        if choice in {"1", "2", "3"}:
            return choice
        print("Invalid selection. Please enter 1, 2, or 3.")

def _resolve_folder_path(dataset_choice: str | None) -> list[str]:
    base_dir = r"C:\Users\PC\Code\UMAP_HDBSCAN"
    dataset_map = {"1": "gps_files_corrected", 
                   "2": "oneweb_files_corrected",
                   "3": "starlink_files_corrected"}

    if dataset_choice is None:
        if sys.stdin is not None and sys.stdin.isatty():
            dataset_choice = _choose_dataset_interactive()
        else:
            dataset_choice = "3"

    folder_name = dataset_map[dataset_choice]
    selected = os.path.join(base_dir, folder_name)
    print(f"Dataset selection: {dataset_choice} -> {folder_name}")
    return [selected]

def _parse_cluster_count_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    parsed = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(f"Invalid cluster count in --cluster-counts: '{tok}'")
        parsed.append(int(tok))
    if not parsed:
        return None
    return sorted(set(parsed))

def _set_thread_envvars(threads: int) -> None:
    # For best reproducibility, single-thread math is the safest default
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(threads))

def _set_global_seed(seed: int) -> None:
    # Note: PYTHONHASHSEED must be set before interpreter start to fully apply.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as _np  # local import to allow early seeding call
        _np.random.seed(seed)
    except Exception:
        # Numpy may not be importable yet; we'll seed again after importing numpy.
        pass

def _safe_pkg_version(pkg_name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return None

def _write_run_metadata(outdir: str, seed: int, threads: int, run_name: str | None,
                        extra: dict | None = None) -> str | None:
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{run_name}_" if run_name else ""
    path = os.path.join(outdir, f"{prefix}run_{timestamp}_seed{seed}.json")

    metadata = {
        "timestamp_local": timestamp,
        "seed": seed,
        "threads": threads,
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "env": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
                "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS")},
        "packages": {"numpy": _safe_pkg_version("numpy"),
                     "pandas": _safe_pkg_version("pandas"),
                     "scipy": _safe_pkg_version("scipy"),
                     "scikit-learn": _safe_pkg_version("scikit-learn"),
                     "matplotlib": _safe_pkg_version("matplotlib"),
                     "seaborn": _safe_pkg_version("seaborn"),
                     "umap-learn": _safe_pkg_version("umap-learn"),
                     "hdbscan": _safe_pkg_version("hdbscan"),
                     "numba": _safe_pkg_version("numba")}}
    if extra:
        metadata["extra"] = extra

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return path
    except Exception:
        return None

ARGS = _parse_args()
_set_thread_envvars(ARGS.threads)
_set_global_seed(ARGS.seed)

# Seed numpy again after import to guarantee it applied.
np.random.seed(ARGS.seed)
if not ARGS.no_metadata:
    _meta_path = _write_run_metadata(outdir=ARGS.outdir, seed=ARGS.seed, threads=ARGS.threads,
                                     run_name=ARGS.run_name, extra={"note": "For best reproducibility, run with --threads 1."})
    if _meta_path:
        print(f"Run metadata written to: {_meta_path}\n")

# UPDATE FIGURE SETTINGS & DEFINE CUSTOM PALETTE
plt.rcParams.update({'figure.figsize': (10.0, 7.5),
                     'xtick.direction': 'in', 'xtick.labelsize': 22, 'xtick.major.size': 3,
                     'xtick.major.width': 0.5, 'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5,
                     'xtick.minor.visible': True, 'xtick.top': True,
                     'ytick.direction': 'in', 'ytick.labelsize': 22, 'ytick.major.size': 3,
                     'ytick.major.width': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
                     'ytick.minor.visible': True, 'ytick.right': True,
                     'axes.linewidth': 0.5, 'grid.linewidth': 0.5, 'lines.linewidth': 1.0,
                     'legend.fontsize': 20, 'legend.frameon': False,
                     'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 24, 'axes.labelsize': 24, 'axes.titlesize': 18,
                     'axes.grid': True, 'grid.linestyle': '--', 'grid.color': '0.5',
                     'lines.markersize': 8, 'axes.spines.top': True, 'axes.spines.right': True})

# Define the custom 20-color palette (darkened colors)
colors = ['#15528e', '#b25800', '#1e701e', '#951c1c', '#673284', 
          '#623c34', '#9e5387', '#585858', '#848417', '#108590',
          '#798ba2', '#b28254', '#6a9c60', '#b26a68', '#8a7b94',
          '#896d67', '#ac7f93', '#8b8b8b', '#999962', '#6f989f']

# Define 20 distinct marker styles for cycling in scatter plots
markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', 'h', 'H',
           'X', '*', '+', 'x', '1', '2', '3', '4', 'd', 'P']

# Pairplots use Seaborn style mapping, which cannot mix filled and line-art markers.
pairplot_markers_filled = ['o', 's', '^', 'v', '<', '>', 'D', 'd', 'p', 'P', 'h', 'H', 'X', '8']

def build_pairplot_style_maps(unique_cluster_labels, color_cycle, marker_cycle):
    """Build pairplot color/marker maps with marker offset per color-cycle pass."""
    cluster_colors_out = {}
    cluster_markers_out = []
    n_colors = len(color_cycle)
    n_markers = len(marker_cycle)

    for i, label in enumerate(unique_cluster_labels):
        color_idx = i % n_colors
        cycle_idx = i // n_colors
        marker_idx = (i + cycle_idx) % n_markers
        cluster_colors_out[label] = color_cycle[color_idx]
        cluster_markers_out.append(marker_cycle[marker_idx])

    return cluster_colors_out, cluster_markers_out

def _format_plain_pairplot_tick(value, _pos):
    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0"

    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    elif abs_value >= 1:
        text = f"{value:.2f}"
    elif abs_value >= 0.01:
        text = f"{value:.3f}"
    else:
        text = f"{value:.4f}"
    return text.rstrip("0").rstrip(".")

def _format_pairplot_axis(axis):
    use_scientific = getattr(ARGS, "pairplot_scientific_notation", False)
    if use_scientific:
        formatter = ScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
    else:
        formatter = FuncFormatter(_format_plain_pairplot_tick)
    axis.set_major_locator(MaxNLocator(nbins=1, min_n_ticks=2))
    axis.set_major_formatter(formatter)

def _format_degree_tick(value, _pos):
    return f"{value:.0f}"

def _set_pairplot_raan_axis(axis):
    axis.set_major_locator(FixedLocator([0, 360]))
    axis.set_major_formatter(FuncFormatter(_format_degree_tick))

def _set_pairplot_eccentricity_x_axis(ax):
    lower, upper = ax.get_xlim()
    ticks = [tick for tick in ax.get_xticks() if np.isfinite(tick) and lower <= tick <= upper]
    if len(ticks) < 2:
        return

    left_tick = ticks[0]
    right_tick = ticks[-1]
    if right_tick <= left_tick:
        return

    span = right_tick - left_tick
    ax.set_xlim(lower, right_tick + 0.25 * span)
    ax.xaxis.set_major_locator(FixedLocator([left_tick, right_tick]))

def _align_pairplot_x_tick_labels(ax, x_var=None):
    labels = [label for label in ax.get_xticklabels() if label.get_visible()]
    if not labels:
        return
    for label in labels:
        label.set_horizontalalignment("center")
    labels[0].set_horizontalalignment("left")
    labels[-1].set_horizontalalignment("right")

def _apply_orbital_pairplot_axis_style(grid, raan_label="RAAN [deg]"):
    for row_idx, ax_row in enumerate(grid.axes):
        for col_idx, ax in enumerate(ax_row):
            if ax is None:
                continue
            x_var = str(grid.x_vars[col_idx])
            y_var = str(grid.y_vars[row_idx])
            _format_pairplot_axis(ax.xaxis)
            _format_pairplot_axis(ax.yaxis)
            if x_var == raan_label:
                ax.set_xlim(0, 360)
                _set_pairplot_raan_axis(ax.xaxis)
            elif x_var == "Eccentricity":
                _set_pairplot_eccentricity_x_axis(ax)
            if y_var == raan_label:
                ax.set_ylim(0, 360)
                _set_pairplot_raan_axis(ax.yaxis)
            _align_pairplot_x_tick_labels(ax, x_var=x_var)

    if hasattr(grid, "diag_axes"):
        for col_idx, ax in enumerate(grid.diag_axes):
            if ax is None:
                continue
            x_var = str(grid.x_vars[col_idx])
            _format_pairplot_axis(ax.xaxis)
            _format_pairplot_axis(ax.yaxis)
            if x_var == raan_label:
                ax.set_xlim(0, 360)
                _set_pairplot_raan_axis(ax.xaxis)
            elif x_var == "Eccentricity":
                _set_pairplot_eccentricity_x_axis(ax)
            _align_pairplot_x_tick_labels(ax, x_var=x_var)

# Define marker sizes using a switch-case like dictionary.
# Markers not specified in this dictionary will default to 25.
marker_sizes = {'x': 30, '+': 35, '1': 40,
                '2': 40, '3': 40, '4': 40}

# Define 20 line styles (cycling through four basic styles)
linestyles = ['-', '--', '-.', ':'] * 5

custom_cmap = ListedColormap(colors)

@njit
def orb2xyz(mu, oe):
    p = oe[0] * (1 - oe[1]**2)
    # Compute position and velocity in perifocal coordinates using true anomaly (nu).
    cos_nu = np.cos(oe[5])
    sin_nu = np.sin(oe[5])
    r      = p / (1 + oe[1] * cos_nu)
    rf_vec = np.array([r * cos_nu, r * sin_nu, 0.0])
    factor = np.sqrt(mu / p)
    vf_vec = np.array([-factor * sin_nu,
                        factor * (oe[1] + cos_nu),
                        0.0])
    # Rotation from perifocal to inertial frame
    cos_w = np.cos(oe[3])
    sin_w = np.sin(oe[3])
    cos_O = np.cos(oe[4])
    sin_O = np.sin(oe[4])
    cos_i = np.cos(oe[2])
    sin_i = np.sin(oe[2])
    R = np.array([[cos_O * cos_w - sin_O * sin_w * cos_i,
                   -cos_O * sin_w - sin_O * cos_w * cos_i,
                    sin_O * sin_i],
                  [sin_O * cos_w + cos_O * sin_w * cos_i,
                   -sin_O * sin_w + cos_O * cos_w * cos_i,
                   -cos_O * sin_i],
                  [sin_w * sin_i,
                   cos_w * sin_i,
                   cos_i]])
    r_vec_inertial = R @ rf_vec
    v_vec_inertial = R @ vf_vec
    state_vec = np.concatenate((r_vec_inertial, v_vec_inertial))
    return state_vec

def _safe_row_normalize(vectors: np.ndarray, eps: float = 1e-12):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = (norms[:, 0] > eps)
    unit = np.zeros_like(vectors)
    unit[valid] = vectors[valid] / norms[valid]
    return unit, valid

def _print_vector_consistency_checks(r_vectors: np.ndarray, v_vectors: np.ndarray,
                                     h_vectors: np.ndarray, e_vectors: np.ndarray,
                                     n_vectors: np.ndarray, mu_val: float) -> None:
    h_from_rv = np.cross(r_vectors, v_vectors)
    h_residual = np.linalg.norm(h_vectors - h_from_rv, axis=1)

    r_norms = np.linalg.norm(r_vectors, axis=1, keepdims=True)
    valid_r = (r_norms[:, 0] > 1e-12)
    e_from_rv = np.zeros_like(e_vectors)
    e_from_rv[valid_r] = (np.cross(v_vectors[valid_r], h_from_rv[valid_r]) / mu_val
                          - (r_vectors[valid_r] / r_norms[valid_r]))
    e_residual = np.linalg.norm(e_vectors - e_from_rv, axis=1)

    n_from_h = np.cross(np.array([0.0, 0.0, 1.0]), h_from_rv)
    n_residual = np.linalg.norm(n_vectors - n_from_h, axis=1)

    print("Vector consistency checks:")
    print(f"  |h - (r x v)| median={np.median(h_residual):.3e}, max={np.max(h_residual):.3e}")
    print(f"  |e - e(r,v)| median={np.median(e_residual):.3e}, max={np.max(e_residual):.3e}")
    print(f"  |n - (k x h)| median={np.median(n_residual):.3e}, max={np.max(n_residual):.3e}\n")

def _save_figure(path: str, dpi: int) -> None:
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()

def _plot_condensed_tree(clusterer_obj, selection_palette, path: str | None = None,
                         dpi: int | None = None, show: bool = False) -> None:
    tree_style = {
        'figure.figsize': (10.0, 7.5),
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 18,
        'font.size': 24,
        'legend.fontsize': 20,
    }
    with plt.rc_context(tree_style):
        fig, ax = plt.subplots(figsize=(10.0, 7.5))
        clusterer_obj.condensed_tree_.plot(
            select_clusters=True,
            selection_palette=selection_palette,
            axis=ax,
        )
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

def _sanitize_export_tag(tag: str | int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(tag))
    return safe.strip("_") or "default"

def _write_dataframe_csv(df: pd.DataFrame, output_dir: str, stem: str,
                         float_format: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{stem}.csv")
    df.to_csv(out_path, index=False,
              float_format=float_format,
              date_format="%Y-%m-%d %H:%M:%S")
    return out_path

def _build_cluster_labels_export_df(data_local: pd.DataFrame,
                                    embedding_local: np.ndarray,
                                    labels_local: np.ndarray) -> pd.DataFrame:
    if len(data_local) != len(labels_local):
        raise ValueError("Data row count does not match label count for cluster label export.")
    if embedding_local.shape[0] != len(labels_local):
        raise ValueError("Embedding row count does not match label count for cluster label export.")

    selected_columns = [
        "timestamp", "filename", "file_name", "object_name", "sat_name", "sat_id",
        "norad_id", "satnum", "international_designator", "launch_year", "launch_num",
        "launch_piece", "sma", "ecc", "inc", "raan", "aop", "true_anomaly"
    ]
    available_columns = [col for col in selected_columns if col in data_local.columns]
    export_df = data_local.loc[:, available_columns].copy()
    export_df.insert(0, "record_index", np.arange(len(labels_local), dtype=np.int64))
    export_df["cluster_label"] = labels_local.astype(np.int64)
    export_df["is_noise"] = (labels_local == -1).astype(np.int8)
    export_df["umap_1"] = embedding_local[:, 0]
    export_df["umap_2"] = embedding_local[:, 1] if embedding_local.shape[1] > 1 else np.nan
    export_df["umap_3"] = embedding_local[:, 2] if embedding_local.shape[1] > 2 else np.nan
    return export_df

def _build_cluster_stats_export_df(data_local: pd.DataFrame,
                                   labels_local: np.ndarray) -> pd.DataFrame:
    if len(data_local) != len(labels_local):
        raise ValueError("Data row count does not match label count for cluster stats export.")

    cluster_counts = pd.Series(labels_local, name="cluster_label").value_counts().sort_index()
    stats_df = cluster_counts.rename("count").reset_index()
    stats_df["fraction"] = stats_df["count"] / float(len(labels_local))
    stats_df["is_noise"] = (stats_df["cluster_label"] == -1).astype(np.int8)

    numeric_columns = [col for col in ["sma", "ecc", "inc", "raan", "aop", "true_anomaly"]
                       if col in data_local.columns]
    if numeric_columns:
        grouped = data_local.loc[:, numeric_columns].copy()
        grouped["cluster_label"] = labels_local
        grouped = grouped.groupby("cluster_label")[numeric_columns].agg(["mean", "std", "min", "max"])
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
        grouped = grouped.reset_index()
        stats_df = stats_df.merge(grouped, on="cluster_label", how="left")

    return stats_df.sort_values("cluster_label").reset_index(drop=True)

def _export_cluster_labels_and_stats(output_dir: str, export_tag: str | int,
                                     data_local: pd.DataFrame,
                                     embedding_local: np.ndarray,
                                     labels_local: np.ndarray,
                                                                         float_format: str) -> tuple[str, str]:
    safe_tag = _sanitize_export_tag(export_tag)
    labels_df = _build_cluster_labels_export_df(data_local, embedding_local, labels_local)
    stats_df = _build_cluster_stats_export_df(data_local, labels_local)
    labels_path = _write_dataframe_csv(labels_df, output_dir,
                                       stem=f"cluster_labels_{safe_tag}",
                                                                             float_format=float_format)
    stats_path = _write_dataframe_csv(stats_df, output_dir,
                                      stem=f"cluster_stats_{safe_tag}",
                                                                            float_format=float_format)
    return labels_path, stats_path

def _export_cluster_plot_bundle(output_dir: str, cluster_count: int, embedding: np.ndarray, 
                                labels_local: np.ndarray, clusterer_local, sma_local: np.ndarray,
                                ecc_local: np.ndarray, inc_local: np.ndarray, raan_local: np.ndarray,
                                angular_momentum_pca_local: np.ndarray, eccentricity_pca_local: np.ndarray,
                                nodal_pca_local: np.ndarray, colors_local: list, markers_local: list,
                                marker_sizes_local: dict, rad_to_deg_local: float, dpi: int) -> None:
    bundle_dir = os.path.join(output_dir, f"clusters_{cluster_count}")
    os.makedirs(bundle_dir, exist_ok=True)

    # 1) Condensed tree plot
    _plot_condensed_tree(
        clusterer_local,
        colors_local,
        path=os.path.join(bundle_dir, "01_condensed_tree.png"),
        dpi=dpi,
    )

    # 2) UMAP cluster scatter
    fig = plt.figure(figsize=(10.0, 7.5))
    plt.subplots_adjust(right=0.755)
    for i, lbl in enumerate(np.unique(labels_local)):
        mask = labels_local == lbl
        marker = markers_local[i % len(markers_local)]
        color = 'gray' if lbl == -1 else colors_local[i % len(colors_local)]
        marker_size = marker_sizes_local.get(marker, 25)
        plt.scatter(embedding[mask, 0], embedding[mask, 1], label=f"Cluster {lbl}",
                    marker=marker, color=color, s=marker_size)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.025, 1), loc='upper left', fontsize=12, ncol=2)
    _save_figure(os.path.join(bundle_dir, "02_umap_scatter_clusters.png"), dpi=dpi)

    # 3) UMAP density ellipses
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    color_dict = {lbl: colors_local[i % len(colors_local)] if lbl != -1 else 'gray'
                  for i, lbl in enumerate(np.unique(labels_local))}
    for lbl in np.unique(labels_local):
        if lbl == -1:
            continue
        mask = labels_local == lbl
        color = color_dict[lbl]
        sns.kdeplot(x=embedding[mask, 0], y=embedding[mask, 1], fill=True, alpha=0.3,
                    levels=5, thresh=0.05, ax=ax, color=color)
        if np.sum(mask) > 1:
            mean_xy = (np.mean(embedding[mask, 0]), np.mean(embedding[mask, 1]))
            cov_xy = np.cov(embedding[mask, 0], embedding[mask, 1])
            eigvals, eigvecs = np.linalg.eig(cov_xy)
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * 2.0 * np.sqrt(eigvals)
            ellipse = mpatches.Ellipse(xy=mean_xy, width=width, height=height, angle=angle,
                                       edgecolor='black', facecolor='none', lw=2)
            ax.add_patch(ellipse)

    for i, lbl in enumerate(np.unique(labels_local)):
        mask = labels_local == lbl
        marker = markers_local[i % len(markers_local)]
        marker_size = marker_sizes_local.get(marker, 25)
        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=f"Cluster {lbl}",
                   marker=marker, color=color_dict[lbl], s=marker_size)
    ax.set_xlabel("UMAP Embedding 1")
    ax.set_ylabel("UMAP Embedding 2")
    _save_figure(os.path.join(bundle_dir, "03_umap_density_ellipses.png"), dpi=dpi)

    # 4) Pairplot of raw orbital elements
    unique_clusters_local = np.unique(labels_local)
    cluster_colors_local, cluster_markers_local = build_pairplot_style_maps(unique_clusters_local, colors_local,
                                                                            pairplot_markers_filled)
    df_orb_local = pd.DataFrame({"Semi-Major Axis [km]": sma_local, "Eccentricity": ecc_local,
                                 "Inclination [deg]": inc_local * rad_to_deg_local,
                                 "RAAN [deg]": raan_local * rad_to_deg_local, "cluster": labels_local})
    g = sns.pairplot(df_orb_local, hue="cluster", hue_order=unique_clusters_local.tolist(), aspect=1.15,
                     markers=cluster_markers_local, diag_kind="kde", corner=True,
                     palette=cluster_colors_local, plot_kws={'s': 20})
    if hasattr(g, "_legend") and g._legend is not None:
        g._legend.remove()
    for ax_row in g.axes:
        for ax in ax_row:
            if ax is not None:
                ax.grid(False)
                ax.tick_params(top=False)
                ax.spines["top"].set_visible(False)
    for ax in g.diag_axes:
        ax.tick_params(top=False)
        ax.tick_params(bottom=False)
        ax.spines["top"].set_visible(False)
    _apply_orbital_pairplot_axis_style(g)
    g.fig.savefig(os.path.join(bundle_dir, "04_pairplot_orbital_elements.png"), dpi=dpi, bbox_inches='tight')
    plt.close(g.fig)

    # 5) Pairplot of PCA-reduced vectors
    df_pca_local = pd.DataFrame({"Angular Momentum Vector": angular_momentum_pca_local.ravel(),
                                 "Eccentricty Vector": eccentricity_pca_local.ravel(),
                                 "Nodal Vector": nodal_pca_local.ravel(), "cluster": labels_local})
    g2 = sns.pairplot(df_pca_local, hue="cluster", hue_order=unique_clusters_local.tolist(), markers=cluster_markers_local,
                      diag_kind="kde", corner=True, palette=cluster_colors_local, plot_kws={'s': 20})
    if hasattr(g2, "_legend") and g2._legend is not None:
        g2._legend.remove()
    for ax_row in g2.axes:
        for ax in ax_row:
            if ax is not None:
                ax.grid(False)
                ax.tick_params(top=False)
                ax.spines["top"].set_visible(False)
    for ax in g2.diag_axes:
        ax.tick_params(top=False)
        ax.tick_params(bottom=False)
        ax.spines["top"].set_visible(False)
    g2.fig.savefig(os.path.join(bundle_dir, "05_pairplot_pca_vectors.png"), dpi=dpi, bbox_inches='tight')
    plt.close(g2.fig)

# GLOBAL SETTINGS & CONSTANTS
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")

mu           = 398600.4418    # Gravitational Parameter for Earth, km^3/s^2
r_E          = 6378.145       # Radius of Earth, km
J2           = 1.082635854e-3 # J2 Second Zonal Harmonic Perturbation Constant
mean_day     = 86164.09054    # Mean Sideral Day, s
sideral_year = 365.25636      # Sideral Year, d
grav_const   = 6.67430e-11    # Newtonian Gravitational Constant, m^3/kg/s^2
DEG_TO_RAD   = np.pi / 180    # Degrees to radians conversion
RAD_TO_DEG   = 180 / np.pi    # Radians to degrees conversion

# Specify the folders containing TLE files
folder_path = _resolve_folder_path(ARGS.dataset_choice)
total_files = sum(len(os.listdir(folder)) for folder in folder_path if os.path.exists(folder))
print(f"Total Files Processed: {total_files}\n")

# Check which folders exist
for folder in folder_path:
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' not found.")

is_starlink_dataset = any("starlink_files_corrected" in p for p in folder_path)
generation_selection = ARGS.starlink_generation
if is_starlink_dataset and generation_selection is None:
    if sys.stdin is not None and sys.stdin.isatty():
        generation_selection = _choose_starlink_generation_interactive()
    else:
        generation_selection = "all"

only_files_generation = None
if is_starlink_dataset:
    only_files_generation = _resolve_generation_only_files(generation_selection, ARGS.generation_split_root)
    if only_files_generation is None:
        print("Starlink generation filter: all (no generation file allowlist).")
    else:
        print("Starlink generation filter applied: "
              f"{generation_selection} ({len(only_files_generation)} files from {ARGS.generation_split_root}).")
else:
    print("Starlink generation filter skipped (dataset is not starlink).")

# DATA LOADING & PREPROCESSING
all_tle_data, filenames_array = load_all_tle_data(folder_path, only_files=only_files_generation)
print(f"Loaded TLE data shape: {all_tle_data.shape}\n")
fileNames = np.unique(filenames_array).tolist()
filenames_array_np = np.array(filenames_array)

# Target date for initial clustering
target_date    = datetime(2025, 4, 10)
time_tolerance = timedelta(days=0.5)

print(all_tle_data.columns)
print(all_tle_data['timestamp'].head())
print(f"Number of records before filtering: {len(all_tle_data)}\n")
initial_tle_data = filter_tle_data_by_date(all_tle_data, target_date, time_tolerance)
print(f"Number of records after filtering: {len(initial_tle_data)}\n")
initial_tle_data = _filter_starlink_inclination_band(initial_tle_data, folder_path, 
                                                           ARGS.starlink_inc_band)
export_data_base = initial_tle_data.reset_index(drop=True)

# Extract orbital elements and launch information
sma          = initial_tle_data['sma'].values
ecc          = initial_tle_data['ecc'].values
inc          = initial_tle_data['inc'].values * DEG_TO_RAD
raan         = initial_tle_data['raan'].values * DEG_TO_RAD
aop          = initial_tle_data['aop'].values * DEG_TO_RAD
ta           = initial_tle_data['true_anomaly'].values * DEG_TO_RAD
launch_year  = initial_tle_data['launch_year'].values
launch_num   = initial_tle_data['launch_num'].values
launch_piece = initial_tle_data['launch_piece'].values
launch_info  = np.column_stack((launch_year, launch_num, launch_piece))
epoch       = initial_tle_data['timestamp'].values

print(f"Epoch: {epoch[:1]}")

# Combine the orbital elements into a single array
orbital_elements = np.column_stack((sma, ecc, inc, aop, raan, ta))

# Calculate radius and velocity vectors for each set of orbital elements
radius_vectors   = []
velocity_vectors = []
for elements in orbital_elements:
    state_vec = orb2xyz(mu, elements)
    radius_vectors.append(state_vec[:3])
    velocity_vectors.append(state_vec[3:])

angular_momentum_vectors = []
eccentricity_vectors     = []
nodal_vectors            = []
k_vec = np.array([0, 0, 1])
for r_vec, v_vec in zip(radius_vectors, velocity_vectors):
    h_vec = np.cross(r_vec, v_vec)
    angular_momentum_vectors.append(h_vec)
    norm_r = np.linalg.norm(r_vec)
    e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / norm_r)
    eccentricity_vectors.append(e_vec)
    n_vec = np.cross(k_vec, h_vec)
    nodal_vectors.append(n_vec)

radius_vectors_np           = np.array(radius_vectors)
angular_momentum_vectors_np = np.array(angular_momentum_vectors)
eccentricity_vectors_np     = np.array(eccentricity_vectors)
nodal_vectors_np            = np.array(nodal_vectors)
velocity_vectors_np         = np.array(velocity_vectors)

_print_vector_consistency_checks(radius_vectors_np, velocity_vectors_np, angular_momentum_vectors_np,
                                 eccentricity_vectors_np, nodal_vectors_np, mu)

e_unit, e_valid = _safe_row_normalize(eccentricity_vectors_np)
h_unit, h_valid = _safe_row_normalize(angular_momentum_vectors_np)
perifocal_minor_axis_np = np.zeros_like(e_unit)
valid_perifocal = e_valid & h_valid
perifocal_minor_axis_np[valid_perifocal] = np.cross(h_unit[valid_perifocal], e_unit[valid_perifocal])
perifocal_minor_axis_np, _ = _safe_row_normalize(perifocal_minor_axis_np)

invalid_perifocal_count = np.count_nonzero(~valid_perifocal)
if invalid_perifocal_count:
    print(f"Warning: {invalid_perifocal_count} rows had near-zero eccentricity or angular momentum; "
          "perifocal minor-axis unit vectors were set to [0, 0, 0].\n")

pca = PCA(n_components=1)
radius_vectors_pca           = pca.fit_transform(radius_vectors_np)
angular_momentum_vectors_pca = pca.fit_transform(angular_momentum_vectors_np)
eccentricity_vectors_pca     = pca.fit_transform(eccentricity_vectors_np)
nodal_vectors_pca            = pca.fit_transform(nodal_vectors_np)
velocity_vectors_pca         = pca.fit_transform(velocity_vectors_np)

# Create a combined array of angular momentum and nodal vectors and scale it
ROPV = np.hstack((radius_vectors_np, velocity_vectors_np, eccentricity_vectors_np, angular_momentum_vectors_np, nodal_vectors_np))
#scaler = StandardScaler()
#ROPV_scaled = scaler.fit_transform(ROPV)
ROPV_scaled = ROPV

# 2D Visualization of Orbital Vectors and Full Orbits
fig = plt.figure(figsize=(10.0, 7.5))
plt.subplots_adjust(hspace=0.238)

ax1 = fig.add_subplot(2, 3, 1)
ax1.set_title("Radius Vectors")
ax1.set_xlabel("X (km)")
ax1.set_ylabel("Y (km)")
ax1.scatter(radius_vectors_np[:, 0], radius_vectors_np[:, 1],
            c=colors[0], marker=markers[0], s=marker_sizes.get(markers[0], 25))

ax2 = fig.add_subplot(2, 3, 2)
ax2.set_title("Angular Momentum Vectors")
ax2.set_xlabel("X (km²/s)")
ax2.set_ylabel("Y (km²/s)")
ax2.scatter(angular_momentum_vectors_np[:, 0], angular_momentum_vectors_np[:, 1],
            c=colors[1], marker=markers[0], s=marker_sizes.get(markers[0], 25))

ax3 = fig.add_subplot(2, 3, 3)
ax3.set_title("Eccentricity Vectors")
ax3.set_xlabel("X (dimensionless)")
ax3.set_ylabel("Y (dimensionless)")
ax3.scatter(eccentricity_vectors_np[:, 0], eccentricity_vectors_np[:, 1],
            c=colors[2], marker=markers[0], s=marker_sizes.get(markers[0], 25))

ax4 = fig.add_subplot(2, 3, 4)
ax4.set_title("Nodal Vectors")
ax4.set_xlabel("X (km²/s)")
ax4.set_ylabel("Y (km²/s)")
ax4.scatter(nodal_vectors_np[:, 0], nodal_vectors_np[:, 1],
            c=colors[3], marker=markers[0], s=marker_sizes.get(markers[0], 25))

ax5 = fig.add_subplot(2, 3, 5)
ax5.set_title("Velocity Vectors")
ax5.set_xlabel("X (km/s)")
ax5.set_ylabel("Y (km/s)")
ax5.scatter(velocity_vectors_np[:, 0], velocity_vectors_np[:, 1],
            c=colors[4], marker=markers[0], s=marker_sizes.get(markers[0], 25))

ax6 = fig.add_subplot(2, 3, 6)
ax6.set_title("Perifocal Minor Axis Vectors")
ax6.set_xlabel("X (dimensionless)")
ax6.set_ylabel("Y (dimensionless)")
ax6.scatter(perifocal_minor_axis_np[:, 0], perifocal_minor_axis_np[:, 1],
            c=colors[5], marker=markers[0], s=marker_sizes.get(markers[0], 25))
    
plt.tight_layout()
plt.subplots_adjust(top=0.92)
# save the figure
_save_figure(os.path.join(ARGS.outdir, "00_orbital_vectors_2d.png"), dpi=600)
#plt.show()

'''
# 3D Visualization of Orbital Vectors Subplots
fig = plt.figure(figsize=(10.0, 7.5))
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.set_title("Radius Vectors")
ax1.set_xlabel("X (km)")
ax1.set_ylabel("Y (km)")
ax1.set_zlabel("Z (km)")
ax1.scatter(radius_vectors_np[:, 0], radius_vectors_np[:, 1], radius_vectors_np[:, 2],
            c=colors[0], marker=markers[0], s=marker_sizes.get(markers[0], 25))
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.set_title("Angular Momentum Vectors")
ax2.set_xlabel("X (km²/s)")
ax2.set_ylabel("Y (km²/s)")
ax2.set_zlabel("Z (km²/s)")
ax2.scatter(angular_momentum_vectors_np[:, 0], angular_momentum_vectors_np[:, 1], angular_momentum_vectors_np[:, 2],
            c=colors[1], marker=markers[0], s=marker_sizes.get(markers[0], 25))
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.set_title("Eccentricity Vectors")
ax3.set_xlabel("X (dimensionless)")
ax3.set_ylabel("Y (dimensionless)")
ax3.set_zlabel("Z (dimensionless)")
ax3.scatter(eccentricity_vectors_np[:, 0], eccentricity_vectors_np[:, 1], eccentricity_vectors_np[:, 2],
            c=colors[2], marker=markers[0], s=marker_sizes.get(markers[0], 25))
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.set_title("Nodal Vectors")
ax4.set_xlabel("X (km²/s)")
ax4.set_ylabel("Y (km²/s)")
ax4.set_zlabel("Z (km²/s)")
ax4.scatter(nodal_vectors_np[:, 0], nodal_vectors_np[:, 1], nodal_vectors_np[:, 2],
            c=colors[3], marker=markers[0], s=marker_sizes.get(markers[0], 25))
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax5.set_title("Velocity Vectors")
ax5.set_xlabel("X (km/s)")
ax5.set_ylabel("Y (km/s)")
ax5.set_zlabel("Z (km/s)")
ax5.scatter(velocity_vectors_np[:, 0], velocity_vectors_np[:, 1], velocity_vectors_np[:, 2],
            c=colors[4], marker=markers[0], s=marker_sizes.get(markers[0], 25))
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
ax6.set_title("Perifocal Minor Axis Vectors")
ax6.set_xlabel("X (dimensionless)")
ax6.set_ylabel("Y (dimensionless)")
ax6.set_zlabel("Z (dimensionless)")
ax6.scatter(perifocal_minor_axis_np[:, 0], perifocal_minor_axis_np[:, 1], perifocal_minor_axis_np[:, 2],
            c=colors[5], marker=markers[0], s=marker_sizes.get(markers[0], 25))
plt.tight_layout()
plt.subplots_adjust(top=0.92)
#plt.show()
'''

# UMAP PARAMETER SWEEP (Stage 1)
umap_n_neighbors_list = list(range(2, 13))
umap_min_dist_list = np.linspace(0.1, 0.95, 20)
# Take half the list of min_dist values 
#umap_min_dist_list = umap_min_dist_list[:len(umap_min_dist_list) // 2]
umap_results = []
print("Sweeping UMAP parameters using trustworthiness...")
for n_neighbors, min_dist in itertools.product(umap_n_neighbors_list, umap_min_dist_list):
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                        metric='euclidean', random_state=ARGS.seed, densmap=False, n_jobs=ARGS.threads)
    embedding = reducer.fit_transform(ROPV_scaled)
    tw = trustworthiness(ROPV_scaled, embedding, n_neighbors=n_neighbors)
    umap_results.append({'n_neighbors': n_neighbors,
                         'min_dist': min_dist,
                         'trustworthiness': tw,
                         'embedding': embedding})
    print(f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist:.3f}): Trustworthiness = {tw:.3f}")
umap_df = pd.DataFrame(umap_results)
best_umap_idx = umap_df['trustworthiness'].idxmax()
best_umap_row = umap_df.loc[best_umap_idx]
best_umap_params = (best_umap_row['n_neighbors'], best_umap_row['min_dist'])
best_umap_embedding = best_umap_row['embedding']
print(f"\nBest UMAP parameters: n_neighbors={best_umap_params[0]}, min_dist={best_umap_params[1]:.3f} with Trustworthiness={best_umap_row['trustworthiness']:.3f}")

# HDBSCAN PARAMETER SWEEP (Stage 2)
hdbscan_min_cluster_size_list = list(range(2, 15))
hdbscan_min_samples_list = list(range(1, 14))
hdbscan_results = []
best_hdbscan_score = -np.inf
best_hdbscan_config = None
best_cluster_labels = None
print("\nSweeping HDBSCAN parameters using DBCV on best UMAP embedding...")
for min_cluster_size, min_samples in itertools.product(hdbscan_min_cluster_size_list, hdbscan_min_samples_list):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='euclidean',
                                core_dist_n_jobs=ARGS.threads,
                                approx_min_span_tree=False)
    clusterer.fit(best_umap_embedding)
    try:
        score = DBCV_fixed.DBCV(best_umap_embedding, clusterer.labels_, dist_function='euclidean', use_gpu=True)
    except Exception as e:
        score = None
        print(f"DBCV failed for HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}): {e}")
    # Compute number of clusters (excluding noise -1)
    num_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    print(f"HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}): DBCV = {score}", 
          f"Number of clusters = {num_clusters}")
    hdbscan_results.append({'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'DBCV_score': score,
                            'labels': clusterer.labels_,
                            'num_clusters': num_clusters})
    # Also track the overall best
    if score is not None and score > best_hdbscan_score:
        best_hdbscan_score = score
        best_hdbscan_config = (min_cluster_size, min_samples)
        best_cluster_labels = clusterer.labels_

# Group results by unique number of clusters and pick best config for each group
best_config_for_clusters = {}
for result in hdbscan_results:
    if result['DBCV_score'] is None:
        continue
    n_clusters = result['num_clusters']
    if n_clusters not in best_config_for_clusters or result['DBCV_score'] > best_config_for_clusters[n_clusters]['DBCV_score']:
        best_config_for_clusters[n_clusters] = result

print("\nBest HDBSCAN parameter configurations for unique number of clusters:")
for n_clusters in sorted(best_config_for_clusters.keys()):
    config = best_config_for_clusters[n_clusters]
    print(f"For {n_clusters} clusters: min_cluster_size = {config['min_cluster_size']}, min_samples = {config['min_samples']}, DBCV = {config['DBCV_score']}")

# Ask user to choose desired number of clusters or "highest" for the best overall DBCV configuration
user_input = input("\nInput the desired number of clusters from the above list, or type 'highest' for the best overall configuration based on DBCV: ").strip().lower()

if user_input == "highest":
    print("Using best overall configuration with highest DBCV.")
    # The overall best configuration is already stored in best_hdbscan_config, best_hdbscan_score, and best_cluster_labels.
elif user_input.isdigit():
    user_choice = int(user_input)
    if user_choice in best_config_for_clusters:
        chosen_config = best_config_for_clusters[user_choice]
        print(f"Chosen configuration for {user_choice} clusters: min_cluster_size = {chosen_config['min_cluster_size']}, min_samples = {chosen_config['min_samples']}, DBCV = {chosen_config['DBCV_score']}")
        best_hdbscan_config = (chosen_config['min_cluster_size'], chosen_config['min_samples'])
        best_hdbscan_score = chosen_config['DBCV_score']
        best_cluster_labels = chosen_config['labels']
    else:
        print("Invalid choice or number of clusters not available. Using overall best configuration.")
else:
    print("Invalid input. Using overall best configuration.")

requested_cluster_counts = _parse_cluster_count_list(ARGS.cluster_counts)
if requested_cluster_counts is None:
    requested_cluster_counts = sorted(best_config_for_clusters.keys())
else:
    requested_cluster_counts = [c for c in requested_cluster_counts if c in best_config_for_clusters]
    if not requested_cluster_counts:
        requested_cluster_counts = sorted(best_config_for_clusters.keys())
        print("No requested cluster counts matched available configurations; "
              "defaulting to all available cluster counts.")
    
# Convert HDBSCAN results to DataFrame for any further analysis if needed
hdbscan_df = pd.DataFrame(hdbscan_results)

# FINAL CLUSTERING VISUALIZATION COMPUTATIONS
umap_reducer = umap.UMAP(n_components=3, n_neighbors=best_umap_params[0],
                         min_dist=best_umap_params[1], metric='euclidean', random_state=ARGS.seed, densmap=False, n_jobs=ARGS.threads)
embedding_umap = umap_reducer.fit_transform(ROPV_scaled)

clusterer = hdbscan.HDBSCAN(min_cluster_size=best_hdbscan_config[0],
                            min_samples=best_hdbscan_config[1],
                            metric='euclidean',
                            core_dist_n_jobs=ARGS.threads,
                            approx_min_span_tree=False)
clusterer.fit(embedding_umap)
labels = clusterer.labels_

best_cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
best_labels_path, best_stats_path = _export_cluster_labels_and_stats(
    output_dir=ARGS.cluster_labels_dir,
    export_tag=f"best_{best_cluster_count}_clusters",
    data_local=export_data_base,
    embedding_local=embedding_umap,
    labels_local=labels,
    float_format=ARGS.labels_float_format,
)
print(f"Saved best-clustering label CSV to: {best_labels_path}")
print(f"Saved best-clustering stats CSV to: {best_stats_path}\n")

# Smooth Contour Maps for UMAP and HDBSCAN Parameter Sweeps (Subplots)
# Prepare data for UMAP sweep
x_umap = umap_df['n_neighbors'].values
y_umap = umap_df['min_dist'].values
z_umap = umap_df['trustworthiness'].values
xi_umap = np.linspace(x_umap.min(), x_umap.max(), 300)
yi_umap = np.linspace(y_umap.min(), y_umap.max(), 300)
Xi_umap, Yi_umap = np.meshgrid(xi_umap, yi_umap)
Zi_umap = griddata((x_umap, y_umap), z_umap, (Xi_umap, Yi_umap), method='cubic')

# Prepare data for HDBSCAN sweep
x_hdb = hdbscan_df['min_cluster_size'].values
y_hdb = hdbscan_df['min_samples'].values
z_hdb = pd.to_numeric(hdbscan_df['DBCV_score'], errors='coerce').values
xi_hdb = np.linspace(x_hdb.min(), x_hdb.max(), 300)
yi_hdb = np.linspace(y_hdb.min(), y_hdb.max(), 300)
Xi_hdb, Yi_hdb = np.meshgrid(xi_hdb, yi_hdb)
Zi_hdb = griddata((x_hdb, y_hdb), z_hdb, (Xi_hdb, Yi_hdb), method='cubic')

# Prepare combined data
combined_df = pd.merge(umap_df[['n_neighbors', 'min_dist', 'trustworthiness']],
                       hdbscan_df[['min_cluster_size', 'min_samples', 'DBCV_score']],
                       left_index=True, right_index=True, how='outer').dropna(subset=['trustworthiness','DBCV_score'])
combined_df['combined_score'] = (combined_df['trustworthiness'] + 
                                 pd.to_numeric(combined_df['DBCV_score'], errors='coerce')) / 2
scaler_umap = StandardScaler()
scaler_hdb = StandardScaler()
umap_params = scaler_umap.fit_transform(combined_df[['n_neighbors', 'min_dist']])
hdb_params = scaler_hdb.fit_transform(combined_df[['min_cluster_size', 'min_samples']])
pca_umap = PCA(n_components=1)
pca_hdb = PCA(n_components=1)
umap_1d = pca_umap.fit_transform(umap_params)
hdb_1d = pca_hdb.fit_transform(hdb_params)
combined_df['UMAP_1D'] = umap_1d.flatten()
combined_df['HDBSCAN_1D'] = hdb_1d.flatten()
xi_comb = np.linspace(combined_df['UMAP_1D'].min(), combined_df['UMAP_1D'].max(), 300)
yi_comb = np.linspace(combined_df['HDBSCAN_1D'].min(), combined_df['HDBSCAN_1D'].max(), 300)
zi_comb = griddata((combined_df['UMAP_1D'], combined_df['HDBSCAN_1D']),
                   combined_df['combined_score'], (xi_comb[None, :], yi_comb[:, None]), method='cubic')

# Create a figure with 3 subplots for smooth contour maps
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 7.5))

# UMAP smooth contour map
contour_umap = ax1.contourf(Xi_umap, Yi_umap, Zi_umap, levels=750, cmap='viridis')
fig.colorbar(contour_umap, ax=ax1, label='Trustworthiness')
ax1.set_xlabel('n_neighbors')
ax1.set_ylabel('min_dist')
ax1.grid(False)

# HDBSCAN smooth contour map
contour_hdb = ax2.contourf(Xi_hdb, Yi_hdb, Zi_hdb, levels=750, cmap='viridis')
fig.colorbar(contour_hdb, ax=ax2, label='DBCV Score')
ax2.set_xlabel('min_cluster_size')
ax2.set_ylabel('min_samples')
ax2.grid(False)

# Combined parameter space contour map
contour_comb = ax3.contourf(xi_comb, yi_comb, zi_comb, levels=750, cmap='viridis')
fig.colorbar(contour_comb, ax=ax3, label='Combined Score')
ax3.set_xlabel("UMAP Parameters (1D)")
ax3.set_ylabel("HDBSCAN Parameters (1D)")
ax3.grid(False)

plt.tight_layout()
#plt.show()

print("\nBest overall HDBSCAN parameters:")
print(f"min_cluster_size = {best_hdbscan_config[0]}, min_samples = {best_hdbscan_config[1]}, with DBCV = {best_hdbscan_score}")

# ------------------------------
# FINAL CLUSTERING VISUALIZATION
# ------------------------------
def plot_umap_scatter(embedding, labels=None):
    plt.figure(figsize=(10.0, 7.5))
    plt.subplots_adjust(right=0.755)
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            marker = markers[i % len(markers)]
            # Use gray for noise (-1), otherwise pick from the custom colors
            color = 'gray' if lbl == -1 else colors[i % len(colors)]
            marker_size = marker_sizes.get(marker, 25)
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                        label=f"Cluster {lbl}", marker=marker, color=color, s=marker_size)
        plt.legend(bbox_to_anchor=(1.025, 1), loc='upper left', fontsize=12, ncol=2)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], color='b', s=25)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    #plt.show()

def get_color_dict(labels):
    unique = np.unique(labels)
    color_dict = {-1: 'gray'}
    for idx, lbl in enumerate(unique):
        if lbl >= 0:
            color_dict[lbl] = colors[idx % len(colors)]
    return color_dict

def plot_density_ellipses(embedding, labels):
    def plot_confidence(ax, mean, cov, n_std=2.0):
        eigvals, eigvecs = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        ellipse = mpatches.Ellipse(xy=mean, width=width, height=height,
                                   angle=angle, edgecolor='black', facecolor='none', lw=2)
        ax.add_patch(ellipse)
    
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    
    # Create a color dictionary for consistent coloring
    color_dict = {lbl: colors[i % len(colors)] if lbl != -1 else 'gray' 
                  for i, lbl in enumerate(np.unique(labels))}
    
    # First plot the density contours with matching colors
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        mask = (labels == lbl)
        color = color_dict[lbl]
        sns.kdeplot(x=embedding[mask, 0], y=embedding[mask, 1],
                    fill=True, alpha=0.3, levels=5, thresh=0.05, ax=ax,
                    color=color)  # Use the same color as the scatter points
        if np.sum(mask) > 1:
            mean_xy = (np.mean(embedding[mask, 0]), np.mean(embedding[mask, 1]))
            cov_xy = np.cov(embedding[mask, 0], embedding[mask, 1])
            plot_confidence(ax, mean_xy, cov_xy)
    
    # Plot scatter points with the same colors
    unique_labels = np.unique(labels)
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        marker = markers[i % len(markers)]
        color = color_dict[lbl]
        marker_size = marker_sizes.get(marker, 25)
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   label=f"Cluster {lbl}", marker=marker, color=color, s=marker_size)
    
    ax.set_xlabel("UMAP Embedding 1")
    ax.set_ylabel("UMAP Embedding 2")
    plt.tight_layout()
    #plt.show()

# Plot the condensed tree of HDBSCAN
_plot_condensed_tree(clusterer, colors)

# Plot final results using the helper functions
plot_umap_scatter(embedding_umap)
plot_umap_scatter(embedding_umap, labels=labels)
plot_density_ellipses(embedding_umap, labels)

# Plot 3d UMAP embedding with cluster coloring
fig = plt.figure(figsize=(10.0, 7.5))
ax = fig.add_subplot(111, projection='3d')
color_dict = get_color_dict(labels)
for lbl in np.unique(labels):
    mask = labels == lbl
    marker = markers[np.where(np.unique(labels) == lbl)[0][0] % len(markers)]
    color = color_dict[lbl]
    marker_size = marker_sizes.get(marker, 25)
    ax.scatter(embedding_umap[mask, 0], embedding_umap[mask, 1], embedding_umap[mask, 2],
               label=f"Cluster {lbl}", marker=marker, color=color, s=marker_size)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")
ax.legend(bbox_to_anchor=(1.025, 1), loc='upper left', fontsize=12, ncol=2)
plt.tight_layout()
#plt.show()

# Pairplots for Raw Orbital Elements
unique_clusters = np.unique(labels)
cluster_colors, cluster_markers = build_pairplot_style_maps(unique_clusters, colors, pairplot_markers_filled)

df_orb = pd.DataFrame({"Semi-Major Axis [km]": sma,
                       "Eccentricity": ecc,
                       "Inclination [deg]": inc * RAD_TO_DEG,
                       "RAAN [deg]": raan * RAD_TO_DEG,
                       "cluster": labels})

g = sns.pairplot(df_orb, hue="cluster", hue_order=unique_clusters.tolist(),
                 markers=cluster_markers, diag_kind="kde", corner=True,
                 palette=cluster_colors, plot_kws={'s': 20})

# Remove the legend from the PairGrid if it exists
if hasattr(g, "_legend") and g._legend is not None:
    g._legend.remove()

for ax_row in g.axes:
    for ax in ax_row:
        if ax is not None:
            ax.grid(False)
            ax.tick_params(top=False)
            ax.spines["top"].set_visible(False)

for ax in g.diag_axes:
    ax.tick_params(top=False)
    ax.tick_params(bottom=False)
    ax.spines["top"].set_visible(False)

_apply_orbital_pairplot_axis_style(g)

#plt.show()
plt.close(g.fig)

# Pairplots for PCA-Reduced Vectors
df_pca = pd.DataFrame({"Angular Momentum Vector": angular_momentum_vectors_pca.ravel(),
                       "Eccentricty Vector": eccentricity_vectors_pca.ravel(),
                       "Nodal Vector": nodal_vectors_pca.ravel(),
                       "cluster": labels})

g2 = sns.pairplot(df_pca, hue="cluster", hue_order=unique_clusters.tolist(),
                 markers=cluster_markers, diag_kind="kde", corner=True,
                 palette=cluster_colors, plot_kws={'s': 20})

if hasattr(g2, "_legend") and g2._legend is not None:
    g2._legend.remove()

for ax_row in g2.axes:
    for ax in ax_row:
        if ax is not None:
            ax.grid(False)
            ax.tick_params(top=False)
            ax.spines["top"].set_visible(False)

for ax in g2.diag_axes:
    ax.tick_params(top=False)
    ax.tick_params(bottom=False)
    ax.spines["top"].set_visible(False)

#plt.show()
plt.close(g2.fig)

os.makedirs(ARGS.cluster_plots_dir, exist_ok=True)
print(f"\nExporting high-DPI cluster plot bundles to '{ARGS.cluster_plots_dir}' "
      f"for cluster counts: {requested_cluster_counts}")

for cluster_count in requested_cluster_counts:
    chosen_cfg = best_config_for_clusters[cluster_count]
    export_clusterer = hdbscan.HDBSCAN(min_cluster_size=chosen_cfg['min_cluster_size'],
                                       min_samples=chosen_cfg['min_samples'],
                                       metric='euclidean',
                                       core_dist_n_jobs=ARGS.threads,
                                       approx_min_span_tree=False)
    export_clusterer.fit(embedding_umap)
    labels_export = export_clusterer.labels_
    labels_path_export, stats_path_export = _export_cluster_labels_and_stats(
        output_dir=ARGS.cluster_labels_dir,
        export_tag=f"{cluster_count}_clusters",
        data_local=export_data_base,
        embedding_local=embedding_umap,
        labels_local=labels_export,
        float_format=ARGS.labels_float_format,
    )
    _export_cluster_plot_bundle(output_dir=ARGS.cluster_plots_dir, cluster_count=cluster_count, embedding=embedding_umap,
                                labels_local=labels_export, clusterer_local=export_clusterer, sma_local=sma, ecc_local=ecc,
                                inc_local=inc, raan_local=raan, angular_momentum_pca_local=angular_momentum_vectors_pca,
                                eccentricity_pca_local=eccentricity_vectors_pca, nodal_pca_local=nodal_vectors_pca,
                                colors_local=colors, markers_local=markers, marker_sizes_local=marker_sizes,
                                rad_to_deg_local=RAD_TO_DEG, dpi=ARGS.plot_dpi)
    print(f"Saved plot bundle for {cluster_count} clusters "
          f"(min_cluster_size={chosen_cfg['min_cluster_size']}, "
          f"min_samples={chosen_cfg['min_samples']})")
    print(f"Saved label CSV: {labels_path_export}")
    print(f"Saved stats CSV: {stats_path_export}")

print(f"\nAll plot bundles saved in: {os.path.abspath(ARGS.cluster_plots_dir)}")
print(f"All label/stat CSV files saved in: {os.path.abspath(ARGS.cluster_labels_dir)}")