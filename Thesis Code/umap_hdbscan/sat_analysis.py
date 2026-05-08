import os
import sys
import json
import argparse
import random
import platform
import warnings
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from numba import njit

from filter_tle_data_by_date import filter_tle_data_by_date
from load_all_tle_data import load_all_tle_data


DEFAULT_STARLINK_PROFILE = "fullrun"
STARLINK_PROFILE_DIR_MAP = {DEFAULT_STARLINK_PROFILE: "starlink_generation_split"}

STARLINK_GENERATIONS = {"gen1", "gen2", "proto", "unknown"}
CLUSTER_COLOR_MODES = {"auto", "on", "off"}

_GLOBAL_CLUSTER_LABELS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "clusters", "global_analysis", "tables", "combined_cluster_labels_global.csv",
)

_CLUSTER_COLORS = [
    "#15528e", "#b25800", "#1e701e", "#951c1c", "#673284",
    "#623c34", "#9e5387", "#585858", "#848417", "#108590",
    "#798ba2", "#b28254", "#6a9c60", "#b26a68", "#8a7b94",
    "#896d67", "#ac7f93", "#8b8b8b", "#999962", "#6f989f",
]

_CLUSTER_MARKERS = [
    "o", "s", "^", "v", "<", ">", "D", "p", "h", "H",
    "X", "*", "+", "x", "1", "2", "3", "4", "d", "P",
]


def _load_global_cluster_label_map() -> dict[str, int] | None:
    """Load sat_id -> global_cluster_id mapping from pre-computed global analysis."""
    if not os.path.isfile(_GLOBAL_CLUSTER_LABELS_CSV):
        return None
    try:
        df = pd.read_csv(_GLOBAL_CLUSTER_LABELS_CSV, usecols=["sat_id", "global_cluster_id"])
        return dict(zip(df["sat_id"].astype(str), df["global_cluster_id"].astype(int)))
    except Exception as exc:
        print(f"Warning: could not load global cluster labels: {exc}")
        return None


def _build_cluster_style_maps(cluster_ids: np.ndarray) -> tuple[dict[int, str], dict[int, str]]:
    """Build color and marker maps for global cluster IDs (0 = noise)."""
    positive_ids = sorted(int(v) for v in set(cluster_ids) if int(v) > 0)
    color_map: dict[int, str] = {0: "gray"}
    marker_map: dict[int, str] = {0: "x"}
    n_colors = len(_CLUSTER_COLORS)
    n_markers = len(_CLUSTER_MARKERS)
    for i, cid in enumerate(positive_ids):
        color_idx = i % n_colors
        marker_repeat_idx = i // n_markers
        marker_idx = (i + marker_repeat_idx) % n_markers
        color_map[cid] = _CLUSTER_COLORS[color_idx]
        marker_map[cid] = _CLUSTER_MARKERS[marker_idx]
    return color_map, marker_map


def _resolve_cluster_arrays(
    tle_data: pd.DataFrame,
    label_map: dict[str, int] | None,
) -> tuple[np.ndarray | None, dict[int, str] | None, dict[int, str] | None]:
    """Map TLE rows to global cluster IDs and return (labels, color_map, marker_map) or Nones."""
    if label_map is None:
        return None, None, None
    sat_ids = tle_data["sat_id"].astype(str).to_numpy()
    labels = np.array([label_map.get(sid, -1) for sid in sat_ids], dtype=int)
    matched = int(np.sum(labels >= 0))
    if matched == 0:
        print("Warning: no cluster labels matched any TLE records.")
        return None, None, None
    print(f"Cluster labels matched {matched}/{len(labels)} TLE records.")
    # Treat unmatched as noise
    labels[labels < 0] = 0
    color_map, marker_map = _build_cluster_style_maps(labels)
    return labels, color_map, marker_map


def _scatter_by_cluster(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_color_map: dict[int, str],
    cluster_marker_map: dict[int, str],
    alpha_noise: float = 0.25,
    alpha_cluster: float = 0.70,
) -> None:
    """Plot scatter with per-cluster color/marker, noise behind."""
    unique_ids = sorted(int(v) for v in np.unique(cluster_labels))
    # Draw noise first
    if 0 in unique_ids:
        mask = cluster_labels == 0
        ax.scatter(x[mask], y[mask], c="gray", marker="x", s=16, alpha=alpha_noise, linewidths=0.0)
        unique_ids = [cid for cid in unique_ids if cid != 0]
    for cid in unique_ids:
        mask = cluster_labels == cid
        color = cluster_color_map.get(cid, "#222222")
        mkr = cluster_marker_map.get(cid, "o")
        ms = marker_sizes.get(mkr, 16)
        ax.scatter(x[mask], y[mask], c=color, marker=mkr, s=ms, alpha=alpha_cluster, linewidths=0.0)

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
    parser.add_argument("--cluster-colors", type=str, default=os.environ.get("HDBSCAN_UMAP_CLUSTER_COLORS"),
                        help="Global cluster plot coloring mode: auto, on, or off. If unset in an interactive Starlink run, prompts.")
    parser.add_argument("--dataset-choice", type=str, default=os.environ.get("HDBSCAN_UMAP_DATASET_CHOICE"),
                        help="Dataset folder selection: 1=gps_files_corrected, 2=oneweb_files_corrected, 3=starlink_files_corrected.")
    parser.add_argument("--starlink-inc-band", type=str, default=os.environ.get("HDBSCAN_UMAP_STARLINK_INC_BAND"),
                        help="Starlink inclination band option: 1..8 (used only when starlink dataset is selected).")
    parser.add_argument(
        "--starlink-profile",
        type=str,
        default=os.environ.get("HDBSCAN_UMAP_STARLINK_PROFILE"),
        help=("Optional Starlink profile key for CSV-based generation filtering. "
              "Choices: fullrun"))
    parser.add_argument(
        "--starlink-profile-dir",
        type=str,
        default=os.environ.get("HDBSCAN_UMAP_STARLINK_PROFILE_DIR"),
        help=(
            "Optional path to a profile folder containing satellite_labels.csv. "
            "Overrides --starlink-profile when both are provided."
        ),
    )
    parser.add_argument(
        "--starlink-generation",
        type=str,
        default=os.environ.get("HDBSCAN_UMAP_STARLINK_GENERATION"),
        help="Generation to include from profile CSV: gen1, gen2, proto, unknown.",
    )
    parser.add_argument(
        "--starlink-source-dir",
        type=str,
        default=os.environ.get("HDBSCAN_UMAP_STARLINK_SOURCE_DIR"),
        help="Optional source folder for Starlink .txt files (used with profile filtering).",
    )
    args, _unknown = parser.parse_known_args()
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if args.plot_dpi < 72:
        raise ValueError("--plot-dpi must be >= 72")
    if args.dataset_choice is not None and args.dataset_choice not in {"1", "2", "3"}:
        raise ValueError("--dataset-choice must be one of: 1, 2, 3")
    if args.cluster_colors is not None:
        args.cluster_colors = args.cluster_colors.lower()
        if args.cluster_colors not in CLUSTER_COLOR_MODES:
            valid_modes = ", ".join(sorted(CLUSTER_COLOR_MODES))
            raise ValueError(f"--cluster-colors must be one of: {valid_modes}")
    if args.starlink_inc_band is not None and args.starlink_inc_band not in {"1", "2", "3", "4", "5", "6", "7", "8"}:
        raise ValueError("--starlink-inc-band must be one of: 1, 2, 3, 4, 5, 6, 7, 8")
    if args.starlink_profile is not None and args.starlink_profile not in STARLINK_PROFILE_DIR_MAP:
        valid_profiles = ", ".join(sorted(STARLINK_PROFILE_DIR_MAP.keys()))
        raise ValueError(f"--starlink-profile must be one of: {valid_profiles}")
    if args.starlink_generation is not None:
        args.starlink_generation = args.starlink_generation.lower()
        if args.starlink_generation not in STARLINK_GENERATIONS:
            raise ValueError("--starlink-generation must be one of: gen1, gen2, proto, unknown")
    return args

def _choose_starlink_inclination_band_interactive() -> str:
    print("Select Starlink inclination filter band:")
    print("  1) 42.95 - 43.05")
    print("  2) 52.95 - 53.10")
    print("  3) 53.10 - 53.17")
    print("  4) 53.17 - 53.25")
    print("  5) 52.95 - 53.25")
    print("  6) 69.90 - 70.10")
    print("  7) 97.50 - 97.75")
    print("  8) 0.00 - 360.00 (no effective inclination filtering)")
    while True:
        choice = input("Enter 1-8 [default 7]: ").strip()
        if choice == "":
            return "7"
        if choice in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            return choice
        print("Invalid selection. Please enter an integer from 1 to 8.")

def _maybe_filter_starlink_inclination_band(
    data: pd.DataFrame,
    folder_paths: list[str],
    band_choice: str | None,
) -> pd.DataFrame:
    is_starlink = any("starlink" in str(p).lower() for p in folder_paths)
    if not is_starlink:
        print("Inclination band filter skipped (dataset is not starlink).\n")
        return data

    band_map = {
        "1": (42.95, 43.05),
        "2": (52.95, 53.10),
        "3": (53.10, 53.17),
        "4": (53.17, 53.25),
        "5": (52.95, 53.25),
        "6": (69.90, 70.10),
        "7": (97.50, 97.75),
        "8": (0.00, 360.00),
    }

    if band_choice is None:
        if sys.stdin is not None and sys.stdin.isatty():
            band_choice = _choose_starlink_inclination_band_interactive()
        else:
            band_choice = "7"

    low, high = band_map[band_choice]
    filtered = data[(data["inc"] > low) & (data["inc"] < high)]
    print(
        "Applied Starlink inclination band "
        f"{band_choice}: {low:.2f} - {high:.2f} deg"
    )
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

def _resolve_dataset_choice(dataset_choice: str | None) -> str:
    if dataset_choice is None:
        if sys.stdin is not None and sys.stdin.isatty():
            return _choose_dataset_interactive()
        return "3"
    return dataset_choice


def _resolve_folder_path(dataset_choice: str) -> list[str]:
    base_dir = r"C:\Users\PC\Code\UMAP_HDBSCAN"
    dataset_map = {
        "1": "gps_files_corrected",
        "2": "oneweb_files_corrected",
        "3": "starlink_files_corrected",
    }

    folder_name = dataset_map[dataset_choice]
    selected = os.path.join(base_dir, folder_name)
    print(f"Dataset selection: {dataset_choice} -> {folder_name}")
    return [selected]


def _choose_starlink_generation_interactive() -> str:
    print("Select Starlink generation filter:")
    print("  1) gen1")
    print("  2) gen2")
    print("  3) proto")
    print("  4) unknown")
    while True:
        choice = input("Enter 1-4 [default 1]: ").strip()
        if choice == "":
            return "gen1"
        if choice == "1":
            return "gen1"
        if choice == "2":
            return "gen2"
        if choice == "3":
            return "proto"
        if choice == "4":
            return "unknown"
        print("Invalid selection. Please enter an integer from 1 to 4.")


def _choose_cluster_colors_interactive() -> bool:
    print("Use global cluster colors in plots?")
    print("  1) yes, color/mark points by global cluster")
    print("  2) no, use the default plot colors")
    while True:
        choice = input("Enter 1 or 2 [default 1]: ").strip().lower()
        if choice in {"", "1", "y", "yes"}:
            return True
        if choice in {"2", "n", "no"}:
            return False
        print("Invalid selection. Please enter 1 for yes or 2 for no.")


def _resolve_cluster_color_request(cluster_color_mode: str | None, is_starlink_dataset: bool) -> bool:
    if not is_starlink_dataset:
        return False
    if cluster_color_mode == "off":
        return False
    if cluster_color_mode in {"auto", "on"}:
        return True
    if sys.stdin is not None and sys.stdin.isatty():
        return _choose_cluster_colors_interactive()
    return True


def _resolve_starlink_profile_dir(profile_key: str | None, profile_dir: str | None) -> str | None:
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    parent_dir = os.path.dirname(this_file_dir)

    def _resolve_path(path_value: str) -> str:
        expanded = os.path.expandvars(os.path.expanduser(path_value))
        if os.path.isabs(expanded):
            return expanded

        candidates = [
            os.path.join(this_file_dir, expanded),
            os.path.join(parent_dir, expanded),
            os.path.join(os.getcwd(), expanded),
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                return cand
        return candidates[0]

    if profile_dir:
        return _resolve_path(profile_dir)
    if profile_key:
        return _resolve_path(STARLINK_PROFILE_DIR_MAP[profile_key])
    return None


def _resolve_starlink_source_dir(starlink_source_dir: str | None) -> str:
    if starlink_source_dir:
        return starlink_source_dir

    candidates = [
        r"C:\Users\PC\Code\starlink_tles",
        r"C:\Users\PC\Code\UMAP_HDBSCAN\starlink_files_corrected",
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return candidates[-1]


def _load_profile_only_files(profile_dir: str, generation: str) -> set[str]:
    csv_path = os.path.join(profile_dir, "satellite_labels.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Profile CSV not found: {csv_path}")

    labels = pd.read_csv(csv_path)
    required = {"filename", "generation"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Profile CSV missing required columns: {sorted(missing)}")

    selected = labels.loc[labels["generation"].astype(str).str.lower() == generation, "filename"]
    files = {str(name).strip() for name in selected if str(name).strip()}
    return files


def _intldes_to_cospar_root(intldes: str) -> str | None:
    s = str(intldes).strip()
    if len(s) < 5:
        return None
    try:
        yy = int(s[0:2])
        launch_num = int(s[2:5])
    except Exception:
        return None
    year_full = 2000 + yy if yy < 57 else 1900 + yy
    return f"{year_full:04d}-{launch_num:03d}"


def _load_profile_launch_roots(profile_dir: str, generation: str) -> set[str]:
    launch_csv = os.path.join(profile_dir, "launch_manifest_inferred.csv")
    if os.path.isfile(launch_csv):
        launch_df = pd.read_csv(launch_csv)
        required = {"cospar_root", "generation"}
        missing = required - set(launch_df.columns)
        if missing:
            raise ValueError(f"Launch manifest missing required columns: {sorted(missing)}")
        selected = launch_df.loc[
            launch_df["generation"].astype(str).str.lower() == generation,
            "cospar_root",
        ]
        return {str(root).strip() for root in selected if str(root).strip()}

    # Fallback to per-satellite labels if launch manifest is unavailable.
    sat_csv = os.path.join(profile_dir, "satellite_labels.csv")
    if not os.path.isfile(sat_csv):
        raise FileNotFoundError(
            f"Profile CSV not found: expected {launch_csv} or {sat_csv}"
        )
    sat_df = pd.read_csv(sat_csv)
    required = {"cospar_root", "generation"}
    missing = required - set(sat_df.columns)
    if missing:
        raise ValueError(f"Satellite labels missing required columns: {sorted(missing)}")
    selected = sat_df.loc[
        sat_df["generation"].astype(str).str.lower() == generation,
        "cospar_root",
    ]
    return {str(root).strip() for root in selected if str(root).strip()}

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


def _safe_slug(value: str | None, default: str = "none") -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = default
    text = text.lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or default


def _dataset_label(dataset_choice: str | None) -> str:
    mapping = {
        "1": "gps_files_corrected",
        "2": "oneweb_files_corrected",
        "3": "starlink_files_corrected",
    }
    return mapping.get(dataset_choice or "", "unknown_dataset")


def _build_plot_output_dir(args: argparse.Namespace) -> str:
    profile_label = args.starlink_profile
    if profile_label is None and args.starlink_profile_dir:
        profile_label = os.path.basename(os.path.normpath(args.starlink_profile_dir))

    subdir = os.path.join(
        "sat_analysis_plot",
        f"dataset_{_safe_slug(_dataset_label(args.dataset_choice), 'unknown_dataset')}",
        f"profile_{_safe_slug(profile_label, 'none')}",
        f"generation_{_safe_slug(args.starlink_generation, 'none')}",
        f"incband_{_safe_slug(args.starlink_inc_band, 'none')}",
    )
    os.makedirs(subdir, exist_ok=True)
    return subdir


def _save_figure(fig, output_dir: str, filename: str, dpi: int) -> str | None:
    try:
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure: {out_path}")
        return out_path
    except Exception as exc:
        print(f"Warning: failed to save figure '{filename}': {exc}")
        return None


def _save_plotly_figure(fig, output_dir: str, filename: str) -> str | None:
    try:
        out_path = os.path.join(output_dir, filename)
        # Inline plotly.js for fully local/offline viewing and version-safe rendering.
        fig.write_html(out_path, include_plotlyjs=True, full_html=True)
        print(f"Saved interactive plot: {out_path}")
        return out_path
    except Exception as exc:
        print(f"Warning: failed to save interactive plot '{filename}': {exc}")
        return None


def _save_plotly_png(fig, output_dir: str, filename: str,
                     width: int = 2200, height: int = 1400, scale: float = 1.0) -> str | None:
    try:
        out_path = os.path.join(output_dir, filename)
        # Kaleido can fail on very large stepwise colorscales from textured surfaces.
        # For static export, fallback to a simpler Earth shading while keeping orbit geometry.
        try:
            import copy
            fig_png = copy.deepcopy(fig)
            if len(fig_png.data) > 0 and getattr(fig_png.data[0], "type", "") == "surface":
                z_surface = np.asarray(fig_png.data[0]["z"], dtype=np.float64)
                fig_png.data[0].update(
                    surfacecolor=z_surface.tolist(),
                    colorscale=[
                        [0.00, "#0a1f4d"],
                        [0.35, "#123b72"],
                        [0.50, "#1f6a47"],
                        [0.68, "#5f7d3a"],
                        [0.82, "#9d8a5e"],
                        [1.00, "#eceff5"],
                    ],
                    cmin=None,
                    cmax=None,
                )
        except Exception:
            fig_png = fig

        fig_png.write_image(out_path, format="png", width=width, height=height, scale=scale)
        print(f"Saved Plotly PNG: {out_path}")
        return out_path
    except Exception as exc:
        print(
            f"Warning: failed to save Plotly PNG '{filename}': {exc}. "
            "Static Plotly image export typically requires the 'kaleido' package."
        )
        return None


def _resolve_earth_texture_path() -> str | None:
    cwd = os.getcwd()
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else cwd
    parent_dir = os.path.dirname(this_file_dir)
    candidates = [
        os.environ.get("HDBSCAN_UMAP_EARTH_TEXTURE"),
        os.path.join(cwd, "earth_texture_4k.jpg"),
        os.path.join(this_file_dir, "earth_texture_4k.jpg"),
        os.path.join(this_file_dir, "assets", "earth_texture_4k.jpg"),
        os.path.join(parent_dir, "earth_texture_4k.jpg"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _kepler_eccentric_anomaly_from_mean(mean_anomaly: np.ndarray, eccentricity: float,
                                        max_iter: int = 30, tol: float = 1e-11) -> np.ndarray:
    e = float(eccentricity)
    M = np.asarray(mean_anomaly, dtype=np.float64)
    if e < 0.8:
        E = M.copy()
    else:
        E = np.full_like(M, np.pi)

    for _ in range(max_iter):
        f_val = E - e * np.sin(E) - M
        f_prime = 1.0 - e * np.cos(E)
        delta = f_val / np.maximum(f_prime, 1e-14)
        E -= delta
        if np.nanmax(np.abs(delta)) < tol:
            break
    return E


def _propagate_one_period_kepler_xyz(mu_val: float, sma_km: float, ecc_val: float, inc_rad: float,
                                     aop_rad: float, raan_rad: float, ta0_rad: float,
                                     points_per_orbit: int) -> tuple[np.ndarray | None, float | None]:
    if (not np.isfinite(sma_km)) or (not np.isfinite(ecc_val)):
        return None, None
    if sma_km <= 0.0 or ecc_val < 0.0 or ecc_val >= 1.0:
        return None, None

    mean_motion_rad_s = np.sqrt(mu_val / (sma_km ** 3))
    if not np.isfinite(mean_motion_rad_s) or mean_motion_rad_s <= 0.0:
        return None, None
    period_s = (2.0 * np.pi) / mean_motion_rad_s

    # Start at the current catalog true anomaly and propagate one full Keplerian period.
    e_safe = float(np.clip(ecc_val, 0.0, 1.0 - 1e-12))
    e0 = 2.0 * np.arctan2(
        np.sqrt(1.0 - e_safe) * np.sin(ta0_rad / 2.0),
        np.sqrt(1.0 + e_safe) * np.cos(ta0_rad / 2.0),
    )
    m0 = np.mod(e0 - e_safe * np.sin(e0), 2.0 * np.pi)

    times = np.linspace(0.0, period_s, points_per_orbit)
    mean_anomaly = np.mod(m0 + mean_motion_rad_s * times, 2.0 * np.pi)
    eccentric_anomaly = _kepler_eccentric_anomaly_from_mean(mean_anomaly, e_safe)

    true_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 + e_safe) * np.sin(eccentric_anomaly / 2.0),
        np.sqrt(1.0 - e_safe) * np.cos(eccentric_anomaly / 2.0),
    )
    radius = sma_km * (1.0 - e_safe * np.cos(eccentric_anomaly))

    x_pf = radius * np.cos(true_anomaly)
    y_pf = radius * np.sin(true_anomaly)

    cos_w = np.cos(aop_rad)
    sin_w = np.sin(aop_rad)
    cos_O = np.cos(raan_rad)
    sin_O = np.sin(raan_rad)
    cos_i = np.cos(inc_rad)
    sin_i = np.sin(inc_rad)

    r11 = cos_O * cos_w - sin_O * sin_w * cos_i
    r12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    r21 = sin_O * cos_w + cos_O * sin_w * cos_i
    r22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    r31 = sin_w * sin_i
    r32 = cos_w * sin_i

    x_eci = r11 * x_pf + r12 * y_pf
    y_eci = r21 * x_pf + r22 * y_pf
    z_eci = r31 * x_pf + r32 * y_pf
    xyz = np.column_stack((x_eci, y_eci, z_eci))
    return xyz, period_s


def _build_textured_earth_surface_trace(r_e_val: float, texture_path: str | None):
    import plotly.graph_objects as go

    lon_count = 360
    lat_count = 181
    lon = np.linspace(-np.pi, np.pi, lon_count)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, lat_count)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    x_earth = r_e_val * np.cos(lat_grid) * np.cos(lon_grid)
    y_earth = r_e_val * np.cos(lat_grid) * np.sin(lon_grid)
    z_earth = r_e_val * np.sin(lat_grid)

    if texture_path:
        try:
            from PIL import Image

            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
            adaptive_palette = getattr(getattr(Image, "Palette", Image), "ADAPTIVE", getattr(Image, "ADAPTIVE", 0))

            img = Image.open(texture_path).convert("RGB").resize((lon_count, lat_count), resampling)
            quantized = img.convert("P", palette=adaptive_palette, colors=256)
            texture_index = np.flipud(np.asarray(quantized, dtype=np.float64))

            palette = quantized.getpalette()
            if palette is None:
                raise ValueError("Quantized texture palette is unavailable.")

            colorscale = []
            for i in range(256):
                r, g, b = palette[3 * i: 3 * i + 3]
                color = f"rgb({int(r)},{int(g)},{int(b)})"
                lo = i / 255.0
                hi = min((i + 1) / 255.0, 1.0)
                colorscale.append([lo, color])
                colorscale.append([hi, color])

            return go.Surface(
                x=x_earth.tolist(),
                y=y_earth.tolist(),
                z=z_earth.tolist(),
                surfacecolor=texture_index.tolist(),
                cmin=0,
                cmax=255,
                colorscale=colorscale,
                showscale=False,
                hoverinfo="skip",
                opacity=1.0,
                lighting={"ambient": 0.8, "diffuse": 0.6, "specular": 0.05, "roughness": 0.9},
                lightposition={"x": 12000, "y": 0, "z": 9000},
            )
        except Exception as exc:
            print(f"Warning: failed to apply Earth texture '{texture_path}': {exc}")

    return go.Surface(
        x=x_earth.tolist(),
        y=y_earth.tolist(),
        z=z_earth.tolist(),
        surfacecolor=z_earth.tolist(),
        colorscale=[
            [0.00, "#0a1f4d"],
            [0.35, "#123b72"],
            [0.50, "#1f6a47"],
            [0.68, "#5f7d3a"],
            [0.82, "#9d8a5e"],
            [1.00, "#eceff5"],
        ],
        showscale=False,
        hoverinfo="skip",
        opacity=1.0,
        lighting={"ambient": 0.8, "diffuse": 0.6, "specular": 0.05, "roughness": 0.9},
        lightposition={"x": 12000, "y": 0, "z": 9000},
    )


def _plot_plotly_satellite_periods(
    tle_data: pd.DataFrame,
    sma_km: np.ndarray,
    ecc: np.ndarray,
    inc_rad: np.ndarray,
    aop_rad: np.ndarray,
    raan_rad: np.ndarray,
    ta_rad: np.ndarray,
    color_cycle: list[str],
    output_dir: str,
    r_e_val: float,
    mu_val: float,
    points_per_orbit: int,
    cluster_labels: np.ndarray | None = None,
    cluster_color_map: dict[int, str] | None = None,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print(f"Warning: Plotly is unavailable; skipping 3D interactive orbit plot ({exc}).")
        return

    if len(sma_km) == 0:
        print("Warning: No satellites available for Plotly propagation.")
        return

    points_per_orbit = max(32, int(points_per_orbit))
    texture_path = _resolve_earth_texture_path()
    if texture_path:
        print(f"Using Earth texture: {texture_path}")
    else:
        print("Earth texture not found; using shaded Earth surface.")

    fig = go.Figure()
    fig.add_trace(_build_textured_earth_surface_trace(r_e_val, texture_path))

    n_colors = max(1, len(color_cycle))
    _use_cl = cluster_labels is not None and cluster_color_map is not None
    grouped_lines = [{"x": [], "y": [], "z": []} for _ in range(n_colors)]
    # When using cluster colors, group orbit lines by cluster color
    _cl_line_groups: dict[str, dict[str, list]] = {} if _use_cl else {}

    sat_x = []
    sat_y = []
    sat_z = []
    sat_colors = []
    sat_hover = []

    label_columns = ["satellite_name", "name", "sat_name", "object_name", "norad_id", "norad_number", "filename"]
    labels = None
    for col in label_columns:
        if col in tle_data.columns:
            labels = tle_data[col].astype(str).to_numpy()
            break
    if labels is None:
        labels = np.array([f"sat_{idx + 1}" for idx in range(len(sma_km))], dtype=object)

    propagated_count = 0
    apoapsis_km = sma_km * (1.0 + np.clip(ecc, 0.0, None))

    for idx in range(len(sma_km)):
        xyz, period_s = _propagate_one_period_kepler_xyz(
            mu_val=mu_val,
            sma_km=float(sma_km[idx]),
            ecc_val=float(ecc[idx]),
            inc_rad=float(inc_rad[idx]),
            aop_rad=float(aop_rad[idx]),
            raan_rad=float(raan_rad[idx]),
            ta0_rad=float(ta_rad[idx]),
            points_per_orbit=points_per_orbit,
        )
        if xyz is None or period_s is None:
            continue

        color_idx = idx % n_colors
        if _use_cl:
            cid = int(cluster_labels[idx])
            orbit_color = cluster_color_map.get(cid, "#222222")
            if orbit_color not in _cl_line_groups:
                _cl_line_groups[orbit_color] = {"x": [], "y": [], "z": []}
            _cl_line_groups[orbit_color]["x"].extend(xyz[:, 0].tolist())
            _cl_line_groups[orbit_color]["x"].append(np.nan)
            _cl_line_groups[orbit_color]["y"].extend(xyz[:, 1].tolist())
            _cl_line_groups[orbit_color]["y"].append(np.nan)
            _cl_line_groups[orbit_color]["z"].extend(xyz[:, 2].tolist())
            _cl_line_groups[orbit_color]["z"].append(np.nan)
        else:
            grouped_lines[color_idx]["x"].extend(xyz[:, 0].tolist())
            grouped_lines[color_idx]["x"].append(np.nan)
            grouped_lines[color_idx]["y"].extend(xyz[:, 1].tolist())
            grouped_lines[color_idx]["y"].append(np.nan)
            grouped_lines[color_idx]["z"].extend(xyz[:, 2].tolist())
            grouped_lines[color_idx]["z"].append(np.nan)

        sat_x.append(float(xyz[0, 0]))
        sat_y.append(float(xyz[0, 1]))
        sat_z.append(float(xyz[0, 2]))
        if _use_cl:
            cid = int(cluster_labels[idx])
            sat_colors.append(cluster_color_map.get(cid, "#222222"))
        else:
            sat_colors.append(color_cycle[color_idx])
        sat_hover.append(
            f"{labels[idx]}<br>period={period_s / 60.0:.2f} min"
        )
        propagated_count += 1

    if propagated_count == 0:
        print("Warning: No valid elliptical satellite states available for Plotly propagation.")
        return

    if _use_cl:
        for line_color, packed in _cl_line_groups.items():
            if not packed["x"]:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=packed["x"],
                    y=packed["y"],
                    z=packed["z"],
                    mode="lines",
                    line={"color": line_color, "width": 2},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    else:
        for color_idx, packed in enumerate(grouped_lines):
            if not packed["x"]:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=packed["x"],
                    y=packed["y"],
                    z=packed["z"],
                    mode="lines",
                    line={"color": color_cycle[color_idx], "width": 2},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.add_trace(
        go.Scatter3d(
            x=sat_x,
            y=sat_y,
            z=sat_z,
            mode="markers",
            marker={"size": 3.5, "color": sat_colors, "opacity": 0.95},
            text=sat_hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    finite_apo = apoapsis_km[np.isfinite(apoapsis_km)]
    if finite_apo.size:
        bound = float(max(r_e_val, np.max(finite_apo))) * 1.08
    else:
        bound = float(r_e_val) * 2.0

    axis_layout = {
        "visible": False,
        "showgrid": False,
        "zeroline": False,
        "showticklabels": False,
        "showbackground": False,
        "range": [-bound, bound],
    }

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={
            "xaxis": axis_layout,
            "yaxis": axis_layout,
            "zaxis": axis_layout,
            "aspectmode": "cube",
            "bgcolor": "rgba(0,0,0,0)",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 0.95}},
        },
    )

    _save_plotly_figure(fig, output_dir, "satellite_orbits_3d_plotly.html")
    _save_plotly_png(fig, output_dir, "satellite_orbits_3d_plotly.png")
    print(
        "Plotly propagation summary: "
        f"{propagated_count} satellites, {points_per_orbit} points per orbit.\n"
    )

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
                     'xtick.direction': 'in', 'xtick.labelsize': 14*1.5, 'xtick.major.size': 3,
                     'xtick.major.width': 0.5, 'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5,
                     'xtick.minor.visible': True, 'xtick.top': True,
                     'ytick.direction': 'in', 'ytick.labelsize': 14*1.5, 'ytick.major.size': 3,
                     'ytick.major.width': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
                     'ytick.minor.visible': True, 'ytick.right': True,
                     'axes.linewidth': 0.5, 'grid.linewidth': 0.5, 'lines.linewidth': 1.0,
                     'legend.fontsize': 14*1.5, 'legend.frameon': False,
                     'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 12*1.5, 'axes.labelsize': 16*1.5, 'axes.titlesize': 18*1.5,
                     'axes.grid': True, 'grid.linestyle': '--', 'grid.color': '0.5',
                     'lines.markersize': 8, 'axes.spines.top': True, 'axes.spines.right': True})

# Define the custom 20-color palette (darkened colors)
colors = ['#15528e', '#b25800', '#1e701e', '#951c1c', '#673284', 
          '#623c34', '#9e5387', '#585858', '#848417', '#108590',
          '#798ba2', '#b28254', '#6a9c60', '#b26a68', '#8a7b94',
          '#896d67', '#ac7f93', '#8b8b8b', '#999962', '#6f989f']

# Define 20 distinct marker styles for cycling in scatter plots
markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', 'h', 'H',
           'X', '*', '+', 'x', '1', '2', '3', '4', 'd', 'P', '8']

# Define marker sizes using a switch-case like dictionary.
# Markers not specified in this dictionary will default to 25.
marker_sizes = {'x': 30, '+': 35, '1': 40,
                '2': 40, '3': 40, '4': 40}

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

def _print_vector_consistency_checks(
    r_vectors: np.ndarray,
    v_vectors: np.ndarray,
    h_vectors: np.ndarray,
    e_vectors: np.ndarray,
    n_vectors: np.ndarray,
    mu_val: float,
) -> None:
    h_from_rv = np.cross(r_vectors, v_vectors)
    h_residual = np.linalg.norm(h_vectors - h_from_rv, axis=1)

    r_norms = np.linalg.norm(r_vectors, axis=1, keepdims=True)
    valid_r = (r_norms[:, 0] > 1e-12)
    e_from_rv = np.zeros_like(e_vectors)
    e_from_rv[valid_r] = (
        np.cross(v_vectors[valid_r], h_from_rv[valid_r]) / mu_val
        - (r_vectors[valid_r] / r_norms[valid_r])
    )
    e_residual = np.linalg.norm(e_vectors - e_from_rv, axis=1)

    n_from_h = np.cross(np.array([0.0, 0.0, 1.0]), h_from_rv)
    n_residual = np.linalg.norm(n_vectors - n_from_h, axis=1)

    print("Vector consistency checks:")
    print(
        f"  |h - (r x v)| median={np.median(h_residual):.3e}, max={np.max(h_residual):.3e}"
    )
    print(
        f"  |e - e(r,v)| median={np.median(e_residual):.3e}, max={np.max(e_residual):.3e}"
    )
    print(
        f"  |n - (k x h)| median={np.median(n_residual):.3e}, max={np.max(n_residual):.3e}\n"
    )

def _nan_percentiles(values: np.ndarray, q: list[float]) -> list[float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return [np.nan for _ in q]
    return [float(np.percentile(clean, qi)) for qi in q]

def _print_orbital_diagnostics_summary(diag: dict[str, np.ndarray]) -> None:
    def _line(label: str, key: str, unit: str = ""):
        p05, p50, p95 = _nan_percentiles(diag[key], [5, 50, 95])
        suffix = f" {unit}" if unit else ""
        print(
            f"  {label:<28} p05={p05:>11.4f}{suffix} | "
            f"p50={p50:>11.4f}{suffix} | p95={p95:>11.4f}{suffix}"
        )

    print("Additional astrodynamics diagnostics (5th/50th/95th percentiles):")
    _line("Perigee altitude", "perigee_alt_km", "km")
    _line("Apogee altitude", "apogee_alt_km", "km")
    _line("Specific orbital energy", "specific_energy_km2_s2", "km^2/s^2")
    _line("Orbital period", "period_min", "min")
    _line("Mean motion", "rev_per_day", "rev/day")
    _line("Flight-path angle", "flight_path_angle_deg", "deg")
    _line("J2 RAAN drift", "raan_drift_deg_day", "deg/day")
    _line("J2 AOP drift", "aop_drift_deg_day", "deg/day")
    _line("Nodal-Kepler period delta", "nodal_minus_kepler_sec", "s")
    _line("Sun-sync inclination error", "sso_inc_error_deg", "deg")
    _line("Relative drag proxy", "drag_proxy_rel")
    _line("Speed minus circular", "delta_v_circ_km_s", "km/s")
    _line("h consistency residual", "h_residual", "km^2/s")
    _line("vis-viva residual", "vis_viva_residual", "km^2/s^2")
    print("")

def _compute_orbital_diagnostics(
    mu_val: float,
    r_e_val: float,
    j2_val: float,
    sma_local: np.ndarray,
    ecc_local: np.ndarray,
    inc_local: np.ndarray,
    raan_local: np.ndarray,
    ta_local: np.ndarray,
    r_vec_local: np.ndarray,
    v_vec_local: np.ndarray,
    h_vec_local: np.ndarray,
) -> dict[str, np.ndarray]:
    r_norm = np.linalg.norm(r_vec_local, axis=1)
    v_norm = np.linalg.norm(v_vec_local, axis=1)
    h_norm = np.linalg.norm(h_vec_local, axis=1)

    perigee_radius = sma_local * (1.0 - ecc_local)
    apogee_radius = sma_local * (1.0 + ecc_local)
    perigee_alt_km = perigee_radius - r_e_val
    apogee_alt_km = apogee_radius - r_e_val

    specific_energy = 0.5 * (v_norm ** 2) - (mu_val / r_norm)
    valid_bound = specific_energy < 0.0
    a_from_state = np.full_like(specific_energy, np.nan)
    a_from_state[valid_bound] = -mu_val / (2.0 * specific_energy[valid_bound])

    valid_ellipse = sma_local > 0.0
    mean_motion_rad_s = np.full_like(sma_local, np.nan)
    mean_motion_rad_s[valid_ellipse] = np.sqrt(mu_val / (sma_local[valid_ellipse] ** 3))
    period_s = np.full_like(sma_local, np.nan)
    period_s[valid_ellipse] = (2.0 * np.pi) / mean_motion_rad_s[valid_ellipse]
    period_min = period_s / 60.0
    rev_per_day = mean_motion_rad_s * 86400.0 / (2.0 * np.pi)

    radial_speed = np.sum(r_vec_local * v_vec_local, axis=1) / np.maximum(r_norm, 1e-12)
    transverse_speed = np.sqrt(np.maximum(v_norm ** 2 - radial_speed ** 2, 0.0))
    flight_path_angle_deg = np.arctan2(radial_speed, transverse_speed) * RAD_TO_DEG

    circular_speed = np.sqrt(mu_val / np.maximum(r_norm, 1e-12))
    delta_v_circ_km_s = v_norm - circular_speed

    one_minus_e2 = np.maximum(1.0 - ecc_local ** 2, 1e-12)
    prefactor = (3.0 / 2.0) * j2_val * np.sqrt(mu_val) * (r_e_val ** 2)
    denom = np.maximum(sma_local ** (3.5) * (one_minus_e2 ** 2), 1e-12)
    raan_dot_rad_s = -prefactor * np.cos(inc_local) / denom
    aop_dot_rad_s = 0.5 * prefactor * (5.0 * np.cos(inc_local) ** 2 - 1.0) / denom
    raan_drift_deg_day = raan_dot_rad_s * RAD_TO_DEG * 86400.0
    aop_drift_deg_day = aop_dot_rad_s * RAD_TO_DEG * 86400.0

    nodal_freq_rad_s = mean_motion_rad_s - raan_dot_rad_s
    nodal_period_s = np.full_like(sma_local, np.nan)
    valid_nodal = np.isfinite(nodal_freq_rad_s) & (np.abs(nodal_freq_rad_s) > 1e-12)
    nodal_period_s[valid_nodal] = (2.0 * np.pi) / nodal_freq_rad_s[valid_nodal]
    nodal_minus_kepler_sec = nodal_period_s - period_s

    # Relative drag proxy: rho(h_p) * v^2 using an exponential atmosphere proxy.
    perigee_alt_km = np.maximum(perigee_alt_km, 0.0)
    rho_ref = np.exp(-(perigee_alt_km - 200.0) / 60.0)
    drag_proxy_rel = rho_ref * (v_norm ** 2)

    # Sun-synchronous target RAAN precession (~+360 deg/tropical year in inertial frame).
    sso_target_deg_day = 360.0 / 365.2422
    sso_target_rad_s = sso_target_deg_day * DEG_TO_RAD / 86400.0
    cos_i_sso = -sso_target_rad_s * denom / np.maximum(prefactor, 1e-20)
    i_sso_target = np.full_like(inc_local, np.nan)
    valid_sso = np.abs(cos_i_sso) <= 1.0
    i_sso_target[valid_sso] = np.arccos(cos_i_sso[valid_sso])
    sso_inc_error_deg = (inc_local - i_sso_target) * RAD_TO_DEG

    sqrt_one_minus_e2 = np.sqrt(np.maximum(1.0 - ecc_local ** 2, 0.0))
    ecc_anomaly = np.arctan2(sqrt_one_minus_e2 * np.sin(ta_local), ecc_local + np.cos(ta_local))
    ecc_anomaly = np.mod(ecc_anomaly, 2.0 * np.pi)
    mean_anomaly = np.mod(ecc_anomaly - ecc_local * np.sin(ecc_anomaly), 2.0 * np.pi)

    vis_viva_residual = (v_norm ** 2) - (mu_val * ((2.0 / r_norm) - (1.0 / sma_local)))
    h_from_elements = np.sqrt(np.maximum(mu_val * sma_local * (1.0 - ecc_local ** 2), 0.0))
    h_residual = h_norm - h_from_elements

    return {
        "r_norm_km": r_norm,
        "v_norm_km_s": v_norm,
        "perigee_alt_km": perigee_alt_km,
        "apogee_alt_km": apogee_alt_km,
        "specific_energy_km2_s2": specific_energy,
        "semi_major_axis_from_state_km": a_from_state,
        "period_min": period_min,
        "rev_per_day": rev_per_day,
        "flight_path_angle_deg": flight_path_angle_deg,
        "circular_speed_km_s": circular_speed,
        "delta_v_circ_km_s": delta_v_circ_km_s,
        "raan_drift_deg_day": raan_drift_deg_day,
        "aop_drift_deg_day": aop_drift_deg_day,
        "nodal_period_min": nodal_period_s / 60.0,
        "nodal_minus_kepler_sec": nodal_minus_kepler_sec,
        "drag_proxy_rel": drag_proxy_rel,
        "sso_target_inc_deg": i_sso_target * RAD_TO_DEG,
        "sso_inc_error_deg": sso_inc_error_deg,
        "vis_viva_residual": vis_viva_residual,
        "h_residual": h_residual,
        "ta_deg": ta_local * RAD_TO_DEG,
        "mean_anomaly_deg": mean_anomaly * RAD_TO_DEG,
        "raan_deg": np.mod(raan_local * RAD_TO_DEG, 360.0),
        "inc_deg": inc_local * RAD_TO_DEG,
    }

def _plot_orbital_diagnostics(
    diag: dict[str, np.ndarray],
    color_cycle: list[str],
    marker_cycle: list[str],
    output_dir: str,
    dpi: int,
    cluster_labels: np.ndarray | None = None,
    cluster_color_map: dict[int, str] | None = None,
    cluster_marker_map: dict[int, str] | None = None,
) -> None:
    use_cl = cluster_labels is not None and cluster_color_map is not None

    def _diag_scatter(ax, x_key, y_key, fallback_color_idx, fallback_marker_idx, **kw):
        if use_cl:
            _scatter_by_cluster(ax, diag[x_key], diag[y_key],
                                cluster_labels, cluster_color_map, cluster_marker_map, **kw)
        else:
            ax.scatter(diag[x_key], diag[y_key], s=18,
                       c=color_cycle[fallback_color_idx],
                       marker=marker_cycle[fallback_marker_idx], alpha=0.7)

    fig = plt.figure(figsize=(12.5, 10.0))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    ax1 = fig.add_subplot(3, 3, 1)
    ax1.hist(diag["perigee_alt_km"], bins=30, color=color_cycle[0], alpha=0.85)
    ax1.set_xlabel("Perigee Altitude [km]")
    ax1.set_ylabel("Count")

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.hist(diag["apogee_alt_km"], bins=30, color=color_cycle[1], alpha=0.85)
    ax2.set_xlabel("Apogee Altitude [km]")
    ax2.set_ylabel("Count")

    ax3 = fig.add_subplot(3, 3, 3)
    _diag_scatter(ax3, "perigee_alt_km", "apogee_alt_km", 2, 2)
    ax3.set_xlabel("Perigee Altitude [km]")
    ax3.set_ylabel("Apogee Altitude [km]")

    ax4 = fig.add_subplot(3, 3, 4)
    _diag_scatter(ax4, "inc_deg", "raan_drift_deg_day", 3, 3)
    ax4.set_xlabel("Inclination (deg)")
    ax4.set_ylabel("RAAN Drift (deg/day)")

    ax5 = fig.add_subplot(3, 3, 5)
    _diag_scatter(ax5, "inc_deg", "aop_drift_deg_day", 4, 4)
    ax5.set_xlabel("Inclination (deg)")
    ax5.set_ylabel("AOP Drift (deg/day)")

    ax6 = fig.add_subplot(3, 3, 6)
    if use_cl:
        _scatter_by_cluster(ax6, diag["r_norm_km"], diag["v_norm_km_s"],
                            cluster_labels, cluster_color_map, cluster_marker_map)
    else:
        ax6.scatter(diag["r_norm_km"], diag["v_norm_km_s"], s=16, c=color_cycle[5], marker=marker_cycle[5], alpha=0.7, label="Actual")
    order = np.argsort(diag["r_norm_km"])
    ax6.plot(diag["r_norm_km"][order], diag["circular_speed_km_s"][order], color='black', linewidth=1.2, label="Circular speed")
    ax6.set_xlabel("Radius [km]")
    ax6.set_ylabel("Speed [km/s]")

    ax7 = fig.add_subplot(3, 3, 7)
    _diag_scatter(ax7, "ta_deg", "flight_path_angle_deg", 6, 6)
    ax7.set_xlabel("True Anomaly (deg)")
    ax7.set_ylabel("Flight-Path Angle (deg)")

    ax8 = fig.add_subplot(3, 3, 8)
    ax8.hist(diag["specific_energy_km2_s2"], bins=30, color=color_cycle[7], alpha=0.85)
    ax8.set_xlabel("Specific Energy [km^2/s^2]")
    ax8.set_ylabel("Count")

    ax9 = fig.add_subplot(3, 3, 9)
    _diag_scatter(ax9, "period_min", "rev_per_day", 8, 8)
    ax9.set_xlabel("Period [min]")
    ax9.set_ylabel("Mean Motion [rev/day]")

    plt.tight_layout()
    _save_figure(fig, output_dir, "orbital_diagnostics.png", dpi)
    #plt.show()

def _plot_advanced_perturbation_diagnostics(
    diag: dict[str, np.ndarray],
    output_dir: str,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(10.0, 7.5))
    ax = fig.add_subplot(1, 1, 1)

    scatter = ax.scatter(
        diag["mean_anomaly_deg"],
        diag["raan_deg"],
        s=20,
        c=diag["raan_drift_deg_day"],
        cmap='viridis',
        marker='o',
        alpha=0.85,
    )
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label("RAAN Drift (deg/day)")
    ax.set_xlabel("Mean Anomaly (deg)")
    ax.set_ylabel("RAAN (deg)")
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(0.0, 360.0)

    plt.tight_layout()
    _save_figure(fig, output_dir, "raan_vs_mean_anomaly.png", dpi)
    #plt.show()

# GLOBAL SETTINGS & CONSTANTS
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")

mu           = 398600.4418    # Gravitational Parameter for Earth, km^3/s^2
r_E          = 6378.145       # Radius of Earth, km
J2           = 1.082635854e-3 # J2 Second Zonal Harmonic Perturbation Constant
DEG_TO_RAD   = np.pi / 180    # Degrees to radians conversion
RAD_TO_DEG   = 180 / np.pi    # Radians to degrees conversion

# Specify the folders containing TLE files
ARGS.dataset_choice = _resolve_dataset_choice(ARGS.dataset_choice)
if (
    ARGS.dataset_choice == "3"
    and ARGS.starlink_profile is None
    and ARGS.starlink_profile_dir is None
    and (ARGS.starlink_generation is not None or (sys.stdin is not None and sys.stdin.isatty()))
):
    ARGS.starlink_profile = DEFAULT_STARLINK_PROFILE

folder_path = _resolve_folder_path(ARGS.dataset_choice)
only_files_filter = None
profile_launch_roots = None

if ARGS.dataset_choice == "3":
    profile_dir = _resolve_starlink_profile_dir(ARGS.starlink_profile, ARGS.starlink_profile_dir)
    if profile_dir is not None:
        if ARGS.starlink_generation is None:
            if sys.stdin is not None and sys.stdin.isatty():
                ARGS.starlink_generation = _choose_starlink_generation_interactive()
            else:
                ARGS.starlink_generation = "gen1"
        only_files_filter = _load_profile_only_files(profile_dir=profile_dir, generation=ARGS.starlink_generation)
        print(f"Starlink profile filter: {profile_dir}")
        print(f"Starlink generation selected: {ARGS.starlink_generation}")
        if ARGS.starlink_source_dir:
            print("Warning: --starlink-source-dir is ignored for dataset-choice 3; using starlink_files_corrected.\n")
        print(f"Starlink files selected from profile generation: {len(only_files_filter)}\n")
    elif ARGS.starlink_generation is not None:
        print("Warning: --starlink-generation was provided without a profile; ignoring generation filter.\n")

PLOT_OUTPUT_DIR = _build_plot_output_dir(ARGS)
print(f"Figure output directory: {PLOT_OUTPUT_DIR}\n")

total_files = sum(len(os.listdir(folder)) for folder in folder_path if os.path.exists(folder))
print(f"Total Files Processed: {total_files}\n")

# Check which folders exist
for folder in folder_path:
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' not found.")

# DATA LOADING & PREPROCESSING
all_tle_data, _filenames_array = load_all_tle_data(folder_path, only_files=only_files_filter)

if only_files_filter is not None:
    print(f"Rows after Starlink profile filename filtering: {len(all_tle_data)}\n")
elif profile_launch_roots is not None:
    row_roots = all_tle_data["international_designator"].map(_intldes_to_cospar_root)
    keep_mask = row_roots.isin(profile_launch_roots)
    kept_rows = int(keep_mask.sum())
    all_tle_data = all_tle_data.loc[keep_mask].copy()
    print(f"Rows after Starlink launch-root profile filtering: {kept_rows}\n")

print(f"Loaded TLE data shape: {all_tle_data.shape}\n")

# Target date for initial clustering
target_date    = datetime(2025, 4, 10)
time_tolerance = timedelta(days=0.5)

print(all_tle_data.columns)
print(all_tle_data['timestamp'].head())
print(f"Number of records before filtering: {len(all_tle_data)}\n")
initial_tle_data = filter_tle_data_by_date(all_tle_data, target_date, time_tolerance)
print(f"Number of records after filtering: {len(initial_tle_data)}\n")

initial_tle_data = _maybe_filter_starlink_inclination_band(
    initial_tle_data,
    folder_path,
    ARGS.starlink_inc_band,
)

# Extract orbital elements and launch information
sma          = initial_tle_data['sma'].values
ecc          = initial_tle_data['ecc'].values
inc          = initial_tle_data['inc'].values * DEG_TO_RAD
raan         = initial_tle_data['raan'].values * DEG_TO_RAD
aop          = initial_tle_data['aop'].values * DEG_TO_RAD
ta           = initial_tle_data['true_anomaly'].values * DEG_TO_RAD
print(f"Epoch: {initial_tle_data['timestamp'].values[:1]}")

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

_print_vector_consistency_checks(
    radius_vectors_np,
    velocity_vectors_np,
    angular_momentum_vectors_np,
    eccentricity_vectors_np,
    nodal_vectors_np,
    mu,
)

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

# --- Load global cluster labels for Starlink coloring ---
_cluster_label_map = None
_is_starlink_dataset = any("starlink" in str(p).lower() for p in folder_path)
_cluster_colors_requested = _resolve_cluster_color_request(ARGS.cluster_colors, _is_starlink_dataset)
if _cluster_colors_requested:
    _cluster_label_map = _load_global_cluster_label_map()
    if _cluster_label_map is not None:
        print(f"Loaded global cluster label map ({len(_cluster_label_map)} entries).")
elif _is_starlink_dataset:
    print("Cluster colors disabled; using default plot colors.\n")

_cluster_labels, _cluster_color_map, _cluster_marker_map = _resolve_cluster_arrays(
    initial_tle_data, _cluster_label_map
)
_use_cluster_colors = _cluster_colors_requested and _cluster_labels is not None

# 2D Visualization of Orbital Vectors and Full Orbits
fig = plt.figure(figsize=(10.0, 7.5))

_vector_specs = [
    (radius_vectors_np, "X (km)", "Y (km)"),
    (angular_momentum_vectors_np, "X (km²/s)", "Y (km²/s)"),
    (eccentricity_vectors_np, "X", "Y"),
    (nodal_vectors_np, "X (km²/s)", "Y (km²/s)"),
    (velocity_vectors_np, "X (km/s)", "Y (km/s)"),
    (perifocal_minor_axis_np, "X ", "Y"),
]

for _vi, (_vecs, _xl, _yl) in enumerate(_vector_specs, start=1):
    _ax = fig.add_subplot(2, 3, _vi)
    _ax.set_xlabel(_xl)
    _ax.set_ylabel(_yl, labelpad=-8)
    _ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    _finite_x = _vecs[np.isfinite(_vecs[:, 0]), 0]
    if _finite_x.size:
        _max_abs_x = np.max(np.abs(_finite_x))
        if 0 < _max_abs_x < 1e-2:
            _x_exponent = int(np.floor(np.log10(_max_abs_x)))
            _x_scale = 10.0 ** _x_exponent
            _ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _pos, scale=_x_scale: f"{value / scale:g}")
            )
            _ax.text(0.98, -0.12, rf"$\times 10^{{{_x_exponent}}}$",
                     transform=_ax.transAxes, ha="right", va="top",
                     fontsize=12, clip_on=False)
    if _use_cluster_colors:
        _scatter_by_cluster(_ax, _vecs[:, 0], _vecs[:, 1],
                            _cluster_labels, _cluster_color_map, _cluster_marker_map)
    else:
        _ax.scatter(_vecs[:, 0], _vecs[:, 1],
                    c=colors[_vi - 1], marker=markers[0], s=marker_sizes.get(markers[0], 25))

fig.subplots_adjust(
    top=0.92,
    bottom=0.125,
    left=0.141,
    right=0.949,
    hspace=0.336,
    wspace=0.845,
)
_save_figure(fig, PLOT_OUTPUT_DIR, "orbital_vectors_2d.png", ARGS.plot_dpi)
#plt.show()

# Additional astrodynamics diagnostics and visualization suite
orbital_diag = _compute_orbital_diagnostics(mu_val=mu, r_e_val=r_E, j2_val=J2, 
                                            sma_local=sma, ecc_local=ecc, inc_local=inc,
                                             raan_local=raan,
                                            ta_local=ta, r_vec_local=radius_vectors_np,
                                            v_vec_local=velocity_vectors_np, h_vec_local=angular_momentum_vectors_np)
_print_orbital_diagnostics_summary(orbital_diag)

# Quick sanity check for state-derived semimajor axis against catalog semimajor axis
a_state = orbital_diag["semi_major_axis_from_state_km"]
valid_a = np.isfinite(a_state)
if np.any(valid_a):
    a_abs_err = np.abs(a_state[valid_a] - sma[valid_a])
    print("Semimajor-axis consistency: "
          f"median |a_state-a_tle|={np.median(a_abs_err):.6f} km, "
          f"max={np.max(a_abs_err):.6f} km\n")

_plot_orbital_diagnostics(orbital_diag, colors, markers, PLOT_OUTPUT_DIR, ARGS.plot_dpi,
                         cluster_labels=_cluster_labels,
                         cluster_color_map=_cluster_color_map,
                         cluster_marker_map=_cluster_marker_map)
_plot_advanced_perturbation_diagnostics(orbital_diag, PLOT_OUTPUT_DIR, ARGS.plot_dpi)
_plot_plotly_satellite_periods(
    tle_data=initial_tle_data,
    sma_km=sma,
    ecc=ecc,
    inc_rad=inc,
    aop_rad=aop,
    raan_rad=raan,
    ta_rad=ta,
    color_cycle=colors,
    output_dir=PLOT_OUTPUT_DIR,
    r_e_val=r_E,
    mu_val=mu,
    points_per_orbit=int(os.environ.get("HDBSCAN_UMAP_PLOTLY_POINTS", "180")),
    cluster_labels=_cluster_labels,
    cluster_color_map=_cluster_color_map,
)