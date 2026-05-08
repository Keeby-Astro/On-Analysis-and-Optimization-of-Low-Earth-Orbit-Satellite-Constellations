from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

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

COLORS = [
    "#15528e", "#b25800", "#1e701e", "#951c1c", "#673284",
    "#623c34", "#9e5387", "#585858", "#848417", "#108590",
    "#798ba2", "#b28254", "#6a9c60", "#b26a68", "#8a7b94",
    "#896d67", "#ac7f93", "#8b8b8b", "#999962", "#6f989f",
]

MARKERS = [
    "o", "s", "^", "v", "<", ">", "D", "p", "h", "H",
    "X", "*", "+", "x", "1", "2", "3", "4", "d", "P",
]

PAIRPLOT_MARKERS_FILLED = ["o", "s", "^", "v", "<", ">", "D", "d", "p", "P", "h", "H", "X", "8"]

MARKER_SIZES = {
    "x": 26,
    "+": 30,
    "1": 34,
    "2": 34,
    "3": 34,
    "4": 34,
}

MU_EARTH_KM3_S2 = 398600.4418
DEG_TO_RAD = np.pi / 180.0
MIN_TICK_LABEL_SIZE = 12
AXIS_LABEL_SIZE = 16
PAIRPLOT_TICK_LABEL_SIZE = 18
PAIRPLOT_AXIS_LABEL_SIZE = 22
LEGEND_FONT_SIZE = 12


@dataclass
class GroupInfo:
    source_folder: str
    inclination_group: str
    inclination_sort: float
    labels_csv_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate all cluster label CSVs across inclination groups, remap clusters to a global "
            "continuous index, and generate comprehensive plots/statistics."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="clusters",
        help="Folder containing inclination subfolders with cluster_labels_*_clusters.csv files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="clusters/global_analysis",
        help="Output folder for combined CSV tables, plots, and report.",
    )
    parser.add_argument(
        "--expected-total-clusters",
        type=int,
        default=93,
        help="Expected number of non-noise global clusters after remapping. Set <=0 to disable check.",
    )
    parser.add_argument(
        "--allow-cluster-count-mismatch",
        action="store_true",
        help="Warn instead of raising an error when total global clusters does not match expected count.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for saved figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--pairplot-max-clusters",
        type=int,
        default=0,
        help="Maximum number of largest global clusters in pairplots. Use <=0 for all clusters.",
    )
    parser.add_argument(
        "--pairplot-sample-size",
        type=int,
        default=10000,
        help="Maximum number of rows used in each pairplot.",
    )
    parser.add_argument(
        "--density-top-clusters",
        type=int,
        default=0,
        help="Number of largest clusters in density+ellipse plot. Use <=0 for all clusters.",
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=3000,
        help="Maximum rows used for silhouette analysis.",
    )
    args = parser.parse_args()

    if args.dpi < 72:
        raise ValueError("--dpi must be >= 72")
    if args.pairplot_max_clusters < 0:
        raise ValueError("--pairplot-max-clusters must be >= 0")
    if args.pairplot_sample_size < 200:
        raise ValueError("--pairplot-sample-size must be >= 200")
    if args.density_top_clusters < 0:
        raise ValueError("--density-top-clusters must be >= 0")
    if args.silhouette_sample_size < 500:
        raise ValueError("--silhouette-sample-size must be >= 500")

    return args


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10.0, 7.5),
            "xtick.direction": "in",
            "xtick.labelsize": 14,
            "xtick.major.size": 3,
            "xtick.major.width": 0.5,
            "xtick.minor.size": 1.5,
            "xtick.minor.width": 0.5,
            "xtick.minor.visible": True,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": 14,
            "ytick.major.size": 3,
            "ytick.major.width": 0.5,
            "ytick.minor.size": 1.5,
            "ytick.minor.width": 0.5,
            "ytick.minor.visible": True,
            "ytick.right": True,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "legend.fontsize": 14,
            "legend.frameon": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 12,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.color": "0.5",
            "lines.markersize": 8,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def _extract_inclination_group(folder_name: str) -> tuple[str, float]:
    match = re.search(r"starlink_labels_(.+)$", folder_name)
    label = match.group(1) if match else folder_name
    try:
        sort_val = float(label)
    except ValueError:
        sort_val = math.inf
    return label, sort_val


def _resolve_path(base: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else base / p


def discover_cluster_label_files(input_root: Path) -> list[GroupInfo]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    groups: list[GroupInfo] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        label_files = sorted(child.glob("cluster_labels_*_clusters.csv"))
        if not label_files:
            continue
        if len(label_files) > 1:
            print(
                f"Warning: multiple labels CSVs in {child}. Using first: {label_files[0].name}"
            )
        inc_label, inc_sort = _extract_inclination_group(child.name)
        groups.append(
            GroupInfo(
                source_folder=child.name,
                inclination_group=inc_label,
                inclination_sort=inc_sort,
                labels_csv_path=label_files[0],
            )
        )

    if not groups:
        raise FileNotFoundError(
            f"No cluster_labels_*_clusters.csv files found below: {input_root}"
        )

    groups.sort(key=lambda g: (g.inclination_sort, g.inclination_group, g.source_folder))
    return groups


def load_group_dataframe(group: GroupInfo) -> pd.DataFrame:
    df = pd.read_csv(group.labels_csv_path)

    required_cols = {"cluster_label", "umap_1", "umap_2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {group.labels_csv_path}: {sorted(missing)}"
        )

    df["cluster_label"] = pd.to_numeric(df["cluster_label"], errors="coerce")
    if df["cluster_label"].isna().any():
        raise ValueError(f"Found NaN cluster_label values in: {group.labels_csv_path}")
    df["cluster_label"] = df["cluster_label"].astype(int)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_cols = [
        "sma", "ecc", "inc", "raan", "aop", "true_anomaly", "umap_1", "umap_2", "umap_3"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source_folder"] = group.source_folder
    df["inclination_group"] = group.inclination_group
    df["inclination_sort"] = group.inclination_sort
    return df


def remap_clusters_globally(
    group_dfs: list[tuple[GroupInfo, pd.DataFrame]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    all_frames: list[pd.DataFrame] = []
    mapping_rows: list[dict] = []
    group_summary_rows: list[dict] = []

    next_global_id = 1
    for group, df in group_dfs:
        local_labels = sorted(int(v) for v in np.unique(df["cluster_label"].to_numpy()))
        non_noise_local = [lbl for lbl in local_labels if lbl >= 0]

        local_to_global = {
            lbl: gid for lbl, gid in zip(non_noise_local, range(next_global_id, next_global_id + len(non_noise_local)))
        }

        start_id = next_global_id if non_noise_local else 0
        end_id = (next_global_id + len(non_noise_local) - 1) if non_noise_local else 0

        local_series = df["cluster_label"].astype(int)
        global_series = local_series.map(lambda x: 0 if x < 0 else local_to_global[x])

        df_remapped = df.copy()
        df_remapped["local_cluster_label"] = local_series
        df_remapped["global_cluster_id"] = global_series.astype(int)
        all_frames.append(df_remapped)

        for local_lbl in local_labels:
            global_lbl = 0 if local_lbl < 0 else local_to_global[local_lbl]
            mask = local_series == local_lbl
            mapping_rows.append(
                {
                    "source_folder": group.source_folder,
                    "inclination_group": group.inclination_group,
                    "inclination_sort": group.inclination_sort,
                    "local_cluster_label": int(local_lbl),
                    "global_cluster_id": int(global_lbl),
                    "record_count": int(mask.sum()),
                    "is_noise": int(local_lbl < 0),
                }
            )

        noise_count = int((local_series < 0).sum())
        record_count = len(df_remapped)
        group_summary_rows.append(
            {
                "source_folder": group.source_folder,
                "inclination_group": group.inclination_group,
                "inclination_sort": group.inclination_sort,
                "cluster_count": len(non_noise_local),
                "global_cluster_start": int(start_id),
                "global_cluster_end": int(end_id),
                "record_count": int(record_count),
                "noise_count": int(noise_count),
                "noise_fraction": float(noise_count / record_count) if record_count else np.nan,
            }
        )

        next_global_id += len(non_noise_local)

    combined = pd.concat(all_frames, ignore_index=True)
    mapping_df = pd.DataFrame(mapping_rows).sort_values(
        ["inclination_sort", "inclination_group", "local_cluster_label"]
    ).reset_index(drop=True)
    group_summary_df = pd.DataFrame(group_summary_rows).sort_values(
        ["inclination_sort", "inclination_group"]
    ).reset_index(drop=True)

    total_non_noise_clusters = next_global_id - 1
    return combined, mapping_df, group_summary_df, total_non_noise_clusters


def build_global_cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    group = df.groupby("global_cluster_id", dropna=False)
    stats = group.size().rename("count").to_frame()
    stats["fraction"] = stats["count"] / float(len(df))
    stats["is_noise"] = (stats.index == 0).astype(int)

    numeric_cols = [
        "sma", "ecc", "inc", "raan", "aop", "true_anomaly", "umap_1", "umap_2", "umap_3"
    ]
    available = [col for col in numeric_cols if col in df.columns]
    if available:
        moments = group[available].agg(["mean", "std", "min", "max"])
        moments.columns = [f"{col}_{stat}" for col, stat in moments.columns]
        stats = stats.join(moments)

    stats = stats.reset_index().sort_values("global_cluster_id").reset_index(drop=True)
    return stats


def build_style_maps(cluster_ids: np.ndarray) -> tuple[dict[int, str], dict[int, str]]:
    positive_ids = sorted(int(v) for v in cluster_ids if int(v) > 0)
    color_map: dict[int, str] = {0: "gray"}
    marker_map: dict[int, str] = {0: "x"}

    n_colors = len(COLORS)
    n_markers = len(MARKERS)
    for i, cid in enumerate(positive_ids):
        color_idx = i % n_colors
        marker_repeat_idx = i // n_markers
        marker_idx = (i + marker_repeat_idx) % n_markers
        color_map[cid] = COLORS[color_idx]
        marker_map[cid] = MARKERS[marker_idx]

    return color_map, marker_map


def build_pairplot_style_maps(
    unique_cluster_labels: list[int],
    color_cycle: list[str],
    marker_cycle: list[str],
) -> tuple[dict[int, str], list[str]]:
    """Build pairplot color/marker maps with marker offset per color-cycle pass."""
    cluster_colors_out: dict[int, str] = {}
    cluster_markers_out: list[str] = []
    n_colors = len(color_cycle)
    n_markers = len(marker_cycle)

    for i, label in enumerate(unique_cluster_labels):
        color_idx = i % n_colors
        cycle_idx = i // n_colors
        marker_idx = (i + cycle_idx) % n_markers
        cluster_colors_out[int(label)] = color_cycle[color_idx]
        cluster_markers_out.append(marker_cycle[marker_idx])

    return cluster_colors_out, cluster_markers_out


def orb2xyz(mu: float, oe: np.ndarray) -> np.ndarray:
    p = oe[0] * (1.0 - oe[1] ** 2)
    cos_nu = np.cos(oe[5])
    sin_nu = np.sin(oe[5])
    r = p / (1.0 + oe[1] * cos_nu)
    rf_vec = np.array([r * cos_nu, r * sin_nu, 0.0], dtype=float)
    factor = np.sqrt(mu / p)
    vf_vec = np.array(
        [-factor * sin_nu, factor * (oe[1] + cos_nu), 0.0],
        dtype=float,
    )

    cos_w = np.cos(oe[3])
    sin_w = np.sin(oe[3])
    cos_O = np.cos(oe[4])
    sin_O = np.sin(oe[4])
    cos_i = np.cos(oe[2])
    sin_i = np.sin(oe[2])
    rot = np.array(
        [
            [
                cos_O * cos_w - sin_O * sin_w * cos_i,
                -cos_O * sin_w - sin_O * cos_w * cos_i,
                sin_O * sin_i,
            ],
            [
                sin_O * cos_w + cos_O * sin_w * cos_i,
                -sin_O * sin_w + cos_O * cos_w * cos_i,
                -cos_O * sin_i,
            ],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ],
        dtype=float,
    )
    r_vec_inertial = rot @ rf_vec
    v_vec_inertial = rot @ vf_vec
    return np.concatenate((r_vec_inertial, v_vec_inertial))


def _safe_row_normalize(vectors: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = norms[:, 0] > eps
    unit = np.zeros_like(vectors)
    unit[valid] = vectors[valid] / norms[valid]
    return unit, valid


def _compute_orbital_vectors(orbital_elements: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = orbital_elements.shape[0]
    radius_vectors = np.zeros((n_rows, 3), dtype=float)
    velocity_vectors = np.zeros((n_rows, 3), dtype=float)

    for idx, oe in enumerate(orbital_elements):
        state_vec = orb2xyz(mu, oe)
        radius_vectors[idx] = state_vec[:3]
        velocity_vectors[idx] = state_vec[3:]

    angular_momentum_vectors = np.cross(radius_vectors, velocity_vectors)
    r_norm = np.linalg.norm(radius_vectors, axis=1, keepdims=True)
    valid_r = r_norm[:, 0] > 1e-12

    eccentricity_vectors = np.zeros_like(radius_vectors)
    eccentricity_vectors[valid_r] = (
        np.cross(velocity_vectors[valid_r], angular_momentum_vectors[valid_r]) / mu
        - (radius_vectors[valid_r] / r_norm[valid_r])
    )

    nodal_vectors = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), angular_momentum_vectors)
    e_unit, e_valid = _safe_row_normalize(eccentricity_vectors)
    h_unit, h_valid = _safe_row_normalize(angular_momentum_vectors)
    perifocal_minor_axis = np.zeros_like(e_unit)
    valid_perifocal = e_valid & h_valid
    perifocal_minor_axis[valid_perifocal] = np.cross(
        h_unit[valid_perifocal],
        e_unit[valid_perifocal],
    )
    perifocal_minor_axis, _ = _safe_row_normalize(perifocal_minor_axis)

    return (
        radius_vectors,
        velocity_vectors,
        angular_momentum_vectors,
        eccentricity_vectors,
        nodal_vectors,
        perifocal_minor_axis,
    )


def _apply_export_text_style(
    fig: plt.Figure,
    tick_label_size: int = MIN_TICK_LABEL_SIZE,
    axis_label_size: int = AXIS_LABEL_SIZE,
    legend_font_size: int = LEGEND_FONT_SIZE,
) -> None:
    for ax in fig.axes:
        ax.set_title("")
        ax.tick_params(axis="both", which="both", labelsize=tick_label_size)
        ax.xaxis.label.set_size(axis_label_size)
        ax.yaxis.label.set_size(axis_label_size)
        if hasattr(ax, "zaxis"):
            ax.zaxis.label.set_size(axis_label_size)
            ax.tick_params(axis="z", which="both", labelsize=tick_label_size)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(legend_font_size)
            if legend.get_title() is not None:
                legend.get_title().set_fontsize(legend_font_size)

    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")


def _save_figure(fig: plt.Figure, out_path: Path, dpi: int, show: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _apply_export_text_style(fig)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _sample_dataframe(
    df: pd.DataFrame,
    cluster_col: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()

    frac = max_rows / float(len(df))
    sampled_parts = []
    for _, group in df.groupby(cluster_col):
        n = max(1, int(round(len(group) * frac)))
        n = min(n, len(group))
        sampled_parts.append(group.sample(n=n, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    return sampled.reset_index(drop=True)


def plot_umap_scatter_2d(
    df: pd.DataFrame,
    color_map: dict[int, str],
    marker_map: dict[int, str],
    out_path: Path,
    dpi: int,
) -> None:
    if not {"umap_1", "umap_2"}.issubset(df.columns):
        print("Skipping 2D UMAP scatter: umap_1/umap_2 not found.")
        return

    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    cluster_ids = sorted(int(v) for v in np.unique(df["global_cluster_id"].to_numpy()))
    legend_handles = []
    for cid in cluster_ids:
        mask = df["global_cluster_id"] == cid
        marker = marker_map.get(cid, "o")
        color = color_map.get(cid, "#222222")
        size = MARKER_SIZES.get(marker, 16)
        alpha = 0.25 if cid == 0 else 0.7
        ax.scatter(
            df.loc[mask, "umap_1"],
            df.loc[mask, "umap_2"],
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            linewidths=0.0,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle="None",
                markersize=7,
                markerfacecolor="none" if marker in {"x", "+", "1", "2", "3", "4"} else color,
                markeredgecolor=color,
                label="Noise" if cid == 0 else f"Cluster {cid}",
                alpha=alpha,
            )
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if legend_handles:
        legend_cols = max(1, math.ceil(len(legend_handles) / 32))
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=LEGEND_FONT_SIZE,
            title="Clusters",
            title_fontsize=LEGEND_FONT_SIZE,
            ncol=legend_cols,
            handletextpad=0.3,
            columnspacing=0.8,
        )

    _save_figure(fig, out_path, dpi, show=False)


def plot_umap_scatter_3d(
    df: pd.DataFrame,
    color_map: dict[int, str],
    marker_map: dict[int, str],
    out_path: Path,
    dpi: int,
) -> None:
    if not {"umap_1", "umap_2", "umap_3"}.issubset(df.columns):
        print("Skipping 3D UMAP scatter: umap_1/umap_2/umap_3 not found.")
        return

    fig = plt.figure(figsize=(11.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    cluster_ids = sorted(int(v) for v in np.unique(df["global_cluster_id"].to_numpy()))
    for cid in cluster_ids:
        mask = df["global_cluster_id"] == cid
        marker = marker_map.get(cid, "o")
        color = color_map.get(cid, "#222222")
        size = MARKER_SIZES.get(marker, 16)
        alpha = 0.25 if cid == 0 else 0.72
        ax.scatter(
            df.loc[mask, "umap_1"],
            df.loc[mask, "umap_2"],
            df.loc[mask, "umap_3"],
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            linewidths=0.0,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    _save_figure(fig, out_path, dpi, show=False)


def plot_density_ellipses(
    df: pd.DataFrame,
    color_map: dict[int, str],
    marker_map: dict[int, str],
    out_path: Path,
    dpi: int,
    top_n_clusters: int,
) -> None:
    if not {"umap_1", "umap_2"}.issubset(df.columns):
        print("Skipping density plot: umap_1/umap_2 not found.")
        return

    non_noise = df[df["global_cluster_id"] > 0]
    cluster_counts = non_noise["global_cluster_id"].value_counts()
    if top_n_clusters > 0:
        focus_clusters = cluster_counts.head(top_n_clusters).index.tolist()
    else:
        focus_clusters = cluster_counts.index.tolist()
    if not focus_clusters:
        print("Skipping density plot: no non-noise clusters found.")
        return

    fig, ax = plt.subplots(figsize=(11.5, 8.0))

    # Plot all noise points faintly in the background.
    noise = df[df["global_cluster_id"] == 0]
    if not noise.empty:
        ax.scatter(noise["umap_1"], noise["umap_2"], c="gray", s=8, alpha=0.15, marker="x")

    for cid in focus_clusters:
        sub = df[df["global_cluster_id"] == cid]
        if sub.empty:
            continue
        color = color_map.get(int(cid), "#222222")
        marker = marker_map.get(int(cid), "o")
        marker_size = MARKER_SIZES.get(marker, 18)

        if len(sub) > 5:
            try:
                sns.kdeplot(
                    data=sub,
                    x="umap_1",
                    y="umap_2",
                    fill=True,
                    alpha=0.20,
                    levels=5,
                    thresh=0.05,
                    ax=ax,
                    color=color,
                )
            except Exception:
                pass

        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            marker=marker,
            c=color,
            s=marker_size,
            alpha=0.70,
            linewidths=0.0,
        )

        # Add 2-sigma confidence ellipse where covariance is valid.
        if len(sub) > 2:
            cov = np.cov(sub["umap_1"], sub["umap_2"])
            if np.isfinite(cov).all():
                eigvals, eigvecs = np.linalg.eigh(cov)
                if np.all(eigvals > 0):
                    order = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]
                    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                    width, height = 2 * 2.0 * np.sqrt(eigvals)
                    ellipse = mpatches.Ellipse(
                        xy=(sub["umap_1"].mean(), sub["umap_2"].mean()),
                        width=width,
                        height=height,
                        angle=angle,
                        edgecolor="black",
                        facecolor="none",
                        lw=1.5,
                    )
                    ax.add_patch(ellipse)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    _save_figure(fig, out_path, dpi, show=False)


def _is_degree_axis_label(label: str) -> bool:
    return "[deg]" in label


def _apply_pairplot_degree_axis_limits(grid) -> None:
    for row_idx, ax_row in enumerate(grid.axes):
        for col_idx, ax in enumerate(ax_row):
            if ax is None:
                continue
            x_var = grid.x_vars[col_idx]
            y_var = grid.y_vars[row_idx]
            if _is_degree_axis_label(str(x_var)):
                ax.set_xlim(0, 360)
            if _is_degree_axis_label(str(y_var)):
                ax.set_ylim(0, 360)

    if hasattr(grid, "diag_axes"):
        for col_idx, ax in enumerate(grid.diag_axes):
            if ax is None:
                continue
            x_var = grid.x_vars[col_idx]
            if _is_degree_axis_label(str(x_var)):
                ax.set_xlim(0, 360)


def plot_orbital_elements_pairplot(
    df: pd.DataFrame,
    color_map: dict[int, str],
    out_path: Path,
    dpi: int,
    max_clusters: int,
    sample_size: int,
    seed: int,
) -> None:
    feature_cols = ["sma", "ecc", "inc", "raan", "aop", "true_anomaly"]
    available = [col for col in feature_cols if col in df.columns]
    if len(available) < 3:
        print("Skipping orbital pairplot: fewer than 3 orbital element columns available.")
        return

    non_noise = df[df["global_cluster_id"] > 0].dropna(subset=available)
    if non_noise.empty:
        print("Skipping orbital pairplot: no non-noise records available.")
        return

    if max_clusters > 0:
        focus_ids = non_noise["global_cluster_id"].value_counts().head(max_clusters).index.tolist()
    else:
        focus_ids = sorted(non_noise["global_cluster_id"].unique().tolist())
    pair_df = non_noise[non_noise["global_cluster_id"].isin(focus_ids)].copy()
    pair_df = _sample_dataframe(pair_df, "global_cluster_id", sample_size, seed)

    rename_map = {
        "sma": "Semi-Major Axis [km]",
        "ecc": "Eccentricity",
        "inc": "Inclination [deg]",
        "raan": "RAAN [deg]",
        "aop": "AOP [deg]",
        "true_anomaly": "True Anomaly [deg]",
    }
    use_cols = available + ["global_cluster_id"]
    plot_df = pair_df[use_cols].rename(columns=rename_map)
    hue_order = sorted(plot_df["global_cluster_id"].unique().tolist())

    pairplot_palette, marker_list = build_pairplot_style_maps(
        unique_cluster_labels=[int(cid) for cid in hue_order],
        color_cycle=COLORS,
        marker_cycle=PAIRPLOT_MARKERS_FILLED,
    )
    palette = {cid: pairplot_palette[int(cid)] for cid in hue_order}

    g = sns.pairplot(
        plot_df,
        hue="global_cluster_id",
        hue_order=hue_order,
        markers=marker_list,
        diag_kind="kde",
        corner=True,
        palette=palette,
        aspect=1.15,
        plot_kws={"s": 16, "alpha": 0.75, "linewidth": 0.0},
    )
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
    _apply_pairplot_degree_axis_limits(g)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _apply_export_text_style(
        g.fig,
        tick_label_size=PAIRPLOT_TICK_LABEL_SIZE,
        axis_label_size=PAIRPLOT_AXIS_LABEL_SIZE,
    )
    g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


def plot_orbital_vectors_2d_with_clusters(
    df: pd.DataFrame,
    color_map: dict[int, str],
    marker_map: dict[int, str],
    out_path: Path,
    dpi: int,
) -> None:
    required_cols = {
        "sma",
        "ecc",
        "inc",
        "raan",
        "aop",
        "true_anomaly",
        "global_cluster_id",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Skipping orbital-vector 2D plot: missing columns {sorted(missing)}")
        return

    work_df = df.dropna(
        subset=["sma", "ecc", "inc", "raan", "aop", "true_anomaly", "global_cluster_id"]
    ).copy()
    if work_df.empty:
        print("Skipping orbital-vector 2D plot: no valid rows after dropping NaNs.")
        return

    sma = work_df["sma"].to_numpy(dtype=float)
    ecc = work_df["ecc"].to_numpy(dtype=float)
    inc = work_df["inc"].to_numpy(dtype=float) * DEG_TO_RAD
    raan = work_df["raan"].to_numpy(dtype=float) * DEG_TO_RAD
    aop = work_df["aop"].to_numpy(dtype=float) * DEG_TO_RAD
    true_anomaly = work_df["true_anomaly"].to_numpy(dtype=float) * DEG_TO_RAD
    labels = work_df["global_cluster_id"].to_numpy(dtype=int)

    orbital_elements = np.column_stack((sma, ecc, inc, aop, raan, true_anomaly))
    (
        radius_vectors,
        velocity_vectors,
        angular_momentum_vectors,
        eccentricity_vectors,
        nodal_vectors,
        perifocal_minor_axis,
    ) = _compute_orbital_vectors(orbital_elements, MU_EARTH_KM3_S2)

    fig = plt.figure(figsize=(9.9, 7.5))
    plt.subplots_adjust(hspace=0.238)

    vector_specs = [
        ("Radius Vectors", radius_vectors, "X (km)", "Y (km)"),
        ("Angular Momentum Vectors", angular_momentum_vectors, "X (km^2/s)", "Y (km^2/s)"),
        ("Eccentricity Vectors", eccentricity_vectors, "X (dimensionless)", "Y (dimensionless)"),
        ("Nodal Vectors", nodal_vectors, "X (km^2/s)", "Y (km^2/s)"),
        ("Velocity Vectors", velocity_vectors, "X (km/s)", "Y (km/s)"),
        ("Perifocal Minor Axis Vectors", perifocal_minor_axis, "X (dimensionless)", "Y (dimensionless)"),
    ]

    unique_labels = sorted(int(v) for v in np.unique(labels))
    for idx, (_title, vectors, x_label, y_label) in enumerate(vector_specs, start=1):
        ax = fig.add_subplot(2, 3, idx)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        for cid in unique_labels:
            mask = labels == cid
            if not np.any(mask):
                continue

            marker = marker_map.get(cid, "o")
            color = color_map.get(cid, "#222222")
            marker_size = MARKER_SIZES.get(marker, 16)
            alpha = 0.25 if cid == 0 else 0.75

            ax.scatter(
                vectors[mask, 0],
                vectors[mask, 1],
                c=color,
                marker=marker,
                s=marker_size,
                alpha=alpha,
                linewidths=0.0,
            )

    _save_figure(fig, out_path, dpi, show=False)


def plot_cluster_size_distribution(
    df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    counts = (
        df[df["global_cluster_id"] > 0]["global_cluster_id"]
        .value_counts()
        .sort_index()
    )
    if counts.empty:
        print("Skipping cluster-size distribution plot: no non-noise clusters.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.8))

    bins = min(30, max(8, int(np.sqrt(len(counts)))))
    ax1.hist(counts.values, bins=bins, color=COLORS[0], edgecolor="white")
    ax1.set_xlabel("Records per Cluster")
    ax1.set_ylabel("Number of Clusters")

    sorted_desc = np.sort(counts.values)[::-1]
    cumulative = np.cumsum(sorted_desc) / sorted_desc.sum()
    ax2.plot(np.arange(1, len(sorted_desc) + 1), cumulative, color=COLORS[2], marker="o", markersize=3)
    ax2.set_xlabel("Top-N Clusters")
    ax2.set_ylabel("Cumulative Share of Non-Noise Records")
    ax2.set_ylim(0, 1.02)

    _save_figure(fig, out_path, dpi)


def plot_top_cluster_sizes(
    df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    top_n: int = 30,
) -> None:
    counts = (
        df[df["global_cluster_id"] > 0]["global_cluster_id"]
        .value_counts()
        .head(top_n)
    )
    if counts.empty:
        print("Skipping top cluster-size bar plot: no non-noise clusters.")
        return

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    x = counts.index.astype(int).astype(str).tolist()
    y = counts.values
    ax.bar(x, y, color=COLORS[1])
    ax.set_xlabel("Global Cluster ID")
    ax.set_ylabel("Record Count")
    ax.tick_params(axis="x", rotation=90)
    _save_figure(fig, out_path, dpi)


def plot_inclination_summary(
    group_summary_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    if group_summary_df.empty:
        print("Skipping inclination summary plot: summary table is empty.")
        return

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2))
    x = group_summary_df["inclination_group"].astype(str)
    y_clusters = group_summary_df["cluster_count"]
    y_noise = group_summary_df["noise_fraction"]

    bars = ax1.bar(x, y_clusters, color=COLORS[4], alpha=0.85)
    ax1.set_xlabel("Inclination Group")
    ax1.set_ylabel("Non-Noise Cluster Count")

    for bar, start_id, end_id in zip(
        bars,
        group_summary_df["global_cluster_start"],
        group_summary_df["global_cluster_end"],
    ):
        if int(start_id) > 0 and int(end_id) >= int(start_id):
            txt = f"{int(start_id)}-{int(end_id)}"
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                txt,
                ha="center",
                va="bottom",
                fontsize=12,
            )

    ax2 = ax1.twinx()
    ax2.plot(x, y_noise, color=COLORS[3], marker="o", linewidth=2.0)
    ax2.set_ylabel("Noise Fraction")
    ax2.set_ylim(0, max(0.3, float(np.nanmax(y_noise) * 1.2)))

    _save_figure(fig, out_path, dpi)


def plot_inclination_cluster_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    matrix = pd.crosstab(df["inclination_group"], df["global_cluster_id"])
    matrix = matrix.drop(columns=[0], errors="ignore")
    if matrix.empty:
        print("Skipping inclination/cluster heatmap: no non-noise clusters.")
        return

    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    matrix_log = np.log1p(matrix)

    fig, ax = plt.subplots(figsize=(16.0, 5.8))
    sns.heatmap(
        matrix_log,
        cmap="viridis",
        cbar_kws={"label": "log(1 + record count)"},
        ax=ax,
    )
    ax.set_xlabel("Global Cluster ID")
    ax.set_ylabel("Inclination Group")

    n_cols = matrix.shape[1]
    if n_cols > 30:
        step = max(1, n_cols // 20)
        tick_pos = np.arange(0, n_cols, step) + 0.5
        tick_lbl = [str(matrix.columns[i]) for i in range(0, n_cols, step)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, rotation=45, fontsize=12)

    _save_figure(fig, out_path, dpi)


def plot_feature_boxplots(
    df: pd.DataFrame,
    group_order: list[str],
    out_path: Path,
    dpi: int,
) -> None:
    candidates = ["sma", "ecc", "inc", "raan", "aop", "true_anomaly"]
    features = [c for c in candidates if c in df.columns]
    if not features:
        print("Skipping feature boxplots: no supported orbital element columns found.")
        return

    n = len(features)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.0, 4.8 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, feature in enumerate(features):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        sns.boxplot(
            data=df,
            x="inclination_group",
            y=feature,
            order=group_order,
            ax=ax,
            fliersize=1.0,
            linewidth=0.7,
            color=COLORS[idx % len(COLORS)],
        )
        ax.tick_params(axis="x", rotation=45)

    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    _save_figure(fig, out_path, dpi)


def plot_centroid_feature_correlation(
    df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    feature_candidates = [
        "sma", "ecc", "inc", "raan", "aop", "true_anomaly", "umap_1", "umap_2", "umap_3"
    ]
    features = [c for c in feature_candidates if c in df.columns]
    if len(features) < 2:
        print("Skipping centroid correlation heatmap: insufficient numeric features.")
        return

    centroids = (
        df[df["global_cluster_id"] > 0]
        .groupby("global_cluster_id")[features]
        .mean()
    )
    if centroids.empty:
        print("Skipping centroid correlation heatmap: no non-noise clusters.")
        return

    display_names = {
        "sma": "Semi-major Axis",
        "ecc": "Eccentricity",
        "inc": "Inclination",
        "raan": "RAAN",
        "aop": "AOP",
        "true_anomaly": "True Anomaly",
        "umap_1": "UMAP 1",
        "umap_2": "UMAP 2",
        "umap_3": "UMAP 3",
    }
    corr = centroids.corr().rename(index=display_names, columns=display_names)
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=False,
        square=True,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _save_figure(fig, out_path, dpi)


def run_silhouette_analysis(
    df: pd.DataFrame,
    out_dir: Path,
    sample_size: int,
    seed: int,
    dpi: int,
) -> tuple[float | None, pd.DataFrame | None]:
    feature_candidates = [
        "sma", "ecc", "inc", "raan", "aop", "true_anomaly", "umap_1", "umap_2", "umap_3"
    ]
    features = [col for col in feature_candidates if col in df.columns]
    if len(features) < 2:
        print("Skipping silhouette analysis: need at least two numeric feature columns.")
        return None, None

    valid = df[df["global_cluster_id"] > 0].dropna(subset=features)
    if valid["global_cluster_id"].nunique() < 2 or len(valid) < 10:
        print("Skipping silhouette analysis: not enough non-noise clusters/records.")
        return None, None

    work = _sample_dataframe(valid, "global_cluster_id", sample_size, seed)
    x = work[features].to_numpy(dtype=float)
    y = work["global_cluster_id"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    overall = float(silhouette_score(x_scaled, y, metric="euclidean"))
    samples = silhouette_samples(x_scaled, y, metric="euclidean")

    sil_df = pd.DataFrame(
        {
            "global_cluster_id": y,
            "silhouette": samples,
        }
    )

    summary = (
        sil_df.groupby("global_cluster_id")["silhouette"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "silhouette_cluster_summary.csv", index=False, float_format="%.8g")

    fig1, ax1 = plt.subplots(figsize=(10.5, 5.8))
    ax1.hist(samples, bins=60, color=COLORS[6], edgecolor="white")
    ax1.axvline(overall, color="black", linestyle="--", linewidth=1.5, label=f"Mean = {overall:.4f}")
    ax1.set_xlabel("Silhouette Score")
    ax1.set_ylabel("Frequency")
    ax1.legend(loc="upper left")
    _save_figure(fig1, out_dir / "silhouette_histogram.png", dpi)

    top_clusters = (
        sil_df["global_cluster_id"].value_counts().head(20).index.tolist()
    )
    sil_top = sil_df[sil_df["global_cluster_id"].isin(top_clusters)]
    fig2, ax2 = plt.subplots(figsize=(13.0, 5.8))
    sns.boxplot(
        data=sil_top,
        x="global_cluster_id",
        y="silhouette",
        ax=ax2,
        color=COLORS[11],
        fliersize=1.0,
        linewidth=0.7,
    )
    ax2.set_xlabel("Global Cluster ID")
    ax2.set_ylabel("Silhouette Score")
    ax2.tick_params(axis="x", rotation=45)
    _save_figure(fig2, out_dir / "silhouette_boxplot_top_clusters.png", dpi)

    return overall, summary


def write_report(
    report_path: Path,
    combined_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    global_stats_df: pd.DataFrame,
    total_non_noise_clusters: int,
    expected_total_clusters: int,
    silhouette_overall: float | None,
) -> None:
    non_noise = combined_df[combined_df["global_cluster_id"] > 0]
    noise = combined_df[combined_df["global_cluster_id"] == 0]

    largest_clusters = (
        global_stats_df[global_stats_df["global_cluster_id"] > 0]
        .sort_values("count", ascending=False)
        .head(15)
    )

    lines: list[str] = []
    lines.append("# Global Cluster Analysis Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total records: {len(combined_df)}")
    lines.append(f"Non-noise records: {len(non_noise)}")
    lines.append(f"Noise records: {len(noise)}")
    lines.append(f"Noise fraction: {len(noise) / float(len(combined_df)):.4f}")
    lines.append(f"Total non-noise global clusters: {total_non_noise_clusters}")
    if expected_total_clusters > 0:
        lines.append(f"Expected non-noise global clusters: {expected_total_clusters}")
        lines.append(
            "Cluster count check: "
            + (
                "PASS"
                if total_non_noise_clusters == expected_total_clusters
                else "MISMATCH"
            )
        )
    if silhouette_overall is not None:
        lines.append(f"Sampled silhouette mean score: {silhouette_overall:.6f}")

    lines.append("")
    lines.append("## Inclination Group Ranges")
    lines.append(group_summary_df.to_string(index=False))

    lines.append("")
    lines.append("## Largest Global Clusters")
    if largest_clusters.empty:
        lines.append("No non-noise clusters found.")
    else:
        lines.append(largest_clusters[["global_cluster_id", "count", "fraction"]].to_string(index=False))

    lines.append("")
    lines.append("## Notes")
    lines.append("- Global cluster IDs are continuous across inclination groups; 0 is reserved for noise.")
    lines.append("- Inclination groups are sorted numerically when possible (53.0, 53.2, 70, 97, ...).")
    lines.append("- Figures are designed for offline review and publication-style export.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = parse_args()
    configure_plot_style()
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    input_root = _resolve_path(script_dir, args.input_root)
    output_root = _resolve_path(script_dir, args.output_root)

    tables_dir = output_root / "tables"
    plots_dir = output_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")

    groups = discover_cluster_label_files(input_root)
    print("Discovered group files:")
    for g in groups:
        print(
            f"  - {g.source_folder}: {g.labels_csv_path.name} "
            f"(inclination_group={g.inclination_group})"
        )

    group_dfs: list[tuple[GroupInfo, pd.DataFrame]] = []
    for group in groups:
        df = load_group_dataframe(group)
        group_dfs.append((group, df))
        print(f"Loaded {len(df):6d} rows from {group.labels_csv_path}")

    combined_df, mapping_df, group_summary_df, total_non_noise_clusters = remap_clusters_globally(group_dfs)

    msg = (
        f"Total non-noise global clusters after remapping: {total_non_noise_clusters}"
    )
    print(msg)

    if args.expected_total_clusters > 0 and total_non_noise_clusters != args.expected_total_clusters:
        mismatch_msg = (
            f"Expected {args.expected_total_clusters} non-noise clusters but found "
            f"{total_non_noise_clusters}."
        )
        if args.allow_cluster_count_mismatch:
            print(f"Warning: {mismatch_msg}")
        else:
            raise ValueError(mismatch_msg)

    combined_df = combined_df.sort_values(
        ["inclination_sort", "inclination_group", "global_cluster_id"]
    ).reset_index(drop=True)

    global_stats_df = build_global_cluster_stats(combined_df)

    combined_csv = tables_dir / "combined_cluster_labels_global.csv"
    mapping_csv = tables_dir / "cluster_local_to_global_mapping.csv"
    summary_csv = tables_dir / "inclination_group_summary.csv"
    global_stats_csv = tables_dir / "global_cluster_stats.csv"

    combined_df.to_csv(combined_csv, index=False, float_format="%.8g", date_format="%Y-%m-%d %H:%M:%S")
    mapping_df.to_csv(mapping_csv, index=False, float_format="%.8g")
    group_summary_df.to_csv(summary_csv, index=False, float_format="%.8g")
    global_stats_df.to_csv(global_stats_csv, index=False, float_format="%.8g")

    print(f"Saved: {combined_csv}")
    print(f"Saved: {mapping_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {global_stats_csv}")

    color_map, marker_map = build_style_maps(combined_df["global_cluster_id"].to_numpy())

    plot_umap_scatter_2d(
        combined_df,
        color_map,
        marker_map,
        plots_dir / "01_umap_scatter_global_2d.png",
        args.dpi,
    )
    plot_umap_scatter_3d(
        combined_df,
        color_map,
        marker_map,
        plots_dir / "02_umap_scatter_global_3d.png",
        args.dpi,
    )
    plot_density_ellipses(
        combined_df,
        color_map,
        marker_map,
        plots_dir / "03_umap_density_ellipses_top_clusters.png",
        args.dpi,
        args.density_top_clusters,
    )
    plot_orbital_elements_pairplot(
        combined_df,
        color_map,
        plots_dir / "04_pairplot_orbital_elements_top_clusters.png",
        args.dpi,
        args.pairplot_max_clusters,
        args.pairplot_sample_size,
        args.seed,
    )
    plot_orbital_vectors_2d_with_clusters(
        combined_df,
        color_map,
        marker_map,
        plots_dir / "11_orbital_vectors_2d_cluster_labels.png",
        args.dpi,
    )

    plot_cluster_size_distribution(
        combined_df,
        plots_dir / "05_cluster_size_distribution.png",
        args.dpi,
    )
    plot_top_cluster_sizes(
        combined_df,
        plots_dir / "06_top_cluster_sizes.png",
        args.dpi,
        top_n=30,
    )
    plot_inclination_summary(
        group_summary_df,
        plots_dir / "07_inclination_summary.png",
        args.dpi,
    )
    plot_inclination_cluster_heatmap(
        combined_df,
        plots_dir / "08_inclination_cluster_heatmap.png",
        args.dpi,
    )

    group_order = group_summary_df["inclination_group"].astype(str).tolist()
    plot_feature_boxplots(
        combined_df,
        group_order,
        plots_dir / "09_orbital_feature_boxplots_by_inclination.png",
        args.dpi,
    )
    plot_centroid_feature_correlation(
        combined_df,
        plots_dir / "10_cluster_centroid_feature_correlation.png",
        args.dpi,
    )

    silhouette_overall = None
    silhouette_summary = None
    try:
        silhouette_overall, silhouette_summary = run_silhouette_analysis(
            combined_df,
            plots_dir,
            args.silhouette_sample_size,
            args.seed,
            args.dpi,
        )
        if silhouette_overall is not None:
            print(f"Silhouette mean (sampled): {silhouette_overall:.6f}")
    except Exception as exc:
        print(f"Warning: silhouette analysis failed: {exc}")

    if silhouette_summary is not None:
        silhouette_summary.to_csv(
            tables_dir / "silhouette_cluster_summary.csv",
            index=False,
            float_format="%.8g",
        )

    report_path = output_root / "analysis_report.md"
    write_report(
        report_path=report_path,
        combined_df=combined_df,
        group_summary_df=group_summary_df,
        global_stats_df=global_stats_df,
        total_non_noise_clusters=total_non_noise_clusters,
        expected_total_clusters=args.expected_total_clusters,
        silhouette_overall=silhouette_overall,
    )
    print(f"Saved: {report_path}")

    print("\nDone. Global cluster analysis artifacts are ready.")


if __name__ == "__main__":
    main()
