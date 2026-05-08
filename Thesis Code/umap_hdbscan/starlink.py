from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from load_all_tle_data import load_all_tle_data

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

colors = ["#1965B0", "#E8601C", "#4EB265", "#882E72", "#DC050C", "#896D67"]

# -----------------------------------------------------------------------------
# Load and preprocess TLE data for STARLINK
# -----------------------------------------------------------------------------
MAX_SATELLITES = 10
SATELLITE_SELECTION_MODE = 'specific'  # 'max' or 'specific'
SPECIFIC_SATELLITES = ['sat1010', 'sat1011', 'sat1013', 'sat1014', 'sat1019', 'sat1030']

def normalize_satellite_id(value: object) -> str:
    sat_id = str(value).strip().lower()
    if sat_id.endswith('.txt'):
        sat_id = sat_id[:-4]
    if sat_id.endswith('_decay'):
        sat_id = sat_id[:-6]
    return sat_id

repo_root = Path(__file__).resolve().parents[1]
folder_path = repo_root / 'starlink_decay'
plots_output_dir = repo_root / 'sat_plots'
plots_output_dir.mkdir(parents=True, exist_ok=True)
PLOTS_DPI = 600

def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(plots_output_dir / filename, dpi=PLOTS_DPI, bbox_inches='tight')

df, _ = load_all_tle_data([str(folder_path)])
df['sat_id'] = df['sat_id'].map(normalize_satellite_id)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values(['sat_id', 'timestamp'], inplace=True)
df.reset_index(drop=True, inplace=True)
available_sat_ids = sorted(df['sat_id'].astype(str).str.strip().str.lower().unique())

if SATELLITE_SELECTION_MODE == 'max':
    selected_sat_ids = df['sat_id'].drop_duplicates().head(MAX_SATELLITES)
    df = df[df['sat_id'].isin(selected_sat_ids)].copy()
elif SATELLITE_SELECTION_MODE == 'specific':
    requested_ids = [normalize_satellite_id(s) for s in SPECIFIC_SATELLITES]
    sat_id_norm = df['sat_id'].astype(str).str.strip().str.lower()
    present_ids = set(sat_id_norm.unique())
    missing_ids = [sat for sat in requested_ids if sat not in present_ids]

    df = df[sat_id_norm.isin(requested_ids)].copy()
    order_map = {sat: idx for idx, sat in enumerate(requested_ids)}
    df['_sat_norm'] = df['sat_id'].astype(str).str.strip().str.lower()
    df['_sat_order'] = df['_sat_norm'].map(order_map)
    df.sort_values(['_sat_order', 'timestamp'], inplace=True)
    df.drop(columns=['_sat_norm', '_sat_order'], inplace=True)

    if missing_ids:
        print(f"Warning: requested satellites not found in data: {missing_ids}")
else:
    raise ValueError(
        "SATELLITE_SELECTION_MODE must be either 'max' or 'specific'."
    )

if df.empty:
    raise ValueError(
        "No data remained after satellite selection. "
        f"Mode={SATELLITE_SELECTION_MODE}, requested={SPECIFIC_SATELLITES}, "
        f"available={available_sat_ids[:20]}"
    )

df_all = df.copy()

measurement_cols = ['inc', 'sma', 'ecc', 'aop', 'raan', 'true_anomaly',
                    'mean_anomaly', 'mean_motion', 'specific_angular_momentum']

missing = [c for c in measurement_cols + ['sat_id', 'timestamp', 'tle_epoch'] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Visualization
anomaly_col = 'mean_anomaly' if 'true_anomaly' in df.columns else 'mean_anomaly'
anomaly_label = 'True Anomaly (°)' if anomaly_col == 'true_anomaly' else 'Mean Anomaly (°)'

elements = [('sma', 'Semi-major Axis (km)'), ('ecc', 'Eccentricity'),
            ('inc', 'Inclination (°)'), ('aop', 'Argument of Perigee (°)'),
            (anomaly_col, anomaly_label), ('mean_motion', 'Mean Motion (rev/day)')]

stats_elements = [('sma', 'Semi-major Axis (km)'), ('ecc', 'Eccentricity'),
                  ('inc', 'Inclination (°)')]

def plot_element_all_sats(dataframe: pd.DataFrame, *, y_col: str, y_label: str, show_legend: bool = False,
                          x_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
                          y_range: tuple[float, float] | None = None,
                          save_filename: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    for idx, (sat_id, g) in enumerate(dataframe.groupby('sat_id', sort=False)):
        ax.plot(g['timestamp'], pd.to_numeric(g[y_col], errors='coerce'),
                label=str(sat_id), color=colors[idx % len(colors)], linewidth=1.0)

    ax.set_title(f"{y_label} vs. Epoch (UTC)")
    ax.set_xlabel("Epoch (UTC)")
    ax.set_ylabel(y_label)
    if x_range is not None:
        ax.set_xlim(x_range[0], x_range[1])
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])

    if show_legend:
        ax.legend(title='sat_id', ncol=2, fontsize=10)

    fig.tight_layout()
    if save_filename is not None:
        save_figure(fig, save_filename)
    plt.show()

def compute_global_statistics(dataframe: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    numeric = dataframe[cols].apply(pd.to_numeric, errors='coerce')
    summary = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    summary['iqr'] = summary['75%'] - summary['25%']
    summary['skew'] = numeric.skew(numeric_only=True)
    summary['kurtosis'] = numeric.kurt(numeric_only=True)
    summary['missing_count'] = numeric.isna().sum()
    summary['missing_fraction'] = summary['missing_count'] / len(numeric)
    return summary

def compute_per_satellite_statistics(dataframe: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    numeric = dataframe[['sat_id'] + cols].copy()
    grouped = numeric.groupby('sat_id')[cols].agg(['count', 'mean', 'std', 'min', 'median', 'max'])
    grouped.columns = ['_'.join(c) for c in grouped.columns]
    return grouped.reset_index()

def compute_trend_slopes_per_day(dataframe: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for sat_id, g in dataframe.groupby('sat_id', sort=False):
        g = g.sort_values('timestamp')
        t_days = (g['timestamp'] - g['timestamp'].iloc[0]).dt.total_seconds().to_numpy() / 86400.0

        row = {'sat_id': sat_id, 'num_points': len(g)}
        for c in cols:
            y = pd.to_numeric(g[c], errors='coerce').to_numpy()
            valid = np.isfinite(t_days) & np.isfinite(y)

            if valid.sum() >= 2 and np.nanstd(t_days[valid]) > 0:
                slope = np.polyfit(t_days[valid], y[valid], 1)[0]
            else:
                slope = np.nan
            row[f'{c}_slope_per_day'] = slope

        rows.append(row)

    return pd.DataFrame(rows)

def plot_feature_distributions(dataframe: pd.DataFrame, cols_and_labels: list[tuple[str, str]],
                               save_filename: str | None = None) -> None:
    num_plots = len(cols_and_labels)
    fig, axes = plt.subplots(1, num_plots, figsize=(4.2 * num_plots, 5.0))
    axes = np.atleast_1d(axes).ravel()

    for idx, (col, label) in enumerate(cols_and_labels):
        series = pd.to_numeric(dataframe[col], errors='coerce').dropna()
        ax = axes[idx]
        ax.hist(series, bins=70, alpha=0.85, color=colors[idx % len(colors)])
        ax.set_title(label)
        ax.set_ylabel('Count')

    for ax in axes[len(cols_and_labels):]:
        ax.axis('off')

    fig.suptitle('Global Distributions of Starlink Orbital Elements', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_filename is not None:
        save_figure(fig, save_filename)
    plt.show()

def plot_per_satellite_mean_boxplots(per_sat_stats: pd.DataFrame, cols_and_labels: list[tuple[str, str]],
                                     save_filename: str | None = None) -> None:
    plot_items: list[tuple[np.ndarray, str]] = []
    for col, label in cols_and_labels:
        mean_col = f'{col}_mean'
        if mean_col not in per_sat_stats.columns:
            continue
        vals = pd.to_numeric(per_sat_stats[mean_col], errors='coerce').dropna()
        if len(vals) > 0:
            plot_items.append((vals.to_numpy(), label))

    if not plot_items:
        return

    num_plots = min(len(plot_items), 3)
    fig, axes = plt.subplots(1, num_plots, figsize=(4.8 * num_plots, 5.2))
    axes = axes.ravel()

    for idx, (vals, label) in enumerate(plot_items[:3]):
        ax = axes[idx]
        bp = ax.boxplot(
            [vals],
            tick_labels=[label],
            showfliers=True,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            meanprops={'color': 'black', 'linewidth': 1.5},
            medianprops={'color': 'gray', 'linewidth': 1.5},
        )
        bp['boxes'][0].set_facecolor(colors[idx % len(colors)])
        bp['boxes'][0].set_alpha(0.55)
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=0)

    for ax in axes[len(plot_items[:3]):]:
        ax.axis('off')

    fig.suptitle('Distribution of Per-Satellite Means', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_filename is not None:
        save_figure(fig, save_filename)
    plt.show()

def plot_correlation_heatmap(dataframe: pd.DataFrame, cols_and_labels: list[tuple[str, str]],
                             save_filename: str | None = None) -> None:
    cols = [c for c, _ in cols_and_labels]
    labels = [l for _, l in cols_and_labels]
    corr = dataframe[cols].apply(pd.to_numeric, errors='coerce').corr()

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(False)

    rounded = np.round(corr.values, 3)
    for i in range(rounded.shape[0]):
        for j in range(rounded.shape[1]):
            text_color = 'white' if abs(rounded[i, j]) > 0.55 else 'black'
            ax.text(j, i, f"{rounded[i, j]:.3f}", ha='center', va='center', color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('ρ')
    ax.set_title('Feature Correlation Matrix')
    fig.tight_layout()
    if save_filename is not None:
        save_figure(fig, save_filename)
    plt.show()

def plot_trend_slope_distributions(trend_df: pd.DataFrame, cols_and_labels: list[tuple[str, str]],
                                   save_filename: str | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, (c, label) in enumerate(cols_and_labels):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            continue
        slope_col = f'{c}_slope_per_day'
        if slope_col not in trend_df.columns:
            ax.axis('off')
            continue
        vals = pd.to_numeric(trend_df[slope_col], errors='coerce').dropna()
        if len(vals) == 0:
            ax.axis('off')
            continue
        ax.hist(vals, bins=10, alpha=0.75, label=f'{c} slope/day',
                color=colors[idx % len(colors)])
        ax.set_title(f'{label} slope/day')
        ax.set_xlabel('Slope per day')
        ax.set_ylabel('Count')
        if c == 'inc':
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    for ax in axes[len(cols_and_labels):]:
        ax.axis('off')

    fig.suptitle('Distribution of Per-Satellite Trend Slopes', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_filename is not None:
        save_figure(fig, save_filename)
    plt.show()

sma_x_range = (df_all['timestamp'].min(), df_all['timestamp'].max())
sma_y_min = pd.to_numeric(df_all['sma'], errors='coerce').min()
sma_y_max = pd.to_numeric(df_all['sma'], errors='coerce').max()
sma_y_range = (float(sma_y_min), float(sma_y_max))

for col, ylabel in elements:
    plot_element_all_sats(df, y_col=col, y_label=ylabel, show_legend=False,
                          x_range=sma_x_range if col == 'sma' else None,
                          y_range=sma_y_range if col == 'sma' else None,
                          save_filename=f'{col}_vs_epoch.png')

stats_cols = [c for c, _ in stats_elements]
global_stats = compute_global_statistics(df, stats_cols)
per_satellite_stats = compute_per_satellite_statistics(df, stats_cols)
trend_slopes = compute_trend_slopes_per_day(df, ['sma', 'ecc', 'inc'])

print('\n=== Global descriptive statistics ===')
print(global_stats.round(6))

print('\n=== Per-satellite summary table shape ===')
print(per_satellite_stats.shape)

print('\n=== Trend slope summary ===')
print(trend_slopes.drop(columns=['sat_id'], errors='ignore').describe().round(6))

plot_feature_distributions(df, stats_elements, save_filename='global_distributions_sma_ecc_inc.png')
plot_per_satellite_mean_boxplots(per_satellite_stats, stats_elements,
                                 save_filename='per_satellite_means_sma_ecc_inc.png')
plot_correlation_heatmap(df, stats_elements, save_filename='feature_correlation_matrix.png')
plot_trend_slope_distributions(
    trend_slopes,
    [('sma', 'Semi-major Axis (km)'), ('ecc', 'Eccentricity'), ('inc', 'Inclination (°)')],
                               save_filename='trend_slope_distributions_sma_ecc_inc.png')