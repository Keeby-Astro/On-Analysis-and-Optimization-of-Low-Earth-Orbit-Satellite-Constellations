from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from pathlib import Path

import pymsis

# ------------------------------
# Plotting settings
# ------------------------------
plt.rcParams.update({
    'figure.figsize': (10.74, 5.175),
    'xtick.direction': 'in', 'xtick.labelsize': 10, 'xtick.major.size': 3,
    'xtick.major.width': 0.5, 'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5,
    'xtick.minor.visible': True, 'xtick.top': False,
    'ytick.direction': 'in', 'ytick.labelsize': 10, 'ytick.major.size': 3,
    'ytick.major.width': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
    'ytick.minor.visible': True, 'ytick.right': False,
    'axes.linewidth': 0.5, 'grid.linewidth': 0.5, 'lines.linewidth': 1.5,
    'legend.fontsize': 10, 'legend.frameon': False,
    'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'dejavuserif',
    'font.size': 8, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'axes.grid': False, 'grid.linestyle': '--', 'grid.color': '0.5',
    'lines.markersize': 8, 'axes.spines.top': True, 'axes.spines.right': True,
    'savefig.transparent': True,
})

# Space-weather inputs
def _get_single_day_inputs(target_date: pd.Timestamp) -> tuple[float, float, float]:
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent  # keep your existing convention

    solar_flux_df = pd.read_csv(data_dir / 'solar_flux_forecast_only.csv')
    solar_flux_df['date'] = pd.to_datetime(solar_flux_df['date'])

    ap_forecast_df = pd.read_csv(data_dir / 'ap_conditional_forecast.csv')
    ap_forecast_df['date'] = pd.to_datetime(ap_forecast_df['date'])

    # 81-day running mean for f107a (proxy consistent with common MSIS usage)
    solar_flux_df['f107a'] = solar_flux_df['Obs'].rolling(window=81, min_periods=1).mean()

    full_forecast_data = pd.merge(solar_flux_df, ap_forecast_df, on='date', how='inner')

    row = full_forecast_data.loc[full_forecast_data['date'] == target_date]
    if row.empty:
        raise RuntimeError(f'No data found for target date: {target_date.date()}')

    row = row.iloc[0]
    f107 = float(row['Obs'])
    f107a = float(row['f107a'])
    ap = float(row['ap_mean'])
    return f107, f107a, ap

# Texture helpers (equirectangular)
def _load_texture_rgb(texture_path: Path) -> np.ndarray:
    img = mpimg.imread(str(texture_path))

    # mpimg.imread returns float in [0,1] for many formats, uint8 for some.
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0

    # Drop alpha if present
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image; got shape={img.shape} from {texture_path}")

    return img

def _sample_equirectangular_rgb(img_rgb: np.ndarray, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    H, W, _ = img_rgb.shape

    # u in [0,1), v in [0,1]
    u = (lon_deg + 180.0) / 360.0
    u = u % 1.0
    v = (90.0 - lat_deg) / 180.0
    v = np.clip(v, 0.0, 1.0)

    x = u * (W - 1)
    y = v * (H - 1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0 + 1) % W
    y1 = np.clip(y0 + 1, 0, H - 1)

    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]

    c00 = img_rgb[y0, x0]
    c10 = img_rgb[y0, x1]
    c01 = img_rgb[y1, x0]
    c11 = img_rgb[y1, x1]

    c0 = (1.0 - wx) * c00 + wx * c10
    c1 = (1.0 - wx) * c01 + wx * c11
    c = (1.0 - wy) * c0 + wy * c1
    return c

def _find_texture_path() -> Path:
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent

    candidates = [script_dir / "earth_texture_4k.jpg", data_dir / "earth_texture_4k.jpg",
                  data_dir / "earth_texture_equirectangular.jpg"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find Earth texture. Place 'earth_texture_4k.jpg' next to this script or in your data_dir.")

# Main
def main() -> None:
    target_date = pd.Timestamp('2026-02-26')
    altitude_km = 200

    # 1-degree global grid
    lons = np.arange(-180, 181, 1)
    lats = np.arange(-90, 91, 1)
    lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='xy')  # shape (len(lats), len(lons))

    f107, f107a, ap = _get_single_day_inputs(target_date)
    aps_input = [[ap] * 7]  # pymsis expects 7-element Ap history per time

    print(f'Running pymsis for {target_date.date()} at {altitude_km} km...')
    output = pymsis.calculate([target_date.to_pydatetime()], lons, lats, altitude_km,
                              [f107], [f107a], aps_input)

    output = np.squeeze(output)  # remove time axis
    # Keep your existing orientation choice; if you see a 90-degree rotation, remove .T
    density = output[:, :, pymsis.Variable.MASS_DENSITY].T  # (lat, lon)

    # Robustify NaNs
    density_safe = np.nan_to_num(density, nan=np.nanmedian(density))

    # Density colormap + alpha (log-scaled)
    # Use percentiles to avoid outliers dominating normalization
    dmin = np.percentile(density_safe, 5)
    dmax = np.percentile(density_safe, 95)
    norm = matplotlib.colors.Normalize(vmin=dmin, vmax=dmax)
    cmap = plt.get_cmap('plasma')

    # Alpha from log10(density), typically improves contrast for MSIS dynamic range
    logd = np.log10(np.maximum(density_safe, 1e-30))
    lo = np.percentile(logd, 5)
    hi = np.percentile(logd, 95)
    alpha = np.clip((logd - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    # Optional shaping: gamma < 1 increases opacity of lower values; gamma > 1 reduces it
    alpha_gamma = 0.8
    alpha = alpha ** alpha_gamma

    # Overall transparency scale so the texture remains visible
    alpha_scale = 0.75
    alpha = alpha_scale * alpha

    density_rgba = cmap(norm(density_safe))
    density_rgba[..., 3] = alpha

    # Sphere geometry
    radius = 1.0
    lon_rad = np.radians(lon_grid)
    lat_rad = np.radians(lat_grid)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    # Earth texture (equirectangular)
    texture_path = _find_texture_path()
    tex_img = _load_texture_rgb(texture_path)
    tex_rgb = _sample_equirectangular_rgb(tex_img, lon_grid, lat_grid)
    tex_rgba = np.dstack([tex_rgb, np.ones(tex_rgb.shape[:2], dtype=tex_rgb.dtype)])

    # Plot
    fig = plt.figure(figsize=(12, 12))
    elev = 25
    azims = [45, 135, 225, 315]
    shell = 1.002

    for idx, azim in enumerate(azims, start=1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')

        # Base textured Earth (opaque)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=tex_rgba,
                        linewidth=0, edgecolor='none', antialiased=True, shade=False)

        # Density overlay shell (slightly larger radius to prevent z-fighting)
        ax.plot_surface(shell * x, shell * y, shell * z, rstride=1, cstride=1, facecolors=density_rgba,
                        linewidth=0, edgecolor='none', antialiased=True, shade=False)

        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

    output_file = f'msis_density_3d_textured_{target_date.strftime("%Y-%m-%d")}.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f'Saved {output_file}')

if __name__ == '__main__':
    main()