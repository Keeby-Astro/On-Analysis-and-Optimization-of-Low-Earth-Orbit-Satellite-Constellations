import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import pymsis

# Plotting settings
plt.rcParams.update({'figure.figsize': (10.74, 5.175),
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
                     'savefig.transparent': True})


def _get_single_day_inputs(target_date):
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent

    solar_flux_df = pd.read_csv(data_dir / 'solar_flux_forecast_only.csv')
    solar_flux_df['date'] = pd.to_datetime(solar_flux_df['date'])

    ap_forecast_df = pd.read_csv(data_dir / 'ap_conditional_forecast.csv')
    ap_forecast_df['date'] = pd.to_datetime(ap_forecast_df['date'])

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


def main():
    target_date = pd.Timestamp('2026-02-26')
    altitude_km = 200

    lons = np.arange(-180, 181, 1)
    lats = np.arange(-90, 91, 1)
    lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='xy')

    f107, f107a, ap = _get_single_day_inputs(target_date)
    aps_input = [[ap] * 7]

    print(f'Running pymsis for {target_date.date()} at {altitude_km} km...')
    output = pymsis.calculate(
        [target_date.to_pydatetime()],
        lons,
        lats,
        altitude_km,
        [f107],
        [f107a],
        aps_input,
    )

    output = np.squeeze(output)
    density = output[:, :, pymsis.Variable.MASS_DENSITY].T

    density_safe = np.nan_to_num(density, nan=np.nanmedian(density))
    dmin = np.percentile(density_safe, 5)
    dmax = np.percentile(density_safe, 95)
    norm = matplotlib.colors.Normalize(vmin=dmin, vmax=dmax)
    cmap = plt.get_cmap('viridis')

    radius = 1.0
    lon_rad = np.radians(lon_grid)
    lat_rad = np.radians(lat_grid)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    facecolors = cmap(norm(density_safe))
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=45)
    ax.set_axis_off()

    output_file = f'msis_density_3d_{target_date.strftime("%Y-%m-%d")}.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f'Saved {output_file}')


if __name__ == '__main__':
    main()
