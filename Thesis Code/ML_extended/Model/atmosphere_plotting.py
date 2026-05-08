import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pymsis

# SETTINGS
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
                     'lines.markersize': 8, 'axes.spines.top': True, 'axes.spines.right': True})

# Color palette for plots
colors = ['#1965B0', '#E8601C', '#4EB265', '#72190E', '#882E72',
          '#437DBF', '#F1932D', '#90C987', '#A5170E', '#994F88',
          '#6195CF', '#F6C141', '#CAE0AB', '#DC050C', '#AA6F9E',
          '#7BAFDE', '#F7F056', '#8B8B8B', '#896D67', '#BA8DB4']

# Load forecast data (CSV files are two directories up from this script)
script_dir = Path(__file__).parent
data_dir = script_dir.parent.parent

solar_flux_df = pd.read_csv(data_dir / "solar_flux_forecast_only.csv")
solar_flux_df["date"] = pd.to_datetime(solar_flux_df["date"])

ap_forecast_df = pd.read_csv(data_dir / "ap_conditional_forecast.csv")
ap_forecast_df["date"] = pd.to_datetime(ap_forecast_df["date"])

# Calculate 81 day rolling average for F10.7a
solar_flux_df["f107a"] = solar_flux_df["Obs"].rolling(window=81, min_periods=1).mean()

# Merge the dataframes on date
full_forecast_data = pd.merge(solar_flux_df, ap_forecast_df, on="date", how="inner")

# Define target dates
target_dates = pd.to_datetime(["2025-01-01", "2025-06-15", "2026-01-01", "2026-06-15", 
                               "2027-01-01", "2027-06-15", "2028-01-01", "2028-06-15",
                               "2029-01-01"])

# Filter for target dates
forecast_data = full_forecast_data[full_forecast_data["date"].isin(target_dates)].copy()
forecast_data = forecast_data.sort_values("date")

if forecast_data.empty:
    raise RuntimeError("No data found for the specified target dates.")

print(f"Processing {len(forecast_data)} target dates.")

# Define global grid (centers)
lons = np.arange(-180, 181, 1)  # degrees
lats = np.arange(-90, 91, 1)    # degrees
alt = 200                       # km

# Precompute grid cell edges for pcolormesh
lon_edges = np.linspace(lons[0] - 2.5, lons[-1] + 2.5, len(lons) + 1)
lat_edges = np.linspace(lats[0] - 2.5, lats[-1] + 2.5, len(lats) + 1)

dates = forecast_data["date"].values
f107s = forecast_data["Obs"].values
f107as = forecast_data["f107a"].values
aps = forecast_data["ap_mean"].values
# MSIS Ap input is a 7 element vector
aps_input = [[ap] * 7 for ap in aps]

# Run MSIS for global grid on target dates
print("Running pymsis.calculate on global grid for target dates...")
output = pymsis.calculate(dates, lons, lats, alt, f107s, f107as, aps_input)
# output has shape (ndates, nlons, nlats, 1, 11) for a single altitude
output = np.squeeze(output)  # -> (ndates, nlons, nlats, 11)
print(f"Calculation complete. Output shape: {output.shape}")

# Extract total mass density (kg/m^3)
output_mass = output[:, :, :, pymsis.Variable.MASS_DENSITY]  # shape (ndates, nlons, nlats)

# Global statistics for consistent color normalization
vmin, vmax = np.percentile(output_mass, [5, 95])
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

# Loop to create and save figures
for i, date in enumerate(dates):
    dt = pd.Timestamp(date)
    date_str = dt.strftime("%Y-%m-%d")
    
    # Determine interval for time series
    if dt.month == 1:
        interval_end = pd.Timestamp(year=dt.year, month=6, day=14)
    else:
        interval_end = pd.Timestamp(year=dt.year, month=12, day=31)
        
    interval_mask = (full_forecast_data["date"] >= dt) & (full_forecast_data["date"] <= interval_end)
    interval_data = full_forecast_data[interval_mask].sort_values("date")
    
    if interval_data.empty:
        print(f"Warning: No data for interval starting {date_str}")
        continue

    ts_dates = interval_data["date"].values
    ts_f107s = interval_data["Obs"].values
    ts_f107as = interval_data["f107a"].values
    ts_aps = interval_data["ap_mean"].values
    ts_aps_input = [[ap] * 7 for ap in ts_aps]
    
    # Calculate daily time series for this interval at (0, 0)
    ts_lons = np.array([0])
    ts_lats = np.array([0])
    ts_output = pymsis.calculate(ts_dates, ts_lons, ts_lats, alt, ts_f107s, ts_f107as, ts_aps_input)
    ts_output = np.squeeze(ts_output)
    
    if ts_output.ndim == 1:
        ts_mass_density = np.array([ts_output[pymsis.Variable.MASS_DENSITY]])
    else:
        ts_mass_density = ts_output[:, pymsis.Variable.MASS_DENSITY]

    # Create figure with subplots
    fig = plt.figure(figsize=(10, 8))
    # Make top subplot more compact (1:5 ratio)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 5])
    
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    
    # Plot Time Series
    ax_ts.plot(ts_dates, ts_mass_density, color='k', linewidth=1)
    ax_ts.set_xlim(ts_dates[0], ts_dates[-1])
    ax_ts.set_ylabel("Density (kg/m$^3$)")
    ax_ts.set_title(f"Daily Total Mass Density - {date_str} to {pd.Timestamp(ts_dates[-1]).strftime('%Y-%m-%d')}")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # Lower x-axis labels to avoid overlap
    ax_ts.tick_params(axis='x', which='major', pad=10)

    gs.xlabel_style = {'size': 12}
    gs.ylabel_style = {'size': 12}

    # Add colorbar for the map at the bottom
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.035])  # [left, bottom, width, height]
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
                        cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Total Mass Density (kg/m$^3$)")
    
    # Plot Map
    ax_map.set_global()
    ax_map.coastlines()
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax_map.add_feature(cfeature.LAND, facecolor="lightgray")
    ax_map.add_feature(cfeature.OCEAN, facecolor="white")
    
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    data = output_mass[i].T
    mesh = ax_map.pcolormesh(lon_edges, lat_edges, data, cmap="viridis",
                             norm=norm, shading="auto", transform=ccrs.PlateCarree())

    # Make the axis numbers bigger for readability for just the map
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    
    filename = f"msis_density_{date_str}.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close(fig)