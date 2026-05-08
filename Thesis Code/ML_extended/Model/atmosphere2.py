import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
import subprocess
import shutil
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pymsis


# =========================
# SETTINGS
# =========================

# Animation constraints
FPS = 30
DURATION_SECONDS = 60
NFRAMES = FPS * DURATION_SECONDS  # 1800 frames
STEP_PER_FRAME = pd.Timedelta(hours=1)
TOTAL_SPAN = (NFRAMES - 1) * STEP_PER_FRAME

START_DATE = pd.Timestamp("2026-01-01 00:00:00")

# MSIS / grid settings
ALT_KM = 200
GRID_DEG = 1
LON0, LAT0 = 0.0, 0.0

# Precompute in chunks for MSIS calls (compute-time only, not rendering)
MSIS_CHUNK = 96  # hours per MSIS call for global grid; tune 48-240 based on RAM

# Color scaling (log10 density)
PERCENTILES_FOR_NORM = (2, 98)

# Output files
OUT_MP4 = "msis_total_mass_density_hourly_75days_30fps_60s_480dpi.mp4"
OUT_GIF = "msis_total_mass_density_hourly_75days_30fps_60s_480dpi.gif"

# Output resolution control
# Aim for ~1280 px width for presentations: width_px ≈ figsize_in * dpi.
SAVE_DPI = 480

# Optional: downscale GIF output width (reduces file size and conversion time)
GIF_WIDTH = 1600  # pixels; keep aspect ratio (-1). Increase for sharper GIFs.


plt.rcParams.update({
    "figure.figsize": (10.74, 5.75),
    "xtick.direction": "in", "xtick.labelsize": 10, "xtick.major.size": 3,
    "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.5,
    "xtick.minor.visible": True, "xtick.top": False,
    "ytick.direction": "in", "ytick.labelsize": 10, "ytick.major.size": 3,
    "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.5,
    "ytick.minor.visible": True, "ytick.right": False,
    "axes.linewidth": 0.5, "grid.linewidth": 0.5, "lines.linewidth": 1.25,
    "legend.fontsize": 10, "legend.frameon": False,
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 10, "axes.titlesize": 12,
    "axes.grid": False, "grid.linestyle": "--", "grid.color": "0.5",
})


# =========================
# HELPERS
# =========================

def resolve_data_dir() -> Path:
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    return script_dir.parent.parent


def build_hourly_drivers(full_daily: pd.DataFrame, start: pd.Timestamp, nframes: int) -> pd.DataFrame:
    end = start + (nframes - 1) * STEP_PER_FRAME
    hourly_index = pd.date_range(start=start, end=end, freq="h")
    daily = full_daily.set_index("date").sort_index()
    hourly = daily.reindex(hourly_index, method="ffill")

    if hourly.isna().any().any():
        hourly = hourly.bfill()

    required = ["Obs", "f107a", "ap_mean"]
    missing = [c for c in required if c not in hourly.columns]
    if missing:
        raise RuntimeError(f"Missing required columns after reindexing: {missing}")

    hourly = hourly[required].copy()
    hourly.index.name = "date"
    return hourly


def make_ap7(ap_series: np.ndarray) -> list:
    return [[float(ap)] * 7 for ap in ap_series]


def msis_mass_density_log10(dates64, lons, lats, alt_km, f107s, f107as, ap7_list) -> np.ndarray:
    out = pymsis.calculate(dates64, lons, lats, alt_km, f107s, f107as, ap7_list)
    out = np.squeeze(out)

    if out.ndim == 3:
        rho = out[..., pymsis.Variable.MASS_DENSITY]  # (nlons, nlats)
        rho = np.transpose(rho, (1, 0))               # (nlats, nlons)
        return np.log10(rho)
    elif out.ndim == 4:
        rho = out[..., pymsis.Variable.MASS_DENSITY]  # (ndates, nlons, nlats)
        rho = np.transpose(rho, (0, 2, 1))            # (ndates, nlats, nlons)
        return np.log10(rho)
    else:
        raise RuntimeError(f"Unexpected pymsis output ndim={out.ndim}, shape={out.shape}")


def ensure_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg and ensure 'ffmpeg' is available in your terminal."
        )


# =========================
# LOAD INPUTS
# =========================

data_dir = resolve_data_dir()

solar_flux_df = pd.read_csv(data_dir / "solar_flux_forecast_only.csv")
solar_flux_df["date"] = pd.to_datetime(solar_flux_df["date"])

ap_forecast_df = pd.read_csv(data_dir / "ap_conditional_forecast.csv")
ap_forecast_df["date"] = pd.to_datetime(ap_forecast_df["date"])

solar_flux_df = solar_flux_df.sort_values("date").copy()
solar_flux_df["f107a"] = solar_flux_df["Obs"].rolling(window=81, min_periods=1).mean()

full_forecast_data = pd.merge(solar_flux_df, ap_forecast_df, on="date", how="inner").sort_values("date")

drivers_hourly = build_hourly_drivers(full_forecast_data, START_DATE, NFRAMES)

dates_hourly = drivers_hourly.index.to_numpy(dtype="datetime64[ns]")
f107s = drivers_hourly["Obs"].to_numpy(dtype=float)
f107as = drivers_hourly["f107a"].to_numpy(dtype=float)
aps = drivers_hourly["ap_mean"].to_numpy(dtype=float)
aps_input = make_ap7(aps)

print(f"Hourly frames: {NFRAMES}")
print(f"Time span: {drivers_hourly.index[0]} to {drivers_hourly.index[-1]} ({TOTAL_SPAN})")


# =========================
# GRID
# =========================

lons = np.arange(-180, 180 + GRID_DEG, GRID_DEG, dtype=float)
lats = np.arange(-90, 90 + GRID_DEG, GRID_DEG, dtype=float)
nlons, nlats = len(lons), len(lats)
extent = [-180, 180, -90, 90]


# =========================
# PRECOMPUTE GLOBAL DENSITY CUBE (no MSIS calls during saving)
# =========================

print("Precomputing global log10 density cube...")
logrho_all = np.empty((NFRAMES, nlats, nlons), dtype=np.float32)

for start in range(0, NFRAMES, MSIS_CHUNK):
    end = min(start + MSIS_CHUNK, NFRAMES)
    d = dates_hourly[start:end]
    f = f107s[start:end]
    fa = f107as[start:end]
    ap7 = aps_input[start:end]
    logrho_chunk = msis_mass_density_log10(d, lons, lats, ALT_KM, f, fa, ap7).astype(np.float32)
    logrho_all[start:end, :, :] = logrho_chunk
    print(f"  MSIS computed frames {start}..{end-1}")

vmin, vmax = np.nanpercentile(logrho_all, PERCENTILES_FOR_NORM)
print(f"Color norm (log10 kg/m^3): vmin={vmin:.3f}, vmax={vmax:.3f}")


# =========================
# PRECOMPUTE TIME SERIES (single point)
# =========================

print("Computing hourly time series at (lon, lat) = (0, 0)...")
ts_lons = np.array([LON0], dtype=float)
ts_lats = np.array([LAT0], dtype=float)

ts_out = pymsis.calculate(dates_hourly, ts_lons, ts_lats, ALT_KM, f107s, f107as, aps_input)
ts_out = np.squeeze(ts_out)
ts_rho = ts_out[:, pymsis.Variable.MASS_DENSITY]
ts_logrho = np.log10(ts_rho)


# =========================
# FIGURE SETUP
# =========================

fig = plt.figure(figsize=(10.74, 5.75))
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 5], hspace=0.38)

ax_ts = fig.add_subplot(gs[0, 0])
ax_map = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())

ax_ts.plot(drivers_hourly.index, ts_logrho, color="k", linewidth=1.0)
ax_ts.set_xlim(drivers_hourly.index[0], drivers_hourly.index[-1])
ax_ts.set_ylabel(r"$\log_{10}(\rho)$")
ax_ts.grid(True, alpha=0.25)
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax_ts.tick_params(axis="x", which="major", pad=10)
cursor_line = ax_ts.axvline(drivers_hourly.index[0], color="k", linewidth=1.0, alpha=0.8)

ax_map.set_global()
ax_map.add_feature(cfeature.LAND, facecolor="0.9", zorder=0)
ax_map.add_feature(cfeature.OCEAN, facecolor="1.0", zorder=0)
ax_map.coastlines(linewidth=0.5, zorder=2)

img = ax_map.imshow(
    logrho_all[0],
    origin="lower",
    extent=extent,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
    zorder=1,
)

title_text = ax_map.set_title(
    f"NRLMSIS Total Mass Density at {ALT_KM} km\n"
    f"{pd.Timestamp(dates_hourly[0]).strftime('%Y-%m-%d %H:%M')}",
    pad=2,
)

cbar = fig.colorbar(img, ax=ax_map, orientation="horizontal", pad=0.085, fraction=0.05)
cbar.set_label(r"$\log_{10}(\rho)$ (kg/m$^3$)")


# =========================
# ANIMATION UPDATE
# =========================

def update(frame_idx: int):
    img.set_data(logrho_all[frame_idx])

    t = pd.Timestamp(dates_hourly[frame_idx])
    cursor_line.set_xdata([t, t])
    title_text.set_text(
        f"NRLMSIS Total Mass Density at {ALT_KM} km\n"
        f"{t.strftime('%Y-%m-%d %H:%M')}"
    )
    return (img, cursor_line, title_text)


ani = FuncAnimation(
    fig,
    update,
    frames=range(NFRAMES),
    interval=1000 / FPS,
    blit=False,
    repeat=False,
    cache_frame_data=False,
)


# =========================
# SAVE MP4 (streaming) WITH PROGRESS CALLBACK
# =========================

def progress_cb(i, n):
    # i is 0-indexed current frame
    if (i % 60) == 0:  # every 2 seconds of video at 30 fps
        if n is None:
            print(f"Saving frame {i} ...")
        else:
            print(f"Saving frame {i}/{n}")

print(f"Saving MP4: {OUT_MP4} (FPS={FPS}, frames={NFRAMES}, dpi={SAVE_DPI})")

writer = FFMpegWriter(
    fps=FPS,
    codec="libx264",
    # libx264 requires even pixel dimensions; 480 DPI can yield an odd width.
    extra_args=[
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
    ],
)

ani.save(
    OUT_MP4,
    writer=writer,
    dpi=SAVE_DPI,
    progress_callback=progress_cb,  # supported by Animation.save :contentReference[oaicite:4]{index=4}
)

plt.close(fig)
print("MP4 complete.")


# =========================
# CONVERT MP4 -> GIF (high quality palette workflow)
# =========================

print("Converting MP4 to GIF via ffmpeg (palettegen/paletteuse)...")
ensure_ffmpeg_available()

vf = (
    f"fps={FPS},scale={GIF_WIDTH}:-1:flags=lanczos,"
    "split[s0][s1];"
    "[s0]palettegen=stats_mode=diff[p];"
    "[s1][p]paletteuse=dither=sierra2_4a"
)

cmd = ["ffmpeg", "-y", "-i", OUT_MP4, "-vf", vf, "-loop", "0", OUT_GIF]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

print(f"GIF complete: {OUT_GIF}")