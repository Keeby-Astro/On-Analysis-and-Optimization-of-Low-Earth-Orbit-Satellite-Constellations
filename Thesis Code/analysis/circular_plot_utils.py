"""Circular plotting helpers for geometric visualizations.

These utilities make circular assumptions explicit for angular variables and
provide deterministic helpers for large catalog visualization workflows.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from scipy.special import i0

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:
    njit = None
    prange = range
    _HAS_NUMBA = False

_NUMBA_DISABLED_ENV = {"0", "false", "no", "off"}
_USE_NUMBA = _HAS_NUMBA and str(os.getenv("CIRCULAR_PLOT_USE_NUMBA", "1")).strip().lower() not in _NUMBA_DISABLED_ENV

try:
    _NUMBA_MIN_SAMPLES = int(os.getenv("CIRCULAR_PLOT_NUMBA_MIN_SAMPLES", "4096"))
except Exception:
    _NUMBA_MIN_SAMPLES = 4096

_NUMBA_MIN_SAMPLES = max(128, _NUMBA_MIN_SAMPLES)


if _HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _torus_kde_accumulate_numba(x_rad, y_rad, weights, gx, gy, kappa_x, kappa_y, norm_x, norm_y):
        out = np.zeros((gx.size, gy.size), dtype=np.float64)
        for ix in prange(gx.size):
            gxv = gx[ix]
            for iy in range(gy.size):
                gyv = gy[iy]
                acc = 0.0
                for k in range(x_rad.size):
                    kx = np.exp(kappa_x * np.cos(gxv - x_rad[k])) * norm_x
                    ky = np.exp(kappa_y * np.cos(gyv - y_rad[k])) * norm_y
                    acc += weights[k] * kx * ky
                out[ix, iy] = acc
        return out


    @njit(cache=True, parallel=True)
    def _circular_linear_kde_accumulate_numba(theta_rad, z_vals, theta_grid, z_grid, kappa, vm_norm, gauss_norm, bandwidth_linear):
        out = np.zeros((theta_grid.size, z_grid.size), dtype=np.float64)
        inv_h = 1.0 / bandwidth_linear
        for ix in prange(theta_grid.size):
            thg = theta_grid[ix]
            for iy in range(z_grid.size):
                zg = z_grid[iy]
                acc = 0.0
                for k in range(theta_rad.size):
                    ktheta = np.exp(kappa * np.cos(thg - theta_rad[k])) * vm_norm
                    dz = (zg - z_vals[k]) * inv_h
                    kz = np.exp(-0.5 * dz * dz) * gauss_norm
                    acc += ktheta * kz
                out[ix, iy] = acc
        return out

def wrap_degrees_360(values: Iterable[float]) -> np.ndarray:
    """Wrap angles in degrees to [0, 360)."""
    arr = np.asarray(values, dtype=np.float64)
    wrapped = np.mod(arr, 360.0)
    wrapped[wrapped < 0.0] += 360.0
    return wrapped

def wrap_degrees_180(values: Iterable[float]) -> np.ndarray:
    """Wrap angles in degrees to [-180, 180)."""
    arr = np.asarray(values, dtype=np.float64)
    wrapped = np.mod(arr + 180.0, 360.0) - 180.0
    return wrapped

def unwrap_degrees(values: Iterable[float]) -> np.ndarray:
    """Unwrap angular degree samples preserving continuity."""
    arr = np.asarray(values, dtype=np.float64)
    return np.rad2deg(np.unwrap(np.deg2rad(arr)))

def circular_pad_histogram_2d(hist2d: np.ndarray, pad_x: int = 1, pad_y: int = 1,
                              circular_x: bool = True, circular_y: bool = True) -> np.ndarray:
    """Pad a 2D histogram with optional circular wrapping by axis.

    The padded matrix can be filtered and then cropped to restore periodic
    continuity near angular seams.
    """
    hist = np.asarray(hist2d, dtype=np.float64)
    if hist.ndim != 2:
        raise ValueError("hist2d must be 2D")

    mode = "constant"
    if circular_x and circular_y:
        mode = "wrap"
    elif circular_x and not circular_y:
        mode = "wrap"
    elif not circular_x and circular_y:
        mode = "wrap"

    padded = np.pad(hist, ((pad_x, pad_x), (pad_y, pad_y)), mode=mode)

    # Override non-circular sides with edge values to avoid artificial drop-off.
    if circular_x and not circular_y:
        padded[:, :pad_y] = padded[:, pad_y:pad_y + 1]
        padded[:, -pad_y:] = padded[:, -pad_y - 1:-pad_y]
    if circular_y and not circular_x:
        padded[:pad_x, :] = padded[pad_x:pad_x + 1, :]
        padded[-pad_x:, :] = padded[-pad_x - 1:-pad_x, :]

    return padded

def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    arr = np.asarray(centers, dtype=np.float64)
    if arr.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    if arr.size == 1:
        c = float(arr[0])
        return np.array([c - 0.5, c + 0.5], dtype=np.float64)

    mids = 0.5 * (arr[:-1] + arr[1:])
    left = arr[0] - (mids[0] - arr[0])
    right = arr[-1] + (arr[-1] - mids[-1])
    return np.concatenate(([left], mids, [right]))


def circular_linear_density(circular_degrees: Iterable[float], linear_values: Iterable[float],
                            circular_bins: int = 72, linear_bins: int = 72,
                            sigma: float = 0.0, kappa: float = 25.0,
                            bandwidth_linear: float | None = None) -> Dict[str, Any]:
    """Directional-linear density using von-Mises x Gaussian kernels.

    Returns edge-based payloads for imshow compatibility with existing plotting
    code paths.
    """
    payload = circular_linear_kde(
        circular_degrees,
        linear_values,
        circular_bins=int(circular_bins),
        linear_bins=int(linear_bins),
        kappa=float(kappa),
        bandwidth_linear=bandwidth_linear,
    )

    density = np.asarray(payload["density"], dtype=np.float64)
    sigma_val = float(max(0.0, sigma))
    if sigma_val > 0.0:
        # Smooth only along the linear axis by default; circular axis is already kernelized.
        density = gaussian_filter(density, sigma=(0.0, sigma_val))

    theta_grid = np.asarray(payload["theta_grid_deg"], dtype=np.float64)
    z_grid = np.asarray(payload["z_grid"], dtype=np.float64)
    xedges = np.linspace(0.0, 360.0, int(circular_bins) + 1)
    yedges = _centers_to_edges(z_grid)

    return {
        "density": density,
        "xedges": xedges,
        "yedges": yedges,
        "theta_grid_deg": theta_grid,
        "z_grid": z_grid,
        "is_placeholder": False,
        "method": "circular_linear_kde",
        "kappa": float(payload.get("kappa", kappa)),
        "bandwidth_linear": float(payload.get("bandwidth_linear", np.nan)),
    }


def circular_linear_density_placeholder(circular_degrees: Iterable[float], linear_values: Iterable[float],
                                        circular_bins: int = 72, linear_bins: int = 72,
                                        sigma: float = 1.0) -> Dict[str, Any]:
    """Backward-compatible alias for directional-linear KDE density."""
    return circular_linear_density(
        circular_degrees,
        linear_values,
        circular_bins=circular_bins,
        linear_bins=linear_bins,
        sigma=sigma,
    )


def torus_kde_von_mises(
    x_degrees: Iterable[float],
    y_degrees: Iterable[float],
    bins_x: int = 96,
    bins_y: int = 96,
    kappa_x: float = 25.0,
    kappa_y: float = 25.0,
    chunk_size: int = 4096,
    weights: Iterable[float] | None = None,
) -> Dict[str, Any]:
    """Estimate density on a torus using product von-Mises kernels.

    Scientific notes:
    - This treats (x, y) as coordinates on S1 x S1.
    - Kernel form follows directional-statistics conventions for circular data.
    """
    x = wrap_degrees_360(x_degrees)
    y = wrap_degrees_360(y_degrees)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return {
            "density": np.zeros((bins_x, bins_y), dtype=np.float64),
            "xgrid_deg": np.linspace(0.0, 360.0, bins_x, endpoint=False),
            "ygrid_deg": np.linspace(0.0, 360.0, bins_y, endpoint=False),
            "method": "torus_kde_von_mises",
            "sample_count": 0,
        }

    if weights is None:
        w = np.ones(x.size, dtype=np.float64)
    else:
        w_raw = np.asarray(weights, dtype=np.float64)
        if w_raw.shape != np.asarray(x_degrees).shape:
            raise ValueError("weights must match original input shape")
        w = w_raw[mask]

    x_rad = np.deg2rad(x)
    y_rad = np.deg2rad(y)
    gx = np.linspace(0.0, 2.0 * np.pi, bins_x, endpoint=False)
    gy = np.linspace(0.0, 2.0 * np.pi, bins_y, endpoint=False)

    norm_x = 1.0 / (2.0 * np.pi * i0(float(kappa_x)))
    norm_y = 1.0 / (2.0 * np.pi * i0(float(kappa_y)))
    use_numba = (
        _USE_NUMBA
        and x_rad.size >= _NUMBA_MIN_SAMPLES
        and bins_x <= 192
        and bins_y <= 192
    )

    if use_numba:
        density = _torus_kde_accumulate_numba(
            x_rad,
            y_rad,
            w,
            gx,
            gy,
            float(kappa_x),
            float(kappa_y),
            float(norm_x),
            float(norm_y),
        )
    else:
        density = np.zeros((bins_x, bins_y), dtype=np.float64)
        chunk = int(max(1, chunk_size))
        for start in range(0, x_rad.size, chunk):
            end = min(start + chunk, x_rad.size)
            xr = x_rad[start:end]
            yr = y_rad[start:end]
            ww = w[start:end]

            kx = np.exp(float(kappa_x) * np.cos(gx[:, None] - xr[None, :])) * norm_x
            ky = np.exp(float(kappa_y) * np.cos(gy[:, None] - yr[None, :])) * norm_y
            # Weighted separable accumulation on S1 x S1.
            density += (kx * ww[None, :]) @ ky.T

    total_w = float(np.sum(w))
    if total_w > 0.0:
        density /= total_w

    return {
        "density": density,
        "xgrid_deg": np.rad2deg(gx),
        "ygrid_deg": np.rad2deg(gy),
        "method": "torus_kde_von_mises",
        "sample_count": int(x.size),
        "kappa_x": float(kappa_x),
        "kappa_y": float(kappa_y),
    }


def circular_linear_kde(
    circular_degrees: Iterable[float],
    linear_values: Iterable[float],
    circular_bins: int = 96,
    linear_bins: int = 96,
    kappa: float = 25.0,
    bandwidth_linear: float | None = None,
    chunk_size: int = 4096,
) -> Dict[str, Any]:
    """Directional-linear KDE using von-Mises (circular) x Gaussian (linear) kernels."""
    theta = wrap_degrees_360(circular_degrees)
    z = np.asarray(linear_values, dtype=np.float64)
    mask = np.isfinite(theta) & np.isfinite(z)
    theta = theta[mask]
    z = z[mask]

    if theta.size == 0:
        return {
            "density": np.zeros((circular_bins, linear_bins), dtype=np.float64),
            "theta_grid_deg": np.linspace(0.0, 360.0, circular_bins, endpoint=False),
            "z_grid": np.linspace(0.0, 1.0, linear_bins),
            "method": "circular_linear_kde",
            "sample_count": 0,
        }

    theta_rad = np.deg2rad(theta)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, circular_bins, endpoint=False)

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max <= z_min:
        z_max = z_min + 1.0
    z_grid = np.linspace(z_min, z_max, linear_bins)

    if bandwidth_linear is None:
        sigma = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
        h = 1.06 * sigma * (z.size ** (-1.0 / 5.0))
        bandwidth_linear = float(max(h, 1.0e-6))
    else:
        bandwidth_linear = float(max(bandwidth_linear, 1.0e-6))

    vm_norm = 1.0 / (2.0 * np.pi * i0(float(kappa)))
    gauss_norm = 1.0 / (np.sqrt(2.0 * np.pi) * bandwidth_linear)

    use_numba = (
        _USE_NUMBA
        and theta_rad.size >= _NUMBA_MIN_SAMPLES
        and circular_bins <= 192
        and linear_bins <= 192
    )

    if use_numba:
        density = _circular_linear_kde_accumulate_numba(
            theta_rad,
            z,
            theta_grid,
            z_grid,
            float(kappa),
            float(vm_norm),
            float(gauss_norm),
            float(bandwidth_linear),
        )
    else:
        density = np.zeros((circular_bins, linear_bins), dtype=np.float64)
        chunk = int(max(1, chunk_size))
        for start in range(0, theta_rad.size, chunk):
            end = min(start + chunk, theta_rad.size)
            th = theta_rad[start:end]
            zz = z[start:end]

            ktheta = np.exp(float(kappa) * np.cos(theta_grid[:, None] - th[None, :])) * vm_norm
            kz = np.exp(-0.5 * ((z_grid[:, None] - zz[None, :]) / bandwidth_linear) ** 2) * gauss_norm
            density += ktheta @ kz.T

    density /= float(theta.size)

    return {
        "density": density,
        "theta_grid_deg": np.rad2deg(theta_grid),
        "z_grid": z_grid,
        "method": "circular_linear_kde",
        "sample_count": int(theta.size),
        "kappa": float(kappa),
        "bandwidth_linear": float(bandwidth_linear),
    }


def stable_category_color_map(categories: Iterable[Any], cmap_name: str = "tab20") -> Dict[str, Any]:
    """Build deterministic category-to-color mapping stable across updates."""
    arr = np.asarray(list(categories), dtype=object)
    labels = np.array(["__nan__" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x) for x in arr], dtype=object)
    unique_sorted = sorted(set(labels.tolist()))

    cmap = colormaps.get_cmap(cmap_name).resampled(max(len(unique_sorted), 1))
    mapping = {label: idx for idx, label in enumerate(unique_sorted)}
    colors = [cmap(i) for i in range(max(len(unique_sorted), 1))]

    codes = np.array([mapping[label] for label in labels], dtype=np.int64)

    return {
        "codes": codes,
        "mapping": mapping,
        "labels": unique_sorted,
        "cmap": ListedColormap(colors),
    }


def angular_axis_ticks(mode: str = "360") -> Tuple[np.ndarray, list[str]]:
    """Return standard angular ticks and labels for wrapped axes."""
    if str(mode) == "180":
        ticks = np.array([-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0])
    else:
        ticks = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0])
    labels = [f"{int(v)}" for v in ticks]
    return ticks, labels


def duplicate_torus_points_for_display(
    x_degrees: Iterable[float],
    y_degrees: Iterable[float],
    margin_deg: float = 10.0,
    wrap_mode: str = "360",
) -> Dict[str, np.ndarray]:
    """Duplicate points near seams for torus-style scatter rendering.

    Returns original+duplicated arrays and source indices identifying which
    original point each rendered sample came from.
    """
    margin = float(max(margin_deg, 0.0))
    if str(wrap_mode) == "180":
        x = wrap_degrees_180(x_degrees)
        y = wrap_degrees_180(y_degrees)
        lo, hi, period = -180.0, 180.0, 360.0
    else:
        x = wrap_degrees_360(x_degrees)
        y = wrap_degrees_360(y_degrees)
        lo, hi, period = 0.0, 360.0, 360.0

    n = x.size
    source = np.arange(n, dtype=np.int64)
    x_all = [x]
    y_all = [y]
    s_all = [source]

    near_lo_x = x < (lo + margin)
    near_hi_x = x > (hi - margin)
    near_lo_y = y < (lo + margin)
    near_hi_y = y > (hi - margin)

    if np.any(near_lo_x):
        x_all.append(x[near_lo_x] + period)
        y_all.append(y[near_lo_x])
        s_all.append(source[near_lo_x])
    if np.any(near_hi_x):
        x_all.append(x[near_hi_x] - period)
        y_all.append(y[near_hi_x])
        s_all.append(source[near_hi_x])
    if np.any(near_lo_y):
        x_all.append(x[near_lo_y])
        y_all.append(y[near_lo_y] + period)
        s_all.append(source[near_lo_y])
    if np.any(near_hi_y):
        x_all.append(x[near_hi_y])
        y_all.append(y[near_hi_y] - period)
        s_all.append(source[near_hi_y])

    return {
        "x": np.concatenate(x_all),
        "y": np.concatenate(y_all),
        "source_index": np.concatenate(s_all),
    }
