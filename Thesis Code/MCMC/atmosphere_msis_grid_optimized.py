"""
Numba-accelerated density lookup + drag using precomputed NRLMSIS daily grids.
"""

from __future__ import annotations

import numpy as np
from numba import njit

@njit(cache=True, inline='always')
def _wrap_lon_deg(lon_deg: float) -> float:
    return lon_deg % 360.0

@njit(cache=True, inline='always')
def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo: return lo
    if x > hi: return hi
    return x

@njit(cache=True, nogil=True)
def _interp_trilinear_dual(grid0: np.ndarray, grid1: np.ndarray, mix: float, lat_deg: float, lon_deg: float,
                           alt_km: float, lat0: float, dlat: float, lon0: float, dlon: float,
                           alt_min_km: float, alt_step_km: float) -> float:
    # Trilinear interpolation on two grids simultaneously  
    nlat, nlon, nz = grid0.shape

    # Clamp lat
    lat_max = lat0 + dlat * (nlat - 1)
    if lat_deg < lat0: lat_deg = lat0
    elif lat_deg > lat_max: lat_deg = lat_max

    # Check Alt bounds (return 0 if outside)
    alt_max_km = alt_min_km + alt_step_km * (nz - 1)
    if alt_km < alt_min_km or alt_km > alt_max_km:
        return 0.0

    lon_deg = lon_deg % 360.0

    # Indices (Float)
    ilat_f = (lat_deg - lat0) / dlat
    ilon_f = (lon_deg - lon0) / dlon
    iz_f = (alt_km - alt_min_km) / alt_step_km

    # Indices (Int)
    ilat0 = int(ilat_f)
    iz0 = int(iz_f)
    
    # Boundary checks
    if ilat0 >= nlat - 1: ilat0 = nlat - 2
    if iz0 >= nz - 1: iz0 = nz - 2
    
    ilon0 = int(np.floor(ilon_f))

    # Weights
    wlat = ilat_f - ilat0
    wlon = ilon_f - np.floor(ilon_f)
    wz = iz_f - iz0

    # Neighbor indices
    ilat1 = ilat0 + 1
    iz1 = iz0 + 1
    j0 = ilon0 % nlon
    j1 = (ilon0 + 1) % nlon

    # Define helper to fetch and interp one grid
    # Grid 0
    c000 = float(grid0[ilat0, j0, iz0])
    c001 = float(grid0[ilat0, j0, iz1])
    c010 = float(grid0[ilat0, j1, iz0])
    c011 = float(grid0[ilat0, j1, iz1])
    c100 = float(grid0[ilat1, j0, iz0])
    c101 = float(grid0[ilat1, j0, iz1])
    c110 = float(grid0[ilat1, j1, iz0])
    c111 = float(grid0[ilat1, j1, iz1])

    # Grid 1
    d000 = float(grid1[ilat0, j0, iz0])
    d001 = float(grid1[ilat0, j0, iz1])
    d010 = float(grid1[ilat0, j1, iz0])
    d011 = float(grid1[ilat0, j1, iz1])
    d100 = float(grid1[ilat1, j0, iz0])
    d101 = float(grid1[ilat1, j0, iz1])
    d110 = float(grid1[ilat1, j1, iz0])
    d111 = float(grid1[ilat1, j1, iz1])

    # Interp Lon
    c00 = c000 + (c010 - c000) * wlon
    c01 = c001 + (c011 - c001) * wlon
    c10 = c100 + (c110 - c100) * wlon
    c11 = c101 + (c111 - c101) * wlon

    d00 = d000 + (d010 - d000) * wlon
    d01 = d001 + (d011 - d001) * wlon
    d10 = d100 + (d110 - d100) * wlon
    d11 = d101 + (d111 - d101) * wlon

    # Interp Alt
    c0 = c00 + (c01 - c00) * wz
    c1 = c10 + (c11 - c10) * wz
    
    d0 = d00 + (d01 - d00) * wz
    d1 = d10 + (d11 - d10) * wz

    # Interp Lat
    val0 = c0 + (c1 - c0) * wlat
    val1 = d0 + (d1 - d0) * wlat
    
    # Temporal mix
    return val0 + (val1 - val0) * mix

@njit(cache=True, nogil=True)
def ecef_from_eci_zrot(x_eci: float, y_eci: float, z_eci: float, theta: float) -> tuple[float, float, float]:
    c = np.cos(theta)
    s = np.sin(theta)
    return (x_eci * c + y_eci * s, -x_eci * s + y_eci * c, z_eci)

@njit(cache=True, nogil=True)
def geodetic_lat_lon_alt_wgs84_km(x: float, y: float, z: float) -> tuple[float, float, float]:
    # WGS84 constants
    a = 6378.137
    e2 = 6.69437999014e-3
    
    p2 = x*x + y*y
    p = np.sqrt(p2)
    lon = np.arctan2(y, x)

    if p < 1e-12:
        return (90.0 if z >= 0 else -90.0, _wrap_lon_deg(np.degrees(lon)), np.abs(z) - 6356.7523)

    # Fast iterative
    lat = np.arctan2(z, p * (1.0 - e2))
    alt = 0.0
    
    for _ in range(4):
        s = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * s * s)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - e2 * N / (N + alt)))

    return (np.degrees(lat), _wrap_lon_deg(np.degrees(lon)), alt)

@njit(cache=True, nogil=True)
def _atm_drag_msis_grid_daysec(Xsat: np.ndarray, Cd: float, AtoM: float, earth_spin: float, GST0: float,
                               t_abs: float, sec_in_day: float,
                               grid_ut00: np.ndarray, grid_ut06: np.ndarray, grid_ut12: np.ndarray, grid_ut18: np.ndarray,
                               grid_tomorrow_ut00: np.ndarray, earth_Re: float, lat0: float, dlat: float, lon0: float,
                               dlon: float, alt_min_km: float, alt_step_km: float, out: np.ndarray) -> None:

    # Coordinate Transforms
    theta = GST0 + earth_spin * t_abs
    rx, ry, rz = Xsat[0], Xsat[1], Xsat[2]
    vx, vy, vz = Xsat[3], Xsat[4], Xsat[5]

    x_ecef, y_ecef, z_ecef = ecef_from_eci_zrot(rx, ry, rz, theta)
    lat_deg, lon_deg, alt_km = geodetic_lat_lon_alt_wgs84_km(x_ecef, y_ecef, z_ecef)

    # Grid Selection & Weighting (6-hour bins: 0, 21600, 43200, 64800)
    if sec_in_day < 21600.0:
        g0, g1 = grid_ut00, grid_ut06
        w = sec_in_day * 4.62962962962963e-5 # 1/21600
    elif sec_in_day < 43200.0:
        g0, g1 = grid_ut06, grid_ut12
        w = (sec_in_day - 21600.0) * 4.62962962962963e-5
    elif sec_in_day < 64800.0:
        g0, g1 = grid_ut12, grid_ut18
        w = (sec_in_day - 43200.0) * 4.62962962962963e-5
    else:
        g0, g1 = grid_ut18, grid_tomorrow_ut00
        w = (sec_in_day - 64800.0) * 4.62962962962963e-5

    rho = _interp_trilinear_dual(g0, g1, w, lat_deg, lon_deg, alt_km,
                                 lat0, dlat, lon0, dlon, alt_min_km, alt_step_km)

    rho_km3 = rho * 1.0e9 # kg/m3 -> kg/km3

    vrel0 = vx - (-earth_spin * ry)
    vrel1 = vy - (earth_spin * rx)
    vrel2 = vz

    vrel = np.sqrt(vrel0*vrel0 + vrel1*vrel1 + vrel2*vrel2)
    fac = -0.5 * rho_km3 * Cd * (AtoM * 1e-6) * vrel

    out[0] = fac * vrel0
    out[1] = fac * vrel1
    out[2] = fac * vrel2

@njit(cache=True, nogil=True)
def atm_drag_msis_grid(Xsat: np.ndarray, Cd: float, AtoM: float, earth_spin: float, GST0: float, t_abs: float,
                       grid_ut00: np.ndarray, grid_ut06: np.ndarray, grid_ut12: np.ndarray, grid_ut18: np.ndarray,
                       grid_tomorrow_ut00: np.ndarray, earth_Re: float, lat0: float, dlat: float, lon0: float,
                       dlon: float, alt_min_km: float, alt_step_km: float, out: np.ndarray) -> None:
    sec_in_day = t_abs % 86400.0
    _atm_drag_msis_grid_daysec(Xsat, Cd, AtoM, earth_spin, GST0, t_abs, sec_in_day,
                               grid_ut00, grid_ut06, grid_ut12, grid_ut18, grid_tomorrow_ut00,
                               earth_Re, lat0, dlat, lon0, dlon, alt_min_km, alt_step_km, out)