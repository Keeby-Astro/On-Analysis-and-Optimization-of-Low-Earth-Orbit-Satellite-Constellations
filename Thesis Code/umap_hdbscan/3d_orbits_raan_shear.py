from __future__ import annotations

import argparse
import os

import numpy as np


MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.145
J2_EARTH = 1.08262668e-3
SIDEREAL_DAY_S = 86164.0905
DEFAULT_PNG_DPI = 600
DEFAULT_PNG_WIDTH_IN = 7.5
DEFAULT_PNG_HEIGHT_IN = 7.5
DEFAULT_LAYOUT_SIZE_PX = 1000
ORBIT_COLOR = "#000000"
ORBIT_LINE_WIDTH = 8


def _parse_csv_ints(text: str, expected_len: int | None = None) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} integer values, received {len(values)}.")
    return values


def _parse_csv_floats(text: str, expected_len: int | None = None) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} numeric values, received {len(values)}.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render synthetic or canonical satellite constellation families as an interactive 3D Plotly figure."
        )
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic orbit generation.")
    parser.add_argument(
        "--count",
        type=int,
        default=24,
        help="Number of satellites to generate for synthetic and generic Walker-style modes.",
    )
    parser.add_argument(
        "--constellation",
        type=str,
        default="raan-shear",
        choices=[
            "synthetic",
            "walker-delta",
            "walker-star",
            "classical-walker",
            "mega-constellation",
            "walker",
            "ballard-rosette",
            "streets-of-coverage",
            "flower",
            "raan-shear",
        ],
        help="Constellation family used to generate the orbital elements.",
    )

    parser.add_argument("--walker-planes", type=int, default=6, help="Number of orbital planes P for Walker modes.")
    parser.add_argument("--walker-phasing", type=int, default=1, help="Relative phasing integer F for Walker modes.")
    parser.add_argument("--walker-altitude-km", type=float, default=550.0, help="Altitude used for Walker modes.")
    parser.add_argument(
        "--walker-eccentricity",
        type=float,
        default=0.0001,
        help="Eccentricity used for Walker modes.",
    )
    parser.add_argument(
        "--walker-inclination-deg",
        type=float,
        default=None,
        help=(
            "Inclination used for Walker modes. If omitted, the code uses 53.0 deg for Delta/Walker and "
            "87.9 deg for Star."
        ),
    )

    parser.add_argument("--classical-planes", type=int, default=6, help="Plane count used by the simple Walker preset.")
    parser.add_argument(
        "--classical-sats-per-plane",
        type=int,
        default=4,
        help="Satellites per plane used by the simple Walker preset.",
    )
    parser.add_argument(
        "--classical-phasing",
        type=int,
        default=1,
        help="Walker phasing used by the simple Walker preset.",
    )
    parser.add_argument(
        "--classical-altitude-km",
        type=float,
        default=550.0,
        help="Altitude used by the simple Walker preset.",
    )
    parser.add_argument(
        "--classical-inclination-deg",
        type=float,
        default=53.0,
        help="Inclination used by the simple Walker preset.",
    )
    parser.add_argument(
        "--classical-eccentricity",
        type=float,
        default=0.0001,
        help="Eccentricity used by the simple Walker preset.",
    )

    parser.add_argument(
        "--mega-shell-planes",
        type=str,
        default="40,40,40",
        help="Comma-separated plane counts for the three-shell mega-constellation preset.",
    )
    parser.add_argument(
        "--mega-shell-sats-per-plane",
        type=str,
        default="8,10,12",
        help="Comma-separated satellites-per-plane values for the three-shell mega-constellation preset.",
    )
    parser.add_argument(
        "--mega-shell-altitudes-km",
        type=str,
        default="340,550,1200",
        help="Comma-separated altitudes for the three-shell mega-constellation preset.",
    )
    parser.add_argument(
        "--mega-shell-inclinations-deg",
        type=str,
        default="53.0,70.0,97.6",
        help="Comma-separated inclinations for the three-shell mega-constellation preset.",
    )
    parser.add_argument(
        "--mega-shell-phasing",
        type=str,
        default="1,3,1",
        help="Comma-separated Walker phasing integers for the mega-constellation preset.",
    )
    parser.add_argument(
        "--mega-shell-eccentricities",
        type=str,
        default="0.0001,0.0002,0.0005",
        help="Comma-separated eccentricities for the mega-constellation preset.",
    )

    parser.add_argument("--rosette-planes", type=int, default=9, help="Plane count P for the Ballard rosette preset.")
    parser.add_argument(
        "--rosette-sats-per-plane",
        type=int,
        default=4,
        help="Satellites per plane for the Ballard rosette preset.",
    )
    parser.add_argument(
        "--rosette-m",
        type=int,
        default=2,
        help="Ballard rosette shift integer m in the (t,p,m) style parameterization.",
    )
    parser.add_argument(
        "--rosette-altitude-km",
        type=float,
        default=1400.0,
        help="Altitude used by the Ballard rosette preset.",
    )
    parser.add_argument(
        "--rosette-inclination-deg",
        type=float,
        default=56.0,
        help="Inclination used by the Ballard rosette preset.",
    )
    parser.add_argument(
        "--rosette-eccentricity",
        type=float,
        default=0.0001,
        help="Eccentricity used by the Ballard rosette preset.",
    )

    parser.add_argument(
        "--soc-planes",
        type=int,
        default=12,
        help="Plane count for the Streets-of-Coverage preset.",
    )
    parser.add_argument(
        "--soc-sats-per-plane",
        type=int,
        default=3,
        help="Satellites per plane for the Streets-of-Coverage preset.",
    )
    parser.add_argument(
        "--soc-altitude-km",
        type=float,
        default=780.0,
        help="Altitude used by the Streets-of-Coverage preset.",
    )
    parser.add_argument(
        "--soc-inclination-deg",
        type=float,
        default=87.0,
        help="Inclination used by the Streets-of-Coverage preset.",
    )
    parser.add_argument(
        "--soc-eccentricity",
        type=float,
        default=0.0001,
        help="Eccentricity used by the Streets-of-Coverage preset.",
    )
    parser.add_argument(
        "--soc-plane-phase-fraction",
        type=float,
        default=0.5,
        help="Adjacent-plane phase fraction used to create synchronized polar streets.",
    )

    parser.add_argument("--flower-petals", type=int, default=6, help="Representative petal count for the Flower preset.")
    parser.add_argument("--flower-satellites", type=int, default=12, help="Satellite count for the Flower preset.")
    parser.add_argument(
        "--flower-revs-per-day",
        type=float,
        default=13.0,
        help="Compatible repeat-ground-track revolution count per sidereal day for the Flower preset.",
    )
    parser.add_argument(
        "--flower-eccentricity",
        type=float,
        default=0.12,
        help="Common eccentricity used by the Flower preset.",
    )
    parser.add_argument(
        "--flower-inclination-deg",
        type=float,
        default=63.4,
        help="Inclination used by the Flower preset.",
    )
    parser.add_argument(
        "--flower-arg-perigee-deg",
        type=float,
        default=270.0,
        help="Argument of perigee used by the Flower preset.",
    )
    parser.add_argument(
        "--flower-raan0-deg",
        type=float,
        default=0.0,
        help="Reference RAAN used by the Flower preset.",
    )
    parser.add_argument(
        "--flower-true-anomaly0-deg",
        type=float,
        default=0.0,
        help="Reference true anomaly used by the Flower preset.",
    )
    parser.add_argument(
        "--flower-phase-numerator",
        type=int,
        default=1,
        help="Phase numerator used by the Flower preset.",
    )
    parser.add_argument(
        "--flower-phase-denominator",
        type=int,
        default=12,
        help="Phase denominator used by the Flower preset.",
    )


    parser.add_argument(
        "--raan-shear-sats-per-plane",
        type=int,
        default=4,
        help="Satellites per plane for the RAAN shear sequence.",
    )
    parser.add_argument(
        "--raan-shear-altitudes-km",
        type=str,
        default="540,575",
        help="Comma-separated altitudes for the two nearby planes in the RAAN shear sequence.",
    )
    parser.add_argument(
        "--raan-shear-inclinations-deg",
        type=str,
        default="53.0,53.0",
        help="Comma-separated inclinations for the two nearby planes in the RAAN shear sequence.",
    )
    parser.add_argument(
        "--raan-shear-eccentricities",
        type=str,
        default="0.0001,0.0001",
        help="Comma-separated eccentricities for the two nearby planes in the RAAN shear sequence.",
    )
    parser.add_argument(
        "--raan-shear-initial-raans-deg",
        type=str,
        default="20.0,40.0",
        help="Comma-separated initial RAAN values for the two nearby planes in the RAAN shear sequence.",
    )
    parser.add_argument(
        "--raan-shear-plane-phase-deg",
        type=str,
        default="0.0,45.0",
        help="Comma-separated in-plane phase offsets applied to the two planes in the RAAN shear sequence.",
    )
    parser.add_argument(
        "--raan-shear-aop-deg",
        type=str,
        default="0.0,0.0",
        help="Comma-separated arguments of perigee used by the RAAN shear sequence.",
    )
    parser.add_argument(
        "--raan-shear-times-days",
        type=str,
        default="0,80,400",
        help="Comma-separated snapshot times in days for the RAAN shear sequence.",
    )

    parser.add_argument(
        "--points-per-orbit",
        type=int,
        default=180,
        help="Samples used to propagate each orbit.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="3d_orbits_plot",
        help="Directory where the Plotly exports are written.",
    )
    parser.add_argument(
        "--earth-texture",
        type=str,
        default=os.environ.get("HDBSCAN_UMAP_EARTH_TEXTURE"),
        help="Optional path to an Earth texture image.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=DEFAULT_PNG_DPI,
        help="DPI used for the default PNG export.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip static PNG export and only write the interactive HTML file.",
    )
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if args.points_per_orbit < 32:
        raise ValueError("--points-per-orbit must be >= 32")
    if args.png_dpi < 72:
        raise ValueError("--png-dpi must be >= 72")
    if args.walker_planes < 1:
        raise ValueError("--walker-planes must be >= 1")
    if args.walker_altitude_km <= 0.0:
        raise ValueError("--walker-altitude-km must be > 0")
    if not (0.0 <= args.walker_eccentricity < 1.0):
        raise ValueError("--walker-eccentricity must satisfy 0 <= e < 1")
    if args.constellation in {"walker-delta", "walker-star", "walker"} and (args.count % args.walker_planes != 0):
        raise ValueError("Walker constellations require --count to be divisible by --walker-planes")
    if args.classical_planes < 1 or args.classical_sats_per_plane < 1:
        raise ValueError("Classical Walker preset requires positive plane and satellite counts")
    if args.rosette_planes < 1 or args.rosette_sats_per_plane < 1:
        raise ValueError("Ballard rosette preset requires positive plane and satellite counts")
    if args.soc_planes < 1 or args.soc_sats_per_plane < 1:
        raise ValueError("Streets-of-Coverage preset requires positive plane and satellite counts")
    if not (0.0 <= args.rosette_eccentricity < 1.0):
        raise ValueError("--rosette-eccentricity must satisfy 0 <= e < 1")
    if not (0.0 <= args.soc_eccentricity < 1.0):
        raise ValueError("--soc-eccentricity must satisfy 0 <= e < 1")
    if args.flower_satellites < 1 or args.flower_petals < 1:
        raise ValueError("Flower preset requires positive petal and satellite counts")
    if args.flower_revs_per_day <= 0.0:
        raise ValueError("--flower-revs-per-day must be > 0")
    if not (0.0 <= args.flower_eccentricity < 1.0):
        raise ValueError("--flower-eccentricity must satisfy 0 <= e < 1")

    shell_planes = _parse_csv_ints(args.mega_shell_planes, expected_len=3)
    shell_sats_per_plane = _parse_csv_ints(args.mega_shell_sats_per_plane, expected_len=3)
    shell_altitudes = _parse_csv_floats(args.mega_shell_altitudes_km, expected_len=3)
    shell_inclinations = _parse_csv_floats(args.mega_shell_inclinations_deg, expected_len=3)
    shell_phasing = _parse_csv_ints(args.mega_shell_phasing, expected_len=3)
    shell_ecc = _parse_csv_floats(args.mega_shell_eccentricities, expected_len=3)

    if sum(shell_planes) != 120:
        raise ValueError("The mega-constellation preset must have exactly 120 total planes.")
    if any(val < 1 for val in shell_planes + shell_sats_per_plane):
        raise ValueError("Mega-constellation preset requires positive plane and satellite counts.")
    if any(val <= 0.0 for val in shell_altitudes):
        raise ValueError("Mega-constellation altitudes must be positive.")
    if any((val < 0.0 or val >= 1.0) for val in shell_ecc):
        raise ValueError("Mega-constellation eccentricities must satisfy 0 <= e < 1.")


    raan_shear_altitudes = _parse_csv_floats(args.raan_shear_altitudes_km, expected_len=2)
    raan_shear_inclinations = _parse_csv_floats(args.raan_shear_inclinations_deg, expected_len=2)
    raan_shear_ecc = _parse_csv_floats(args.raan_shear_eccentricities, expected_len=2)
    raan_shear_initial_raans = _parse_csv_floats(args.raan_shear_initial_raans_deg, expected_len=2)
    raan_shear_plane_phase = _parse_csv_floats(args.raan_shear_plane_phase_deg, expected_len=2)
    raan_shear_aop = _parse_csv_floats(args.raan_shear_aop_deg, expected_len=2)
    raan_shear_times = _parse_csv_floats(args.raan_shear_times_days, expected_len=3)

    if args.raan_shear_sats_per_plane < 1:
        raise ValueError("--raan-shear-sats-per-plane must be >= 1")
    if any(val <= 0.0 for val in raan_shear_altitudes):
        raise ValueError("RAAN shear altitudes must be positive.")
    if any((val < 0.0 or val >= 1.0) for val in raan_shear_ecc):
        raise ValueError("RAAN shear eccentricities must satisfy 0 <= e < 1.")

    args.raan_shear_altitudes_km_list = raan_shear_altitudes
    args.raan_shear_inclinations_deg_list = raan_shear_inclinations
    args.raan_shear_eccentricities_list = raan_shear_ecc
    args.raan_shear_initial_raans_deg_list = raan_shear_initial_raans
    args.raan_shear_plane_phase_deg_list = raan_shear_plane_phase
    args.raan_shear_aop_deg_list = raan_shear_aop
    args.raan_shear_times_days_list = raan_shear_times

    args.mega_shell_planes_list = shell_planes
    args.mega_shell_sats_per_plane_list = shell_sats_per_plane
    args.mega_shell_altitudes_km_list = shell_altitudes
    args.mega_shell_inclinations_deg_list = shell_inclinations
    args.mega_shell_phasing_list = shell_phasing
    args.mega_shell_eccentricities_list = shell_ecc
    return args


def _resolve_earth_texture_path(explicit_path: str | None) -> str | None:
    if explicit_path:
        expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(explicit_path)))
        if os.path.isfile(expanded):
            return expanded
        print(f"Warning: Earth texture not found: {expanded}")

    cwd = os.getcwd()
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else cwd
    parent_dir = os.path.dirname(this_file_dir)
    candidates = [
        os.path.join(cwd, "earth_texture_4k.jpg"),
        os.path.join(this_file_dir, "earth_texture_4k.jpg"),
        os.path.join(this_file_dir, "assets", "earth_texture_4k.jpg"),
        os.path.join(parent_dir, "earth_texture_4k.jpg"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _save_plotly_figure(fig, output_dir: str, filename: str) -> str | None:
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.write_html(out_path, include_plotlyjs=True, full_html=True)
        print(f"Saved interactive plot: {out_path}")
        return out_path
    except Exception as exc:
        print(f"Warning: failed to save interactive plot '{filename}': {exc}")
        return None


def _save_plotly_png(
    fig,
    output_dir: str,
    filename: str,
    dpi: int = DEFAULT_PNG_DPI,
    width_inches: float = DEFAULT_PNG_WIDTH_IN,
    height_inches: float = DEFAULT_PNG_HEIGHT_IN,
    scale: float = 1.0,
) -> str | None:
    try:
        import copy

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        width = max(1, int(round(width_inches * dpi)))
        height = max(1, int(round(height_inches * dpi)))
        fig_png = copy.deepcopy(fig)
        fig_png.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        fig_png.update_scenes(bgcolor="white")
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
        fig_png.write_image(out_path, format="png", width=width, height=height, scale=scale)
        try:
            from PIL import Image

            with Image.open(out_path) as img:
                img.save(out_path, dpi=(dpi, dpi))
        except Exception:
            pass
        print(f"Saved Plotly PNG: {out_path}")
        return out_path
    except Exception as exc:
        print(
            f"Warning: failed to save Plotly PNG '{filename}': {exc}. "
            "Static Plotly image export typically requires the 'kaleido' package."
        )
        return None


def _kepler_eccentric_anomaly_from_mean(
    mean_anomaly: np.ndarray,
    eccentricity: float,
    max_iter: int = 30,
    tol: float = 1e-11,
) -> np.ndarray:
    e = float(eccentricity)
    mean_anomaly = np.asarray(mean_anomaly, dtype=np.float64)
    eccentric_anomaly = mean_anomaly.copy() if e < 0.8 else np.full_like(mean_anomaly, np.pi)

    for _ in range(max_iter):
        f_val = eccentric_anomaly - e * np.sin(eccentric_anomaly) - mean_anomaly
        f_prime = 1.0 - e * np.cos(eccentric_anomaly)
        delta = f_val / np.maximum(f_prime, 1e-14)
        eccentric_anomaly -= delta
        if np.nanmax(np.abs(delta)) < tol:
            break
    return eccentric_anomaly


def _true_from_mean_anomaly(mean_anomaly_rad: float, eccentricity: float) -> float:
    mean_array = np.asarray([mean_anomaly_rad], dtype=np.float64)
    eccentric_anomaly = _kepler_eccentric_anomaly_from_mean(mean_array, eccentricity)[0]
    e_safe = float(np.clip(eccentricity, 0.0, 1.0 - 1e-12))
    return float(
        2.0
        * np.arctan2(
            np.sqrt(1.0 + e_safe) * np.sin(eccentric_anomaly / 2.0),
            np.sqrt(1.0 - e_safe) * np.cos(eccentric_anomaly / 2.0),
        )
    )


def _propagate_one_period_kepler_xyz(
    mu_val: float,
    sma_km: float,
    ecc_val: float,
    inc_rad: float,
    aop_rad: float,
    raan_rad: float,
    ta0_rad: float,
    points_per_orbit: int,
) -> tuple[np.ndarray | None, float | None]:
    if (not np.isfinite(sma_km)) or (not np.isfinite(ecc_val)):
        return None, None
    if sma_km <= 0.0 or ecc_val < 0.0 or ecc_val >= 1.0:
        return None, None

    mean_motion_rad_s = np.sqrt(mu_val / (sma_km ** 3))
    if not np.isfinite(mean_motion_rad_s) or mean_motion_rad_s <= 0.0:
        return None, None
    period_s = (2.0 * np.pi) / mean_motion_rad_s

    e_safe = float(np.clip(ecc_val, 0.0, 1.0 - 1e-12))
    eccentric_anomaly_0 = 2.0 * np.arctan2(
        np.sqrt(1.0 - e_safe) * np.sin(ta0_rad / 2.0),
        np.sqrt(1.0 + e_safe) * np.cos(ta0_rad / 2.0),
    )
    mean_anomaly_0 = np.mod(eccentric_anomaly_0 - e_safe * np.sin(eccentric_anomaly_0), 2.0 * np.pi)

    times = np.linspace(0.0, period_s, points_per_orbit)
    mean_anomaly = np.mod(mean_anomaly_0 + mean_motion_rad_s * times, 2.0 * np.pi)
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
    cos_raan = np.cos(raan_rad)
    sin_raan = np.sin(raan_rad)
    cos_inc = np.cos(inc_rad)
    sin_inc = np.sin(inc_rad)

    r11 = cos_raan * cos_w - sin_raan * sin_w * cos_inc
    r12 = -cos_raan * sin_w - sin_raan * cos_w * cos_inc
    r21 = sin_raan * cos_w + cos_raan * sin_w * cos_inc
    r22 = -sin_raan * sin_w + cos_raan * cos_w * cos_inc
    r31 = sin_w * sin_inc
    r32 = cos_w * sin_inc

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
                r_val, g_val, b_val = palette[3 * i: 3 * i + 3]
                color = f"rgb({int(r_val)},{int(g_val)},{int(b_val)})"
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


def _generate_synthetic_orbits(count: int, seed: int) -> dict[str, np.ndarray | list[str]]:
    rng = np.random.default_rng(seed)
    shell_names = np.array(["mid-inclination", "starlink-like", "high-inclination", "sun-sync-like"], dtype=object)
    shell_inclinations_deg = np.array([43.0, 53.2, 70.0, 97.6], dtype=np.float64)
    shell_altitudes_km = np.array([360.0, 540.0, 1150.0, 580.0], dtype=np.float64)
    shell_altitude_jitter_km = np.array([18.0, 28.0, 45.0, 22.0], dtype=np.float64)
    shell_ecc_base = np.array([0.0008, 0.0013, 0.0026, 0.0011], dtype=np.float64)
    shell_count = len(shell_inclinations_deg)
    planes_per_shell = max(1, int(np.ceil(count / shell_count)))

    sma_km = np.zeros(count, dtype=np.float64)
    ecc = np.zeros(count, dtype=np.float64)
    inc_rad = np.zeros(count, dtype=np.float64)
    aop_rad = np.zeros(count, dtype=np.float64)
    raan_rad = np.zeros(count, dtype=np.float64)
    ta_rad = np.zeros(count, dtype=np.float64)
    labels = []

    for idx in range(count):
        shell_idx = idx % shell_count
        plane_idx = idx // shell_count

        altitude_km = shell_altitudes_km[shell_idx] + rng.normal(0.0, shell_altitude_jitter_km[shell_idx])
        altitude_km = max(220.0, altitude_km)
        sma_km[idx] = R_EARTH_KM + altitude_km

        ecc[idx] = float(np.clip(shell_ecc_base[shell_idx] + rng.normal(0.0, 0.0007), 0.0001, 0.02))
        inc_deg = shell_inclinations_deg[shell_idx] + rng.normal(0.0, 0.35)
        inc_rad[idx] = np.deg2rad(inc_deg)

        raan_deg = (
            plane_idx * (360.0 / planes_per_shell)
            + shell_idx * 11.0
            + rng.normal(0.0, 1.8)
        ) % 360.0
        aop_deg = (shell_idx * 35.0 + plane_idx * 19.0 + rng.normal(0.0, 8.0)) % 360.0
        ta_deg = (idx * (360.0 / count) + rng.normal(0.0, 10.0)) % 360.0

        raan_rad[idx] = np.deg2rad(raan_deg)
        aop_rad[idx] = np.deg2rad(aop_deg)
        ta_rad[idx] = np.deg2rad(ta_deg)
        labels.append(f"synthetic_sat_{idx + 1:02d}")

    return {
        "sma_km": sma_km,
        "ecc": ecc,
        "inc_rad": inc_rad,
        "aop_rad": aop_rad,
        "raan_rad": raan_rad,
        "ta_rad": ta_rad,
        "labels": labels,
    }


def _build_constellation_block(
    sma_km: np.ndarray,
    ecc: np.ndarray,
    inc_rad: np.ndarray,
    aop_rad: np.ndarray,
    raan_rad: np.ndarray,
    ta_rad: np.ndarray,
    label_prefix: str,
) -> dict[str, np.ndarray | list[str]]:
    count = len(sma_km)
    return {
        "sma_km": np.asarray(sma_km, dtype=np.float64),
        "ecc": np.asarray(ecc, dtype=np.float64),
        "inc_rad": np.asarray(inc_rad, dtype=np.float64),
        "aop_rad": np.asarray(aop_rad, dtype=np.float64),
        "raan_rad": np.asarray(raan_rad, dtype=np.float64),
        "ta_rad": np.asarray(ta_rad, dtype=np.float64),
        "labels": [f"{label_prefix}_{idx + 1:03d}" for idx in range(count)],
    }


def _generate_walker_orbits(
    count: int,
    planes: int,
    phasing: int,
    altitude_km: float,
    inclination_deg: float,
    eccentricity: float,
    walker_type: str,
    family_name: str | None = None,
) -> dict[str, np.ndarray | list[str]]:
    satellites_per_plane = count // planes
    sma_km = np.full(count, R_EARTH_KM + altitude_km, dtype=np.float64)
    ecc = np.full(count, eccentricity, dtype=np.float64)
    inc_rad = np.full(count, np.deg2rad(inclination_deg), dtype=np.float64)
    aop_rad = np.zeros(count, dtype=np.float64)
    raan_rad = np.zeros(count, dtype=np.float64)
    ta_rad = np.zeros(count, dtype=np.float64)

    if walker_type == "walker-delta":
        raan_span_deg = 360.0
        default_family_name = "walker_delta"
    elif walker_type == "walker-star":
        raan_span_deg = 180.0
        default_family_name = "walker_star"
    else:
        raise ValueError(f"Unsupported Walker type: {walker_type}")

    family_name = family_name or default_family_name
    delta_mean_anomaly_deg = 360.0 / count
    plane_spacing_deg = raan_span_deg / planes
    in_plane_spacing_deg = 360.0 / satellites_per_plane

    sat_idx = 0
    for plane_idx in range(planes):
        raan_deg = plane_idx * plane_spacing_deg
        plane_phase_deg = (phasing * plane_idx * delta_mean_anomaly_deg) % 360.0
        for slot_idx in range(satellites_per_plane):
            ta_deg = (slot_idx * in_plane_spacing_deg + plane_phase_deg) % 360.0
            raan_rad[sat_idx] = np.deg2rad(raan_deg)
            ta_rad[sat_idx] = np.deg2rad(ta_deg)
            sat_idx += 1

    return _build_constellation_block(sma_km, ecc, inc_rad, aop_rad, raan_rad, ta_rad, family_name)


def _generate_classical_walker_orbits(args: argparse.Namespace) -> dict[str, np.ndarray | list[str]]:
    count = args.classical_planes * args.classical_sats_per_plane
    return _generate_walker_orbits(
        count=count,
        planes=args.classical_planes,
        phasing=args.classical_phasing,
        altitude_km=float(args.classical_altitude_km),
        inclination_deg=float(args.classical_inclination_deg),
        eccentricity=float(args.classical_eccentricity),
        walker_type="walker-delta",
        family_name="classical_walker",
    )


def _generate_ballard_rosette_orbits(args: argparse.Namespace) -> dict[str, np.ndarray | list[str]]:
    planes = int(args.rosette_planes)
    sats_per_plane = int(args.rosette_sats_per_plane)
    count = planes * sats_per_plane
    sma_km = np.full(count, R_EARTH_KM + float(args.rosette_altitude_km), dtype=np.float64)
    ecc = np.full(count, float(args.rosette_eccentricity), dtype=np.float64)
    inc_rad = np.full(count, np.deg2rad(float(args.rosette_inclination_deg)), dtype=np.float64)
    aop_rad = np.zeros(count, dtype=np.float64)
    raan_rad = np.zeros(count, dtype=np.float64)
    ta_rad = np.zeros(count, dtype=np.float64)

    plane_spacing_deg = 360.0 / planes
    in_plane_spacing_deg = 360.0 / sats_per_plane
    equivalent_shift_deg = (int(args.rosette_m) * 360.0) / count

    sat_idx = 0
    for plane_idx in range(planes):
        raan_deg = plane_idx * plane_spacing_deg
        plane_phase_deg = (plane_idx * equivalent_shift_deg) % 360.0
        for slot_idx in range(sats_per_plane):
            ta_deg = (slot_idx * in_plane_spacing_deg + plane_phase_deg) % 360.0
            raan_rad[sat_idx] = np.deg2rad(raan_deg)
            ta_rad[sat_idx] = np.deg2rad(ta_deg)
            sat_idx += 1

    return _build_constellation_block(sma_km, ecc, inc_rad, aop_rad, raan_rad, ta_rad, "ballard_rosette")


def _generate_streets_of_coverage_orbits(args: argparse.Namespace) -> dict[str, np.ndarray | list[str]]:
    planes = int(args.soc_planes)
    sats_per_plane = int(args.soc_sats_per_plane)
    count = planes * sats_per_plane
    sma_km = np.full(count, R_EARTH_KM + float(args.soc_altitude_km), dtype=np.float64)
    ecc = np.full(count, float(args.soc_eccentricity), dtype=np.float64)
    inc_rad = np.full(count, np.deg2rad(float(args.soc_inclination_deg)), dtype=np.float64)
    aop_rad = np.zeros(count, dtype=np.float64)
    raan_rad = np.zeros(count, dtype=np.float64)
    ta_rad = np.zeros(count, dtype=np.float64)

    plane_spacing_deg = 360.0 / planes
    in_plane_spacing_deg = 360.0 / sats_per_plane
    adjacent_plane_shift_deg = float(args.soc_plane_phase_fraction) * in_plane_spacing_deg

    sat_idx = 0
    for plane_idx in range(planes):
        raan_deg = plane_idx * plane_spacing_deg
        plane_phase_deg = (plane_idx * adjacent_plane_shift_deg) % 360.0
        for slot_idx in range(sats_per_plane):
            ta_deg = (slot_idx * in_plane_spacing_deg + plane_phase_deg) % 360.0
            raan_rad[sat_idx] = np.deg2rad(raan_deg)
            ta_rad[sat_idx] = np.deg2rad(ta_deg)
            sat_idx += 1

    return _build_constellation_block(sma_km, ecc, inc_rad, aop_rad, raan_rad, ta_rad, "streets_of_coverage")


def _semi_major_axis_from_revs_per_sidereal_day(revs_per_day: float) -> float:
    mean_motion_rad_s = 2.0 * np.pi * revs_per_day / SIDEREAL_DAY_S
    return (MU_EARTH_KM3_S2 / (mean_motion_rad_s ** 2)) ** (1.0 / 3.0)



def _j2_nodal_regression_rate_rad_s(sma_km: float, eccentricity: float, inclination_rad: float) -> float:
    p_km = sma_km * (1.0 - eccentricity ** 2)
    mean_motion_rad_s = np.sqrt(MU_EARTH_KM3_S2 / (sma_km ** 3))
    return -1.5 * J2_EARTH * mean_motion_rad_s * (R_EARTH_KM / p_km) ** 2 * np.cos(inclination_rad)


def _generate_raan_shear_snapshot(
    altitudes_km: list[float],
    inclinations_deg: list[float],
    eccentricities: list[float],
    initial_raans_deg: list[float],
    plane_phase_deg: list[float],
    aop_deg: list[float],
    sats_per_plane: int,
    time_days: float,
) -> dict[str, np.ndarray | list[str]]:
    plane_blocks: list[dict[str, np.ndarray | list[str]]] = []
    in_plane_spacing_deg = 360.0 / sats_per_plane
    elapsed_s = float(time_days) * 86400.0

    for plane_idx in range(2):
        sma_val = R_EARTH_KM + float(altitudes_km[plane_idx])
        ecc_val = float(eccentricities[plane_idx])
        inc_val_rad = np.deg2rad(float(inclinations_deg[plane_idx]))
        rate_rad_s = _j2_nodal_regression_rate_rad_s(sma_val, ecc_val, inc_val_rad)
        raan_val_rad = np.mod(np.deg2rad(float(initial_raans_deg[plane_idx])) + rate_rad_s * elapsed_s, 2.0 * np.pi)

        sma_km = np.full(sats_per_plane, sma_val, dtype=np.float64)
        ecc = np.full(sats_per_plane, ecc_val, dtype=np.float64)
        inc_rad = np.full(sats_per_plane, inc_val_rad, dtype=np.float64)
        aop_rad = np.full(sats_per_plane, np.deg2rad(float(aop_deg[plane_idx])), dtype=np.float64)
        raan_rad = np.full(sats_per_plane, raan_val_rad, dtype=np.float64)
        ta_rad = np.zeros(sats_per_plane, dtype=np.float64)

        for slot_idx in range(sats_per_plane):
            ta_deg = (slot_idx * in_plane_spacing_deg + float(plane_phase_deg[plane_idx])) % 360.0
            ta_rad[slot_idx] = np.deg2rad(ta_deg)

        plane_blocks.append(
            _build_constellation_block(
                sma_km=sma_km,
                ecc=ecc,
                inc_rad=inc_rad,
                aop_rad=aop_rad,
                raan_rad=raan_rad,
                ta_rad=ta_rad,
                label_prefix=f"raan_shear_plane_{plane_idx + 1}",
            )
        )

    return _concatenate_constellation_blocks(plane_blocks)


def _generate_raan_shear_snapshots(args: argparse.Namespace) -> list[tuple[dict[str, np.ndarray | list[str]], str]]:
    snapshots: list[tuple[dict[str, np.ndarray | list[str]], str]] = []
    for snap_idx, time_days in enumerate(args.raan_shear_times_days_list):
        snapshot = _generate_raan_shear_snapshot(
            altitudes_km=args.raan_shear_altitudes_km_list,
            inclinations_deg=args.raan_shear_inclinations_deg_list,
            eccentricities=args.raan_shear_eccentricities_list,
            initial_raans_deg=args.raan_shear_initial_raans_deg_list,
            plane_phase_deg=args.raan_shear_plane_phase_deg_list,
            aop_deg=args.raan_shear_aop_deg_list,
            sats_per_plane=int(args.raan_shear_sats_per_plane),
            time_days=float(time_days),
        )
        snapshots.append((snapshot, f"raan_shear_t{snap_idx}"))
    return snapshots


def _generate_flower_orbits(args: argparse.Namespace) -> dict[str, np.ndarray | list[str]]:
    count = int(args.flower_satellites)
    petals = int(args.flower_petals)
    sma_val = _semi_major_axis_from_revs_per_sidereal_day(float(args.flower_revs_per_day))
    ecc_val = float(args.flower_eccentricity)
    inc_val_rad = np.deg2rad(float(args.flower_inclination_deg))
    aop_val_rad = np.deg2rad(float(args.flower_arg_perigee_deg))
    raan0_rad = np.deg2rad(float(args.flower_raan0_deg))
    ta0_rad = np.deg2rad(float(args.flower_true_anomaly0_deg))
    phase_num = int(args.flower_phase_numerator)
    phase_den = max(1, int(args.flower_phase_denominator))

    sma_km = np.full(count, sma_val, dtype=np.float64)
    ecc = np.full(count, ecc_val, dtype=np.float64)
    inc_rad = np.full(count, inc_val_rad, dtype=np.float64)
    aop_rad = np.full(count, aop_val_rad, dtype=np.float64)
    raan_rad = np.zeros(count, dtype=np.float64)
    ta_rad = np.zeros(count, dtype=np.float64)

    for sat_idx in range(count):
        petal_idx = sat_idx % petals
        phase_idx = sat_idx // petals
        raan_rad[sat_idx] = np.mod(raan0_rad + (2.0 * np.pi * petal_idx / petals), 2.0 * np.pi)
        mean_phase_rad = 2.0 * np.pi * ((phase_num * sat_idx) / phase_den + phase_idx / max(1, count // petals))
        ta_rad[sat_idx] = np.mod(ta0_rad + _true_from_mean_anomaly(mean_phase_rad, ecc_val), 2.0 * np.pi)

    return _build_constellation_block(sma_km, ecc, inc_rad, aop_rad, raan_rad, ta_rad, "flower")


def _concatenate_constellation_blocks(blocks: list[dict[str, np.ndarray | list[str]]]) -> dict[str, np.ndarray | list[str]]:
    return {
        "sma_km": np.concatenate([np.asarray(block["sma_km"], dtype=np.float64) for block in blocks]),
        "ecc": np.concatenate([np.asarray(block["ecc"], dtype=np.float64) for block in blocks]),
        "inc_rad": np.concatenate([np.asarray(block["inc_rad"], dtype=np.float64) for block in blocks]),
        "aop_rad": np.concatenate([np.asarray(block["aop_rad"], dtype=np.float64) for block in blocks]),
        "raan_rad": np.concatenate([np.asarray(block["raan_rad"], dtype=np.float64) for block in blocks]),
        "ta_rad": np.concatenate([np.asarray(block["ta_rad"], dtype=np.float64) for block in blocks]),
        "labels": [label for block in blocks for label in list(block["labels"])],
    }


def _generate_mega_constellation_orbits(args: argparse.Namespace) -> dict[str, np.ndarray | list[str]]:
    blocks: list[dict[str, np.ndarray | list[str]]] = []
    plane_offset_deg = 0.0
    shell_true_anomaly_bias_deg = [0.0, 7.5, 15.0]

    for shell_idx in range(3):
        planes = int(args.mega_shell_planes_list[shell_idx])
        sats_per_plane = int(args.mega_shell_sats_per_plane_list[shell_idx])
        count = planes * sats_per_plane
        shell_block = _generate_walker_orbits(
            count=count,
            planes=planes,
            phasing=int(args.mega_shell_phasing_list[shell_idx]),
            altitude_km=float(args.mega_shell_altitudes_km_list[shell_idx]),
            inclination_deg=float(args.mega_shell_inclinations_deg_list[shell_idx]),
            eccentricity=float(args.mega_shell_eccentricities_list[shell_idx]),
            walker_type="walker-delta",
            family_name=f"mega_shell_{shell_idx + 1}",
        )
        shell_block["raan_rad"] = np.mod(
            np.asarray(shell_block["raan_rad"], dtype=np.float64) + np.deg2rad(plane_offset_deg),
            2.0 * np.pi,
        )
        shell_block["ta_rad"] = np.mod(
            np.asarray(shell_block["ta_rad"], dtype=np.float64) + np.deg2rad(shell_true_anomaly_bias_deg[shell_idx]),
            2.0 * np.pi,
        )
        shell_block["aop_rad"] = np.full(count, np.deg2rad(10.0 * shell_idx), dtype=np.float64)
        plane_offset_deg += 360.0 / max(planes, 1)
        blocks.append(shell_block)

    return _concatenate_constellation_blocks(blocks)


def _generate_constellation_orbits(args: argparse.Namespace) -> tuple[dict[str, np.ndarray | list[str]], str]:
    if args.constellation == "synthetic":
        return _generate_synthetic_orbits(count=args.count, seed=args.seed), "synthetic"

    if args.constellation == "classical-walker":
        return _generate_classical_walker_orbits(args), "classical_walker"

    if args.constellation == "mega-constellation":
        return _generate_mega_constellation_orbits(args), "mega_constellation"

    if args.constellation == "ballard-rosette":
        return _generate_ballard_rosette_orbits(args), "ballard_rosette"

    if args.constellation == "streets-of-coverage":
        return _generate_streets_of_coverage_orbits(args), "streets_of_coverage"

    if args.constellation == "flower":
        return _generate_flower_orbits(args), "flower"

    default_inclination_deg = 53.0 if args.constellation in {"walker-delta", "walker"} else 87.9
    inclination_deg = (
        float(args.walker_inclination_deg)
        if args.walker_inclination_deg is not None
        else default_inclination_deg
    )
    walker_type = "walker-delta" if args.constellation == "walker" else args.constellation
    family_name = "walker" if args.constellation == "walker" else None
    return (
        _generate_walker_orbits(
            count=args.count,
            planes=args.walker_planes,
            phasing=args.walker_phasing,
            altitude_km=float(args.walker_altitude_km),
            inclination_deg=inclination_deg,
            eccentricity=float(args.walker_eccentricity),
            walker_type=walker_type,
            family_name=family_name,
        ),
        args.constellation.replace("-", "_"),
    )


def _plot_constellation_orbits(
    constellation_orbits: dict[str, np.ndarray | list[str]],
    output_dir: str,
    output_stem: str,
    r_e_val: float,
    mu_val: float,
    points_per_orbit: int,
    texture_path: str | None,
    export_png: bool,
    png_dpi: int,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError("Plotly is required for 3D orbit plotting.") from exc

    sma_km = np.asarray(constellation_orbits["sma_km"], dtype=np.float64)
    ecc = np.asarray(constellation_orbits["ecc"], dtype=np.float64)
    inc_rad = np.asarray(constellation_orbits["inc_rad"], dtype=np.float64)
    aop_rad = np.asarray(constellation_orbits["aop_rad"], dtype=np.float64)
    raan_rad = np.asarray(constellation_orbits["raan_rad"], dtype=np.float64)
    ta_rad = np.asarray(constellation_orbits["ta_rad"], dtype=np.float64)

    if len(sma_km) == 0:
        raise ValueError("No satellites were generated.")

    fig = go.Figure()
    fig.add_trace(_build_textured_earth_surface_trace(r_e_val, texture_path))

    orbit_line_x = []
    orbit_line_y = []
    orbit_line_z = []
    sat_x = []
    sat_y = []
    sat_z = []
    periods_min = []

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

        orbit_line_x.extend(xyz[:, 0].tolist())
        orbit_line_x.append(np.nan)
        orbit_line_y.extend(xyz[:, 1].tolist())
        orbit_line_y.append(np.nan)
        orbit_line_z.extend(xyz[:, 2].tolist())
        orbit_line_z.append(np.nan)

        sat_x.append(float(xyz[0, 0]))
        sat_y.append(float(xyz[0, 1]))
        sat_z.append(float(xyz[0, 2]))
        periods_min.append(period_s / 60.0)

    if not sat_x:
        raise RuntimeError("No valid orbit states were available for propagation.")

    fig.add_trace(
        go.Scatter3d(
            x=orbit_line_x,
            y=orbit_line_y,
            z=orbit_line_z,
            mode="lines",
            line={"color": ORBIT_COLOR, "width": ORBIT_LINE_WIDTH},
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
            marker={"size": 4.5, "color": ORBIT_COLOR, "opacity": 0.98},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    apoapsis_km = sma_km * (1.0 + np.clip(ecc, 0.0, None))
    finite_apo = apoapsis_km[np.isfinite(apoapsis_km)]
    bound = float(max(r_e_val, np.max(finite_apo))) * 1.08 if finite_apo.size else float(r_e_val) * 2.0
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
        width=DEFAULT_LAYOUT_SIZE_PX,
        height=DEFAULT_LAYOUT_SIZE_PX,
        scene={
            "xaxis": axis_layout,
            "yaxis": axis_layout,
            "zaxis": axis_layout,
            "aspectmode": "cube",
            "bgcolor": "rgba(0,0,0,0)",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 0.95}},
        },
    )

    _save_plotly_figure(fig, output_dir, f"{output_stem}.html")
    if export_png:
        _save_plotly_png(fig, output_dir, f"{output_stem}.png", dpi=png_dpi)
    print(
        "Orbit summary: "
        f"{len(sat_x)} satellites, {points_per_orbit} points per orbit, "
        f"period range {min(periods_min):.2f}-{max(periods_min):.2f} min."
    )


def main() -> None:
    args = _parse_args()
    texture_path = _resolve_earth_texture_path(args.earth_texture)
    if texture_path:
        print(f"Using Earth texture: {texture_path}")
    else:
        print("Earth texture not found; using shaded Earth surface.")

    if args.constellation == "raan-shear":
        for constellation_orbits, output_stem in _generate_raan_shear_snapshots(args):
            _plot_constellation_orbits(
                constellation_orbits=constellation_orbits,
                output_dir=args.output_dir,
                output_stem=output_stem,
                r_e_val=R_EARTH_KM,
                mu_val=MU_EARTH_KM3_S2,
                points_per_orbit=args.points_per_orbit,
                texture_path=texture_path,
                export_png=not args.no_png,
                png_dpi=args.png_dpi,
            )
        return

    constellation_orbits, output_stem = _generate_constellation_orbits(args)
    _plot_constellation_orbits(
        constellation_orbits=constellation_orbits,
        output_dir=args.output_dir,
        output_stem=output_stem,
        r_e_val=R_EARTH_KM,
        mu_val=MU_EARTH_KM3_S2,
        points_per_orbit=args.points_per_orbit,
        texture_path=texture_path,
        export_png=not args.no_png,
        png_dpi=args.png_dpi,
    )


if __name__ == "__main__":
    main()
