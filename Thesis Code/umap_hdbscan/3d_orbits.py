from __future__ import annotations

import argparse
import os

import numpy as np


MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.145
DEFAULT_PNG_DPI = 600
DEFAULT_PNG_WIDTH_IN = 7.5
DEFAULT_PNG_HEIGHT_IN = 7.5
DEFAULT_LAYOUT_SIZE_PX = 1000
ORBIT_COLOR = "#000000"
ORBIT_LINE_WIDTH = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render synthetic satellite orbits as an interactive 3D Plotly figure."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic orbit generation.")
    parser.add_argument("--count", type=int, default=8, help="Number of synthetic satellites to generate.")
    parser.add_argument(
        "--points-per-orbit",
        type=int,
        default=180,
        help="Samples used to propagate each synthetic orbit.",
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

        labels.append(
            f"synthetic_sat_{idx + 1:02d}<br>"
            f"shell={shell_names[shell_idx]}<br>"
            f"altitude={altitude_km:.1f} km<br>"
            f"inclination={inc_deg:.2f} deg"
        )

    return {
        "sma_km": sma_km,
        "ecc": ecc,
        "inc_rad": inc_rad,
        "aop_rad": aop_rad,
        "raan_rad": raan_rad,
        "ta_rad": ta_rad,
        "labels": labels,
    }


def _plot_synthetic_orbits(
    synthetic_orbits: dict[str, np.ndarray | list[str]],
    output_dir: str,
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

    sma_km = np.asarray(synthetic_orbits["sma_km"], dtype=np.float64)
    ecc = np.asarray(synthetic_orbits["ecc"], dtype=np.float64)
    inc_rad = np.asarray(synthetic_orbits["inc_rad"], dtype=np.float64)
    aop_rad = np.asarray(synthetic_orbits["aop_rad"], dtype=np.float64)
    raan_rad = np.asarray(synthetic_orbits["raan_rad"], dtype=np.float64)
    ta_rad = np.asarray(synthetic_orbits["ta_rad"], dtype=np.float64)
    labels = list(synthetic_orbits["labels"])

    if len(sma_km) == 0:
        raise ValueError("No synthetic satellites were generated.")

    fig = go.Figure()
    fig.add_trace(_build_textured_earth_surface_trace(r_e_val, texture_path))

    orbit_line_x = []
    orbit_line_y = []
    orbit_line_z = []
    sat_x = []
    sat_y = []
    sat_z = []
    sat_colors = []
    sat_hover = []
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
        sat_colors.append(ORBIT_COLOR)
        sat_hover.append(f"{labels[idx]}<br>period={period_s / 60.0:.2f} min")
        periods_min.append(period_s / 60.0)

    if not sat_x:
        raise RuntimeError("No valid synthetic orbit states were available for propagation.")

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
            marker={"size": 4.5, "color": sat_colors, "opacity": 0.98},
            text=sat_hover,
            hovertemplate="%{text}<extra></extra>",
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
        title={
            "text": (
                f"Synthetic satellite constellation | n={len(sat_x)} | "
                f"period range={min(periods_min):.1f}-{max(periods_min):.1f} min"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        scene={
            "xaxis": axis_layout,
            "yaxis": axis_layout,
            "zaxis": axis_layout,
            "aspectmode": "cube",
            "bgcolor": "rgba(0,0,0,0)",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 0.95}},
        },
    )

    _save_plotly_figure(fig, output_dir, "3d_orbits_plotly.html")
    if export_png:
        _save_plotly_png(fig, output_dir, "3d_orbits_plotly.png", dpi=png_dpi)
    print(
        "Synthetic orbit summary: "
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

    synthetic_orbits = _generate_synthetic_orbits(count=args.count, seed=args.seed)
    _plot_synthetic_orbits(
        synthetic_orbits=synthetic_orbits,
        output_dir=args.output_dir,
        r_e_val=R_EARTH_KM,
        mu_val=MU_EARTH_KM3_S2,
        points_per_orbit=args.points_per_orbit,
        texture_path=texture_path,
        export_png=not args.no_png,
        png_dpi=args.png_dpi,
    )


if __name__ == "__main__":
    main()