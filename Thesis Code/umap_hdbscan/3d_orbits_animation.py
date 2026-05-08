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
            "tle-backup",
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
        default="16,18,20",
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
        default=360,
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
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Export a smoothly looping GIF animation of the satellites along their orbits.",
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="Skip HTML/PNG export and only write the looping GIF animation.",
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=120,
        help="Number of frames in the looping GIF.",
    )
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=60.0,
        help="Playback frame rate for the GIF.",
    )
    parser.add_argument(
        "--gif-dpi",
        type=int,
        default=600,
        help="Effective DPI of each GIF frame (>= 600 recommended).",
    )
    parser.add_argument(
        "--gif-width-in",
        type=float,
        default=5.0,
        help="Frame width in inches (multiplied by --gif-dpi to set pixel size).",
    )
    parser.add_argument(
        "--gif-height-in",
        type=float,
        default=5.0,
        help="Frame height in inches (multiplied by --gif-dpi to set pixel size).",
    )
    parser.add_argument(
        "--gif-line-width",
        type=float,
        default=0.8,
        help="Orbit polyline width used in the GIF (Plotly Scatter3d line width).",
    )
    parser.add_argument(
        "--gif-marker-size",
        type=float,
        default=2.0,
        help="Satellite marker size used in the GIF.",
    )
    parser.add_argument(
        "--gif-overlay-color",
        type=str,
        default="#000000",
        help="Color of the satellite markers in the GIF.",
    )
    parser.add_argument(
        "--gif-orbit-color",
        type=str,
        default="rgba(64,64,64,0.85)",
        help="Color (any Plotly color string, including rgba) of the orbit polylines in the GIF.",
    )
    parser.add_argument(
        "--gif-transparent",
        action="store_true",
        default=True,
        help="Render the GIF background as transparent (default).",
    )
    parser.add_argument(
        "--gif-no-transparent",
        dest="gif_transparent",
        action="store_false",
        help="Render the GIF on a solid white background instead of transparent.",
    )
    parser.add_argument(
        "--gif-earth-grid",
        type=int,
        default=720,
        help="Earth surface longitude samples for the GIF (latitude uses half + 1).",
    )
    parser.add_argument(
        "--gif-loop-duration-s",
        type=float,
        default=0.0,
        help=(
            "Loop duration in seconds of simulated time. 0 (default) sets it to the longest "
            "satellite orbital period so the slowest satellite completes exactly one orbit per loop."
        ),
    )
    parser.add_argument(
        "--gif-workers",
        type=int,
        default=8,
        help=(
            "Number of parallel processes used to render GIF frames. 0 (default) auto-selects "
            "min(8, CPU count). 1 disables multiprocessing."
        ),
    )
    parser.add_argument(
        "--gif-frame-timeout",
        type=float,
        default=60.0,
        help=(
            "Per-frame timeout in seconds for the parallel GIF renderer. If no frame completes "
            "within this window, the worker pool is restarted and the unfinished frames are retried. "
            "Set <= 0 to disable the watchdog."
        ),
    )
    parser.add_argument(
        "--gif-frame-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for stalled/failed GIF frames before giving up on them.",
    )
    parser.add_argument(
        "--tle-backup-dir",
        type=str,
        default="starlink_backup",
        help="Directory containing per-satellite TLE files (mode: tle-backup).",
    )
    parser.add_argument(
        "--tle-target-epoch",
        type=str,
        default="2025-12-31T00:00:00",
        help="Target UTC epoch (ISO format) at which to start the TLE-backup propagation.",
    )
    parser.add_argument(
        "--tle-cluster-csv",
        type=str,
        default=os.path.join(
            "clusters", "global_analysis", "tables", "combined_cluster_labels_global.csv"
        ),
        help="CSV mapping sat_id -> global_cluster_id used to color the TLE-backup satellites.",
    )
    parser.add_argument(
        "--tle-realtime-duration-s",
        type=float,
        default=180.0,
        help=(
            "Total simulated wall-clock duration in seconds for the TLE-backup real-time GIF. "
            "Total frames = round(duration * --gif-fps). Playback is real-time (1:1)."
        ),
    )
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if args.points_per_orbit < 32:
        raise ValueError("--points-per-orbit must be >= 32")
    if args.png_dpi < 72:
        raise ValueError("--png-dpi must be >= 72")
    if args.gif_frames < 8:
        raise ValueError("--gif-frames must be >= 8")
    if args.gif_fps <= 0.0:
        raise ValueError("--gif-fps must be > 0")
    if args.gif_dpi < 72:
        raise ValueError("--gif-dpi must be >= 72")
    if args.gif_width_in <= 0.0 or args.gif_height_in <= 0.0:
        raise ValueError("--gif-width-in and --gif-height-in must be > 0")
    if args.gif_line_width <= 0.0 or args.gif_marker_size <= 0.0:
        raise ValueError("--gif-line-width and --gif-marker-size must be > 0")
    if args.gif_earth_grid < 90:
        raise ValueError("--gif-earth-grid must be >= 90")
    if args.gif_loop_duration_s < 0.0:
        raise ValueError("--gif-loop-duration-s must be >= 0")
    if args.gif_workers < 0:
        raise ValueError("--gif-workers must be >= 0")
    if args.gif_only:
        args.gif = True
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


def _render_plotly_frame_to_pil(
    fig,
    width_px: int,
    height_px: int,
    scale: float,
    keep_alpha: bool = False,
):
    import io

    from PIL import Image

    png_bytes = fig.to_image(format="png", width=width_px, height=height_px, scale=scale)
    img = Image.open(io.BytesIO(png_bytes))
    return img.convert("RGBA") if keep_alpha else img.convert("RGB")


# Multiprocessing worker state. Module-level so child processes (Windows spawn)
# can import it. The init payload (the figure dict including the Earth surface)
# is shipped once per worker; each task only sends the satellite XYZ for a frame.
_GIF_WORKER_STATE: dict = {}


_KALEIDO_NOISE_PATTERNS = (
    "Resorting to unclean kill browser",
    "unclean kill",
)


class _FilteredStderr:
    """Wrapper that drops lines matching known-noisy Kaleido/Chromium shutdown
    messages but otherwise behaves like the underlying stream."""

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""

    def write(self, data):
        if not data:
            return 0
        self._buf += data
        total = len(data)
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(p in line for p in _KALEIDO_NOISE_PATTERNS):
                self._stream.write(line + "\n")
        return total

    def flush(self):
        if self._buf and not any(p in self._buf for p in _KALEIDO_NOISE_PATTERNS):
            self._stream.write(self._buf)
        self._buf = ""
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _silence_kaleido_noise() -> None:
    import logging
    import sys

    for name in ("kaleido", "choreographer", "logistro"):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass
    if not isinstance(sys.stderr, _FilteredStderr):
        sys.stderr = _FilteredStderr(sys.stderr)


def _gif_worker_init(fig_dict: dict, width_px: int, height_px: int, scale: float) -> None:
    import plotly.graph_objects as go

    _silence_kaleido_noise()
    _GIF_WORKER_STATE["fig"] = go.Figure(fig_dict)
    _GIF_WORKER_STATE["w"] = int(width_px)
    _GIF_WORKER_STATE["h"] = int(height_px)
    _GIF_WORKER_STATE["scale"] = float(scale)
    # Index of the trace whose x/y/z get updated each frame. Defaults to the
    # last trace for backward compatibility with the legacy 3-trace base
    # figure (Earth + orbits + sats); the overlay path uses a single-trace
    # figure and sets this to 0 via _GIF_WORKER_TRACE_IDX.
    _GIF_WORKER_STATE["trace_idx"] = len(_GIF_WORKER_STATE["fig"].data) - 1


def _gif_worker_init_colored(
    fig_dict: dict, width_px: int, height_px: int, scale: float, marker_size: float
) -> None:
    _gif_worker_init(fig_dict, width_px, height_px, scale)
    _GIF_WORKER_STATE["marker_size"] = float(marker_size)


def _gif_worker_render(payload: tuple) -> tuple:
    frame_idx, x_list, y_list, z_list = payload
    fig = _GIF_WORKER_STATE["fig"]
    trace = fig.data[_GIF_WORKER_STATE["trace_idx"]]
    # Single batched update -> one Plotly validation pass instead of three.
    trace.update(x=x_list, y=y_list, z=z_list)
    png_bytes = fig.to_image(
        format="png",
        width=_GIF_WORKER_STATE["w"],
        height=_GIF_WORKER_STATE["h"],
        scale=_GIF_WORKER_STATE["scale"],
    )
    return frame_idx, png_bytes


def _gif_worker_render_colored(payload: tuple) -> tuple:
    frame_idx, x_list, y_list, z_list, colors = payload
    fig = _GIF_WORKER_STATE["fig"]
    trace = fig.data[_GIF_WORKER_STATE["trace_idx"]]
    trace.update(x=x_list, y=y_list, z=z_list, marker={"color": colors,
                 "size": _GIF_WORKER_STATE.get("marker_size", 3.0),
                 "opacity": 1.0,
                 "line": {"color": "rgba(40,40,40,0.85)", "width": 0.4}})
    png_bytes = fig.to_image(
        format="png",
        width=_GIF_WORKER_STATE["w"],
        height=_GIF_WORKER_STATE["h"],
        scale=_GIF_WORKER_STATE["scale"],
    )
    return frame_idx, png_bytes


def _generate_distinct_colors(n: int) -> list[str]:
    """Generate n visually distinct RGB color strings using golden-ratio hue
    spacing combined with a small saturation/value cycle to maximize
    perceptual separation between neighboring indices."""
    import colorsys

    if n <= 0:
        return []
    golden_ratio_conjugate = 0.61803398875
    colors: list[str] = []
    h = 0.137  # arbitrary phase
    sv_pairs = [
        (0.85, 0.95),
        (0.65, 0.85),
        (0.95, 0.75),
        (0.55, 1.00),
        (0.80, 0.65),
    ]
    for i in range(n):
        s, v = sv_pairs[i % len(sv_pairs)]
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
        colors.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")
        h += golden_ratio_conjugate
    return colors


def _datetime_to_jd_fr(dt) -> tuple[float, float]:
    """Convert a Python datetime (UTC) to (jd, fr) suitable for sgp4."""
    from sgp4.api import jday

    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                dt.second + dt.microsecond * 1e-6)


def _parse_tle_file(path: str) -> list[tuple[str, str, str]]:
    """Parse a multi-TLE text file into a list of (name, line1, line2) tuples."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw_lines = [ln.rstrip("\r\n") for ln in fh if ln.strip()]
    tles: list[tuple[str, str, str]] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if line.startswith("1 ") and i + 1 < len(raw_lines) and raw_lines[i + 1].startswith("2 "):
            name = ""
            tles.append((name, line, raw_lines[i + 1]))
            i += 2
        elif (
            i + 2 < len(raw_lines)
            and raw_lines[i + 1].startswith("1 ")
            and raw_lines[i + 2].startswith("2 ")
        ):
            tles.append((line, raw_lines[i + 1], raw_lines[i + 2]))
            i += 3
        else:
            i += 1
    return tles


def _load_tle_satellites(
    backup_dir: str, target_jd: float, target_fr: float
) -> tuple[list[str], list]:
    """For each .txt file in backup_dir, pick the TLE whose epoch is closest
    to target_(jd+fr) and return parallel lists (sat_id, Satrec)."""
    from sgp4.api import Satrec

    if not os.path.isdir(backup_dir):
        raise FileNotFoundError(f"TLE backup directory not found: {backup_dir}")

    target_total = float(target_jd) + float(target_fr)
    sat_ids: list[str] = []
    satrecs: list = []
    skipped = 0
    for fname in sorted(os.listdir(backup_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(backup_dir, fname)
        try:
            tles = _parse_tle_file(path)
        except Exception:
            skipped += 1
            continue
        best: tuple[float, object] | None = None
        for _name, l1, l2 in tles:
            try:
                sat = Satrec.twoline2rv(l1, l2)
            except Exception:
                continue
            try:
                sat_total = float(sat.jdsatepoch) + float(sat.jdsatepochF)
            except Exception:
                continue
            dt = abs(sat_total - target_total)
            if best is None or dt < best[0]:
                best = (dt, sat)
        if best is None:
            skipped += 1
            continue
        sat_ids.append(fname)
        satrecs.append(best[1])
    if skipped:
        print(f"TLE loader: skipped {skipped} file(s) without usable TLEs.")
    print(f"TLE loader: loaded {len(satrecs)} satellites from {backup_dir}.")
    return sat_ids, satrecs


def _load_cluster_color_map(
    csv_path: str, sat_ids: list[str]
) -> tuple[list[str], list[int], int]:
    """Return (per-sat color string, per-sat global cluster id, num_clusters).
    Satellites missing from the CSV get a neutral light-gray color and id -1."""
    import pandas as pd

    df = pd.read_csv(csv_path, usecols=["sat_id", "global_cluster_id"])
    sat_to_cid = dict(zip(df["sat_id"].astype(str), df["global_cluster_id"].astype(int)))
    unique_ids = sorted(int(v) for v in df["global_cluster_id"].unique())
    n_unique = len(unique_ids)
    palette_n = max(n_unique, 103)  # honor user request for ~103 distinct colors
    palette = _generate_distinct_colors(palette_n)
    id_to_color = {cid: palette[i] for i, cid in enumerate(unique_ids)}
    colors: list[str] = []
    cids: list[int] = []
    missing = 0
    for sid in sat_ids:
        cid = sat_to_cid.get(str(sid))
        if cid is None:
            colors.append("rgb(180,180,180)")
            cids.append(-1)
            missing += 1
        else:
            colors.append(id_to_color[cid])
            cids.append(int(cid))
    print(
        f"Cluster colors: {n_unique} unique global clusters, palette of "
        f"{palette_n} colors; {missing} satellite(s) had no cluster assignment."
    )
    return colors, cids, n_unique


def _compute_camera_world_position(
    eye_xyz: tuple[float, float, float], bound: float
) -> np.ndarray:
    """Plotly's scene.camera.eye is given in normalized scene units where the
    cube spans [-1, 1]. Under aspectmode='cube' with axis range [-bound, bound]
    the world-coordinate camera position equals eye * bound."""
    return np.asarray(eye_xyz, dtype=np.float64) * float(bound)


def _compute_visible_mask(
    positions: np.ndarray, camera_world: np.ndarray, r_e: float
) -> np.ndarray:
    """Return a boolean mask of satellites NOT occluded by the Earth sphere
    when viewed from `camera_world`. Vectorized; positions has shape (N, 3)."""
    if positions.size == 0:
        return np.zeros(0, dtype=bool)
    d = positions - camera_world[None, :]
    a = np.einsum("ij,ij->i", d, d)
    b = 2.0 * (d @ camera_world)
    c = float(camera_world @ camera_world) - float(r_e) ** 2
    disc = b * b - 4.0 * a * c
    visible = np.ones(positions.shape[0], dtype=bool)
    hits = disc > 0.0
    if np.any(hits):
        sqrt_disc = np.sqrt(np.maximum(disc[hits], 0.0))
        t_min = (-b[hits] - sqrt_disc) / (2.0 * a[hits])
        # Sat is occluded iff sphere is hit between camera (t=0) and sat (t=1).
        occluded_local = (t_min > 0.0) & (t_min < 1.0)
        visible_idx = np.where(hits)[0]
        visible[visible_idx[occluded_local]] = False
    return visible


def _rgba_to_transparent_palette(rgba_img):
    """Convert an RGBA PIL image to mode 'P' with palette index 255 reserved
    for transparent pixels. Quantization uses 255 colors so the transparent
    slot is unused by any opaque pixel."""
    from PIL import Image

    if rgba_img.mode != "RGBA":
        rgba_img = rgba_img.convert("RGBA")
    alpha = rgba_img.split()[3]
    rgb = rgba_img.convert("RGB")
    palette_img = rgb.quantize(colors=255, method=2, dither=0)
    # Build a mask where alpha < 128 -> transparent.
    transparent_mask = alpha.point(lambda a: 255 if a < 128 else 0).convert("L")
    palette_img.paste(255, mask=transparent_mask)
    palette_img.info["transparency"] = 255
    return palette_img


def _save_looping_gif(
    frames: list,
    output_dir: str,
    filename: str,
    fps: float,
    dpi: int,
    transparent: bool = False,
) -> str | None:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover - tqdm is optional
        def tqdm(iterable=None, **_kwargs):  # type: ignore
            return iterable if iterable is not None else iter(())

    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        duration_ms = max(1, int(round(1000.0 / float(fps))))
        if transparent:
            quantized = [
                _rgba_to_transparent_palette(frame)
                for frame in tqdm(frames, desc="Quantizing (transparent)", unit="frame")
            ]
            save_kwargs = {
                "save_all": True,
                "append_images": quantized[1:],
                "duration": duration_ms,
                "loop": 0,
                "disposal": 2,
                "optimize": False,
                "dpi": (dpi, dpi),
                "transparency": 255,
            }
        else:
            quantized = [
                frame.quantize(colors=256, method=2, dither=0)
                for frame in tqdm(frames, desc="Quantizing", unit="frame")
            ]
            save_kwargs = {
                "save_all": True,
                "append_images": quantized[1:],
                "duration": duration_ms,
                "loop": 0,
                "disposal": 2,
                "optimize": False,
                "dpi": (dpi, dpi),
            }
        print(f"Encoding GIF ({len(quantized)} frames @ {duration_ms} ms/frame, loop=infinite)...")
        quantized[0].save(out_path, **save_kwargs)
        try:
            file_mb = os.path.getsize(out_path) / (1024.0 * 1024.0)
            print(f"Saved looping GIF: {out_path} ({file_mb:.2f} MB)")
        except Exception:
            print(f"Saved looping GIF: {out_path}")
        return out_path
    except Exception as exc:
        print(f"Warning: failed to save GIF '{filename}': {exc}")
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


def _kepler_eccentric_anomaly_from_mean_vec(
    mean_anomaly: np.ndarray,
    eccentricities: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-11,
) -> np.ndarray:
    """Vectorized Kepler solver supporting per-element eccentricities.

    `mean_anomaly` and `eccentricities` must broadcast to the same shape.
    """
    M = np.asarray(mean_anomaly, dtype=np.float64)
    e = np.asarray(eccentricities, dtype=np.float64)
    E = np.where(e < 0.8, M, np.full_like(M, np.pi))
    for _ in range(max_iter):
        f_val = E - e * np.sin(E) - M
        f_prime = 1.0 - e * np.cos(E)
        delta = f_val / np.maximum(f_prime, 1e-14)
        E = E - delta
        if np.nanmax(np.abs(delta)) < tol:
            break
    return E


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


def _build_textured_earth_surface_trace(
    r_e_val: float,
    texture_path: str | None,
    lon_count: int = 360,
    lat_count: int = 181,
):
    import plotly.graph_objects as go

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
                stop = i / 255.0
                colorscale.append([stop, color])

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


def _calibrate_camera_dlt(
    template_layout: dict,
    width_px: int,
    height_px: int,
    scale: float,
    bound: float,
):
    """Render a small calibration figure (no Earth, no orbits) with 8 known
    3D markers in distinct primary colors, detect each marker's pixel centroid
    by exact-color matching, and solve a 3x4 DLT projection matrix.

    Returns (P, (final_w, final_h)) so per-frame projection becomes a single
    matmul + perspective divide -- no Kaleido per frame.
    """
    import plotly.graph_objects as go
    from PIL import Image

    palette_rgb = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]
    color_strs = [f"rgb({r},{g},{b})" for (r, g, b) in palette_rgb]
    b = 0.85 * float(bound)  # keep markers comfortably inside the cube range
    pts = np.asarray(
        [
            (b, b, b),
            (-b, b, b),
            (b, -b, b),
            (b, b, -b),
            (-b, -b, b),
            (-b, b, -b),
            (b, -b, -b),
            (-b, -b, -b),
        ],
        dtype=np.float64,
    )

    cal_fig = go.Figure()
    cal_fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0].tolist(),
            y=pts[:, 1].tolist(),
            z=pts[:, 2].tolist(),
            mode="markers",
            marker={"size": 16, "color": color_strs, "opacity": 1.0,
                    "line": {"width": 0}},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    cal_fig.update_layout(**template_layout)
    img = _render_plotly_frame_to_pil(cal_fig, width_px, height_px, scale, keep_alpha=False)
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    H, W = arr.shape[:2]

    pixel_centers = []
    for (r, g, b_c) in palette_rgb:
        diff = np.abs(arr - np.array([r, g, b_c], dtype=np.int16))
        dist = diff.sum(axis=2)
        mask = dist < 60
        if not mask.any():
            return None, (W, H)
        ys, xs = np.where(mask)
        pixel_centers.append((float(xs.mean()), float(ys.mean())))

    A_rows = []
    rhs = []
    for (X, Y, Z), (u, v) in zip(pts, pixel_centers):
        A_rows.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z])
        rhs.append(u)
        A_rows.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z])
        rhs.append(v)
    A = np.asarray(A_rows, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    p, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    P = np.zeros((3, 4), dtype=np.float64)
    P[0, :] = p[0:4]
    P[1, :] = p[4:8]
    P[2, 0:3] = p[8:11]
    P[2, 3] = 1.0
    return P, (W, H)


def _project_world_to_pixels(P: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    """Project (N,3) world points -> (N,2) pixel coords via projection matrix P (3x4)."""
    if pts_world.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    homog = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
    proj = homog @ P.T  # (N,3)
    w = proj[:, 2]
    w_safe = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return proj[:, :2] / w_safe[:, None]


def _hex_or_rgb_to_tuple(c: str) -> tuple[int, int, int]:
    s = c.strip()
    if s.startswith("rgb"):
        nums = s[s.index("(") + 1 : s.index(")")].split(",")
        return (int(float(nums[0])), int(float(nums[1])), int(float(nums[2])))
    if s.startswith("#"):
        s = s[1:]
        if len(s) == 3:
            return (int(s[0] * 2, 16), int(s[1] * 2, 16), int(s[2] * 2, 16))
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    return (200, 200, 200)


def _export_tle_realtime_gif(
    sat_ids: list[str],
    satrecs: list,
    sat_colors: list[str],
    target_epoch_dt,
    duration_s: float,
    output_dir: str,
    output_stem: str,
    r_e_val: float,
    texture_path: str | None,
    gif_fps: float,
    gif_dpi: int,
    gif_width_in: float,
    gif_height_in: float,
    gif_marker_size: float,
    gif_earth_grid: int,
    gif_workers: int,
    gif_transparent: bool,
    gif_frame_timeout_s: float,
    gif_frame_retries: int,
) -> None:
    """Render a real-time GIF of TLE-propagated satellites colored by global
    cluster id. Faint orbit polylines included.

    Fast path: vectorized SGP4 (SatrecArray) + render Earth/orbits ONCE via
    Plotly + per-frame markers drawn directly with PIL using a calibrated
    perspective camera matrix. No Kaleido in the per-frame loop.
    """
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError("Plotly is required for GIF animation export.") from exc

    from PIL import Image, ImageDraw

    _silence_kaleido_noise()

    n_sats = len(satrecs)
    if n_sats == 0:
        raise RuntimeError("No satellites available for the TLE-backup animation.")

    n_frames = max(2, int(round(float(duration_s) * float(gif_fps))))
    print(
        f"TLE realtime GIF (fast PIL path): {n_sats} satellites, "
        f"duration {duration_s:.1f} s @ {gif_fps:.1f} fps -> {n_frames} frames "
        f"(1:1 real-time)."
    )

    target_jd, target_fr = _datetime_to_jd_fr(target_epoch_dt)

    # ---------- Vectorized SGP4 propagation (all sats x all frames) ----------
    from sgp4.api import SatrecArray

    dt_s_per_frame = float(duration_s) / float(n_frames)
    jd_arr = np.full(n_frames, float(target_jd), dtype=np.float64)
    fr_arr = float(target_fr) + (np.arange(n_frames, dtype=np.float64) * dt_s_per_frame) / 86400.0
    sat_array = SatrecArray(list(satrecs))
    import time as _time
    t0 = _time.perf_counter()
    err_arr, r_arr, _v_arr = sat_array.sgp4(jd_arr, fr_arr)  # (M, K, ...) shapes
    print(
        f"Vectorized SGP4: propagated {n_sats}*{n_frames}="
        f"{n_sats * n_frames:,} states in {_time.perf_counter() - t0:.2f}s."
    )
    # err_arr: (M, K); r_arr: (M, K, 3). Transpose to (K, M, 3).
    pos_grid = np.transpose(r_arr, (1, 0, 2))
    err_grid = np.transpose(err_arr, (1, 0))
    bad = err_grid != 0
    if bad.any():
        pos_grid[bad] = np.nan
        print(f"TLE realtime GIF: {int(bad.sum())} SGP4 errors masked out.")

    # ---------- Scene/camera bounds ----------
    finite = np.isfinite(pos_grid[..., 0])
    norms = np.linalg.norm(np.where(finite[..., None], pos_grid, 0.0), axis=-1)
    max_norm = float(np.nanmax(norms)) if norms.size else float(r_e_val)
    bound = float(max(r_e_val, max_norm)) * 1.08
    axis_layout = {
        "visible": False,
        "showgrid": False,
        "zeroline": False,
        "showticklabels": False,
        "showbackground": False,
        "range": [-bound, bound],
    }
    lon_count = max(90, int(gif_earth_grid))
    lat_count = lon_count // 2 + 1
    camera_eye = (1.55, 1.55, 0.95)
    scene_layout = {
        "xaxis": axis_layout,
        "yaxis": axis_layout,
        "zaxis": axis_layout,
        "aspectmode": "cube",
        "bgcolor": "rgba(0,0,0,0)",
        "camera": {"eye": {"x": camera_eye[0], "y": camera_eye[1], "z": camera_eye[2]}},
    }
    common_layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "width": DEFAULT_LAYOUT_SIZE_PX,
        "height": DEFAULT_LAYOUT_SIZE_PX,
        "scene": scene_layout,
    }

    # ---------- Static base figure: Earth + faint orbit polylines ----------
    base_fig = go.Figure()
    base_fig.add_trace(
        _build_textured_earth_surface_trace(
            r_e_val=r_e_val,
            texture_path=texture_path,
            lon_count=lon_count,
            lat_count=lat_count,
        )
    )

    orbit_groups: dict[str, dict[str, list[float]]] = {}
    points_per_orbit = 180
    for sat, color_str in zip(satrecs, sat_colors):
        try:
            n_rad_per_min = float(sat.no_kozai)
            n_rad_per_s = n_rad_per_min / 60.0
            if n_rad_per_s <= 0.0:
                continue
            sma_km_i = (MU_EARTH_KM3_S2 / (n_rad_per_s ** 2)) ** (1.0 / 3.0)
            xyz, _period = _propagate_one_period_kepler_xyz(
                mu_val=MU_EARTH_KM3_S2,
                sma_km=sma_km_i,
                ecc_val=float(sat.ecco),
                inc_rad=float(sat.inclo),
                aop_rad=float(sat.argpo),
                raan_rad=float(sat.nodeo),
                ta0_rad=_true_from_mean_anomaly(float(sat.mo), float(sat.ecco)),
                points_per_orbit=points_per_orbit,
            )
        except Exception:
            continue
        if xyz is None:
            continue
        bucket = orbit_groups.setdefault(color_str, {"x": [], "y": [], "z": []})
        bucket["x"].extend(xyz[:, 0].tolist()); bucket["x"].append(np.nan)
        bucket["y"].extend(xyz[:, 1].tolist()); bucket["y"].append(np.nan)
        bucket["z"].extend(xyz[:, 2].tolist()); bucket["z"].append(np.nan)

    orbit_alpha = 0.05
    for color_str, bucket in orbit_groups.items():
        r, g, b = _hex_or_rgb_to_tuple(color_str)
        rgba = f"rgba({r},{g},{b},{orbit_alpha})"
        base_fig.add_trace(
            go.Scatter3d(
                x=bucket["x"],
                y=bucket["y"],
                z=bucket["z"],
                mode="lines",
                line={"color": rgba, "width": 1.0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    base_fig.update_layout(**common_layout)

    width_px = max(1, int(round(gif_width_in * 96.0)))
    height_px = max(1, int(round(gif_height_in * 96.0)))
    scale = float(gif_dpi) / 96.0
    final_w = int(round(width_px * scale))
    final_h = int(round(height_px * scale))

    print(
        f"Rendering Earth+orbits base ({final_w}x{final_h} px @ {gif_dpi} dpi) once via Plotly..."
    )
    t0 = _time.perf_counter()
    base_rgba = _render_plotly_frame_to_pil(
        base_fig, width_px, height_px, scale, keep_alpha=True
    )
    print(f"Base render: {_time.perf_counter() - t0:.2f}s")
    if base_rgba.size != (final_w, final_h):
        base_rgba = base_rgba.resize((final_w, final_h))
    if not gif_transparent:
        white_bg = Image.new("RGBA", base_rgba.size, (255, 255, 255, 255))
        base_rgba = Image.alpha_composite(white_bg, base_rgba)

    # ---------- Calibrate camera (one Plotly render) -> 3x4 DLT matrix ----------
    print("Calibrating camera projection (single Plotly render of marker grid)...")
    t0 = _time.perf_counter()
    P_proj, (cal_w, cal_h) = _calibrate_camera_dlt(
        template_layout=common_layout,
        width_px=width_px,
        height_px=height_px,
        scale=scale,
        bound=bound,
    )
    if P_proj is None:
        raise RuntimeError("Camera calibration failed (couldn't detect markers).")
    # Calibration was rendered at the same size as base, so pixel coords match.
    if (cal_w, cal_h) != base_rgba.size:
        # Rescale projection to base image size in case Kaleido produced a
        # slightly different output size.
        sx = base_rgba.size[0] / float(cal_w)
        sy = base_rgba.size[1] / float(cal_h)
        S = np.diag([sx, sy, 1.0])
        P_proj = S @ P_proj
    print(f"Calibration: {_time.perf_counter() - t0:.2f}s")

    camera_world = _compute_camera_world_position(camera_eye, bound)

    # ---------- Per-frame: project + occlude + draw markers via PIL ----------
    sat_color_tuples = [_hex_or_rgb_to_tuple(c) for c in sat_colors]
    marker_radius_px = max(1, int(round(float(gif_marker_size) * scale * 0.55)))

    print(
        f"Rendering {n_frames} frames via PIL (marker radius {marker_radius_px}px)..."
    )
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        def tqdm(iterable=None, **_kwargs):  # type: ignore
            return iterable if iterable is not None else iter(())

    # Stream frames straight to MP4 via imageio-ffmpeg. This avoids holding
    # all RGBA frames in memory (a 600 dpi 4MP * 3600 frame run would be
    # ~83 GB). MP4 (H.264, yuv420p) gives lossless-looking compression at
    # tens of MB instead of GBs.
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        import imageio  # type: ignore

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{output_stem}.mp4")

    base_rgb_static = base_rgba.convert("RGB")
    writer = imageio.get_writer(
        out_path,
        fps=float(gif_fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    t0 = _time.perf_counter()
    try:
        for i in tqdm(range(n_frames), desc="Rendering+encoding", unit="frame"):
            positions = pos_grid[i]
            finite_mask = np.isfinite(positions[:, 0])
            idx_finite = np.where(finite_mask)[0]
            if idx_finite.size == 0:
                writer.append_data(np.asarray(base_rgb_static))
                continue
            pos_finite = positions[idx_finite]
            vis_mask = _compute_visible_mask(pos_finite, camera_world, r_e_val)
            idx_vis = idx_finite[vis_mask]
            if idx_vis.size == 0:
                writer.append_data(np.asarray(base_rgb_static))
                continue
            vis_pts = positions[idx_vis]
            uv = _project_world_to_pixels(P_proj, vis_pts)

            frame = base_rgb_static.copy()
            draw = ImageDraw.Draw(frame)
            rad = marker_radius_px
            for (u, v), sat_idx in zip(uv, idx_vis):
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                color = sat_color_tuples[int(sat_idx)]
                draw.ellipse(
                    (u - rad, v - rad, u + rad, v + rad),
                    fill=color,
                    outline=(40, 40, 40),
                    width=1,
                )
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    elapsed = _time.perf_counter() - t0
    print(
        f"Render+encode: {elapsed:.2f}s "
        f"({elapsed / max(n_frames, 1) * 1000.0:.1f} ms/frame)."
    )
    try:
        size_mb = os.path.getsize(out_path) / (1024.0 * 1024.0)
        print(f"Saved MP4: {out_path} ({size_mb:.2f} MB)")
    except Exception:
        print(f"Saved MP4: {out_path}")



def _export_constellation_gif(
    constellation_orbits: dict[str, np.ndarray | list[str]],
    output_dir: str,
    output_stem: str,
    r_e_val: float,
    mu_val: float,
    points_per_orbit: int,
    texture_path: str | None,
    gif_frames: int,
    gif_fps: float,
    gif_dpi: int,
    gif_width_in: float,
    gif_height_in: float,
    gif_line_width: float = 2.0,
    gif_marker_size: float = 3.0,
    gif_earth_grid: int = 720,
    gif_loop_duration_s: float = 0.0,
    gif_workers: int = 0,
    gif_overlay_color: str = "#FFFFFF",
    gif_orbit_color: str = "rgba(200,200,200,0.75)",
    gif_transparent: bool = True,
    gif_frame_timeout_s: float = 60.0,
    gif_frame_retries: int = 3,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError("Plotly is required for GIF animation export.") from exc

    sma_km = np.asarray(constellation_orbits["sma_km"], dtype=np.float64)
    ecc = np.asarray(constellation_orbits["ecc"], dtype=np.float64)
    inc_rad = np.asarray(constellation_orbits["inc_rad"], dtype=np.float64)
    aop_rad = np.asarray(constellation_orbits["aop_rad"], dtype=np.float64)
    raan_rad = np.asarray(constellation_orbits["raan_rad"], dtype=np.float64)
    ta_rad = np.asarray(constellation_orbits["ta_rad"], dtype=np.float64)

    if len(sma_km) == 0:
        raise ValueError("No satellites were generated.")

    _silence_kaleido_noise()

    orbit_line_x: list[float] = []
    orbit_line_y: list[float] = []
    orbit_line_z: list[float] = []

    valid_indices: list[int] = []
    mean_anomaly_0_per_sat: list[float] = []
    period_per_sat: list[float] = []

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

        e_safe = float(np.clip(float(ecc[idx]), 0.0, 1.0 - 1e-12))
        ta0 = float(ta_rad[idx])
        eccentric_anomaly_0 = 2.0 * np.arctan2(
            np.sqrt(1.0 - e_safe) * np.sin(ta0 / 2.0),
            np.sqrt(1.0 + e_safe) * np.cos(ta0 / 2.0),
        )
        mean_anomaly_0 = float(np.mod(eccentric_anomaly_0 - e_safe * np.sin(eccentric_anomaly_0), 2.0 * np.pi))
        mean_anomaly_0_per_sat.append(mean_anomaly_0)
        period_per_sat.append(float(period_s))
        valid_indices.append(idx)

    if not valid_indices:
        raise RuntimeError("No valid orbit states were available for the GIF animation.")

    sat_count = len(valid_indices)
    sma_v = sma_km[valid_indices]
    ecc_v = np.clip(ecc[valid_indices], 0.0, 1.0 - 1e-12)
    inc_v = inc_rad[valid_indices]
    aop_v = aop_rad[valid_indices]
    raan_v = raan_rad[valid_indices]
    mean_anomaly_0 = np.asarray(mean_anomaly_0_per_sat, dtype=np.float64)
    periods = np.asarray(period_per_sat, dtype=np.float64)

    # Realistic timing: pick a loop duration in simulated seconds and let each
    # satellite advance through an integer number of revolutions per loop equal
    # to round(loop_duration / period_i). This preserves the true relative
    # speeds (low altitude => fast, high altitude => slow) while keeping the
    # animation perfectly seamless. When all periods agree (Walker shells),
    # everyone completes exactly one orbit per loop at realistic speed.
    if gif_loop_duration_s > 0.0:
        loop_duration_s = float(gif_loop_duration_s)
    else:
        loop_duration_s = float(np.max(periods))
    revs_per_loop = np.maximum(1.0, np.round(loop_duration_s / periods))
    period_min_s = float(np.min(periods))
    period_max_s = float(np.max(periods))
    period_spread = (period_max_s - period_min_s) / period_max_s if period_max_s > 0.0 else 0.0
    print(
        f"GIF timing: loop_duration={loop_duration_s:.1f} s "
        f"({loop_duration_s / 60.0:.2f} min), period range "
        f"{period_min_s / 60.0:.2f}-{period_max_s / 60.0:.2f} min "
        f"({period_spread * 100.0:.2f}% spread), "
        f"revs/loop range {int(revs_per_loop.min())}-{int(revs_per_loop.max())}."
    )

    cos_w = np.cos(aop_v)
    sin_w = np.sin(aop_v)
    cos_raan = np.cos(raan_v)
    sin_raan = np.sin(raan_v)
    cos_inc = np.cos(inc_v)
    sin_inc = np.sin(inc_v)

    r11 = cos_raan * cos_w - sin_raan * sin_w * cos_inc
    r12 = -cos_raan * sin_w - sin_raan * cos_w * cos_inc
    r21 = sin_raan * cos_w + cos_raan * sin_w * cos_inc
    r22 = -sin_raan * sin_w + cos_raan * cos_w * cos_inc
    r31 = sin_w * sin_inc
    r32 = cos_w * sin_inc

    apoapsis_km = sma_v * (1.0 + ecc_v)
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

    lon_count = max(90, int(gif_earth_grid))
    lat_count = lon_count // 2 + 1

    camera_eye = (1.55, 1.55, 0.95)
    scene_layout = {
        "xaxis": axis_layout,
        "yaxis": axis_layout,
        "zaxis": axis_layout,
        "aspectmode": "cube",
        "bgcolor": "rgba(0,0,0,0)",
        "camera": {"eye": {"x": camera_eye[0], "y": camera_eye[1], "z": camera_eye[2]}},
    }
    common_layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "width": DEFAULT_LAYOUT_SIZE_PX,
        "height": DEFAULT_LAYOUT_SIZE_PX,
        "scene": scene_layout,
    }

    # Static base figure: Earth surface + orbit polylines. Rendered ONCE at
    # the final pixel size, then composited under each per-frame marker
    # overlay. This removes the dominant per-frame Kaleido cost (re-shipping
    # the textured Earth surface JSON to Chromium for every frame).
    base_fig = go.Figure()
    base_fig.add_trace(
        _build_textured_earth_surface_trace(
            r_e_val=r_e_val,
            texture_path=texture_path,
            lon_count=lon_count,
            lat_count=lat_count,
        )
    )
    base_fig.add_trace(
        go.Scatter3d(
            x=orbit_line_x,
            y=orbit_line_y,
            z=orbit_line_z,
            mode="lines",
            line={"color": gif_orbit_color, "width": float(gif_line_width)},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    base_fig.update_layout(**common_layout)

    # Lightweight per-frame figure: single Scatter3d trace with the
    # satellite markers, transparent background, identical scene/camera.
    overlay_fig = go.Figure()
    overlay_fig.add_trace(
        go.Scatter3d(
            x=[0.0] * sat_count,
            y=[0.0] * sat_count,
            z=[0.0] * sat_count,
            mode="markers",
            marker={
                "size": float(gif_marker_size),
                "color": gif_overlay_color,
                "opacity": 1.0,
                "line": {"color": "rgba(80,80,80,0.85)", "width": 0.5},
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    overlay_fig.update_layout(**common_layout)

    width_px = max(1, int(round(gif_width_in * 96.0)))
    height_px = max(1, int(round(gif_height_in * 96.0)))
    scale = float(gif_dpi) / 96.0

    # Frame fraction in [0, 1) so frame N == frame 0 (seamless loop).
    frame_fractions = np.arange(gif_frames, dtype=np.float64) / float(gif_frames)

    # Vectorize all per-frame satellite positions in a single shot.
    # Shape: (n_frames, n_sats).
    mean_anom_grid = np.mod(
        mean_anomaly_0[None, :]
        + 2.0 * np.pi * revs_per_loop[None, :] * frame_fractions[:, None],
        2.0 * np.pi,
    )
    ecc_grid = np.broadcast_to(ecc_v[None, :], mean_anom_grid.shape)
    eccentric_grid = _kepler_eccentric_anomaly_from_mean_vec(mean_anom_grid, ecc_grid)
    true_anom_grid = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc_grid) * np.sin(eccentric_grid / 2.0),
        np.sqrt(1.0 - ecc_grid) * np.cos(eccentric_grid / 2.0),
    )
    radius_grid = sma_v[None, :] * (1.0 - ecc_grid * np.cos(eccentric_grid))
    x_pf_grid = radius_grid * np.cos(true_anom_grid)
    y_pf_grid = radius_grid * np.sin(true_anom_grid)
    x_eci_grid = r11[None, :] * x_pf_grid + r12[None, :] * y_pf_grid
    y_eci_grid = r21[None, :] * x_pf_grid + r22[None, :] * y_pf_grid
    z_eci_grid = r31[None, :] * x_pf_grid + r32[None, :] * y_pf_grid

    # ---------------- Static-base + overlay compositing setup ----------------
    # Render Earth+orbits ONCE at the final pixel size; per-frame work then
    # only pushes ~sat_count markers through Kaleido. Cull satellites
    # occluded by the Earth sphere using an analytic ray/sphere test so
    # depth ordering matches a full 3D render.
    final_w = int(round(width_px * scale))
    final_h = int(round(height_px * scale))
    try:
        base_rgba = _render_plotly_frame_to_pil(
            base_fig, width_px, height_px, scale, keep_alpha=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to render the static Earth/orbit base frame: {exc}. "
            "Static Plotly image export typically requires the 'kaleido' package."
        )
    if base_rgba.size != (final_w, final_h):
        base_rgba = base_rgba.resize((final_w, final_h))
    if not gif_transparent:
        # Pre-composite onto white so per-frame composite stays cheap.
        from PIL import Image as _PILImage

        white_bg = _PILImage.new("RGBA", base_rgba.size, (255, 255, 255, 255))
        base_rgba = _PILImage.alpha_composite(white_bg, base_rgba)

    camera_world = _compute_camera_world_position(camera_eye, bound)

    # Resolve worker count.
    if gif_workers <= 0:
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1
        worker_count = max(1, min(8, cpu_count, gif_frames))
    else:
        worker_count = max(1, min(int(gif_workers), gif_frames))

    print(
        f"Rendering {gif_frames} GIF frames at "
        f"{final_w}x{final_h} px ({gif_dpi} dpi, "
        f"{gif_fps:.1f} fps, loop {gif_frames / gif_fps:.2f} s) "
        f"using {worker_count} worker process{'es' if worker_count != 1 else ''} "
        f"[static-base compositing]..."
    )

    # Build per-frame payloads containing only the VISIBLE satellite positions
    # for that frame (analytic occlusion against the Earth sphere).
    payloads: list = []
    for frame_idx in range(gif_frames):
        positions = np.column_stack(
            (x_eci_grid[frame_idx], y_eci_grid[frame_idx], z_eci_grid[frame_idx])
        )
        mask = _compute_visible_mask(positions, camera_world, r_e_val)
        if np.any(mask):
            visible = positions[mask]
            xs = visible[:, 0].tolist()
            ys = visible[:, 1].tolist()
            zs = visible[:, 2].tolist()
        else:
            # Plotly Scatter3d needs at least one point; place it at the
            # camera (always occluded by definition... but here we just feed
            # an off-screen point that will be invisible past the axis range).
            xs = [float(2.0 * bound)]
            ys = [float(2.0 * bound)]
            zs = [float(2.0 * bound)]
        payloads.append((frame_idx, xs, ys, zs))

    if worker_count == 1:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            def tqdm(iterable=None, **_kwargs):  # type: ignore
                return iterable if iterable is not None else iter(())

        from PIL import Image as _PILImage

        frames_pil: list = []
        marker_trace = overlay_fig.data[0]
        for payload in tqdm(payloads, desc="Rendering (serial)", unit="frame"):
            frame_idx, x_list, y_list, z_list = payload
            marker_trace.update(x=x_list, y=y_list, z=z_list)
            try:
                overlay = _render_plotly_frame_to_pil(
                    overlay_fig, width_px, height_px, scale, keep_alpha=True
                )
            except Exception as exc:
                print(
                    f"Warning: failed to render GIF frame {frame_idx + 1}/{gif_frames}: {exc}. "
                    "Static Plotly image export typically requires the 'kaleido' package."
                )
                return
            if overlay.size != base_rgba.size:
                overlay = overlay.resize(base_rgba.size)
            composed = _PILImage.alpha_composite(base_rgba, overlay)
            if not gif_transparent:
                composed = composed.convert("RGB")
            frames_pil.append(composed)
    else:
        try:
            import io
            from concurrent.futures import (
                FIRST_COMPLETED,
                ProcessPoolExecutor,
                wait,
            )

            from PIL import Image
        except Exception as exc:
            raise RuntimeError("Parallel GIF rendering requires concurrent.futures and PIL.") from exc

        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            def tqdm(iterable=None, **_kwargs):  # type: ignore
                return iterable if iterable is not None else iter(())

        # Worker uses the lightweight overlay figure (1 trace, no Earth/orbits).
        fig_dict = overlay_fig.to_dict()
        results: list = [None] * gif_frames

        per_frame_timeout = float(gif_frame_timeout_s) if gif_frame_timeout_s > 0 else None
        max_attempts = max(1, int(gif_frame_retries) + 1)
        remaining = list(range(gif_frames))
        attempt = 0
        completed_total = 0
        progress = tqdm(
            total=gif_frames,
            desc=f"Rendering ({worker_count} workers)",
            unit="frame",
            smoothing=0.1,
        )
        try:
            while remaining and attempt < max_attempts:
                attempt += 1
                if attempt > 1:
                    print(
                        f"\nRetry attempt {attempt}/{max_attempts} for "
                        f"{len(remaining)} stalled/failed frame(s)..."
                    )
                executor = ProcessPoolExecutor(
                    max_workers=min(worker_count, len(remaining)),
                    initializer=_gif_worker_init,
                    initargs=(fig_dict, width_px, height_px, scale),
                )
                stalled = False
                try:
                    future_to_idx = {
                        executor.submit(_gif_worker_render, payloads[i]): i
                        for i in remaining
                    }
                    pending = set(future_to_idx)
                    while pending:
                        done, pending = wait(
                            pending,
                            timeout=per_frame_timeout,
                            return_when=FIRST_COMPLETED,
                        )
                        if not done:
                            print(
                                f"\nWarning: no frame completed within "
                                f"{per_frame_timeout:.0f}s; restarting worker pool "
                                f"({len(pending)} frame(s) still pending)."
                            )
                            stalled = True
                            break
                        for fut in done:
                            idx = future_to_idx[fut]
                            try:
                                frame_idx, png_bytes = fut.result()
                                results[frame_idx] = png_bytes
                                completed_total += 1
                                progress.update(1)
                            except Exception as exc:
                                print(
                                    f"\nWarning: frame {idx} failed: {exc}"
                                )
                finally:
                    try:
                        executor.shutdown(wait=not stalled, cancel_futures=True)
                    except TypeError:
                        # Python < 3.9 fallback
                        executor.shutdown(wait=not stalled)
                remaining = [i for i in range(gif_frames) if results[i] is None]
        finally:
            progress.close()

        if remaining:
            print(
                f"Warning: {len(remaining)} frame(s) failed to render after "
                f"{attempt} attempt(s); they will be skipped."
            )
            if completed_total == 0:
                print(
                    "Static Plotly image export typically requires the 'kaleido' package."
                )
                return

        # Composite each transparent overlay over the prerendered Earth/orbit base.
        frames_pil = []
        target_mode = "RGBA" if gif_transparent else "RGB"
        for png_bytes in tqdm(
            [r for r in results if r is not None],
            desc="Compositing",
            unit="frame",
        ):
            overlay = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            if overlay.size != base_rgba.size:
                overlay = overlay.resize(base_rgba.size)
            composed = Image.alpha_composite(base_rgba, overlay)
            if target_mode == "RGB":
                composed = composed.convert("RGB")
            frames_pil.append(composed)
        if len(frames_pil) != gif_frames:
            print(
                f"Warning: only {len(frames_pil)}/{gif_frames} frames rendered successfully."
            )
            if not frames_pil:
                return

    _save_looping_gif(
        frames=frames_pil,
        output_dir=output_dir,
        filename=f"{output_stem}.gif",
        fps=gif_fps,
        dpi=gif_dpi,
        transparent=gif_transparent,
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

    def _process(orbits: dict, stem: str) -> None:
        if not args.gif_only:
            _plot_constellation_orbits(
                constellation_orbits=orbits,
                output_dir=args.output_dir,
                output_stem=stem,
                r_e_val=R_EARTH_KM,
                mu_val=MU_EARTH_KM3_S2,
                points_per_orbit=args.points_per_orbit,
                texture_path=texture_path,
                export_png=not args.no_png,
                png_dpi=args.png_dpi,
            )
        if args.gif:
            _export_constellation_gif(
                constellation_orbits=orbits,
                output_dir=args.output_dir,
                output_stem=stem,
                r_e_val=R_EARTH_KM,
                mu_val=MU_EARTH_KM3_S2,
                points_per_orbit=args.points_per_orbit,
                texture_path=texture_path,
                gif_frames=int(args.gif_frames),
                gif_fps=float(args.gif_fps),
                gif_dpi=int(args.gif_dpi),
                gif_width_in=float(args.gif_width_in),
                gif_height_in=float(args.gif_height_in),
                gif_line_width=float(args.gif_line_width),
                gif_marker_size=float(args.gif_marker_size),
                gif_earth_grid=int(args.gif_earth_grid),
                gif_loop_duration_s=float(args.gif_loop_duration_s),
                gif_workers=int(args.gif_workers),
                gif_overlay_color=str(args.gif_overlay_color),
                gif_orbit_color=str(args.gif_orbit_color),
                gif_transparent=bool(args.gif_transparent),
                gif_frame_timeout_s=float(args.gif_frame_timeout),
                gif_frame_retries=int(args.gif_frame_retries),
            )

    if args.constellation == "raan-shear":
        for constellation_orbits, output_stem in _generate_raan_shear_snapshots(args):
            _process(constellation_orbits, output_stem)
        return

    if args.constellation == "tle-backup":
        from datetime import datetime, timezone

        target_dt = datetime.fromisoformat(args.tle_target_epoch)
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=timezone.utc)
        target_jd, target_fr = _datetime_to_jd_fr(target_dt)
        sat_ids, satrecs = _load_tle_satellites(
            backup_dir=args.tle_backup_dir,
            target_jd=target_jd,
            target_fr=target_fr,
        )
        sat_colors, _cids, _n_clusters = _load_cluster_color_map(
            csv_path=args.tle_cluster_csv,
            sat_ids=sat_ids,
        )
        stem = (
            f"tle_backup_{target_dt.strftime('%Y%m%dT%H%M%S')}_"
            f"{int(round(args.tle_realtime_duration_s))}s"
        )
        _export_tle_realtime_gif(
            sat_ids=sat_ids,
            satrecs=satrecs,
            sat_colors=sat_colors,
            target_epoch_dt=target_dt,
            duration_s=float(args.tle_realtime_duration_s),
            output_dir=args.output_dir,
            output_stem=stem,
            r_e_val=R_EARTH_KM,
            texture_path=texture_path,
            gif_fps=float(args.gif_fps),
            gif_dpi=int(args.gif_dpi),
            gif_width_in=float(args.gif_width_in),
            gif_height_in=float(args.gif_height_in),
            gif_marker_size=float(args.gif_marker_size),
            gif_earth_grid=int(args.gif_earth_grid),
            gif_workers=int(args.gif_workers),
            gif_transparent=bool(args.gif_transparent),
            gif_frame_timeout_s=float(args.gif_frame_timeout),
            gif_frame_retries=int(args.gif_frame_retries),
        )
        return

    constellation_orbits, output_stem = _generate_constellation_orbits(args)
    _process(constellation_orbits, output_stem)


if __name__ == "__main__":
    main()
