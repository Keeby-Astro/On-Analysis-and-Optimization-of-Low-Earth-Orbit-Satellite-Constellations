"""Generic capture of matplotlib figure data to CSV + JSON manifest.

Layer 1 of the plot-data export pipeline.  Walks every currently open
``matplotlib`` figure and serialises every supported artist (``Line2D``,
``PathCollection``, ``QuadMesh``, ``AxesImage``, ``PolyCollection``, bar
``Rectangle`` containers) into one CSV per artist plus a per-figure JSON
descriptor.  A top-level ``manifest.json`` lists every figure that was
captured so a downstream script can reconstruct the plots without rerunning
the analytical pipeline.

The module is intentionally module-agnostic and does not require any change
to the individual plotting functions: capture happens after the figures have
been built, while they are still open.

CSV files larger than ``compress_threshold_bytes`` are transparently
recompressed with ``zstandard`` (if installed); otherwise they are left as
plain ``.csv``.  The replot script auto-detects either form.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import (
    LineCollection,
    PathCollection,
    PolyCollection,
    QuadMesh,
)
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


try:  # optional dependency
    import zstandard as _zstd  # type: ignore
except Exception:  # pragma: no cover - optional
    _zstd = None


DEFAULT_COMPRESS_THRESHOLD_BYTES = 256 * 1024  # 256 KB
MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sanitize(text: Any, fallback: str = "untitled", max_len: int = 80) -> str:
    s = re.sub(r"\s+", "_", str(text or "").strip())
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    s = s.strip("._-")
    return (s or fallback)[:max_len]


def _figure_label(fig, ordinal: int) -> str:
    label = str(fig.get_label() or "").strip()
    if label:
        return label
    for ax in fig.get_axes():
        title = str(ax.get_title() or "").strip()
        if title:
            return title
        sup = getattr(fig, "_suptitle", None)
        if sup is not None:
            t = str(sup.get_text() or "").strip()
            if t:
                return t
    return f"figure_{ordinal:03d}"


def _color_to_hex(c) -> Optional[str]:
    try:
        return matplotlib.colors.to_hex(c, keep_alpha=False)
    except Exception:
        return None


def _is_datetime_array(arr: np.ndarray) -> bool:
    if arr.dtype.kind == "M":
        return True
    # Matplotlib stores datetimes as floats (days since epoch) once drawn;
    # we cannot reliably detect those without axis context.
    return False


def _serialise_xy(x: np.ndarray, y: np.ndarray, axis_x_is_date: bool) -> pd.DataFrame:
    if axis_x_is_date:
        try:
            x = matplotlib.dates.num2date(np.asarray(x, dtype=float))
            x = pd.to_datetime([t.replace(tzinfo=None) for t in x])
        except Exception:
            pass
    return pd.DataFrame({"x": x, "y": y})


def _axis_x_is_date(ax) -> bool:
    try:
        return isinstance(ax.xaxis.converter, matplotlib.dates.DateConverter)
    except Exception:
        # Fallback: inspect major formatter
        try:
            return isinstance(ax.xaxis.get_major_formatter(), matplotlib.dates.DateFormatter)
        except Exception:
            return False


def _axis_y_is_date(ax) -> bool:
    try:
        return isinstance(ax.yaxis.converter, matplotlib.dates.DateConverter)
    except Exception:
        try:
            return isinstance(ax.yaxis.get_major_formatter(), matplotlib.dates.DateFormatter)
        except Exception:
            return False


def _safe(v: Any) -> Any:
    """Coerce numpy / pandas scalars to JSON-friendly values."""
    if v is None:
        return None
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.ndarray,)):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    return v


def _write_csv(
    df: pd.DataFrame,
    path: str,
    compress_threshold_bytes: int,
) -> str:
    """Write ``df`` to ``path`` (.csv).  If the resulting file exceeds the
    threshold and ``zstandard`` is available, recompress to ``.csv.zst`` and
    drop the plain CSV.  Returns the final on-disk filename (basename)."""
    df.to_csv(path, index=False)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    if size > compress_threshold_bytes and _zstd is not None:
        zst_path = path + ".zst"
        try:
            cctx = _zstd.ZstdCompressor(level=10)
            with open(path, "rb") as f_in, open(zst_path, "wb") as f_out:
                cctx.copy_stream(f_in, f_out)
            os.remove(path)
            return os.path.basename(zst_path)
        except Exception:
            # Compression failed — keep plain CSV
            if os.path.exists(zst_path):
                try:
                    os.remove(zst_path)
                except OSError:
                    pass
    return os.path.basename(path)


# --------------------------------------------------------------------------- #
# Per-artist serialisers
# --------------------------------------------------------------------------- #
def _serialise_line2d(
    artist: Line2D,
    ax,
    out_dir: str,
    base_name: str,
    compress_threshold_bytes: int,
) -> Optional[Dict[str, Any]]:
    # Detect axhline / axvline: their transform is a blended axes-fraction /
    # data transform, so xdata/ydata cannot be interpreted directly in data
    # coordinates.  Capture the constant data-coordinate value instead.
    try:
        if artist.get_transform() is ax.get_yaxis_transform():
            # axhline: ydata is in data coords (constant), xdata is axes-fraction.
            yd = np.asarray(artist.get_ydata(), dtype=float)
            if yd.size and np.all(yd == yd[0]):
                return {
                    "type": "axhline",
                    "y": float(yd[0]),
                    "label": str(artist.get_label() or ""),
                    "color": _color_to_hex(artist.get_color()),
                    "linestyle": str(artist.get_linestyle()),
                    "linewidth": _safe(artist.get_linewidth()),
                    "alpha": _safe(artist.get_alpha()),
                    "zorder": _safe(artist.get_zorder()),
                }
        if artist.get_transform() is ax.get_xaxis_transform():
            xd = np.asarray(artist.get_xdata(), dtype=float)
            if xd.size and np.all(xd == xd[0]):
                return {
                    "type": "axvline",
                    "x": float(xd[0]),
                    "label": str(artist.get_label() or ""),
                    "color": _color_to_hex(artist.get_color()),
                    "linestyle": str(artist.get_linestyle()),
                    "linewidth": _safe(artist.get_linewidth()),
                    "alpha": _safe(artist.get_alpha()),
                    "zorder": _safe(artist.get_zorder()),
                }
    except Exception:
        pass

    x = np.asarray(artist.get_xdata())
    y = np.asarray(artist.get_ydata())
    if x.size == 0:
        return None
    df = _serialise_xy(x, y, _axis_x_is_date(ax))
    fname = _write_csv(df, os.path.join(out_dir, base_name + ".csv"), compress_threshold_bytes)
    return {
        "type": "line",
        "data_file": fname,
        "label": str(artist.get_label() or ""),
        "color": _color_to_hex(artist.get_color()),
        "linestyle": str(artist.get_linestyle()),
        "linewidth": _safe(artist.get_linewidth()),
        "marker": str(artist.get_marker()),
        "markersize": _safe(artist.get_markersize()),
        "alpha": _safe(artist.get_alpha()),
        "zorder": _safe(artist.get_zorder()),
        "drawstyle": str(artist.get_drawstyle()),
    }


def _serialise_pathcollection(
    artist: PathCollection,
    ax,
    out_dir: str,
    base_name: str,
    compress_threshold_bytes: int,
) -> Optional[Dict[str, Any]]:
    offsets = np.asarray(artist.get_offsets())
    if offsets.size == 0:
        return None
    if offsets.ndim == 1:
        offsets = offsets.reshape(-1, 2)
    x = offsets[:, 0]
    y = offsets[:, 1]
    df = _serialise_xy(x, y, _axis_x_is_date(ax))

    # Per-point sizes / colors if present
    sizes = artist.get_sizes()
    if sizes is not None and len(sizes) == len(df):
        df["size"] = np.asarray(sizes)
    collection_color = None
    fc = artist.get_facecolors()
    if fc is not None and len(fc) == len(df):
        try:
            df["color"] = [matplotlib.colors.to_hex(c, keep_alpha=False) for c in fc]
        except Exception:
            pass
    if fc is not None and len(fc) == 1:
        try:
            collection_color = matplotlib.colors.to_hex(fc[0], keep_alpha=False)
        except Exception:
            collection_color = None
    arr = artist.get_array()
    if arr is not None and len(arr) == len(df):
        df["c_value"] = np.asarray(arr)

    linewidths = artist.get_linewidths()
    default_linewidth = None
    if linewidths is not None and len(linewidths):
        default_linewidth = _safe(float(np.nanmean(linewidths)))

    fname = _write_csv(df, os.path.join(out_dir, base_name + ".csv"), compress_threshold_bytes)

    cmap = None
    vmin = vmax = None
    try:
        cmap = artist.get_cmap().name if artist.get_cmap() is not None else None
        norm = artist.norm
        if norm is not None:
            vmin = _safe(norm.vmin)
            vmax = _safe(norm.vmax)
    except Exception:
        pass

    return {
        "type": "scatter",
        "data_file": fname,
        "label": str(artist.get_label() or ""),
        "default_size": _safe(np.mean(sizes)) if sizes is not None and len(sizes) else None,
        "color": collection_color,
        "linewidths": default_linewidth,
        "alpha": _safe(artist.get_alpha()),
        "zorder": _safe(artist.get_zorder()),
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "marker": "o",  # PathCollection markers are paths; default to 'o' for replot
    }


def _serialise_quadmesh(
    artist: QuadMesh,
    ax,
    out_dir: str,
    base_name: str,
    compress_threshold_bytes: int,
) -> Optional[Dict[str, Any]]:
    arr = artist.get_array()
    if arr is None:
        return None
    coords = artist._coordinates  # (ny+1, nx+1, 2) corner grid
    if coords is None:
        return None
    coords = np.asarray(coords)
    if coords.ndim != 3:
        return None
    ny_p1, nx_p1, _ = coords.shape
    ny, nx = ny_p1 - 1, nx_p1 - 1

    # Column-major flatten matches QuadMesh ordering
    values = np.ma.asarray(arr).filled(np.nan).reshape(ny, nx)

    x_edges = coords[0, :, 0]  # row 0 x-coords across columns
    y_edges = coords[:, 0, 1]  # col 0 y-coords across rows

    values_path = os.path.join(out_dir, base_name + "_values.csv")
    df_vals = pd.DataFrame(values)
    fname_values = _write_csv(df_vals, values_path, compress_threshold_bytes)

    edges_path = os.path.join(out_dir, base_name + "_edges.csv")
    edges_df = pd.DataFrame({
        "x_edges": pd.Series(x_edges),
        "y_edges": pd.Series(y_edges),
    })
    fname_edges = _write_csv(edges_df, edges_path, compress_threshold_bytes)

    cmap = None
    vmin = vmax = None
    try:
        cmap = artist.get_cmap().name if artist.get_cmap() is not None else None
        norm = artist.norm
        if norm is not None:
            vmin = _safe(norm.vmin)
            vmax = _safe(norm.vmax)
    except Exception:
        pass

    return {
        "type": "quadmesh",
        "values_file": fname_values,
        "edges_file": fname_edges,
        "label": str(artist.get_label() or ""),
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "alpha": _safe(artist.get_alpha()),
        "zorder": _safe(artist.get_zorder()),
        "x_is_date": _axis_x_is_date(ax),
        "y_is_date": _axis_y_is_date(ax),
    }


def _serialise_image(
    artist: AxesImage,
    ax,
    out_dir: str,
    base_name: str,
    compress_threshold_bytes: int,
) -> Optional[Dict[str, Any]]:
    arr = artist.get_array()
    if arr is None:
        return None
    values = np.ma.asarray(arr).filled(np.nan)
    if values.ndim == 3:  # RGB(A) image -- skip generic capture
        return None
    df = pd.DataFrame(values)
    fname = _write_csv(df, os.path.join(out_dir, base_name + "_values.csv"), compress_threshold_bytes)
    extent = artist.get_extent()
    cmap = None
    vmin = vmax = None
    try:
        cmap = artist.get_cmap().name if artist.get_cmap() is not None else None
        norm = artist.norm
        if norm is not None:
            vmin = _safe(norm.vmin)
            vmax = _safe(norm.vmax)
    except Exception:
        pass
    return {
        "type": "image",
        "values_file": fname,
        "label": str(artist.get_label() or ""),
        "extent": [_safe(v) for v in extent],
        "origin": str(artist.origin),
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "aspect": str(getattr(ax, "get_aspect", lambda: "auto")()),
        "alpha": _safe(artist.get_alpha()),
        "zorder": _safe(artist.get_zorder()),
    }


def _serialise_polycollection(
    artist: PolyCollection,
    ax,
    out_dir: str,
    base_name: str,
    compress_threshold_bytes: int,
) -> Optional[Dict[str, Any]]:
    paths = artist.get_paths()
    if not paths:
        return None
    rows = []
    for poly_idx, p in enumerate(paths):
        verts = np.asarray(p.vertices)
        for v_idx, (xv, yv) in enumerate(verts):
            rows.append({"polygon_id": poly_idx, "vertex_idx": v_idx, "x": xv, "y": yv})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fname = _write_csv(df, os.path.join(out_dir, base_name + ".csv"), compress_threshold_bytes)

    fc = artist.get_facecolors()
    color = None
    if fc is not None and len(fc):
        color = matplotlib.colors.to_hex(fc[0], keep_alpha=False)
    return {
        "type": "polygon",
        "data_file": fname,
        "label": str(artist.get_label() or ""),
        "color": color,
        "alpha": _safe(artist.get_alpha()),
        "zorder": _safe(artist.get_zorder()),
    }


def _serialise_linecollection(
    artist: LineCollection,
    ax,
    out_dir: str,
    base_name: str,
    compress_threshold_bytes: int,
) -> Optional[Dict[str, Any]]:
    segs = artist.get_segments()
    if not segs:
        return None
    rows = []
    for seg_idx, seg in enumerate(segs):
        seg = np.asarray(seg)
        for v_idx, (xv, yv) in enumerate(seg):
            rows.append({"segment_id": seg_idx, "vertex_idx": v_idx, "x": xv, "y": yv})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fname = _write_csv(df, os.path.join(out_dir, base_name + ".csv"), compress_threshold_bytes)
    colors = artist.get_colors()
    color = None
    if colors is not None and len(colors):
        color = matplotlib.colors.to_hex(colors[0], keep_alpha=False)
    return {
        "type": "linecollection",
        "data_file": fname,
        "label": str(artist.get_label() or ""),
        "color": color,
        "linewidth": _safe(np.asarray(artist.get_linewidths()).mean()) if len(artist.get_linewidths()) else None,
        "alpha": _safe(artist.get_alpha()),
        "zorder": _safe(artist.get_zorder()),
    }


def _serialise_bar_container(
    container: BarContainer,
    ax,
    out_dir: str,
    base_name: str,
    compress_threshold_bytes: int,
) -> Optional[Dict[str, Any]]:
    rects = [c for c in container if isinstance(c, Rectangle)]
    if not rects:
        return None
    rows = []
    for r in rects:
        rows.append({
            "x": r.get_x(),
            "y": r.get_y(),
            "width": r.get_width(),
            "height": r.get_height(),
        })
    df = pd.DataFrame(rows)
    fname = _write_csv(df, os.path.join(out_dir, base_name + ".csv"), compress_threshold_bytes)
    color = _color_to_hex(rects[0].get_facecolor())
    # Heuristic for orientation
    widths = df["width"].to_numpy()
    heights = df["height"].to_numpy()
    orientation = "vertical"
    if np.nanmedian(np.abs(widths)) > np.nanmedian(np.abs(heights)) and np.nanmedian(np.abs(heights)) > 0:
        orientation = "horizontal"
    return {
        "type": "bar",
        "data_file": fname,
        "label": str(container.get_label() or ""),
        "color": color,
        "alpha": _safe(rects[0].get_alpha()),
        "orientation": orientation,
    }


def _serialise_errorbar_container(
    container: ErrorbarContainer,
    ax,
    out_dir: str,
    base_name: str,
    compress_threshold_bytes: int,
) -> Optional[Dict[str, Any]]:
    line, caps, bars = container
    if line is None:
        return None
    x = np.asarray(line.get_xdata())
    y = np.asarray(line.get_ydata())
    if x.size == 0:
        return None
    df = _serialise_xy(x, y, _axis_x_is_date(ax))
    # Bars: list of LineCollection (xerr, yerr)
    if bars:
        try:
            for bi, bar in enumerate(bars):
                segs = bar.get_segments()
                # We attempt to recover symmetric err magnitudes
                if len(segs) == len(df):
                    arr = np.array([np.asarray(s) for s in segs])
                    # For yerr: each seg is [[x, y-err],[x, y+err]]
                    if bi == 0:
                        # Determine direction by checking variance
                        d_y = arr[:, 1, 1] - arr[:, 0, 1]
                        d_x = arr[:, 1, 0] - arr[:, 0, 0]
                        if np.nanmean(np.abs(d_y)) >= np.nanmean(np.abs(d_x)):
                            df["yerr"] = (d_y / 2.0)
                        else:
                            df["xerr"] = (d_x / 2.0)
        except Exception:
            pass
    fname = _write_csv(df, os.path.join(out_dir, base_name + ".csv"), compress_threshold_bytes)
    return {
        "type": "errorbar",
        "data_file": fname,
        "label": str(container.get_label() or ""),
        "color": _color_to_hex(line.get_color()),
        "marker": str(line.get_marker()),
        "linestyle": str(line.get_linestyle()),
        "alpha": _safe(line.get_alpha()),
    }


# --------------------------------------------------------------------------- #
# Per-axes / per-figure serialisation
# --------------------------------------------------------------------------- #
def _axes_metadata(ax) -> Dict[str, Any]:
    pos = ax.get_position()
    geom = None
    try:
        ss = ax.get_subplotspec()
        if ss is not None:
            gs = ss.get_gridspec()
            geom = {
                "nrows": int(gs.nrows),
                "ncols": int(gs.ncols),
                "row_start": int(ss.rowspan.start),
                "row_stop": int(ss.rowspan.stop),
                "col_start": int(ss.colspan.start),
                "col_stop": int(ss.colspan.stop),
            }
    except Exception:
        geom = None

    legend = ax.get_legend()
    legend_meta = None
    if legend is not None:
        try:
            legend_meta = {
                "labels": [t.get_text() for t in legend.get_texts()],
                "loc": str(getattr(legend, "_loc", "best")),
            }
        except Exception:
            legend_meta = None

    return {
        "title": str(ax.get_title() or ""),
        "xlabel": str(ax.get_xlabel() or ""),
        "ylabel": str(ax.get_ylabel() or ""),
        "xscale": str(ax.get_xscale()),
        "yscale": str(ax.get_yscale()),
        "xlim": [_safe(v) for v in ax.get_xlim()],
        "ylim": [_safe(v) for v in ax.get_ylim()],
        "aspect": str(ax.get_aspect()) if hasattr(ax, "get_aspect") else "auto",
        "grid_visible": bool(any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())),
        "x_is_date": _axis_x_is_date(ax),
        "y_is_date": _axis_y_is_date(ax),
        "geometry": geom,
        "position_bbox": [pos.x0, pos.y0, pos.width, pos.height],
        "legend": legend_meta,
    }


def _is_probable_control_axes(ax) -> bool:
    """Detect low-profile Matplotlib widget axes such as Slider controls."""
    try:
        if ax.get_title() or ax.get_xlabel() or ax.get_ylabel():
            return False
        pos = ax.get_position()
        if pos.height > 0.08 or pos.width < 0.25 or pos.y0 > 0.16:
            return False
        if len(ax.collections) > 0 or len(ax.images) > 0 or len(ax.containers) > 0:
            return False
        return True
    except Exception:
        return False


def _capture_axes(
    ax,
    axes_dir: str,
    axis_id: str,
    compress_threshold_bytes: int,
) -> Dict[str, Any]:
    os.makedirs(axes_dir, exist_ok=True)
    artists_meta: List[Dict[str, Any]] = []
    counter = 0

    # Containers (bars, errorbars) own multiple raw artists; capture container-level first
    for container in list(getattr(ax, "containers", []) or []):
        counter += 1
        base = f"{axis_id}_a{counter:03d}_{type(container).__name__}"
        try:
            if isinstance(container, BarContainer):
                meta = _serialise_bar_container(container, ax, axes_dir, base, compress_threshold_bytes)
            elif isinstance(container, ErrorbarContainer):
                meta = _serialise_errorbar_container(container, ax, axes_dir, base, compress_threshold_bytes)
            else:
                meta = None
        except Exception as exc:  # pragma: no cover
            meta = {"type": "error", "error": f"{type(container).__name__}: {exc}"}
        if meta:
            artists_meta.append(meta)

    # Track lines that belong to errorbar/bar containers so we don't double-capture
    consumed: set = set()
    for container in getattr(ax, "containers", []) or []:
        if isinstance(container, ErrorbarContainer):
            line, caps, bars = container
            if line is not None:
                consumed.add(id(line))
            for c in caps or ():
                consumed.add(id(c))
            for b in bars or ():
                consumed.add(id(b))
        elif isinstance(container, BarContainer):
            for r in container:
                consumed.add(id(r))

    # Lines
    for artist in ax.get_lines():
        if id(artist) in consumed:
            continue
        counter += 1
        base = f"{axis_id}_a{counter:03d}_line"
        try:
            meta = _serialise_line2d(artist, ax, axes_dir, base, compress_threshold_bytes)
        except Exception as exc:  # pragma: no cover
            meta = {"type": "error", "error": f"Line2D: {exc}"}
        if meta:
            artists_meta.append(meta)

    # Collections (scatter, pcolormesh, fill_between, etc.)
    for artist in list(ax.collections):
        if id(artist) in consumed:
            continue
        counter += 1
        base = f"{axis_id}_a{counter:03d}_{type(artist).__name__}"
        try:
            if isinstance(artist, QuadMesh):
                meta = _serialise_quadmesh(artist, ax, axes_dir, base, compress_threshold_bytes)
            elif isinstance(artist, PathCollection):
                meta = _serialise_pathcollection(artist, ax, axes_dir, base, compress_threshold_bytes)
            elif isinstance(artist, PolyCollection):
                meta = _serialise_polycollection(artist, ax, axes_dir, base, compress_threshold_bytes)
            elif isinstance(artist, LineCollection):
                meta = _serialise_linecollection(artist, ax, axes_dir, base, compress_threshold_bytes)
            else:
                meta = {"type": "unsupported_collection", "class": type(artist).__name__}
        except Exception as exc:  # pragma: no cover
            meta = {"type": "error", "error": f"{type(artist).__name__}: {exc}"}
        if meta:
            artists_meta.append(meta)

    # Images
    for artist in list(ax.get_images()):
        counter += 1
        base = f"{axis_id}_a{counter:03d}_image"
        try:
            meta = _serialise_image(artist, ax, axes_dir, base, compress_threshold_bytes)
        except Exception as exc:  # pragma: no cover
            meta = {"type": "error", "error": f"AxesImage: {exc}"}
        if meta:
            artists_meta.append(meta)

    return {
        "axis_id": axis_id,
        "metadata": _axes_metadata(ax),
        "artists": artists_meta,
    }


def capture_open_figures(
    output_dir: str,
    compress_threshold_bytes: int = DEFAULT_COMPRESS_THRESHOLD_BYTES,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Walk every open matplotlib figure and write per-artist CSVs and a
    ``manifest.json`` into ``output_dir``.

    Returns a summary dict suitable for embedding in pipeline metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    fignums = sorted(int(n) for n in plt.get_fignums())
    figures_meta: List[Dict[str, Any]] = []
    used_stems: Dict[str, int] = {}
    errors: List[str] = []

    for ordinal, fignum in enumerate(fignums, start=1):
        try:
            fig = plt.figure(fignum)
        except Exception as exc:
            errors.append(f"figure {fignum}: cannot reopen ({exc})")
            continue

        label = _figure_label(fig, ordinal)
        stem = _sanitize(label, fallback=f"figure_{ordinal:03d}")
        n_used = used_stems.get(stem, 0) + 1
        used_stems[stem] = n_used
        if n_used > 1:
            stem = f"{stem}__{n_used}"
        fig_dir_name = f"figure_{ordinal:03d}_{stem}"
        fig_dir = os.path.join(output_dir, fig_dir_name)
        os.makedirs(fig_dir, exist_ok=True)

        axes = list(fig.get_axes())

        # Identify colorbar axes so we can (a) skip capturing them as
        # standalone subplots and (b) annotate the parent axes that owns the
        # colorbar so the replot can re-attach one.
        colorbar_axes_ids: set = set()
        colorbar_info_by_parent: Dict[int, Dict[str, Any]] = {}
        for ax in axes:
            # Matplotlib labels colorbar axes with '<colorbar>'.
            try:
                if str(ax.get_label()) == "<colorbar>":
                    colorbar_axes_ids.add(id(ax))
            except Exception:
                pass
        control_axes_ids = {
            id(ax) for ax in axes
            if id(ax) not in colorbar_axes_ids and _is_probable_control_axes(ax)
        }

        # Walk every artist on every non-colorbar/non-control axes; if it has a .colorbar
        # attribute (set by fig.colorbar), record metadata against the parent.
        for ax in axes:
            if id(ax) in colorbar_axes_ids or id(ax) in control_axes_ids:
                continue
            mappables = list(ax.collections) + list(ax.get_images())
            for m in mappables:
                cb = getattr(m, "colorbar", None)
                if cb is None:
                    continue
                cb_ax = getattr(cb, "ax", None)
                if cb_ax is not None:
                    colorbar_axes_ids.add(id(cb_ax))
                cmap = None
                vmin = vmax = None
                try:
                    cmap = m.get_cmap().name if m.get_cmap() is not None else None
                    norm = m.norm
                    if norm is not None:
                        vmin = _safe(norm.vmin)
                        vmax = _safe(norm.vmax)
                except Exception:
                    pass
                colorbar_info_by_parent[id(ax)] = {
                    "orientation": str(getattr(cb, "orientation", "vertical")),
                    "label": str(cb.ax.get_ylabel() or cb.ax.get_xlabel() or ""),
                    "cmap": cmap,
                    "vmin": vmin,
                    "vmax": vmax,
                }
                break

        axes_meta: List[Dict[str, Any]] = []
        for ax_idx, ax in enumerate(axes):
            if id(ax) in colorbar_axes_ids or id(ax) in control_axes_ids:
                continue
            axis_id = f"ax{ax_idx:02d}"
            try:
                ax_record = _capture_axes(ax, fig_dir, axis_id, compress_threshold_bytes)
                cb_info = colorbar_info_by_parent.get(id(ax))
                if cb_info is not None:
                    ax_record["metadata"]["colorbar"] = cb_info
                axes_meta.append(ax_record)
            except Exception as exc:
                errors.append(f"figure {ordinal} axes {ax_idx}: {exc}")

        # Suptitle if present
        suptitle = ""
        try:
            sup = getattr(fig, "_suptitle", None)
            if sup is not None:
                suptitle = str(sup.get_text() or "")
        except Exception:
            pass

        fig_record = {
            "figure_id": ordinal,
            "fignum": int(fignum),
            "label": label,
            "directory": fig_dir_name,
            "suptitle": suptitle,
            "figsize_inches": [float(v) for v in fig.get_size_inches()],
            "dpi": float(fig.get_dpi()),
            "tight_layout": bool(getattr(fig, "_tight", False)),
            "constrained_layout": bool(fig.get_constrained_layout()) if hasattr(fig, "get_constrained_layout") else False,
            "axes": axes_meta,
        }
        figures_meta.append(fig_record)

        # Per-figure manifest sidecar (handy for downstream filtering)
        try:
            with open(os.path.join(fig_dir, "figure.json"), "w", encoding="utf-8") as f:
                json.dump(fig_record, f, indent=2, default=str)
        except Exception as exc:
            errors.append(f"figure {ordinal} sidecar: {exc}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "matplotlib_version": matplotlib.__version__,
        "compression_threshold_bytes": int(compress_threshold_bytes),
        "zstandard_available": _zstd is not None,
        "extra_metadata": _safe(extra_metadata) if extra_metadata else {},
        "figures": figures_meta,
        "errors": errors,
    }

    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
    except Exception as exc:
        errors.append(f"manifest write: {exc}")

    return {
        "enabled": True,
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "figure_count": len(figures_meta),
        "errors": errors,
    }
