"""Pairwise coupling analysis for orbital dynamics.

This module preserves the existing lagged cross-correlation entry point and
adds optional frequency-domain coupling products (cross-spectrum/CSD/coherence).

Scientific note:
- Cross-correlation is descriptive in lag domain.
- Cross-spectrum and coherence are stronger frequency-domain coupling tools.
"""

from __future__ import annotations

import warnings
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate, correlation_lags

try:
    import pywt
    _PYWT_AVAILABLE = True
except Exception:
    pywt = None
    _PYWT_AVAILABLE = False

try:
    from scipy.signal import coherence as scipy_coherence
    from scipy.signal import csd as scipy_csd
    from scipy.signal import welch as scipy_welch
    _SCIPY_CPL_AVAILABLE = True
except Exception:
    scipy_coherence = None
    scipy_csd = None
    scipy_welch = None
    _SCIPY_CPL_AVAILABLE = False

from spectral_preprocessing import (
    SECONDS_PER_DAY,
    build_uniform_resampling_metadata,
    build_elapsed_seconds,
    build_satellite_index_map,
    is_irregular_time_axis,
    preprocess_series,
    resample_pair_to_uniform_grid,
    substitute_low_e_phase,
    unique_in_order,
)


def compute_cross_correlation(x, y, use_fft=True, lag_step_days=None, normalization="legacy"):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
        return np.array([]), np.array([]), np.array([])

    x_zm = x_arr - x_arr.mean()
    y_zm = y_arr - y_arr.mean()
    method = "fft" if bool(use_fft) else "direct"
    corr = correlate(x_zm, y_zm, mode="full", method=method)

    lags = correlation_lags(len(x_zm), len(y_zm), mode="full").astype(np.float64)
    n_overlap_by_lag = np.maximum(0.0, float(len(x_zm)) - np.abs(lags))

    std_prod = np.std(x_zm) * np.std(y_zm)
    if std_prod > 0:
        mode = str(normalization).strip().lower()
        if mode in {"overlap_aware", "overlap-aware", "overlap"}:
            denom = std_prod * np.where(n_overlap_by_lag > 0.0, n_overlap_by_lag, np.nan)
            corr = np.divide(corr, denom, out=np.zeros_like(corr, dtype=np.float64), where=np.isfinite(denom) & (denom > 0.0))
        else:
            corr = corr / (std_prod * len(x_zm))

    if lag_step_days is not None:
        lags = lags * float(lag_step_days)
    return corr, lags, n_overlap_by_lag


def compute_cross_spectrum(x, y, cadence_seconds, nperseg=256, noverlap=None):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 8 or y_arr.size < 8 or x_arr.size != y_arr.size:
        return {
            "frequency_cpd": np.array([]),
            "csd_real": np.array([]),
            "csd_imag": np.array([]),
            "csd_abs": np.array([]),
        }
    if not _SCIPY_CPL_AVAILABLE:
        return {
            "frequency_cpd": np.array([]),
            "csd_real": np.array([]),
            "csd_imag": np.array([]),
            "csd_abs": np.array([]),
            "note": "scipy_unavailable",
        }

    fs = 1.0 / float(cadence_seconds)
    nper = int(max(8, min(int(nperseg), x_arr.size)))
    nov = int(nper // 2) if noverlap is None else int(max(0, min(nper - 1, noverlap)))
    f_hz, pxy = scipy_csd(x_arr, y_arr, fs=fs, nperseg=nper, noverlap=nov, window="hann", scaling="density")
    return {
        "frequency_cpd": f_hz * SECONDS_PER_DAY,
        "csd_real": np.real(pxy),
        "csd_imag": np.imag(pxy),
        "csd_abs": np.abs(pxy),
        "nperseg": int(nper),
        "noverlap": int(nov),
    }


def compute_coherence(x, y, cadence_seconds, nperseg=256, noverlap=None):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 8 or y_arr.size < 8 or x_arr.size != y_arr.size:
        return {"frequency_cpd": np.array([]), "coherence": np.array([])}
    if not _SCIPY_CPL_AVAILABLE:
        return {
            "frequency_cpd": np.array([]),
            "coherence": np.array([]),
            "note": "scipy_unavailable",
        }

    fs = 1.0 / float(cadence_seconds)
    nper = int(max(8, min(int(nperseg), x_arr.size)))
    nov = int(nper // 2) if noverlap is None else int(max(0, min(nper - 1, noverlap)))
    if scipy_csd is not None and scipy_welch is not None:
        f_hz, pxy = scipy_csd(x_arr, y_arr, fs=fs, nperseg=nper, noverlap=nov, window="hann", scaling="density")
        _, pxx = scipy_welch(x_arr, fs=fs, nperseg=nper, noverlap=nov, window="hann", scaling="density")
        _, pyy = scipy_welch(y_arr, fs=fs, nperseg=nper, noverlap=nov, window="hann", scaling="density")
        denom = pxx * pyy
        cxy = np.zeros_like(f_hz, dtype=np.float64)
        valid = np.isfinite(denom) & (denom > 0.0)
        cxy[valid] = (np.abs(pxy[valid]) ** 2) / denom[valid]
        cxy = np.clip(cxy, 0.0, 1.0)
    else:
        f_hz, cxy = scipy_coherence(x_arr, y_arr, fs=fs, nperseg=nper, noverlap=nov, window="hann")
        cxy = np.nan_to_num(cxy, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "frequency_cpd": f_hz * SECONDS_PER_DAY,
        "coherence": cxy,
        "nperseg": int(nper),
        "noverlap": int(nov),
    }


def compute_cross_wavelet(
    x,
    y,
    cadence_seconds,
    *,
    wavelet_name="cmor1.5-1.0",
    n_scales=64,
    min_scale=2.0,
    max_scale=None,
    smoothing_sigma_scale=1.0,
    smoothing_sigma_time=1.5,
):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 16 or y_arr.size < 16 or x_arr.size != y_arr.size:
        return {
            "status": "insufficient_overlap",
            "cross_wavelet_power": np.zeros((0, 0), dtype=np.float64),
            "wavelet_coherence": np.zeros((0, 0), dtype=np.float64),
            "frequency_cpd": np.array([]),
            "period_days": np.array([]),
        }

    if not _PYWT_AVAILABLE:
        return {
            "status": "pywt_unavailable",
            "cross_wavelet_power": np.zeros((0, 0), dtype=np.float64),
            "wavelet_coherence": np.zeros((0, 0), dtype=np.float64),
            "frequency_cpd": np.array([]),
            "period_days": np.array([]),
        }

    if not np.isfinite(cadence_seconds) or float(cadence_seconds) <= 0.0:
        return {
            "status": "invalid_cadence",
            "cross_wavelet_power": np.zeros((0, 0), dtype=np.float64),
            "wavelet_coherence": np.zeros((0, 0), dtype=np.float64),
            "frequency_cpd": np.array([]),
            "period_days": np.array([]),
        }

    max_scale_val = float(max_scale) if max_scale is not None else max(8.0, float(x_arr.size) / 4.0)
    max_scale_val = max(max_scale_val, float(min_scale) + 1.0)
    scales = np.geomspace(float(min_scale), max_scale_val, int(max(16, n_scales))).astype(np.float64)

    try:
        wx, freqs = pywt.cwt(x_arr, scales, wavelet_name, sampling_period=float(cadence_seconds), method="fft")
        wy, _ = pywt.cwt(y_arr, scales, wavelet_name, sampling_period=float(cadence_seconds), method="fft")
    except Exception as exc:
        return {
            "status": "cwt_error",
            "error": str(exc),
            "cross_wavelet_power": np.zeros((0, 0), dtype=np.float64),
            "wavelet_coherence": np.zeros((0, 0), dtype=np.float64),
            "frequency_cpd": np.array([]),
            "period_days": np.array([]),
        }

    wxy = wx * np.conj(wy)
    sxx = gaussian_filter(np.abs(wx) ** 2, sigma=(float(smoothing_sigma_scale), float(smoothing_sigma_time)))
    syy = gaussian_filter(np.abs(wy) ** 2, sigma=(float(smoothing_sigma_scale), float(smoothing_sigma_time)))
    sxy_real = gaussian_filter(np.real(wxy), sigma=(float(smoothing_sigma_scale), float(smoothing_sigma_time)))
    sxy_imag = gaussian_filter(np.imag(wxy), sigma=(float(smoothing_sigma_scale), float(smoothing_sigma_time)))
    sxy = sxy_real + 1j * sxy_imag

    denom = sxx * syy
    coherence = np.zeros_like(np.real(wxy), dtype=np.float64)
    valid = np.isfinite(denom) & (denom > 0.0)
    coherence[valid] = (np.abs(sxy[valid]) ** 2) / denom[valid]
    coherence = np.clip(np.nan_to_num(coherence, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    freq_hz = np.asarray(freqs, dtype=np.float64)
    freq_cpd = freq_hz * SECONDS_PER_DAY
    period_days = np.full(freq_cpd.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        good_f = np.isfinite(freq_cpd) & (np.abs(freq_cpd) > 1.0e-12)
        period_days[good_f] = 1.0 / np.abs(freq_cpd[good_f])

    return {
        "status": "ok",
        "method": "cross_wavelet_cwt",
        "wavelet": str(wavelet_name),
        "frequency_cpd": freq_cpd,
        "period_days": period_days,
        "cross_wavelet_power": np.abs(wxy),
        "cross_wavelet_phase_rad": np.angle(wxy),
        "wavelet_coherence": coherence,
        "n_scales": int(scales.size),
    }


def compute_cross_wavelet_placeholder(*args, **kwargs):
    """Backward-compatible alias for cross-wavelet/coherence computation."""
    if len(args) < 2 and not ({"x", "y"} <= set(kwargs)):
        return {
            "status": "invalid_call",
            "note": "Provide x, y, and cadence_seconds for cross-wavelet computation.",
            "cross_wavelet_power": np.zeros((0, 0), dtype=np.float64),
            "wavelet_coherence": np.zeros((0, 0), dtype=np.float64),
            "frequency_cpd": np.array([]),
            "period_days": np.array([]),
        }
    return compute_cross_wavelet(*args, **kwargs)


def _decimate_for_plot(x, y, max_points=50_000):
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    n = x_arr.size
    if n <= int(max_points):
        return x_arr, y_arr
    stride = int(np.ceil(n / float(max_points)))
    return x_arr[::stride], y_arr[::stride]


def _aggregate_correlations(results_for_pair):
    sats = list(results_for_pair.keys())
    if not sats:
        return np.array([]), np.array([]), np.array([]), []

    lag_ref = None
    for sat in sats:
        lags = np.asarray(results_for_pair[sat].get("lags_days", np.array([])), dtype=np.float64)
        corr = np.asarray(results_for_pair[sat].get("correlation", np.array([])), dtype=np.float64)
        if lags.size == 0 or corr.size == 0:
            continue
        if lag_ref is None or lags.size < lag_ref.size:
            lag_ref = lags

    if lag_ref is None:
        return np.array([]), np.array([]), np.array([]), []

    corr_rows = []
    used_sats = []
    for sat in sats:
        lags = np.asarray(results_for_pair[sat].get("lags_days", np.array([])), dtype=np.float64)
        corr = np.asarray(results_for_pair[sat].get("correlation", np.array([])), dtype=np.float64)
        if lags.size == 0 or corr.size == 0:
            continue
        corr_i = np.interp(lag_ref, lags, corr, left=np.nan, right=np.nan)
        if not np.any(np.isfinite(corr_i)):
            continue
        corr_rows.append(corr_i)
        used_sats.append(sat)

    if not corr_rows:
        return np.array([]), np.array([]), np.array([]), []

    mat = np.vstack(corr_rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr_mean = np.nanmean(mat, axis=0)
    return lag_ref, corr_mean, mat, used_sats


def _aligned_pair_products(
    t_seconds,
    x,
    y,
    *,
    use_fft=True,
    preprocessing="raw",
    cadence_seconds=None,
    min_overlap=32,
    max_grid_points=200_000,
    include_frequency_products=False,
    freq_nperseg=256,
    freq_noverlap=None,
    include_cross_wavelet=False,
    normalization="legacy",
    interpolation="linear",
):
    t_input = np.asarray(t_seconds, dtype=np.float64)
    input_irregular = is_irregular_time_axis(t_input)
    grid, x_u, y_u, used_cadence = resample_pair_to_uniform_grid(
        t_seconds,
        x,
        y,
        cadence_seconds=cadence_seconds,
        max_grid_points=max_grid_points,
    )
    resample_meta = build_uniform_resampling_metadata(
        t_input,
        grid,
        used_cadence_seconds=used_cadence,
        interpolation_method=interpolation,
        resampled=True,
        input_irregular=input_irregular,
    )

    if x_u.size < max(4, int(min_overlap)):
        out = {
            "correlation": np.array([]),
            "lags_days": np.array([]),
            "n_overlap_by_lag": np.array([]),
            "n_overlap": int(x_u.size),
            "cadence_seconds": used_cadence,
            "warning": "overlap_too_short",
            "uniform_grid_seconds": grid,
            "normalization_mode": str(normalization),
        }
        out.update(resample_meta)
        return out

    x_p = preprocess_series(x_u, mode=preprocessing)
    y_p = preprocess_series(y_u, mode=preprocessing)
    n = min(x_p.size, y_p.size)
    x_p = x_p[:n]
    y_p = y_p[:n]
    if n < max(4, int(min_overlap)):
        out = {
            "correlation": np.array([]),
            "lags_days": np.array([]),
            "n_overlap_by_lag": np.array([]),
            "n_overlap": int(n),
            "cadence_seconds": used_cadence,
            "warning": "overlap_too_short",
            "uniform_grid_seconds": grid,
            "normalization_mode": str(normalization),
        }
        out.update(resample_meta)
        return out

    lag_step_days = used_cadence / SECONDS_PER_DAY if np.isfinite(used_cadence) else None
    corr, lags, n_overlap_by_lag = compute_cross_correlation(
        x_p,
        y_p,
        use_fft=use_fft,
        lag_step_days=lag_step_days,
        normalization=normalization,
    )
    out = {
        "correlation": corr,
        "lags_days": lags,
        "n_overlap_by_lag": n_overlap_by_lag,
        "n_overlap": int(n),
        "cadence_seconds": used_cadence,
        "warning": None,
        "uniform_grid_seconds": grid,
        "normalization_mode": str(normalization),
    }
    out.update(resample_meta)

    if include_frequency_products and np.isfinite(used_cadence):
        csd = compute_cross_spectrum(
            x_p,
            y_p,
            cadence_seconds=used_cadence,
            nperseg=freq_nperseg,
            noverlap=freq_noverlap,
        )
        coh = compute_coherence(
            x_p,
            y_p,
            cadence_seconds=used_cadence,
            nperseg=freq_nperseg,
            noverlap=freq_noverlap,
        )
        csd.update(resample_meta)
        coh.update(resample_meta)
        csd["uniform_grid_seconds"] = grid
        coh["uniform_grid_seconds"] = grid
        csd["normalization_mode"] = str(normalization)
        coh["normalization_mode"] = str(normalization)
        out["cross_spectrum"] = csd
        out["coherence"] = coh

    if include_cross_wavelet:
        cw = compute_cross_wavelet(
            x_p,
            y_p,
            cadence_seconds=used_cadence,
        )
        if isinstance(cw, dict):
            cw.update(resample_meta)
            cw["uniform_grid_seconds"] = grid
            cw["normalization_mode"] = str(normalization)
        out["cross_wavelet"] = cw
    return out


def cross_correlation(
    inclinations,
    semi_major_axes,
    args_of_perigee,
    right_ascensions,
    use_fft=True,
    timestamps=None,
    satellite_ids=None,
    selected_satellites=None,
    preprocessing="raw",
    cadence_seconds=None,
    min_overlap=32,
    plot_heatmap=False,
    show_plots=True,
    return_results=False,
    phase_mode=None,
    phase_series=None,
    eccentricities=None,
    ecc_threshold=1e-3,
    max_grid_points=200_000,
    max_plot_points=50_000,
    include_frequency_products=False,
    include_cross_wavelet=False,
    freq_nperseg=256,
    freq_noverlap=None,
    normalization="legacy",
    interpolation="linear",
):
    """Lag-domain cross-correlation with optional frequency-domain coupling outputs."""
    t_all = perf_counter()
    inclinations_arr = np.asarray(inclinations, dtype=np.float64)
    semi_major_axes_arr = np.asarray(semi_major_axes, dtype=np.float64)
    args_of_perigee_arr = np.asarray(args_of_perigee, dtype=np.float64)
    right_ascensions_arr = np.asarray(right_ascensions, dtype=np.float64)

    n = inclinations_arr.size
    elapsed_seconds, lag_meta = build_elapsed_seconds(
        timestamps,
        n,
        fallback_units="samples",
        warning_prefix="cross_correlation",
    )

    if satellite_ids is None:
        sat_ids = np.zeros(n, dtype=np.int64)
    else:
        sat_ids = np.asarray(satellite_ids)
        if sat_ids.size != n:
            sat_ids = np.zeros(n, dtype=np.int64)

    phase_proxy, phase_meta = substitute_low_e_phase(
        args_of_perigee_arr,
        phase_series,
        eccentricities,
        ecc_threshold=float(ecc_threshold),
    )

    sat_ids_str = sat_ids.astype(str)
    unique_sat_ids = unique_in_order(sat_ids_str)
    if selected_satellites is None:
        selected_ids = unique_sat_ids
    else:
        selected_set = {str(x) for x in selected_satellites}
        selected_ids = [sid for sid in unique_sat_ids if sid in selected_set]
    if not selected_ids:
        selected_ids = unique_sat_ids[:1]

    pairs = [
        ("inc_vs_sma", inclinations_arr, semi_major_axes_arr, "Cross-Correlation: Inclination vs. Semi-major Axis"),
        ("raan_vs_phase", right_ascensions_arr, phase_proxy, "Cross-Correlation: RAAN vs. Phase Proxy"),
        ("inc_vs_raan", inclinations_arr, right_ascensions_arr, "Cross-Correlation: Inclination vs. RAAN"),
    ]

    results = {pair[0]: {} for pair in pairs}
    get_sat_indices = build_satellite_index_map(sat_ids_str)
    for sat in selected_ids:
        idx = get_sat_indices(sat)
        if idx.size == 0:
            continue
        t_sat = elapsed_seconds[idx]
        if t_sat.size < max(4, int(min_overlap)):
            continue

        for key, x_series, y_series, _ in pairs:
            out = _aligned_pair_products(
                t_sat,
                x_series[idx],
                y_series[idx],
                use_fft=use_fft,
                preprocessing=preprocessing,
                cadence_seconds=cadence_seconds,
                min_overlap=min_overlap,
                max_grid_points=max_grid_points,
                include_frequency_products=include_frequency_products,
                freq_nperseg=freq_nperseg,
                freq_noverlap=freq_noverlap,
                include_cross_wavelet=include_cross_wavelet,
                normalization=normalization,
                interpolation=interpolation,
            )
            if out["warning"] == "overlap_too_short" and show_plots:
                warnings.warn(
                    f"cross_correlation: overlap too short for {key} on satellite {sat}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            results[key][sat] = out

    plot_limit = 50_000
    try:
        if max_plot_points is not None:
            plot_limit = int(max(1000, max_plot_points))
    except Exception:
        plot_limit = 50_000

    if show_plots:
        for key, _, _, title in pairs:
            lag_ref, corr_mean, _, used_sats = _aggregate_correlations(results[key])
            if lag_ref.size == 0 or corr_mean.size == 0:
                continue
            lags_plot, corr_plot = _decimate_for_plot(lag_ref, corr_mean, max_points=plot_limit)
            fig, ax = plt.subplots()
            ax.plot(lags_plot, corr_plot)
            ax.set_title(f"{title} [combined {len(used_sats)} satellites] | {preprocessing}")
            ax.set_xlabel("Lag (days)" if lag_meta["units"] == "seconds" else "Lag (samples)")
            ax.set_ylabel("Normalized Correlation")

        if plot_heatmap:
            for key, _, _, title in pairs:
                lag_ref, _, mat, used_sats = _aggregate_correlations(results[key])
                if lag_ref.size == 0 or mat.size == 0:
                    continue
                if mat.shape[1] > plot_limit:
                    stride = int(np.ceil(mat.shape[1] / float(plot_limit)))
                    mat = mat[:, ::stride]
                    lag_ref = lag_ref[::stride]
                mat = np.asarray(mat, dtype=np.float32)
                fig_hm, ax_hm = plt.subplots(figsize=(8, 5))
                im = ax_hm.imshow(
                    mat,
                    aspect="auto",
                    origin="lower",
                    extent=[float(lag_ref[0]), float(lag_ref[-1]), 0, mat.shape[0]],
                    cmap="coolwarm",
                )
                ax_hm.set_title(f"Aligned Cross-Correlation Heatmap: {title} [n={len(used_sats)}]")
                ax_hm.set_xlabel("Lag (days)")
                ax_hm.set_ylabel("Satellite index")
                fig_hm.colorbar(im, ax=ax_hm, label="Correlation")

        plt.show()

    print(f"[cross_correlation] Ready in {perf_counter() - t_all:.2f}s")
    if return_results:
        return {
            "results": results,
            "selected_satellites": selected_ids,
            "preprocessing": preprocessing,
            "normalization_mode": str(normalization),
            "lag_units": "days" if lag_meta["units"] == "seconds" else "samples",
            "cadence_seconds": cadence_seconds,
            "phase_mode": phase_mode,
            "phase_metadata": phase_meta,
            "frequency_products_enabled": bool(include_frequency_products),
            "cross_wavelet_enabled": bool(include_cross_wavelet),
            "interpolation_method": str(interpolation),
        }
