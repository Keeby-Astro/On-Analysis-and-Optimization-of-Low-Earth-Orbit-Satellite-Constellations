"""Global spectral analysis for orbital dynamics.

Scientific notes:
- Uniform FFT uses an explicit Hann window + coherent-gain correction.
- Outputs are one-sided amplitude spectrum products on a uniform grid.
"""

from __future__ import annotations

import warnings
from time import perf_counter
import importlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from spectral_preprocessing import (
    SECONDS_PER_DAY,
    build_uniform_resampling_metadata,
    build_elapsed_seconds,
    build_satellite_index_map,
    is_irregular_time_axis,
    resample_to_uniform_grid,
    unique_in_order,
    warn_small_sample,
)

try:
    from scipy.signal import find_peaks, lombscargle as scipy_lombscargle, stft, welch
    _SCIPY_SIGNAL_AVAILABLE = True
except Exception:
    find_peaks = None
    scipy_lombscargle = None
    stft = None
    welch = None
    _SCIPY_SIGNAL_AVAILABLE = False

try:
    cp = importlib.import_module("cupy")
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False

try:
    torch = importlib.import_module("torch")
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


def _resolve_fft_backend(fft_backend="auto", n_samples=0, gpu_min_samples=4096):
    backend = str(fft_backend).lower()
    if backend in {"numpy", "cpu"}:
        return "numpy"

    if backend == "cupy":
        if not _CUPY_AVAILABLE:
            warnings.warn(
                "fft_orbital_elements: cupy requested but unavailable; using numpy.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "numpy"
        return "cupy"

    if backend == "torch":
        if not (_TORCH_AVAILABLE and torch.cuda.is_available()):
            warnings.warn(
                "fft_orbital_elements: torch CUDA requested but unavailable; using numpy.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "numpy"
        return "torch"

    if backend == "auto":
        if int(n_samples) < int(gpu_min_samples):
            return "numpy"
        if _CUPY_AVAILABLE:
            return "cupy"
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return "torch"
        return "numpy"

    warnings.warn(
        f"fft_orbital_elements: unknown fft_backend '{fft_backend}', using numpy.",
        RuntimeWarning,
        stacklevel=3,
    )
    return "numpy"


def _rfft_amplitude_backend(centered, window, coherent_gain, backend="numpy"):
    n = centered.size
    if n == 0:
        return np.array([], dtype=np.float64)

    if backend == "cupy" and _CUPY_AVAILABLE:
        try:
            x_gpu = cp.asarray(centered * window, dtype=cp.float64)
            spec_gpu = cp.fft.rfft(x_gpu)
            amp_gpu = cp.abs(spec_gpu) / (n * coherent_gain)
            if amp_gpu.size > 2:
                amp_gpu[1:-1] *= 2.0
            return cp.asnumpy(amp_gpu)
        except Exception:
            warnings.warn("fft_orbital_elements: CuPy FFT failed; falling back to numpy.", RuntimeWarning, stacklevel=3)

    if backend == "torch" and _TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            dev = torch.device("cuda")
            x_t = torch.as_tensor(centered * window, dtype=torch.float64, device=dev)
            spec_t = torch.fft.rfft(x_t)
            amp_t = torch.abs(spec_t) / (n * coherent_gain)
            if amp_t.numel() > 2:
                amp_t[1:-1] *= 2.0
            return amp_t.detach().cpu().numpy()
        except Exception:
            warnings.warn("fft_orbital_elements: Torch CUDA FFT failed; falling back to numpy.", RuntimeWarning, stacklevel=3)

    spec = np.fft.rfft(centered * window)
    amp = np.abs(spec) / (n * coherent_gain)
    if amp.size > 2:
        amp[1:-1] *= 2.0
    return amp


def _normalize_amplitude(amp):
    a = np.asarray(amp, dtype=np.float64)
    if a.size == 0:
        return a
    m = np.nanmax(a)
    if not np.isfinite(m) or m <= 0:
        return a
    return a / m


def _frequency_to_period_days(freq_cpd):
    f = np.asarray(freq_cpd, dtype=np.float64)
    out = np.full(f.shape, np.inf, dtype=np.float64)
    mask = f > 0
    out[mask] = 1.0 / f[mask]
    return out


def _finite_frequency_xlim(freq_values, require_positive=False):
    f = np.asarray(freq_values, dtype=np.float64)
    finite = np.isfinite(f)
    if require_positive:
        finite &= (f > 0.0)
    vals = f[finite]
    if vals.size == 0:
        return None
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return lo, hi


def _build_frequency_grid(t_days, min_freq_cpd=None, max_freq_cpd=None, n_freqs=512, grid_kind="linear"):
    span_days = max(float(t_days[-1] - t_days[0]), 1e-9)
    diffs = np.diff(t_days)
    diffs = diffs[diffs > 0]
    nyquist_like = 0.5 / np.median(diffs) if diffs.size > 0 else 4.0

    f_min = (1.0 / span_days) if min_freq_cpd is None else max(float(min_freq_cpd), 1e-9)
    f_max = nyquist_like if max_freq_cpd is None else max(float(max_freq_cpd), f_min * 1.01)
    n = int(max(64, n_freqs))
    if str(grid_kind).lower() == "log":
        return np.geomspace(f_min, f_max, n, dtype=np.float64)
    return np.linspace(f_min, f_max, n, dtype=np.float64)


def extract_spectral_peaks(spectrum, top_k=5, min_prominence=None, min_distance_bins=1):
    freq = np.asarray(spectrum.get("frequency_cpd", []), dtype=np.float64)
    amp = np.asarray(spectrum.get("amplitude", []), dtype=np.float64)
    if freq.size < 3 or amp.size != freq.size:
        return []

    if find_peaks is None:
        idx = np.argsort(amp)[::-1][: int(max(1, top_k))]
    else:
        kwargs = {"distance": int(max(1, min_distance_bins))}
        if min_prominence is not None:
            kwargs["prominence"] = float(min_prominence)
        peaks, _ = find_peaks(amp, **kwargs)
        if peaks.size == 0:
            return []
        idx = peaks[np.argsort(amp[peaks])[::-1][: int(max(1, top_k))]]

    out = []
    for i in np.asarray(idx, dtype=np.int64):
        out.append(
            {
                "frequency_cpd": float(freq[i]),
                "period_days": float(1.0 / freq[i]) if float(freq[i]) > 0 else np.inf,
                "amplitude": float(amp[i]),
                "bin_index": int(i),
            }
        )
    return out


def _bootstrap_stack_band(stacked_freq, member_spectra, normalize_each=True, n_bootstrap=0, random_seed=0):
    n_boot = int(max(0, n_bootstrap))
    if n_boot <= 1:
        return None
    valid = [s for s in member_spectra if s["frequency_cpd"].size > 1 and s["amplitude"].size == s["frequency_cpd"].size]
    if len(valid) < 2:
        return None

    rng = np.random.default_rng(int(random_seed))
    f_common = np.asarray(stacked_freq, dtype=np.float64)
    mats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(valid), size=len(valid))
        rows = []
        for j in pick:
            spec = valid[int(j)]
            y = np.asarray(spec["amplitude"], dtype=np.float64)
            if normalize_each:
                y = _normalize_amplitude(y)
            if spec["frequency_cpd"].shape != f_common.shape or not np.allclose(spec["frequency_cpd"], f_common):
                y = np.interp(f_common, spec["frequency_cpd"], y, left=np.nan, right=np.nan)
            rows.append(y)
        mats.append(np.nanmedian(np.vstack(rows), axis=0))

    arr = np.vstack(mats)
    return {
        "p05": np.nanpercentile(arr, 5.0, axis=0),
        "p95": np.nanpercentile(arr, 95.0, axis=0),
        "n_bootstrap": int(n_boot),
    }


def _median_stack_spectra(spectra, normalize_each=True):
    valid = [s for s in spectra if s["frequency_cpd"].size > 1 and s["amplitude"].size == s["frequency_cpd"].size]
    if not valid:
        return None

    f_common = valid[0]["frequency_cpd"]
    amps = []
    for spec in valid:
        y = np.asarray(spec["amplitude"], dtype=np.float64)
        if normalize_each:
            y = _normalize_amplitude(y)
        if spec["frequency_cpd"].shape != f_common.shape or not np.allclose(spec["frequency_cpd"], f_common):
            y = np.interp(f_common, spec["frequency_cpd"], y, left=np.nan, right=np.nan)
        amps.append(y)

    if not amps:
        return None
    arr = np.vstack(amps)
    med = np.nanmedian(arr, axis=0)
    return {
        "frequency_cpd": f_common,
        "period_days": _frequency_to_period_days(f_common),
        "amplitude": med,
        "mode": "median_stack",
        "output_type": "morphology_summary" if normalize_each else "amplitude_spectrum",
        "n_members": int(arr.shape[0]),
        "normalized_before_stack": bool(normalize_each),
        "note": "Normalized stacked spectra summarize morphology, not physical amplitudes.",
    }


def _compute_uniform_fft_periodogram(
    t_seconds,
    y,
    cadence_seconds=None,
    interpolation="linear",
    fft_backend="auto",
    gpu_min_samples=4096,
):
    t_input = np.asarray(t_seconds, dtype=np.float64)
    input_irregular = is_irregular_time_axis(t_input)
    grid, values, used_cad = resample_to_uniform_grid(
        t_seconds,
        y,
        cadence_seconds=cadence_seconds,
        interpolation=interpolation,
    )
    resample_meta = build_uniform_resampling_metadata(
        t_input,
        grid,
        used_cadence_seconds=used_cad,
        interpolation_method=interpolation,
        resampled=True,
        input_irregular=input_irregular,
    )
    if values.size < 4:
        payload = {
            "frequency_cpd": np.array([]),
            "period_days": np.array([]),
            "amplitude": np.array([]),
            "n_samples": int(values.size),
            "cadence_seconds": np.nan,
            "mode": "uniform_fft",
            "output_type": "amplitude_spectrum",
            "window": "hann",
            "coherent_gain_correction": True,
            "uniform_grid_seconds": grid,
        }
        payload.update(resample_meta)
        return payload

    centered = values - np.mean(values)
    window = np.hanning(centered.size)
    coherent_gain = np.mean(window)
    backend_used = _resolve_fft_backend(fft_backend=fft_backend, n_samples=centered.size, gpu_min_samples=gpu_min_samples)
    amp = _rfft_amplitude_backend(centered, window, coherent_gain, backend=backend_used)

    freq_hz = np.fft.rfftfreq(centered.size, d=used_cad)
    freq_cpd = freq_hz * SECONDS_PER_DAY
    period_days = _frequency_to_period_days(freq_cpd)
    psd = (amp ** 2) / np.maximum(np.finfo(np.float64).eps, np.nanmax(np.diff(freq_hz)) if freq_hz.size > 1 else 1.0)
    payload = {
        "frequency_cpd": freq_cpd,
        "period_days": period_days,
        "amplitude": amp,
        "psd_onesided": psd,
        "n_samples": int(centered.size),
        "cadence_seconds": float(used_cad),
        "mode": "uniform_fft",
        "output_type": "amplitude_spectrum",
        "window": "hann",
        "coherent_gain_correction": True,
        "fft_backend": backend_used,
        "uniform_grid_seconds": grid,
    }
    payload.update(resample_meta)
    return payload


def _compute_lomb_scargle_periodogram(
    t_seconds,
    y,
    min_freq_cpd=None,
    max_freq_cpd=None,
    n_freqs=512,
    grid_kind="linear",
):
    t = np.asarray(t_seconds, dtype=np.float64)
    v = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(v)
    t = t[valid]
    v = v[valid]
    if t.size < 5:
        return {
            "frequency_cpd": np.array([]),
            "period_days": np.array([]),
            "amplitude": np.array([]),
            "n_samples": int(t.size),
            "mode": "lomb_scargle",
            "output_type": "normalized_periodogram_amplitude_like",
            "resampled": False,
        }

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[order] - np.mean(v[order])
    signal_energy = float(np.dot(v, v))
    if not np.isfinite(signal_energy) or signal_energy <= 1e-20:
        return {
            "frequency_cpd": np.array([]),
            "period_days": np.array([]),
            "amplitude": np.array([]),
            "n_samples": int(t.size),
            "mode": "lomb_scargle",
            "output_type": "normalized_periodogram_amplitude_like",
            "resampled": False,
        }

    if not _SCIPY_SIGNAL_AVAILABLE or scipy_lombscargle is None:
        warnings.warn(
            "fft_orbital_elements: scipy.signal.lombscargle unavailable; falling back to uniform FFT.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _compute_uniform_fft_periodogram(t, v)

    t_days = (t - t[0]) / SECONDS_PER_DAY
    f_grid = _build_frequency_grid(
        t_days,
        min_freq_cpd=min_freq_cpd,
        max_freq_cpd=max_freq_cpd,
        n_freqs=n_freqs,
        grid_kind=grid_kind,
    )

    angular = 2.0 * np.pi * f_grid
    max_working_bytes = 256 * 1024 * 1024
    max_working_elems = max(1, max_working_bytes // np.dtype(np.float64).itemsize)
    n_time = max(1, int(t_days.size))
    chunk_n_freqs = max(1, int(max_working_elems // n_time))

    if angular.size <= chunk_n_freqs:
        pgram_raw = scipy_lombscargle(t_days, v, angular, precenter=False, normalize=False)
    else:
        pgram_raw = np.empty_like(angular, dtype=np.float64)
        for start in range(0, angular.size, chunk_n_freqs):
            end = min(start + chunk_n_freqs, angular.size)
            pgram_raw[start:end] = scipy_lombscargle(
                t_days,
                v,
                angular[start:end],
                precenter=False,
                normalize=False,
            )
    pgram_norm = pgram_raw * (0.5 / signal_energy)
    pgram_norm = np.nan_to_num(pgram_norm, nan=0.0, posinf=0.0, neginf=0.0)
    amp_like = np.sqrt(np.clip(pgram_norm, 0.0, None))
    caution_note = (
        "Amplitudes from uniform FFT and Lomb-Scargle are not directly "
        "interchangeable without explicit normalization and calibration."
    )

    return {
        "frequency_cpd": f_grid,
        "period_days": _frequency_to_period_days(f_grid),
        "amplitude": amp_like,
        "power_normalized": pgram_norm,
        "amplitude_caution": caution_note,
        "n_samples": int(t.size),
        "mode": "lomb_scargle",
        "output_type": "normalized_periodogram_amplitude_like",
        "resampled": False,
        "grid_kind": str(grid_kind).lower(),
        "min_freq_cpd": float(np.min(f_grid)) if f_grid.size else np.nan,
        "max_freq_cpd": float(np.max(f_grid)) if f_grid.size else np.nan,
        "n_freqs": int(f_grid.size),
    }


def _compute_welch_psd_periodogram(
    t_seconds,
    y,
    cadence_seconds=None,
    interpolation="linear",
    nperseg=256,
    noverlap=None,
):
    grid, values, used_cad = resample_to_uniform_grid(
        t_seconds,
        y,
        cadence_seconds=cadence_seconds,
        interpolation=interpolation,
    )
    if values.size < 8:
        return {
            "frequency_cpd": np.array([]),
            "period_days": np.array([]),
            "amplitude": np.array([]),
            "psd": np.array([]),
            "n_samples": int(values.size),
            "mode": "welch_psd",
            "output_type": "power_spectral_density",
            "resampled": True,
            "cadence_seconds": np.nan,
        }

    if welch is None:
        warnings.warn("fft_orbital_elements: scipy.signal.welch unavailable; using uniform_fft.", RuntimeWarning, stacklevel=2)
        return _compute_uniform_fft_periodogram(grid, values, cadence_seconds=used_cad, interpolation="linear")

    fs = 1.0 / float(used_cad)
    nper = int(max(8, min(int(nperseg), values.size)))
    nov = int(nper // 2) if noverlap is None else int(max(0, min(nper - 1, noverlap)))
    freq_hz, psd = welch(values - np.mean(values), fs=fs, window="hann", nperseg=nper, noverlap=nov, scaling="density")
    freq_cpd = freq_hz * SECONDS_PER_DAY
    amp = np.sqrt(np.clip(psd, 0.0, None))
    return {
        "frequency_cpd": freq_cpd,
        "period_days": _frequency_to_period_days(freq_cpd),
        "amplitude": amp,
        "psd": psd,
        "n_samples": int(values.size),
        "mode": "welch_psd",
        "output_type": "power_spectral_density",
        "resampled": True,
        "cadence_seconds": float(used_cad),
        "nperseg": int(nper),
        "noverlap": int(nov),
    }


def _compute_stft_summary_periodogram(
    t_seconds,
    y,
    cadence_seconds=None,
    interpolation="linear",
    nperseg=256,
    noverlap=None,
):
    grid, values, used_cad = resample_to_uniform_grid(
        t_seconds,
        y,
        cadence_seconds=cadence_seconds,
        interpolation=interpolation,
    )
    if values.size < 8:
        return {
            "frequency_cpd": np.array([]),
            "period_days": np.array([]),
            "amplitude": np.array([]),
            "n_samples": int(values.size),
            "mode": "stft",
            "output_type": "time_localized_amplitude_summary",
            "resampled": True,
            "cadence_seconds": np.nan,
        }

    if stft is None:
        warnings.warn("fft_orbital_elements: scipy.signal.stft unavailable; using uniform_fft.", RuntimeWarning, stacklevel=2)
        return _compute_uniform_fft_periodogram(grid, values, cadence_seconds=used_cad, interpolation="linear")

    fs = 1.0 / float(used_cad)
    nper = int(max(8, min(int(nperseg), values.size)))
    nov = int(nper // 2) if noverlap is None else int(max(0, min(nper - 1, noverlap)))
    f_hz, t_stft, zxx = stft(values - np.mean(values), fs=fs, window="hann", nperseg=nper, noverlap=nov, boundary=None)
    amp2d = np.abs(zxx)
    amp_summary = np.nanmedian(amp2d, axis=1) if amp2d.ndim == 2 and amp2d.shape[1] > 0 else np.array([])
    freq_cpd = f_hz * SECONDS_PER_DAY
    return {
        "frequency_cpd": freq_cpd,
        "period_days": _frequency_to_period_days(freq_cpd),
        "amplitude": amp_summary,
        "stft_time_seconds": np.asarray(t_stft, dtype=np.float64),
        "stft_amplitude": amp2d,
        "n_samples": int(values.size),
        "mode": "stft",
        "output_type": "time_localized_amplitude_summary",
        "resampled": True,
        "cadence_seconds": float(used_cad),
        "nperseg": int(nper),
        "noverlap": int(nov),
    }


def uniform_fft(*args, **kwargs):
    return _compute_uniform_fft_periodogram(*args, **kwargs)


def lomb_scargle_spectrum(*args, **kwargs):
    raise ValueError("lomb_scargle_spectrum has been removed; use uniform_fft().")


def lomb_scargle(*args, **kwargs):
    raise ValueError("lomb_scargle has been removed; use uniform_fft().")


def welch_psd(*args, **kwargs):
    raise ValueError("welch_psd has been removed; use uniform_fft().")


def fft_orbital_elements(
    inclinations,
    semi_major_axes,
    right_ascensions,
    args_of_perigee,
    eccentricities,
    true_anomalies,
    fileNames,
    filenames_array,
    phase_mode=None,
    phase_series=None,
    timestamps=None,
    satellite_ids=None,
    selected_satellites=None,
    mode="uniform_fft",
    cadence_seconds=None,
    interpolation="linear",
    stack_mode="none",
    normalize_before_stack=True,
    show_period_axis=False,
    show_plots=True,
    return_results=False,
    fft_backend="auto",
    gpu_min_samples=4096,
    extract_peaks=False,
    peak_top_k=5,
    peak_min_prominence=None,
    peak_min_distance_bins=1,
    peak_overlay=False,
    bootstrap_replicates=0,
    bootstrap_seed=0,
):
    """Perform uniform-FFT spectral analysis on orbital elements.

    Uniform FFT formulation:
    - Window: Hann, w[n]
    - Coherent gain: CG = mean(w)
    - One-sided amplitude spectrum:
      A1(f_k) = |RFFT((x-mean(x))*w)| / (N*CG), with interior bins doubled.
    """
    t0 = perf_counter()

    requested_mode = str(mode).lower()
    if requested_mode != "uniform_fft":
        raise ValueError("fft_orbital_elements only supports mode='uniform_fft'.")
    mode = "uniform_fft"

    gui_data = {
        "inclinations": np.asarray(inclinations, dtype=np.float64),
        "semi_major_axes": np.asarray(semi_major_axes, dtype=np.float64),
        "right_ascensions": np.asarray(right_ascensions, dtype=np.float64),
        "args_of_perigee": np.asarray(args_of_perigee, dtype=np.float64),
        "eccentricities": np.asarray(eccentricities, dtype=np.float64),
        "true_anomalies": np.asarray(true_anomalies, dtype=np.float64),
        "filenames_array": np.asarray(filenames_array),
    }

    if phase_series is not None:
        phase_arr = np.asarray(phase_series, dtype=np.float64)
        if phase_arr.shape != gui_data["true_anomalies"].shape:
            raise ValueError("phase_series must match true_anomalies shape")
        gui_data["selected_phase"] = phase_arr

    n_rows = gui_data["inclinations"].size
    elapsed_seconds, time_meta = build_elapsed_seconds(
        timestamps,
        n_rows,
        fallback_units="samples",
        warning_prefix="fft_orbital_elements",
    )

    if satellite_ids is not None:
        sat_ids = np.asarray(satellite_ids)
    else:
        sat_ids = np.asarray(filenames_array)
    if sat_ids.size != n_rows:
        warnings.warn(
            "fft_orbital_elements: satellite_ids length mismatch; falling back to filenames_array grouping.",
            RuntimeWarning,
            stacklevel=2,
        )
        sat_ids = np.asarray(filenames_array)

    sat_ids_str = sat_ids.astype(str)
    available_ids = unique_in_order(sat_ids_str)
    if selected_satellites is None:
        selected_ids = available_ids
    else:
        selected_set = {str(x) for x in selected_satellites}
        selected_ids = [sid for sid in available_ids if str(sid) in selected_set]
    if not selected_ids:
        selected_ids = available_ids[:1]

    display_names = [str(x) for x in selected_ids]
    if stack_mode.lower() == "median":
        display_names.append("Stacked Median")

    initial_idx = 0
    init_name = display_names[initial_idx] if display_names else None

    y_keys = [
        "inclinations",
        "semi_major_axes",
        "right_ascensions",
        "args_of_perigee",
        "eccentricities",
        "true_anomalies",
    ]

    spectrum_keys = y_keys + (["selected_phase"] if "selected_phase" in gui_data else [])
    spectra_by_var = {k: {} for k in spectrum_keys}
    get_sat_indices = build_satellite_index_map(sat_ids_str)

    def compute_mode_spectrum(t_data, y_data, sat_id):
        if y_data.size < 8:
            warn_small_sample("fft_orbital_elements", str(sat_id), int(y_data.size), 16)
        return _compute_uniform_fft_periodogram(
            t_data,
            y_data,
            cadence_seconds=cadence_seconds,
            interpolation=interpolation,
            fft_backend=fft_backend,
            gpu_min_samples=gpu_min_samples,
        )

    def get_cached_spectrum(y_key, sat_id):
        sat_key = str(sat_id)
        cached = spectra_by_var[y_key].get(sat_key)
        if cached is not None:
            return cached

        idx = get_sat_indices(sat_key)
        if idx.size < 4:
            spec = {
                "frequency_cpd": np.array([]),
                "period_days": np.array([]),
                "amplitude": np.array([]),
                "n_samples": int(idx.size),
                "mode": str(mode).lower(),
                "output_type": "amplitude_spectrum",
            }
            spectra_by_var[y_key][sat_key] = spec
            return spec

        t_sat = elapsed_seconds[idx]
        y_sat = gui_data[y_key][idx]
        spec = compute_mode_spectrum(t_sat, y_sat, sat_key)
        if extract_peaks:
            spec["peaks"] = extract_spectral_peaks(
                spec,
                top_k=peak_top_k,
                min_prominence=peak_min_prominence,
                min_distance_bins=peak_min_distance_bins,
            )
        spectra_by_var[y_key][sat_key] = spec
        return spec

    stacked_by_var = {}
    if stack_mode.lower() == "median":
        for y_key in spectrum_keys:
            sat_specs = [get_cached_spectrum(y_key, sid) for sid in selected_ids]
            stacked = _median_stack_spectra(sat_specs, normalize_each=normalize_before_stack)
            if stacked is not None:
                band = _bootstrap_stack_band(
                    stacked["frequency_cpd"],
                    sat_specs,
                    normalize_each=normalize_before_stack,
                    n_bootstrap=bootstrap_replicates,
                    random_seed=bootstrap_seed,
                )
                if band is not None:
                    stacked["bootstrap_band"] = band
                stacked_by_var[y_key] = stacked

    def extract_series(y_key, selected_name):
        if selected_name == "Stacked Median":
            spec = stacked_by_var.get(y_key)
        else:
            spec = get_cached_spectrum(y_key, selected_name)
        if not spec:
            return np.array([]), np.array([]), spec
        return spec["frequency_cpd"], spec["amplitude"], spec

    def fft_plot(y_data_key, title, ylabel, color, ylim):
        if init_name is None:
            return
        f_init, p_init, spec_init = extract_series(y_data_key, init_name)
        if p_init.size == 0:
            print(f"No data available for {y_data_key}")
            return

        fig_fft, ax_fft = plt.subplots()
        fft_line, = ax_fft.plot(f_init, p_init, color)
        ax_fft.set_title(title)
        ax_fft.set_xlabel("Frequency (cycles/day)")
        ax_fft.set_ylabel(ylabel)
        ax_fft.set_ylim(ylim)
        subtitle = (
            f"mode={spec_init.get('mode', mode)} | type={spec_init.get('output_type', 'unknown')} "
            f"| resampled={spec_init.get('resampled', False)}"
        )
        ax_fft.text(0.01, 0.98, subtitle, transform=ax_fft.transAxes, va="top", ha="left", fontsize=9)

        if peak_overlay and spec_init.get("peaks"):
            pk_f = [p["frequency_cpd"] for p in spec_init["peaks"]]
            pk_a = [p["amplitude"] for p in spec_init["peaks"]]
            ax_fft.scatter(pk_f, pk_a, c="k", s=18, zorder=3)

        period_axis = None
        if show_period_axis:
            eps = 1e-12
            period_axis = ax_fft.secondary_xaxis(
                "top",
                functions=(
                    lambda f: 1.0 / np.maximum(np.asarray(f, dtype=np.float64), eps),
                    lambda p: 1.0 / np.maximum(np.asarray(p, dtype=np.float64), eps),
                ),
            )
            period_axis.set_xlabel("Period (days)")
            xlim = _finite_frequency_xlim(f_init, require_positive=True)
            if xlim is not None:
                ax_fft.set_xlim(xlim[0], xlim[1])
        else:
            xlim = _finite_frequency_xlim(f_init, require_positive=False)
            if xlim is not None:
                ax_fft.set_xlim(xlim[0], xlim[1])

        ax_slider_fft = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor="lightgoldenrodyellow", figure=fig_fft)
        slider_fft = Slider(ax_slider_fft, "File Index", 0, len(display_names) - 1, valinit=initial_idx, valstep=1)

        def update_fft_plot(_):
            idx = int(slider_fft.val)
            selected_filename = display_names[idx]
            t_update = perf_counter()
            f_new, p_new, spec_new = extract_series(y_data_key, selected_filename)
            if p_new.size == 0:
                fft_line.set_ydata([])
                fft_line.set_xdata([])
                ax_fft.relim()
                ax_fft.autoscale_view()
                fig_fft.canvas.draw_idle()
                return

            fft_line.set_xdata(f_new)
            fft_line.set_ydata(p_new)
            xlim = _finite_frequency_xlim(f_new, require_positive=bool(period_axis is not None))
            if xlim is not None:
                ax_fft.set_xlim(xlim[0], xlim[1])
            ax_fft.set_ylim(ylim)
            ax_fft.set_title(f"{title} [{selected_filename}]")
            mode_used = spec_new.get("mode", mode) if spec_new else mode
            for txt in list(ax_fft.texts):
                txt.remove()
            ax_fft.text(
                0.01,
                0.98,
                f"mode={mode_used} | type={spec_new.get('output_type', 'unknown')} | resampled={spec_new.get('resampled', False)}",
                transform=ax_fft.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )
            fig_fft.canvas.draw_idle()
            print(f"[fft_orbital_elements] {y_data_key} -> {selected_filename} in {perf_counter() - t_update:.2f}s")

        slider_fft.on_changed(update_fft_plot)
        if show_plots:
            plt.show()

    if show_plots:
        fft_plot("inclinations", "Amplitude Spectrum of Inclinations", "|P1(f)|", "r", [0, 0.005])
        fft_plot("semi_major_axes", "Amplitude Spectrum of Semi-Major Axes", "|P1(f)|", "g", [0, 100])
        fft_plot("right_ascensions", "Amplitude Spectrum of Right Ascensions", "|P1(f)|", "b", [0, 140])
        fft_plot("args_of_perigee", "Amplitude Spectrum of Arguments of Perigee", "|P1(f)|", "m", [0, 25])
        fft_plot("eccentricities", "Amplitude Spectrum of Eccentricities", "|P1(f)|", "c", [0, 0.5e-4])
        if phase_mode is not None and "selected_phase" in gui_data:
            fft_plot("selected_phase", "Amplitude Spectrum of True Anomaly (TLE Kepler proxy)", "|P1(f)|", "k", [0, 25])
        else:
            fft_plot("true_anomalies", "Amplitude Spectrum of True Anomaly (TLE Kepler proxy)", "|P1(f)|", "k", [0, 25])

    if return_results:
        for y_key in spectrum_keys:
            for sid in selected_ids:
                get_cached_spectrum(y_key, sid)
        if stack_mode.lower() == "median":
            for y_key in spectrum_keys:
                if y_key not in stacked_by_var:
                    sat_specs = [spectra_by_var[y_key].get(str(sid), get_cached_spectrum(y_key, sid)) for sid in selected_ids]
                    stacked = _median_stack_spectra(sat_specs, normalize_each=normalize_before_stack)
                    if stacked is not None:
                        band = _bootstrap_stack_band(
                            stacked["frequency_cpd"],
                            sat_specs,
                            normalize_each=normalize_before_stack,
                            n_bootstrap=bootstrap_replicates,
                            random_seed=bootstrap_seed,
                        )
                        if band is not None:
                            stacked["bootstrap_band"] = band
                        stacked_by_var[y_key] = stacked
        return {
            "spectra_by_variable": spectra_by_var,
            "stacked_by_variable": stacked_by_var,
            "selected_satellites": [str(x) for x in selected_ids],
            "mode": mode,
            "spectral_mode": "uniform_fft",
            "phase_variable": "true_anomaly",
            "phase_semantics": "TLE-derived Kepler proxy from mean anomaly",
            "time_meta": time_meta,
            "interpolation": interpolation,
            "cadence_seconds": cadence_seconds,
            "fft_backend": fft_backend,
            "gpu_min_samples": int(gpu_min_samples),
            "method_warning": None,
        }

    print(f"[fft_orbital_elements] Ready in {perf_counter() - t0:.2f}s")


# Consolidated from frequency_map_orbital_elements.py
def _window_starts(n_samples: int, window_size: int, step_size: int) -> np.ndarray:
    if n_samples <= 0 or window_size <= 0 or step_size <= 0 or n_samples < window_size:
        return np.array([], dtype=np.int64)
    return np.arange(0, n_samples - window_size + 1, step_size, dtype=np.int64)


def extract_windowed_dominant_frequencies(
    timestamps,
    values,
    *,
    window_size=256,
    step_size=64,
    cadence_seconds=None,
    min_freq_cpd=None,
    max_freq_cpd=None,
):
    t = np.asarray(timestamps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    t_u, y_u, cad = resample_to_uniform_grid(t, y, cadence_seconds=cadence_seconds)
    if y_u.size < int(window_size):
        return {
            "dominant_frequency_cpd": np.array([], dtype=np.float64),
            "window_center_days": np.array([], dtype=np.float64),
            "cadence_seconds": float(cad) if np.isfinite(cad) else np.nan,
            "status": "insufficient_samples",
        }

    starts = _window_starts(y_u.size, int(window_size), int(step_size))
    f_dom = []
    t_center = []

    for s in starts:
        e = s + int(window_size)
        t_seg = t_u[s:e]
        y_seg = y_u[s:e]
        spec = uniform_fft(t_seg, y_seg, cadence_seconds=cad, interpolation="linear")
        f = np.asarray(spec.get("frequency_cpd", []), dtype=np.float64)
        a = np.asarray(spec.get("amplitude", []), dtype=np.float64)
        if f.size == 0 or a.size != f.size:
            f_dom.append(np.nan)
            t_center.append(np.nan)
            continue

        mask = np.isfinite(f) & np.isfinite(a) & (f > 0)
        if min_freq_cpd is not None:
            mask &= f >= float(min_freq_cpd)
        if max_freq_cpd is not None:
            mask &= f <= float(max_freq_cpd)
        if not np.any(mask):
            f_dom.append(np.nan)
            t_center.append(np.nan)
            continue

        ff = f[mask]
        aa = a[mask]
        f_dom.append(float(ff[int(np.argmax(aa))]))
        t_center.append(float(np.mean(t_seg) / SECONDS_PER_DAY))

    return {
        "dominant_frequency_cpd": np.asarray(f_dom, dtype=np.float64),
        "window_center_days": np.asarray(t_center, dtype=np.float64),
        "cadence_seconds": float(cad),
        "window_size": int(window_size),
        "step_size": int(step_size),
        "status": "ok",
    }


def compare_dominant_frequencies(windowed_freqs_a, windowed_freqs_b):
    fa = np.asarray(windowed_freqs_a, dtype=np.float64)
    fb = np.asarray(windowed_freqs_b, dtype=np.float64)
    n = min(fa.size, fb.size)
    if n == 0:
        return {"delta_frequency_cpd": np.array([], dtype=np.float64), "n_compared": 0}
    d = fa[:n] - fb[:n]
    return {
        "delta_frequency_cpd": d,
        "mean_abs_delta_cpd": float(np.nanmean(np.abs(d))) if d.size else np.nan,
        "n_compared": int(n),
    }


def compute_frequency_diffusion_proxy(dominant_frequency_cpd):
    f = np.asarray(dominant_frequency_cpd, dtype=np.float64)
    valid = np.isfinite(f)
    f = f[valid]
    if f.size < 2:
        return {
            "diffusion_proxy": np.nan,
            "method": "std_of_adjacent_dominant_frequency_deltas",
            "n_valid": int(f.size),
        }
    df = np.diff(f)
    return {
        "diffusion_proxy": float(np.nanstd(df)),
        "method": "std_of_adjacent_dominant_frequency_deltas",
        "n_valid": int(f.size),
    }

