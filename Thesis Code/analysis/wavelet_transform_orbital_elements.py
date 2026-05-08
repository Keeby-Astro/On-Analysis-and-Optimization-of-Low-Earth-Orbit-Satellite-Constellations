"""Local time-frequency analysis

Scientific notes:
- Standard CWT assumes uniform sampling.
- WWZ provides an uneven-sampling time-frequency pathway.
"""

from __future__ import annotations

import warnings
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch

try:
    import ptwt
    _PTWT_AVAILABLE = True
except Exception:
    ptwt = None
    _PTWT_AVAILABLE = False

from spectral_preprocessing import (SECONDS_PER_DAY, build_uniform_resampling_metadata, build_elapsed_seconds,
                                    infer_positive_cadence_seconds, is_irregular_time_axis,
                                    resample_to_uniform_grid, unique_in_order, warn_irregular_resampled,
                                    zscore)

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def downsample_series(data, max_points):
    arr = np.asarray(data, dtype=np.float64)
    n = arr.size
    if n <= int(max_points):
        return arr, 1
    stride = int(np.ceil(n / float(max_points)))
    return arr[::stride], stride

def compute_cwt(data, scales, wavelet_name, sampling_period, return_complex=False):
    if _PTWT_AVAILABLE:
        scales_arr = np.asarray(scales, dtype=np.float32)
        try:
            signal = torch.as_tensor(data, dtype=torch.float32, device=TORCH_DEVICE)
            coeffs_t, freqs = ptwt.cwt(signal, scales_arr, wavelet_name, sampling_period=sampling_period)
        except Exception as exc:
            msg = str(exc).lower()
            is_oom = isinstance(exc, torch.OutOfMemoryError) or ("out of memory" in msg)
            if TORCH_DEVICE.type == "cuda" and is_oom:
                torch.cuda.empty_cache()
                signal = torch.as_tensor(data, dtype=torch.float32, device="cpu")
                coeffs_t, freqs = ptwt.cwt(signal, scales_arr, wavelet_name, sampling_period=sampling_period)
            else:
                raise

        coeffs_complex = coeffs_t.detach().cpu().numpy()
        abs_coeffs = np.abs(coeffs_complex)
        if return_complex:
            return abs_coeffs, np.asarray(freqs), coeffs_complex
        return abs_coeffs, np.asarray(freqs), None

    coeffs, freqs = pywt.cwt(data, scales, wavelet_name, sampling_period=sampling_period, method="fft")
    abs_coeffs = np.abs(coeffs)
    if return_complex:
        return abs_coeffs, np.asarray(freqs), coeffs
    return abs_coeffs, np.asarray(freqs), None

def _build_period_grid(period_min_days, period_max_days, n_periods):
    pmin = max(float(period_min_days), 1e-6)
    pmax = max(float(period_max_days), pmin * 1.05)
    return np.geomspace(pmin, pmax, int(max(16, n_periods)), dtype=np.float64)

def _compute_wwz(t_days, y, period_grid_days, n_time_centers=96,
                 decay=0.0125):
    """First-pass WWZ implementation for uneven sampling.

    Uses a local weighted least-squares fit per (tau, frequency):
        y(t) ~ a*cos(w(t-tau)) + b*sin(w(t-tau)) + c
    with Gaussian weights exp(-decay*w^2*(t-tau)^2).
    """
    t = np.asarray(t_days, dtype=np.float64)
    v = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(v)
    t = t[valid]
    v = v[valid]
    if t.size < 16:
        return {"wwz_power": np.zeros((0, 0), dtype=np.float64), "tau_days": np.array([], dtype=np.float64),
                "frequency_cpd": np.array([], dtype=np.float64), "effective_n": np.zeros((0, 0), dtype=np.float64),
                "status": "insufficient_samples"}

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[order]

    tau = np.linspace(float(t[0]), float(t[-1]), int(max(8, n_time_centers)), dtype=np.float64)
    freq_cpd = 1.0 / np.asarray(period_grid_days, dtype=np.float64)

    wwz = np.full((freq_cpd.size, tau.size), np.nan, dtype=np.float64)
    neff = np.zeros((freq_cpd.size, tau.size), dtype=np.float64)

    for i, f in enumerate(freq_cpd):
        w = 2.0 * np.pi * float(f)
        for j, tj in enumerate(tau):
            arg = w * (t - tj)
            wt = np.exp(-float(decay) * (w ** 2) * ((t - tj) ** 2))
            if not np.any(np.isfinite(wt)) or np.sum(wt) <= 1e-12:
                continue

            cos_col = np.cos(arg)
            sin_col = np.sin(arg)
            one_col = np.ones_like(cos_col)
            X = np.column_stack((cos_col, sin_col, one_col))

            sw = np.sqrt(wt)
            Xw = X * sw[:, None]
            yw = v * sw
            try:
                beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            except Exception:
                continue

            fit = X @ beta
            ybar = np.sum(wt * v) / np.sum(wt)
            sst = np.sum(wt * (v - ybar) ** 2)
            ssr = np.sum(wt * (fit - ybar) ** 2)
            if sst <= 1e-18:
                continue

            n_eff = (np.sum(wt) ** 2) / max(np.sum(wt ** 2), 1e-18)
            neff[i, j] = n_eff
            # Foster-style statistic scaffold, stabilized for low effective sample counts.
            z = ((n_eff - 3.0) * ssr) / max(2.0 * (sst - ssr), 1e-18)
            wwz[i, j] = max(float(z), 0.0)

    return {"wwz_power": wwz, "tau_days": tau, "frequency_cpd": freq_cpd,
            "effective_n": neff, "status": "ok", "decay": float(decay)}

def _compute_wwz_torch(t_days, y, period_grid_days, n_time_centers=96, decay=0.0125,
                       device=None, freq_batch_size=16, compute_dtype=torch.float32,
                       max_working_elements=20_000_000):
    """Torch WWZ path with optional CUDA acceleration.

    Falls back to CPU execution if CUDA is unavailable or errors occur upstream.
    """
    t_np = np.asarray(t_days, dtype=np.float64)
    v_np = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(t_np) & np.isfinite(v_np)
    t_np = t_np[valid]
    v_np = v_np[valid]
    if t_np.size < 16:
        return {"wwz_power": np.zeros((0, 0), dtype=np.float64), "tau_days": np.array([], dtype=np.float64),
                "frequency_cpd": np.array([], dtype=np.float64), "effective_n": np.zeros((0, 0), dtype=np.float64),
                "status": "insufficient_samples"}

    order = np.argsort(t_np, kind="mergesort")
    t_np = t_np[order]
    v_np = v_np[order]

    tau_np = np.linspace(float(t_np[0]), float(t_np[-1]), int(max(8, n_time_centers)), dtype=np.float64)
    freq_np = 1.0 / np.asarray(period_grid_days, dtype=np.float64)

    torch_device = device if device is not None else TORCH_DEVICE
    freq_batch_size = int(max(1, freq_batch_size))
    with torch.no_grad():
        t = torch.as_tensor(t_np, dtype=compute_dtype, device=torch_device)
        v = torch.as_tensor(v_np, dtype=compute_dtype, device=torch_device)
        tau = torch.as_tensor(tau_np, dtype=compute_dtype, device=torch_device)

        wwz = torch.full((freq_np.size, tau_np.size), float("nan"), dtype=compute_dtype, device=torch_device)
        neff = torch.zeros((freq_np.size, tau_np.size), dtype=compute_dtype, device=torch_device)

        eye3 = torch.eye(3, dtype=compute_dtype, device=torch_device)
        tiny = torch.tensor(1e-18, dtype=compute_dtype, device=torch_device)
        ridge_eps = torch.tensor(1e-6, dtype=compute_dtype, device=torch_device)

        freq_w = torch.as_tensor(2.0 * np.pi * freq_np, dtype=compute_dtype, device=torch_device)
        n_samples = int(t.numel())
        target_elems = int(max(1_000_000, max_working_elements))

        start = 0
        while start < freq_w.numel():
            current_batch = min(freq_batch_size, freq_w.numel() - start)
            while True:
                end = start + current_batch
                w = freq_w[start:end].view(-1, 1, 1)
                try:
                    # Process tau centers in adaptive blocks to bound temporary tensor size.
                    max_tau_chunk = max(1, int(target_elems // max(1, n_samples * current_batch)))
                    tau_start = 0
                    while tau_start < tau.numel():
                        tau_end = min(tau_start + max_tau_chunk, tau.numel())
                        tau_blk = tau[tau_start:tau_end]

                        d = t[:, None] - tau_blk[None, :]
                        d2 = d * d

                        arg = w * d.unsqueeze(0)
                        wt = torch.exp(-float(decay) * (w * w) * d2.unsqueeze(0))

                        sum_w = torch.sum(wt, dim=1)
                        valid_tau = torch.isfinite(sum_w) & (sum_w > 1e-12)
                        if not torch.any(valid_tau):
                            wwz[start:end, tau_start:tau_end] = torch.nan
                            neff[start:end, tau_start:tau_end] = 0.0
                            tau_start = tau_end
                            continue

                        x0 = torch.cos(arg)
                        x1 = torch.sin(arg)
                        x2_b = torch.ones_like(x0)
                        X = torch.stack((x0, x1, x2_b), dim=-1)

                        xtwx = torch.einsum("btn,btnk,btnm->bnkm", wt, X, X)
                        xtwy = torch.einsum("btn,btnk,t->bnk", wt, X, v)

                        # Exclude tau bins with effectively no support from the solve.
                        valid_tau_4d = valid_tau[:, :, None, None]
                        valid_tau_3d = valid_tau[:, :, None]
                        xtwx = torch.where(valid_tau_4d, xtwx, eye3[None, None, :, :])
                        xtwy = torch.where(valid_tau_3d, xtwy, torch.zeros_like(xtwy))

                        # Tikhonov-style ridge regularization for small 3x3 normal equations.
                        xtwx = xtwx + ridge_eps * eye3[None, None, :, :]

                        # Use stable float64 solve on GPU when available; fallback to pseudo-inverse if singular.
                        solve_dtype = torch.float64 if torch_device.type == "cuda" else compute_dtype
                        xtwx_solve = xtwx.to(solve_dtype)
                        xtwy_solve = xtwy.to(solve_dtype)
                        try:
                            beta = torch.linalg.solve(xtwx_solve, xtwy_solve.unsqueeze(-1)).squeeze(-1)
                        except RuntimeError as solve_exc:
                            msg_solve = str(solve_exc).lower()
                            if "singular" in msg_solve or "not invertible" in msg_solve or "ill-conditioned" in msg_solve:
                                beta = torch.matmul(torch.linalg.pinv(xtwx_solve), xtwy_solve.unsqueeze(-1)).squeeze(-1)
                            else:
                                raise
                        beta = beta.to(compute_dtype)

                        fit = torch.einsum("btnk,bnk->btn", X, beta)
                        ybar = torch.sum(wt * v[None, :, None], dim=1) / torch.clamp(sum_w, min=1e-18)
                        sst = torch.sum(wt * (v[None, :, None] - ybar[:, None, :]) ** 2, dim=1)
                        ssr = torch.sum(wt * (fit - ybar[:, None, :]) ** 2, dim=1)

                        sum_w2 = torch.sum(wt * wt, dim=1)
                        n_eff = (sum_w * sum_w) / torch.clamp(sum_w2, min=1e-18)
                        z = ((n_eff - 3.0) * ssr) / torch.clamp(2.0 * (sst - ssr), min=1e-18)
                        z = torch.where(valid_tau & (sst > tiny), torch.clamp(z, min=0.0), torch.nan)

                        wwz[start:end, tau_start:tau_end] = z
                        neff[start:end, tau_start:tau_end] = torch.where(valid_tau, n_eff, torch.zeros_like(n_eff))
                        tau_start = tau_end
                    break
                except Exception as exc:
                    msg = str(exc).lower()
                    is_oom = isinstance(exc, torch.OutOfMemoryError) or ("out of memory" in msg)
                    if torch_device.type == "cuda" and is_oom and current_batch > 1:
                        torch.cuda.empty_cache()
                        current_batch = max(1, current_batch // 2)
                        continue
                    raise

            start += current_batch

    return {"wwz_power": wwz.detach().cpu().numpy(), "tau_days": tau_np, "frequency_cpd": freq_np,
            "effective_n": neff.detach().cpu().numpy(), "status": "ok", "decay": float(decay),}

def _coi_scaffold(elapsed_days, period_days):
    t = np.asarray(elapsed_days, dtype=np.float64)
    p = np.asarray(period_days, dtype=np.float64)
    if t.size == 0 or p.size == 0:
        return {"left_days": np.array([]), "right_days": np.array([]), "status": "empty"}
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    if span <= 0:
        return {"left_days": np.array([]), "right_days": np.array([]), "status": "degenerate"}
    edge = 0.5 * p
    return {"left_days": t[0] + edge, "right_days": t[-1] - edge, "status": "scaffold"}

def _ridge_scaffold(power_2d, frequency_cpd):
    pw = np.asarray(power_2d, dtype=np.float64)
    f = np.asarray(frequency_cpd, dtype=np.float64)
    if pw.ndim != 2 or pw.shape[0] == 0 or pw.shape[1] == 0 or f.size != pw.shape[0]:
        return {"status": "empty", "ridge_frequency_cpd": np.array([], dtype=np.float64),
                "ridge_amplitude": np.array([], dtype=np.float64), "ridge_valid_mask": np.array([], dtype=bool)}
    finite_mask = np.isfinite(pw)
    if not np.any(finite_mask):
        return {"status": "empty", "ridge_frequency_cpd": np.array([], dtype=np.float64),
                "ridge_amplitude": np.array([], dtype=np.float64), "ridge_valid_mask": np.array([], dtype=bool)}

    pw_work = np.where(finite_mask, pw, -np.inf)
    idx = np.argmax(pw_work, axis=0)
    col_has_finite = np.any(finite_mask, axis=0)

    ridge_f = np.full(pw.shape[1], np.nan, dtype=np.float64)
    ridge_a = np.full(pw.shape[1], np.nan, dtype=np.float64)
    valid_cols = np.where(col_has_finite)[0]
    ridge_f[valid_cols] = f[idx[valid_cols]]
    ridge_a[valid_cols] = pw_work[idx[valid_cols], valid_cols]
    ridge_valid = np.isfinite(ridge_f) & np.isfinite(ridge_a)
    return {"status": "scaffold", "ridge_frequency_cpd": ridge_f,
            "ridge_amplitude": ridge_a, "ridge_valid_mask": ridge_valid}

def _local_maxima_mask(surface_2d):
    arr = np.asarray(surface_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros_like(arr, dtype=bool)

    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=bool)

    work = np.where(valid, arr, -np.inf)
    is_local_max = valid.copy()

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue

            shifted = np.full_like(work, -np.inf)

            src_i0 = max(0, -di)
            src_i1 = work.shape[0] - max(0, di)
            src_j0 = max(0, -dj)
            src_j1 = work.shape[1] - max(0, dj)
            if src_i1 <= src_i0 or src_j1 <= src_j0:
                continue

            dst_i0 = max(0, di)
            dst_j0 = max(0, dj)
            dst_i1 = dst_i0 + (src_i1 - src_i0)
            dst_j1 = dst_j0 + (src_j1 - src_j0)
            shifted[dst_i0:dst_i1, dst_j0:dst_j1] = work[src_i0:src_i1, src_j0:src_j1]
            is_local_max &= work >= shifted

    return is_local_max

def _extract_wwz_peaks(power_2d, tau_days, period_days, top_k=5, min_prominence=None,
                       min_separation_tau_bins=2, min_separation_period_bins=2):
    pw = np.asarray(power_2d, dtype=np.float64)
    tau = np.asarray(tau_days, dtype=np.float64)
    periods = np.asarray(period_days, dtype=np.float64)

    if pw.ndim != 2 or pw.shape[0] == 0 or pw.shape[1] == 0:
        return []
    if tau.size != pw.shape[1] or periods.size != pw.shape[0]:
        return []

    finite = np.isfinite(pw)
    if not np.any(finite):
        return []

    local_mask = _local_maxima_mask(pw)
    candidates = np.argwhere(local_mask)
    if candidates.size == 0:
        max_idx = np.unravel_index(int(np.nanargmax(np.where(finite, pw, -np.inf))), pw.shape)
        candidates = np.asarray([[max_idx[0], max_idx[1]]], dtype=np.int64)

    baseline = float(np.nanmedian(pw[finite]))
    values = pw[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(values)[::-1]

    sep_tau = int(max(0, min_separation_tau_bins))
    sep_period = int(max(0, min_separation_period_bins))
    k_limit = int(max(1, top_k))

    selected = []
    for idx in order.tolist():
        p_idx = int(candidates[idx, 0])
        t_idx = int(candidates[idx, 1])
        val = float(pw[p_idx, t_idx])
        if not np.isfinite(val):
            continue

        prominence = float(val - baseline)
        if min_prominence is not None:
            try:
                min_prom = float(min_prominence)
            except Exception:
                min_prom = None
            if min_prom is not None and np.isfinite(min_prom) and prominence < min_prom:
                continue

        too_close = False
        for row in selected:
            if (abs(int(row["period_index"]) - p_idx) < sep_period
                and abs(int(row["tau_index"]) - t_idx) < sep_tau):
                too_close = True
                break
        if too_close:
            continue

        selected.append({"tau_index": int(t_idx), "period_index": int(p_idx),
                         "tau_days": float(tau[t_idx]), "period_days": float(periods[p_idx]),
                         "power": val, "prominence": prominence})
        if len(selected) >= k_limit:
            break

    for rank, row in enumerate(selected, start=1):
        row["rank"] = int(rank)
    return selected

def _aggregate_wavelet_items(sat_items, method):
    method_key = str(method).lower()
    if method_key == "wwz":
        power_key = "wwz_power"
        x_key = "tau_days"
    else:
        power_key = "coeff_abs"
        x_key = "elapsed_days"

    prepared = []
    for sat_id, item in sat_items:
        pw = np.asarray(item.get(power_key, np.array([])), dtype=np.float64)
        x = np.asarray(item.get(x_key, np.array([])), dtype=np.float64)
        periods = np.asarray(item.get("period_days", np.array([])), dtype=np.float64)
        eff_n = np.asarray(item.get("effective_n", np.array([])), dtype=np.float64)
        wwz_backend = str(item.get("wwz_backend", "unknown"))
        if pw.ndim != 2 or pw.shape[0] == 0 or pw.shape[1] == 0:
            continue
        if x.size != pw.shape[1] or periods.size != pw.shape[0]:
            continue
        if eff_n.shape != pw.shape:
            eff_n = np.full(pw.shape, np.nan, dtype=np.float64)
        prepared.append((sat_id, pw, x, periods, eff_n, wwz_backend))

    if not prepared:
        return None

    x_ref = prepared[0][2]
    periods_ref = prepared[0][3]
    mats = []
    effective_n_mats = []
    used_sats = []
    backend_counts = {}

    for sat_id, pw, x_i, periods_i, eff_n_i, backend_i in prepared:
        if periods_i.size != periods_ref.size or not np.allclose(periods_i, periods_ref, rtol=1e-6, atol=1e-8, equal_nan=True):
            continue

        if x_i.size != x_ref.size or not np.allclose(x_i, x_ref, rtol=1e-6, atol=1e-8, equal_nan=True):
            pw_i = np.full((periods_ref.size, x_ref.size), np.nan, dtype=np.float64)
            eff_i = np.full((periods_ref.size, x_ref.size), np.nan, dtype=np.float64)
            for row in range(periods_ref.size):
                pw_i[row, :] = np.interp(x_ref, x_i, pw[row, :], left=np.nan, right=np.nan)
                eff_i[row, :] = np.interp(x_ref, x_i, eff_n_i[row, :], left=np.nan, right=np.nan)
        else:
            pw_i = pw
            eff_i = eff_n_i

        if not np.any(np.isfinite(pw_i)):
            continue
        mats.append(pw_i)
        effective_n_mats.append(eff_i)
        used_sats.append(sat_id)
        backend_counts[backend_i] = int(backend_counts.get(backend_i, 0) + 1)

    if not mats:
        return None

    stack = np.stack(mats, axis=0)
    stack_eff = np.stack(effective_n_mats, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pw_combined = np.nanmean(stack, axis=0)
        eff_combined = np.nanmean(stack_eff, axis=0)

    return {"power": pw_combined, "effective_n": eff_combined, "x": np.asarray(x_ref, dtype=np.float64),
            "period_days": np.asarray(periods_ref, dtype=np.float64), "used_satellites": used_sats,
            "candidate_count": int(len(sat_items)), "backend_counts": backend_counts}


def wavelet_transform_orbital_elements(inclinations, semi_major_axes, right_ascensions, args_of_perigee,
                                       eccentricities, true_anomalies, phase_mode=None, phase_series=None,
                                       timestamps=None, satellite_ids=None, selected_satellites=None,
                                       cadence_seconds=None, interpolation="linear", irregular_policy="resample",
                                       show_plots=True, return_results=False, method="wwz", scales=None,
                                       wavelet_name="cmor1.5-1.0", return_complex=False, include_significance=False,
                                       include_coi=True, include_ridge=True, wwz_period_min_days=0.5,
                                       wwz_period_max_days=120.0, wwz_n_periods=96, wwz_n_time_centers=96,
                                       wwz_decay=0.0125, wwz_use_gpu=True, wwz_freq_batch_size=16,
                                       irregular_warning_mode="once", wwz_extract_peaks=False, wwz_peak_top_k=5,
                                       wwz_peak_min_prominence=None, wwz_peak_min_separation_tau_bins=2,
                                       wwz_peak_min_separation_period_bins=2, wwz_overlay_ridge=False,
                                       wwz_annotate_panels=False, wwz_min_effective_n=None, wwz_export_combined_summary=True):
    """WWZ analysis with fixed uneven-sampling semantics.

    Hard-cleanup policy:
    - method is fixed to `wwz`.
    - irregular_policy is fixed to `resample` metadata semantics.
    """
    t0 = perf_counter()
    backend = "torch-cuda" if (bool(wwz_use_gpu) and TORCH_DEVICE.type == "cuda") else "numpy"

    inclinations = np.asarray(inclinations, dtype=np.float64)
    semi_major_axes = np.asarray(semi_major_axes, dtype=np.float64)
    right_ascensions = np.asarray(right_ascensions, dtype=np.float64)
    args_of_perigee = np.asarray(args_of_perigee, dtype=np.float64)
    eccentricities = np.asarray(eccentricities, dtype=np.float64)
    true_anomalies = np.asarray(true_anomalies, dtype=np.float64)

    n_rows = inclinations.size
    elapsed_seconds, time_meta = build_elapsed_seconds(timestamps, n_rows, fallback_units="samples",
                                                       warning_prefix="wavelet_transform_orbital_elements")

    if satellite_ids is None:
        sat_ids = np.array(["sat_0"] * n_rows)
    else:
        sat_ids = np.asarray(satellite_ids)
        if sat_ids.size != n_rows:
            warnings.warn("wavelet_transform_orbital_elements: satellite_ids length mismatch; treating as single satellite.",
                          RuntimeWarning, stacklevel=2)
            sat_ids = np.array(["sat_0"] * n_rows)

    if phase_mode is not None and phase_series is not None:
        phase_arr = np.asarray(phase_series, dtype=np.float64)
        if phase_arr.shape != true_anomalies.shape:
            raise ValueError("phase_series must match true_anomalies shape")

    unique_sats = unique_in_order(sat_ids.astype(str))
    if selected_satellites is None:
        selected_ids = unique_sats
    else:
        selected_set = {str(x) for x in selected_satellites}
        selected_ids = [sid for sid in unique_sats if sid in selected_set]
    if not selected_ids:
        selected_ids = unique_sats[:1]

    data_dict = {"Inclinations": inclinations, "Semi-major Axis": semi_major_axes,
                 "RAAN": right_ascensions, "Argument of Perigee": args_of_perigee,
                 "Eccentricities": eccentricities, "True Anomaly (TLE Kepler proxy)": true_anomalies}

    compute_ids = selected_ids
    results_by_satellite = {}
    sat_ids_str = sat_ids.astype(str)
    irregular_count = 0
    method_input = str(method).lower()
    if method_input != "wwz":
        raise ValueError("wavelet_transform_orbital_elements only supports method='wwz'.")
    irregular_policy = "resample"

    period_grid = _build_period_grid(wwz_period_min_days, wwz_period_max_days, wwz_n_periods)
    wwz_gpu_failed = False

    min_effective_n = None
    if wwz_min_effective_n is not None:
        try:
            min_effective_n = float(wwz_min_effective_n)
        except Exception:
            min_effective_n = None
        if min_effective_n is not None and (not np.isfinite(min_effective_n) or min_effective_n <= 0.0):
            min_effective_n = None

    combined_summary_by_element = {}
    combined_peak_table_by_element = {}
    combined_ridge_summary_by_element = {}
    combined_ridge_candidates_by_element = {}
    combined_support_mask_by_element = {}

    for sat_id in compute_ids:
        sat_mask = sat_ids_str == sat_id
        t_sat_seconds = elapsed_seconds[sat_mask]
        if t_sat_seconds.size < 8:
            continue

        irregular = is_irregular_time_axis(t_sat_seconds)
        if irregular:
            irregular_count += 1

        sat_series = {}
        for key, values in data_dict.items():
            y_sat_raw = np.asarray(values, dtype=np.float64)[sat_mask]

            t_days = t_sat_seconds / SECONDS_PER_DAY
            y_w = zscore(y_sat_raw)
            wwz_backend = "numpy"
            if bool(wwz_use_gpu) and TORCH_DEVICE.type == "cuda" and not wwz_gpu_failed:
                try:
                    wwz_out = _compute_wwz_torch(t_days, y_w, period_grid, n_time_centers=wwz_n_time_centers,
                                                 decay=wwz_decay, device=TORCH_DEVICE, freq_batch_size=wwz_freq_batch_size,
                                                 compute_dtype=torch.float32)
                    wwz_backend = "torch-cuda"
                except Exception as exc:
                    wwz_gpu_failed = True
                    warnings.warn(f"wavelet_transform_orbital_elements: GPU WWZ failed ({exc}); falling back to NumPy WWZ.",
                                  RuntimeWarning, stacklevel=2)
                    if TORCH_DEVICE.type == "cuda":
                        torch.cuda.empty_cache()
                    wwz_out = _compute_wwz(t_days, y_w, period_grid, n_time_centers=wwz_n_time_centers, decay=wwz_decay)
            else:
                wwz_out = _compute_wwz(t_days, y_w, period_grid, n_time_centers=wwz_n_time_centers, decay=wwz_decay)
            freq_cpd = np.asarray(wwz_out["frequency_cpd"], dtype=np.float64)
            period_days = np.full(freq_cpd.shape, np.inf, dtype=np.float64)
            pos = freq_cpd > 0
            period_days[pos] = 1.0 / freq_cpd[pos]

            cadence_used = infer_positive_cadence_seconds(t_sat_seconds)
            resample_meta = build_uniform_resampling_metadata(t_sat_seconds, np.asarray([], dtype=np.float64),
                                                              used_cadence_seconds=cadence_used, interpolation_method=interpolation,
                                                              resampled=False, input_irregular=irregular)

            sat_item = {"method": "wwz", "wwz_power": np.asarray(wwz_out["wwz_power"], dtype=np.float32),
                        "frequency_cpd": freq_cpd, "period_days": period_days, "tau_days": wwz_out["tau_days"],
                        "effective_n": np.asarray(wwz_out["effective_n"], dtype=np.float32), "irregular_input": bool(irregular),
                        "status": wwz_out.get("status", "ok"), "decay": wwz_out.get("decay"), "wwz_backend": wwz_backend,
                        "uniform_grid_seconds": np.asarray([], dtype=np.float64)}
            sat_item.update(resample_meta)
            if include_coi:
                sat_item["coi"] = _coi_scaffold(wwz_out["tau_days"], period_days)
            if include_ridge:
                sat_item["ridge"] = _ridge_scaffold(wwz_out["wwz_power"], freq_cpd)
            if include_significance:
                sat_item["significance"] = {"status": "scaffold", "note": "WWZ significance bootstrap scaffold not yet implemented.",
                                            "significance_status": "not_computed",
                                            "significance_note": "Descriptive WWZ power map only; no surrogate significance test executed.",
                                            "surrogate_test_status": "not_run"}
            sat_series[key] = sat_item

        if sat_series:
            results_by_satellite[sat_id] = sat_series

    if results_by_satellite:
        element_names = []
        for sat_results in results_by_satellite.values():
            for name in sat_results.keys():
                if name not in element_names:
                    element_names.append(name)

        combined_payload_by_element = {}
        for name in element_names:
            sat_items = []
            for sat_id, sat_results in results_by_satellite.items():
                if name in sat_results:
                    sat_items.append((sat_id, sat_results[name]))

            agg = _aggregate_wavelet_items(sat_items, "wwz")
            if agg is None:
                continue

            coeff_raw = np.asarray(agg.get("power", np.array([])), dtype=np.float64)
            x_axis = np.asarray(agg.get("x", np.array([])), dtype=np.float64)
            period_days = np.asarray(agg.get("period_days", np.array([])), dtype=np.float64)
            effective_n_map = np.asarray(agg.get("effective_n", np.array([])), dtype=np.float64)
            backend_counts = dict(agg.get("backend_counts", {}))
            n_used = int(len(agg.get("used_satellites", [])))
            n_total = int(agg.get("candidate_count", 0))

            if coeff_raw.ndim != 2 or coeff_raw.size == 0 or x_axis.size == 0 or period_days.size == 0:
                continue

            low_support_mask = np.zeros_like(coeff_raw, dtype=bool)
            if min_effective_n is not None and effective_n_map.shape == coeff_raw.shape:
                low_support_mask = (~np.isfinite(effective_n_map)) | (effective_n_map < float(min_effective_n))

            coeff_analysis = np.where(low_support_mask, np.nan, coeff_raw) if np.any(low_support_mask) else coeff_raw
            finite_mask = np.isfinite(coeff_analysis)
            finite_count = int(np.sum(finite_mask))
            total_cells = int(coeff_analysis.size)

            strongest_tau = np.nan
            strongest_period = np.nan
            strongest_power = np.nan
            strongest_tau_idx = -1
            strongest_period_idx = -1
            if finite_count > 0:
                work = np.where(finite_mask, coeff_analysis, -np.inf)
                flat_idx = int(np.argmax(work))
                strongest_period_idx, strongest_tau_idx = np.unravel_index(flat_idx, coeff_analysis.shape)
                strongest_tau = float(x_axis[strongest_tau_idx])
                strongest_period = float(period_days[strongest_period_idx])
                strongest_power = float(coeff_analysis[strongest_period_idx, strongest_tau_idx])

            period_profile = np.nanmean(coeff_analysis, axis=1)
            tau_profile = np.nanmean(coeff_analysis, axis=0)
            dominant_period_days = np.nan
            dominant_time_center_days = np.nan
            if np.any(np.isfinite(period_profile)):
                dominant_period_days = float(period_days[int(np.nanargmax(period_profile))])
            if np.any(np.isfinite(tau_profile)):
                dominant_time_center_days = float(x_axis[int(np.nanargmax(tau_profile))])

            peak_rows = []
            if bool(wwz_extract_peaks):
                peak_rows = _extract_wwz_peaks(coeff_analysis, x_axis, period_days, top_k=int(max(1, wwz_peak_top_k)),
                                               min_prominence=wwz_peak_min_prominence,
                                               min_separation_tau_bins=int(max(0, wwz_peak_min_separation_tau_bins)),
                                               min_separation_period_bins=int(max(0, wwz_peak_min_separation_period_bins)))
            combined_peak_table_by_element[name] = peak_rows

            ridge_export = {"ridge_tau_days": [], "ridge_period_days": [], "ridge_power": [], "ridge_valid_mask": []}
            ridge_summary = {"status": "not_requested" if not include_ridge else "empty", "median_ridge_period_days": np.nan,
                             "ridge_period_iqr_days": np.nan, "ridge_tau_support_span_days": np.nan, "max_ridge_power": np.nan}
            ridge_plot_tau = np.array([], dtype=np.float64)
            ridge_plot_period = np.array([], dtype=np.float64)
            if include_ridge:
                freq_for_ridge = np.full(period_days.shape, np.nan, dtype=np.float64)
                pos_period = period_days > 0
                freq_for_ridge[pos_period] = 1.0 / period_days[pos_period]
                ridge = _ridge_scaffold(coeff_analysis, freq_for_ridge)
                ridge_freq = np.asarray(ridge.get("ridge_frequency_cpd", np.array([])), dtype=np.float64)
                ridge_amp = np.asarray(ridge.get("ridge_amplitude", np.array([])), dtype=np.float64)
                ridge_valid = np.asarray(ridge.get("ridge_valid_mask", np.array([])), dtype=bool)

                ridge_period = np.full(ridge_freq.shape, np.nan, dtype=np.float64)
                pos_freq = ridge_freq > 0
                ridge_period[pos_freq] = 1.0 / ridge_freq[pos_freq]
                ridge_tau = np.asarray(x_axis, dtype=np.float64)

                if ridge_tau.size != ridge_period.size:
                    n_min = int(min(ridge_tau.size, ridge_period.size))
                    ridge_tau = ridge_tau[:n_min]
                    ridge_period = ridge_period[:n_min]
                    ridge_amp = ridge_amp[:n_min]
                    ridge_valid = ridge_valid[:n_min]

                ridge_valid = ridge_valid & np.isfinite(ridge_tau) & np.isfinite(ridge_period) & np.isfinite(ridge_amp)
                ridge_export = {"ridge_tau_days": ridge_tau.tolist(), "ridge_period_days": ridge_period.tolist(),
                                "ridge_power": ridge_amp.tolist(), "ridge_valid_mask": ridge_valid.tolist()}

                ridge_summary["status"] = str(ridge.get("status", "scaffold"))
                if np.any(ridge_valid):
                    ridge_period_valid = ridge_period[ridge_valid]
                    ridge_tau_valid = ridge_tau[ridge_valid]
                    ridge_amp_valid = ridge_amp[ridge_valid]
                    q75 = float(np.nanquantile(ridge_period_valid, 0.75))
                    q25 = float(np.nanquantile(ridge_period_valid, 0.25))
                    ridge_summary.update({"median_ridge_period_days": float(np.nanmedian(ridge_period_valid)),
                                          "ridge_period_iqr_days": float(q75 - q25),
                                          "ridge_tau_support_span_days": float(np.nanmax(ridge_tau_valid) - np.nanmin(ridge_tau_valid)),
                                          "max_ridge_power": float(np.nanmax(ridge_amp_valid))})

                    ridge_plot_tau = ridge_tau_valid
                    ridge_plot_period = ridge_period_valid

            combined_ridge_candidates_by_element[name] = ridge_export
            combined_ridge_summary_by_element[name] = ridge_summary
            combined_support_mask_by_element[name] = low_support_mask

            combined_summary_by_element[name] = {"element": str(name),
                                                 "strongest_tau_days": float(strongest_tau) if np.isfinite(strongest_tau) else np.nan,
                                                 "strongest_period_days": float(strongest_period) if np.isfinite(strongest_period) else np.nan,
                                                 "strongest_tau_index": int(strongest_tau_idx), "strongest_period_index": int(strongest_period_idx),
                                                 "max_wwz_power": float(strongest_power) if np.isfinite(strongest_power) else np.nan,
                                                 "median_wwz_power_finite": float(np.nanmedian(coeff_analysis[finite_mask])) if finite_count > 0 else np.nan,
                                                 "fraction_finite_cells": float(finite_count / max(1, total_cells)),
                                                 "dominant_period_days": float(dominant_period_days) if np.isfinite(dominant_period_days) else np.nan,
                                                 "dominant_time_center_days": float(dominant_time_center_days) if np.isfinite(dominant_time_center_days) else np.nan,
                                                 "used_satellite_count": int(n_used), "candidate_satellite_count": int(n_total), "backend_counts": backend_counts,
                                                 "wwz_min_effective_n": None if min_effective_n is None else float(min_effective_n),
                                                 "low_support_fraction": float(np.mean(low_support_mask)) if low_support_mask.size > 0 else 0.0,
                                                 "significance_status": "not_computed",
                                                 "significance_note": "Descriptive WWZ prioritization map; surrogate significance is not implemented in this run.",
                                                 "surrogate_test_status": "not_run"}

            combined_payload_by_element[name] = {"coeff_raw": coeff_raw, "coeff_plot": coeff_analysis, "x_axis": x_axis,"period_days": period_days,
                                                 "n_used": n_used, "n_total": n_total, "ridge_tau": ridge_plot_tau, "ridge_period": ridge_plot_period}

        if show_plots and combined_payload_by_element:
            n_series = len(element_names)
            n_cols = 3
            n_rows_grid = int(np.ceil(n_series / n_cols))
            fig, axes = plt.subplots(n_rows_grid, n_cols, figsize=(16, 4 * n_rows_grid), constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()

            for i, name in enumerate(element_names):
                ax = axes[i]
                payload = combined_payload_by_element.get(name)
                if payload is None:
                    ax.set_title(f"WWZ: {name} [combined] (empty)")
                    continue

                coeff = np.asarray(payload["coeff_plot"], dtype=np.float64)
                x_axis = np.asarray(payload["x_axis"], dtype=np.float64)
                period_days = np.asarray(payload["period_days"], dtype=np.float64)
                n_used = int(payload["n_used"])
                n_total = int(payload["n_total"])

                if coeff.size == 0 or x_axis.size == 0 or period_days.size == 0:
                    ax.set_title(f"WWZ: {name} [combined] (empty)")
                    continue

                coeff_masked = np.ma.masked_invalid(coeff)
                im = ax.imshow(coeff_masked, extent=[x_axis[0], x_axis[-1], period_days[-1], period_days[0]], cmap="viridis",
                               aspect="auto", origin="lower", rasterized=True)
                ax.set_title(f"WWZ: {name} [combined {n_used}/{n_total} sats]")
                ax.set_xlabel("Time center tau (days)")
                ax.set_ylabel("Period (days)")

                if bool(wwz_overlay_ridge) and include_ridge:
                    ridge_tau = np.asarray(payload.get("ridge_tau", np.array([])), dtype=np.float64)
                    ridge_period = np.asarray(payload.get("ridge_period", np.array([])), dtype=np.float64)
                    if ridge_tau.size > 1 and ridge_period.size > 1:
                        ax.plot(ridge_tau, ridge_period, color="#f8fafc", linewidth=1.1, alpha=0.9)

                if bool(wwz_annotate_panels):
                    summary = combined_summary_by_element.get(name, {})
                    ann_text = (f"P*={summary.get('strongest_period_days', np.nan):.2f} d\n"
                                f"tau*={summary.get('strongest_tau_days', np.nan):.2f} d\n"
                                f"max={summary.get('max_wwz_power', np.nan):.2f}\n"
                                f"n={summary.get('used_satellite_count', 0)}")
                    ax.text(0.02, 0.98, ann_text, transform=ax.transAxes, va="top",  ha="left", fontsize=8, color="#f8fafc",
                            bbox={"facecolor": "black", "alpha": 0.35, "pad": 3, "edgecolor": "none"})

                fig.colorbar(im, ax=ax, label="WWZ power")

            for j in range(n_series, axes.size):
                axes[j].set_visible(False)

            backend_hist = {}
            for summary in combined_summary_by_element.values():
                counts = summary.get("backend_counts", {})
                for key, val in counts.items():
                    backend_hist[key] = int(backend_hist.get(key, 0) + int(val))
            backend_label = ", ".join(f"{k}:{v}" for k, v in sorted(backend_hist.items())) if backend_hist else backend

            fig.suptitle(f"Wavelet WWZ combined ({len(results_by_satellite)} processed) | time={time_meta['time_basis']} | irregular_policy={irregular_policy} | backend={backend_label}",
                         fontsize=12)
            plt.show()

    print(f"[wavelet_transform_orbital_elements] Ready in {perf_counter() - t0:.2f}s")
    if return_results:
        payload = {"results_by_satellite": results_by_satellite, "selected_satellites": selected_ids, "time_meta": time_meta,
                   "irregular_policy": irregular_policy, "irregular_warning_mode": irregular_warning_mode,
                   "irregular_satellite_count": int(irregular_count), "processed_satellite_count": int(len(compute_ids)),
                   "interpolation": interpolation, "cadence_seconds": cadence_seconds, "method": "wwz", "wavelet_backend": backend,
                   "phase_variable": "true_anomaly", "phase_semantics": "TLE-derived Kepler proxy from mean anomaly",
                   "wwz_extract_peaks": bool(wwz_extract_peaks), "wwz_peak_top_k": int(max(1, wwz_peak_top_k)),
                   "wwz_peak_min_prominence": wwz_peak_min_prominence,
                   "wwz_peak_min_separation_tau_bins": int(max(0, wwz_peak_min_separation_tau_bins)),
                   "wwz_peak_min_separation_period_bins": int(max(0, wwz_peak_min_separation_period_bins)),
                   "wwz_overlay_ridge": bool(wwz_overlay_ridge), "wwz_annotate_panels": bool(wwz_annotate_panels),
                   "wwz_min_effective_n": None if min_effective_n is None else float(min_effective_n),
                   "wwz_export_combined_summary": bool(wwz_export_combined_summary)}
        if bool(wwz_export_combined_summary):
            payload["combined_summary_by_element"] = combined_summary_by_element
            payload["combined_peak_table_by_element"] = combined_peak_table_by_element
            payload["combined_ridge_summary_by_element"] = combined_ridge_summary_by_element
            payload["combined_ridge_candidates_by_element"] = combined_ridge_candidates_by_element
            payload["combined_support_mask_by_element"] = combined_support_mask_by_element
        return payload