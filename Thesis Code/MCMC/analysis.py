"""Postprocess constellation MCMC outputs for ballistic coefficient studies.

This script loads per-satellite MCMC CSV outputs, computes per-satellite diagnostics
and summaries, builds constellation-level combined analyses, generates figures, and
writes machine-readable and markdown reports.

Expected inputs per satellite stem:
	- *_posterior_samples.csv
	- *_chain_traces.csv (optional / may be incomplete)
	- *_observations_used.csv

The pipeline is designed to work in headless batch mode and avoid rerunning MCMC.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, wasserstein_distance

matplotlib.use("Agg")

LOGGER = logging.getLogger("constellation_mcmc_analysis")
POSTERIOR_SUFFIX = "_posterior_samples.csv"
TRACE_SUFFIX = "_chain_traces.csv"
OBS_SUFFIX = "_observations_used.csv"
DEFAULT_SEED = 42

plt.rcParams.update({
    "figure.figsize": (9.9, 7.5),
    "xtick.direction": "in", "xtick.labelsize": 14, "xtick.major.size": 3,
    "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.5,
    "xtick.minor.visible": True, "xtick.top": True,
    "ytick.direction": "in", "ytick.labelsize": 14, "ytick.major.size": 3,
    "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.5,
    "ytick.minor.visible": True, "ytick.right": True,
    "axes.linewidth": 0.5, "grid.linewidth": 0.5, "lines.linewidth": 1.0,
    "legend.fontsize": 14, "legend.frameon": False,
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.labelsize": 16, "axes.titlesize": 18,
    "axes.grid": False, "grid.linestyle": "--", "grid.color": "0.5",
    "lines.markersize": 8, "axes.spines.top": True, "axes.spines.right": True
})

colors = ["#1965B0", "#E8601C", "#4EB265", "#72190E", "#882E72",
          "#437DBF", "#F1932D", "#90C987", "#A5170E", "#994F88",
          "#6195CF", "#F6C141", "#CAE0AB", "#DC050C", "#AA6F9E",
          "#7BAFDE", "#F7F056", "#8B8B8B", "#896D67", "#BA8DB4"]

@dataclass(slots=True)
class SatelliteGroup:
	"""Container for file paths associated with one satellite result set."""
	stem: str
	posterior_path: Path | None
	trace_path: Path | None
	obs_path: Path | None

	@property
	def status(self) -> str:
		"""Return matching status across expected artifacts."""
		has_posterior = self.posterior_path is not None
		has_trace = self.trace_path is not None
		has_obs = self.obs_path is not None
		if has_posterior and has_obs and has_trace:
			return "complete"
		if has_posterior and has_obs:
			return "missing_trace"
		if has_posterior:
			return "posterior_only"
		if has_obs:
			return "obs_only"
		if has_trace:
			return "trace_only"
		return "unknown"

@dataclass(slots=True)
class SatelliteData:
	"""Loaded and normalized data for one satellite."""
	stem: str
	satellite_id: str
	posterior: pd.DataFrame
	traces: pd.DataFrame | None
	observations: pd.DataFrame
	warnings: list[str]

@dataclass(slots=True)
class ChainArrays:
	"""Chain-by-draw arrays used for ArviZ diagnostics."""
	bc: np.ndarray
	beta: np.ndarray
	log_posterior: np.ndarray
	chain_ids: list[int]
	draw_ids: list[int]

@dataclass(slots=True)
class PerSatDiagnostics:
	"""Diagnostic values and warnings per satellite."""

	rhat_bc: float | None
	ess_bulk_bc: float | None
	ess_tail_bc: float | None
	mcse_bc: float | None
	rhat_beta: float | None
	ess_bulk_beta: float | None
	ess_tail_beta: float | None
	mcse_beta: float | None
	warnings: list[str]

def setup_logging(verbose: bool) -> None:
	"""Configure root logging format and level."""
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Analyze constellation-level MCMC outputs for ballistic coefficient inference.")
	parser.add_argument("--input-dir", type=Path, required=True, help="Directory with MCMC CSV outputs.")
	parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"),
					 help="Output directory for reports, figures, and CSV artifacts.")
	parser.add_argument("--burnin-iters", type=int, default=500,
					 help="Burn-in iterations to discard (used if --burnin-frac is not provided).")
	parser.add_argument("--burnin-frac", type=float, default=None,
					 help="Burn-in fraction in [0,1). If provided, overrides --burnin-iters.")
	parser.add_argument("--hdi-prob", type=float, default=0.95,
					 help="Probability mass for highest-density intervals.")
	parser.add_argument("--min-draws-per-sat", type=int, default=None,
					 help="Optional lower bound on draws sampled per satellite for equal-weight mixture.")
	parser.add_argument("--kde-grid", type=int, default=400,
					 help="Number of points for KDE grid evaluations.")
	parser.add_argument("--save-combined-samples", action="store_true",
					 help="Save equal-weight combined posterior samples to CSV.")
	parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
					 help="Random seed for deterministic sampling operations.")
	return parser.parse_args()

def derive_stem(path: Path, suffix: str) -> str:
	"""Return stem base for a file that ends with a known suffix."""
	name = path.name
	if not name.endswith(suffix):
		raise ValueError(f"File {path} does not end with expected suffix {suffix}")
	return name[: -len(suffix)]

def discover_satellite_groups(input_dir: Path) -> tuple[list[SatelliteGroup], pd.DataFrame]:
	"""Discover and match satellite artifact groups by shared file stem."""
	posterior_files = sorted(input_dir.glob(f"*{POSTERIOR_SUFFIX}"))
	trace_files = sorted(input_dir.glob(f"*{TRACE_SUFFIX}"))
	obs_files = sorted(input_dir.glob(f"*{OBS_SUFFIX}"))

	stem_map: dict[str, SatelliteGroup] = {}

	for path in posterior_files:
		stem = derive_stem(path, POSTERIOR_SUFFIX)
		stem_map[stem] = SatelliteGroup(stem=stem, posterior_path=path, trace_path=None, obs_path=None)

	for path in trace_files:
		stem = derive_stem(path, TRACE_SUFFIX)
		group = stem_map.get(stem)
		if group is None:
			stem_map[stem] = SatelliteGroup(stem=stem, posterior_path=None, trace_path=path, obs_path=None)
		else:
			group.trace_path = path

	for path in obs_files:
		stem = derive_stem(path, OBS_SUFFIX)
		group = stem_map.get(stem)
		if group is None:
			stem_map[stem] = SatelliteGroup(stem=stem, posterior_path=None, trace_path=None, obs_path=path)
		else:
			group.obs_path = path

	groups = sorted(stem_map.values(), key=lambda g: g.stem)
	manifest = pd.DataFrame([{"stem": g.stem, "posterior_path": str(g.posterior_path) if g.posterior_path else "",
						   "trace_path": str(g.trace_path) if g.trace_path else "",
						   "obs_path": str(g.obs_path) if g.obs_path else "", "status": g.status} for g in groups])
	return groups, manifest

def parse_satellite_id(stem: str, posterior_df: pd.DataFrame) -> str:
	"""Parse stable satellite identifier from data and filename stem."""
	if "sat_file" in posterior_df.columns and posterior_df["sat_file"].notna().any():
		sat_value = str(posterior_df["sat_file"].dropna().iloc[0])
		sat_match = re.search(r"sat\d+", sat_value)
		if sat_match:
			return sat_match.group(0)
	stem_match = re.search(r"sat\d+", stem)
	if stem_match:
		return stem_match.group(0)
	return stem

def require_columns(df: pd.DataFrame, required: Iterable[str], file_path: Path) -> None:
	"""Raise an informative error if required columns are missing."""
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in {file_path}: {missing}")

def normalize_columns(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
	"""Rename known alias columns to canonical names when canonical is absent."""
	renamed = df.copy()
	for canonical, alt_names in aliases.items():
		if canonical in renamed.columns:
			continue
		for alt in alt_names:
			if alt in renamed.columns:
				renamed = renamed.rename(columns={alt: canonical})
				break
	return renamed

def load_summary_chain_hints(input_dir: Path) -> dict[str, int]:
	"""Load per-stem chain count hints from summary CSV if present."""
	summary_files = sorted(input_dir.glob("*mcmc_summary.csv"))
	if not summary_files:
		return {}
	try:
		summary_df = pd.read_csv(summary_files[0])
	except Exception as exc:
		LOGGER.warning("Failed to read summary file %s: %s", summary_files[0], exc)
		return {}
	if "sat_id" not in summary_df.columns or "n_chains" not in summary_df.columns:
		return {}
	hints: dict[str, int] = {}
	for _, row in summary_df.iterrows():
		try:
			hints[str(row["sat_id"])] = int(row["n_chains"])
		except Exception:
			continue
	return hints

def coerce_numeric(df: pd.DataFrame, columns: Iterable[str], file_path: Path) -> pd.DataFrame:
	"""Coerce selected columns to numeric and drop rows with NaN in those columns."""
	result = df.copy()
	for col in columns:
		result[col] = pd.to_numeric(result[col], errors="coerce")
	before = len(result)
	result = result.dropna(subset=list(columns)).reset_index(drop=True)
	dropped = before - len(result)
	if dropped > 0:
		LOGGER.warning("Dropped %d rows with invalid numeric values from %s", dropped, file_path)
	return result

def load_satellite_data(group: SatelliteGroup, n_chains_hint: int | None = None) -> SatelliteData:
	"""Load and normalize data for one satellite artifact group."""
	if group.posterior_path is None:
		raise ValueError(f"Satellite stem {group.stem} has no posterior samples file.")
	if group.obs_path is None:
		raise ValueError(f"Satellite stem {group.stem} has no observations file.")

	warnings: list[str] = []

	posterior = pd.read_csv(group.posterior_path)
	posterior = normalize_columns(posterior, {"chain_id": ["chain", "chain_index"],
										      "sample_id": ["sample", "draw", "draw_id"],
											  "log_posterior": ["log_post", "logprob", "log_probability"]})
	require_columns(posterior, ["beta_m2_per_kg", "bc_kg_per_m2"], group.posterior_path)

	if "log_posterior" not in posterior.columns:
		posterior["log_posterior"] = np.nan
		warnings.append("log_posterior missing in posterior samples; log-posterior diagnostics limited.")

	if "chain_id" not in posterior.columns or "sample_id" not in posterior.columns:
		n_rows = len(posterior)
		n_chains = int(n_chains_hint) if n_chains_hint is not None and int(n_chains_hint) > 0 else 1
		if n_rows % n_chains != 0:
			warnings.append(f"Cannot evenly split {n_rows} posterior rows across n_chains={n_chains}; using single synthetic chain.")
			n_chains = 1
		draws_per_chain = n_rows // n_chains if n_chains > 0 else n_rows
		posterior["chain_id"] = np.repeat(np.arange(n_chains), draws_per_chain)
		posterior["sample_id"] = np.tile(np.arange(draws_per_chain), n_chains)
		warnings.append(f"chain_id/sample_id missing; reconstructed synthetic chains with n_chains={n_chains}, draws_per_chain={draws_per_chain}.")
		posterior["__synthetic_chain"] = True
	else:
		posterior["__synthetic_chain"] = False
	posterior = coerce_numeric(posterior, ["chain_id", "sample_id", "beta_m2_per_kg", "bc_kg_per_m2"], group.posterior_path)
	posterior["log_posterior"] = pd.to_numeric(posterior["log_posterior"], errors="coerce")
	posterior["chain_id"] = posterior["chain_id"].astype(int)
	posterior["sample_id"] = posterior["sample_id"].astype(int)
	posterior = posterior.sort_values(["chain_id", "sample_id"]).reset_index(drop=True)

	observations = pd.read_csv(group.obs_path)
	observations = normalize_columns(observations, {"t_days": ["t_obs_days", "time_days", "t"],
												    "sma_obs_km": ["sma_km", "sma_obs"]})
	require_columns(observations, ["t_days", "sma_obs_km"], group.obs_path)
	observations = coerce_numeric(observations, ["t_days", "sma_obs_km"], group.obs_path)
	observations = observations.sort_values("t_days").reset_index(drop=True)

	traces: pd.DataFrame | None = None
	if group.trace_path is not None:
		traces_raw = pd.read_csv(group.trace_path)
		traces_raw = normalize_columns(traces_raw, {"chain_id": ["chain", "chain_index"],
											        "iter": ["iteration", "step"],
													"log_posterior": ["log_post", "logprob", "log_probability"]})
		if "bc_kg_per_m2" not in traces_raw.columns and "beta_m2_per_kg" in traces_raw.columns:
			beta_vals = pd.to_numeric(traces_raw["beta_m2_per_kg"], errors="coerce")
			with np.errstate(divide="ignore", invalid="ignore"):
				traces_raw["bc_kg_per_m2"] = 1.0 / beta_vals
			warnings.append("bc_kg_per_m2 missing in traces; derived as 1/beta_m2_per_kg.")

		require_columns(traces_raw, ["chain_id", "iter", "beta_m2_per_kg", "bc_kg_per_m2",
							         "log_posterior", "accepted", "proposal_sd_logbeta"], group.trace_path)
		traces = coerce_numeric(traces_raw, ["chain_id", "iter", "beta_m2_per_kg", "bc_kg_per_m2",
									         "log_posterior", "accepted", "proposal_sd_logbeta"], group.trace_path)
		traces["chain_id"] = traces["chain_id"].astype(int)
		traces["iter"] = traces["iter"].astype(int)
		traces["accepted"] = traces["accepted"].astype(int)
		traces = traces.sort_values(["chain_id", "iter"]).reset_index(drop=True)

	satellite_id = parse_satellite_id(group.stem, posterior)
	return SatelliteData(stem=group.stem, satellite_id=satellite_id, posterior=posterior,
					     traces=traces, observations=observations, warnings=warnings)

def get_burnin_cutoff(n_draws: int, burnin_iters: int, burnin_frac: float | None) -> int:
	"""Compute burn-in draw count from fraction or iteration count."""
	if burnin_frac is not None:
		if not (0 <= burnin_frac < 1):
			raise ValueError("--burnin-frac must be in [0, 1).")
		cutoff = int(math.floor(n_draws * burnin_frac))
	else:
		cutoff = max(0, burnin_iters)
	return min(cutoff, max(0, n_draws - 1))

def reconstruct_chain_draw_arrays(posterior: pd.DataFrame, burnin_iters: int, 
								  burnin_frac: float | None) -> tuple[ChainArrays | None, list[str], pd.DataFrame]:
	"""Reconstruct chain x draw arrays for key posterior variables.

	Returns chain arrays, warning messages, and posterior dataframe filtered for
	post-burn-in draws.
	"""
	warnings: list[str] = []
	chain_lengths = posterior.groupby("chain_id")["sample_id"].nunique().sort_index()
	if chain_lengths.empty:
		warnings.append("No valid chain samples found.")
		return None, warnings, posterior.iloc[0:0]

	min_len = int(chain_lengths.min())
	max_len = int(chain_lengths.max())
	if min_len != max_len:
		warnings.append(f"Inconsistent chain lengths detected (min={min_len}, max={max_len}); truncating to min length.")

	is_synthetic_chain = bool(posterior.get("__synthetic_chain", pd.Series([False])).iloc[0])
	cutoff = get_burnin_cutoff(min_len, burnin_iters=burnin_iters, burnin_frac=burnin_frac)
	if is_synthetic_chain and burnin_frac is None and cutoff >= min_len - 1:
		warnings.append("Posterior samples appear post-burn (synthetic chain reconstruction); overriding burnin-iters to 0.")
		cutoff = 0
	usable_len = min_len - cutoff
	if usable_len <= 1:
		warnings.append("Insufficient post-burn-in draws for diagnostics.")
		return None, warnings, posterior.iloc[0:0]

	chain_ids = sorted(posterior["chain_id"].unique().tolist())
	per_chain_frames: list[pd.DataFrame] = []
	for chain_id in chain_ids:
		chain_df = posterior[posterior["chain_id"] == chain_id].sort_values("sample_id").head(min_len)
		chain_df = chain_df.iloc[cutoff:].copy()
		per_chain_frames.append(chain_df)

	filtered = pd.concat(per_chain_frames, ignore_index=True)
	draw_ids = list(range(usable_len))
	n_chains = len(chain_ids)

	bc = np.empty((n_chains, usable_len), dtype=float)
	beta = np.empty((n_chains, usable_len), dtype=float)
	log_post = np.empty((n_chains, usable_len), dtype=float)

	for i, chain_df in enumerate(per_chain_frames):
		bc[i, :] = chain_df["bc_kg_per_m2"].to_numpy(dtype=float)
		beta[i, :] = chain_df["beta_m2_per_kg"].to_numpy(dtype=float)
		log_post[i, :] = chain_df["log_posterior"].to_numpy(dtype=float)

	arrays = ChainArrays(bc=bc, beta=beta, log_posterior=log_post,
					     chain_ids=chain_ids, draw_ids=draw_ids)
	return arrays, warnings, filtered

def build_inferencedata(arrays: ChainArrays) -> az.InferenceData:
	"""Build ArviZ InferenceData from chain arrays."""
	posterior_dict: dict[str, np.ndarray] = {"bc_kg_per_m2": arrays.bc, "beta_m2_per_kg": arrays.beta}
	if np.isfinite(arrays.log_posterior).any():
		posterior_dict["log_posterior"] = arrays.log_posterior
	return az.from_dict(posterior=posterior_dict)

def safe_extract_scalar(ds: xr.Dataset | xr.DataArray | None, var: str) -> float | None:
	"""Extract scalar float from ArviZ xarray output if available."""
	if ds is None:
		return None
	try:
		if isinstance(ds, xr.DataArray):
			val = ds.values
		else:
			if var not in ds:
				return None
			val = ds[var].values
		return float(np.asarray(val).squeeze())
	except Exception:
		return None

def compute_arviz_diagnostics(idata: az.InferenceData | None) -> PerSatDiagnostics:
	"""Compute R-hat, ESS, and MCSE diagnostics with graceful fallback."""
	warnings: list[str] = []
	if idata is None:
		return PerSatDiagnostics(rhat_bc=None, ess_bulk_bc=None, ess_tail_bc=None,
						         mcse_bc=None, rhat_beta=None, ess_bulk_beta=None,
		                         ess_tail_beta=None, mcse_beta=None,
								 warnings=["InferenceData unavailable; diagnostics skipped."])

	try:
		rhat_ds = az.rhat(idata, method="rank")
		ess_bulk_ds = az.ess(idata, method="bulk")
		ess_tail_ds = az.ess(idata, method="tail")
		mcse_ds = az.mcse(idata, method="mean")
	except Exception as exc:
		warnings.append(f"ArviZ diagnostics failed: {exc}")
		return PerSatDiagnostics(rhat_bc=None, ess_bulk_bc=None, ess_tail_bc=None,
						         mcse_bc=None, rhat_beta=None, ess_bulk_beta=None,
								 ess_tail_beta=None, mcse_beta=None, warnings=warnings)

	return PerSatDiagnostics(rhat_bc=safe_extract_scalar(rhat_ds, "bc_kg_per_m2"),
						     ess_bulk_bc=safe_extract_scalar(ess_bulk_ds, "bc_kg_per_m2"),
							 ess_tail_bc=safe_extract_scalar(ess_tail_ds, "bc_kg_per_m2"),
							 mcse_bc=safe_extract_scalar(mcse_ds, "bc_kg_per_m2"),
							 rhat_beta=safe_extract_scalar(rhat_ds, "beta_m2_per_kg"),
							 ess_bulk_beta=safe_extract_scalar(ess_bulk_ds, "beta_m2_per_kg"),
							 ess_tail_beta=safe_extract_scalar(ess_tail_ds, "beta_m2_per_kg"),
							 mcse_beta=safe_extract_scalar(mcse_ds, "beta_m2_per_kg"),
							 warnings=warnings)

def flatten_chain_arrays(arr: np.ndarray) -> np.ndarray:
	"""Return flattened array preserving order chain-major then draw."""
	return np.asarray(arr, dtype=float).reshape(-1)

def compute_interval(values: np.ndarray, prob: float) -> tuple[float, float]:
	"""Compute highest-density interval bounds for 1D samples."""
	hdi = az.hdi(values, hdi_prob=prob)
	return float(hdi[0]), float(hdi[1])

def compute_acceptance_metrics(traces: pd.DataFrame | None, burnin_iters: int,
							   burnin_frac: float | None) -> tuple[float | None, float | None, pd.DataFrame | None]:
	"""Compute mean and final acceptance rates from chain traces."""
	if traces is None or traces.empty:
		return None, None, None

	metrics_frames: list[pd.DataFrame] = []
	final_rates: list[float] = []
	mean_rates: list[float] = []

	for chain_id, df_chain in traces.groupby("chain_id"):
		chain_df = df_chain.sort_values("iter").copy()
		n_chain = len(chain_df)
		cutoff = get_burnin_cutoff(n_chain, burnin_iters=burnin_iters, burnin_frac=burnin_frac)
		if cutoff >= n_chain - 1:
			continue
		chain_df = chain_df.iloc[cutoff:].copy()
		accepted = chain_df["accepted"].astype(float)
		chain_df["accept_cum_rate"] = accepted.expanding().mean()
		chain_df["accept_roll_25"] = accepted.rolling(25, min_periods=5).mean()
		chain_df["accept_roll_50"] = accepted.rolling(50, min_periods=10).mean()
		chain_df["chain_id"] = int(chain_id)
		metrics_frames.append(chain_df)
		mean_rates.append(float(accepted.mean()))
		final_rates.append(float(chain_df["accept_cum_rate"].iloc[-1]))

	if not metrics_frames:
		return None, None, None

	metrics_df = pd.concat(metrics_frames, ignore_index=True)
	return float(np.mean(mean_rates)), float(np.mean(final_rates)), metrics_df

def kde_curve(samples: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
	"""Estimate KDE curve from 1D samples."""
	samples = np.asarray(samples, dtype=float)
	if len(samples) < 3 or np.allclose(samples.std(ddof=0), 0.0):
		x = np.linspace(samples.min() - 1e-6, samples.max() + 1e-6, max(grid_size, 10))
		y = np.zeros_like(x)
		y[len(y) // 2] = 1.0
		return x, y
	kde = gaussian_kde(samples)
	x = np.linspace(samples.min(), samples.max(), grid_size)
	y = kde.evaluate(x)
	return x, y

def weighted_quantile(values: np.ndarray, quantiles: list[float], weights: np.ndarray) -> np.ndarray:
	"""Compute weighted quantiles for 1D values."""
	values = np.asarray(values, dtype=float)
	weights = np.asarray(weights, dtype=float)
	sorter = np.argsort(values)
	values = values[sorter]
	weights = weights[sorter]
	cdf = np.cumsum(weights)
	cdf /= cdf[-1]
	return np.interp(np.asarray(quantiles), cdf, values)

def plot_autocorrelation(ax: plt.Axes, values: np.ndarray, max_lag: int = 50) -> None:
	"""Plot simple autocorrelation function for one chain."""
	series = np.asarray(values, dtype=float)
	series = series - np.mean(series)
	denom = np.dot(series, series)
	if denom <= 0:
		acf = np.zeros(max_lag + 1)
		acf[0] = 1.0
	else:
		acf = np.array([np.dot(series[: len(series) - lag], series[lag:]) / denom if lag < len(series) else np.nan
				        for lag in range(max_lag + 1)])
	ax.vlines(range(max_lag + 1), 0, acf, colors="tab:blue", linewidth=1.2)
	ax.axhline(0, color="black", linewidth=0.8)
	ax.set_title("Autocorrelation")
	ax.set_xlabel("Lag")
	ax.set_ylabel("ACF")

def make_per_satellite_plots(sat: SatelliteData, arrays: ChainArrays | None, filtered_posterior: pd.DataFrame,
							 accept_df: pd.DataFrame | None, out_dir: Path, hdi_prob: float,
							 kde_grid: int) -> list[str]:
	"""Create required per-satellite diagnostic plots and return warnings."""
	warnings: list[str] = []
	sat_dir = out_dir / "per_satellite" / sat.satellite_id
	sat_dir.mkdir(parents=True, exist_ok=True)

	if arrays is None:
		warnings.append("Missing chain arrays; trace/rank/autocorrelation plots skipped.")
	else:
		idata = build_inferencedata(arrays)

		trace_axes = az.plot_trace(idata, var_names=["bc_kg_per_m2", "beta_m2_per_kg"], compact=False)
		trace_fig = np.asarray(trace_axes).ravel()[0].figure
		trace_fig.suptitle(f"Trace Plot - {sat.satellite_id}", fontsize=11)
		trace_fig.tight_layout()
		trace_fig.savefig(sat_dir / "trace_plot.png", dpi=600, bbox_inches="tight")
		plt.close(trace_fig)

		rank_axes = az.plot_rank(idata, var_names=["bc_kg_per_m2", "beta_m2_per_kg"], kind="vlines")
		rank_fig = np.asarray(rank_axes).ravel()[0].figure
		rank_fig.suptitle(f"Rank Plot - {sat.satellite_id}", fontsize=11)
		rank_fig.tight_layout()
		rank_fig.savefig(sat_dir / "rank_plot.png", dpi=600, bbox_inches="tight")
		plt.close(rank_fig)

		fig, axes = plt.subplots(nrows=max(1, len(arrays.chain_ids)), ncols=1, figsize=(8, 2.6 * max(1, len(arrays.chain_ids))))
		if len(arrays.chain_ids) == 1:
			axes = np.array([axes])
		for i, chain_id in enumerate(arrays.chain_ids):
			plot_autocorrelation(axes[i], arrays.bc[i, :], max_lag=min(80, max(10, arrays.bc.shape[1] // 4)))
			axes[i].set_title(f"BC ACF - chain {chain_id}")
		fig.tight_layout()
		fig.savefig(sat_dir / "bc_autocorrelation.png", dpi=600, bbox_inches="tight")
		plt.close(fig)

		if np.isfinite(arrays.log_posterior).any():
			fig, ax = plt.subplots(figsize=(8, 4))
			for i, chain_id in enumerate(arrays.chain_ids):
				ax.plot(arrays.log_posterior[i, :], linewidth=0.9, alpha=0.9, label=f"chain {chain_id}")
			ax.set_title(f"Log Posterior Trace - {sat.satellite_id}")
			ax.set_xlabel("Draw")
			ax.set_ylabel("log_posterior")
			ax.legend(loc="best", fontsize=8)
			fig.tight_layout()
			fig.savefig(sat_dir / "log_posterior_trace.png", dpi=600, bbox_inches="tight")
			plt.close(fig)
		else:
			warnings.append("log_posterior unavailable; skipped log-posterior trace plot.")

	bc_samples = filtered_posterior["bc_kg_per_m2"].to_numpy(dtype=float)
	if len(bc_samples) > 0:
		fig, ax = plt.subplots(figsize=(8, 4.5))
		ax.hist(bc_samples, bins=40, density=True, alpha=0.35, color="tab:blue", edgecolor="white")
		x, y = kde_curve(bc_samples, grid_size=kde_grid)
		ax.plot(x, y, color="tab:blue", linewidth=1.8, label="KDE")
		median = float(np.median(bc_samples))
		hdi_low, hdi_high = compute_interval(bc_samples, hdi_prob)
		ax.axvline(median, color="black", linestyle="--", linewidth=1.2, label="Median")
		ax.axvline(hdi_low, color="tab:red", linestyle=":", linewidth=1.2)
		ax.axvline(hdi_high, color="tab:red", linestyle=":", linewidth=1.2, label=f"{int(hdi_prob*100)}% HDI")
		ax.set_title(f"BC Posterior - {sat.satellite_id}")
		ax.set_xlabel("BC [kg/m^2]")
		ax.set_ylabel("Density")
		ax.legend(loc="best", fontsize=8)
		fig.tight_layout()
		fig.savefig(sat_dir / "bc_posterior_hist_kde.png", dpi=600, bbox_inches="tight")
		plt.close(fig)
	else:
		warnings.append("No post-burn-in posterior samples available for BC histogram/KDE plot.")

	if accept_df is not None and not accept_df.empty:
		fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), sharex=True)
		for chain_id, df_chain in accept_df.groupby("chain_id"):
			axes[0].plot(df_chain["iter"], df_chain["accept_cum_rate"], label=f"chain {chain_id}", linewidth=1.2)
			axes[1].plot(df_chain["iter"], df_chain["accept_roll_25"], linewidth=1.2, alpha=0.8)
			axes[1].plot(df_chain["iter"], df_chain["accept_roll_50"], linewidth=1.2, alpha=0.8, linestyle="--")
		axes[0].set_title(f"Cumulative Acceptance - {sat.satellite_id}")
		axes[0].set_ylabel("Rate")
		axes[0].legend(loc="best", fontsize=8)
		axes[1].set_title("Rolling Acceptance (25 solid, 50 dashed)")
		axes[1].set_xlabel("Iteration")
		axes[1].set_ylabel("Rate")
		fig.tight_layout()
		fig.savefig(sat_dir / "acceptance_diagnostics.png", dpi=600, bbox_inches="tight")
		plt.close(fig)
	else:
		warnings.append("Acceptance diagnostics unavailable (missing/incomplete chain traces).")

	return warnings

def compute_per_satellite_summary(sat: SatelliteData, arrays: ChainArrays | None, filtered_posterior: pd.DataFrame,
								  diagnostics: PerSatDiagnostics, mean_accept_rate: float | None,
								  final_accept_rate: float | None, hdi_prob: float) -> dict[str, float | int | str | None]:
	"""Compute per-satellite summary row."""
	bc = filtered_posterior["bc_kg_per_m2"].to_numpy(dtype=float)
	beta = filtered_posterior["beta_m2_per_kg"].to_numpy(dtype=float)
	n_obs = int(len(sat.observations))
	obs_span = (float(sat.observations["t_days"].max() - sat.observations["t_days"].min()) if n_obs > 1 else 0.0)

	def stats_block(values: np.ndarray, prefix: str) -> dict[str, float | None]:
		if len(values) == 0:
			return {f"{prefix}_mean": None,
		            f"{prefix}_median": None,
				    f"{prefix}_sd": None,
				    f"{prefix}_hdi_95_low": None,
				    f"{prefix}_hdi_95_high": None}
		low, high = compute_interval(values, hdi_prob)
		return {f"{prefix}_mean": float(np.mean(values)),
			    f"{prefix}_median": float(np.median(values)),
			    f"{prefix}_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
			    f"{prefix}_hdi_95_low": low,
			    f"{prefix}_hdi_95_high": high}

	row: dict[str, float | int | str | None] = {
		"satellite_id": sat.satellite_id,
		"n_chains": int(arrays.bc.shape[0]) if arrays is not None else int(sat.posterior["chain_id"].nunique()),
		"n_draws_per_chain": int(arrays.bc.shape[1]) if arrays is not None else None,
		"n_obs": n_obs,
		"obs_span_days": obs_span,
		"mean_accept_rate": mean_accept_rate,
		"final_accept_rate": final_accept_rate,
		"rhat_bc": diagnostics.rhat_bc,
		"ess_bulk_bc": diagnostics.ess_bulk_bc,
		"ess_tail_bc": diagnostics.ess_tail_bc,
		"mcse_bc": diagnostics.mcse_bc,
		"rhat_beta": diagnostics.rhat_beta,
		"ess_bulk_beta": diagnostics.ess_bulk_beta,
		"ess_tail_beta": diagnostics.ess_tail_beta,
		"mcse_beta": diagnostics.mcse_beta,
	}
	row.update(stats_block(beta, "beta"))
	row.update(stats_block(bc, "bc"))

	if len(bc) > 0:
		row["bc_q05"] = float(np.quantile(bc, 0.05))
		row["bc_q95"] = float(np.quantile(bc, 0.95))
		row["bc_skew"] = float(stats.skew(bc, bias=False)) if len(bc) > 2 else None
		row["bc_kurtosis"] = float(stats.kurtosis(bc, fisher=True, bias=False)) if len(bc) > 3 else None
	else:
		row["bc_q05"] = None
		row["bc_q95"] = None
		row["bc_skew"] = None
		row["bc_kurtosis"] = None

	return row

def build_constellation_mixtures(
	per_sat_samples: dict[str, np.ndarray],
	ess_bulk_map: dict[str, float | None],
	rng: np.random.Generator,
	min_draws_per_sat: int | None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float | None]]:
	"""Build equal-weight and ESS-weighted constellation mixture summaries for BC."""
	valid = {k: v for k, v in per_sat_samples.items() if len(v) > 0}
	if not valid:
		return np.array([]), pd.DataFrame(), {
			"equal_median": None,
			"equal_hdi95_low": None,
			"equal_hdi95_high": None,
			"ess_weighted_median": None,
			"ess_weighted_q025": None,
			"ess_weighted_q975": None,
			"precision_weighted_mean": None,
			"precision_weighted_se": None,
		}

	min_draws = min(len(v) for v in valid.values())
	if min_draws_per_sat is not None:
		min_draws = min(min_draws, max(1, min_draws_per_sat))

	eq_blocks = []
	records = []
	for sat_id, samples in sorted(valid.items()):
		if len(samples) < min_draws:
			continue
		pick_idx = rng.choice(len(samples), size=min_draws, replace=False)
		picked = samples[pick_idx]
		eq_blocks.append(picked)
		for val in picked:
			records.append({"satellite_id": sat_id, "bc_kg_per_m2": float(val)})

	equal_samples = np.concatenate(eq_blocks) if eq_blocks else np.array([])
	equal_df = pd.DataFrame.from_records(records)

	ess_rows = []
	ess_sats = []
	ess_weights = []
	for sat_id, samples in sorted(valid.items()):
		w = ess_bulk_map.get(sat_id)
		if w is None or not np.isfinite(w) or w <= 0:
			continue
		ess_sats.append(sat_id)
		ess_weights.append(float(w))
		ess_rows.extend(
			[{"satellite_id": sat_id, "bc_kg_per_m2": float(v)} for v in samples]
		)
	ess_df = pd.DataFrame.from_records(ess_rows)

	summary: dict[str, float | None] = {
		"equal_median": None,
		"equal_hdi95_low": None,
		"equal_hdi95_high": None,
		"ess_weighted_median": None,
		"ess_weighted_q025": None,
		"ess_weighted_q975": None,
		"precision_weighted_mean": None,
		"precision_weighted_se": None,
	}

	if len(equal_samples) > 0:
		low, high = compute_interval(equal_samples, 0.95)
		summary["equal_median"] = float(np.median(equal_samples))
		summary["equal_hdi95_low"] = low
		summary["equal_hdi95_high"] = high

	if not ess_df.empty and ess_weights:
		sat_weight_map = {s: w for s, w in zip(ess_sats, ess_weights)}
		total_w = sum(ess_weights)
		if total_w > 0:
			sat_weight_map = {k: v / total_w for k, v in sat_weight_map.items()}
			weight_col = ess_df["satellite_id"].map(sat_weight_map).to_numpy(dtype=float)
			samples = ess_df["bc_kg_per_m2"].to_numpy(dtype=float)
			q50, q025, q975 = weighted_quantile(samples, [0.5, 0.025, 0.975], weight_col)
			summary["ess_weighted_median"] = float(q50)
			summary["ess_weighted_q025"] = float(q025)
			summary["ess_weighted_q975"] = float(q975)

	means = []
	variances = []
	for _, arr in valid.items():
		if len(arr) > 1:
			means.append(float(np.mean(arr)))
			variances.append(float(np.var(arr, ddof=1)))
	if means and variances:
		variances_arr = np.asarray(variances, dtype=float)
		means_arr = np.asarray(means, dtype=float)
		ok = variances_arr > 0
		if np.any(ok):
			inv_var = 1.0 / variances_arr[ok]
			w = inv_var / np.sum(inv_var)
			weighted_mean = float(np.sum(w * means_arr[ok]))
			weighted_se = float(math.sqrt(1.0 / np.sum(inv_var)))
			summary["precision_weighted_mean"] = weighted_mean
			summary["precision_weighted_se"] = weighted_se

	return equal_samples, ess_df, summary

def compute_pairwise_distances(per_sat_samples: dict[str, np.ndarray]) -> pd.DataFrame:
	"""Compute pairwise Wasserstein and Jensen-Shannon distances for BC posteriors."""
	rows: list[dict[str, float | str]] = []
	sat_ids = sorted(per_sat_samples.keys())
	for i, sat_i in enumerate(sat_ids):
		s_i = per_sat_samples[sat_i]
		for j, sat_j in enumerate(sat_ids):
			if j <= i:
				continue
			s_j = per_sat_samples[sat_j]
			if len(s_i) == 0 or len(s_j) == 0:
				continue
			wd = float(wasserstein_distance(s_i, s_j))

			lo = min(float(np.min(s_i)), float(np.min(s_j)))
			hi = max(float(np.max(s_i)), float(np.max(s_j)))
			if hi <= lo:
				jsd = 0.0
			else:
				bins = np.linspace(lo, hi, 80)
				p, _ = np.histogram(s_i, bins=bins, density=True)
				q, _ = np.histogram(s_j, bins=bins, density=True)
				p = p + 1e-12
				q = q + 1e-12
				p = p / p.sum()
				q = q / q.sum()
				jsd = float(jensenshannon(p, q, base=2.0))

			rows.append(
				{
					"satellite_i": sat_i,
					"satellite_j": sat_j,
					"wasserstein_bc": wd,
					"js_divergence_bc": jsd,
				}
			)
	return pd.DataFrame(rows)

def make_constellation_plots(
	per_sat_df: pd.DataFrame,
	per_sat_samples: dict[str, np.ndarray],
	per_sat_beta_samples: dict[str, np.ndarray],
	equal_samples: np.ndarray,
	ess_df: pd.DataFrame,
	pairwise_df: pd.DataFrame,
	out_dir: Path,
	kde_grid: int,
	rng: np.random.Generator,
) -> None:
	"""Create constellation-level plots."""
	plot_dir = out_dir / "constellation"
	plot_dir.mkdir(parents=True, exist_ok=True)

	def _plot_combined_constellation_diagnostics() -> None:
		valid_ids = [
			sat_id
			for sat_id in sorted(set(per_sat_samples).intersection(set(per_sat_beta_samples)))
			if len(per_sat_samples[sat_id]) > 0 and len(per_sat_beta_samples[sat_id]) > 0
		]
		if not valid_ids:
			return

		min_draws = min(min(len(per_sat_samples[s]), len(per_sat_beta_samples[s])) for s in valid_ids)
		if min_draws < 2:
			return

		bc_matrix = np.empty((len(valid_ids), min_draws), dtype=float)
		beta_matrix = np.empty((len(valid_ids), min_draws), dtype=float)
		for i, sat_id in enumerate(valid_ids):
			bc = per_sat_samples[sat_id]
			be = per_sat_beta_samples[sat_id]
			idx_bc = rng.choice(len(bc), size=min_draws, replace=False)
			idx_be = rng.choice(len(be), size=min_draws, replace=False)
			bc_matrix[i, :] = bc[idx_bc]
			beta_matrix[i, :] = be[idx_be]

		idata_combined = az.from_dict(
			posterior={
				"bc_kg_per_m2": bc_matrix,
				"beta_m2_per_kg": beta_matrix,
			}
		)

		trace_axes = az.plot_trace(idata_combined, var_names=["bc_kg_per_m2", "beta_m2_per_kg"], compact=False)
		trace_fig = np.asarray(trace_axes).ravel()[0].figure
		trace_fig.suptitle("Combined Constellation Trace Plot")
		trace_fig.tight_layout()
		trace_fig.savefig(plot_dir / "combined_trace_plot.png", dpi=600, bbox_inches="tight")
		plt.close(trace_fig)

		rank_axes = az.plot_rank(idata_combined, var_names=["bc_kg_per_m2", "beta_m2_per_kg"], kind="vlines")
		rank_fig = np.asarray(rank_axes).ravel()[0].figure
		rank_fig.suptitle("Combined Constellation Rank Plot")
		rank_fig.tight_layout()
		rank_fig.savefig(plot_dir / "combined_rank_plot.png", dpi=600, bbox_inches="tight")
		plt.close(rank_fig)

		bc_combined = bc_matrix.reshape(-1)
		fig, ax = plt.subplots(figsize=(8, 4.5))
		ax.hist(bc_combined, bins=50, density=True, alpha=0.35, color="tab:blue", edgecolor="white")
		x_bc, y_bc = kde_curve(bc_combined, grid_size=kde_grid)
		ax.plot(x_bc, y_bc, color="tab:blue", linewidth=1.8, label="KDE")
		med_bc = float(np.median(bc_combined))
		low_bc, high_bc = compute_interval(bc_combined, 0.95)
		ax.axvline(med_bc, color="black", linestyle="--", linewidth=1.2, label="Median")
		ax.axvline(low_bc, color="tab:red", linestyle=":", linewidth=1.2)
		ax.axvline(high_bc, color="tab:red", linestyle=":", linewidth=1.2, label="95% HDI")
		ax.set_title("Combined BC Posterior (Constellation)")
		ax.set_xlabel("BC [kg/m^2]")
		ax.set_ylabel("Density")
		ax.legend(loc="best", fontsize=8)
		fig.tight_layout()
		fig.savefig(plot_dir / "combined_bc_posterior_hist_kde.png", dpi=600, bbox_inches="tight")
		plt.close(fig)

		fig, ax = plt.subplots(figsize=(8, 4))
		plot_autocorrelation(ax, bc_combined, max_lag=min(120, max(20, len(bc_combined) // 20)))
		ax.set_title("Combined BC Autocorrelation")
		fig.tight_layout()
		fig.savefig(plot_dir / "combined_bc_autocorrelation.png", dpi=600, bbox_inches="tight")
		plt.close(fig)

		n_sat = bc_matrix.shape[0]
		bc_avg = np.mean(bc_matrix, axis=0)
		beta_avg = np.mean(beta_matrix, axis=0)
		fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
		for ax, samples, title, xlabel, color in [
			(
				axes[0],
				beta_avg,
				"Combined Posterior Beta Average (Constellation)",
				"Average beta per satellite [m^2/kg]",
				"tab:green",
			),
			(
				axes[1],
				bc_avg,
				"Combined Posterior BC Average (Constellation)",
				"Average BC per satellite [kg/m^2]",
				"tab:purple",
			),
		]:
			ax.hist(samples, bins=40, density=True, alpha=0.35, color=color, edgecolor="white")
			x, y = kde_curve(samples, grid_size=kde_grid)
			ax.plot(x, y, color=color, linewidth=1.8)
			med = float(np.median(samples))
			lo, hi = compute_interval(samples, 0.95)
			ax.axvline(med, color="black", linestyle="--", linewidth=1.2)
			ax.axvline(lo, color="tab:red", linestyle=":", linewidth=1.2)
			ax.axvline(hi, color="tab:red", linestyle=":", linewidth=1.2)
			ax.set_title(title)
			ax.set_xlabel(xlabel)
			ax.set_ylabel("Density")
		fig.tight_layout()
		fig.savefig(plot_dir / "combined_posterior_beta_bc_sum.png", dpi=600, bbox_inches="tight")
		plt.close(fig)

	_plot_combined_constellation_diagnostics()

	if not per_sat_df.empty:
		forest_df = per_sat_df.dropna(subset=["bc_median", "bc_hdi_95_low", "bc_hdi_95_high"]).copy()
		forest_df = forest_df.sort_values("bc_median", ascending=True).reset_index(drop=True)
		if not forest_df.empty:
			y = np.arange(len(forest_df))
			fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(forest_df))))
			med = forest_df["bc_median"].to_numpy(dtype=float)
			low95 = forest_df["bc_hdi_95_low"].to_numpy(dtype=float)
			high95 = forest_df["bc_hdi_95_high"].to_numpy(dtype=float)
			low50 = forest_df.get("bc_q05", pd.Series([np.nan] * len(forest_df))).to_numpy(dtype=float)
			high50 = forest_df.get("bc_q95", pd.Series([np.nan] * len(forest_df))).to_numpy(dtype=float)

			ax.hlines(y=y, xmin=low95, xmax=high95, color="tab:blue", alpha=0.9, linewidth=2.0, label="95% interval")
			ax.hlines(y=y, xmin=low50, xmax=high50, color="tab:blue", alpha=0.5, linewidth=5.0, label="~central interval")
			ax.plot(med, y, "o", color="black", markersize=4, label="Median")
			ax.set_yticks(y)
			ax.set_yticklabels(forest_df["satellite_id"].tolist())
			ax.set_xlabel("BC [kg/m^2]")
			ax.set_title("Forest Plot of Satellite BC Posteriors")
			handles, labels = ax.get_legend_handles_labels()
			if handles:
				ax.legend(loc="best", fontsize=8)
			fig.tight_layout()
			fig.savefig(plot_dir / "forest_plot_bc.png", dpi=600, bbox_inches="tight")
			plt.close(fig)

			fig, ax = plt.subplots(figsize=(9, max(6, 0.45 * len(forest_df))))
			offset = 0.0
			step = 1.0
			for sat_id in forest_df["satellite_id"]:
				samples = per_sat_samples.get(sat_id, np.array([]))
				if len(samples) == 0:
					continue
				x, yk = kde_curve(samples, grid_size=kde_grid)
				yk = yk / np.max(yk) if np.max(yk) > 0 else yk
				ax.fill_between(x, offset, offset + yk * 0.8, alpha=0.35)
				ax.plot(x, offset + yk * 0.8, linewidth=1.0)
				ax.text(float(np.min(x)), offset + 0.1, sat_id, fontsize=7, va="bottom")
				offset += step
			ax.set_title("Ridge/Stacked BC Densities by Satellite")
			ax.set_xlabel("BC [kg/m^2]")
			ax.set_ylabel("Stack index")
			fig.tight_layout()
			fig.savefig(plot_dir / "ridge_bc_posteriors.png", dpi=600, bbox_inches="tight")
			plt.close(fig)

			fig, ax = plt.subplots(figsize=(8, 4.5))
			if len(equal_samples) > 0:
				x_eq, y_eq = kde_curve(equal_samples, grid_size=kde_grid)
				ax.plot(x_eq, y_eq, color="tab:blue", linewidth=2.0, label="Equal-weight")
			if not ess_df.empty:
				x_ess = ess_df["bc_kg_per_m2"].to_numpy(dtype=float)
				sat_counts = ess_df.groupby("satellite_id").size().to_dict()
				w = np.array([1.0 / sat_counts[s] for s in ess_df["satellite_id"]], dtype=float)
				w = w / w.sum()
				bins = np.linspace(float(np.min(x_ess)), float(np.max(x_ess)), 120)
				hist, edges = np.histogram(x_ess, bins=bins, weights=w, density=False)
				ctr = 0.5 * (edges[:-1] + edges[1:])
				ax.plot(ctr, hist, color="tab:orange", linewidth=1.8, label="ESS-weighted (hist approx)")
			ax.set_title("Combined Constellation BC Distributions")
			ax.set_xlabel("BC [kg/m^2]")
			ax.set_ylabel("Density (relative)")
			handles, labels = ax.get_legend_handles_labels()
			if handles:
				ax.legend(loc="best", fontsize=8)
			fig.tight_layout()
			fig.savefig(plot_dir / "combined_bc_distributions.png", dpi=600, bbox_inches="tight")
			plt.close(fig)

			fig, ax = plt.subplots(figsize=(8, 4.5))
			width95 = forest_df["bc_hdi_95_high"].to_numpy(dtype=float) - forest_df["bc_hdi_95_low"].to_numpy(dtype=float)
			ax.scatter(forest_df["obs_span_days"].to_numpy(dtype=float), width95, s=30, alpha=0.8)
			for _, r in forest_df.iterrows():
				ax.annotate(r["satellite_id"], (float(r["obs_span_days"]), float(r["bc_hdi_95_high"] - r["bc_hdi_95_low"])), fontsize=7)
			ax.set_title("BC 95% Interval Width vs Observation Span")
			ax.set_xlabel("Observation span [days]")
			ax.set_ylabel("BC interval width")
			fig.tight_layout()
			fig.savefig(plot_dir / "interval_width_vs_obs_span.png", dpi=600, bbox_inches="tight")
			plt.close(fig)

			fig, ax = plt.subplots(figsize=(8, 4.5))
			ax.scatter(forest_df["n_obs"].to_numpy(dtype=float), forest_df["bc_median"].to_numpy(dtype=float), s=30, alpha=0.8)
			for _, r in forest_df.iterrows():
				ax.annotate(r["satellite_id"], (float(r["n_obs"]), float(r["bc_median"])), fontsize=7)
			ax.set_title("BC Median vs Number of Observations")
			ax.set_xlabel("Number of observations")
			ax.set_ylabel("BC median [kg/m^2]")
			fig.tight_layout()
			fig.savefig(plot_dir / "bc_median_vs_n_obs.png", dpi=600, bbox_inches="tight")
			plt.close(fig)

	if not pairwise_df.empty:
		sat_ids = sorted(set(pairwise_df["satellite_i"]).union(set(pairwise_df["satellite_j"])))
		idx_map = {s: i for i, s in enumerate(sat_ids)}
		matrix = np.zeros((len(sat_ids), len(sat_ids)), dtype=float)
		np.fill_diagonal(matrix, 0.0)
		for _, row in pairwise_df.iterrows():
			i = idx_map[str(row["satellite_i"])]
			j = idx_map[str(row["satellite_j"])]
			val = float(row["wasserstein_bc"])
			matrix[i, j] = val
			matrix[j, i] = val

		fig, ax = plt.subplots(figsize=(8, 6))
		im = ax.imshow(matrix, aspect="auto", cmap="viridis")
		ax.set_xticks(np.arange(len(sat_ids)))
		ax.set_yticks(np.arange(len(sat_ids)))
		ax.set_xticklabels(sat_ids, rotation=45, ha="right")
		ax.set_yticklabels(sat_ids)
		ax.set_title("Pairwise Wasserstein Distance Heatmap (BC)")
		fig.colorbar(im, ax=ax, label="Wasserstein distance")
		fig.tight_layout()
		fig.savefig(plot_dir / "pairwise_wasserstein_heatmap.png", dpi=600, bbox_inches="tight")
		plt.close(fig)

def write_per_satellite_markdown(
	out_dir: Path,
	per_sat_df: pd.DataFrame,
	sat_warnings: dict[str, list[str]],
) -> None:
	"""Write per-satellite diagnostics markdown report."""
	path = out_dir / "per_satellite_diagnostics.md"
	lines: list[str] = ["# Per-Satellite Diagnostics", ""]
	if per_sat_df.empty:
		lines.append("No satellites available for diagnostics.")
	else:
		for _, row in per_sat_df.sort_values("satellite_id").iterrows():
			sat_id = str(row["satellite_id"])
			lines.append(f"## {sat_id}")
			lines.append("")
			lines.append(f"- Chains: {row['n_chains']}")
			lines.append(f"- Draws/chain (post-burn): {row['n_draws_per_chain']}")
			lines.append(f"- Observations: {row['n_obs']}")
			lines.append(f"- BC median: {row['bc_median']}")
			lines.append(f"- BC 95% HDI: [{row['bc_hdi_95_low']}, {row['bc_hdi_95_high']}]")
			lines.append(f"- R-hat (BC): {row['rhat_bc']}")
			lines.append(f"- ESS bulk/tail (BC): {row['ess_bulk_bc']}, {row['ess_tail_bc']}")
			lines.append(f"- MCSE (BC): {row['mcse_bc']}")
			warn_list = sat_warnings.get(sat_id, [])
			if warn_list:
				lines.append("- Warnings:")
				for w in warn_list:
					lines.append(f"  - {w}")
			lines.append("")
	path.write_text("\n".join(lines), encoding="utf-8")

def write_constellation_markdown(
	out_dir: Path,
	n_sats: int,
	mix_summary: dict[str, float | None],
	pairwise_df: pd.DataFrame,
) -> None:
	"""Write constellation-level markdown report."""
	path = out_dir / "constellation_report.md"
	lines = [
		"# Constellation Report",
		"",
		f"- Number of satellites analyzed: {n_sats}",
		f"- Equal-weight BC median: {mix_summary.get('equal_median')}",
		f"- Equal-weight BC 95% HDI: [{mix_summary.get('equal_hdi95_low')}, {mix_summary.get('equal_hdi95_high')} ]",
		f"- ESS-weighted BC median: {mix_summary.get('ess_weighted_median')}",
		f"- ESS-weighted BC 95% quantile interval: [{mix_summary.get('ess_weighted_q025')}, {mix_summary.get('ess_weighted_q975')} ]",
		f"- Precision-weighted mean BC (summary statistic): {mix_summary.get('precision_weighted_mean')}",
		f"- Precision-weighted standard error: {mix_summary.get('precision_weighted_se')}",
		"",
		"## Pairwise Distances",
		"",
		f"- Pair count: {len(pairwise_df)}",
	]
	path.write_text("\n".join(lines), encoding="utf-8")


def make_posterior_predictive_hooks_warning(out_dir: Path) -> None:
	"""Write a note describing posterior predictive hook availability."""
	note = out_dir / "posterior_predictive_note.md"
	note.write_text(
		"# Posterior Predictive Hooks\n\n"
		"Posterior predictive checks are not executed by default.\n"
		"Provide an external propagator integration that maps (satellite_id, posterior_draws, t_days) "
		"to predicted sma_km trajectories to enable residual, RMSE/MAE/bias, and coverage analyses.\n",
		encoding="utf-8",
	)


def analyze_satellite(
	sat: SatelliteData,
	args: argparse.Namespace,
	out_dir: Path,
) -> tuple[dict[str, float | int | str | None], np.ndarray, np.ndarray, float | None, list[str]]:
	"""Run complete per-satellite analysis and return summary outputs."""
	warnings: list[str] = list(sat.warnings)
	arrays, arr_warnings, filtered_post = reconstruct_chain_draw_arrays(
		sat.posterior,
		burnin_iters=args.burnin_iters,
		burnin_frac=args.burnin_frac,
	)
	warnings.extend(arr_warnings)

	idata = build_inferencedata(arrays) if arrays is not None else None
	diagnostics = compute_arviz_diagnostics(idata)
	warnings.extend(diagnostics.warnings)

	mean_accept, final_accept, accept_df = compute_acceptance_metrics(
		sat.traces,
		burnin_iters=args.burnin_iters,
		burnin_frac=args.burnin_frac,
	)

	plot_warnings = make_per_satellite_plots(
		sat=sat,
		arrays=arrays,
		filtered_posterior=filtered_post,
		accept_df=accept_df,
		out_dir=out_dir,
		hdi_prob=args.hdi_prob,
		kde_grid=args.kde_grid,
	)
	warnings.extend(plot_warnings)

	summary_row = compute_per_satellite_summary(
		sat=sat,
		arrays=arrays,
		filtered_posterior=filtered_post,
		diagnostics=diagnostics,
		mean_accept_rate=mean_accept,
		final_accept_rate=final_accept,
		hdi_prob=args.hdi_prob,
	)
	ess_bulk_bc = diagnostics.ess_bulk_bc
	return (
		summary_row,
		filtered_post["bc_kg_per_m2"].to_numpy(dtype=float),
		filtered_post["beta_m2_per_kg"].to_numpy(dtype=float),
		ess_bulk_bc,
		warnings,
	)

def run_analysis(args: argparse.Namespace) -> None:
	"""Execute end-to-end analysis workflow."""
	input_dir = args.input_dir
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	groups, manifest = discover_satellite_groups(input_dir)
	manifest.to_csv(output_dir / "artifact_manifest.csv", index=False)
	chain_hints = load_summary_chain_hints(input_dir)

	matched = [g for g in groups if g.posterior_path is not None and g.obs_path is not None]
	if not matched:
		raise RuntimeError(f"No valid satellite groups found in {input_dir}")

	rng = np.random.default_rng(args.seed)
	per_sat_rows: list[dict[str, float | int | str | None]] = []
	per_sat_samples: dict[str, np.ndarray] = {}
	per_sat_beta_samples: dict[str, np.ndarray] = {}
	ess_bulk_map: dict[str, float | None] = {}
	sat_warnings: dict[str, list[str]] = {}

	for group in matched:
		LOGGER.info("Analyzing satellite stem: %s", group.stem)
		sat = load_satellite_data(group, n_chains_hint=chain_hints.get(group.stem))
		summary_row, bc_samples, beta_samples, ess_bulk_bc, warnings = analyze_satellite(sat, args=args, out_dir=output_dir)
		per_sat_rows.append(summary_row)
		per_sat_samples[sat.satellite_id] = bc_samples
		per_sat_beta_samples[sat.satellite_id] = beta_samples
		ess_bulk_map[sat.satellite_id] = ess_bulk_bc
		sat_warnings[sat.satellite_id] = warnings

	per_sat_df = pd.DataFrame(per_sat_rows)
	per_sat_df = per_sat_df.sort_values("satellite_id").reset_index(drop=True)
	per_sat_df.to_csv(output_dir / "per_satellite_summary.csv", index=False)

	equal_samples, ess_df, mix_summary = build_constellation_mixtures(
		per_sat_samples=per_sat_samples,
		ess_bulk_map=ess_bulk_map,
		rng=rng,
		min_draws_per_sat=args.min_draws_per_sat,
	)

	pairwise_df = compute_pairwise_distances(per_sat_samples)
	pairwise_df.to_csv(output_dir / "constellation_pairwise_distances.csv", index=False)

	if args.save_combined_samples and len(equal_samples) > 0:
		pd.DataFrame({"bc_kg_per_m2": equal_samples}).to_csv(
			output_dir / "constellation_combined_samples_equal_weight.csv",
			index=False,
		)

	constellation_summary = {
		"n_satellites": int(len(per_sat_df)),
		**mix_summary,
	}
	pd.DataFrame([constellation_summary]).to_csv(output_dir / "constellation_summary.csv", index=False)

	make_constellation_plots(
		per_sat_df=per_sat_df,
		per_sat_samples=per_sat_samples,
		per_sat_beta_samples=per_sat_beta_samples,
		equal_samples=equal_samples,
		ess_df=ess_df,
		pairwise_df=pairwise_df,
		out_dir=output_dir,
		kde_grid=args.kde_grid,
		rng=rng,
	)

	write_per_satellite_markdown(output_dir, per_sat_df, sat_warnings)
	write_constellation_markdown(output_dir, n_sats=len(per_sat_df), mix_summary=mix_summary, pairwise_df=pairwise_df)
	make_posterior_predictive_hooks_warning(output_dir)

	LOGGER.info("Analysis complete. Outputs written to %s", output_dir)

def main() -> None:
	"""CLI entrypoint."""
	args = parse_args()
	setup_logging(args.verbose)
	run_analysis(args)

if __name__ == "__main__":
	main()