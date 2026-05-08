"""Full baseline-vs-optimized comparison for all stitched cluster policies.

Generates an artifact bundle:
    - comparison_summary.csv
    - run_metadata.json
    - initial_policy_offsets.png
    - final_altitude_comparison.png
    - altitude_timeseries_comparison.png
    - raan_timeseries_comparison.png
    - mean_anomaly_timeseries_comparison.png

Runs all 93 cluster medoids through the live simulator twice over 1 year:
    1. baseline  : raw TLE-derived orbital elements
    2. optimized : TLE-derived orbital elements after stitched policy offsets

All comparison time-series are overlaid on single figures (one trace per
cluster) rather than subplots, color-coded by cluster.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CLUSTERS = list(range(1, 94))  # all 93 clusters
DEFAULT_HOURS = 8766.0 * 5 # 1 year (365.25 days)
DEFAULT_OUTPUT_STRIDE = 500
DEFAULT_WORKERS = 12


def _load_simulator(module_path: Path):
    module_dir = module_path.parent
    workspace_root = module_dir.parent
    sys.path.insert(0, str(module_dir))
    sys.path.insert(0, str(workspace_root))
    importlib.invalidate_caches()
    if "constellation_simulator_optimized_thrust" in sys.modules:
        del sys.modules["constellation_simulator_optimized_thrust"]
    return importlib.import_module("constellation_simulator_optimized_thrust")


def _select_sample_satellites(sim, cluster_ids: list[int]) -> pd.DataFrame:
    overview_path = Path(sim.optimized_cluster_overview_csv)
    if not overview_path.is_absolute():
        overview_path = sim._base_dir_cluster / overview_path
    if not overview_path.is_file():
        raise FileNotFoundError(f"Cluster overview CSV not found: {overview_path}")

    overview = pd.read_csv(overview_path)
    overview["cluster_id"] = pd.to_numeric(overview["cluster_id"], errors="coerce").astype("Int64")
    overview = overview[overview["cluster_id"].isin(cluster_ids)].copy()
    overview = overview.sort_values("cluster_id").reset_index(drop=True)
    if overview.empty:
        raise RuntimeError("Requested sample clusters were not found in cluster_overview.csv")

    required = ["cluster_id", "medoid_sat_id", "cost_scalar"]
    missing = [c for c in required if c not in overview.columns]
    if missing:
        raise KeyError(f"cluster_overview.csv missing required columns: {missing}")

    return overview[[c for c in overview.columns if c in (
        "cluster_id", "medoid_sat_id", "cost_scalar", "policy_delta_a",
        "policy_delta_Omega", "policy_delta_lambda"
    )]].copy()


def _load_raw_latest_tle(sim, sat_ids: list[str]) -> pd.DataFrame:
    root = Path(sim.__file__).resolve().parent.parent
    tle_folder_paths = []
    for p in sim.tle_data_folders:
        pp = Path(p)
        if not pp.is_absolute():
            pp = root / pp
        tle_folder_paths.append(str(pp))

    tle_df, _ = sim.load_all_tle_data(tle_folder_paths, only_files=sim.tle_only_files, derived={"sma"})
    latest = (
        tle_df.sort_values("timestamp")
        .groupby("sat_id", as_index=False)
        .head(1)
        .sort_values("sat_id")
        .reset_index(drop=True)
    )
    latest = latest[latest["sat_id"].isin(sat_ids)].copy()
    if latest.empty:
        raise RuntimeError("No requested satellites were found in the loaded TLE data")
    return latest.set_index("sat_id", drop=False)


def _build_baseline_oe_cases(raw_latest: pd.DataFrame, sat_ids: list[str]) -> np.ndarray:
    oe = np.zeros((len(sat_ids), 6), dtype=np.float64)
    for idx, sat_id in enumerate(sat_ids):
        row = raw_latest.loc[sat_id]
        oe[idx, 0] = max(6378.1366 + 120.0, float(row["sma"]))
        oe[idx, 1] = np.clip(float(row["ecc"]), 0.0, 0.95)
        oe[idx, 2] = np.deg2rad(float(row["inc"]))
        oe[idx, 3] = np.deg2rad(float(row["aop"]))
        oe[idx, 4] = np.deg2rad(float(row["raan"]))
        oe[idx, 5] = np.deg2rad(float(row["mean_anomaly"]))
    return np.ascontiguousarray(oe, dtype=np.float64)


def _series_altitude_km(sim, state_sat: np.ndarray) -> np.ndarray:
    radii = np.sqrt(np.sum(state_sat[0:3, :] ** 2, axis=0))
    return radii - sim.earth_Re


def _extract_orbital_series(sim, state_sat: np.ndarray) -> dict[str, np.ndarray]:
    n = state_sat.shape[1]
    raan = np.zeros(n, dtype=np.float64)
    mean_anomaly = np.zeros(n, dtype=np.float64)
    for idx in range(n):
        oe = sim.xyz2orb(sim.earth_GM, state_sat[0:3, idx], state_sat[3:6, idx])
        raan[idx] = float(oe[4])
        mean_anomaly[idx] = float(oe[5])
    return {
        "raan_deg": np.rad2deg(np.unwrap(raan)),
        "mean_anomaly_deg_wrapped": np.mod(np.rad2deg(mean_anomaly), 360.0),
        "mean_anomaly_deg_unwrapped": np.rad2deg(np.unwrap(mean_anomaly)),
    }


def _final_altitude_km(sim, result: dict) -> float:
    rf = np.sqrt(result["final_x_km"] ** 2 + result["final_y_km"] ** 2 + result["final_z_km"] ** 2)
    return float(rf - sim.earth_Re)


def _final_orbital_angles(sim, result: dict) -> tuple[float, float]:
    r = np.array([result["final_x_km"], result["final_y_km"], result["final_z_km"]], dtype=np.float64)
    v = np.array([result["final_vx_kms"], result["final_vy_kms"], result["final_vz_kms"]], dtype=np.float64)
    oe = sim.xyz2orb(sim.earth_GM, r, v)
    return float(np.rad2deg(oe[4]) % 360.0), float(np.rad2deg(oe[5]) % 360.0)


def _wrapped_deg_delta(a_deg: float, b_deg: float) -> float:
    return float(((a_deg - b_deg + 180.0) % 360.0) - 180.0)


def _make_summary_df(sim, sample_df: pd.DataFrame, raw_latest: pd.DataFrame,
                     baseline_results: list[dict], optimized_results: list[dict],
                     cluster_meta: dict | None = None) -> pd.DataFrame:
    baseline_by_sat = {item["sat_id"]: item for item in baseline_results}
    optimized_by_sat = {item["sat_id"]: item for item in optimized_results}
    meta_map = cluster_meta or sim.cluster_metadata_by_sat or {}

    rows = []
    for row in sample_df.itertuples(index=False):
        sat_id = str(row.medoid_sat_id)
        raw = raw_latest.loc[sat_id]
        meta = meta_map.get(sat_id, {})
        base = baseline_by_sat[sat_id]
        opt = optimized_by_sat[sat_id]
        base_final_raan_deg, base_final_ma_deg = _final_orbital_angles(sim, base)
        opt_final_raan_deg, opt_final_ma_deg = _final_orbital_angles(sim, opt)

        rows.append({
            "cluster_id": int(row.cluster_id),
            "sat_id": sat_id,
            "policy_applied": bool(meta.get("cluster_policy_applied", False)),
            "policy_delta_a_km": float(meta.get("policy_delta_a_km", 0.0) or 0.0),
            "policy_delta_Omega_deg": float(np.rad2deg(meta.get("policy_delta_Omega_rad", 0.0) or 0.0)),
            "policy_delta_lambda_deg": float(np.rad2deg(meta.get("policy_delta_lambda_rad", 0.0) or 0.0)),
            "raw_sma_km": float(raw["sma"]),
            "baseline_initial_a_km": float(base["a_km"]),
            "optimized_initial_a_km": float(opt["a_km"]),
            "optimized_minus_baseline_a_km": float(opt["a_km"] - base["a_km"]),
            "baseline_final_alt_km": _final_altitude_km(sim, base),
            "optimized_final_alt_km": _final_altitude_km(sim, opt),
            "optimized_minus_baseline_final_alt_km": _final_altitude_km(sim, opt) - _final_altitude_km(sim, base),
            "baseline_final_raan_deg": base_final_raan_deg,
            "optimized_final_raan_deg": opt_final_raan_deg,
            "optimized_minus_baseline_final_raan_deg": _wrapped_deg_delta(opt_final_raan_deg, base_final_raan_deg),
            "baseline_final_mean_anomaly_deg": base_final_ma_deg,
            "optimized_final_mean_anomaly_deg": opt_final_ma_deg,
            "optimized_minus_baseline_final_mean_anomaly_deg": _wrapped_deg_delta(opt_final_ma_deg, base_final_ma_deg),
            "baseline_propellant_used_kg": float(base.get("propellant_used_kg", 0.0) or 0.0),
            "optimized_propellant_used_kg": float(opt.get("propellant_used_kg", 0.0) or 0.0),
            "optimized_minus_baseline_propellant_kg": float((opt.get("propellant_used_kg", 0.0) or 0.0) - (base.get("propellant_used_kg", 0.0) or 0.0)),
            "baseline_n_points": int(base["n_points"]),
            "optimized_n_points": int(opt["n_points"]),
            "policy_total_cost": float(meta.get("policy_total_cost")) if meta.get("policy_total_cost") is not None else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["cluster_id", "sat_id"]).reset_index(drop=True)


def _plot_initial_offsets(summary_df: pd.DataFrame, out_path: Path):
    x = np.arange(len(summary_df))
    cids = summary_df["cluster_id"].values

    fig, ax = plt.subplots(figsize=(18, 6))
    width = 0.28
    ax.bar(x - width, summary_df["policy_delta_a_km"], width=width, color="#15528e", label="delta a (km)")
    ax.bar(x, summary_df["policy_delta_Omega_deg"], width=width, color="#b25800", label="delta RAAN (deg)")
    ax.bar(x + width, summary_df["policy_delta_lambda_deg"], width=width, color="#1e701e", label="delta phase (deg)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Offset value")
    ax.set_title("Applied Stitched Policy Offsets (all clusters)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in cids], rotation=90, fontsize=6)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_final_altitudes(summary_df: pd.DataFrame, out_path: Path):
    x = np.arange(len(summary_df))
    cids = summary_df["cluster_id"].values
    width = 0.36

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - width / 2, summary_df["baseline_final_alt_km"], width=width, label="Baseline", color="#798ba2")
    ax.bar(x + width / 2, summary_df["optimized_final_alt_km"], width=width, label="Optimized", color="#15528e")
    ax.set_ylabel("Final altitude (km)")
    ax.set_title("Baseline vs Optimized Final Altitude (1-year run, all clusters)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in cids], rotation=90, fontsize=6)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _cluster_cmap(n_clusters: int):
    """Return a list of *n_clusters* distinct colors from tab20 + tab20b."""
    cmap1 = plt.cm.get_cmap("tab20", 20)
    cmap2 = plt.cm.get_cmap("tab20b", 20)
    colors = [cmap1(i % 20) for i in range(min(n_clusters, 20))]
    if n_clusters > 20:
        colors += [cmap2(i % 20) for i in range(n_clusters - 20)]
    return colors


def _plot_altitude_timeseries(sim, sample_df: pd.DataFrame,
                              baseline_results: list[dict], optimized_results: list[dict],
                              out_path: Path):
    baseline_by_sat = {item["sat_id"]: item for item in baseline_results}
    optimized_by_sat = {item["sat_id"]: item for item in optimized_results}
    n = len(sample_df)
    cluster_colors = _cluster_cmap(n)

    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, row in enumerate(sample_df.itertuples(index=False)):
        sat_id = str(row.medoid_sat_id)
        base = baseline_by_sat[sat_id]
        opt = optimized_by_sat[sat_id]
        color = cluster_colors[idx]

        t_base_days = np.asarray(base["times"], dtype=np.float64) / 86400.0
        t_opt_days = np.asarray(opt["times"], dtype=np.float64) / 86400.0
        alt_base = _series_altitude_km(sim, np.asarray(base["state_sat"], dtype=np.float64))
        alt_opt = _series_altitude_km(sim, np.asarray(opt["state_sat"], dtype=np.float64))

        ax.plot(t_base_days, alt_base, color=color, alpha=0.35, linewidth=0.5)
        ax.plot(t_opt_days, alt_opt, color=color, alpha=0.8, linewidth=0.7,
                label=f"C{int(row.cluster_id)}" if idx < 30 else None)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude Timeseries: Baseline (faint) vs Optimized (solid)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=6, ncol=5, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_orbital_series_comparison(sim, sample_df: pd.DataFrame,
                                    baseline_results: list[dict], optimized_results: list[dict],
                                    out_path: Path, series_key: str,
                                    y_label: str, title: str):
    baseline_by_sat = {item["sat_id"]: item for item in baseline_results}
    optimized_by_sat = {item["sat_id"]: item for item in optimized_results}
    n = len(sample_df)
    cluster_colors = _cluster_cmap(n)

    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, row in enumerate(sample_df.itertuples(index=False)):
        sat_id = str(row.medoid_sat_id)
        base = baseline_by_sat[sat_id]
        opt = optimized_by_sat[sat_id]
        color = cluster_colors[idx]

        t_base_days = np.asarray(base["times"], dtype=np.float64) / 86400.0
        t_opt_days = np.asarray(opt["times"], dtype=np.float64) / 86400.0
        base_series = _extract_orbital_series(sim, np.asarray(base["state_sat"], dtype=np.float64))[series_key]
        opt_series = _extract_orbital_series(sim, np.asarray(opt["state_sat"], dtype=np.float64))[series_key]

        ax.plot(t_base_days, base_series, color=color, alpha=0.35, linewidth=0.5)
        ax.plot(t_opt_days, opt_series, color=color, alpha=0.8, linewidth=0.7,
                label=f"C{int(row.cluster_id)}" if idx < 30 else None)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}: Baseline (faint) vs Optimized (solid)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=6, ncol=5, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _build_optimized_oe_cases(sim, baseline_oe: np.ndarray, sat_ids: list[str],
                              sample_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Apply stitched policy offsets to baseline OE and build cluster metadata."""
    optimized_oe = baseline_oe.copy()

    # Load policy and cluster assignment maps from the simulator
    cluster_assign_map = sim._load_cluster_assignments_map(sim.cluster_assignments_csv)
    policy_map, policy_source = sim._load_cluster_policy_map()

    cluster_meta = {}
    for idx, sat_id in enumerate(sat_ids):
        sat_key = sim._normalize_cluster_sat_id(sat_id)
        cid = int(cluster_assign_map.get(str(sat_key), 0))
        policy = policy_map.get(cid)

        meta = {
            'cluster_id': cid,
            'is_noise': cid <= 0,
            'pooled_role': 'medoid',
            'cluster_weight_active': 1,
            'cluster_weight_global': 1,
            'representative_sat_id': sat_id,
            'color_hex': sim.cluster_noise_color if cid <= 0 else '',
            'cluster_policy_applied': False,
            'cluster_policy_source': '',
        }
        for field in sim._CLUSTER_POLICY_OUTPUT_FIELDS:
            meta.setdefault(field, sim._CLUSTER_POLICY_DEFAULTS.get(field, 0.0))

        if policy is not None:
            optimized_oe[idx, 0] = max(6378.1366 + 120.0,
                                       optimized_oe[idx, 0] + policy['policy_delta_a_km'])
            optimized_oe[idx, 4] = (optimized_oe[idx, 4] + policy['policy_delta_Omega_rad']) % (2.0 * np.pi)
            optimized_oe[idx, 5] = (optimized_oe[idx, 5] + policy['policy_delta_lambda_rad']) % (2.0 * np.pi)
            meta['cluster_policy_applied'] = True
            meta['cluster_policy_source'] = policy_source or ''
            for field in sim._CLUSTER_POLICY_OUTPUT_FIELDS:
                if field in policy:
                    meta[field] = policy[field]

        cluster_meta[sat_id] = meta

    return np.ascontiguousarray(optimized_oe, dtype=np.float64), cluster_meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full baseline-vs-optimized stitched policy comparison (1-year, all clusters)")
    parser.add_argument("--clusters", nargs="*", type=int, default=DEFAULT_CLUSTERS,
                        help="Cluster IDs to sample (default: %(default)s)")
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS,
                        help="Short-run propagation horizon in hours")
    parser.add_argument("--output-stride", type=int, default=DEFAULT_OUTPUT_STRIDE,
                        help="Trajectory thinning factor")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Parallel workers for the short runs")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "optimization_outputs" / "verification_quick_compare"),
                        help="Directory for CSV/JSON/plot outputs")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module_path = Path(__file__).resolve().parent / "constellation_simulator_optimized_thrust.py"
    t0 = time.perf_counter()
    sim = _load_simulator(module_path)
    import_seconds = time.perf_counter() - t0

    sample_df = _select_sample_satellites(sim, list(args.clusters))
    sat_ids = sample_df["medoid_sat_id"].astype(str).tolist()

    # Load raw TLE data for all medoid satellites directly (bypasses sim's tle_satellite_limit)
    raw_latest = _load_raw_latest_tle(sim, sat_ids)

    # Filter to satellites that actually have TLE data
    available_sat_ids = [sid for sid in sat_ids if sid in raw_latest.index]
    n_before = len(sat_ids)
    if len(available_sat_ids) < n_before:
        print(f"[INFO] {n_before - len(available_sat_ids)} medoid satellites not in TLE data, "
              f"continuing with {len(available_sat_ids)}/{n_before} clusters.")
        sample_df = sample_df[sample_df["medoid_sat_id"].isin(available_sat_ids)].reset_index(drop=True)
        sat_ids = sample_df["medoid_sat_id"].astype(str).tolist()
    if not sat_ids:
        print(f"ERROR: None of the {n_before} medoid satellites were found in TLE data.")
        return 1

    # Build start timestamps from raw TLE epochs
    start_timestamps = [pd.Timestamp(raw_latest.loc[sid, "timestamp"]) for sid in sat_ids]

    # Build OE cases
    baseline_oe = _build_baseline_oe_cases(raw_latest, sat_ids)
    optimized_oe, optimized_meta = _build_optimized_oe_cases(sim, baseline_oe, sat_ids, sample_df)
    ballistic_coefficients = np.full(len(sat_ids), sim.ballistic_coefficient_nominal, dtype=np.float64)

    baseline_meta = None

    run_kwargs = dict(
        sat_ids=sat_ids,
        start_timestamps=start_timestamps,
        ballistic_coefficients=ballistic_coefficients,
        write_outputs=False,
        return_trajectories=True,
        return_mass_series=False,
        workers_override=max(1, int(args.workers)),
        max_prop_time_s=float(args.hours) * 3600.0,
        output_stride=max(1, int(args.output_stride)),
    )

    t1 = time.perf_counter()
    baseline_results = sim.run_batch_cases(oe_cases=baseline_oe, cluster_meta_map=baseline_meta, **run_kwargs)
    baseline_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    optimized_results = sim.run_batch_cases(oe_cases=optimized_oe, cluster_meta_map=optimized_meta, **run_kwargs)
    optimized_seconds = time.perf_counter() - t2

    summary_df = _make_summary_df(sim, sample_df, raw_latest, baseline_results, optimized_results,
                                   cluster_meta=optimized_meta)
    summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)

    metadata = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "policy_source": str(next((m.get("cluster_policy_source") for m in optimized_meta.values() if m.get("cluster_policy_source")), "")),
        "selected_clusters": [int(c) for c in sample_df["cluster_id"].tolist()],
        "selected_satellites": sat_ids,
        "import_seconds": round(import_seconds, 3),
        "baseline_run_seconds": round(baseline_seconds, 3),
        "optimized_run_seconds": round(optimized_seconds, 3),
        "hours": float(args.hours),
        "output_stride": int(args.output_stride),
        "workers": int(args.workers),
        "artifacts": {
            "summary_csv": str(output_dir / "comparison_summary.csv"),
            "offset_plot": str(output_dir / "initial_policy_offsets.png"),
            "final_altitude_plot": str(output_dir / "final_altitude_comparison.png"),
            "timeseries_plot": str(output_dir / "altitude_timeseries_comparison.png"),
            "raan_plot": str(output_dir / "raan_timeseries_comparison.png"),
            "mean_anomaly_plot": str(output_dir / "mean_anomaly_timeseries_comparison.png"),
        },
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    _plot_initial_offsets(summary_df, output_dir / "initial_policy_offsets.png")
    _plot_final_altitudes(summary_df, output_dir / "final_altitude_comparison.png")
    _plot_altitude_timeseries(sim, sample_df, baseline_results, optimized_results,
                              output_dir / "altitude_timeseries_comparison.png")
    _plot_orbital_series_comparison(
        sim,
        sample_df,
        baseline_results,
        optimized_results,
        output_dir / "raan_timeseries_comparison.png",
        "raan_deg",
        "RAAN (deg, unwrapped)",
        "Baseline vs Optimized RAAN",
    )
    _plot_orbital_series_comparison(
        sim,
        sample_df,
        baseline_results,
        optimized_results,
        output_dir / "mean_anomaly_timeseries_comparison.png",
        "mean_anomaly_deg_wrapped",
        "Mean anomaly (deg, wrapped)",
        "Baseline vs Optimized Mean Anomaly",
    )

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())