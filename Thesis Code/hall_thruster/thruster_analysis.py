"""
Thruster Performance Analysis
=============================
Computes impulse, thrust, delta-v, propellant consumption, power,
and per-phase summaries from Stage A predictions.

Usage
-----
    python thruster_analysis.py [run_dir]

    run_dir defaults to ../outputs  (relative to this script).

Outputs are printed to stdout and saved to:
    <run_dir>/latest/reports/thruster_performance_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

G0 = 9.80665  # m/s^2

# ── Resolve run directory ────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
default_run = script_dir.parent / "outputs"
run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_run
latest = run_dir / "latest"

pred_path = latest / "tables" / "stage_a_predictions.csv"
param_path = latest / "checkpoints" / "stage_a" / "stage_a_parameter_summary.json"
out_path = latest / "reports" / "thruster_performance_summary.json"

if not pred_path.exists():
    sys.exit(f"Predictions file not found: {pred_path}")

# ── Load data ────────────────────────────────────────────────────────────────
cols = [
    "sat_id", "phase", "phase_sign", "dt_s", "dt_days",
    "a0_km", "a1_km", "da_obs_km",
    "thrust_pred_N", "isp_pred_s", "eta_total_phase",
    "power_in_W", "power_nominal_W",
    "mass_end_kg", "mdot_a_kg_s", "mdot_c_kg_s",
    "gamma", "eta_b", "eta_v", "eta_m", "eta_o",
    "vb_V", "ib_A", "vd_V",
    "da_pred_km", "da_resid_km",
    "segment_quality_flag",
]
df = pd.read_csv(pred_path, usecols=cols)

# Load global fitted parameters
params = {}
if param_path.exists():
    with open(param_path) as f:
        params = json.load(f)

n_sats = df["sat_id"].nunique()
n_segs = len(df)

# ── Per-phase duty cycle from parameter summary ─────────────────────────────
phase_duty = {}
if "phase_parameters" in params:
    for phase, pp in params["phase_parameters"].items():
        phase_duty[phase] = pp.get("duty", 1.0)

# ── Derived columns ─────────────────────────────────────────────────────────
# Duty fraction for each segment (use fitted duty if available, else 1)
df["duty"] = df["phase"].map(phase_duty).fillna(1.0)

# Thrust duration accounting for duty cycle [s]
df["thrust_on_time_s"] = df["dt_s"] * df["duty"]

# Total impulse per segment  I = T * duty * dt  [N·s]
df["impulse_Ns"] = df["thrust_pred_N"] * df["thrust_on_time_s"]

# Exhaust velocity [m/s]
df["ve_m_s"] = df["isp_pred_s"] * G0

# Mass flow rate [kg/s] from thrust equation: mdot = T / ve
df["mdot_thrust_kg_s"] = np.where(
    df["ve_m_s"] > 0,
    df["thrust_pred_N"] / df["ve_m_s"],
    0.0,
)

# Propellant consumed per segment [kg]   dm = mdot * thrust_on_time
df["propellant_kg"] = df["mdot_thrust_kg_s"] * df["thrust_on_time_s"]

# Delta-v per segment via Tsiolkovsky: dv = ve * ln(m0/m1)
# Approximate m0 from mass_end + propellant consumed
df["mass_start_kg"] = df["mass_end_kg"] + df["propellant_kg"]
df["delta_v_m_s"] = np.where(
    (df["mass_end_kg"] > 0) & (df["mass_start_kg"] > df["mass_end_kg"]),
    df["ve_m_s"] * np.log(df["mass_start_kg"] / df["mass_end_kg"]),
    0.0,
)

# Delta-v cross-check from thrust acceleration: dv ≈ (T/m_avg) * duty * dt
df["mass_avg_kg"] = 0.5 * (df["mass_start_kg"] + df["mass_end_kg"])
df["delta_v_accel_m_s"] = np.where(
    df["mass_avg_kg"] > 0,
    (df["thrust_pred_N"] / df["mass_avg_kg"]) * df["thrust_on_time_s"],
    0.0,
)

# Electrical energy consumed per segment [J] and [W·h]
df["energy_J"] = df["power_in_W"] * df["thrust_on_time_s"]
df["energy_Wh"] = df["energy_J"] / 3600.0

# Thermal dissipation [W]
df["power_thermal_W"] = df["power_in_W"] * (1.0 - df["eta_total_phase"])

# Thrust-to-power ratio [mN/kW]
df["thrust_to_power_mN_kW"] = np.where(
    df["power_in_W"] > 0,
    (df["thrust_pred_N"] * 1e3) / (df["power_in_W"] * 1e-3),
    0.0,
)

# ── Aggregate per phase ─────────────────────────────────────────────────────
phase_order = ["insertion_or_orbit_raise", "operational_shell", "disposal_lowering"]
phases_present = [p for p in phase_order if p in df["phase"].values]

phase_summary = {}
for phase in phases_present:
    g = df[df["phase"] == phase]
    n = len(g)
    total_time_days = g["dt_days"].sum()
    thrust_on_days = g["thrust_on_time_s"].sum() / 86400.0

    total_impulse = g["impulse_Ns"].sum()
    total_dv_tsiolkovsky = g["delta_v_m_s"].sum()
    total_dv_accel = g["delta_v_accel_m_s"].sum()
    total_propellant = g["propellant_kg"].sum()
    total_energy_kWh = g["energy_Wh"].sum() / 1e3

    mean_thrust = g["thrust_pred_N"].mean()
    mean_isp = g["isp_pred_s"].mean()
    mean_power = g["power_in_W"].mean()
    mean_eta = g["eta_total_phase"].mean()
    mean_tp = g["thrust_to_power_mN_kW"].mean()

    # Per-satellite averages
    per_sat = g.groupby("sat_id").agg(
        impulse_Ns=("impulse_Ns", "sum"),
        delta_v_m_s=("delta_v_m_s", "sum"),
        propellant_kg=("propellant_kg", "sum"),
        dt_days=("dt_days", "sum"),
    )

    phase_summary[phase] = {
        "n_segments": int(n),
        "n_satellites": int(g["sat_id"].nunique()),
        "total_elapsed_days": round(total_time_days, 2),
        "total_thrust_on_days": round(thrust_on_days, 2),
        "duty_cycle": round(phase_duty.get(phase, 1.0), 4),
        "mean_thrust_N": round(float(mean_thrust), 6),
        "mean_thrust_mN": round(float(mean_thrust * 1e3), 3),
        "mean_isp_s": round(float(mean_isp), 2),
        "mean_efficiency": round(float(mean_eta), 4),
        "mean_power_W": round(float(mean_power), 2),
        "mean_thrust_to_power_mN_kW": round(float(mean_tp), 2),
        "total_impulse_Ns": round(float(total_impulse), 2),
        "total_impulse_kNs": round(float(total_impulse / 1e3), 4),
        "total_delta_v_tsiolkovsky_m_s": round(float(total_dv_tsiolkovsky), 2),
        "total_delta_v_accel_m_s": round(float(total_dv_accel), 2),
        "total_propellant_kg": round(float(total_propellant), 4),
        "total_energy_kWh": round(float(total_energy_kWh), 2),
        "per_satellite_median": {
            "impulse_Ns": round(float(per_sat["impulse_Ns"].median()), 2),
            "delta_v_m_s": round(float(per_sat["delta_v_m_s"].median()), 2),
            "propellant_kg": round(float(per_sat["propellant_kg"].median()), 4),
            "duration_days": round(float(per_sat["dt_days"].median()), 2),
        },
    }

# ── Fleet-wide totals ───────────────────────────────────────────────────────
fleet = {
    "n_satellites": int(n_sats),
    "n_segments": int(n_segs),
    "total_impulse_kNs": round(float(df["impulse_Ns"].sum() / 1e3), 4),
    "total_delta_v_m_s": round(float(df["delta_v_m_s"].sum()), 2),
    "total_propellant_kg": round(float(df["propellant_kg"].sum()), 4),
    "total_energy_kWh": round(float(df["energy_Wh"].sum() / 1e3), 2),
}

# Per-satellite lifetime totals
per_sat_all = df.groupby("sat_id").agg(
    impulse_Ns=("impulse_Ns", "sum"),
    delta_v_m_s=("delta_v_m_s", "sum"),
    propellant_kg=("propellant_kg", "sum"),
    dt_days=("dt_days", "sum"),
)
fleet["per_satellite_median"] = {
    "impulse_Ns": round(float(per_sat_all["impulse_Ns"].median()), 2),
    "delta_v_m_s": round(float(per_sat_all["delta_v_m_s"].median()), 2),
    "propellant_kg": round(float(per_sat_all["propellant_kg"].median()), 4),
    "total_observed_days": round(float(per_sat_all["dt_days"].median()), 2),
}
fleet["per_satellite_p10_p90"] = {
    "delta_v_m_s": [
        round(float(per_sat_all["delta_v_m_s"].quantile(0.10)), 2),
        round(float(per_sat_all["delta_v_m_s"].quantile(0.90)), 2),
    ],
    "propellant_kg": [
        round(float(per_sat_all["propellant_kg"].quantile(0.10)), 4),
        round(float(per_sat_all["propellant_kg"].quantile(0.90)), 4),
    ],
}

# ── Global fitted parameters echo ───────────────────────────────────────────
fitted = {}
if params:
    fitted = {
        "mass_kg": params.get("mass_kg"),
        "dry_mass_kg": params.get("dry_mass_kg"),
        "isp_s": params.get("isp_s"),
        "eta_total": params.get("eta_total"),
    }

# ── Assemble final report ───────────────────────────────────────────────────
report = {
    "description": "Thruster performance derived from Stage A trajectory fits",
    "fitted_parameters": fitted,
    "fleet_totals": fleet,
    "per_phase": phase_summary,
}

# ── Print ────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  THRUSTER PERFORMANCE SUMMARY")
print("=" * 70)
if fitted:
    print(f"\n  Fitted globals:  mass={fitted['mass_kg']:.1f} kg,  "
          f"Isp={fitted['isp_s']:.1f} s,  η={fitted['eta_total']:.3f}")
print(f"  Fleet: {fleet['n_satellites']} satellites, {fleet['n_segments']} segments")
print(f"  Total impulse:     {fleet['total_impulse_kNs']:.2f} kN·s")
print(f"  Total Δv (fleet):  {fleet['total_delta_v_m_s']:.1f} m/s")
print(f"  Total propellant:  {fleet['total_propellant_kg']:.2f} kg")
print(f"  Total energy:      {fleet['total_energy_kWh']:.1f} kW·h")
print(f"\n  Per-satellite medians:")
m = fleet["per_satellite_median"]
print(f"    Impulse:     {m['impulse_Ns']:.1f} N·s")
print(f"    Δv:          {m['delta_v_m_s']:.1f} m/s")
print(f"    Propellant:  {m['propellant_kg']:.3f} kg")
print(f"    Observed:    {m['total_observed_days']:.0f} days")
ci = fleet["per_satellite_p10_p90"]
print(f"    Δv 10–90%:   [{ci['delta_v_m_s'][0]:.1f}, {ci['delta_v_m_s'][1]:.1f}] m/s")

for phase in phases_present:
    ps = phase_summary[phase]
    label = phase.replace("_", " ").title()
    print(f"\n  ── {label} ──")
    print(f"    Segments: {ps['n_segments']},  Sats: {ps['n_satellites']}")
    print(f"    Thrust:     {ps['mean_thrust_mN']:.3f} mN  (mean)")
    print(f"    Isp:        {ps['mean_isp_s']:.1f} s")
    print(f"    Efficiency: {ps['mean_efficiency']:.3f}")
    print(f"    Power:      {ps['mean_power_W']:.1f} W")
    print(f"    T/P ratio:  {ps['mean_thrust_to_power_mN_kW']:.1f} mN/kW")
    print(f"    Duty cycle: {ps['duty_cycle']:.3f}")
    print(f"    Impulse:    {ps['total_impulse_kNs']:.3f} kN·s  (total across fleet)")
    print(f"    Δv:         {ps['total_delta_v_tsiolkovsky_m_s']:.1f} m/s  (Tsiolkovsky sum)")
    print(f"    Propellant: {ps['total_propellant_kg']:.3f} kg")
    pm = ps["per_satellite_median"]
    print(f"    Per-sat median: Δv={pm['delta_v_m_s']:.1f} m/s, "
          f"prop={pm['propellant_kg']:.3f} kg, "
          f"dur={pm['duration_days']:.0f} d")

print("\n" + "=" * 70)

# ── Save JSON ────────────────────────────────────────────────────────────────
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved → {out_path}")
