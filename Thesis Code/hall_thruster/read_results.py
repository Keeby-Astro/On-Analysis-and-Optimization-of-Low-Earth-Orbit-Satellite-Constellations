"""Quick script to read and display trajectory run results."""
import sys, json
import pandas as pd

run_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/traj_run_01"

h = pd.read_csv(f"{run_dir}/latest/checkpoints/stage_a/stage_a_history.csv")
print("=== Trajectory Training History ===")
print(f"  Epoch 1:  path_rmse={h.iloc[0]['path_rmse_km']:.2f} km, total={h.iloc[0]['loss_total']:.4f}")
print(f"  Epoch {int(h.iloc[-1]['epoch'])}:  path_rmse={h.iloc[-1]['path_rmse_km']:.2f} km, total={h.iloc[-1]['loss_total']:.4f}")
print(f"  Min RMSE: {h['path_rmse_km'].min():.2f} km at epoch {int(h.loc[h['path_rmse_km'].idxmin(), 'epoch'])}")
print()

try:
    r = json.load(open(f"{run_dir}/latest/reports/trajectory_validation_report.json"))
    agg = r["aggregate"]
    print("=== Validation Report ===")
    print(f"  n_arcs: {r['n_arcs']}")
    print(f"  path_rmse_km_mean:   {agg['path_rmse_km_mean']:.2f}")
    print(f"  path_rmse_km_median: {agg['path_rmse_km_median']:.2f}")
    print(f"  path_rmse_km_p90:    {agg['path_rmse_km_p90']:.2f}")
    print(f"  raan_mae_rad:        {agg['raan_mae_rad']:.4f}")
    print(f"  lam_mae_rad:         {agg['lam_mae_rad']:.4f}")
    print()
    print("Per-phase:")
    for phase, m in r["per_phase"].items():
        print(f"  {phase}: n={m['n_arcs']}, rmse_mean={m['path_rmse_km_mean']:.2f}, rmse_med={m['path_rmse_km_median']:.2f}, raan_mae={m['raan_mae_rad']:.4f}")
except FileNotFoundError:
    print("  (no validation report)")
print()

try:
    p = json.load(open(f"{run_dir}/latest/checkpoints/stage_a/stage_a_parameter_summary.json"))
    print("=== Key Parameters ===")
    print(f"  mass_kg:   {p['mass_kg']:.2f}")
    print(f"  isp_s:     {p['isp_s']:.2f}")
    print(f"  eta_total: {p['eta_total']:.4f}")
    for phase, pp in p["phase_parameters"].items():
        print(f"  {phase}: thrust={pp['thrust_N']:.5f} N, duty={pp['duty']:.4f}, drag={pp['drag_kmps2']:.2e}, shell_comp={pp['shell_drag_comp_fraction']:.4f}, sign={pp['sign']}")
except FileNotFoundError:
    print("  (no parameter summary)")
print()

# Check for plots
from pathlib import Path
plots_fit = Path(f"{run_dir}/latest/plots/fit")
plots_train = Path(f"{run_dir}/latest/plots/train")
plots_params = Path(f"{run_dir}/latest/plots/parameters")
for d in [plots_fit, plots_train, plots_params]:
    if d.exists():
        files = list(d.glob("trajectory_*"))
        if files:
            print(f"Plots in {d.name}/: {[f.name for f in files]}")
