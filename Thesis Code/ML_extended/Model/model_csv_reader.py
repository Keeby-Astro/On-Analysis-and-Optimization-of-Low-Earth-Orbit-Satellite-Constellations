from pathlib import Path
from simple_model_plotting import generate_visualizations_from_csv

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[2]

    predictions_csv = project_root / 'solar_flux_predictions.csv'
    ap_conditional_csv = project_root / 'ap_conditional_forecast.csv'

    if not predictions_csv.exists():
        raise FileNotFoundError(f"Missing required CSV: {predictions_csv}. "
                                "Expected a file with columns like date, Obs, Adj, URSI-D, ap, data_type.")

    ap_csv_arg = str(ap_conditional_csv) if ap_conditional_csv.exists() else None
    generate_visualizations_from_csv(predictions_csv=str(predictions_csv), ap_conditional_csv=ap_csv_arg)