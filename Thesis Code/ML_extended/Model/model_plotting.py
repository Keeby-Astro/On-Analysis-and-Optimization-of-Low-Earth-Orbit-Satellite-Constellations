# Model Plotting and Visualization Module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import torch
from astropy.timeseries import LombScargle

from model_visualization import (visualize_resnext_features, list_conv_modules,
                                 visualize_all_kernels, visualize_activations)

def generate_visualizations_and_predictions(final_model, dm, loss_history, cv_losses, cv_maes, 
                                           forecast_years, pred_len, colors, HARMONICS=3,
                                           ap_regression_model=None, ap_ar_coeffs=None, 
                                           ap_sigma=None, ap_eps_init=None, simulate_ap_fn=None,
                                           last_train_idx=None, df_hist=None):
    ap_conditional_mean = None
    ap_conditional_sample = None

    # Plot RMSE per fold
    rmses = [np.sqrt(loss) for loss in cv_losses]
    plt.figure(figsize=(7.16, 3.45))
    plt.bar([f'Fold {i+1}' for i in range(len(rmses))], rmses, color=colors[0])
    plt.ylabel('RMSE')
    plt.title('Cross-Validation RMSE per Fold')
    plt.savefig('cv_rmse.png', dpi=600, bbox_inches='tight')
    #plt.show()

    # Plot MAE per fold
    plt.figure(figsize=(7.16, 3.45))
    plt.bar([f'Fold {i+1}' for i in range(len(cv_maes))], cv_maes, color=colors[1])
    plt.ylabel('MAE')
    plt.title('Cross-Validation MAE per Fold')
    plt.savefig('cv_mae.png', dpi=600, bbox_inches='tight')
    #plt.show()

    # Recursive 22-year forecast & plotting
    ds_full    = dm.full_ds
    orig_T     = len(ds_full.series)
    fut_len    = forecast_years * pred_len
    start_date = ds_full.dates.iloc[-1] + pd.Timedelta(days=1)
    fut_idx    = pd.date_range(start=start_date, periods=fut_len, freq='D')

    full_day1 = np.concatenate([ds_full.dates.dt.dayofyear.values.astype(np.float32) - 1,
                               fut_idx.dayofyear.values.astype(np.float32) - 1])
    rad1_full = 2*np.pi * full_day1 / 365.0
    sin1_full = np.stack([np.sin((h+1)*rad1_full) for h in range(HARMONICS)], axis=1)
    cos1_full = np.stack([np.cos((h+1)*rad1_full) for h in range(HARMONICS)], axis=1)

    full_day2 = np.arange(orig_T + fut_len, dtype=np.float32)
    rad2_full = 2*np.pi * (full_day2 % (11*365)) / (11*365)
    sin2_full = np.stack([np.sin((h+1)*rad2_full) for h in range(HARMONICS)], axis=1)
    cos2_full = np.stack([np.cos((h+1)*rad2_full) for h in range(HARMONICS)], axis=1)

    final_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model.to(device)
    norm_series = ds_full.series.copy()

    residuals = []
    for x_batch, y_batch in dm.val_dataloader():
        x_batch = x_batch.to(device)
        with torch.no_grad():
            y_pred = final_model(x_batch)
        residuals.append((y_pred.cpu().numpy() - y_batch.numpy()).flatten())
    residuals = np.concatenate(residuals) if residuals else np.array([])

    # Histogram of residuals
    plt.figure(figsize=(7.16, 3.45))
    plt.hist(residuals, bins=50, color=colors[0])
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Residuals')
    plt.grid()
    plt.savefig('residuals_histogram.png', dpi=600, bbox_inches='tight')
    #plt.show()

    preds = []
    idx   = orig_T
    seq_len = ds_full.seq_length
    
    for year_step in range(forecast_years):
        win_s = norm_series[-seq_len:]
        # Slice the HARMONICS-order blocks
        s1 = sin1_full[idx-seq_len:idx] 
        c1 = cos1_full[idx-seq_len:idx]
        s2 = sin2_full[idx-seq_len:idx]
        c2 = cos2_full[idx-seq_len:idx]
        # Concatenate all harmonics
        cyc = np.concatenate([s1, c1, s2, c2], axis=1) * ds_full.std.mean()
        # Final input has only original vars + cyc (no extra engineered features)
        x_in = np.concatenate([win_s, cyc], axis=1).T
        inp  = torch.from_numpy(x_in).unsqueeze(0).to(device).float()
        with torch.no_grad():
            p = final_model(inp).squeeze(0).cpu().numpy()
        preds.append(p)
        
        if len(norm_series) > 365:  # Use seasonal average if we have enough history
            # Use same-season values from 1 solar cycle ago as proxy
            lookback_idx = max(0, len(norm_series) - 11*365)
            exog_proxy = norm_series[lookback_idx:lookback_idx + p.shape[0], ds_full.n_target_vars:]
            if len(exog_proxy) < p.shape[0]:
                # Fallback to last known values if not enough history
                exog_proxy = np.tile(norm_series[-1, ds_full.n_target_vars:], (p.shape[0], 1))
        else:
            # Fallback to last known values
            exog_proxy = np.tile(norm_series[-1, ds_full.n_target_vars:], (p.shape[0], 1))
        
        p_full = np.concatenate([p, exog_proxy], axis=1)  # (pred_len, n_all_vars)
        norm_series = np.vstack([norm_series, p_full])
        idx += pred_len

    # Denormalize only the predicted columns (first n_target_vars)
    multi = np.vstack(preds) * ds_full.std[:, :ds_full.n_target_vars] + ds_full.mean[:, :ds_full.n_target_vars]
    df_fut = pd.DataFrame(multi, index=fut_idx, columns=ds_full.cols[:ds_full.n_target_vars])

    # Also denormalize all model channels for SW-All compatible export construction.
    fut_norm_all = norm_series[orig_T:orig_T + fut_len, :len(ds_full.cols)]
    fut_all = fut_norm_all * ds_full.std + ds_full.mean
    df_fut_all = pd.DataFrame(fut_all, index=fut_idx, columns=ds_full.cols)

    # === DIAGNOSTIC VISUALIZATION MODULE ==
    # 1. F10.7 time series with 13-month smoothed curve and cycle maxima/minima
    f107 = ds_full.raw[:, ds_full.cols.index('Obs')]

    # 2. Frequency-domain plots: periodogram and Lomb-Scargle
    # (a) Periodogram
    f107_nonan = np.nan_to_num(f107, nan=np.nanmean(f107))
    fs = 1.0  # 1/day
    freqs, pxx = scipy.signal.periodogram(f107_nonan, fs=fs)
    plt.figure(figsize=(10.74, 4))
    plt.semilogy(freqs, pxx, color=colors[0])
    plt.xlabel('Frequency (1/day)')
    plt.ylabel('Power')
    plt.title('F10.7 Periodogram')
    plt.xlim(0, 0.05)
    plt.grid()
    plt.savefig('f107_periodogram.png', dpi=600, bbox_inches='tight')
    #plt.show()

    # (b) Lomb-Scargle (for uneven sampling, but here for completeness)
    t = np.arange(len(f107_nonan))
    frequency = np.linspace(1/5000, 0.5, 10000)
    power = LombScargle(t, f107_nonan).power(frequency)
    plt.figure(figsize=(10.74, 4))
    plt.plot(1/frequency, power, color=colors[1])
    plt.xlabel('Period (days)')
    plt.ylabel('Lomb-Scargle Power')
    plt.title('F10.7 Lomb-Scargle Periodogram')
    plt.xscale('log')
    plt.xlim(10, 5000)
    plt.grid()
    plt.savefig('f107_lombscargle.png', dpi=600, bbox_inches='tight')
    #plt.show()

    # (c) Residuals spectrum
    if len(residuals) > 0:
        res_nonan = np.nan_to_num(residuals, nan=np.nanmean(residuals))
        freqs_r, pxx_r = scipy.signal.periodogram(res_nonan, fs=fs)
        plt.figure(figsize=(10.74, 4))
        plt.semilogy(freqs_r, pxx_r, color=colors[2])
        plt.xlabel('Frequency (1/day)')
        plt.ylabel('Power')
        plt.title('Residuals Periodogram')
        plt.xlim(0, 0.05)
        plt.grid()
        plt.savefig('residuals_periodogram.png', dpi=600, bbox_inches='tight')
        #plt.show()
        # Lomb-Scargle for residuals
        t_res = np.arange(len(res_nonan))
        power_r = LombScargle(t_res, res_nonan).power(frequency)
        plt.figure(figsize=(10.74, 4))
        plt.plot(1/frequency, power_r, color=colors[3])
        plt.xlabel('Period (days)')
        plt.ylabel('Lomb-Scargle Power')
        plt.title('Residuals Lomb-Scargle Periodogram')
        plt.xscale('log')
        plt.xlim(10, 5000)
        plt.grid()
        plt.savefig('residuals_lombscargle.png', dpi=600, bbox_inches='tight')
        #plt.show()

    # 3. Forecast errors as a function of time and solar cycle phase
    # (a) Residuals over time
    if len(residuals) > 0:
        # Try to reshape residuals to (n_samples, forecast_horizon, n_vars) if possible
        n_val = len(ds_full.dates)
        n_out = ds_full.n_target_vars  # Use actual number of target variables
        try:
            # Try to infer forecast_horizon from y_batch shape
            forecast_horizon = pred_len
            n_samples = len(residuals) // (forecast_horizon * n_out)
            residuals_reshaped = residuals.reshape(n_samples, forecast_horizon, n_out)
            # Take mean over all samples and variables for each forecast step
            residuals_mean = residuals_reshaped.mean(axis=(0,2))
            res_dates = ds_full.dates[-forecast_horizon:]
            plt.figure(figsize=(10.74, 4))
            plt.plot(res_dates, residuals_mean, color=colors[4], alpha=0.7)
            plt.xlabel('Date')
            plt.ylabel('Mean Prediction Residual')
            plt.title('Forecast Residuals Over Time')
            plt.grid()
            plt.savefig('residuals_time.png', dpi=600, bbox_inches='tight')
            #plt.show()
        except Exception as e:
            print(f"Could not reshape residuals for time plot: {e}")

    # Plot Training vs. Validation Loss
    train = loss_history.train_losses
    val   = loss_history.val_losses

    # Detect-and-drop the extra sanity-check entry
    if len(val) == len(train) + 1:
        val = val[1:]
    epochs = range(1, len(train) + 1)
    plt.figure(figsize=(7.16, 3.45))
    plt.plot(epochs, train, label='Training Loss',   linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig('train_loss.png', dpi=600, bbox_inches='tight')
    #plt.show()

    # Plot Obs + Forecast
    obs_i = ds_full.cols.index('Obs')
    obs_hist = ds_full.raw[:, obs_i].copy()
    obs_hist[obs_hist == 0] = np.nan
    obs_hist = pd.Series(obs_hist, index=ds_full.dates).ffill().values

    plt.figure(figsize=(7.16, 3.45))
    plt.plot(ds_full.dates, obs_hist, label='Historical Obs')
    plt.plot(df_fut.index, df_fut['Obs'], label='22-Year Forecast Obs')
    plt.xlabel('Date')
    plt.ylabel('10.7 cm Solar Radio Flux')
    plt.title('Historical + 22-Year Obs Forecast')
    plt.legend()
    plt.grid()
    plt.savefig('obs_forecast.png', dpi=600, bbox_inches='tight')
    #plt.show()

    # Plot Adj + Forecast
    adj_i = ds_full.cols.index('Adj')
    adj_hist = ds_full.raw[:, adj_i].copy()
    adj_hist[adj_hist == 0] = np.nan
    adj_hist = pd.Series(adj_hist, index=ds_full.dates).ffill().values

    plt.figure(figsize=(7.16, 3.45))
    plt.plot(ds_full.dates, adj_hist, label='Historical Adj')
    plt.plot(df_fut.index, df_fut['Adj'], label='22-Year Forecast Adj')
    plt.xlabel('Date')
    plt.ylabel('10.7 cm Solar Radio Flux')
    plt.title('Historical + 22-Year Adj Forecast')
    plt.legend()
    plt.grid()
    plt.savefig('adj_forecast.png', dpi=600, bbox_inches='tight')
    #plt.show()

    # Plot URSI-D + Forecast
    ursi_i = ds_full.cols.index('URSI-D')
    ursi_hist = ds_full.raw[:, ursi_i].copy()
    ursi_hist[ursi_hist == 0] = np.nan
    ursi_hist = pd.Series(ursi_hist, index=ds_full.dates).ffill().values

    plt.figure(figsize=(7.16, 3.45))
    plt.plot(ds_full.dates, ursi_hist, label='Historical URSI-D')
    plt.plot(df_fut.index, df_fut['URSI-D'], label='22-Year Forecast URSI-D')
    plt.xlabel('Date')
    plt.ylabel('10.7 cm Solar Radio Flux')
    plt.title('Historical + 22-Year URSI-D Forecast')
    plt.legend()
    plt.grid()
    plt.savefig('ursi_forecast.png', dpi=600, bbox_inches='tight')
    #plt.show()

    # Plot ap + Forecast
    ap_i = ds_full.cols.index('ap')
    ap_hist = ds_full.raw[:, ap_i].copy()
    ap_hist[ap_hist == 0] = np.nan
    ap_hist = pd.Series(ap_hist, index=ds_full.dates).ffill().values
    if ap_regression_model is not None and ap_ar_coeffs is not None and simulate_ap_fn is not None:
        # Use the forecasted F10.7 (Adj) to generate conditional Ap
        f107_forecast = df_fut['Adj'].values
        dates_forecast = df_fut.index
        
        # Provide 40 days of historical F10.7 for F81c calculation
        if df_hist is not None and len(df_hist) >= 40:
            historical_f107 = df_hist["Adj"].values[-40:]
        else:
            historical_f107 = None  # Will fall back to beginning of forecast in simulate_ap_forecast
        
        ap_conditional_mean, ap_conditional_sample, _ = simulate_ap_fn(
            f107_forecast=f107_forecast,
            dates_forecast=dates_forecast,
            harmonics=HARMONICS,
            regression_model=ap_regression_model,
            ar_coeffs=ap_ar_coeffs,
            sigma=ap_sigma,
            eps_init=ap_eps_init,
            historical_f107=historical_f107,
            last_train_idx=last_train_idx if last_train_idx is not None else len(ds_full.dates)-1
        )
        
        # Plot conditional Ap forecast
        plt.figure(figsize=(7.16, 3.45))
        plt.plot(ds_full.dates, ap_hist, label='Historical ap')
        plt.plot(dates_forecast, ap_conditional_mean, label='Conditional ap Mean', linewidth=2)
        plt.plot(dates_forecast, ap_conditional_sample, label='Conditional ap Sample', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('ap (Geomagnetic Activity Index)')
        plt.title('Historical + Conditional ap Forecast')
        plt.legend()
        plt.grid()
        plt.savefig('ap_conditional_forecast.png', dpi=600, bbox_inches='tight')
        #plt.show()
        
        # Save conditional Ap forecast to CSV
        df_ap_conditional = pd.DataFrame({
            'date': dates_forecast,
            'ap_mean': ap_conditional_mean,
            'ap_sample': ap_conditional_sample
        })
        df_ap_conditional.to_csv('ap_conditional_forecast.csv', index=False)
        print(f"Conditional Ap forecast saved to ap_conditional_forecast.csv")
    
    # Fourier‐Feature Visualization (all HARMONICS)
    full_days = np.arange(orig_T + forecast_years * pred_len, dtype=np.float32)
    cycle_len = 11 * 365

    # Plot all seasonal harmonics (sin/cos)
    fig, axs = plt.subplots(HARMONICS, 1, figsize=(7.16, 1.15 * HARMONICS), sharex=True)
    for h in range(HARMONICS):
        rad1 = 2 * np.pi * (h + 1) * full_days[:orig_T] / 365.0
        sin1 = np.sin(rad1)
        cos1 = np.cos(rad1)
        axs[h].plot(ds_full.dates, sin1, color=colors[2*h % len(colors)])
        axs[h].plot(ds_full.dates, cos1, color=colors[(2*h+1) % len(colors)])
        axs[h].set_ylabel('Amplitude')
        axs[h].set_title(f'Seasonal Fourier Features (Order {h+1})')
        axs[h].grid()
    axs[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('seasonal_harmonics.png', dpi=600, bbox_inches='tight')
    #plt.show()

    # Plot all solar-cycle harmonics (sin/cos)
    fig, axs = plt.subplots(HARMONICS, 1, figsize=(7.16, 1.15 * HARMONICS), sharex=True)
    for h in range(HARMONICS):
        rad2 = 2 * np.pi * (h + 1) * (full_days[:orig_T] % cycle_len) / cycle_len
        sin2 = np.sin(rad2)
        cos2 = np.cos(rad2)
        axs[h].plot(ds_full.dates, sin2, color=colors[2*h % len(colors)])
        axs[h].plot(ds_full.dates, cos2, color=colors[(2*h+1) % len(colors)])
        axs[h].set_ylabel('Amplitude')
        axs[h].set_title(f'Solar-Cycle Fourier Features (Order {h+1})')
        axs[h].grid()
    axs[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('solar_cycle_harmonics.png', dpi=600, bbox_inches='tight')
    #plt.show()

    # Print final model summary
    print(final_model.model)
    print(f"Total parameters: {sum(p.numel() for p in final_model.model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in final_model.model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"Non-trainable parameters: {sum(p.numel() for p in final_model.model.parameters() if not p.requires_grad) / 1e6:.2f}M")
    print(f"Total layers: {sum(1 for _ in final_model.model.modules())}")
    print(f"Total modules: {sum(1 for _ in final_model.model.modules() if isinstance(_, torch.nn.Module))}")
    print(f"Total submodules: {sum(1 for _ in final_model.model.modules() if isinstance(_, torch.nn.ModuleList))}")
    print(f"Total buffers: {sum(1 for _ in final_model.model.buffers())}")
    print(f"Total parameters in ResNeXt: {sum(p.numel() for p in final_model.model.cnn.parameters()) / 1e6:.2f}M")
    print(f"Total parameters in iTransformer: {sum(p.numel() for p in final_model.model.itransformer.parameters()) / 1e6:.2f}M")

    # Generate ResNeXt feature visualizations
    print("\n" + "="*60)
    print("GENERATING RESNEXT FEATURE VISUALIZATIONS")
    print("="*60)
    feature_files = visualize_resnext_features(model=final_model, dataloader=dm.val_dataloader(),
                                               output_dir="visualizations_features", max_samples=None)
    print(f"Generated {len(feature_files)} feature visualization files")

    # 1-D CNN VISUALIZATION
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE 1-D CNN VISUALIZATION")
    print("="*80)
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model.eval().to(device)
    
    # Get batch from dataloader
    batch_x, _ = next(iter(dm.full_dataloader()))
    batch_x = batch_x.to(device)
    
    print(f"Device: {device}")
    print(f"Batch shape: {batch_x.shape}")
    print(f"Model: {final_model.__class__.__name__}")
    
    # Create output directories
    kernel_dir = "visualization_kernels"
    activation_dir = "visualization_activations"
    
    print(f"\nOutput directories:")
    print(f"  Kernels: {kernel_dir}")
    print(f"  Activations: {activation_dir}")
    
    # 1. Visualize all kernels/filters
    print("\n" + "-"*50)
    print("VISUALIZING KERNELS/FILTERS")
    print("-"*50)
    
    kernel_files = visualize_all_kernels(model=final_model, root_name="ResNeXt_iTransformer",
                                         out_dir=kernel_dir)
    print(f"Generated {len(kernel_files)} kernel visualization files")
    
    # 2. Visualize activations/feature maps
    print("\n" + "-"*50)
    print("VISUALIZING ACTIVATIONS/FEATURE MAPS")
    print("-"*50)
    
    activation_files = visualize_activations(model=final_model, batch_x=batch_x,
                                             root_name="ResNeXt_iTransformer", out_dir=activation_dir)
    print(f"Generated {len(activation_files)} activation visualization files")
    
    # Summary
    print("\n" + "="*50)
    print("FINAL VISUALIZATION SUMMARY")
    print("="*50)
    print(f"ResNeXt feature files: {len(feature_files)}")
    print(f"Kernel visualization files: {len(kernel_files)}")
    print(f"Activation visualization files: {len(activation_files)}")
    print(f"Total visualization files: {len(feature_files) + len(kernel_files) + len(activation_files)}")
    
    # List all Conv1d layers found
    conv_modules = list_conv_modules(final_model, include_1d=True)
    print(f"\nConv1d layers found: {len(conv_modules)}")
    for i, (name, module) in enumerate(conv_modules.items()):
        weight_shape = module.weight.shape
        print(f"  {i+1}. {name}: {weight_shape}")
    
    # Save predictions to CSV file
    print("\n" + "="*60)
    print("SAVING PREDICTIONS TO CSV")
    print("="*60)
    
    # Combine historical and forecast data
    historical_df = pd.DataFrame(ds_full.raw[:, :ds_full.n_target_vars], 
                                 index=ds_full.dates, 
                                 columns=ds_full.cols[:ds_full.n_target_vars])
    
    # Mark data type
    historical_df['data_type'] = 'historical'
    df_fut['data_type'] = 'forecast'
    
    # Combine into single dataframe
    combined_df = pd.concat([historical_df, df_fut])
    
    # Save to CSV
    csv_path = 'solar_flux_predictions.csv'
    combined_df.to_csv(csv_path, index_label='date')
    print(f"Predictions saved to {csv_path}")
    print(f"  - Historical records: {len(historical_df)}")
    print(f"  - Forecast records: {len(df_fut)}")
    print(f"  - Total records: {len(combined_df)}")
    
    # Also save just the forecast
    forecast_csv_path = 'solar_flux_forecast_only.csv'
    df_fut.to_csv(forecast_csv_path, index_label='date')
    print(f"Forecast-only data saved to {forecast_csv_path}")

    # Save SW-All schema compatible outputs for downstream Fortran workflows.
    sw_cols = [
        'DATE', 'BSRN', 'ND',
        'KP1', 'KP2', 'KP3', 'KP4', 'KP5', 'KP6', 'KP7', 'KP8', 'KP_SUM',
        'AP1', 'AP2', 'AP3', 'AP4', 'AP5', 'AP6', 'AP7', 'AP8', 'AP_AVG',
        'CP', 'C9', 'ISN',
        'F10.7_OBS', 'F10.7_ADJ', 'F10.7_DATA_TYPE',
        'F10.7_OBS_CENTER81', 'F10.7_OBS_LAST81', 'F10.7_ADJ_CENTER81', 'F10.7_ADJ_LAST81',
    ]

    project_root = Path(__file__).resolve().parents[2]
    sw_input_path = project_root / 'SW-All.csv'

    if sw_input_path.exists():
        sw_hist_raw = pd.read_csv(sw_input_path, skipinitialspace=True)
        sw_hist_raw.columns = [str(col).strip() for col in sw_hist_raw.columns]
        sw_hist_raw['__date'] = pd.to_datetime(sw_hist_raw['DATE'], errors='coerce')
        sw_hist_raw = sw_hist_raw.dropna(subset=['__date']).sort_values('__date')
        sw_hist = sw_hist_raw[[col for col in sw_cols if col in sw_hist_raw.columns]].copy()
        for missing_col in [c for c in sw_cols if c not in sw_hist.columns]:
            sw_hist[missing_col] = np.nan
        sw_hist = sw_hist[sw_cols]

        hist_dates = sw_hist_raw['__date'].to_numpy()
        if len(hist_dates) > 0:
            last_hist_row = sw_hist_raw.iloc[-1]
            last_bsrn = int(pd.to_numeric(last_hist_row.get('BSRN', np.nan), errors='coerce')) if pd.notna(pd.to_numeric(last_hist_row.get('BSRN', np.nan), errors='coerce')) else 0
            last_nd = int(pd.to_numeric(last_hist_row.get('ND', np.nan), errors='coerce')) if pd.notna(pd.to_numeric(last_hist_row.get('ND', np.nan), errors='coerce')) else 0
        else:
            last_bsrn = 0
            last_nd = 0
    else:
        sw_hist = pd.DataFrame(columns=sw_cols)
        hist_dates = np.array([], dtype='datetime64[ns]')
        last_bsrn = 0
        last_nd = 0

    ap_source = ap_conditional_mean if ap_conditional_mean is not None else df_fut_all['ap'].values
    ap_source = np.clip(np.asarray(ap_source, dtype=float), 0.0, 400.0)
    kp_avg = np.clip(np.asarray(df_fut_all['Kp'].values, dtype=float), 0.0, 9.0)
    kp_slot = np.rint(kp_avg * 10.0).astype(int)
    kp_sum = np.rint(kp_avg * 80.0).astype(int)
    ap_slot = np.rint(ap_source).astype(int)

    nd_vals = np.array([((last_nd + i) % 27) + 1 for i in range(len(df_fut_all))], dtype=int)
    bsrn_vals = np.array([last_bsrn + ((last_nd + i) // 27) for i in range(len(df_fut_all))], dtype=int)

    ison = np.rint(np.asarray(df_fut_all['Total'].values, dtype=float)).astype(int)
    cp = np.round(np.clip(kp_avg / 3.0, 0.0, 2.5), 1)
    c9 = np.rint(np.clip(cp * 3.6, 0.0, 9.0)).astype(int)

    sw_forecast = pd.DataFrame({
        'DATE': [f"{d.month}/{d.day}/{d.year}" for d in df_fut_all.index],
        'BSRN': bsrn_vals,
        'ND': nd_vals,
        'KP1': kp_slot, 'KP2': kp_slot, 'KP3': kp_slot, 'KP4': kp_slot,
        'KP5': kp_slot, 'KP6': kp_slot, 'KP7': kp_slot, 'KP8': kp_slot,
        'KP_SUM': kp_sum,
        'AP1': ap_slot, 'AP2': ap_slot, 'AP3': ap_slot, 'AP4': ap_slot,
        'AP5': ap_slot, 'AP6': ap_slot, 'AP7': ap_slot, 'AP8': ap_slot,
        'AP_AVG': np.round(ap_source, 1),
        'CP': cp,
        'C9': c9,
        'ISN': ison,
        'F10.7_OBS': np.round(np.asarray(df_fut_all['Obs'].values, dtype=float), 1),
        'F10.7_ADJ': np.round(np.asarray(df_fut_all['Adj'].values, dtype=float), 1),
        'F10.7_DATA_TYPE': 'PRD',
        'F10.7_OBS_CENTER81': np.nan,
        'F10.7_OBS_LAST81': np.nan,
        'F10.7_ADJ_CENTER81': np.nan,
        'F10.7_ADJ_LAST81': np.nan,
    })[sw_cols]

    sw_combined = pd.concat([sw_hist, sw_forecast], ignore_index=True)

    sw_dates = pd.to_datetime(sw_combined['DATE'], errors='coerce')
    obs_vals = pd.to_numeric(sw_combined['F10.7_OBS'], errors='coerce')
    adj_vals = pd.to_numeric(sw_combined['F10.7_ADJ'], errors='coerce')

    obs_center_calc = obs_vals.rolling(window=81, center=True, min_periods=40).mean().ffill().bfill()
    obs_last_calc = obs_vals.rolling(window=81, min_periods=1).mean().ffill().bfill()
    adj_center_calc = adj_vals.rolling(window=81, center=True, min_periods=40).mean().ffill().bfill()
    adj_last_calc = adj_vals.rolling(window=81, min_periods=1).mean().ffill().bfill()

    for col_name, calc_vals in [
        ('F10.7_OBS_CENTER81', obs_center_calc),
        ('F10.7_OBS_LAST81', obs_last_calc),
        ('F10.7_ADJ_CENTER81', adj_center_calc),
        ('F10.7_ADJ_LAST81', adj_last_calc),
    ]:
        current = pd.to_numeric(sw_combined[col_name], errors='coerce')
        sw_combined[col_name] = current.where(current.notna(), np.round(calc_vals, 1))

    # Preserve historical rows from input SW-All and append forecast rows.
    sw_forecast_only_path = 'SW-All_forecast_only.csv'
    sw_predictions_path = 'SW-All_predictions.csv'

    sw_forecast.to_csv(sw_forecast_only_path, index=False)
    sw_combined.to_csv(sw_predictions_path, index=False)

    print(f"SW-All format forecast saved to {sw_forecast_only_path}")
    print(f"SW-All format combined history+forecast saved to {sw_predictions_path}")
    print(f"  - SW format forecast records: {len(sw_forecast)}")
    print(f"  - SW format combined records: {len(sw_combined)}")
    
    print("\n" + "="*80)
    print("SCRIPT EXECUTION COMPLETE")
    print("="*80)