# Model Plotting and Visualization Module
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import torch
from astropy.timeseries import LombScargle

plt.rcParams['savefig.transparent'] = True


def generate_visualizations_from_csv(predictions_csv, ap_conditional_csv=None, colors=None):
    if colors is None:
        colors = ['#1965B0', '#E8601C', '#4EB265', '#72190E', '#882E72',
                  '#437DBF', '#F1932D', '#90C987', '#A5170E', '#994F88',
                  '#6195CF', '#F6C141', '#CAE0AB', '#DC050C', '#AA6F9E',
                  '#7BAFDE', '#F7F056', '#8B8B8B', '#896D67', '#BA8DB4']

    df = pd.read_csv(predictions_csv)
    if 'date' not in df.columns:
        raise ValueError(f"Expected a 'date' column in {predictions_csv}")

    df['date'] = pd.to_datetime(df['date'])
    if 'data_type' in df.columns:
        hist_df = df[df['data_type'].str.lower() == 'historical'].copy()
        fut_df = df[df['data_type'].str.lower() == 'forecast'].copy()
    else:
        hist_df = pd.DataFrame(columns=df.columns)
        fut_df = df.copy()

    if fut_df.empty:
        raise ValueError("No forecast rows found. Ensure CSV contains data_type='forecast' rows.")

    # Prepare historical columns with same behavior as model-based plotting
    for col in ['Obs', 'Adj', 'URSI-D', 'ap']:
        if col in hist_df.columns:
            series = hist_df[col].astype(float).copy()
            series[series == 0] = np.nan
            hist_df[col] = pd.Series(series.values, index=hist_df['date']).ffill().values

    # Plot Obs + Forecast
    if 'Obs' in fut_df.columns:
        plt.figure(figsize=(8, 3))
        if not hist_df.empty and 'Obs' in hist_df.columns:
            plt.plot(hist_df['date'], hist_df['Obs'], label='Historical Obs')
        plt.plot(fut_df['date'], fut_df['Obs'], label='Forecast Obs')
        plt.xlabel('Date')
        plt.ylabel('10.7 cm Solar Radio Flux (sfu)')
        plt.title('Historical + Forecast Observed Solar Radio Flux')
        plt.legend()
        plt.grid()
        plt.savefig('obs_forecast.png', dpi=600, bbox_inches='tight')

    # Plot Adj + Forecast
    if 'Adj' in fut_df.columns:
        plt.figure(figsize=(8, 3))
        if not hist_df.empty and 'Adj' in hist_df.columns:
            plt.plot(hist_df['date'], hist_df['Adj'], label='Historical Adj')
        plt.plot(fut_df['date'], fut_df['Adj'], label='Forecast Adj')
        plt.xlabel('Date')
        plt.ylabel('10.7 cm Solar Radio Flux (sfu)')
        plt.title('Historical + Forecast Adjusted Solar Radio Flux')
        plt.legend()
        plt.grid()
        plt.savefig('adj_forecast.png', dpi=600, bbox_inches='tight')

    # Plot URSI-D + Forecast
    if 'URSI-D' in fut_df.columns:
        plt.figure(figsize=(8, 3))
        if not hist_df.empty and 'URSI-D' in hist_df.columns:
            plt.plot(hist_df['date'], hist_df['URSI-D'], label='Historical URSI-D')
        plt.plot(fut_df['date'], fut_df['URSI-D'], label='Forecast URSI-D')
        plt.xlabel('Date')
        plt.ylabel('10.7 cm Solar Radio Flux (sfu)')
        plt.title('Historical + Forecast URSI-D Solar Radio Flux')
        plt.legend()
        plt.grid()
        plt.savefig('ursi_forecast.png', dpi=600, bbox_inches='tight')

    # Plot ap using conditional forecast if provided, otherwise direct ap forecast
    ap_used_conditional = False
    if ap_conditional_csv is not None:
        df_ap = pd.read_csv(ap_conditional_csv)
        if 'date' in df_ap.columns:
            df_ap['date'] = pd.to_datetime(df_ap['date'])
            if {'ap_mean', 'ap_sample'}.issubset(df_ap.columns):
                plt.figure(figsize=(7.16, 3.45))
                if not hist_df.empty and 'ap' in hist_df.columns:
                    plt.plot(hist_df['date'], hist_df['ap'], label='Historical ap')
                plt.plot(df_ap['date'], df_ap['ap_mean'], label='Conditional ap Mean', linewidth=2)
                plt.plot(df_ap['date'], df_ap['ap_sample'], label='Conditional ap Sample', alpha=0.5)
                plt.xlabel('Date')
                plt.ylabel('ap (Geomagnetic Activity Index)')
                plt.title('Historical + Conditional ap Forecast')
                plt.legend()
                plt.grid()
                plt.savefig('ap_conditional_forecast.png', dpi=600, bbox_inches='tight')
                ap_used_conditional = True

    if not ap_used_conditional and 'ap' in fut_df.columns:
        plt.figure(figsize=(7.16, 3.45))
        if not hist_df.empty and 'ap' in hist_df.columns:
            plt.plot(hist_df['date'], hist_df['ap'], label='Historical ap')
        plt.plot(fut_df['date'], fut_df['ap'], label='Forecast ap')
        plt.xlabel('Date')
        plt.ylabel('ap (Geomagnetic Activity Index)')
        plt.title('Historical + Forecast ap')
        plt.legend()
        plt.grid()
        plt.savefig('ap_forecast.png', dpi=600, bbox_inches='tight')
        
    print("CSV-based plotting complete.")
    print(f"Loaded predictions: {predictions_csv}")
    if ap_conditional_csv is not None:
        print(f"Loaded conditional ap: {ap_conditional_csv}")
    print("Generated files include: obs_forecast.png, adj_forecast.png, ursi_forecast.png, and ap plot outputs.")

def generate_visualizations_and_predictions(final_model, dm, loss_history, cv_losses, cv_maes, 
                                           forecast_years, pred_len, colors, HARMONICS=3,
                                           ap_regression_model=None, ap_ar_coeffs=None, 
                                           ap_sigma=None, ap_eps_init=None, simulate_ap_fn=None,
                                           last_train_idx=None, df_hist=None):
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