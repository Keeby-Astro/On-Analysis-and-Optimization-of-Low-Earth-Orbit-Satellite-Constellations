# CONDITIONAL AP MODEL - UTILITY FUNCTIONS
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def _resolve_input_path(path_value):
    if path_value is None:
        return None

    path_str = str(path_value)
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str

    search_roots = [
        os.getcwd(),
        os.path.dirname(__file__),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ]
    for root in search_roots:
        candidate = os.path.abspath(os.path.join(root, path_str))
        if os.path.exists(candidate):
            return candidate
    return path_str


def _pick_column(columns, candidates):
    candidate_set = {c.lower() for c in candidates}
    for col in columns:
        if str(col).strip().lower() in candidate_set:
            return col
    return None


def _load_sunspot_daily(sunspot_path, start_ts, end_ts):
    with open(sunspot_path, 'r', encoding='utf-8', errors='ignore') as fh:
        first_line = fh.readline()

    if ';' in first_line and first_line.count(';') >= 6:
        sunspot = pd.read_csv(
            sunspot_path,
            sep=';',
            engine='python',
            header=None,
            names=['Year', 'Month', 'Day', 'Fraction', 'Total', 'Observations', 'Std', 'Indicator'],
        )
    else:
        sunspot = pd.read_csv(sunspot_path, sep=',', engine='python', skipinitialspace=True)
        sunspot.columns = [str(col).strip().replace('"', '') for col in sunspot.columns]

    year_col = _pick_column(sunspot.columns, ['Year'])
    month_col = _pick_column(sunspot.columns, ['Month'])
    day_col = _pick_column(sunspot.columns, ['Day'])
    total_col = _pick_column(sunspot.columns, ['Total'])

    if None in (year_col, month_col, day_col, total_col):
        raise ValueError("Sunspot file must contain Year, Month, Day, and Total columns")

    sunspot[year_col] = pd.to_numeric(sunspot[year_col], errors='coerce')
    sunspot[month_col] = pd.to_numeric(sunspot[month_col], errors='coerce')
    sunspot[day_col] = pd.to_numeric(sunspot[day_col], errors='coerce')
    sunspot[total_col] = pd.to_numeric(sunspot[total_col], errors='coerce')

    sunspot = sunspot.dropna(subset=[year_col, month_col, day_col])
    sunspot['date'] = pd.to_datetime(
        dict(
            year=sunspot[year_col].astype(int),
            month=sunspot[month_col].astype(int),
            day=sunspot[day_col].astype(int),
        ),
        errors='coerce',
    )
    sunspot = sunspot.dropna(subset=['date'])
    sunspot = sunspot[(sunspot['date'] >= start_ts) & (sunspot['date'] <= end_ts)].copy()

    # SIDC daily files use -1 for missing values.
    sunspot[total_col] = sunspot[total_col].where(sunspot[total_col] >= 0, np.nan)

    return sunspot.groupby('date', as_index=False)[total_col].mean().rename(columns={total_col: 'Total_SN'})


def load_merged_solar_geomag_data(
    filename,
    kp_ap_file,
    sunspot_file='SN_d_tot_V2.0.csv',
    start_date="1957-01-01",
    end_date="2025-12-31",
):
    """Load SW-All, DST, and sunspot daily data into the model's expected schema."""

    sw_path = _resolve_input_path(filename)
    dst_path = _resolve_input_path(kp_ap_file)
    sunspot_path = _resolve_input_path(sunspot_file)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    sw = pd.read_csv(sw_path, skipinitialspace=True)
    sw.columns = [col.strip() for col in sw.columns]
    if 'DATE' not in sw.columns:
        raise ValueError("SW-All CSV must contain a 'DATE' column")

    sw['date'] = pd.to_datetime(sw['DATE'], errors='coerce')
    sw = sw.dropna(subset=['date'])
    sw = sw[(sw['date'] >= start_ts) & (sw['date'] <= end_ts)].copy()

    sw_required = ['BSRN', 'F10.7_OBS', 'F10.7_ADJ', 'F10.7_ADJ_CENTER81', 'KP_SUM', 'AP_AVG']
    missing_sw = [col for col in sw_required if col not in sw.columns]
    if missing_sw:
        raise ValueError(f"SW-All CSV is missing required columns: {missing_sw}")
    for col in sw_required:
        sw[col] = pd.to_numeric(sw[col], errors='coerce')

    dst = pd.read_csv(dst_path, skipinitialspace=True)
    dst.columns = [col.strip() for col in dst.columns]

    dst_date_col = 'Date' if 'Date' in dst.columns else ('DATE' if 'DATE' in dst.columns else None)
    if dst_date_col is None:
        raise ValueError("DST CSV must contain a 'Date' column")

    dst_value_col = next((col for col in dst.columns if col.upper() == 'DST'), None)
    if dst_value_col is None:
        raise ValueError("DST CSV must contain a 'DST' column")

    dst['date'] = pd.to_datetime(dst[dst_date_col], errors='coerce')
    dst[dst_value_col] = pd.to_numeric(dst[dst_value_col], errors='coerce')
    dst = dst.dropna(subset=['date'])
    dst = dst[(dst['date'] >= start_ts) & (dst['date'] <= end_ts)].copy()
    dst_daily = dst.groupby('date', as_index=False)[dst_value_col].mean().rename(columns={dst_value_col: 'DST'})

    sunspot_daily = _load_sunspot_daily(sunspot_path, start_ts, end_ts)

    full_dates = pd.DataFrame({'date': pd.date_range(start=start_ts, end=end_ts, freq='D')})
    df_merged = full_dates.merge(
        sw[['date', 'BSRN', 'F10.7_OBS', 'F10.7_ADJ', 'F10.7_ADJ_CENTER81', 'KP_SUM', 'AP_AVG']],
        on='date',
        how='left',
    ).merge(dst_daily, on='date', how='left').merge(sunspot_daily, on='date', how='left')

    df_merged['Rotation'] = df_merged['BSRN']
    df_merged['Obs'] = df_merged['F10.7_OBS']
    df_merged['Adj'] = df_merged['F10.7_ADJ']
    df_merged['URSI-D'] = df_merged['F10.7_ADJ_CENTER81']
    df_merged['ap'] = df_merged['AP_AVG']
    df_merged['Kp'] = df_merged['KP_SUM'] / 80.0
    df_merged['Kp'] = df_merged['Kp'].fillna(df_merged['ap'] / 8.0)
    df_merged['Total'] = df_merged['Total_SN']
    df_merged['Dst'] = df_merged['DST']

    value_cols = ['Rotation', 'Obs', 'Adj', 'URSI-D', 'Kp', 'ap', 'Total', 'Dst']
    df_merged[value_cols] = df_merged[value_cols].replace([np.inf, -np.inf], np.nan)
    df_merged[value_cols] = df_merged[value_cols].interpolate(method='linear', limit_direction='both')
    df_merged[value_cols] = df_merged[value_cols].ffill().bfill()
    df_merged['Kp'] = df_merged['Kp'].clip(0, 9)
    df_merged['ap'] = df_merged['ap'].clip(0, 400)

    return df_merged[['date', 'Rotation', 'Obs', 'Adj', 'URSI-D', 'Kp', 'ap', 'Total', 'Dst']]

def build_ap_regression_data(df, harmonics, cycle_len_days=11*365, f_col="Adj"):
    """Build regression features and target for Ap prediction.
    
    Drops rows with missing target (ap) or key predictors (F_t, F81c_t).
    Returns cleaned X, y, and the corresponding time indices.
    """
    df = df.sort_values("date").reset_index(drop=True)
    N = len(df)
    
    # Target: Ap
    y = df['ap'].values.astype(np.float32)
    
    # Feature 1: Daily F10.7
    F_t = df[f_col].values.astype(np.float32)
    
    # Feature 2: 81-day centered mean F81c_t
    F81c_t = df[f_col].rolling(window=81, center=True, min_periods=40).mean().values.astype(np.float32)
    # Handle edges with forward/backward fill
    F81c_t = pd.Series(F81c_t).ffill().bfill().values.astype(np.float32)
    
    # Feature 3: Solar cycle harmonics
    t_idx = np.arange(N, dtype=np.float32)
    harmonic_features = []
    for k in range(1, harmonics + 1):
        phi_k = np.sin(2 * np.pi * k * t_idx / cycle_len_days).astype(np.float32)
        psi_k = np.cos(2 * np.pi * k * t_idx / cycle_len_days).astype(np.float32)
        harmonic_features.append(phi_k)
        harmonic_features.append(psi_k)
    
    # Stack all features (NO constant column - bias handled by linear layer)
    X = np.column_stack([F_t, F81c_t] + harmonic_features)
    
    # Drop rows with NaN in target or key predictors
    valid_mask = ~(np.isnan(y) | np.isnan(F_t) | np.isnan(F81c_t))
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    t_idx_clean = t_idx[valid_mask].astype(np.int32)
    
    n_dropped = N - np.sum(valid_mask)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with missing Ap or F10.7 data ({100*n_dropped/N:.1f}%)")
    
    return X_clean, y_clean, t_idx_clean

class ConditionalApRegression(torch.nn.Module):
    """Linear regression model for conditional Ap prediction."""
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)

def fit_conditional_ap_regression(X, y, lr=1e-3, weight_decay=0.0, max_epochs=2000, tol=1e-7, device="cpu"):
    """Fit the conditional Ap regression model using gradient descent."""
    N, D = X.shape
    model = ConditionalApRegression(D).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    X_tensor = torch.from_numpy(X).float().to(device)
    y_tensor = torch.from_numpy(y).float().to(device)
    
    prev_loss = float('inf')
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Early stopping check
        if epoch > 0 and epoch % 100 == 0:
            loss_val = loss.item()
            if abs(prev_loss - loss_val) < tol:
                break
            prev_loss = loss_val
    
    # Move model to CPU and compute fitted values
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        y_hat = model(torch.from_numpy(X).float()).numpy()
    
    return model, y_hat

def fit_ar_residuals(eps, p=1):
    """Fit AR(p) model to residuals using least squares."""
    N = len(eps)
    if N <= 2 * p:
        raise ValueError(f"Insufficient data: N={N}, need N > 2*p={2*p}")
    
    # Construct design matrix Z and target Y
    Y = eps[p:]
    Z = np.zeros((N - p, p), dtype=np.float32)
    for i in range(N - p):
        for j in range(p):
            Z[i, j] = eps[p + i - 1 - j]
    
    # Solve via least squares
    a, residuals, rank, s = np.linalg.lstsq(Z, Y, rcond=None)
    
    # Compute innovation variance
    eta_hat = Y - Z @ a
    sigma = np.sqrt(np.mean(eta_hat ** 2))
    
    # Last p residuals for initialization (physical initial conditions)
    eps_init = eps[-p:].copy()
    
    return a.astype(np.float32), float(sigma), eps_init.astype(np.float32)

def simulate_ap_forecast(f107_forecast, dates_forecast, harmonics, regression_model, ar_coeffs, sigma, eps_init, 
                        cycle_len_days=11*365, f_col_name="Adj", rng=None, historical_f107=None, last_train_idx=None):
    """Simulate Ap forecast conditioned on F10.7 forecast using regression + AR residuals."""
    if rng is None:
        rng = np.random.default_rng()
    
    T = len(f107_forecast)
    p = len(ar_coeffs)
    
    # Construct future DataFrame
    df_future = pd.DataFrame({'date': pd.to_datetime(dates_forecast), f_col_name: f107_forecast})
    df_future = df_future.sort_values('date').reset_index(drop=True)
    
    # Build F81c_t with historical context for proper 81-day centered mean
    # Physics: F81c is NRLMSIS standard, requires 40 days before and after
    if historical_f107 is not None and len(historical_f107) >= 40:
        # Concatenate historical context with forecast
        f107_extended = np.concatenate([historical_f107[-40:], f107_forecast])
        F81c_extended = pd.Series(f107_extended).rolling(window=81, center=True, min_periods=40).mean()
        # Extract forecast portion (skip historical context)
        F81c_future = F81c_extended.values[40:].astype(np.float32)
    else:
        # Fallback: use forecast only (will have edge effects)
        F81c_future = df_future[f_col_name].rolling(window=81, center=True, min_periods=1).mean().values.astype(np.float32)
    
    # Handle any remaining NaN values
    F81c_future = pd.Series(F81c_future).ffill().bfill().values.astype(np.float32)
    
    # Build design matrix X_future (NO constant column - bias in linear layer)
    F_t = df_future[f_col_name].values.astype(np.float32)
    
    # Maintain harmonic phase continuity with training data
    # Physics: Solar cycle phase must be continuous across train/forecast boundary
    if last_train_idx is not None:
        t_idx = np.arange(last_train_idx + 1, last_train_idx + 1 + T, dtype=np.float32)
    else:
        # Fallback: start from 0 (will have phase discontinuity)
        t_idx = np.arange(T, dtype=np.float32)
    
    harmonic_features = []
    for k in range(1, harmonics + 1):
        phi_k = np.sin(2 * np.pi * k * t_idx / cycle_len_days).astype(np.float32)
        psi_k = np.cos(2 * np.pi * k * t_idx / cycle_len_days).astype(np.float32)
        harmonic_features.append(phi_k)
        harmonic_features.append(psi_k)
    
    X_future = np.column_stack([F_t, F81c_future] + harmonic_features)
    
    # Compute conditional mean Ap
    regression_model.eval()
    with torch.no_grad():
        mu_t = regression_model(torch.from_numpy(X_future).float()).numpy()
    
    # Simulate AR(p) residuals with physical initial conditions
    # eps_init contains the last p residuals from the training period
    # Use buffer of length T + p to properly seed the AR recursion
    eps_buf = np.zeros(T + p, dtype=np.float32)
    eps_buf[:p] = eps_init  # Seed with historical residuals
    
    for t in range(p, T + p):
        eta_t = sigma * rng.standard_normal()
        eps_buf[t] = np.sum(ar_coeffs * eps_buf[t-p:t][::-1]) + eta_t
    
    eps_sim = eps_buf[p:]  # Extract the T forecast residuals
    
    # Construct Ap forecast with physically realistic bounds
    # Physics: Ap index ranges from 0 (quiet) to 400 (extreme storm)
    Ap_sample = mu_t + eps_sim
    Ap_sample = np.clip(Ap_sample, 0.0, 400.0)  # Clip to realistic range [0, 400]
    
    return mu_t, Ap_sample, eps_sim

def train_conditional_ap_model(filename, kp_ap_file, sunspot_file, harmonics, f_col="Adj", p=1):
    df = load_merged_solar_geomag_data(filename, kp_ap_file, sunspot_file)
    X, y, t_idx = build_ap_regression_data(df, harmonics, f_col=f_col)
    regression_model, y_hat = fit_conditional_ap_regression(X, y)
    eps = y - y_hat
    ar_coeffs, sigma, eps_init = fit_ar_residuals(eps, p=p)
    last_train_idx = int(t_idx[-1])  # For harmonic phase continuity
    return regression_model, ar_coeffs, sigma, eps_init, df, last_train_idx