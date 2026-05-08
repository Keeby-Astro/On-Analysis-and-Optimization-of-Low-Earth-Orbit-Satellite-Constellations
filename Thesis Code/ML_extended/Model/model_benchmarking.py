# IMPORTS
import os
import random
from types import SimpleNamespace
from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import tqdm
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError

from Embed import DataEmbedding_inverted
from SelfAttention_Family import AttentionLayer, FullAttention
from Transformer_EncDec import Encoder, EncoderLayer
from model_plotting import generate_visualizations_and_predictions
from conditional_ap_model import (load_merged_solar_geomag_data, simulate_ap_forecast, 
                                  train_conditional_ap_model)
from benchmark import run_benchmarks, plot_benchmark_results, print_benchmark_summary

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
pl.seed_everything(SEED, workers=True)

# ACCELERATOR
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# Plotting settings
plt.rcParams.update({'figure.figsize': (10.74, 5.175),
                     'xtick.direction': 'in', 'xtick.labelsize': 10, 'xtick.major.size': 3,
                     'xtick.major.width': 0.5, 'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5,
                     'xtick.minor.visible': True, 'xtick.top': False,
                     'ytick.direction': 'in', 'ytick.labelsize': 10, 'ytick.major.size': 3,
                     'ytick.major.width': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
                     'ytick.minor.visible': True, 'ytick.right': False,
                     'axes.linewidth': 0.5, 'grid.linewidth': 0.5, 'lines.linewidth': 1.5,
                     'legend.fontsize': 10, 'legend.frameon': False,
                     'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 8, 'axes.labelsize': 12, 'axes.titlesize': 14,
                     'axes.grid': False, 'grid.linestyle': '--', 'grid.color': '0.5',
                     'lines.markersize': 8, 'axes.spines.top': True, 'axes.spines.right': True})

# Color palette for plots
colors = ['#1965B0', '#E8601C', '#4EB265', '#72190E', '#882E72',
          '#437DBF', '#F1932D', '#90C987', '#A5170E', '#994F88',
          '#6195CF', '#F6C141', '#CAE0AB', '#DC050C', '#AA6F9E',
          '#7BAFDE', '#F7F056', '#8B8B8B', '#896D67', '#BA8DB4']

# DATASET
class SolarFluxDataset(Dataset):
    def __init__(self, filename, seq_length, forecast_horizon,
                 columns = ['Rotation','Obs','Adj','URSI-D','ap'],  # Added 'ap' to defaults
                 kp_ap_file='kp_ap_index.txt', sunspot_file='sunspot_number.txt'):
        # Use the shared helper to load and merge data
        df_merged = load_merged_solar_geomag_data(filename, kp_ap_file, sunspot_file)

        # Convert year, month, day columns to pandas datetime
        self.dates = df_merged['date']
        # Add new features to columns (ap is now in target columns)
        all_columns = columns + ['Kp','Total']  # Only Kp and Total are exogenous now
        data = df_merged[all_columns].astype('float32').values
        self.raw   = data.copy() # Store raw data for later use
        self.cols  = all_columns     # Store column names
        self.n_target_vars = len(columns)  # Number of target variables (now includes ap)

        # --- Remove all engineered features except harmonics ---
        # No extra engineered features - keeping only solar cycle and seasonal harmonics
        self.mean   = data.mean(axis=0, keepdims=True)
        self.std    = data.std(axis=0, keepdims=True)
        self.series = (data - self.mean) / self.std
        self.seq_length       = seq_length
        self.forecast_horizon = forecast_horizon
        T = len(self.series)
        doy  = self.dates.dt.dayofyear.values.astype(np.float32) - 1
        rad1 = 2 * np.pi * doy / 365.0
        self.doy_sin = np.stack([np.sin((h+1) * rad1) for h in range(HARMONICS)], axis = 1) 
        self.doy_cos = np.stack([np.cos((h+1) * rad1) for h in range(HARMONICS)], axis = 1)
        idx = np.arange(len(self.series), dtype=np.float32)
        cycle_len = 11 * 365
        rad2 = 2 * np.pi * (idx % cycle_len) / cycle_len
        self.cyc_sin = np.stack([np.sin((h+1) * rad2) for h in range(HARMONICS)], axis = 1)
        self.cyc_cos = np.stack([np.cos((h+1) * rad2) for h in range(HARMONICS)], axis = 1)
        self.n_samples = T - (seq_length + forecast_horizon) + 1
        if self.n_samples < 1:
            raise ValueError(f"T = {T}, need at least {seq_length + forecast_horizon}")
        self._cyc_scale = float(self.std.mean())

    def set_normalization_from_indices(self, train_indices):
        train_data = self.raw[train_indices]
        self.mean = train_data.mean(axis=0, keepdims=True)
        self.std = train_data.std(axis=0, keepdims=True)
        # Recompute normalized series with training-only statistics
        self.series = (self.raw - self.mean) / self.std
        self._cyc_scale = float(self.std.mean())

    # Return the total number of samples in the dataset
    def __len__(self):
        return self.n_samples

    # Get the normalized input sequence for the given index
    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_length]  # (seq_length, n_features)

        # Slice out the harmonic features for this window
        sin1 = self.doy_sin[idx : idx + self.seq_length]   
        cos1 = self.doy_cos[idx : idx + self.seq_length]
        sin2 = self.cyc_sin[idx : idx + self.seq_length]
        cos2 = self.cyc_cos[idx : idx + self.seq_length]

        # Concatenate all harmonics: seasonal and cycle
        cyc_feats = np.concatenate([sin1, cos1, sin2, cos2], axis = 1)  
        cyc_feats *= self._cyc_scale

        # Combine with normalized series and cyclical features only (no extra engineered features)
        x_aug = np.concatenate([x, cyc_feats], axis=1).T        
        x_tensor = torch.from_numpy(x_aug).float()  # Ensure float32

        # Only use the original columns for the target y (exclude exogenous features)
        y_full = self.series[idx + self.seq_length : idx + self.seq_length + self.forecast_horizon]
        y_tensor = torch.from_numpy(y_full[:, :self.n_target_vars]).float()  # Only target columns
        return x_tensor, y_tensor

# 1D ResNeXt-101
class ResNeXtBottleneck1D(nn.Module):
    # ResNeXt expansion factor
    expansion = 4

    def __init__(self, in_ch, planes, stride = 1, dilation = 1,
                 cardinality = 32, bottleneck_width = 4):
        super().__init__()
        # Compute the width of the bottleneck (number of channels per group)
        D = cardinality * bottleneck_width

        # 1x1 convolution to reduce dimensionality
        self.conv1 = nn.Conv1d(in_ch, D, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm1d(D)
        # 3x3 grouped convolution for ResNeXt, with dilation and stride
        self.conv2 = nn.Conv1d(D, D, kernel_size = 3, stride = stride,
                               padding = dilation, dilation = dilation,
                               groups = cardinality, bias = False)
        self.bn2   = nn.BatchNorm1d(D)
        # 1x1 convolution to expand back to output channels
        self.conv3 = nn.Conv1d(D, planes * self.expansion,
                               kernel_size = 1, bias = False)
        self.bn3   = nn.BatchNorm1d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace = True)
        self.downsample = None
        # Downsample the identity if shape changes (stride or channel mismatch)
        if stride != 1 or in_ch != planes * self.expansion:
            self.downsample = nn.Sequential(nn.Conv1d(in_ch, planes * self.expansion,
                                                      kernel_size = 1, stride = stride, bias = False),
                                                      nn.BatchNorm1d(planes * self.expansion))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))   # 1x1 conv + BN + ReLU
        out = self.relu(self.bn2(self.conv2(out))) # 3x3 grouped conv + BN + ReLU
        out = self.bn3(self.conv3(out))            # 1x1 conv + BN
        
        # Downsample identity if needed
        if self.downsample is not None:
            identity = self.downsample(identity)
        # Add skip connection and apply ReLU
        return self.relu(out + identity)

# Multi-Scale ResNeXt-101
class MultiScaleResNeXt1D(nn.Module):
    def __init__(self, in_ch, layers, cardinality, width, base_ch, kernel_sizes = None):
        super().__init__()
        # Define dilations for each ResNeXt stage to capture multi-scale features
        dilations = [1, 2, 4, 8]
        # Set initial input channels for the first stage
        self.in_ch = base_ch

        # Initial convolutional layer, increases channel dimension and reduces sequence length
        self.conv1   = nn.Conv1d(in_ch, base_ch, kernel_size = 7,
                     stride = 2, padding = 3, bias = False)
        self.bn1     = nn.BatchNorm1d(base_ch)
        self.relu    = nn.ReLU(inplace = True)
        # Max pooling to further reduce sequence length
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)

        # Four sequential ResNeXt stages with increasing channels and dilation
        self.layer1 = self._make_stage(base_ch,    layers[0], stride = 1, dilation = dilations[0], cardinality = cardinality, bottleneck_width = width)
        self.layer2 = self._make_stage(base_ch*2,  layers[1], stride = 2, dilation = dilations[1], cardinality = cardinality, bottleneck_width = width)
        self.layer3 = self._make_stage(base_ch*4,  layers[2], stride = 2, dilation = dilations[2], cardinality = cardinality, bottleneck_width = width)
        self.layer4 = self._make_stage(base_ch*8,  layers[3], stride = 2, dilation = dilations[3], cardinality = cardinality, bottleneck_width = width)

    def _make_stage(self, planes, blocks, stride, dilation, cardinality, bottleneck_width):
        layers = []
        # First block may have stride > 1 and sets the input channels
        layers.append(ResNeXtBottleneck1D(self.in_ch, planes, stride = stride, dilation = dilation,
                                          cardinality = cardinality, bottleneck_width = bottleneck_width))
        # Update input channels for subsequent blocks
        self.in_ch = planes * ResNeXtBottleneck1D.expansion
        # Add remaining blocks (stride=1 for all except first)
        for _ in range(1, blocks):
            layers.append(ResNeXtBottleneck1D(self.in_ch, planes, stride = 1, dilation = dilation,
                                              cardinality = cardinality, bottleneck_width = bottleneck_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) # Initial convolution + batch norm + ReLU
        x = self.maxpool(x)                    # Max pooling to reduce sequence length
        x = self.layer1(x)                     # First ResNeXt stage
        x = self.layer2(x)                     # Second ResNeXt stage
        x = self.layer3(x)                     # Third ResNeXt stage
        x = self.layer4(x)                     # Fourth ResNeXt stage
        return x 

# iTRANSFORMER
class ITransformerModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len          = configs.seq_len          # Input sequence length
        self.pred_len         = configs.pred_len         # Prediction length
        self.output_attention = configs.output_attention # Attention weights
        self.use_norm         = configs.use_norm         # Normalization

        # Embedding layer for input sequence
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model,
                                                    configs.embed, configs.freq, configs.dropout)
        # Encoder with multiple attention layers
        self.encoder = Encoder([EncoderLayer(AttentionLayer(FullAttention(False, configs.factor,
                                                                          attention_dropout=configs.dropout,
                                                                          output_attention=configs.output_attention),
                                                            configs.d_model, configs.n_heads),
                                            configs.d_model, configs.d_ff, dropout=configs.dropout,
                                            activation=configs.activation) for _ in range(configs.e_layers)],
                                norm_layer=torch.nn.LayerNorm(configs.d_model))
        # Projector for mapping encoder output to prediction length
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Normalize input sequence (zero mean, unit variance)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # Pass through embedding and encoder (with attention)
        enc_out, attns = self.encoder(self.enc_embedding(x_enc, x_mark_enc), attn_mask=None)
        # Project encoder output to prediction length and permute dimensions
        dec_out = self.projector(enc_out).permute(0, 2, 1)

        # Denormalize output
        if self.use_norm:
            dec_out = (dec_out * stdev[:,0,:].unsqueeze(1).repeat(1, self.pred_len, 1)
            ) + means[:,0,:].unsqueeze(1).repeat(1, self.pred_len, 1)

        # Return output and attention weights
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

# ResNeXt + iTransformer
class ResNeXtITransformer(nn.Module):
    def __init__(self, cnn_params, ts_params, in_vars, out_vars, seq_len):
        super().__init__()
        # Initialize ResNeXt-101 backbone with specified parameters
        self.cnn = MultiScaleResNeXt1D(in_ch = in_vars, layers = cnn_params['layers'],
                                       cardinality = cnn_params['cardinality'],
                                       width = cnn_params['width'],
                                       base_ch = cnn_params['base_channels'])
        # Compute the number of output channels after the ResNeXt backbone
        in_c = cnn_params['base_channels'] * 8 * ResNeXtBottleneck1D.expansion
        self.reducer = nn.Conv1d(in_c, ts_params['d_model'], kernel_size=1)

        # Downsample factor for sequence length reduction
        with torch.no_grad():
            dummy = torch.zeros(1, in_vars, seq_len)
            cnn_out = self.cnn(dummy)
            L_out = cnn_out.shape[-1]

        # Initialize iTransformer model with specified parameters
        configs = SimpleNamespace(seq_len = L_out, pred_len = ts_params['pred_len'],
                                  output_attention=False, use_norm=True, embed = 'fixed',
                                  freq = 'd', dropout = ts_params['dropout'], factor = None,
                                  d_model = ts_params['d_model'], n_heads = ts_params['n_heads'],
                                  d_ff = ts_params['d_ff'], e_layers = ts_params['num_layers'],
                                  activation = 'relu')
        self.itransformer = ITransformerModel(configs)
        self.fc_out = nn.Linear(ts_params['d_model'], out_vars)

    def forward(self, x):
        f      = self.cnn(x)              # Pass through ResNeXt-101 backbone
        r      = self.reducer(f)          # Reduce channels to match iTransformer input
        x_enc  = r.permute(0, 2, 1)       # Permute dimensions for iTransformer
        y_pred = self.itransformer(x_enc) # Pass through iTransformer
        return self.fc_out(y_pred)

# LIGHTNING MODULE
class SolarFluxLitModel(pl.LightningModule):
    def __init__(self, cnn_params, ts_params, in_vars, out_vars, seq_len, lr, max_epochs):
        super().__init__()
        self.save_hyperparameters()

        # Model and loss
        self.model     = ResNeXtITransformer(cnn_params, ts_params, in_vars, out_vars, seq_len)
        self.criterion = nn.MSELoss()

        # Validation metrics
        self.val_mae  = MeanAbsoluteError()
        self.val_mape = MeanAbsolutePercentageError()
        self.val_mse  = MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    # Training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss  = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    # Validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Compute and log validation metrics
        self.val_mae(y_hat, y)
        self.val_mape(y_hat, y)
        self.val_mse(y_hat, y)
        self.log('val_mae', self.val_mae, on_epoch=True, prog_bar=True)
        self.log('val_mape', self.val_mape, on_epoch=True, prog_bar=False)
        self.log('val_mse', self.val_mse, on_epoch=True, prog_bar=False)

    # Optimizer configuration
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs, eta_min=1e-8)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

# DATA MODULE
class SolarFluxDataModule(pl.LightningDataModule):
    def __init__(self, filename, seq_length, pred_length,
                 batch_size = 64, val_split_days = 730,
                 kp_ap_file='kp_ap_index.txt', sunspot_file='sunspot_number.txt'):
        super().__init__()
        self.filename       = filename       # Path to the CSV file
        self.seq_length     = seq_length     # Input sequence length
        self.pred_length    = pred_length    # Forecast horizon
        self.batch_size     = batch_size     # Batch size for training and validation
        self.val_split_days = val_split_days # Validation split in days
        self.kp_ap_file     = kp_ap_file
        self.sunspot_file   = sunspot_file

    def setup(self, stage=None):
        # Load dataset and split into training and validation sets
        columns = ['Rotation','Obs','Adj','URSI-D','ap']  # Added 'ap' to target variables
        # Add new features to columns
        all_columns = columns + ['Kp','Total']  # Removed 'ap' from exogenous since it's now a target
        ds     = SolarFluxDataset(self.filename, self.seq_length, self.pred_length,
                                 columns=columns,
                                 kp_ap_file=self.kp_ap_file,
                                 sunspot_file=self.sunspot_file)

        cutoff = ds.dates.max() - pd.Timedelta(days=self.val_split_days)
        train_idx = [i for i in range(len(ds))
                     if ds.dates.iloc[i + self.seq_length + self.pred_length - 1] < cutoff]
        val_idx   = [i for i in range(len(ds))
                     if ds.dates.iloc[i + self.seq_length - 1] >= cutoff]
        
        # Compute normalization statistics from training data only (prevent leakage)
        min_i = min(train_idx)
        max_i = max(train_idx) + self.seq_length + self.pred_length - 1
        train_data_indices = list(range(min_i, max_i + 1))
        ds.set_normalization_from_indices(train_data_indices)
        self.train_ds = Subset(ds, train_idx)
        self.val_ds   = Subset(ds, val_idx)
        self.full_ds  = ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.batch_size,
                          shuffle = True, num_workers = 4, pin_memory = True,
                          persistent_workers = True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle = False, num_workers = 4, pin_memory = True,
                          persistent_workers = True)

    def full_dataloader(self):
        return DataLoader(self.full_ds, batch_size=self.batch_size,
                          shuffle = False, num_workers = 4, pin_memory = True,
                          persistent_workers = True)

# CALLBACKS
class LossHistory(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('train_loss')
        if loss is not None:
            # detach and convert to Python float
            self.train_losses.append(loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('val_loss')
        if loss is not None:
            self.val_losses.append(loss.item())

# MAIN SCRIPT
if __name__ == '__main__':
    # Harmonics for cyclical encoding
    HARMONICS = 3

    # Hyperparameters (using default values)
    seq_len        = 365
    pred_len       = 365
    batch_size     = 128
    max_epochs     = 20
    lr             = 1e-4
    filename       = 'solar_flux.txt'
    kp_ap_file     = 'kp_ap_index.txt'
    sunspot_file   = 'sunspot_number.txt'
    columns        = ['Rotation','Obs','Adj','URSI-D','ap']  # Added 'ap' to target variables
    all_columns    = columns + ['Kp','Total']  # Only Kp and Total are exogenous
    n_cyc_feats    = 4 * HARMONICS
    forecast_years = 22
    n_splits       = 5

    # Default hyperparameters (no optimization)
    cnn_params = {'layers': [3, 4, 23, 3], # Default ResNeXt layers
                  'cardinality': 32,       # Default cardinality
                  'width': 4,              # Default width
                  'base_channels': 64}     # Default base channels
    
    ts_params = {'d_model': 512,           # Default model dimension
                 'n_heads': 8,             # Default attention heads
                 'd_ff': 2048,             # Default feed-forward dimension
                 'num_layers': 2,          # Default transformer layers
                 'dropout': 0.25,          # Default dropout
                 'pred_len': pred_len}     # Prediction length

    # Initialize data module with default batch_size
    dm = SolarFluxDataModule(filename, seq_len, pred_len, batch_size,
                             kp_ap_file=kp_ap_file, sunspot_file=sunspot_file)
    dm.setup()

    # Determine correct number of input features
    sample_x, _ = dm.full_ds[0]
    in_vars     = sample_x.shape[0] # Number of input channels for the model
    out_vars    = len(columns)      # Only predict the original solar flux variables

    # Rolling-window cross-validation
    tscv     = TimeSeriesSplit(n_splits=n_splits)
    cv_losses = []
    cv_maes   = [] # Store MAE per fold
    for fold, (train_idx, val_idx) in enumerate(tscv.split(dm.full_ds)):
        print(f"CV Fold {fold+1}/{n_splits}")
        
        # Recompute normalization for each fold using training data only
        min_i = min(train_idx)
        max_i = max(train_idx) + dm.full_ds.seq_length + dm.full_ds.forecast_horizon - 1
        max_i = min(max_i, len(dm.full_ds.raw) - 1)  # safe guard
        train_data_indices = list(range(min_i, max_i + 1))
        dm.full_ds.set_normalization_from_indices(train_data_indices)
        
        dm.train_ds = Subset(dm.full_ds, train_idx)
        dm.val_ds   = Subset(dm.full_ds, val_idx)

        # Re-initialize model & trainer per fold
        model = SolarFluxLitModel(cnn_params, ts_params,
                                  in_vars    = in_vars,
                                  out_vars   = out_vars,
                                  seq_len    = seq_len,
                                  lr         = lr,
                                  max_epochs = max_epochs)
        
        cb_early = EarlyStopping('val_loss', patience=7, mode='min')
        cb_lr    = LearningRateMonitor('epoch')
        trainer  = pl.Trainer(max_epochs        = max_epochs,
                              callbacks         = [cb_early, cb_lr],
                              accelerator       = 'auto',
                              devices           = 1,
                              precision         = '16-mixed',
                              gradient_clip_val = 1.0,
                              log_every_n_steps = 10)
        trainer.fit(model, dm)
        cv_losses.append(trainer.callback_metrics['val_loss'].item())

        # Compute MAE for this fold
        device = model.device
        model.eval()
        mae_numer = 0.0
        mae_denom = 0

        with torch.no_grad():
            for x, y in dm.val_dataloader():
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                diff = torch.abs(y_hat - y)
                mae_numer += diff.sum().item()
                mae_denom += diff.numel()

        val_mae = mae_numer / mae_denom if mae_denom > 0 else float("nan")
        cv_maes.append(val_mae)

    print("Cross-validation val_loss per fold:", cv_losses)
    print("Cross-validation MAE per fold:", cv_maes)

    # Final training on full data before forecasting
    # Use last 10% as validation for monitoring
    val_size = max(1, len(dm.full_ds) // 10)
    train_final_idx = list(range(len(dm.full_ds) - val_size))
    val_final_idx = list(range(len(dm.full_ds) - val_size, len(dm.full_ds)))
    
    # Recompute normalization on training portion
    min_i = min(train_final_idx)
    max_j = max(train_final_idx) + dm.full_ds.seq_length + dm.full_ds.forecast_horizon - 1
    max_i = min(max_j, len(dm.full_ds.raw) - 1)
    train_data_indices = list(range(min_i, max_i + 1))
    dm.full_ds.set_normalization_from_indices(train_data_indices)
    
    dm.train_ds = Subset(dm.full_ds, train_final_idx)
    dm.val_ds   = Subset(dm.full_ds, val_final_idx)
    final_model = SolarFluxLitModel(cnn_params, ts_params,
                                    in_vars    = in_vars,
                                    out_vars   = out_vars,
                                    seq_len    = seq_len,
                                    lr         = lr,
                                    max_epochs = max_epochs)
    loss_history  = LossHistory()
    final_trainer = pl.Trainer(max_epochs        = max_epochs,
                               callbacks         = [loss_history],
                               accelerator       = 'auto',
                               devices           = 1,
                               precision         = '16-mixed',
                               gradient_clip_val = 1.0,
                               log_every_n_steps = 10)
    final_trainer.fit(final_model, dm)

    # Save the trained model
    model_save_path = 'solar_flux_model.pth'
    torch.save({'model_state_dict': final_model.state_dict(), 
                'cnn_params': cnn_params,
                'ts_params': ts_params,
                'in_vars': in_vars,
                'out_vars': out_vars,
                'seq_len': seq_len,
                'lr': lr,
                'hyperparameters': final_model.hparams}, 
                model_save_path)
    print(f"Model saved to {model_save_path}")

    # ===========================
    # RUN BENCHMARKS
    # ===========================
    print("\n" + "="*80)
    print("BENCHMARKING AGAINST BASELINE MODELS")
    print("="*80)
    
    # Run comprehensive benchmarks
    benchmark_results = run_benchmarks(
        dm=dm,
        resnext_model=final_model,
        in_vars=in_vars,
        out_vars=out_vars,
        seq_len=seq_len,
        pred_len=pred_len,
        epochs=max_epochs,
        lr=lr,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Print summary
    print_benchmark_summary(benchmark_results)
    
    # Generate benchmark plots
    plot_benchmark_results(benchmark_results, colors, save_dir='plots')
    
    print("\n" + "="*80)
    print("BENCHMARKING COMPLETE")
    print("="*80)

    # Train conditional Ap model on historical data (Ap | F10.7, phase)
    print("\n" + "="*80)
    print("TRAINING CONDITIONAL AP MODEL")
    print("="*80)
    regression_model, ar_coeffs, sigma, eps_init, df_hist, last_train_idx = train_conditional_ap_model(filename=filename,
                kp_ap_file=kp_ap_file, sunspot_file=sunspot_file, harmonics=HARMONICS, f_col="Adj", p=1)
    print("Fitted conditional Ap regression with AR(1) residuals.")
    print(f"AR(1) coefficients: {ar_coeffs}")
    print(f"Innovation std deviation (sigma): {sigma:.4f}")
    print(f"Last training index (for phase continuity): {last_train_idx}")
    
    # Save the conditional Ap model
    ap_model_save_path = 'conditional_ap_model.pth'
    torch.save({'regression_model_state_dict': regression_model.state_dict(), 
                'ar_coeffs': ar_coeffs,
                'sigma': sigma,
                'eps_init': eps_init,
                'harmonics': HARMONICS,
                'f_col': "Adj",
                'last_train_idx': last_train_idx}, ap_model_save_path)
    print(f"Conditional Ap model saved to {ap_model_save_path}")
    
    # Demonstrate a short self-consistency forecast using the last N days of F10.7
    last_n = 365
    f107_future = df_hist["Adj"].values[-last_n:]
    dates_future = df_hist["date"].values[-last_n:]
    # Provide 40 days of historical context for F81c calculation
    historical_f107 = df_hist["Adj"].values[-(last_n+40):-last_n]
    ap_mean, ap_sample, eps_sim = simulate_ap_forecast(f107_forecast=f107_future, dates_forecast=dates_future,
                                                       harmonics=HARMONICS, regression_model=regression_model,
                                                       ar_coeffs=ar_coeffs, sigma=sigma, eps_init=eps_init,
                                                       historical_f107=historical_f107, 
                                                       last_train_idx=len(df_hist)-last_n-1)
    print(f"\nExample Ap forecast (last 365 days self-consistency check):")
    print(f"  Mean Ap [0:5] = {ap_mean[:5]}")
    print(f"  Sample Ap [0:5] = {ap_sample[:5]}")
    print(f"  Historical Ap [0:5] = {df_hist['ap'].values[-last_n:][:5]}")
    print("="*80 + "\n")