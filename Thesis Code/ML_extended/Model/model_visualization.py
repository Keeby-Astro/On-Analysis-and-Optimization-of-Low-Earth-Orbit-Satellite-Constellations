# MODEL VISUALIZATION UTILITIES
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

# Color palette for plots
colors = ['#1965B0', '#E8601C', '#4EB265', '#72190E', '#882E72',
          '#437DBF', '#F1932D', '#90C987', '#A5170E', '#994F88',
          '#6195CF', '#F6C141', '#CAE0AB', '#DC050C', '#AA6F9E',
          '#7BAFDE', '#F7F056', '#8B8B8B', '#896D67', '#BA8DB4']

# FEATURE VISUALIZATION FUNCTION
def visualize_resnext_features(model, dataloader, output_dir=".", max_samples=None):
    print("=== FEATURE VISUALIZATION: ResNeXt Abstract Features ===")
    print("Extracting and visualizing ResNeXt abstract features...")
    
    def extract_resnext_features(model, dataloader, device, num_samples=5):
        model.eval()
        model.to(device)
        features_raw_list = []
        features_reduced_list = []
        input_data_list = []
        
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(dataloader):
                if i >= num_samples:
                    break
                x_batch = x_batch.to(device)
                
                # Extract features at different stages
                f_raw = model.model.cnn(x_batch)       # Raw ResNeXt features
                f_reduced = model.model.reducer(f_raw) # Reduced features
                features_raw_list.append(f_raw[0].cpu().numpy())
                features_reduced_list.append(f_reduced[0].cpu().numpy())
                input_data_list.append(x_batch[0].cpu().numpy())
        
        return features_raw_list, features_reduced_list, input_data_list
    
    # Extract features from validation set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use maximum number of batches available or specified limit
    max_batches = len(dataloader)
    if max_samples is not None:
        max_batches = min(max_batches, max_samples)
        
    print(f"Total validation batches available: {len(dataloader)}")
    print(f"Extracting features from {max_batches} batches for comprehensive analysis...")
    
    features_raw, features_reduced, input_data = extract_resnext_features(
        model, dataloader, device, num_samples=max_batches)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    # Visualize ALL the features (no limits)
    print(f"\nGenerating comprehensive visualizations for ALL {len(features_raw)} samples...")
    print("Note: This will generate many plots - each sample gets multiple visualization types.")
    
    for sample_idx in range(len(features_raw)):
        print(f"\n=== VISUALIZING SAMPLE {sample_idx + 1} OF {len(features_raw)} ===")
        
        # Get the features for this sample
        f_raw = features_raw[sample_idx]         # Shape: (channels, time)
        f_reduced = features_reduced[sample_idx] # Shape: (d_model, time)
        x_input = input_data[sample_idx]         # Shape: (input_features, time)
        
        print(f"Raw ResNeXt features shape: {f_raw.shape}")
        print(f"Reduced features shape: {f_reduced.shape}")
        print(f"Input data shape: {x_input.shape}")
        
        # Visualize raw ResNeXt features (all channels)
        n_raw_channels = f_raw.shape[0]
        time_steps = f_raw.shape[1]
        
        # Plot all raw ResNeXt features (all channels) on a single plot
        fig, ax = plt.subplots(figsize=(15, 6))
        
        if n_raw_channels > 0:
            # Create line segments for all channels
            time_indices = np.arange(time_steps)
            lines = []
            for ch_idx in range(n_raw_channels):
                line_points = np.column_stack([time_indices, f_raw[ch_idx]])
                lines.append(line_points)
            
            # Create colors - cycle through the predefined color palette
            line_colors = [colors[ch_idx % len(colors)] for ch_idx in range(n_raw_channels)]
            
            # Create LineCollection for efficient rendering
            lc = LineCollection(lines, colors=line_colors, alpha=0.8, linewidths=1.0)
            ax.add_collection(lc)
            
            # Set proper axis limits
            ax.set_xlim(time_indices.min(), time_indices.max())
            ax.set_ylim(f_raw.min(), f_raw.max())
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Activation')
        ax.set_title(f'Sample {sample_idx + 1}: All Raw ResNeXt Features (n={n_raw_channels})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, f'resnext_raw_features_sample{sample_idx+1}_all.png')
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close(fig)  # Explicitly close figure to free memory
        
        # Visualize reduced features (input to iTransformer)
        n_reduced_channels = f_reduced.shape[0]
        
        # Plot all reduced features (abstract representations) on a single plot
        fig, ax = plt.subplots(figsize=(15, 6))
        
        if n_reduced_channels > 0:
            # Create line segments for all channels
            time_indices = np.arange(f_reduced.shape[1])
            lines = []
            for ch_idx in range(n_reduced_channels):
                line_points = np.column_stack([time_indices, f_reduced[ch_idx]])
                lines.append(line_points)
            
            # Create colors - cycle through the predefined color palette
            line_colors = [colors[ch_idx % len(colors)] for ch_idx in range(n_reduced_channels)]
            
            # Create LineCollection for efficient rendering
            lc = LineCollection(lines, colors=line_colors, alpha=0.8, linewidths=1.0)
            ax.add_collection(lc)
            
            # Set proper axis limits
            ax.set_xlim(time_indices.min(), time_indices.max())
            ax.set_ylim(f_reduced.min(), f_reduced.max())
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Activation')
        ax.set_title(f'Sample {sample_idx + 1}: All Abstract Features (Input to iTransformer, n={n_reduced_channels})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, f'resnext_abstract_features_sample{sample_idx+1}_all.png')
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close(fig)  # Explicitly close figure to free memory
            
        # 3. Heatmap visualization of all features
        # Raw features heatmap
        fig, ax = plt.subplots(figsize=(15, 8))
        im = ax.imshow(f_raw, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label='Feature Activation')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Feature Channel')
        ax.set_title(f'Sample {sample_idx + 1}: Raw ResNeXt Features Heatmap ({f_raw.shape[0]} channels)')
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, f'resnext_raw_heatmap_sample{sample_idx+1}.png')
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close(fig)  # Explicitly close figure to free memory
        
        # Abstract features heatmap
        fig, ax = plt.subplots(figsize=(15, 6))
        im = ax.imshow(f_reduced, aspect='auto', cmap='plasma', interpolation='nearest')
        plt.colorbar(im, label='Feature Activation')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Abstract Feature Channel')
        ax.set_title(f'Sample {sample_idx + 1}: Abstract Features Heatmap ({f_reduced.shape[0]} channels)')
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, f'resnext_abstract_heatmap_sample{sample_idx+1}.png')
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close(fig)  # Explicitly close figure to free memory

        # 3D Surface plot for Raw ResNeXt Features
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for channels and time steps
        channels_raw = np.arange(f_raw.shape[0])
        time_steps_raw = np.arange(f_raw.shape[1])
        C_raw, T_raw = np.meshgrid(channels_raw, time_steps_raw, indexing='ij')
        
        # Apply Gaussian smoothing for nicer surface visualization
        f_raw_smooth = gaussian_filter(f_raw, sigma=1.0)
        
        # Create 3D surface plot with enhanced visual settings
        surf = ax.plot_surface(C_raw, T_raw, f_raw_smooth, cmap='viridis', alpha=0.85, 
                              linewidth=0, antialiased=True, shade=True, 
                              rcount=min(50, f_raw.shape[0]), ccount=min(50, f_raw.shape[1]))
        
        ax.set_xlabel('Feature Channel', fontsize=12)
        ax.set_ylabel('Time Step', fontsize=12)
        ax.set_zlabel('Activation', fontsize=12)
        ax.set_title(f'Sample {sample_idx + 1}: Landscape of Raw ResNeXt Features\n({f_raw.shape[0]} channels × {f_raw.shape[1]} time steps)', 
                    fontsize=14, pad=20)
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1, label='Feature Activation')
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, f'resnext_raw_3d_landscape_sample{sample_idx+1}.png')
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close(fig)  # Explicitly close figure to free memory
        
        # 3D Surface plot for Abstract Features (Reduced)
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for abstract features
        channels_reduced = np.arange(f_reduced.shape[0])
        time_steps_reduced = np.arange(f_reduced.shape[1])
        C_reduced, T_reduced = np.meshgrid(channels_reduced, time_steps_reduced, indexing='ij')
        
        # Apply Gaussian smoothing for nicer surface visualization
        f_reduced_smooth = gaussian_filter(f_reduced, sigma=1.0)
        
        # Create 3D surface plot with enhanced visual settings
        surf = ax.plot_surface(C_reduced, T_reduced, f_reduced_smooth, cmap='plasma', alpha=0.85,
                              linewidth=0, antialiased=True, shade=True,
                              rcount=min(50, f_reduced.shape[0]), ccount=min(50, f_reduced.shape[1]))
        
        ax.set_xlabel('Abstract Feature Channel', fontsize=12)
        ax.set_ylabel('Time Step', fontsize=12)
        ax.set_zlabel('Activation', fontsize=12)
        ax.set_title(f'Sample {sample_idx + 1}: Landscape of Abstract Features\n({f_reduced.shape[0]} channels × {f_reduced.shape[1]} time steps)', 
                    fontsize=14, pad=20)
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1, label='Feature Activation')
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, f'resnext_abstract_3d_landscape_sample{sample_idx+1}.png')
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close(fig)  # Explicitly close figure to free memory
    
    print("Feature visualization complete!")
    print(f"Generated {len(generated_files)} visualization files in {output_dir}")
    return generated_files

def list_conv_modules(model, include_1d=True):
    conv_modules = OrderedDict()
    
    def _traverse(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if include_1d and isinstance(child, nn.Conv1d):
                conv_modules[full_name] = child
            
            # Recursively traverse child modules
            _traverse(child, full_name)
    
    _traverse(model)
    return conv_modules

def _normalize_img(x):
    x_flat = x.flatten()
    x_min = x_flat.min()
    x_max = x_flat.max()
    
    if x_max == x_min:
        return np.zeros_like(x)
    
    return (x - x_min) / (x_max - x_min)

def visualize_all_kernels(model, root_name, out_dir):
    conv_modules = list_conv_modules(model, include_1d=True)
    output_files = []
    os.makedirs(out_dir, exist_ok=True)

    for layer_name, conv_layer in conv_modules.items():
        print(f"Visualizing kernels (single-figure) for layer: {layer_name}")

        # Get weights: (out_channels, in_channels_per_group, kernel_size)
        weights = conv_layer.weight.detach().cpu().numpy()
        O, Igroup, K = weights.shape
        kernels = weights.reshape(-1, K)  # (n_kernels, K)
        n_kernels = kernels.shape[0]
        print(f"  Weight shape: ({O}, {Igroup}, {K}) -> {n_kernels} kernels")

        # Robust scaling for visualization: avoid division by zero
        max_abs = np.max(np.abs(kernels))
        scale = max_abs if max_abs > 0 else 1.0
        kernels_plot = kernels / scale

        # Downsample to at most 512 kernels for plotting (evenly sampled)
        max_display = 512
        if n_kernels > max_display:
            print(f"  Downsampling kernels for display: {n_kernels} -> {max_display}")
            indices = np.linspace(0, n_kernels - 1, max_display, dtype=int)
            kernels_plot_disp = kernels_plot[indices]
            display_count = max_display
        else:
            kernels_plot_disp = kernels_plot
            display_count = n_kernels

        x = np.arange(K)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Use LineCollection for efficient plotting of many lines
        if display_count > 0:
            # Create line segments for all kernels to be displayed
            lines = []
            for k in kernels_plot_disp:
                # Each line is a sequence of (x, y) points
                line_points = np.column_stack([x, k])
                lines.append(line_points)

            # Create colors for displayed lines - cycle through colormap
            cmap = plt.get_cmap('tab20')
            colors_kernel = [cmap(i % 20) for i in range(display_count)]

            # Create LineCollection with displayed kernels
            lc = LineCollection(lines, colors=colors_kernel, alpha=0.45, linewidths=0.9)
            ax.add_collection(lc)

            # Set proper axis limits for the LineCollection
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(kernels_plot_disp.min(), kernels_plot_disp.max())

        ax.set_xlabel('Kernel Index')
        ax.set_ylabel('Weight (scaled)')
        ax.set_title(f'{root_name} - {layer_name}  |  Kernels: ({O}, {Igroup}, {K})  |  total={n_kernels}')
        ax.grid(True, alpha=0.3)
        # Avoid calling legend with no labeled handles
        plt.tight_layout()

        output_file = os.path.join(out_dir, f"{layer_name.replace('.', '_')}_kernels_singlefig.png")
        output_files.append(output_file)
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        plt.close(fig)  # Explicitly close figure to free memory

    return output_files

def visualize_activations(model, batch_x, root_name, out_dir, forward_fn=None):
    conv_modules = list_conv_modules(model, include_1d=True)
    output_files = []
    activation_dict = {}
    handles = []
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Define hook function to capture activations
    def capture_activation(name):
        def hook(module, input, output):
            activation_dict[name] = output.detach().cpu()
        return hook
    
    # Register hooks on all Conv1d layers
    for layer_name, conv_layer in conv_modules.items():
        handle = conv_layer.register_forward_hook(capture_activation(layer_name))
        handles.append(handle)
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        if forward_fn is not None:
            _ = forward_fn(model, batch_x)
        else:
            _ = model(batch_x)
    
    # Process captured activations
    for layer_name in conv_modules.keys():
        if layer_name not in activation_dict:
            print(f"Warning: No activation captured for layer {layer_name}")
            continue
        print(f"Visualizing activations for layer: {layer_name}")
        
        # Get activations: shape (B, C, T)
        activations = activation_dict[layer_name]
        B, C, T = activations.shape
        print(f"  Activation shape: ({B}, {C}, {T})")
        
        # Compute batch mean: (C, T)
        M = activations.mean(dim=0).numpy()  # Shape: (C, T)
        
        # Heatmap view - optimized for large channel counts
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # For very large channel counts, downsample for display but keep all data
        if C > 2048:  # If more than 2048 channels, downsample for display
            print(f"  Large channel count ({C}), downsampling heatmap for display...")
            downsample_factor = max(1, C // 1024)  # Target ~1024 channels for display
            M_display = M[::downsample_factor, :]
            im = ax.imshow(M_display, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_ylabel(f'Channel (every {downsample_factor}th shown)')
        else:
            im = ax.imshow(M, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_ylabel('Channel')
            
        plt.colorbar(im, label='Activation')
        ax.set_xlabel('Time')
        ax.set_title(f'{root_name} - {layer_name}\nHeatmap: ({C} channels × {T} time steps)')
        
        # Save heatmap
        heatmap_file = os.path.join(out_dir, f"{layer_name.replace('.', '_')}_heatmap.png")
        output_files.append(heatmap_file)
        plt.savefig(heatmap_file, dpi=600, bbox_inches='tight')
        plt.close(fig)  # Explicitly close figure to free memory
        
        # Channel visualizations
        print(f"  Using LineCollection approach for {C} channels...")
        _visualize_channels_single_plot(M, C, T, root_name, layer_name, out_dir, output_files)

    # Remove all hooks
    for handle in handles:
        handle.remove()
    
    return output_files

def _visualize_channels_single_plot(M, C, T, root_name, layer_name, out_dir, output_files):
    time_indices = np.arange(T)
    
    # Create line segments for all channels
    lines = []
    for ch_idx in range(C):
        line_points = np.column_stack([time_indices, M[ch_idx, :]])
        lines.append(line_points)
    
    # Create colors - use a colormap that cycles through many colors
    cmap = plt.get_cmap('tab20')
    colors_channel = [cmap(i % 20) for i in range(C)]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create LineCollection
    lc = LineCollection(lines, colors=colors_channel, alpha=0.6, linewidths=0.8)
    ax.add_collection(lc)
    
    # Set axis limits
    ax.set_xlim(time_indices.min(), time_indices.max())
    ax.set_ylim(M.min(), M.max())
    ax.set_xlabel('Time')
    ax.set_ylabel('Activation')
    ax.set_title(f'{root_name} - {layer_name}\nAll {C} Channels')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the plot
    all_channels_file = os.path.join(out_dir, f"{layer_name.replace('.', '_')}_all_channels.png")
    output_files.append(all_channels_file)
    plt.savefig(all_channels_file, dpi=600, bbox_inches='tight')
    plt.close(fig)