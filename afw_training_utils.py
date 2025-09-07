import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import csv

from torch.nn.functional import softmax

# ---------------------------------------------------------
# Utility 1: Count total number of trainable parameters
# ---------------------------------------------------------
def count_trainable_params(model):
    """
    Prints the total number of trainable parameters in the model.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total Trainable Parameters: {total:,}")


# ---------------------------------------------------------
# Utility 2: Log class distribution from a dataset loader
# ---------------------------------------------------------
def log_class_distribution(dataloader, num_classes=4):
    """
    Computes and prints the voxel distribution per class from the dataset.

    Args:
        dataloader: DataLoader instance yielding (images, masks)
        num_classes: Number of segmentation classes
    """
    counts = torch.zeros(num_classes)

    for _, masks in dataloader:
        for c in range(num_classes):
            counts[c] += (masks == c).sum()

    total_voxels = counts.sum().item()
    print("\n[INFO] Class Voxel Distribution:")
    for i, count in enumerate(counts):
        percentage = 100 * count.item() / total_voxels
        print(f"  Class {i}: {int(count.item())} voxels ({percentage:.2f}%)")

    return counts


# ---------------------------------------------------------
# Utility 3: Compute class weights based on voxel frequencies
# ---------------------------------------------------------
def compute_class_weights(dataloader, num_classes=4, smoothing=1.02):
    """
    Computes inverse log class weights for use in CrossEntropyLoss.

    Args:
        dataloader: DataLoader instance
        num_classes: Total number of classes in segmentation task
        smoothing: Small constant to avoid division by zero

    Returns:
        Tensor of class weights: shape (num_classes,)
    """
    counts = log_class_distribution(dataloader, num_classes)
    class_freq = counts / counts.sum()  # Normalize to get relative frequency
    weights = 1.0 / (torch.log(class_freq + smoothing))  # Inverse log-frequency
    norm_weights = weights / weights.sum()  # Normalize to sum to 1
    print(f"[INFO] Computed Class Weights: {norm_weights.tolist()}")
    return norm_weights


# ---------------------------------------------------------
# Utility 4: AFW Entropy Regularizer
# ---------------------------------------------------------
def afw_entropy_loss(afw_tensor):
    """
    Computes the negative entropy of softmax-normalized AFW weights.
    Improved version that preserves spatial frequency structure.

    Args:
        afw_tensor: (1, C, D, H, W) tensor of learnable frequency weights

    Returns:
        Scalar tensor representing the entropy regularization loss
    """
    import torch.nn.functional as F
    
    # Flatten the spatial dimensions for softmax normalization
    # Shape: (1, C, D, H, W) -> (1, C, D*H*W)
    original_shape = afw_tensor.shape
    afw_flat = afw_tensor.view(original_shape[0], original_shape[1], -1)
    
    # Apply softmax across the flattened spatial dimensions
    afw_norm_flat = F.softmax(afw_flat, dim=-1)
    
    # Reshape back to original shape
    afw_norm = afw_norm_flat.view(original_shape)
    
    # Compute entropy while preserving spatial structure
    # Sum across frequency dimensions, then mean across channels
    entropy = -torch.sum(afw_norm * torch.log(afw_norm + 1e-8), dim=(-3, -2, -1))
    
    # Return mean entropy across channels
    return torch.mean(entropy)


# ---------------------------------------------------------
# Utility 5: Save AFW weights periodically
# ---------------------------------------------------------
def save_afw_weights(epoch, afw_tensor, save_dir="afw_logs"):
    """
    Saves the current AFW weights to a NumPy file for visualization.

    Args:
        epoch: Current training epoch
        afw_tensor: AFW tensor of shape (1, C, D, H, W)
        save_dir: Directory to store NumPy snapshots
    """
    os.makedirs(save_dir, exist_ok=True)
    weights_np = afw_tensor.detach().cpu().squeeze().numpy()
    filename = os.path.join(save_dir, f"afw_epoch{epoch:03d}.npy")
    np.save(filename, weights_np)
    print(f"[AFW] Saved AFW weights to: {filename}")


# ---------------------------------------------------------
# Utility 6: Static Plot of AFW weights
# ---------------------------------------------------------
def plot_afw_static(weights_np):
    """
    Plots the AFW tensor (NumPy format) as multiple 2D slices.

    Args:
        weights_np: NumPy array of shape (C, D, H, W)
    """
    num_channels = weights_np.shape[0]

    for i in range(num_channels):
        plt.figure(figsize=(6, 4))
        plt.imshow(weights_np[i, :, :, :].mean(axis=0), cmap='viridis')  # Average over depth
        plt.title(f"AFW Slice - Channel {i}")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------
# Utility 7: Animate AFW evolution over time (GIF)
# ---------------------------------------------------------
def generate_afw_animation(npy_dir="afw_logs", save_gif="afw_evolution.gif"):
    """
    Creates a GIF showing the evolution of AFW weights over epochs.

    Args:
        npy_dir: Directory containing .npy snapshots of AFW weights
        save_gif: Output path for the generated GIF
    """
    files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])
    frames = []

    for fname in files:
        afw_np = np.load(os.path.join(npy_dir, fname))  # Load weights
        mean_proj = afw_np.mean(axis=(0, 1))  # Average over C and D dims

        # Create the image for this epoch
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(mean_proj, cmap='plasma')
        ax.set_title(fname.replace(".npy", ""))
        plt.axis("off")

        # Save current frame to buffer
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close()

    # Write all frames to a GIF
    imageio.mimsave(save_gif, frames, duration=0.8)
    print(f"[AFW] Animation saved as: {save_gif}")


def monitor_afw_evolution(afw_tensor, epoch, save_dir="afw_logs"):
    """
    Monitor AFW weight evolution during training.
    
    Args:
        afw_tensor: Current AFW weights tensor
        epoch: Current training epoch
        save_dir: Directory to save monitoring data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # Get statistics
        afw_min = afw_tensor.min().item()
        afw_max = afw_tensor.max().item()
        afw_mean = afw_tensor.mean().item()
        afw_std = afw_tensor.std().item()
        
        # Save statistics to CSV
        stats_file = os.path.join(save_dir, "afw_evolution_stats.csv")
        stats_exist = os.path.exists(stats_file)
        
        with open(stats_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not stats_exist:
                writer.writerow(["Epoch", "Min", "Max", "Mean", "Std"])
            writer.writerow([epoch, afw_min, afw_max, afw_mean, afw_std])
        
        # Print current statistics
        print(f"[AFW Monitor] Epoch {epoch}: Min={afw_min:.4f}, Max={afw_max:.4f}, Mean={afw_mean:.4f}, Std={afw_std:.4f}")
        
        return {
            'min': afw_min,
            'max': afw_max,
            'mean': afw_mean,
            'std': afw_std
        }


"""
2. carry out tuning of class_weights based on your actual dataset (provide reasons for the choice of the weights) ---- for class imbalance
3. execute weight decay + AFW entropy regularizer
4. Log Voxel Class Stats
5. Sanity Print for Parameters
6. Monitor AFW During Training by logging and visualizing the AFW weights at intervals (eg, every 5 epochs).
7.  Save AFW as Numpy Files for Plotting
8. create a helper function for plotting the AFW later
9. Add AFW Animation Support


| Component                   | What it does                               | Where to place                      |
| --------------------------- | ------------------------------------------ | ----------------------------------- |
| `compute_class_weights()`   | Balance voxel classes                      | After DataLoader in `train_model()` |
| `weight_decay` in optimizer | Prevent overfitting of AFW or final layers | When creating the optimizer         |
| `afw_entropy_loss()`        | Regularize AFW attention weights           | Inside loss computation             |


"""