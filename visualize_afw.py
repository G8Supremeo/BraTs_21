# visualize_afw.py

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_afw_slice(filepath, out_channel=0, axis='z', slice_idx=None):
    """
    Visualize a slice from a saved AFW tensor (Adaptive Frequency Weights).
    
    Args:
        filepath (str): Path to .npy AFW file.
        out_channel (int): Output channel index to visualize.
        axis (str): One of 'x', 'y', 'z' to define slice direction.
        slice_idx (int): Optional index. If None, uses middle slice.
    """
    afw = np.load(filepath)  # shape: (1, out_channels, M, M, M)
    afw = afw[0, out_channel]

    assert axis in ['x', 'y', 'z'], "axis must be 'x', 'y', or 'z'"

    if axis == 'x':
        idx = slice_idx or afw.shape[0] // 2
        slice_2d = afw[idx, :, :]
    elif axis == 'y':
        idx = slice_idx or afw.shape[1] // 2
        slice_2d = afw[:, idx, :]
    else:  # axis == 'z'
        idx = slice_idx or afw.shape[2] // 2
        slice_2d = afw[:, :, idx]

    plt.figure(figsize=(6, 5))
    plt.imshow(slice_2d, cmap='viridis')
    plt.title(f"AFW Slice @ {axis.upper()}={idx} (Channel {out_channel})")
    plt.colorbar(label='Weight')
    plt.tight_layout()
    plt.show()

def list_afw_files(directory='.'):
    return sorted([f for f in os.listdir(directory) if f.startswith("afw_epoch") and f.endswith(".npy")])

if __name__ == "__main__":
    files = list_afw_files()
    print("Available AFW files:")
    for f in files:
        print(" -", f)
    
    if files:
        print("\nShowing middle Z-slice of first file...")
        plot_afw_slice(files[0], out_channel=0, axis='z')
