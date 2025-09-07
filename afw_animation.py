# afw_animation.py

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def load_afw_slices(directory, out_channel=0, axis='z', slice_idx=None):
    """
    Load and extract 2D slices from multiple AFW .npy files.
    """
    files = sorted([f for f in os.listdir(directory) if f.startswith("afw_epoch") and f.endswith(".npy")],
                   key=lambda x: int(x.split("epoch")[1].split(".")[0]))

    slices = []
    for f in files:
        afw = np.load(os.path.join(directory, f))
        afw = afw[0, out_channel]
        if axis == 'x':
            idx = slice_idx or afw.shape[0] // 2
            slice_2d = afw[idx, :, :]
        elif axis == 'y':
            idx = slice_idx or afw.shape[1] // 2
            slice_2d = afw[:, idx, :]
        else:  # 'z'
            idx = slice_idx or afw.shape[2] // 2
            slice_2d = afw[:, :, idx]
        slices.append((f, slice_2d))
    return slices

def save_afw_gif(directory='.', out_channel=0, axis='z', slice_idx=None, gif_path="afw_evolution.gif"):
    slices = load_afw_slices(directory, out_channel, axis, slice_idx)

    images = []
    for filename, afw_slice in slices:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(afw_slice, cmap='viridis', vmin=0.0, vmax=afw_slice.max())
        ax.set_title(f"{filename}")
        ax.axis('off')
        fig.tight_layout()
        # Save to buffer
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)

    # Save as GIF
    imageio.mimsave(gif_path, images, duration=0.8)
    print(f"✅ Saved AFW evolution GIF to: {gif_path}")

if __name__ == "__main__":
    save_afw_gif(directory='.', out_channel=0, axis='z', gif_path='afw_channel0_z.gif')
