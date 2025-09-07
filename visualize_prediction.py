# visualize_prediction.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import random

from dataset import get_dataloaders
from models import MODEL_REGISTRY

# Set same config
data_dir = "/content/drive/MyDrive/BraTS_Project/BraTS2021_Training_Data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4
batch_size = 1  # One patient at a time

def load_model(model_name, checkpoint_path):
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(in_channels=4, out_channels=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def visualize_slice(model):
    _, _, test_loader = get_dataloaders(data_dir, batch_size=batch_size)

    for images, masks in test_loader:
        images = images.to(device)
        masks = masks[0].cpu().numpy()  # shape: (D, H, W)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)[0].cpu().numpy()  # shape: (D, H, W)

        flair = images[0, 0].cpu().numpy()  # Flair channel of the 1st batch, shape: (D, H, W)
        slice_idx = flair.shape[0] // 2  # Middle slice

        # Plot
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(flair[slice_idx], cmap='gray')
        plt.title("FLAIR Image")

        plt.subplot(1, 4, 2)
        plt.imshow(masks[slice_idx], cmap='nipy_spectral')
        plt.title("Ground Truth Mask")

        plt.subplot(1, 4, 3)
        plt.imshow(preds[slice_idx], cmap='nipy_spectral')
        plt.title("Predicted Mask")

        plt.subplot(1, 4, 4)
        overlay = np.where(preds[slice_idx] == masks[slice_idx], preds[slice_idx], 10)  # mismatch shown
        plt.imshow(flair[slice_idx], cmap='gray')
        plt.imshow(overlay, cmap='autumn', alpha=0.5)
        plt.title("Overlay")

        plt.tight_layout()
        plt.show()

        break  # Only show one patient

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    model = load_model(args.model, args.checkpoint)
    visualize_slice(model)
