# save_predictions.py

import os
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

from dataset import get_dataloaders
from models import MODEL_REGISTRY

# ------------------------
# Configuration
# ------------------------
data_dir = "/content/drive/MyDrive/BraTS_Project/BraTS2021_Training_Data"
save_dir = "saved_predictions"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4
batch_size = 1  # Predict one patient at a time

# ------------------------
# Load Trained Model
# ------------------------
def load_model(model_name, checkpoint_path):
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(in_channels=4, out_channels=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# ------------------------
# Save a prediction as .nii.gz
# ------------------------
def save_nifti(pred_3d, ref_img, out_path):
    # pred_3d: numpy array of shape (D, H, W)
    # ref_img: any reference image (e.g., flair) for affine
    nib_img = nib.Nifti1Image(pred_3d.astype(np.uint8), affine=ref_img.affine)
    nib.save(nib_img, out_path)

# ------------------------
# Main
# ------------------------
@torch.no_grad()
def run_inference(model_name, checkpoint_path):
    _, _, test_loader = get_dataloaders(data_dir, batch_size=batch_size)

    model = load_model(model_name, checkpoint_path)

    print("📦 Saving predictions to:", save_dir)

    for i, (images, masks, paths) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)[0].cpu().numpy()  # shape: (D, H, W)
        masks = masks[0].numpy()

        # Extract patient ID from flair path (optional)
        patient_id = os.path.basename(paths[0]).split("_")[0] if paths else f"patient_{i:03d}"

        # Load a reference image for affine (e.g., flair)
        ref_nii = nib.load(paths[0]) if paths else None

        # Save prediction
        save_nifti(preds, ref_nii, os.path.join(save_dir, f"{patient_id}_pred.nii.gz"))

        # Optionally: Save ground truth
        save_nifti(masks, ref_nii, os.path.join(save_dir, f"{patient_id}_gt.nii.gz"))

    print("✅ All predictions saved.")

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth file")
    args = parser.parse_args()

    run_inference(args.model, args.checkpoint)
