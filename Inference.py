# inference.py

import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
import nibabel as nib

from models import MODEL_REGISTRY
from dataset import BraTSDataset
from torch.utils.data import DataLoader

def load_model(model_name, checkpoint_path, device):
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(in_channels=4, out_channels=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def infer_single_volume(model, volume):
    with torch.no_grad():
        input_tensor = torch.from_numpy(volume).unsqueeze(0).to(torch.float32).to(next(model.parameters()).device)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        return prediction

def save_prediction(pred, affine, save_path):
    pred_npy = pred.squeeze().astype(np.uint8)
    nib_img = nib.Nifti1Image(pred_npy, affine)
    nib.save(nib_img, save_path)

def run_inference(model, dataloader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running batch inference"):
            images = batch["image"].to(device)           # (B, 4, D, H, W)
            affine = batch["affine"][0]                  # affine for NIfTI saving
            name = batch["name"][0]                      # patient/case ID
            outputs = model(images)                      # (B, 4, D, H, W)

            preds = torch.argmax(outputs, dim=1)         # (B, D, H, W)
            save_path = os.path.join(output_dir, f"{name}_pred.nii.gz")
            save_prediction(preds[0].cpu(), affine, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name from MODEL_REGISTRY')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True, help='Path to NIfTI file or test directory')
    parser.add_argument('--output', type=str, default="predictions/", help='Directory to save predictions (for batch)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.checkpoint, device)

    # --- Mode A: Single Volume Inference ---
    if args.input.endswith(".nii") or args.input.endswith(".nii.gz"):
        print("🧠 Performing single-volume inference...")
        nii = nib.load(args.input)
        volume = np.moveaxis(nii.get_fdata(), -1, 0)  # adjust to (C, D, H, W) if needed
        pred = infer_single_volume(model, volume)
        print("✅ Inference complete. Prediction shape:", pred.shape)

    # --- Mode B: Batch Inference via Dataset ---
    elif os.path.isdir(args.input):
        print("🗂️  Performing batch inference from directory...")
        test_dataset = BraTSDataset(root_dir=args.input, mode="test")
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        run_inference(model, test_loader, device, args.output)
        print("✅ Batch inference complete. Predictions saved to:", args.output)

    else:
        raise ValueError("❌ Invalid input. Provide a .nii file or a test directory.")
