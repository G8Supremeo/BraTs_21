
import torch
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
import numpy as np

from dataset import BraTSDataset  # Update this with the correct import

# Parameters
EXPECTED_LABELS = {0, 1, 2, 3}
dataset_root = "/content/drive/MyDrive/BraTS_Project/BraTS2021_Training_Data"
batch_size = 1  # safer per-patient loading

# Load your Dataset
val_dataset = BraTSDataset(dataset_root, mode='val')  # or 'train' or 'test' if needed
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("🔍 Verifying remapped labels from BraTSDataset...")

all_labels = set()
voxel_counter = Counter()
sample_counter = 0

for batch in tqdm(val_loader):
    # Access segmentation mask
    seg = batch['seg']  # shape: [B, H, W, D]
    
    for i in range(seg.size(0)):  # iterate each item in batch
        seg_volume = seg[i].cpu().numpy().astype(int)
        unique, counts = np.unique(seg_volume, return_counts=True)

        all_labels.update(unique)
        voxel_counter.update(dict(zip(unique, counts)))
        sample_counter += 1

print(f"\n✅ Total samples checked: {sample_counter}")
print(f"🧠 Unique labels found: {sorted(all_labels)}")
print(f"📊 Voxel counts: {dict(voxel_counter)}")

# Check if labels are as expected
unexpected = all_labels - EXPECTED_LABELS
if unexpected:
    print(f"\n❌ Unexpected labels found: {unexpected}")
else:
    print("\n✅ All labels were correctly remapped by BraTSDataset to [0, 1, 2, 3]")



with open("/content/drive/MyDrive/BraTS_Project/logs/remap_verification.txt", "w") as f:
    f.write(f"Samples: {sample_counter}\n")
    f.write(f"Labels found: {sorted(all_labels)}\n")
    f.write(f"Voxel counts: {dict(voxel_counter)}\n")
    if unexpected:
        f.write(f"❌ Unexpected labels: {unexpected}\n")
    else:
        f.write(f"✅ All labels correctly mapped to {EXPECTED_LABELS}\n")
