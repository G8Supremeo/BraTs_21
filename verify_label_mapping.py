import torch
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader
import csv, os

from dataset import BraTSDataset

NUM_CLASSES = 4  # Expected: 0,1,2,3
data_dir  = "/content/drive/MyDrive/BraTS_Project/BraTS2021_Training_Data"

logs_dir = "/content/drive/MyDrive/BraTS_Project/logs"
# ✅ Make sure the full log directory exists
os.makedirs(logs_dir, exist_ok=True)


batch_size = 1  # Use 1 to avoid memory issues

def verify_entire_dataset(data_dir):
    dataset = BraTSDataset(data_dir, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_voxel_counts = Counter()
    unexpected_labels = set()

    print("🔍 Verifying class labels across entire dataset...")
    for i, (_, mask) in enumerate(tqdm(loader)):
        unique = torch.unique(mask)
        for label in unique:
            val = label.item()
            if val not in range(NUM_CLASSES):
                unexpected_labels.add(val)
            count = torch.sum(mask == val).item()
            all_voxel_counts[val] += count

    print(f"\n✅ Voxel counts per class: {dict(all_voxel_counts)}")
    if unexpected_labels:
        print(f"❌ Found unexpected label(s): {unexpected_labels}")
        raise ValueError("Label remapping failed.")
    else:
        print("✅ All labels correctly mapped to [0, 1, 2, 3].")

    # Save to TXT
    txt_path = os.path.join(logs_dir, "label_verification_result.txt")
    with open(txt_path, "w") as f:
        f.write(f"Voxel counts: {dict(all_voxel_counts)}\n")
        f.write("Verification successful.\n")

    # Save to CSV
    csv_path = os.path.join(logs_dir, "label_verification_result.txt")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class", "Voxel Count"])
        for cls in range(NUM_CLASSES):
            writer.writerow([cls, all_voxel_counts.get(cls, 0)])
    print("📁 Voxel distribution saved to 'label_distribution.csv'.")

# Run before training
verify_entire_dataset(data_dir)