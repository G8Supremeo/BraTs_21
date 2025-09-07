#!/usr/bin/env python3
"""
Create a test CSV file for voxel distribution
"""

import os
import pandas as pd
import numpy as np

def create_test_csv():
    """Create a test CSV file with voxel distribution data"""
    
    # Create reports directory if it doesn't exist
    reports_dir = "/content/drive/MyDrive/BraTS_Project/reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create test data (simulating realistic class distribution)
    test_data = {
        'class_0': [50000000, 45000000, 48000000, 52000000, 46000000],  # Background (most common)
        'class_1': [500000, 450000, 480000, 520000, 460000],           # Necrotic (rare)
        'class_2': [2000000, 1800000, 2200000, 1900000, 2100000],      # Edema (moderate)
        'class_3': [800000, 750000, 850000, 900000, 820000]            # Enhancing (moderate)
    }
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    # Save to CSV
    csv_path = os.path.join(reports_dir, "brats_mapped_voxel_distribution.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Created test CSV file: {csv_path}")
    print("📊 Test data summary:")
    print(f"   - Background (class 0): ~{df['class_0'].sum():,} voxels")
    print(f"   - Necrotic (class 1): ~{df['class_1'].sum():,} voxels")
    print(f"   - Edema (class 2): ~{df['class_2'].sum():,} voxels")
    print(f"   - Enhancing (class 3): ~{df['class_3'].sum():,} voxels")
    
    return csv_path

if __name__ == "__main__":
    create_test_csv()
