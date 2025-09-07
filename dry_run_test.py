#!/usr/bin/env python3
"""
Dry run test - Test training loop with fake data
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def create_fake_dataloader(batch_size=2, num_samples=10):
    """Create fake data for testing"""
    # Create fake images (4 channels, smaller size for testing)
    images = torch.randn(num_samples, 4, 64, 64, 32)
    # Create fake masks (4 classes)
    masks = torch.randint(0, 4, (num_samples, 64, 64, 32))
    
    dataset = TensorDataset(images, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def dry_run_test():
    """Test training loop with fake data"""
    print("🔍 Dry run test with fake data...")
    
    try:
        import train1
        
        # Create fake data loaders
        print("Creating fake data loaders...")
        train_loader = create_fake_dataloader(batch_size=2, num_samples=8)
        val_loader = create_fake_dataloader(batch_size=2, num_samples=4)
        
        print(f"✅ Fake train loader: {len(train_loader)} batches")
        print(f"✅ Fake val loader: {len(val_loader)} batches")
        
        # Test model creation
        print("Creating model...")
        model = train1.UNet3D_SpectralAFW(4, 4).to(train1.device)
        print("✅ Model created")
        
        # Test loss function
        print("Testing loss function...")
        criterion = train1.DiceCrossEntropyAFWLoss(
            class_weights=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            afw_reg=True,
            lambda_afw=0.01,
            alpha=0.5
        )
        print("✅ Loss function created")
        
        # Test one training step
        print("Testing one training step...")
        model.train()
        for images, masks in train_loader:
            images = images.to(train1.device)
            masks = masks.to(train1.device)
            
            # Forward pass
            outputs = model(images)
            loss, ce_loss, dice_loss, afw_loss = criterion(outputs, masks.unsqueeze(1))
            
            print(f"✅ Forward pass successful")
            print(f"   - Output shape: {outputs.shape}")
            print(f"   - Loss: {loss.item():.4f}")
            break  # Only test one batch
        
        print("\n🎉 Dry run test passed! Training loop should work.")
        return True
        
    except Exception as e:
        print(f"❌ Dry run test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    dry_run_test()
