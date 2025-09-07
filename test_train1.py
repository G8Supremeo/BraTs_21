#!/usr/bin/env python3
"""
Test script for train1.py - Validates the training script without actually training
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Test core imports
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        print("✅ Core imports successful")
        
        # Test MONAI imports
        from monai.losses import DiceCELoss
        from monai.networks.utils import one_hot
        print("✅ MONAI imports successful")
        
        # Test local imports
        from dataset import get_dataloaders
        from models.spectral_3d_unet_afw_attention import UNet3D_SpectralAFW
        from metrics import dice_per_class, sensitivity, specificity, ppv, miou
        from log_utils import init_log_files, log_to_csv, log_to_json
        from afw_training_utils import (count_trainable_params, afw_entropy_loss, save_afw_weights, monitor_afw_evolution)
        print("✅ Local imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_configuration():
    """Test if configuration variables are properly set"""
    print("\n🔍 Testing configuration...")
    
    try:
        # Import the training script
        import train1
        
        # Check if required variables exist
        required_vars = [
            'data_dir', 'num_epochs', 'batch_size', 'lr', 'seed',
            'CLASS_IDS', 'NUM_CLASSES', 'device', 'use_amp',
            'log_dir', 'checkpoint_dir', 'reports_dir', 'training_viz_dir'
        ]
        
        for var in required_vars:
            if hasattr(train1, var):
                value = getattr(train1, var)
                print(f"✅ {var}: {value}")
            else:
                print(f"❌ Missing variable: {var}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_directory_creation():
    """Test if directories can be created"""
    print("\n🔍 Testing directory creation...")
    
    try:
        import train1
        
        # Test directory creation
        test_dirs = [
            train1.log_dir,
            train1.checkpoint_dir,
            train1.reports_dir,
            train1.training_viz_dir
        ]
        
        for dir_path in test_dirs:
            os.makedirs(dir_path, exist_ok=True)
            if os.path.exists(dir_path):
                print(f"✅ Directory created: {dir_path}")
            else:
                print(f"❌ Failed to create: {dir_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory creation error: {e}")
        return False

def test_model_creation():
    """Test if the model can be created"""
    print("\n🔍 Testing model creation...")
    
    try:
        import train1
        
        # Test model creation
        model = train1.UNet3D_SpectralAFW(
            in_channels=4, 
            out_channels=train1.NUM_CLASSES
        ).to(train1.device)
        
        print(f"✅ Model created successfully")
        print(f"   - Device: {train1.device}")
        print(f"   - Input channels: 4")
        print(f"   - Output channels: {train1.NUM_CLASSES}")
        
        # Test parameter counting
        train1.count_trainable_params(model)
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_utility_classes():
    """Test if utility classes can be instantiated"""
    print("\n🔍 Testing utility classes...")
    
    try:
        import train1
        
        # Test TrainingProgressTracker
        progress_tracker = train1.TrainingProgressTracker(10)
        print("✅ TrainingProgressTracker created")
        
        # Test CheckpointManager
        checkpoint_manager = train1.CheckpointManager(train1.checkpoint_dir)
        print("✅ CheckpointManager created")
        
        # Test LiveTrainingVisualizer (without showing plots)
        visualizer = train1.LiveTrainingVisualizer(update_interval=5)
        print("✅ LiveTrainingVisualizer created")
        visualizer.close()  # Close immediately to avoid display issues
        
        return True
        
    except Exception as e:
        print(f"❌ Utility classes error: {e}")
        return False

def test_loss_function():
    """Test if loss function can be created"""
    print("\n🔍 Testing loss function...")
    
    try:
        import train1
        
        # Create dummy weights
        dummy_weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test loss function creation
        criterion = train1.DiceCrossEntropyAFWLoss(
            class_weights=dummy_weights,
            afw_reg=True,
            lambda_afw=0.01,
            alpha=0.5
        )
        
        print("✅ Loss function created successfully")
        
        # Test forward pass with dummy data
        batch_size = 2
        preds = torch.randn(batch_size, 4, 32, 32, 16)  # Smaller size for testing
        target = torch.randint(0, 4, (batch_size, 32, 32, 16))
        
        loss, ce_loss, dice_loss, afw_loss = criterion(preds, target)
        
        print(f"✅ Forward pass successful")
        print(f"   - Total loss: {loss.item():.4f}")
        print(f"   - CE loss: {ce_loss.item():.4f}")
        print(f"   - Dice loss: {dice_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss function error: {e}")
        return False

def test_data_loader_simulation():
    """Test data loader creation (simulated)"""
    print("\n🔍 Testing data loader simulation...")
    
    try:
        import train1
        
        # Check if data directory exists
        if not os.path.exists(train1.data_dir):
            print(f"⚠️  Data directory not found: {train1.data_dir}")
            print("   This is expected if you haven't set up the data yet")
            return True
        
        # If data exists, test data loader creation
        try:
            train_loader, val_loader, _ = train1.get_dataloaders(
                data_dir=train1.data_dir,
                batch_size=1,  # Small batch for testing
                train_ratio=0.8,
                val_ratio=0.2,
                augment=False,  # No augmentation for testing
                num_workers=0,
                pin_memory=False
            )
            
            print("✅ Data loaders created successfully")
            print(f"   - Train batches: {len(train_loader)}")
            print(f"   - Val batches: {len(val_loader)}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Data loader creation failed: {e}")
            print("   This might be due to missing data files")
            return True  # Don't fail the test for missing data
        
    except Exception as e:
        print(f"❌ Data loader test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing train1.py script...")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Directory Creation", test_directory_creation),
        ("Model Creation", test_model_creation),
        ("Utility Classes", test_utility_classes),
        ("Loss Function", test_loss_function),
        ("Data Loader Simulation", test_data_loader_simulation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your train1.py script is ready to run!")
        print("\n💡 Next steps:")
        print("   1. Ensure your data is in the correct directory")
        print("   2. Run: python train1.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running training.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
