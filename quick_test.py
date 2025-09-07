#!/usr/bin/env python3
"""
Quick test for train1.py - Minimal validation
"""

def quick_test():
    """Quick validation of train1.py"""
    print("🔍 Quick test of train1.py...")
    
    try:
        # Test imports without running main block
        import sys
        import importlib.util
        import torch
        
        # Load module without executing main block
        spec = importlib.util.spec_from_file_location("train1", "train1.py")
        train1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train1)
        
        print("✅ Script imports successfully")
        
        # Test key components
        print(f"✅ Device: {train1.device}")
        print(f"✅ Batch size: {train1.batch_size}")
        print(f"✅ Epochs: {train1.num_epochs}")
        print(f"✅ Classes: {train1.NUM_CLASSES}")
        
        # Test model creation
        model = train1.UNet3D_SpectralAFW(4, train1.NUM_CLASSES)
        print("✅ Model can be created")
        
        # Test utility classes
        progress_tracker = train1.TrainingProgressTracker(5)
        checkpoint_manager = train1.CheckpointManager(train1.checkpoint_dir)
        print("✅ Utility classes work")
        
        # Test loss function
        dummy_weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        criterion = train1.DiceCrossEntropyAFWLoss(
            class_weights=dummy_weights,
            afw_reg=True,
            lambda_afw=0.01,
            alpha=0.5
        )
        print("✅ Loss function works")
        
        print("\n🎉 Quick test passed! Script looks good.")
        print("💡 Note: This test doesn't require the CSV file or data directory.")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_test()
