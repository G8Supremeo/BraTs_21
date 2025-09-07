#!/usr/bin/env python3
"""
Find optimal batch size for 80GB GPU
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def get_gpu_memory_info():
    """Get GPU memory information"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free = total_memory - allocated
        
        total_gb = total_memory / (1024**3)
        allocated_gb = allocated / (1024**3)
        free_gb = free / (1024**3)
        
        return {
            'total_gb': total_gb,
            'allocated_gb': allocated_gb,
            'free_gb': free_gb,
            'utilization': (allocated / total_memory) * 100
        }
    return None

def create_fake_batch(batch_size, volume_shape=(240, 240, 152)):
    """Create fake batch for memory testing"""
    images = torch.randn(batch_size, 4, *volume_shape, device='cuda')
    masks = torch.randint(0, 4, (batch_size, *volume_shape), device='cuda')
    return images, masks

def test_batch_size(batch_size, model, volume_shape=(240, 240, 152)):
    """Test if a batch size works without OOM"""
    try:
        # Clear cache
        torch.cuda.empty_cache()
        
        # Create fake batch
        images, masks = create_fake_batch(batch_size, volume_shape)
        
        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, masks)
        
        # Backward pass
        loss.backward()
        
        # Get memory usage
        memory_info = get_gpu_memory_info()
        
        # Clean up
        del images, masks, outputs, loss
        torch.cuda.empty_cache()
        
        return True, memory_info
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return False, None
        else:
            raise e

def find_optimal_batch_size():
    """Find the optimal batch size for 80GB GPU"""
    print("🔍 Finding optimal batch size for 80GB GPU...")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    # Get initial memory info
    initial_memory = get_gpu_memory_info()
    print(f"📊 GPU Memory Info:")
    print(f"   - Total: {initial_memory['total_gb']:.1f} GB")
    print(f"   - Free: {initial_memory['free_gb']:.1f} GB")
    print(f"   - Utilization: {initial_memory['utilization']:.1f}%")
    print()
    
    # Import model
    try:
        from models.spectral_3d_unet_afw_attention import UNet3D_SpectralAFW
        model = UNet3D_SpectralAFW(in_channels=4, out_channels=4).cuda()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test different batch sizes
    batch_sizes_to_test = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    results = []
    
    print("\n🧪 Testing batch sizes...")
    print("-" * 60)
    
    for batch_size in batch_sizes_to_test:
        print(f"Testing batch size: {batch_size:2d}... ", end="", flush=True)
        
        success, memory_info = test_batch_size(batch_size, model)
        
        if success:
            print(f"✅ Success - Memory: {memory_info['allocated_gb']:.1f}GB used")
            results.append({
                'batch_size': batch_size,
                'success': True,
                'memory_used_gb': memory_info['allocated_gb'],
                'memory_utilization': memory_info['utilization']
            })
        else:
            print("❌ OOM")
            results.append({
                'batch_size': batch_size,
                'success': False,
                'memory_used_gb': None,
                'memory_utilization': None
            })
            break  # Stop testing larger batch sizes
    
    # Find optimal batch size
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        # Find batch size that uses ~70-80% of GPU memory
        target_utilization = 75  # 75% utilization
        optimal_batch_size = None
        
        for result in reversed(successful_results):  # Start from largest
            if result['memory_utilization'] <= target_utilization:
                optimal_batch_size = result['batch_size']
                break
        
        if optimal_batch_size is None:
            # If no batch size uses less than 75%, use the largest successful one
            optimal_batch_size = max([r['batch_size'] for r in successful_results])
        
        print("\n" + "=" * 60)
        print("📊 RESULTS SUMMARY:")
        print("=" * 60)
        
        for result in results:
            if result['success']:
                print(f"✅ Batch size {result['batch_size']:2d}: {result['memory_used_gb']:.1f}GB used ({result['memory_utilization']:.1f}% utilization)")
            else:
                print(f"❌ Batch size {result['batch_size']:2d}: Out of Memory")
        
        print(f"\n🎯 RECOMMENDED BATCH SIZE: {optimal_batch_size}")
        print(f"💡 This should use ~{next(r['memory_utilization'] for r in successful_results if r['batch_size'] == optimal_batch_size):.1f}% of your GPU memory")
        
        # Additional recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   - Start with batch_size = {optimal_batch_size}")
        print(f"   - You can try batch_size = {optimal_batch_size + 2} if you want to push limits")
        print(f"   - Monitor memory usage during training")
        print(f"   - Use gradient accumulation if you need larger effective batch sizes")
        
    else:
        print("\n❌ No successful batch sizes found!")
        print("   - Check if your model is too large")
        print("   - Try reducing input volume size")
        print("   - Consider model optimization techniques")

if __name__ == "__main__":
    find_optimal_batch_size()
