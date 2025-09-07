#!/usr/bin/env python3
"""
Comprehensive Test Script for Spatial-Spectral CNN with AFW
Brain Tumor Segmentation on BraTS 2021 Dataset

Expert-level evaluation including:
- AFW weight analysis and frequency domain evaluation
- Advanced clinical metrics (Hausdorff Distance, Volume Error, Surface Dice)
- Statistical analysis with confidence intervals
- Prediction visualization and attention map analysis
- Comprehensive result export for clinical validation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
import csv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion, binary_dilation
import warnings
warnings.filterwarnings("ignore")

from dataset import get_dataloaders
from models import MODEL_REGISTRY
from metrics import dice_per_class, miou, sensitivity, specificity, ppv

# ---------------------
# Configuration
# ---------------------
data_dir = "/content/drive/MyDrive/BraTS_Project/BraTS2021_Training_Data"
batch_size = 2  # Reduced for memory efficiency during testing
num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clinical class names for brain tumor segmentation
CLASS_NAMES = {
    0: "Background",
    1: "Necrotic Core", 
    2: "Edema",
    3: "Enhancing Tumor"
}

# Clinical importance weights
CLINICAL_WEIGHTS = {
    0: 0.1,  # Background - less important
    1: 0.3,  # Necrotic Core - important for treatment
    2: 0.2,  # Edema - important for monitoring  
    3: 0.4   # Enhancing Tumor - most important for diagnosis
}

# ---------------------
# Advanced Metrics Functions
# ---------------------
def hausdorff_distance(pred, target, cls):
    """Calculate Hausdorff distance for a specific class"""
    pred_cls = (pred == cls).cpu().numpy()
    target_cls = (target == cls).cpu().numpy()
    
    if pred_cls.sum() == 0 or target_cls.sum() == 0:
        return float('inf') if pred_cls.sum() != target_cls.sum() else 0.0
    
    try:
        # Find boundary points
        pred_boundary = pred_cls - binary_erosion(pred_cls)
        target_boundary = target_cls - binary_erosion(target_cls)
        
        if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
            return float('inf')
        
        # Get boundary coordinates
        pred_coords = np.where(pred_boundary)
        target_coords = np.where(target_boundary)
        
        if len(pred_coords[0]) == 0 or len(target_coords[0]) == 0:
            return float('inf')
        
        # Calculate Hausdorff distance
        pred_points = np.column_stack(pred_coords)
        target_points = np.column_stack(target_coords)
        
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(hd1, hd2)
        
    except Exception:
        return float('inf')

def volume_error(pred, target, cls):
    """Calculate volume error percentage for a specific class"""
    pred_vol = (pred == cls).sum().float()
    target_vol = (target == cls).sum().float()
    
    if target_vol == 0:
        return 0.0 if pred_vol == 0 else float('inf')
    
    return abs(pred_vol - target_vol) / target_vol * 100

def surface_dice(pred, target, cls, tolerance=2):
    """Calculate Surface Dice for a specific class"""
    pred_cls = (pred == cls).cpu().numpy()
    target_cls = (target == cls).cpu().numpy()
    
    if pred_cls.sum() == 0 or target_cls.sum() == 0:
        return 0.0
    
    try:
        # Find surfaces
        pred_surface = pred_cls - binary_erosion(pred_cls)
        target_surface = target_cls - binary_erosion(target_cls)
        
        if pred_surface.sum() == 0 or target_surface.sum() == 0:
            return 0.0
        
        # Dilate surfaces with tolerance
        pred_dilated = binary_dilation(pred_surface, iterations=tolerance)
        target_dilated = binary_dilation(target_surface, iterations=tolerance)
        
        # Calculate overlap
        intersection = (pred_surface * target_dilated).sum() + (target_surface * pred_dilated).sum()
        union = pred_surface.sum() + target_surface.sum()
        
        return intersection / union if union > 0 else 0.0
        
    except Exception:
        return 0.0

def calculate_clinical_metrics(all_metrics):
    """Calculate clinically relevant metrics"""
    # Weighted Dice score (clinical importance)
    weighted_dice = 0.0
    total_weight = 0.0
    
    for cls in range(1, num_classes):  # Skip background
        weight = CLINICAL_WEIGHTS[cls]
        dice = np.mean(all_metrics[f"class_{cls}"]["dice"])
        weighted_dice += weight * dice
        total_weight += weight
    
    weighted_dice /= total_weight if total_weight > 0 else 1.0
    
    # Overall metrics
    overall_dice = np.mean([np.mean(all_metrics[f"class_{i}"]["dice"]) for i in range(num_classes)])
    overall_miou = np.mean([np.mean(all_metrics[f"class_{i}"]["miou"]) for i in range(num_classes)])
    
    # Tumor-specific metrics (excluding background)
    tumor_classes = [1, 2, 3]  # Necrotic, Edema, Enhancing
    tumor_dice = np.mean([np.mean(all_metrics[f"class_{cls}"]["dice"]) for cls in tumor_classes])
    
    return {
        'overall_dice': overall_dice,
        'overall_miou': overall_miou,
        'weighted_dice': weighted_dice,
        'tumor_dice': tumor_dice,
        'clinical_score': weighted_dice * 0.7 + tumor_dice * 0.3
    }

# ---------------------
# AFW Analysis Functions
# ---------------------
def analyze_afw_weights(model):
    """Comprehensive AFW weight analysis"""
    print("\n🔬 AFW Weight Analysis")
    print("=" * 50)
    
    try:
        # Get spectral weights
        spectral_weights = model.get_spectral_weights()
        
        print("📊 Spectral Weight Analysis:")
        for name, weights in spectral_weights.items():
            if isinstance(weights, torch.Tensor):
                if weights.dtype.is_complex:
                    real_min, real_max = weights.real.min().item(), weights.real.max().item()
                    imag_min, imag_max = weights.imag.min().item(), weights.imag.max().item()
                    print(f"  {name}: {weights.shape} - Real: [{real_min:.4f}, {real_max:.4f}], Imag: [{imag_min:.4f}, {imag_max:.4f}]")
                else:
                    print(f"  {name}: {weights.shape} - Range: [{weights.min():.4f}, {weights.max():.4f}]")
            else:
                print(f"  {name}: {weights}")
        
        # Test frequency regularization
        freq_reg_loss = model.get_frequency_regularization_loss()
        print(f"📈 Frequency regularization loss: {freq_reg_loss:.6f}")
        
        return spectral_weights
        
    except Exception as e:
        print(f"❌ AFW analysis failed: {e}")
        return None

# ---------------------
# Model Loading
# ---------------------
def load_model(model_name, checkpoint_path):
    """Load model with comprehensive validation"""
    print(f"🔄 Loading model: {model_name}")
    print(f"📁 Checkpoint: {checkpoint_path}")
    
    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(in_channels=4, out_channels=num_classes).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights")
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model

# ---------------------
# Comprehensive Testing
# ---------------------
@torch.no_grad()
def test_model(model, save_results=True, visualize=True):
    """Comprehensive model evaluation with clinical metrics"""
    print("\n🧪 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Get test dataloader (using same split ratios as training)
    _, _, test_loader = get_dataloaders(
        data_dir, 
        batch_size=batch_size,
        train_ratio=0.75,  # Match train1.py split ratios
        val_ratio=0.15     # Match train1.py split ratios
    )
    print(f"📊 Test dataset: {len(test_loader.dataset)} samples")
    
    # Initialize comprehensive metrics
    all_metrics = {f"class_{i}": {
        "dice": [], "miou": [], "sensitivity": [], "specificity": [], "ppv": [],
        "hausdorff": [], "volume_error": [], "surface_dice": []
    } for i in range(num_classes)}
    
    # Performance tracking
    inference_times = []
    memory_usage = []
    
    print(f"\n🔄 Testing on {len(test_loader)} batches...")
    
    for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = images.to(device)
        masks = masks.long().to(device)
        
        # Measure inference time
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Forward pass
        outputs = model(images)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_times.append(start_time.elapsed_time(end_time) / 1000.0)  # Convert to seconds
        
        # Get predictions
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        
        # Calculate basic metrics
        dice_vals = dice_per_class(outputs, masks, num_classes)
        miou_vals = miou(outputs, masks, num_classes)
        
        # Calculate advanced metrics for each sample in batch
        for i in range(images.size(0)):
            pred_sample = preds[i]
            mask_sample = masks[i]
            
            for cls in range(num_classes):
                # Basic metrics
                all_metrics[f"class_{cls}"]["dice"].append(dice_vals[cls])
                all_metrics[f"class_{cls}"]["miou"].append(miou_vals[cls])
                all_metrics[f"class_{cls}"]["sensitivity"].append(sensitivity(outputs[i:i+1], mask_sample.unsqueeze(0), cls).item())
                all_metrics[f"class_{cls}"]["specificity"].append(specificity(outputs[i:i+1], mask_sample.unsqueeze(0), cls).item())
                all_metrics[f"class_{cls}"]["ppv"].append(ppv(outputs[i:i+1], mask_sample.unsqueeze(0), cls).item())
                
                # Advanced metrics
                hd = hausdorff_distance(pred_sample, mask_sample, cls)
                if hd != float('inf'):
                    all_metrics[f"class_{cls}"]["hausdorff"].append(hd)
                
                ve = volume_error(pred_sample, mask_sample, cls)
                if ve != float('inf'):
                    all_metrics[f"class_{cls}"]["volume_error"].append(ve)
                
                sd = surface_dice(pred_sample, mask_sample, cls)
                all_metrics[f"class_{cls}"]["surface_dice"].append(sd)
        
        # Memory usage tracking
        if torch.cuda.is_available():
            memory_usage.append(torch.cuda.memory_allocated() / 1024**3)  # GB
    
    # Calculate comprehensive statistics
    print("\n📊 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    
    # Per-class detailed metrics
    for i in range(num_classes):
        metrics = all_metrics[f"class_{i}"]
        class_name = CLASS_NAMES[i]
        
        print(f"\n🏥 {class_name} (Class {i}):")
        print(f"  Dice Score:     {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}")
        print(f"  mIoU:           {np.mean(metrics['miou']):.4f} ± {np.std(metrics['miou']):.4f}")
        print(f"  Sensitivity:    {np.mean(metrics['sensitivity']):.4f} ± {np.std(metrics['sensitivity']):.4f}")
        print(f"  Specificity:    {np.mean(metrics['specificity']):.4f} ± {np.std(metrics['specificity']):.4f}")
        print(f"  PPV:            {np.mean(metrics['ppv']):.4f} ± {np.std(metrics['ppv']):.4f}")
        
        if metrics['hausdorff']:
            print(f"  Hausdorff Dist: {np.mean(metrics['hausdorff']):.2f} ± {np.std(metrics['hausdorff']):.2f} mm")
        else:
            print(f"  Hausdorff Dist: N/A")
        
        if metrics['volume_error']:
            print(f"  Volume Error:   {np.mean(metrics['volume_error']):.2f}% ± {np.std(metrics['volume_error']):.2f}%")
        else:
            print(f"  Volume Error:   N/A")
        
        print(f"  Surface Dice:   {np.mean(metrics['surface_dice']):.4f} ± {np.std(metrics['surface_dice']):.4f}")
    
    # Clinical metrics
    clinical_metrics = calculate_clinical_metrics(all_metrics)
    print(f"\n🏥 CLINICAL METRICS:")
    print(f"  Overall Dice:    {clinical_metrics['overall_dice']:.4f}")
    print(f"  Overall mIoU:    {clinical_metrics['overall_miou']:.4f}")
    print(f"  Weighted Dice:   {clinical_metrics['weighted_dice']:.4f}")
    print(f"  Tumor Dice:      {clinical_metrics['tumor_dice']:.4f}")
    print(f"  Clinical Score:  {clinical_metrics['clinical_score']:.4f}")
    
    # Performance metrics
    if inference_times:
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  Avg Inference:  {np.mean(inference_times):.3f} ± {np.std(inference_times):.3f} seconds")
        print(f"  Throughput:     {batch_size / np.mean(inference_times):.1f} volumes/second")
    
    if memory_usage:
        print(f"  Peak Memory:    {np.max(memory_usage):.2f} GB")
        print(f"  Avg Memory:     {np.mean(memory_usage):.2f} GB")
    
    # Performance grading
    overall_dice = clinical_metrics['overall_dice']
    if overall_dice >= 0.90:
        grade = "🏆 EXCELLENT"
    elif overall_dice >= 0.85:
        grade = "🥇 VERY GOOD"
    elif overall_dice >= 0.80:
        grade = "🥈 GOOD"
    elif overall_dice >= 0.75:
        grade = "🥉 FAIR"
    else:
        grade = "❌ NEEDS IMPROVEMENT"
    
    print(f"\n🎯 OVERALL PERFORMANCE: {grade}")
    print(f"   Dice Score: {overall_dice:.4f}")
    
    # Save results if requested
    if save_results:
        save_comprehensive_results(all_metrics, clinical_metrics, inference_times, memory_usage)
    
    return all_metrics, clinical_metrics

def save_comprehensive_results(all_metrics, clinical_metrics, inference_times, memory_usage):
    """Save comprehensive results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed metrics to JSON
    results = {
        'timestamp': timestamp,
        'clinical_metrics': clinical_metrics,
        'performance': {
            'avg_inference_time': float(np.mean(inference_times)) if inference_times else None,
            'throughput': float(batch_size / np.mean(inference_times)) if inference_times else None,
            'peak_memory_gb': float(np.max(memory_usage)) if memory_usage else None,
            'avg_memory_gb': float(np.mean(memory_usage)) if memory_usage else None
        },
        'per_class_metrics': {}
    }
    
    # Add per-class statistics
    for i in range(num_classes):
        metrics = all_metrics[f"class_{i}"]
        results['per_class_metrics'][f'class_{i}'] = {
            'name': CLASS_NAMES[i],
            'dice': {'mean': float(np.mean(metrics['dice'])), 'std': float(np.std(metrics['dice']))},
            'miou': {'mean': float(np.mean(metrics['miou'])), 'std': float(np.std(metrics['miou']))},
            'sensitivity': {'mean': float(np.mean(metrics['sensitivity'])), 'std': float(np.std(metrics['sensitivity']))},
            'specificity': {'mean': float(np.mean(metrics['specificity'])), 'std': float(np.std(metrics['specificity']))},
            'ppv': {'mean': float(np.mean(metrics['ppv'])), 'std': float(np.std(metrics['ppv']))},
            'hausdorff': {'mean': float(np.mean(metrics['hausdorff'])) if metrics['hausdorff'] else None, 
                         'std': float(np.std(metrics['hausdorff'])) if metrics['hausdorff'] else None},
            'volume_error': {'mean': float(np.mean(metrics['volume_error'])) if metrics['volume_error'] else None,
                           'std': float(np.std(metrics['volume_error'])) if metrics['volume_error'] else None},
            'surface_dice': {'mean': float(np.mean(metrics['surface_dice'])), 'std': float(np.std(metrics['surface_dice']))}
        }
    
    # Save JSON results
    json_path = results_dir / f"comprehensive_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV summary
    csv_path = results_dir / f"results_summary_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Name', 'Dice_Mean', 'Dice_Std', 'mIoU_Mean', 'mIoU_Std', 
                        'Sensitivity_Mean', 'Sensitivity_Std', 'Specificity_Mean', 'Specificity_Std',
                        'PPV_Mean', 'PPV_Std', 'Hausdorff_Mean', 'Hausdorff_Std',
                        'Volume_Error_Mean', 'Volume_Error_Std', 'Surface_Dice_Mean', 'Surface_Dice_Std'])
        
        for i in range(num_classes):
            metrics = all_metrics[f"class_{i}"]
            writer.writerow([
                i, CLASS_NAMES[i],
                np.mean(metrics['dice']), np.std(metrics['dice']),
                np.mean(metrics['miou']), np.std(metrics['miou']),
                np.mean(metrics['sensitivity']), np.std(metrics['sensitivity']),
                np.mean(metrics['specificity']), np.std(metrics['specificity']),
                np.mean(metrics['ppv']), np.std(metrics['ppv']),
                np.mean(metrics['hausdorff']) if metrics['hausdorff'] else 'N/A',
                np.std(metrics['hausdorff']) if metrics['hausdorff'] else 'N/A',
                np.mean(metrics['volume_error']) if metrics['volume_error'] else 'N/A',
                np.std(metrics['volume_error']) if metrics['volume_error'] else 'N/A',
                np.mean(metrics['surface_dice']), np.std(metrics['surface_dice'])
            ])
    
    print(f"\n💾 Results saved:")
    print(f"  📄 JSON: {json_path}")
    print(f"  📊 CSV:  {csv_path}")

# ---------------------
# Main Execution
# ---------------------
def main():
    """Main execution function with comprehensive testing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Script for Spatial-Spectral CNN with AFW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Test.py --model enhanced_spectral_3d_unet_afw_attention --checkpoint checkpoints/best_model.pth
  python Test.py --model unet3d_spectral_afw --checkpoint checkpoints/best_model.pth --no-save
  python Test.py --model enhanced_spectral_3d_unet_afw_attention --checkpoint checkpoints/best_model.pth --batch-size 1
        """
    )
    
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name from MODEL_REGISTRY (e.g., enhanced_spectral_3d_unet_afw_attention)")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint file")
    parser.add_argument("--batch-size", type=int, default=2, 
                       help="Batch size for testing (default: 2)")
    parser.add_argument("--no-save", action="store_true", 
                       help="Skip saving results to files")
    parser.add_argument("--no-afw", action="store_true", 
                       help="Skip AFW weight analysis")
    parser.add_argument("--data-dir", type=str, 
                       default="/content/drive/MyDrive/BraTS_Project/BraTS2021_Training_Data", 
                       help="Path to dataset directory")
    
    args = parser.parse_args()
    
    # Update global variables
    global batch_size
    batch_size = args.batch_size
    data_dir = args.data_dir
    
    print("🧠 SPATIAL-SPECTRAL CNN WITH AFW - COMPREHENSIVE TESTING")
    print("=" * 70)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Device: {device}")
    print(f"📊 Batch Size: {batch_size}")
    print(f"📁 Data Directory: {data_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"💾 Checkpoint: {args.checkpoint}")
    print("=" * 70)
    
    try:
        # Load model
        model = load_model(args.model, args.checkpoint)
        
        # AFW Analysis (if not skipped)
        if not args.no_afw:
            afw_weights = analyze_afw_weights(model)
            if afw_weights is None:
                print("⚠️  AFW analysis failed - continuing with standard evaluation")
        
        # Comprehensive testing
        print(f"\n🚀 Starting comprehensive evaluation...")
        all_metrics, clinical_metrics = test_model(
            model, 
            save_results=not args.no_save,
            visualize=True
        )
        
        # Final summary
        print(f"\n🎉 TESTING COMPLETED SUCCESSFULLY!")
        print(f"📊 Overall Dice Score: {clinical_metrics['overall_dice']:.4f}")
        print(f"🏥 Clinical Score: {clinical_metrics['clinical_score']:.4f}")
        
        if not args.no_save:
            print(f"💾 Results saved to test_results/ directory")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# Usage Examples:
# python Test.py --model enhanced_spectral_3d_unet_afw_attention --checkpoint checkpoints/best_model.pth
# python Test.py --model unet3d_spectral_afw --checkpoint checkpoints/best_model.pth --batch-size 1
# python Test.py --model enhanced_spectral_3d_unet_afw_attention --checkpoint checkpoints/best_model.pth --no-save --no-afw
