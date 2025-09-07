import os
import json
import matplotlib.pyplot as plt
import csv

def load_logs(csv_path, json_path, num_classes=4):
    # Load losses from JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    epochs = []
    losses = {'train': [], 'val': []}
    metrics = {cls: {'dice': [], 'miou': [], 'sensitivity': [], 'specificity': [], 'ppv': []}
               for cls in range(num_classes)}

    for entry in data:
        epochs.append(entry['epoch'])
        losses['train'].append(entry.get('train_loss', 0))  # Optional
        losses['val'].append(entry.get('val_loss', 0))      # Optional

        for cls in range(num_classes):
            m = entry['metrics'][cls]
            metrics[cls]['dice'].append(m['dice'])
            metrics[cls]['miou'].append(m['miou'])
            metrics[cls]['sensitivity'].append(m['sensitivity'])
            metrics[cls]['specificity'].append(m['specificity'])
            metrics[cls]['ppv'].append(m['ppv'])

    return epochs, losses, metrics

def plot_metrics(log_dir, model_name, num_classes=4):
    csv_path = os.path.join(log_dir, "metrics.csv")
    json_path = os.path.join(log_dir, "metrics.json")

    epochs, losses, metrics = load_logs(csv_path, json_path, num_classes)

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses['train'], label='Train Loss')
    plt.plot(epochs, losses['val'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{log_dir}/loss_curve.png")
    plt.show()

    # Plot metrics per class
    for cls in range(num_classes):
        for metric in ['dice', 'miou', 'sensitivity', 'specificity', 'ppv']:
            plt.figure()
            plt.plot(epochs, metrics[cls][metric])
            plt.title(f"{model_name} - {metric.title()} (Class {cls})")
            plt.xlabel("Epoch")
            plt.ylabel(metric.title())
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{log_dir}/{metric}_class_{cls}.png")
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (log folder)")
    args = parser.parse_args()
    plot_metrics(f"logs/{args.model}", args.model)
