import csv
import matplotlib.pyplot as plt
import pandas as pd

def load_logs(csv_path, num_classes=4):
    epochs = []
    train_losses = []
    val_losses = []
    metrics_per_class = {
        cls: {
            'dice': [], 'miou': [], 'sensitivity': [],
            'specificity': [], 'ppv': []
        } for cls in range(num_classes)
    }

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        _ = next(reader)  # Skip header

        for row in reader:
            epoch = int(row[0])
            train_loss = float(row[1])
            val_loss = float(row[2])

            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            for cls in range(num_classes):
                base_idx = 3 + cls * 5  # Start after epoch/train_loss/val_loss
                metrics_per_class[cls]['dice'].append(float(row[base_idx]))
                metrics_per_class[cls]['miou'].append(float(row[base_idx + 1]))
                metrics_per_class[cls]['sensitivity'].append(float(row[base_idx + 2]))
                metrics_per_class[cls]['specificity'].append(float(row[base_idx + 3]))
                metrics_per_class[cls]['ppv'].append(float(row[base_idx + 4]))

    return epochs, train_losses, val_losses, metrics_per_class


def plot_losses(epochs, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()


def plot_metrics_per_class(epochs, metrics_per_class):
    for metric_name in ['dice', 'miou', 'sensitivity', 'specificity', 'ppv']:
        plt.figure(figsize=(10, 6))
        for cls, metrics in metrics_per_class.items():
            plt.plot(epochs, metrics[metric_name], label=f'Class {cls}')
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.title(f"{metric_name.capitalize()} Over Epochs (Per Class)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{metric_name}_per_class.png")
        plt.show()


#plot_afw_loss.py

def plot_afw_loss(log_path="logs/afw_loss_log.csv", save_path="afw_loss_plot.png"):
    try:
        df = pd.read_csv(log_path)

        if df.empty:
            print("Log file is empty.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(df["Epoch"], df["AFW_Loss"], marker='o', label="AFW Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("AFW Entropy Loss")
        plt.title("Adaptive Frequency Weighting (AFW) Loss Trend Over Epochs")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

        print(f"✅ Plot saved as {save_path}")

    except FileNotFoundError:
        print(f"❌ Log file '{log_path}' not found.")


if __name__ == "__main__":
    csv_log_path = "logs/metrics_log.csv"
    num_classes = 4

    epochs, train_losses, val_losses, metrics_per_class = load_logs(csv_log_path, num_classes)

    plot_losses(epochs, train_losses, val_losses)
    plot_metrics_per_class(epochs, metrics_per_class)

    plot_afw_loss()
