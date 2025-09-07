# log_utils.py

import os
import csv
import json

def init_log_files(csv_path, json_path, num_classes):
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['epoch', 'train_loss', 'val_loss']
            for cls in range(num_classes):
                header += [
                    f'dice_class_{cls}',
                    f'miou_class_{cls}',
                    f'sensitivity_class_{cls}',
                    f'specificity_class_{cls}',
                    f'ppv_class_{cls}'
                ]
            writer.writerow(header)

    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump([], f, indent=4)

def log_to_csv(csv_path, epoch_record):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [
            epoch_record['epoch'],
            round(epoch_record['train_loss'], 6),
            round(epoch_record['val_loss'], 6)
        ]
        for cls_metrics in epoch_record['metrics']:
            row.extend([
                round(cls_metrics['dice'], 6),
                round(cls_metrics['miou'], 6),
                round(cls_metrics['sensitivity'], 6),
                round(cls_metrics['specificity'], 6),
                round(cls_metrics['ppv'], 6)
            ])
        writer.writerow(row)


def log_to_json(json_path, epoch_record):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(epoch_record)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
