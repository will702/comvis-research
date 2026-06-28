"""
train_cnn_6535.py
=================
CNN training pipeline with:
  - 65/35 train-test split  (reads from data_split_6535/ built by train_6535.py)
  - Balanced test set: downsampled to smallest class count before evaluation
  - All outputs isolated to models_6535/

Run:
    python -m scripts.train_cnn_6535
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import BASE_DIR, CLASSES

# ── Paths (isolated from original pipeline) ──────────────────────────────────
SPLIT_DIR = os.path.join(BASE_DIR, 'data_split_6535')
OUT_DIR   = os.path.join(BASE_DIR, 'models_6535')
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# Dataset
# ============================================================================

def get_df_from_dir(directory):
    data = []
    for label, cls_name in enumerate(CLASSES):
        cls_path = os.path.join(directory, cls_name)
        if not os.path.exists(cls_path):
            continue
        for f in sorted(os.listdir(cls_path)):
            if f.endswith('.jpg'):
                data.append({'image_path': os.path.join(cls_path, f),
                              'label': label,
                              'class': cls_name})
    return pd.DataFrame(data)


def balance_dataframe(df, random_state=42):
    """Downsample each class to the count of the smallest class."""
    min_count = df['label'].value_counts().min()
    balanced = (df.groupby('label', group_keys=False)
                  .apply(lambda g: g.sample(n=min_count, random_state=random_state)))
    return balanced.reset_index(drop=True)


class AcneDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        label    = self.df.loc[idx, 'label']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (512, 512))
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================================
# Model (same architecture as src/train_cnn.py)
# ============================================================================

class AcneCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),   # 256×256

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),   # 128×128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),   # 64×64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),  # 32×32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================================
# Training + evaluation
# ============================================================================

def train_cnn_6535():
    train_dir = os.path.join(SPLIT_DIR, 'train')
    test_dir  = os.path.join(SPLIT_DIR, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            "data_split_6535/ not found.\n"
            "Run first:  python -m scripts.train_6535 --split-only"
        )

    # ── DataFrames ───────────────────────────────────────────────────────────
    train_df    = get_df_from_dir(train_dir)
    test_df_raw = get_df_from_dir(test_dir)
    test_df     = balance_dataframe(test_df_raw)

    print(f"\nTrain set : {len(train_df)} images")
    print(f"Test  set : {len(test_df_raw)} images (raw) "
          f"→ {len(test_df)} balanced "
          f"({len(test_df) // len(CLASSES)} per class)")
    print(f"Train class counts:\n{train_df['label'].value_counts().sort_index().to_dict()}")
    print(f"Test  class counts (balanced):\n"
          f"{test_df['label'].value_counts().sort_index().to_dict()}")

    # ── Transforms ───────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = AcneDataset(train_df, transform=train_transform)
    test_dataset  = AcneDataset(test_df,  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")

    model     = AcneCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs              = 50
    best_loss           = float('inf')
    patience_counter    = 0
    early_stop_patience = 10
    loss_history        = []
    best_ckpt           = os.path.join(OUT_DIR, 'acne_cnn_best_6535.pth')

    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:>2}/{epochs} | Loss: {epoch_loss:.4f} | "
              f"Best: {best_loss:.4f} | "
              f"Patience: {patience_counter}/{early_stop_patience} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}.")
            break

    # ── Restore best weights ─────────────────────────────────────────────────
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print("Loaded best checkpoint for evaluation.")

    # ── Loss curve ───────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.title('CNN Training Loss — 65/35 Balanced Test')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True); plt.tight_layout()
    curve_path = os.path.join(OUT_DIR, 'cnn_loss_curve_6535.png')
    plt.savefig(curve_path); plt.close()
    print(f"\nLoss curve → {curve_path}")

    # ── Evaluation on balanced test set ──────────────────────────────────────
    print("\nEvaluating on balanced test set...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1   = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_class   = f1_score(all_labels, all_preds, average=None, zero_division=0)
    prec_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    rec_per_class  = recall_score(all_labels, all_preds, average=None, zero_division=0)
    cm             = confusion_matrix(all_labels, all_preds)

    print(f"\n{'=' * 55}")
    print(f"  CNN RESULTS — 65/35 SPLIT, BALANCED TEST")
    print(f"{'=' * 55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Per-class breakdown ({len(test_df)//len(CLASSES)} samples/class):")
    print(f"  {'Class':<18} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print(f"  {'-'*48}")
    for cls_name, p, r, f in zip(CLASSES, prec_per_class, rec_per_class, f1_per_class):
        print(f"  {cls_name:<18} {p:>9.4f} {r:>9.4f} {f:>9.4f}")
    print(f"{'=' * 55}")

    print("\nFull classification report:")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASSES, zero_division=0))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix — CNN (65/35, Balanced Test)')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, 'cm_cnn_6535.png')
    plt.savefig(cm_path); plt.close()
    print(f"Confusion matrix → {cm_path}")

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = os.path.join(OUT_DIR, 'acne_cnn_model_6535.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model      → {final_path}")

    # ── Save text summary ─────────────────────────────────────────────────────
    summary_path = os.path.join(OUT_DIR, 'CNN_Summary_6535.txt')
    with open(summary_path, 'w') as fp:
        fp.write("CNN RESULTS — 65/35 SPLIT, BALANCED TEST\n")
        fp.write("=" * 55 + "\n")
        fp.write(f"Accuracy  : {acc:.4f}\n")
        fp.write(f"Precision : {prec:.4f}\n")
        fp.write(f"Recall    : {rec:.4f}\n")
        fp.write(f"F1-Score  : {f1:.4f}\n\n")
        fp.write(f"Per-class ({len(test_df)//len(CLASSES)} samples/class):\n")
        fp.write(f"{'Class':<18} {'Precision':>9} {'Recall':>9} {'F1':>9}\n")
        fp.write("-" * 48 + "\n")
        for cls_name, p, r, f in zip(CLASSES, prec_per_class, rec_per_class, f1_per_class):
            fp.write(f"{cls_name:<18} {p:>9.4f} {r:>9.4f} {f:>9.4f}\n")
        fp.write("\nFull report:\n")
        fp.write(classification_report(all_labels, all_preds,
                                       target_names=CLASSES, zero_division=0))
    print(f"Text summary     → {summary_path}")


if __name__ == '__main__':
    train_cnn_6535()
