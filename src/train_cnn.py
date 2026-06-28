import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import MODELS_DIR, CLASSES, BASE_DIR

def get_df_from_dir(directory):
    data = []
    for label, cls_name in enumerate(CLASSES):
        cls_path = os.path.join(directory, cls_name)
        if not os.path.exists(cls_path): continue
        for f in os.listdir(cls_path):
            if f.endswith('.jpg'):
                data.append({'image_path': os.path.join(cls_path, f), 'label': label})
    return pd.DataFrame(data)

# 1. Custom Dataset
class AcneDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        label = self.dataframe.loc[idx, 'label']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (512, 512))

        if self.transform:
            image = self.transform(image)
        return image, label

# 2. Custom CNN Architecture
# BatchNorm2d added after each Conv2d:
#   - Normalizes activations per-batch → stabilizes training
#   - Allows higher learning rates, reduces sensitivity to weight init
#   - Acts as mild regularizer → reduces overfitting on small datasets
class AcneCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(AcneCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 256×256

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 128×128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 64×64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32×32
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_cnn():
    train_dir = os.path.join(BASE_DIR, 'data_split', 'train')
    test_dir  = os.path.join(BASE_DIR, 'data_split', 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("Error: data_split directories not found. Run data_prep.py first.")
        return

    train_df = get_df_from_dir(train_dir)
    test_df  = get_df_from_dir(test_dir)

    # Training transform — augmentation applied only on train set
    # Flips and rotation are label-invariant for facial skin photos.
    # ColorJitter simulates lighting/camera variation across subjects.
    # Test transform is deterministic (no augmentation).
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = AcneDataset(train_df, transform=train_transform)
    test_dataset  = AcneDataset(test_df,  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # num_classes tied to config — never hardcoded
    model = AcneCNN(num_classes=len(CLASSES)).to(device)

    # Training set is balanced (300/class via data_prep), so unweighted loss is correct.
    # CrossEntropyLoss with no weight treats all classes equally during training.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ReduceLROnPlateau: halve LR when training loss stops improving for 5 epochs.
    # Prevents oscillating around a local minimum without manual LR tuning.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    epochs             = 50
    best_loss          = float('inf')
    patience_counter   = 0
    early_stop_patience = 10   # stop if no improvement for 10 consecutive epochs

    print("Starting Training...")
    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        loss_history.append(epoch_loss)

        # Step scheduler based on epoch loss
        scheduler.step(epoch_loss)

        # Early stopping + best model checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'acne_cnn_best.pth'))
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:>2}/{epochs} | Loss: {epoch_loss:.4f} | "
              f"Best: {best_loss:.4f} | Patience: {patience_counter}/{early_stop_patience} | "
              f"LR: {current_lr:.6f}")

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    # Restore best weights for evaluation
    best_ckpt = os.path.join(MODELS_DIR, 'acne_cnn_best.pth')
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print("Loaded best model weights for evaluation.")

    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-')
    plt.title('CNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'cnn_loss_curve.png'))
    plt.close()
    print(f"Loss curve saved to {MODELS_DIR}/cnn_loss_curve.png")

    # Evaluation
    print("\nEvaluating CNN Model...")
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1   = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)

    print(f"\n--- CNN Results ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nPer-class F1:")
    for cls_name, cls_f1 in zip(CLASSES, f1_per_class):
        print(f"  {cls_name}: {cls_f1:.4f}")

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix - Custom CNN')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'cm_cnn.png'))
    plt.close()
    print(f"Confusion Matrix saved to {MODELS_DIR}/cm_cnn.png")

    # Save final model
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'acne_cnn_model.pth'))
    print(f"Model saved to {MODELS_DIR}/acne_cnn_model.pth")

if __name__ == "__main__":
    train_cnn()
