"""Train one TCN model for binary occlusion classification from CSV sequences."""

import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from toc_top.TCN.tcn import TemporalConvNet


# Configuration

def parse_args():
    parser = argparse.ArgumentParser(description="Train a TCN occlusion classifier.")
    parser.add_argument("--csv_folder", type=Path, default=Path("train_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("train_result"))
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--early_stop_patience", type=int, default=150)
    return parser.parse_args()


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch before data splitting and model setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Loss

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification with optional class weights."""

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        weights = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", weights)

    def forward(self, logits, targets):
        cross_entropy = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-cross_entropy)
        loss = (1.0 - pt).pow(self.gamma) * cross_entropy
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# Data preparation

def build_sequences(frame, window_size):
    """Use [score, iou, area] as features and the last frame as the label."""
    features = frame[["score", "iou", "area"]].to_numpy(dtype=np.float32)
    labels = frame["label"].to_numpy(dtype=np.int64)

    inputs = []
    targets = []
    for start in range(len(frame) - window_size + 1):
        inputs.append(features[start:start + window_size])
        targets.append(labels[start + window_size - 1])
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def load_csv_data(csv_folder, window_size, train_ratio, seed):
    """Split by CSV file *before* creating overlapping temporal windows."""
    if not 0 < train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1.")
    files = sorted(csv_folder.glob("*.csv"))
    if len(files) < 2:
        raise ValueError(f"Expected at least two CSV files in {csv_folder}; found {len(files)}.")

    rng = random.Random(seed)
    rng.shuffle(files)
    split = max(1, min(int(train_ratio * len(files)), len(files) - 1))
    train_files, val_files = files[:split], files[split:]
    print(f"Training files: {len(train_files)} | Validation files: {len(val_files)}")

    def combine(file_list):
        inputs, targets = [], []
        for path in file_list:
            frame = pd.read_csv(path)
            x, y = build_sequences(frame, window_size)
            if len(x) == 0:
                print(f"Warning: skipping short file {path.name}")
                continue
            inputs.append(x)
            targets.append(y)
        if not inputs:
            raise ValueError("No valid sliding-window samples in the selected file split.")
        return np.concatenate(inputs), np.concatenate(targets)

    return combine(train_files), combine(val_files)


class OcclusionDataset(Dataset):
    """Return feature tensors [channels=3, time] and integer class labels."""

    def __init__(self, inputs, targets):
        self.inputs = torch.from_numpy(inputs).transpose(1, 2)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


# Model

class TCNClassifier(nn.Module):
    """Five-block, 64-channel TCN with a final-time-step classification head."""

    def __init__(self, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(3, [64, 64, 64, 64, 64])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, 2)

    def forward(self, inputs):
        features = self.dropout(self.tcn(inputs))
        return self.classifier(features[:, :, -1])


# Evaluation

def evaluate(model, loader, criterion, device):
    """Calculate validation loss, accuracy, per-class accuracy, and class-1 F1."""
    model.eval()
    loss_sum = 0.0
    predictions, labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss_sum += criterion(logits, targets).item() * targets.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.cpu().tolist())

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )

    def class_accuracy(label):
        mask = labels == label
        return float(100.0 * np.mean(predictions[mask] == labels[mask])) if mask.any() else float("nan")

    return {
        "val_loss": loss_sum / len(labels),
        "val_accuracy": float(100.0 * (predictions == labels).mean()),
        "class0_accuracy": class_accuracy(0),
        "class1_accuracy": class_accuracy(1),
        "class1_f1": float(f1[1]),
    }


def save_curves(history, output_file):
    """Save training loss, validation accuracy, and class-1 F1 curves."""
    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["val_accuracy"] for row in history])
    axes[1].set_title("Validation Accuracy (%)")
    axes[2].plot(epochs, [row["class1_f1"] for row in history])
    axes[2].set_title("Class 1 F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def new_run_directory(base):
    base.mkdir(parents=True, exist_ok=True)
    index = 1
    while (base / f"run{index}").exists():
        index += 1
    output = base / f"run{index}"
    output.mkdir()
    return output


# Training

def train(args):
    """Train once and save both the best (class-1 F1) and final checkpoints."""
    if args.window_size < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Window size, epochs, and batch size must be positive.")
    if args.early_stop_patience < 1:
        raise ValueError("Early stopping patience must be positive.")
    set_seed(args.seed)
    output_dir = new_run_directory(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data = load_csv_data(
        args.csv_folder, args.window_size, args.train_ratio, args.seed
    )
    train_dataset = OcclusionDataset(*train_data)
    val_dataset = OcclusionDataset(*val_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Device: {device} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model = TCNClassifier(dropout=args.dropout).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=[0.2, 0.8]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=15, factor=0.5, min_lr=1e-6
    )

    best_f1 = -1.0
    no_improvement = 0
    history = []
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * targets.size(0)

        train_loss = running_loss / len(train_dataset)
        metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(metrics["val_loss"])
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(
            f"Epoch {epoch:04d}/{args.epochs} | Train {train_loss:.4f} | "
            f"Val {metrics['val_loss']:.4f} | Acc {metrics['val_accuracy']:.2f}% | "
            f"Class 0 Acc {metrics['class0_accuracy']:.2f}% | "
            f"Class 1 Acc {metrics['class1_accuracy']:.2f}% | "
            f"Class 1 F1 {metrics['class1_f1']:.4f}"
        )

        # Select the checkpoint by the F1 score of class 1 (occlusion).
        if metrics["class1_f1"] > best_f1:
            best_f1 = metrics["class1_f1"]
            no_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")
        else:
            no_improvement += 1
            if no_improvement >= args.early_stop_patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, output_dir / "training_curves.png")
    print(f"Training complete. Best class-1 F1: {best_f1:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    train(parse_args())
