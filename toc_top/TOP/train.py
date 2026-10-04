import argparse
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import TCN


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a TCN for multi-step bounding-box trajectory prediction."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="train_data",
    )
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--pred_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--nhid", type=int, default=128)
    parser.add_argument("--ksize", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train_ratio", type=int, default=7)
    parser.add_argument("--val_ratio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output_dir", type=str, default="train_result")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Reproducibility and runtime utilities
# -----------------------------------------------------------------------------
def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name):
    """Resolve the requested computation device."""
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_run_folder(output_dir, base_name="run"):
    """Create a new run directory without overwriting previous results."""
    os.makedirs(output_dir, exist_ok=True)

    run_id = 1
    while os.path.exists(os.path.join(output_dir, f"{base_name}{run_id}")):
        run_id += 1

    run_dir = os.path.join(output_dir, f"{base_name}{run_id}")
    os.makedirs(run_dir)
    return run_dir


# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------
def smooth_labels(bbox_seq, window_size=3):
    """Apply moving-average smoothing to [x, y, w, h] trajectories."""
    smoothed = np.copy(bbox_seq)
    kernel = np.ones(window_size, dtype=np.float32) / window_size

    for i in range(4):
        smoothed[:, i] = np.convolve(
            bbox_seq[:, i],
            kernel,
            mode="same",
        )

    return smoothed


def load_trajectory(file_path):
    """Load one trajectory file and keep its last four bbox columns."""
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".npy":
        array = np.load(file_path)
    else:
        try:
            array = np.loadtxt(file_path, delimiter=",")
        except ValueError:
            array = np.loadtxt(file_path)

    array = np.atleast_2d(array)

    if array.shape[1] < 4:
        raise ValueError(
            f"Expected at least 4 columns, but got {array.shape[1]}."
        )

    return array[:, -4:].astype(np.float32)


def extract_samples(file_list, seq_len, pred_steps):
    """Build sliding-window input/target pairs from trajectory files."""
    inputs = []
    targets = []
    required_length = seq_len + pred_steps

    for file_path in file_list:
        try:
            data = load_trajectory(file_path)
            data = smooth_labels(data, window_size=3)

            if data.shape[0] < required_length:
                continue

            for i in range(data.shape[0] - required_length + 1):
                inputs.append(data[i : i + seq_len])
                targets.append(
                    data[i + seq_len : i + seq_len + pred_steps]
                )

        except (OSError, ValueError) as exc:
            print(f"Warning: skipping '{file_path}': {exc}")

    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def load_data_split_by_files(
    data_dir,
    seq_len,
    train_ratio,
    val_ratio,
    pred_steps,
    seed,
):
    """Split trajectory files into train/validation sets before windowing."""

    if train_ratio + val_ratio != 10:
        raise ValueError("train_ratio + val_ratio must equal 10.")

    patterns = ("*.csv", "*.txt", "*.npy")
    all_files = []

    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(data_dir, pattern)))

    all_files = sorted(all_files)

    if not all_files:
        available = (
            sorted(os.listdir(data_dir))
            if os.path.isdir(data_dir)
            else []
        )
        raise ValueError(
            f"No data files found in {data_dir}. "
            f"Directory listing: {available}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(all_files)

    train_count = len(all_files) * train_ratio // 10
    train_files = all_files[:train_count]
    val_files = all_files[train_count:]

    if not train_files or not val_files:
        raise ValueError(
            "The dataset must contain enough files for both "
            "training and validation."
        )

    train_x, train_y = extract_samples(
        train_files,
        seq_len,
        pred_steps,
    )
    val_x, val_y = extract_samples(
        val_files,
        seq_len,
        pred_steps,
    )

    if len(train_x) == 0:
        raise ValueError("No valid training samples were generated.")

    if len(val_x) == 0:
        raise ValueError("No valid validation samples were generated.")

    return train_x, train_y, val_x, val_y


# -----------------------------------------------------------------------------
# Loss functions and metrics
# -----------------------------------------------------------------------------
def bbox_iou(box1, box2, eps=1e-6):
    """Compute IoU for center-format boxes [x, y, w, h]."""
    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)

    inter_area = (
        (inter_x2 - inter_x1).clamp(min=0)
        * (inter_y2 - inter_y1).clamp(min=0)
    )
    area1 = (
        (b1_x2 - b1_x1).clamp(min=0)
        * (b1_y2 - b1_y1).clamp(min=0)
    )
    area2 = (
        (b2_x2 - b2_x1).clamp(min=0)
        * (b2_y2 - b2_y1).clamp(min=0)
    )

    union = area1 + area2 - inter_area + eps
    return inter_area / union


def temporal_smoothness_loss(pred_seq, weight=0.05):
    """Penalize abrupt changes between consecutive predicted boxes."""
    diff = pred_seq[:, 1:, :] - pred_seq[:, :-1, :]
    return weight * torch.mean(diff**2)


def compute_loss(
    pred,
    target,
    alpha=1.0,
    beta=0.5,
    gamma=0.5,
    smooth_weight=0.05,
):
    """Combine position, size, IoU, and temporal smoothness losses."""

    loss_xy = F.l1_loss(pred[..., :2], target[..., :2])
    loss_wh = F.l1_loss(pred[..., 2:], target[..., 2:])

    ious = bbox_iou(
        pred.reshape(-1, 4),
        target.reshape(-1, 4),
    )
    iou_loss = 1.0 - ious.mean()

    smooth_loss = temporal_smoothness_loss(
        pred,
        weight=smooth_weight,
    )

    total_loss = (
        alpha * loss_xy
        + beta * loss_wh
        + gamma * iou_loss
        + smooth_loss
    )

    return total_loss


def seq2seq_predict(model, inputs, pred_steps):
    """Reshape model output to [batch, pred_steps, 4]."""
    output = model(inputs)

    if output.dim() == 3:
        output = output[:, -1, :]

    return output.view(-1, pred_steps, 4)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def evaluate(model, val_loader, device, pred_steps):
    """Evaluate validation loss, IoU, ADE, and FDE."""
    model.eval()

    val_loss = 0.0
    total_iou = 0.0
    total_boxes = 0
    ade_total = 0.0
    fde_total = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            pred = seq2seq_predict(
                model,
                inputs,
                pred_steps,
            )

            loss = compute_loss(pred, targets)
            val_loss += loss.item()

            ious = bbox_iou(
                pred.reshape(-1, 4),
                targets.reshape(-1, 4),
            )
            total_iou += ious.sum().item()
            total_boxes += ious.numel()

            ade = torch.norm(
                pred[..., :2] - targets[..., :2],
                dim=-1,
            ).mean()

            fde = torch.norm(
                pred[:, -1, :2] - targets[:, -1, :2],
                dim=-1,
            ).mean()

            ade_total += ade.item()
            fde_total += fde.item()
            num_batches += 1

    val_loss /= max(1, len(val_loader))
    mean_iou = total_iou / max(1, total_boxes)
    mean_ade = ade_total / max(1, num_batches)
    mean_fde = fde_total / max(1, num_batches)

    return val_loss, mean_iou, mean_ade, mean_fde


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def save_metric_curves(
    train_loss_history,
    val_loss_history,
    iou_history,
    ade_history,
    fde_history,
    save_path,
):
    """Save training-loss and validation-metric curves."""

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iou_history, label="IoU")
    plt.plot(ade_history, label="ADE")
    plt.plot(fde_history, label="FDE")
    plt.xlabel("Epoch")
    plt.title("Validation Metrics")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(args):
    """Train a single TCN model and save the best checkpoint."""
    set_seed(args.seed)

    device = get_device(args.device)
    run_dir = create_run_folder(args.output_dir)

    print(f"Device: {device}")
    print(f"Results will be saved to: {run_dir}")

    # Split files first to avoid train/validation leakage between windows.
    train_x, train_y, val_x, val_y = load_data_split_by_files(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        pred_steps=args.pred_steps,
        seed=args.seed,
    )

    print(f"Train samples: {len(train_x)}")
    print(f"Validation samples: {len(val_x)}")
    print(f"Total samples: {len(train_x) + len(val_x)}")

    train_x = torch.from_numpy(train_x).transpose(1, 2)
    train_y = torch.from_numpy(train_y)
    val_x = torch.from_numpy(val_x).transpose(1, 2)
    val_y = torch.from_numpy(val_y)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=args.batch_size,
        shuffle=False,
    )

    channel_sizes = [args.nhid] * args.levels

    # The model predicts pred_steps future boxes, each represented by 4 values.
    model = TCN(
        input_size=4,
        output_size=4 * args.pred_steps,
        num_channels=channel_sizes,
        kernel_size=args.ksize,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_metrics = {
        "val_loss": float("inf"),
        "iou": 0.0,
        "ade": float("inf"),
    }

    train_loss_history = []
    val_loss_history = []
    iou_history = []
    ade_history = []
    fde_history = []

    best_model_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            pred = seq2seq_predict(
                model,
                inputs,
                args.pred_steps,
            )
            loss = compute_loss(pred, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        val_loss, mean_iou, mean_ade, mean_fde = evaluate(
            model,
            val_loader,
            device,
            args.pred_steps,
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        iou_history.append(mean_iou)
        ade_history.append(mean_ade)
        fde_history.append(mean_fde)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train={train_loss:.6f} | "
            f"Val={val_loss:.6f} | "
            f"IoU={mean_iou:.4f} | "
            f"ADE={mean_ade:.4f} | "
            f"FDE={mean_fde:.4f}"
        )

        # Preserve the original checkpoint rule: all three criteria must improve.
        improved = (
            val_loss < best_metrics["val_loss"]
            and mean_iou > best_metrics["iou"]
            and mean_ade < best_metrics["ade"]
        )

        if improved:
            best_metrics.update(
                {
                    "val_loss": val_loss,
                    "iou": mean_iou,
                    "ade": mean_ade,
                }
            )

            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            model_name = (
                f"best_model_"
                f"loss{val_loss:.4f}_"
                f"iou{mean_iou:.4f}_"
                f"ade{mean_ade:.4f}.pt"
            )
            best_model_path = os.path.join(
                run_dir,
                model_name,
            )

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print(f"Saved best model: {model_name}")

    save_metric_curves(
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        iou_history=iou_history,
        ade_history=ade_history,
        fde_history=fde_history,
        save_path=os.path.join(
            run_dir,
            "metrics_curve.png",
        ),
    )

    print("Training completed.")

    if best_model_path is not None:
        print(f"Best model: {best_model_path}")
    else:
        print("No checkpoint was saved.")


def main():
    """Program entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
