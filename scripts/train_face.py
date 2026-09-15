"""
Training Script for Face Detection Model on extracted WebFace 112x112 images.

Trains FaceDetectorNet using:
  - Cross-Entropy Loss for face identity classification
  - GPU acceleration (NVIDIA RTX 3050 CUDA)
  - Mixed precision training for speed
  - Cosine annealing learning rate schedule

Usage:
    python scripts/train_face.py
    python scripts/train_face.py --epochs 30 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_dataset import WebFaceFolderDataset
from src.face_model import FaceDetectorNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Face Detection Model on WebFace + MS1M-RetinaFace-t1")
    parser.add_argument("--data-dirs", type=str, nargs="+", default=None, help="One or more extracted face dataset directories")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "models" / "face"))
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler,
) -> float:
    """Train one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            acc = 100.0 * correct / total
            avg_loss = running_loss / total
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    epoch_loss = running_loss / max(1, total)
    return epoch_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / max(1, total)
    val_acc = 100.0 * correct / max(1, total)
    return val_loss, val_acc


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("[Face Detection Model Training — WebFace 112x112]")
    print("=" * 60)
    print(f"Device          : {device}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch Size      : {args.batch_size}")
    print(f"Learning Rate   : {args.lr}")
    print(f"Save Directory  : {args.save_dir}")
    print("-" * 60)

    # Load dataset
    print("\n[1/4] Loading face dataset(s)...")
    dataset = WebFaceFolderDataset(roots=args.data_dirs)
    num_classes = dataset.num_classes

    # Train/Val split (90/10)
    total_len = len(dataset)
    val_len = max(1, int(total_len * 0.1))
    train_len = total_len - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"  Training samples  : {train_len}")
    print(f"  Validation samples: {val_len}")
    print(f"  Num classes       : {num_classes}")
    print(f"  Batches per epoch : {len(train_loader)}")

    # Build model
    print("\n[2/4] Building FaceDetectorNet...")
    model = FaceDetectorNet(num_classes=num_classes)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type)

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # Training loop
    print("\n[3/4] Starting training...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | LR: {lr:.6f} | Time: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save last checkpoint
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch, "num_classes": num_classes},
            save_dir / "last.pt",
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "num_classes": num_classes},
                save_dir / "best.pt",
            )
            print(f"  >>> Best model saved! (Val Loss: {val_loss:.4f})")

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("[4/4] Training complete!")
    print(f"  Best checkpoint : {save_dir / 'best.pt'}")
    print(f"  Last checkpoint : {save_dir / 'last.pt'}")
    print(f"  Training history: {save_dir / 'history.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
