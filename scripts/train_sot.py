"""
Train SiamFC Single Object Tracking Model on VisDrone2019-SOT Dataset.

Run from project root:
    python scripts/train_sot.py --epochs 30 --batch-size 16 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sot_dataset import VisDroneSOTDataset
from src.sot_model import SiamFC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SiamFC for Single Object Tracking")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "sot" / "visdrone2019sot"),
        help="Path to VisDrone2019 SOT dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "sot"),
        help="Where to save SiamFC model checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples-per-epoch", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = ensure_dir(args.output_dir)

    print("=" * 60)
    print("[VisDrone Single Object Tracking (SOT) Model Training]")
    print("=" * 60)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Device             : {device}")
    print(f"Dataset dir        : {args.dataset_dir}")
    print(f"Output dir         : {output_dir}")
    print(f"Epochs             : {args.epochs}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Learning rate      : {args.lr}")
    print(f"Samples per epoch  : {args.samples_per_epoch}")

    dataset = VisDroneSOTDataset(
        dataset_dir=args.dataset_dir,
        samples_per_epoch=args.samples_per_epoch,
    )
    print(f"Loaded dataset     : {len(dataset.sequences)} sequences")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SiamFC().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{args.epochs}", leave=False)

        for batch in pbar:
            template = batch["template"].to(device)
            search = batch["search"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            score_map = model(template, search)
            loss = criterion(score_map, labels)

            loss.backward()
            optimizer.step()

            b_loss = loss.item()
            total_loss += b_loss
            n_batches += 1

            pbar.set_postfix(
                {
                    "batch_loss": f"{b_loss:.4f}",
                    "avg_loss": f"{total_loss / n_batches:.4f}",
                }
            )

        scheduler.step()
        elapsed = time.time() - start_time
        avg_epoch_loss = total_loss / max(n_batches, 1)

        record = {
            "epoch": epoch,
            "loss": avg_epoch_loss,
            "elapsed_s": elapsed,
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss: {avg_epoch_loss:.5f} | "
            f"time: {elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_epoch_loss,
            "config": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"  -> Saved new best SiamFC model (loss: {best_loss:.5f})")

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training Complete! Best loss: {best_loss:.5f}")
    print(f"Model saved to: {output_dir / 'best.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
