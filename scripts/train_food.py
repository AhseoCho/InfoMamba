"""Train InfoMamba for Food classification."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.infomamba import InfoMambaFoodClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="mamba_vision_B_21k")
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, targets in loader:
        targets = targets.to(device, non_blocking=True)
        correct += int((model(images.to(device, non_blocking=True)).argmax(1) == targets).sum())
        total += targets.numel()
    return correct / total


def main() -> None:
    args = parse_args()
    if not (args.data / "train").is_dir() or not (args.data / "val").is_dir():
        raise FileNotFoundError("--data must contain train/ and val/")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.RandAugment(2, 9), transforms.ToTensor(), normalize, transforms.RandomErasing(p=0.25),
    ])
    val_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    train_set = datasets.ImageFolder(args.data / "train", train_tf)
    val_set = datasets.ImageFolder(args.data / "val", val_tf)
    if train_set.classes != val_set.classes:
        raise RuntimeError("train/ and val/ class orders differ")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda")
    model = InfoMambaFoodClassifier(args.backbone, len(train_set.classes), True, args.pretrained_checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=5e-6)
    scaler = torch.amp.GradScaler("cuda")
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = samples = 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = nn.functional.cross_entropy(model(images), targets, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * targets.numel()
            samples += targets.numel()
        top1 = validate(model, val_loader, device)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": loss_sum / samples, "val_top1": top1}
        history.append(record)
        print(json.dumps(record), flush=True)
        if top1 > best:
            best = top1
            torch.save({"state_dict": model.state_dict(), "classes": train_set.classes,
                        "args": vars(args), "epoch": epoch, "val_top1": top1}, args.output / "best.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
