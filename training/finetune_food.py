"""Fine-tune InfoMamba for Food classification."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from training.infomamba_food import InfoMambaFoodClassifier
except ModuleNotFoundError:
    from infomamba_food import InfoMambaFoodClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing train/ and val/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="mamba_vision_B_21k")
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--concepts", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        targets = targets.to(device, non_blocking=True)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += targets.numel()
    return correct / total


def main() -> None:
    args = parse_args()
    if not (args.data / "train").is_dir() or not (args.data / "val").is_dir():
        raise FileNotFoundError("--data must contain train/ and val/ ImageFolder directories")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), transforms.RandomErasing(p=0.25),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_set, val_set = datasets.ImageFolder(args.data / "train", train_tf), datasets.ImageFolder(args.data / "val", val_tf)
    if train_set.classes != val_set.classes:
        raise RuntimeError("train/ and val/ class orders differ")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = InfoMambaFoodClassifier(args.backbone, len(train_set.classes), args.concepts, args.temperature,
                                    pretrained=True, checkpoint_path=args.pretrained_checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = samples = 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = nn.functional.cross_entropy(model(images), targets, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * targets.numel()
            samples += targets.numel()
        val_top1 = evaluate(model, val_loader, device)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": loss_sum / samples, "val_top1": val_top1}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_top1 > best:
            best = val_top1
            torch.save({"state_dict": model.state_dict(), "classes": train_set.classes, "args": vars(args),
                        "epoch": epoch, "val_top1": val_top1}, args.output / "best.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
