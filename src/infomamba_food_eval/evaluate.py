"""Deterministic, strict ImageFolder evaluator for released InfoMamba weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"dataset", "num_classes", "image_size", "resize_size", "mean", "std", "classes_file"}
    missing = required - set(config or {})
    if missing:
        raise ValueError(f"Configuration is missing: {sorted(missing)}")
    return config


def load_classes(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError("Class list has duplicate names")
    return names


def resolve_factory(spec: str):
    try:
        module_name, symbol = spec.split(":", 1)
        factory = getattr(importlib.import_module(module_name), symbol)
    except (ValueError, ImportError, AttributeError) as error:
        raise RuntimeError(f"Cannot resolve --model-factory {spec!r}") from error
    if not callable(factory):
        raise TypeError(f"Model factory {spec!r} is not callable")
    return factory


def safe_load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load tensor-only checkpoints; never unpickle arbitrary release files."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("PyTorch >= 2.1 is required for safe checkpoint loading") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a tensor state dictionary or a dictionary containing one")
    for key in ("state_dict", "model", "module"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in checkpoint.items()):
        raise ValueError("Checkpoint contains non-tensor entries; publish a tensor-only state_dict")
    return {key.removeprefix("module."): value for key, value in checkpoint.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="ImageFolder split directory")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-factory", required=True, help="Exact factory, e.g. package.module:function")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    class_path = args.config.parent.parent / config["classes_file"]
    expected_classes = load_classes(class_path)
    if len(expected_classes) != config["num_classes"]:
        raise ValueError("Configured num_classes differs from the released class list")
    if not args.data.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("Dataset split directory or checkpoint does not exist")

    checkpoint_hash = sha256(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_hash != args.expected_checkpoint_sha256.lower():
        raise ValueError("Checkpoint SHA-256 mismatch")

    transform = transforms.Compose([
        transforms.Resize(config["resize_size"], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["mean"], std=config["std"]),
    ])
    dataset = datasets.ImageFolder(args.data, transform=transform)
    if dataset.classes != expected_classes:
        raise ValueError(
            "Dataset class order differs from the released class list. "
            f"Expected {expected_classes[:3]}..., received {dataset.classes[:3]}..."
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"), persistent_workers=args.workers > 0)

    model = resolve_factory(args.model_factory)(num_classes=config["num_classes"])
    incompatible = model.load_state_dict(safe_load_state_dict(args.checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:  # defensive for non-standard modules
        raise RuntimeError(f"State-dict mismatch: {incompatible}")
    device = torch.device(args.device)
    model.to(device).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    correct1 = correct5 = total = 0
    rows: list[list[Any]] = []
    with torch.inference_mode():
        for images, target in loader:
            logits = model(images.to(device, non_blocking=True))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.ndim != 2 or logits.shape[1] != config["num_classes"]:
                raise RuntimeError(f"Model emitted invalid logits shape {tuple(logits.shape)}")
            top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices.cpu()
            target_cpu = target.cpu()
            correct1 += int((top5[:, 0] == target_cpu).sum())
            correct5 += int((top5 == target_cpu[:, None]).any(dim=1).sum())
            start = total
            for offset, predicted in enumerate(top5.tolist()):
                index = start + offset
                rows.append([index, dataset.samples[index][0], int(target_cpu[offset]), expected_classes[int(target_cpu[offset])], predicted[0], expected_classes[predicted[0]], "|".join(map(str, predicted))])
            total += len(target)

    metrics = {"dataset": config["dataset"], "samples": total, "top1": 100 * correct1 / total,
               "top5": 100 * correct5 / total, "checkpoint_sha256": checkpoint_hash,
               "config_sha256": sha256(args.config), "model_factory": args.model_factory,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
               "python": sys.version, "platform": platform.platform()}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "path", "target", "target_name", "top1", "top1_name", "top5_indices"])
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
