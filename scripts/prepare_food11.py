"""Validate and materialize an authorized Food-11 ImageFolder copy without redistributing it."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

VALID_SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Existing file differs: {destination}")
        return
    if mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Authorized Food-11 ImageFolder directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    class_orders: list[list[str]] = []
    total = 0
    for split in VALID_SPLITS:
        source_split = args.source / split
        if not source_split.is_dir():
            raise FileNotFoundError(f"Expected split directory: {source_split}")
        classes = sorted(item.name for item in source_split.iterdir() if item.is_dir())
        if len(classes) != 11:
            raise RuntimeError(f"{split} has {len(classes)} classes; expected 11")
        class_orders.append(classes)
        for image in source_split.rglob("*"):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                link_or_copy(image, args.output / split / image.relative_to(source_split), args.link_mode)
                total += 1
    if len({tuple(order) for order in class_orders}) != 1:
        raise RuntimeError("Food-11 class names/order differ between splits")
    print(f"Prepared Food-11: {total} images; output={args.output}")


if __name__ == "__main__":
    main()
