"""Download Food-101 via torchvision and materialize the official ImageFolder splits."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from torchvision.datasets import Food101


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


def materialize(root: Path, output: Path, split: str, mode: str) -> int:
    entries = (root / "food-101" / "meta" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    images = root / "food-101" / "images"
    for entry in entries:
        source = images / f"{entry}.jpg"
        if not source.is_file():
            raise FileNotFoundError(source)
        link_or_copy(source, output / split / f"{entry}.jpg", mode)
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Torchvision download cache")
    parser.add_argument("--output", type=Path, required=True, help="Prepared ImageFolder destination")
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    # torchvision owns the canonical download URL/checksum handling.
    Food101(args.root, split="train", download=True)
    Food101(args.root, split="test", download=True)
    train = materialize(args.root, args.output, "train", args.link_mode)
    test = materialize(args.root, args.output, "test", args.link_mode)
    if (train, test) != (75750, 25250):
        raise RuntimeError(f"Unexpected official split counts: train={train}, test={test}")
    print(f"Prepared Food-101: train={train}, test={test}; output={args.output}")


if __name__ == "__main__":
    main()
