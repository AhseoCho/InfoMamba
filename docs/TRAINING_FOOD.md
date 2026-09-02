# Food fine-tuning

This repository provides an independently written reference fine-tuning implementation for Food-11. It imports the visual backbone as an external dependency; no source files from that project are included here.

## Install the external backbone

```bash
pip install mambavision
pip install -e .
```

The backbone source and its license are maintained upstream at [NVlabs/MambaVision](https://github.com/NVlabs/MambaVision). Review and comply with that license before downloading a pretrained checkpoint or distributing a fine-tuned derivative.

## Run Food-11 fine-tuning

Prepare authorized Food-11 data first. `--data` must contain `train/` and `val/` ImageFolder splits. The test split is intentionally not read by this training command.

```bash
python training/finetune_food.py \
  --data artifacts/data/Food-11 \
  --output outputs/food11_finetune \
  --backbone mamba_vision_B_21k \
  --epochs 40 --batch-size 32
```

The command loads the public pretrained backbone, adds this repository's concept readout, and fine-tunes all parameters. It writes `best.pt` according to validation Top-1 accuracy and `history.json`. Evaluate that selected checkpoint once on the held-out test split.

## Scope

This is a clean reference implementation, not a claim of binary or architectural compatibility with historical release checkpoints. It does not reproduce, copy, or redistribute external backbone source, upstream weights, Food-11 images, private paths, or experiment logs.
