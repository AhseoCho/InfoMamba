# InfoMamba

> **InfoMamba: Information-Gated State Space Models for Visual Recognition**
>
> **Resources:** [Model Weights](https://github.com/AhseoCho/InfoMamba/releases/tag/v0.1.0) · [Data Preparation](DATASETS.md) · [Evaluation](src/infomamba_food_eval/evaluate.py) · [Model Card](MODEL_CARD.md)

## About

InfoMamba is an information-gated state space model for visual recognition.
This repository provides released checkpoints and an evaluation implementation
for Food-11 and Food-101.

## Getting Started

### 1. Install Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the InfoMamba/MambaVision inference dependency that provides
`models.mamba_vision:mamba_vision_B`.

### 2. Prepare Data

Prepare the datasets under `artifacts/data/`:

```bash
# Food-101
python scripts/prepare_food101.py \
  --root artifacts/source/food101 \
  --output artifacts/data/Food-101

# Food-11 (after obtaining the dataset from its authorized source)
python scripts/prepare_food11.py \
  --source /path/to/Food-11 \
  --output artifacts/data/Food-11
```

Food-11 uses `train`, `val`, and `test` folders. Food-101 uses the official
`train` and `test` folders. See [DATASETS.md](DATASETS.md) for details.

### 3. Download Checkpoints

Download released checkpoints from the [v0.1.0 release](https://github.com/AhseoCho/InfoMamba/releases/tag/v0.1.0)
and place them under `artifacts/`.

| Dataset | Checkpoint | SHA-256 |
| --- | --- | --- |
| Food-11 | `infomamba_food11.pth.tar` | `5aadc0c8b1b788a905148adb163e38329c8b4c1dc63ff20da8f04d0327a0f2dd` |
| Food-101 | `infomamba_food101.pth.tar` | `0ccd5f2e8cfc0511d4abf8dfe9503a4cbb648aa705aeef82929cfc6e6c914988` |

### 4. Run Evaluation

```bash
# Food-11
python -m infomamba_food_eval.evaluate \
  --config configs/food11.yaml \
  --data artifacts/data/Food-11/test \
  --checkpoint artifacts/infomamba_food11.pth.tar \
  --model-factory models.mamba_vision:mamba_vision_B \
  --output outputs/food11

# Food-101
python -m infomamba_food_eval.evaluate \
  --config configs/food101.yaml \
  --data artifacts/data/Food-101/test \
  --checkpoint artifacts/infomamba_food101.pth.tar \
  --model-factory models.mamba_vision:mamba_vision_B \
  --output outputs/food101
```

## Repository Structure

```text
InfoMamba/
├── classes/                 # fixed class orders
├── configs/                 # evaluation configurations
├── scripts/                 # dataset preparation
├── src/                     # evaluation implementation
├── DATASETS.md
├── MODEL_CARD.md
└── weights_manifest.json
```

## Citation

Citation metadata will be added with the accompanying paper release.

## License

The evaluation code is released under the MIT License. Checkpoint and
third-party dependency notices are documented in [THIRD_PARTY.md](LICENSES/THIRD_PARTY.md).
