# InfoMamba

> **InfoMamba: Information-Gated State Space Models for Visual Recognition**
>
> **Resources:** [Model Weights](https://github.com/AhseoCho/InfoMamba/releases) · [Data Preparation](docs/DATASETS.md) · [Evaluation](evaluation) · [Model Card](docs/MODEL_CARD.md)

## About

InfoMamba provides released checkpoints and concise, reproducible evaluation
workflows for visual and sequence classification tasks.

## Getting Started

### 1. Install Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the compatible inference dependencies required by the released
checkpoints.

### 2. Prepare Data

```bash
# Food-101
python scripts/prepare_food101.py \
  --root artifacts/source/food101 \
  --output artifacts/data/Food-101

# Food-11
python scripts/prepare_food11.py \
  --source /path/to/Food-11 \
  --output artifacts/data/Food-11
```

See [Dataset Preparation](docs/DATASETS.md) for the complete layouts.

### 3. Download Checkpoints

Download the required checkpoint from [Releases](https://github.com/AhseoCho/InfoMamba/releases)
and place it under `artifacts/`.

### 4. Run Evaluation

```bash
# Food-11
python -m infomamba_food_eval.evaluate \
  --config evaluation/food/configs/food11.yaml \
  --data artifacts/data/Food-11/test \
  --checkpoint artifacts/infomamba_food11.pth.tar \
  --model-factory <released-model-factory> \
  --output outputs/food11

# AG News
python evaluation/agnews/evaluate.py \
  --data /path/to/test.csv \
  --checkpoint artifacts/infomamba_agnews.pt \
  --output outputs/agnews
```

## Repository Structure

```text
InfoMamba/
├── docs/                    # usage and release documentation
├── evaluation/
│   ├── food/                # class orders and evaluation configurations
│   └── agnews/              # sequence evaluation assets
├── scripts/                 # dataset preparation
├── src/                     # shared evaluation utilities
├── LICENSE
└── pyproject.toml
```

## Food Fine-tuning

See [Food Fine-tuning](docs/TRAINING_FOOD.md).

## Citation

Citation metadata will be added with the accompanying paper release.

## License

The evaluation code is released under the MIT License. Third-party notices are
available in [THIRD_PARTY.md](LICENSES/THIRD_PARTY.md).
