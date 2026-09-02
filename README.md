# InfoMamba

> **InfoMamba: Recovering Information Lost in Hidden-State Compression**

**Resources:** [Algorithm](models/train.py) | [Data Preparation](docs/DATASETS.md) | [Evaluation](evaluation) | [Model Card](docs/MODEL_CARD.md)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[food]"
```

## Data Preparation

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

## Model Weights

Checkpoints for Food-11 and Food-101 are distributed through an
[encrypted Baidu Netdisk share](https://pan.baidu.com/s/1n5Q1YrkpnhkT_9cHwlXnYw).
Please contact the authors to obtain the extraction code.

## Evaluation

```bash
# Food-11
python -m infomamba_food_eval.evaluate \
  --config evaluation/food/configs/food11.yaml \
  --data artifacts/data/Food-11/test \
  --checkpoint artifacts/infomamba_food11.pth.tar \
  --model-factory models.released_food:build_food_model \
  --output outputs/food11

```

## License

InfoMamba-authored code is released under the MIT License. Third-party notices are available in [THIRD_PARTY.md](LICENSES/THIRD_PARTY.md).
