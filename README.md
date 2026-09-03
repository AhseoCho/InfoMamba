# InfoMamba

> **InfoMamba: Recovering Information Lost in Hidden-State Compression**

**Resources:** [Algorithm](models/train.py) | [Tasks](tasks/README.md) | [Model card](docs/MODEL_CARD.md)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[food]"
```

## Tasks

Task-specific preparation, training, and evaluation instructions are collected
in [tasks/](tasks/README.md).

## Checkpoints

Checkpoints for Food-11 and Food-101 are distributed through an
[encrypted Baidu Netdisk share](https://pan.baidu.com/s/1n5Q1YrkpnhkT_9cHwlXnYw).
Please contact the authors to obtain the extraction code.

## Evaluation

```bash
# Food-11
python -m infomamba_food_eval.evaluate \
  --config tasks/food/evaluation/configs/food11.yaml \
  --data artifacts/data/Food-11/test \
  --checkpoint artifacts/infomamba_food11.pth.tar \
  --model-factory models.released_food:build_food_model \
  --output outputs/food11

```

## AG News

Training and evaluation commands are in [tasks/agnews/README.md](tasks/agnews/README.md).
AG News checkpoints are distributed separately through the authors' encrypted
Baidu Netdisk share; please contact the authors for access.

The public training entry point uses a 40-epoch budget and retains the
checkpoint selected on the validation split.

## License

InfoMamba-authored code is released under the MIT License. Third-party notices are available in [THIRD_PARTY.md](LICENSES/THIRD_PARTY.md).
