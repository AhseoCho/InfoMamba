# InfoMamba

> **Recovering Information Lost in Hidden-State Compression**

InfoMamba provides task-specific resources for Food-11, Food-101, and AG News.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[food,agnews]"
```

## Tasks

| Task | Resources |
| --- | --- |
| Food-11 / Food-101 | [Preparation, training, and evaluation](tasks/food/README.md) |
| AG News | [Training and evaluation](tasks/agnews/README.md) |

Each task page contains its own commands and configuration files.

## Food benchmarks

Food-11 and Food-101 preparation, training, and evaluation instructions are
available in [tasks/food](tasks/food/README.md). The training reference is in
[models/train.py](models/train.py), with the Food-11 configuration in
[tasks/food/training_food11.yaml](tasks/food/training_food11.yaml).

Evaluate a released Food checkpoint with:

```bash
python -m infomamba_food_eval.evaluate \
  --config tasks/food/evaluation/configs/food11.yaml \
  --data artifacts/data/Food-11/test \
  --checkpoint /path/to/infomamba_food11.pth \
  --model-factory models.released_food:build_food_model \
  --output outputs/food11_eval
```

## Checkpoints

Checkpoints are available from [Baidu Netdisk](https://pan.baidu.com/s/1n5Q1YrkpnhkT_9cHwlXnYw).
For the extraction code, please contact the authors.

## Repository structure

```text
tasks/
  food/       Food-11 and Food-101 resources
  agnews/     AG News resources
models/       Algorithm pseudocode and released model factory
src/          Evaluation package
LICENSES/     Third-party notices
```

## License

InfoMamba-authored code is released under the MIT License. See
[THIRD_PARTY.md](LICENSES/THIRD_PARTY.md) for external dependencies and data
terms.
