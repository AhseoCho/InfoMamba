# Dataset preparation

The repository does not redistribute Food-11 or Food-101 image files. Download
or obtain each dataset under its upstream terms, then prepare a deterministic
ImageFolder view locally.

## Food-101

`torchvision.datasets.Food101` downloads the canonical dataset and exposes the
official `train` and `test` splits. This release materializes them as ImageFolder
directories without changing their memberships:

```bash
python tasks/food/prepare_food101.py \
  --root artifacts/source/food101 \
  --output artifacts/data/Food-101
```

The expected counts are 75,750 training and 25,250 test images. For the public
evaluation command, pass `artifacts/data/Food-101/test` as `--data`.

## Food-11

Obtain Food-11 from an authorized upstream source and organize it as
`train/`, `val/`, and `test/`, with eleven class folders in every split. Then:

```bash
python tasks/food/prepare_food11.py \
  --source /path/to/authorized/Food-11 \
  --output artifacts/data/Food-11
```

The script checks that all three splits have the same eleven class names and
uses hard links by default. Use `--link-mode copy` where hard links are not
available.

## Training

The compact InfoMamba training reference is
[`models/train.py`](../../models/train.py). The Food-11 configuration is
[`training_food11.yaml`](training_food11.yaml). It specifies the backbone,
optimization settings, and 40-epoch schedule used for the Food experiment.

## Evaluation

```bash
python -m infomamba_food_eval.evaluate \
  --config tasks/food/evaluation/configs/food11.yaml \
  --data artifacts/data/Food-11/test \
  --checkpoint /path/to/infomamba_food11.pth \
  --model-factory models.released_food:build_food_model \
  --output outputs/food11_eval
```

## Integrity and benchmark rules

- Do not select a Food-11 checkpoint using the test split.
- Keep raw files outside Git; `artifacts/` is intentionally ignored.
- Record the prepared split counts and hashes in the released run manifest.
