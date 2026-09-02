# InfoMamba

> **InfoMamba: Information-Gated State Space Models for Visual Recognition**
>
> **Resources:** [Models](weights_manifest.json) · [Evaluation](src/infomamba_food_eval/evaluate.py) · [Datasets](DATASETS.md) · [Benchmark Card](MODEL_CARD.md) · [Data Audit](DATA_AUDIT.md)

InfoMamba is an evaluation-only release for visual recognition. This repository
ships verified model artifacts and a deterministic evaluator for Food-11 and
Food-101. It intentionally does not include training code, training recipes, or
datasets.

## Status of this repository

The source package is ready for release, but its `weights_manifest.json` still
contains placeholders. Do **not** publish a numerical claim until every weight
URL and SHA-256 value in that file has been filled from the exact released
artifact and the command below has been rerun in a clean environment.

## About

The release is designed to make every reported result independently checkable:
the evaluator fixes preprocessing and class order, verifies the checkpoint hash,
loads the exact state dict strictly, and emits both aggregate metrics and
per-example predictions. A run fails rather than silently substituting a model,
class mapping, or incompatible checkpoint.

## Getting started

### 1. Install environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the separately distributed InfoMamba/MambaVision inference package that
contains the released architecture. Its model factory is supplied explicitly at
runtime; this repository never silently substitutes another architecture.

### 2. Prepare data

See [DATASETS.md](DATASETS.md) for deterministic Food-11/Food-101 preparation.
Raw images remain outside Git and are placed under `artifacts/data/` locally.

```
Food-11/
  test/<class-name>/*.jpg
Food-101/
  test/<class-name>/*.jpg
```

Food-11 checkpoint selection must use a separate `val/` split. The `test/`
split is for the final evaluation only. For Food-101, `test/` denotes the
official Food-101 evaluation split (25,250 images).

### 3. Obtain a verified model

The release manifest records the artifact URL, SHA-256, exact model factory, and
redistribution terms. A release is not ready until those fields are concrete.
Never use a checkpoint whose hash does not match the manifest.

### 4. Run evaluation

Download a checkpoint only after its manifest entry is complete, then run:

```bash
python -m infomamba_food_eval.evaluate \
  --config configs/food101.yaml \
  --data /path/to/Food-101/test \
  --checkpoint artifacts/infomamba_food101.pth.tar \
  --model-factory models.mamba_vision:mamba_vision_B \
  --output outputs/food101
```

`--model-factory` must resolve to the exact released InfoMamba inference model.
The evaluator passes the configured `num_classes` and rejects a state dict with
any missing or unexpected parameter. It writes `metrics.json`,
`predictions.csv`, and `run_manifest.json`.

## Reporting

Report `top1` and `top5` exactly as written in `metrics.json`; they are distinct
sample-weighted metrics. Include the checkpoint SHA-256, config SHA-256,
dataset split size, and commit hash in any table. See [MODEL_CARD.md](MODEL_CARD.md)
and [DATA_AUDIT.md](DATA_AUDIT.md).

## Repository layout

```
InfoMamba/
├── artifacts/              # downloaded checkpoints; gitignored
├── classes/                # fixed class order
├── configs/                # immutable evaluation protocols
├── src/                    # strict evaluation implementation
├── weights_manifest.json   # artifact identity and integrity metadata
├── MODEL_CARD.md
└── DATA_AUDIT.md
```

## Scope and third-party terms

The MIT license applies only to the new evaluation code in this repository.
Checkpoint redistribution and the external architecture dependency can have
their own terms; record them before publishing. See
[THIRD_PARTY.md](LICENSES/THIRD_PARTY.md).
