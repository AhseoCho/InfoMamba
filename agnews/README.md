# AG News

This directory contains the AG News implementation used with the public
Mamba-130M language-model backbone. The concept path applies token-to-concept
routing, concept-space attention, and pre-/post-block residual projections at
layers 5, 11, 17, and 23.

## Install

```bash
pip install -e ".[agnews]"
```

## Train

```bash
python agnews/train.py \
  --output outputs/agnews \
  --base-model state-spaces/mamba-130m-hf \
  --epochs 6 --batch-size 8 --grad-accum 4 --lr 8e-5 \
  --seed 42 --unfreeze-last-n 4 --position-aware-routing
```

The script creates a stratified 6,000-example validation set from AG News
training data, selects the checkpoint by validation accuracy, and evaluates the
selected checkpoint once on the official test set.

## Evaluate a checkpoint

```bash
python agnews/train.py \
  --output outputs/agnews_eval \
  --base-model state-spaces/mamba-130m-hf \
  --checkpoint /path/to/infomamba_agnews_mamba130.pt \
  --evaluate-only --batch-size 8 --seed 42 --position-aware-routing
```

The AG News checkpoint is distributed separately through the authors'
encrypted Baidu Netdisk share. It is an adapter checkpoint and requires the
public `state-spaces/mamba-130m-hf` base model named in the command above.
Please contact the authors for access.
