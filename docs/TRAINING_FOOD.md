# Food fine-tuning

```bash
pip install mambavision
pip install -e .
```

Prepare Food-11 so that the data directory contains `train/` and `val/`.

```bash
python training/finetune_food.py \
  --data artifacts/data/Food-11 \
  --output outputs/food11_finetune \
  --backbone mamba_vision_B_21k \
  --epochs 40 --batch-size 32
```

The best checkpoint is selected by validation Top-1 accuracy and saved as `best.pt`. The held-out test split is not used during training.

The visual backbone is provided by [NVlabs/MambaVision](https://github.com/NVlabs/MambaVision). Please follow its license when obtaining pretrained weights or distributing derivatives.
