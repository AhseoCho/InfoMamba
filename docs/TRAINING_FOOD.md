# Food fine-tuning

```bash
pip install mambavision
pip install -e .

python training/finetune_food.py \
  --data artifacts/data/Food-11 \
  --output outputs/food11_finetune \
  --backbone mamba_vision_B_21k \
  --epochs 40 --batch-size 32
```

`best.pt` is selected by validation Top-1. The test split is not used during training.

The backbone is provided by [NVlabs/MambaVision](https://github.com/NVlabs/MambaVision); use is subject to its license.
