# AG News Evaluation

Obtain the AG News checkpoint from the [encrypted Baidu Netdisk share](https://pan.baidu.com/s/1n5Q1YrkpnhkT_9cHwlXnYw).
Contact the authors for the extraction code. Obtain the standard AG News test CSV from an authorized dataset source. The file
must contain `Class Index`, `Title`, and `Description` columns.

```bash
python evaluation/agnews/evaluate.py \
  --data /path/to/test.csv \
  --checkpoint artifacts/infomamba_agnews.pt \
  --output outputs/agnews
```

The command writes `metrics.json` locally. No training code, training recipe,
or benchmark result is included in this repository.
