# InfoMamba: Information-Gated State Space Models for Visual Recognition

## Intended use

Image classification evaluation on Food-11 and Food-101 only. This repository
does not provide training code and must not be used to infer a training recipe.

## Metrics

`top1` is the percentage of examples whose highest-logit class is correct.
`top5` is the percentage whose target appears among the five highest-logit
classes. Both are computed across all examples, not averaged across batches.

## Release checklist

- [ ] Released checkpoint assets are available from the GitHub Release.
- [ ] Exact architecture and inference dependency version recorded.
- [ ] Clean-environment command produces `metrics.json` and `predictions.csv`.
- [ ] Dataset split count and class ordering match the config.
- [ ] Food-101 duplicate-file decision and result are disclosed in `DATA_AUDIT.md`.
- [ ] Third-party model and checkpoint redistribution terms have been reviewed.

No result should be labelled as an InfoMamba result unless it was generated with
the released InfoMamba checkpoint and this evaluator.
