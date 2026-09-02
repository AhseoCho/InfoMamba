# Dataset and split audit

## Required release checks

1. Record the upstream dataset version, official split source, and file counts.
2. Save a SHA-256 manifest of every evaluated file outside this repository.
3. Check train/validation/test splits for byte-identical files and perceptual near-duplicates.
4. Select checkpoints on Food-11 validation data only; evaluate Food-11 test exactly once.
5. For Food-101, use the official evaluation split and call it `val` only as a directory name.

## Food-101 duplicate-file disclosure

An internal audit found two byte-identical train/evaluation pairs in the local
Food-101 copy. Both pairs have the same label. This is not enough to materially
explain a high aggregate score, but it must be resolved before claiming a clean
benchmark result: either filter the affected evaluation files with a documented
rule or publish the pair list and report both original-split and filtered-split
metrics.

Do not replace this section with an unsupported statement that the dataset is
free of duplicates.
