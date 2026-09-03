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
| Food-11 / Food-101 | [Preparation and evaluation](tasks/food/README.md) |
| AG News | [Training and evaluation](tasks/agnews/README.md) |

Each task page contains its own commands and configuration files.

## Checkpoints

Checkpoints are distributed separately through the authors' encrypted Baidu
Netdisk share. Please contact the authors for access.

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
