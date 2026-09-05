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

| Resource | Download | Access |
| --- | --- | --- |
| InfoMamba checkpoints | [Baidu Netdisk](https://pan.baidu.com/s/1n5Q1YrkpnhkT_9cHwlXnYw) | Extraction code available from the authors. |
| Food-11 / Food-101 data preparation | [Task resources](tasks/food/README.md) | Follow the dataset preparation instructions. |
| AG News data preparation | [Task resources](tasks/agnews/README.md) | Follow the dataset preparation instructions. |

The Netdisk share is password-protected. Please contact the authors to obtain
the extraction code; do not publish it in issues, forks, or mirrors.

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
