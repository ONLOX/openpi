"""Train the tactile state-concat baseline.

Example:
    torchrun --standalone --nproc-per-node=8 \
        scripts/tactile/train.py pi05_kaihand_card_concat \
        --exp-name concat-v1 \
        --checkpoint-base-dir /oss/yeqianyu/openpi-checkpoints
"""

import os
import pathlib

from openpi.tactile import config


if __name__ == "__main__":
    train_config = config.cli()
    dataset_root = pathlib.Path(train_config.data.dataset_root)  # type: ignore[attr-defined]
    os.environ["HF_LEROBOT_HOME"] = str(dataset_root.parent)
    from scripts import train_pytorch

    train_pytorch.train_loop(train_config)
