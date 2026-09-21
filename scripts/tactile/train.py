"""Train the tactile state-concat baseline.

Example:
    torchrun --standalone --nproc-per-node=8 \
        scripts/tactile/train.py pi05_kaihand_usb_concat \
        --checkpoint-base-dir /oss/yeqianyu/openpi-checkpoints
"""

from openpi.tactile import config


if __name__ == "__main__":
    train_config = config.cli()
    from scripts import train_pytorch

    train_pytorch.train_loop(train_config)
