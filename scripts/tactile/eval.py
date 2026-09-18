"""Evaluate a tactile checkpoint with flow-matching loss."""

from __future__ import annotations

import dataclasses

import tyro

from openpi.tactile import config
from openpi.tactile.evaluation import evaluate_flow_loss


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str
    checkpoint_dir: str
    max_batches: int = 10
    device: str = "cuda"


def main(args: Args) -> None:
    result = evaluate_flow_loss(
        config.get_config(args.config_name),
        args.checkpoint_dir,
        max_batches=args.max_batches,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main(tyro.cli(Args))
