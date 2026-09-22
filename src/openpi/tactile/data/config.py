"""Data configs for tactile fields embedded in LeRobot datasets."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import tyro
from typing_extensions import override

from openpi import transforms
from openpi.models import model as model_base
from openpi.shared import normalize
from openpi.tactile.data import kaihand
from openpi.tactile.data.transforms import ClipNormalizedStateAndActions
from openpi.tactile.data.transforms import ConcatTactileState
from openpi.training import config as training_config


def _norm_stats(raw_stats: dict, key: str) -> normalize.NormStats:
    stats = raw_stats[key]
    return normalize.NormStats(
        mean=np.asarray(stats["mean"], dtype=np.float32),
        std=np.asarray(stats["std"], dtype=np.float32),
        q01=np.asarray(stats["q01"], dtype=np.float32),
        q99=np.asarray(stats["q99"], dtype=np.float32),
    )


@dataclasses.dataclass(frozen=True)
class KaiHandTactileDataConfig(training_config.DataConfigFactory):
    """Append embedded KaiHand five-finger normal force to robot state."""

    dataset_root: str = tyro.MISSING

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: model_base.BaseModelConfig
    ) -> training_config.DataConfig:
        del assets_dirs
        dataset_root = pathlib.Path(self.dataset_root)
        stats_path = dataset_root / "meta" / "stats.json"
        raw_stats = json.loads(stats_path.read_text())
        tactile_stats = raw_stats["observation.tactile.right.normal_force"]

        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.head",
                        "observation/wrist_image": "observation.images.right_wrist",
                        "observation/state": "observation.state",
                        "tactile": {
                            "right_normal_force": "observation.tactile.right.normal_force",
                        },
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[kaihand.KaiHandInputs(model_type=model_config.model_type)],
            outputs=[kaihand.KaiHandOutputs()],
        )
        base_model_transforms = training_config.ModelTransformFactory()(model_config)
        model_transforms = transforms.Group(
            inputs=[
                ClipNormalizedStateAndActions(),
                ConcatTactileState(
                    q01=tuple(tactile_stats["q01"]),
                    q99=tuple(tactile_stats["q99"]),
                ),
                *base_model_transforms.inputs,
            ],
            outputs=base_model_transforms.outputs,
        )

        return training_config.DataConfig(
            repo_id=self.repo_id,
            lerobot_root=str(dataset_root),
            asset_id=self.repo_id,
            norm_stats={
                "state": _norm_stats(raw_stats, "observation.state"),
                "actions": _norm_stats(raw_stats, "action"),
            },
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=True,
            action_sequence_keys=("action",),
            prompt_from_task=True,
        )
