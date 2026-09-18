"""Standalone tactile experiment configs.

These configs intentionally live outside `openpi.training.config`: tactile
architectures can evolve without adding experiment-specific switches to the
vanilla π0/π0.5 configuration.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import tyro
from typing_extensions import override

from openpi import transforms
from openpi.models import model as model_base
from openpi.models import pi0_config
from openpi.tactile.data import kaihand
from openpi.tactile.data.sidecar import LoadTactileSidecar
from openpi.tactile.data.sidecar import PreserveTactileAfterRepack
from openpi.tactile.data.transforms import ConcatTactileState
from openpi.training import config as training_config


@dataclasses.dataclass(frozen=True)
class CardTactileDataConfig(training_config.DataConfigFactory):
    """Card LeRobot data joined with its observation-aligned tactile sidecar."""

    dataset_root: str = tyro.MISSING
    tactile_sidecar_dir: str = tyro.MISSING
    extra_delta_transform: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: model_base.BaseModelConfig
    ) -> training_config.DataConfig:
        info_path = pathlib.Path(self.tactile_sidecar_dir) / "info.json"
        info = json.loads(info_path.read_text())
        force_stats = info["right_normal_force"]
        concat = ConcatTactileState(
            q01=tuple(force_stats["q01"]),
            q99=tuple(force_stats["q99"]),
        )

        raw_repack = transforms.RepackTransform(
            {
                "observation/image": "observation.images.head",
                "observation/wrist_image": "observation.images.right_wrist",
                "observation/state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            }
        )
        repack = transforms.Group(
            inputs=[
                LoadTactileSidecar(self.tactile_sidecar_dir),
                PreserveTactileAfterRepack(raw_repack),
            ],
        )
        data_transforms = transforms.Group(
            inputs=[kaihand.KaiHandInputs(model_type=model_config.model_type)],
            outputs=[kaihand.KaiHandOutputs()],
        )
        if self.extra_delta_transform:
            delta_mask = transforms.make_bool_mask(7, -20)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(delta_mask)],
                outputs=[transforms.AbsoluteActions(delta_mask)],
            )

        base_model_transforms = training_config.ModelTransformFactory()(model_config)
        model_transforms = transforms.Group(
            inputs=[concat, *base_model_transforms.inputs],
            outputs=base_model_transforms.outputs,
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


_CONFIGS = [
    training_config.TrainConfig(
        name="pi05_kaihand_card_concat",
        project_name="openpi-tactile",
        exp_name="concat",
        model=pi0_config.Pi0Config(
            action_horizon=16,
            pi05=True,
            discrete_state_input=True,
        ),
        data=CardTactileDataConfig(
            repo_id="0914_200",
            dataset_root="/nas/chenxianchi/datasets/sim/card/pi05/0914_200",
            tactile_sidecar_dir="/nas/chenxianchi/datasets/sim/card/pi05/0914_200/tactile",
            assets=training_config.AssetsConfig(
                assets_dir=(
                    "/nas/chenxianchi/datasets/sim/card/pi05/0914_200/checkpoints/"
                    "pi05_kaihand_card_0914_200/card_full_0914_200_fsdp8/5000/assets"
                ),
                asset_id="0914_200",
            ),
            base_config=training_config.DataConfig(prompt_from_task=True),
        ),
        pytorch_weight_path="/oss/yeqianyu/pi05_base",
        checkpoint_base_dir="/oss/yeqianyu/openpi-checkpoints",
        batch_size=32,
        fsdp_devices=8,
    ),
]

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> training_config.TrainConfig:
    return tyro.extras.overridable_config_cli({name: (name, config) for name, config in _CONFIGS_DICT.items()})


def get_config(name: str) -> training_config.TrainConfig:
    return _CONFIGS_DICT[name]
