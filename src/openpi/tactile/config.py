"""Training configs for the tactile state-concatenation baseline."""

from __future__ import annotations

import tyro

from openpi.models import pi0_config
from openpi.tactile.data import config as tactile_data_config
from openpi.training import config as training_config


_CONFIGS = [
    training_config.TrainConfig(
        name="pi05_kaihand_usb_concat",
        project_name="openpi-tactile",
        exp_name="usb-concat",
        model=pi0_config.Pi0Config(
            action_horizon=20,
            pi05=True,
            discrete_state_input=True,
        ),
        data=tactile_data_config.KaiHandTactileDataConfig(
            repo_id="kaihand/usb_insert_0920_200",
            dataset_root="/nas/chenxianchi/datasets/sim/usb_insert/lerobot_v3/0920_200",
        ),
        pytorch_weight_path="/nas/yeqianyu/checkpoints/pi05_base",
        checkpoint_base_dir="/nas/yeqianyu/checkpoints/openpi",
        batch_size=64,
        num_workers=4,
        num_train_steps=30_000,
        log_interval=100,
        save_interval=1_000,
    ),
    training_config.TrainConfig(
        name="pi05_kaihand_poker_concat",
        project_name="openpi-tactile",
        exp_name="poker-concat",
        model=pi0_config.Pi0Config(
            action_horizon=20,
            pi05=True,
            discrete_state_input=True,
        ),
        data=tactile_data_config.KaiHandTactileDataConfig(
            repo_id="kaihand/poker_draw_0920_200",
            dataset_root="/nas/chenxianchi/datasets/sim/poker-draw/lerobot_v3/0920_200",
        ),
        pytorch_weight_path="/nas/yeqianyu/checkpoints/pi05_base",
        checkpoint_base_dir="/nas/yeqianyu/checkpoints/openpi",
        batch_size=64,
        num_workers=4,
        num_train_steps=30_000,
        log_interval=100,
        save_interval=1_000,
    ),
]

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> training_config.TrainConfig:
    return tyro.extras.overridable_config_cli({name: (name, config) for name, config in _CONFIGS_DICT.items()})


def get_config(name: str) -> training_config.TrainConfig:
    return _CONFIGS_DICT[name]
