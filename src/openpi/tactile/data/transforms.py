"""Data transforms for explicit tactile model variants."""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms

NUM_RIGHT_FINGERS = 5


@dataclasses.dataclass(frozen=True)
class ClipNormalizedStateAndActions(transforms.DataTransformFn):
    """Clip quantile-normalized robot values to their intended range."""

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        data["state"] = np.clip(np.asarray(data["state"], dtype=np.float32), -1.0, 1.0)
        if "actions" in data:
            data["actions"] = np.clip(np.asarray(data["actions"], dtype=np.float32), -1.0, 1.0)
        return data


@dataclasses.dataclass(frozen=True)
class ConcatTactileState(transforms.DataTransformFn):
    """Append normalized five-finger force to the normalized 27-d robot state.

    This transform must run after the robot-state normalization and before
    tokenization/padding. The resulting state has exactly 32 dimensions.
    """

    q01: tuple[float, ...]
    q99: tuple[float, ...]
    joint_dim: int = 27

    def __post_init__(self) -> None:
        if len(self.q01) != NUM_RIGHT_FINGERS or len(self.q99) != NUM_RIGHT_FINGERS:
            raise ValueError("Tactile quantiles must contain one value per right-hand finger")

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        tactile = data.get("tactile")
        if tactile is None:
            raise ValueError("ConcatTactileState requires a tactile sample")

        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] < self.joint_dim:
            raise ValueError(f"Expected at least {self.joint_dim} state dimensions, got {state.shape[-1]}")

        force = np.asarray(tactile["right_normal_force"], dtype=np.float32)
        q01 = np.asarray(self.q01, dtype=np.float32)
        q99 = np.asarray(self.q99, dtype=np.float32)
        normalized_force = (force - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        normalized_force = np.clip(normalized_force, -1.0, 1.0)
        data["state"] = np.concatenate([state[..., : self.joint_dim], normalized_force], axis=-1)
        return data
