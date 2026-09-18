"""KaiHand dataset transforms used only by tactile experiments."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as model_base

KAIHAND_ACTION_DIM = 27


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class KaiHandInputs(transforms.DataTransformFn):
    """Map KaiHand LeRobot fields to π0/π0.5 inputs."""

    model_type: model_base.ModelType

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        base_image = _parse_image(data["observation/image"])
        if "observation/wrist_image" in data:
            wrist_image = _parse_image(data["observation/wrist_image"])
            wrist_mask = np.True_
        else:
            wrist_image = np.zeros_like(base_image)
            wrist_mask = np.False_

        is_fast = self.model_type == model_base.ModelType.PI0_FAST
        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if is_fast else np.False_,
                "right_wrist_0_rgb": np.True_ if is_fast else wrist_mask,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        if "tactile" in data:
            inputs["tactile"] = data["tactile"]
        return inputs


@dataclasses.dataclass(frozen=True)
class KaiHandOutputs(transforms.DataTransformFn):
    """Remove model padding from KaiHand actions."""

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        return {"actions": np.asarray(data["actions"][..., :KAIHAND_ACTION_DIM])}
