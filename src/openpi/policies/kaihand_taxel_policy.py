"""KaiHand transforms for native taxel-conditioned PI0.5."""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    return image


@dataclasses.dataclass(frozen=True)
class KaiHandTaxelInputs(transforms.DataTransformFn):
    """Map two cameras, 27-D state/action, and native taxels to model inputs."""

    def __call__(self, data: dict) -> dict:
        head = _parse_image(data["images"]["head"])
        wrist = _parse_image(data["images"]["right_wrist"])
        taxel_force = np.asarray(data["taxel_force"], dtype=np.float32)
        if taxel_force.shape != (5, 7, 5, 3):
            raise ValueError(f"KaiHand taxel_force must have shape (5, 7, 5, 3), got {taxel_force.shape}")

        result = {
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(head),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": np.asarray(data["state"], dtype=np.float32),
            "taxel_force": taxel_force,
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
        return result


@dataclasses.dataclass(frozen=True)
class KaiHandOutputs(transforms.DataTransformFn):
    action_dim: int = 27

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
