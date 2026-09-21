"""Offline evaluation helpers for tactile model variants."""

from __future__ import annotations

import pathlib

import jax
import torch

from openpi.training import config as training_config


@torch.no_grad()
def evaluate_flow_loss(
    config: training_config.TrainConfig,
    checkpoint_dir: str | pathlib.Path,
    *,
    max_batches: int = 10,
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate mean flow-matching loss on deterministic dataset batches."""

    checkpoint = pathlib.Path(checkpoint_dir)
    weight_path = checkpoint / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(weight_path)

    from openpi.training import data_loader

    torch_device = torch.device(device)
    model = config.model.load_pytorch(config, str(weight_path)).to(torch_device)
    model.eval()
    loader = data_loader.create_data_loader(config, framework="pytorch", shuffle=False)

    total = 0.0
    batches = 0
    for observation, actions in loader:
        observation = jax.tree.map(lambda value: value.to(torch_device), observation)
        actions = actions.to(device=torch_device, dtype=torch.float32)
        total += float(model(observation, actions).mean())
        batches += 1
        if batches >= max_batches:
            break

    if batches == 0:
        raise ValueError("Evaluation data loader produced no batches")
    return {"flow_loss": total / batches, "batches": float(batches)}
