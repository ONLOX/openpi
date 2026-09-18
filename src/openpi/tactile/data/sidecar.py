"""Load observation-aligned tactile arrays without rewriting LeRobot parquet."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import numpy as np

from openpi import transforms
from openpi.tactile.data import schema


@dataclasses.dataclass
class TactileSidecarIndex:
    root: pathlib.Path
    strict: bool = True
    _cache: dict[int, dict[str, Any]] = dataclasses.field(default_factory=dict, repr=False)

    def get(self, episode_index: int, frame_index: int) -> schema.TactileFrame:
        episode = int(episode_index)
        if episode not in self._cache:
            path = self.root / f"episode_{episode:06d}.npz"
            if not path.exists():
                if self.strict:
                    raise FileNotFoundError(f"Missing tactile sidecar: {path}")
                return schema.empty_frame()
            with np.load(path) as data:
                frames = np.asarray(data["frame_index"])
                self._cache[episode] = {
                    "lookup": {int(frame): row for row, frame in enumerate(frames)},
                    "right_normal_force": np.asarray(data["right_normal_force"], dtype=np.float32),
                    "right_taxel_normal": np.asarray(data["right_taxel_normal"], dtype=np.float32),
                    "right_contact": np.asarray(data["right_contact"], dtype=np.bool_),
                }

        packed = self._cache[episode]
        row = packed["lookup"].get(int(frame_index))
        if row is None:
            if self.strict:
                raise KeyError(f"No tactile row for episode={episode}, frame={frame_index}")
            return schema.empty_frame()
        return {
            "right_normal_force": packed["right_normal_force"][row],
            "right_taxel_normal": packed["right_taxel_normal"][row],
            "right_contact": packed["right_contact"][row],
        }


@dataclasses.dataclass(frozen=True)
class LoadTactileSidecar(transforms.DataTransformFn):
    """Attach tactile by `(episode_index, frame_index)` before repacking."""

    sidecar_dir: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", TactileSidecarIndex(pathlib.Path(self.sidecar_dir)))

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        episode = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        frame = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        data["tactile"] = self._index.get(episode, frame)  # type: ignore[attr-defined]
        return data


@dataclasses.dataclass(frozen=True)
class PreserveTactileAfterRepack(transforms.DataTransformFn):
    """Run a normal repack while preserving the sidecar payload."""

    inner: transforms.DataTransformFn

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        tactile = data.get("tactile")
        output = self.inner(data)
        if tactile is not None:
            output["tactile"] = tactile
        return output
