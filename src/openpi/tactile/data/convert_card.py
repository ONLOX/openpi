"""Convert card raw HDF5 tactile streams into LeRobot-aligned sidecars."""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import pyarrow.parquet as pq

from openpi.tactile.data import schema


def convert_episode(
    *,
    parquet_path: pathlib.Path,
    hdf5_path: pathlib.Path,
    output_path: pathlib.Path,
) -> dict:
    table = pq.read_table(parquet_path, columns=["frame_index", "provenance.source_state_index"])
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    state_index = np.asarray(table["provenance.source_state_index"].to_numpy(), dtype=np.int64)

    with h5py.File(hdf5_path, "r") as h5:
        force = np.asarray(h5["tactile_contact_force/normal_force_n"], dtype=np.float32)
        taxel = np.asarray(h5["tactile_contact_force/normal_taxel_force_n"], dtype=np.float32)
        contact = np.asarray(h5["tactile_proxy/contact"], dtype=np.bool_)

    if np.any(state_index < 0) or np.any(state_index >= force.shape[0]):
        raise IndexError(f"{hdf5_path}: source_state_index is outside [0, {force.shape[0]})")

    payload = {
        "frame_index": frame_index,
        "source_state_index": state_index,
        "right_normal_force": force[state_index, schema.RIGHT_FINGER_SLICE],
        "right_taxel_normal": taxel[state_index, schema.RIGHT_FINGER_SLICE],
        "right_contact": contact[state_index, schema.RIGHT_FINGER_SLICE],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        "n_frames": int(len(frame_index)),
        "n_state": int(force.shape[0]),
        "normal_mean": payload["right_normal_force"].mean(axis=0).tolist(),
        "normal_max": payload["right_normal_force"].max(axis=0).tolist(),
        "contact_frac": payload["right_contact"].mean(axis=0).tolist(),
        "npz": str(output_path),
    }


def convert_dataset(
    *,
    lerobot_root: str,
    raw_dir: str,
    manifest_name: str = "kaihand_card_conversion_manifest.json",
    output_dir: str | None = None,
) -> pathlib.Path:
    root = pathlib.Path(lerobot_root)
    raw = pathlib.Path(raw_dir)
    destination = pathlib.Path(output_dir) if output_dir else root / "tactile"
    manifest = json.loads((root / "meta" / manifest_name).read_text())

    summaries = []
    all_force = []
    for episode in manifest["episodes"]:
        episode_index = int(episode["output_episode_index"])
        output_path = destination / f"episode_{episode_index:06d}.npz"
        summary = convert_episode(
            parquet_path=root / episode["parquet_path"],
            hdf5_path=raw / episode["hdf5"],
            output_path=output_path,
        )
        summary["output_episode_index"] = episode_index
        summary["hdf5"] = episode["hdf5"]
        summaries.append(summary)
        with np.load(output_path) as data:
            all_force.append(np.asarray(data["right_normal_force"]))

    force = np.concatenate(all_force, axis=0)
    info = {
        "schema_version": "kaihand-card-tactile-sidecar-v1",
        "source": "tactile_contact_force indexed by provenance.source_state_index",
        "right_fingers": list(schema.FINGER_NAMES),
        "right_normal_force": {
            "shape": [schema.NUM_RIGHT_FINGERS],
            "unit": "N",
            "mean": force.mean(axis=0).tolist(),
            "std": force.std(axis=0).tolist(),
            "q01": np.quantile(force, 0.01, axis=0).tolist(),
            "q99": np.quantile(force, 0.99, axis=0).tolist(),
        },
        "episodes": summaries,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "info.json").write_text(json.dumps(info, indent=2))
    return destination
