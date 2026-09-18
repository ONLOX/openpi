import pathlib

import numpy as np

from openpi.tactile.data.sidecar import TactileSidecarIndex
from openpi.tactile.data.transforms import ConcatTactileState


def test_concat_tactile_state_produces_32_dimensions() -> None:
    transform = ConcatTactileState(q01=(0, 0, 0, 0, 0), q99=(1, 2, 3, 4, 5))
    data = {
        "state": np.zeros(27, dtype=np.float32),
        "tactile": {
            "right_normal_force": np.asarray([0.5, 1, 1.5, 2, 2.5], dtype=np.float32),
        },
    }

    output = transform(data)

    assert output["state"].shape == (32,)
    np.testing.assert_allclose(output["state"][-5:], 0.0, atol=1e-6)


def test_sidecar_index_reads_episode_frame(tmp_path: pathlib.Path) -> None:
    np.savez_compressed(
        tmp_path / "episode_000003.npz",
        frame_index=np.asarray([7]),
        right_normal_force=np.ones((1, 5), dtype=np.float32),
        right_taxel_normal=np.ones((1, 5, 7, 5), dtype=np.float32),
        right_contact=np.ones((1, 5), dtype=np.bool_),
    )

    frame = TactileSidecarIndex(tmp_path).get(3, 7)

    np.testing.assert_array_equal(frame["right_normal_force"], np.ones(5))
    assert frame["right_taxel_normal"].shape == (5, 7, 5)
