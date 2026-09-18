"""Create observation-aligned tactile sidecars for the card dataset."""

from __future__ import annotations

import tyro

from openpi.tactile.data.convert_card import convert_dataset


def main(
    lerobot_root: str = "/nas/chenxianchi/datasets/sim/card/pi05/0914_200",
    raw_dir: str = "/nas/chenxianchi/datasets/sim/card/raw/0914_200",
    manifest_name: str = "kaihand_card_conversion_manifest.json",
    output_dir: str | None = None,
) -> None:
    destination = convert_dataset(
        lerobot_root=lerobot_root,
        raw_dir=raw_dir,
        manifest_name=manifest_name,
        output_dir=output_dir,
    )
    print(f"Wrote tactile sidecars to {destination}")


if __name__ == "__main__":
    tyro.cli(main)
