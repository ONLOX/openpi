"""Canonical tactile tensors shared by conversion and tactile model variants."""

from typing import TypedDict

import numpy as np

RIGHT_FINGER_SLICE = slice(5, 10)
NUM_RIGHT_FINGERS = 5
TAXEL_H = 7
TAXEL_W = 5
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


class TactileFrame(TypedDict):
    """One observation-aligned KaiHand tactile sample."""

    right_normal_force: np.ndarray  # (5,) float32, Newtons
    right_taxel_normal: np.ndarray  # (5, 7, 5) float32, Newtons
    right_contact: np.ndarray  # (5,) bool


def empty_frame() -> TactileFrame:
    return {
        "right_normal_force": np.zeros((NUM_RIGHT_FINGERS,), dtype=np.float32),
        "right_taxel_normal": np.zeros((NUM_RIGHT_FINGERS, TAXEL_H, TAXEL_W), dtype=np.float32),
        "right_contact": np.zeros((NUM_RIGHT_FINGERS,), dtype=np.bool_),
    }
