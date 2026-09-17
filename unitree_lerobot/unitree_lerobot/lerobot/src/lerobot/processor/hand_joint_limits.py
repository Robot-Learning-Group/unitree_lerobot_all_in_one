# 2026/09/16  阪大側で追加したコード。ハンドの関節角を-1〜1の範囲で正規化して学習する。

"""Joint-limit normalization for named G1 Dex3 and BrainCo observations."""

from typing import Any

import torch


# Radians, from eval_robot/assets/unitree_hand/unitree_dex3_{left,right}.urdf.
# Names follow unitree_lerobot/utils/constants.py, not URDF traversal order.
DEX3_LIMITS = {}
for side, thumb1, thumb2, finger0, finger1 in (
    ("Left", (-0.72431163, 0.920), (0.0, 1.74532925), (-1.57079632, 0.0), (-1.74532925, 0.0)),
    ("Right", (-0.920, 0.72431163), (-1.74532925, 0.0), (0.0, 1.57079632), (0.0, 1.74532925)),
):
    DEX3_LIMITS[f"k{side}HandThumb0"] = (-1.04719755, 1.04719755)
    DEX3_LIMITS[f"k{side}HandThumb1"] = thumb1
    DEX3_LIMITS[f"k{side}HandThumb2"] = thumb2
    for finger in ("Index", "Middle"):
        DEX3_LIMITS[f"k{side}Hand{finger}0"] = finger0
        DEX3_LIMITS[f"k{side}Hand{finger}1"] = finger1

# BrainCo DDS state positions are already in [0, 1], not URDF radians.
BRAINCO_LIMITS = {
    f"k{side}Hand{finger}": (0.0, 1.0)
    for side in ("Left", "Right")
    for finger in ("Thumb", "ThumbAux", "Index", "Middle", "Ring", "Pinky")
}


def resolve_hand_joint_limits(features: dict[str, Any]) -> dict[str, Any]:
    """Resolve a complete pair of hands from dataset state names, preserving their order."""
    feature = features.get("observation.state", {})
    names = feature.get("names")
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError("Hand normalization requires observation.state joint names in dataset metadata.")
    if len(set(names)) != len(names):
        raise ValueError("Hand normalization requires unique observation.state joint names.")
    if tuple(feature.get("shape", ())) != (len(names),):
        raise ValueError("observation.state shape does not match its joint names.")

    hand_names = {name for name in names if "Hand" in name}
    for hand_type, limits in (("dex3", DEX3_LIMITS), ("brainco", BRAINCO_LIMITS)):
        if hand_names == set(limits):
            indices = [i for i, name in enumerate(names) if name in limits]
            return {
                "hand_type": hand_type,
                "state_names": list(names),
                "indices": indices,
                "lower": [limits[names[i]][0] for i in indices],
                "upper": [limits[names[i]][1] for i in indices],
            }
    raise ValueError("Hand normalization requires a complete Dex3 or BrainCo joint-name set.")


def normalize_hand_observations(
    raw: torch.Tensor, normalized: torch.Tensor, limits: dict[str, Any] | None
) -> torch.Tensor:
    """Replace only hand coordinates; keep existing arm normalization and do not clip."""
    if limits is None:
        return normalized
    if raw.shape[-1] != len(limits["state_names"]):
        raise ValueError("Observation state dimension does not match saved hand joint limits.")
    indices = limits["indices"]
    lower = raw.new_tensor(limits["lower"])
    upper = raw.new_tensor(limits["upper"])
    result = normalized.clone()
    result[..., indices] = 2 * (raw[..., indices] - lower) / (upper - lower) - 1
    return result


def configure_hand_normalization(
    preprocessor, *, enabled: bool, dataset_meta=None, ee: str | None = None, loaded: bool = False
):
    """Bind or validate saved limits once, at pipeline creation, before robot startup."""
    steps = [step for step in preprocessor.steps if hasattr(step, "hand_joint_limits")]
    saved = [step.hand_joint_limits for step in steps if step.hand_joint_limits is not None]
    if not enabled:
        if saved:
            raise ValueError("Saved hand normalization is enabled; cannot disable it when loading a checkpoint.")
        return
    if len(steps) != 1:
        raise ValueError("Hand normalization requires exactly one supported state normalizer.")
    step = steps[0]
    expected = resolve_hand_joint_limits(dataset_meta.features) if dataset_meta is not None else None
    if step.hand_joint_limits is None:
        if loaded:
            raise ValueError("Checkpoint has no saved hand joint limits. Train with hand normalization enabled.")
        if expected is None:
            raise ValueError("Hand normalization requires dataset metadata when creating a new processor.")
        step.hand_joint_limits = expected
    limits = step.hand_joint_limits
    if expected is not None and limits["state_names"] != expected["state_names"]:
        raise ValueError("Saved hand joint names do not match the dataset.")
    if ee is not None and ee != limits["hand_type"]:
        raise ValueError(f"Robot ee={ee!r} does not match saved hand type {limits['hand_type']!r}.")
