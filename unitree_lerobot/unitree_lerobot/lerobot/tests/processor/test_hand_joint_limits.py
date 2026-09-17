"""Offline regression tests; also runnable with python -m unittest."""

import copy
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import GrootPackInputsStep
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline, TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.processor.hand_joint_limits import (
    DEX3_LIMITS,
    configure_hand_normalization,
    normalize_hand_observations,
    resolve_hand_joint_limits,
)
from unitree_lerobot.utils.constants import G1_BRAINCO_CONFIG, G1_DEX3_CONFIG


def metadata(hand="dex3"):
    names = (G1_DEX3_CONFIG if hand == "dex3" else G1_BRAINCO_CONFIG).motors
    return SimpleNamespace(features={
        "observation.state": {"names": [list(names)], "shape": (len(names),)}
    })


def stats(dim):
    return {
        key: {
            "mean": torch.full((dim,), 0.2), "std": torch.full((dim,), 9.56e-6),
            "min": torch.full((dim,), -0.2), "max": torch.full((dim,), 0.8),
            "q01": torch.full((dim,), -0.1), "q99": torch.full((dim,), 0.9),
            "q10": torch.zeros(dim), "q90": torch.ones(dim),
        }
        for key in ("observation.state", "action")
    }


def config(cls=ACTConfig, hand="dex3", enabled=True):
    dim = 28 if hand == "dex3" else 26
    return cls(
        device="cpu", hand_joint_limit_normalization=enabled,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(dim,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(dim,))},
    )


class HandJointLimitsTest(unittest.TestCase):
    def test_limits_match_urdf(self):
        assets = Path(__file__).resolve().parents[3] / "eval_robot/assets/unitree_hand"
        for side in ("left", "right"):
            root = ET.parse(assets / f"unitree_dex3_{side}.urdf").getroot()
            for finger, count in (("thumb", 3), ("index", 2), ("middle", 2)):
                for index in range(count):
                    joint = root.find(f"joint[@name='{side}_hand_{finger}_{index}_joint']/limit")
                    self.assertEqual(
                        DEX3_LIMITS[f"k{side.title()}Hand{finger.title()}{index}"],
                        (float(joint.attrib["lower"]), float(joint.attrib["upper"])),
                    )

    def test_joint_order_is_resolved_by_name(self):
        meta = metadata()
        names = meta.features["observation.state"]["names"][0]
        names.reverse()
        limits = resolve_hand_joint_limits(meta.features)
        self.assertEqual(limits["indices"], list(range(14)))
        self.assertEqual(limits["lower"][0], 0.0)  # right middle 1, not left thumb 0
        self.assertEqual(limits["upper"][0], 1.74532925)

    def test_invalid_names(self):
        for names in (None, ["unknown"] * 28, list(G1_DEX3_CONFIG.motors[:-1]),
                      G1_DEX3_CONFIG.motors + G1_BRAINCO_CONFIG.motors[14:]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                resolve_hand_joint_limits({"observation.state": {"names": names, "shape": [28]}})

    def test_endpoints_batch_history_and_no_clipping(self):
        for hand in ("dex3", "brainco"):
            limits = resolve_hand_joint_limits(metadata(hand).features)
            dim = len(limits["state_names"])
            lo, hi = torch.tensor(limits["lower"]), torch.tensor(limits["upper"])
            for fraction, expected in ((0, -1), (0.5, 0), (1, 1), (1.5, 2)):
                with self.subTest(hand=hand, fraction=fraction):
                    raw = torch.zeros(2, 3, dim)
                    raw[..., limits["indices"]] = lo + fraction * (hi - lo)
                    original = raw.clone()
                    normalized = torch.full_like(raw, 7)
                    result = normalize_hand_observations(raw, normalized, limits)
                    torch.testing.assert_close(result[..., limits["indices"]],
                                               torch.full((2, 3, len(lo)), float(expected)), atol=1e-6, rtol=0)
                    torch.testing.assert_close(result[..., :14], normalized[..., :14])
                    torch.testing.assert_close(raw, original)
                    self.assertTrue(torch.all(normalized == 7))
            with self.assertRaises(ValueError):
                normalize_hand_observations(torch.zeros(dim - 1), torch.zeros(dim - 1), limits)

    def test_all_normalization_modes_preserve_other_features(self):
        for hand in ("dex3", "brainco"):
            limits = resolve_hand_joint_limits(metadata(hand).features)
            dim = len(limits["state_names"])
            features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(dim,)),
                        "action": PolicyFeature(type=FeatureType.ACTION, shape=(dim,)),
                        "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2))}
            for mode in NormalizationMode:
                with self.subTest(hand=hand, mode=mode):
                    data_stats = stats(dim)
                    data_stats["observation.images.head"] = {"mean": torch.zeros(3, 1, 1),
                                                            "std": torch.ones(3, 1, 1)}
                    original_stats = copy.deepcopy(data_stats)
                    kwargs = dict(features=features, stats=data_stats,
                                  norm_map={FeatureType.STATE: mode, FeatureType.ACTION: mode,
                                            FeatureType.VISUAL: NormalizationMode.MEAN_STD})
                    baseline = NormalizerProcessorStep(**kwargs)
                    processor = NormalizerProcessorStep(**kwargs, hand_joint_limits=limits)
                    raw = torch.full((2, dim), 0.2)
                    obs = {"observation.state": raw, "observation.images.head": torch.ones(2, 3, 2, 2)}
                    transition = create_transition(obs, torch.full((2, dim), 0.3))
                    expected, actual = baseline(transition), processor(transition)
                    torch.testing.assert_close(actual[TransitionKey.ACTION], expected[TransitionKey.ACTION])
                    torch.testing.assert_close(actual[TransitionKey.OBSERVATION]["observation.state"][..., :14],
                                               expected[TransitionKey.OBSERVATION]["observation.state"][..., :14])
                    torch.testing.assert_close(actual[TransitionKey.OBSERVATION]["observation.images.head"],
                                               expected[TransitionKey.OBSERVATION]["observation.images.head"])
                    changed = raw.clone()
                    changed[..., 14:] += 0.068
                    out = processor(create_transition({"observation.state": changed}))
                    delta = out[TransitionKey.OBSERVATION]["observation.state"][..., 14:] - actual[
                        TransitionKey.OBSERVATION]["observation.state"][..., 14:]
                    width = torch.tensor(limits["upper"]) - torch.tensor(limits["lower"])
                    torch.testing.assert_close(delta, (0.136 / width).expand_as(delta), atol=1e-6, rtol=0)
                    if hand == "dex3":
                        self.assertLess(delta.max().item(), 0.09)
                    for key in data_stats:
                        for stat in data_stats[key]:
                            torch.testing.assert_close(data_stats[key][stat], original_stats[key][stat])

    def test_factory_save_reload_and_resume_stats_override(self):
        for cls in (ACTConfig, DiffusionConfig):
            for hand in ("dex3", "brainco"):
                with self.subTest(policy=cls.__name__, hand=hand), tempfile.TemporaryDirectory() as tmp:
                    cfg = config(cls, hand)
                    dim = 28 if hand == "dex3" else 26
                    pre, post = make_pre_post_processors(cfg, dataset_stats=stats(dim), dataset_meta=metadata(hand))
                    sample = {"observation.state": torch.full((dim,), 0.2)}
                    expected = pre(sample)["observation.state"]
                    pre.save_pretrained(tmp)
                    post.save_pretrained(tmp)
                    for overrides in ({}, {"normalizer_processor": {"stats": stats(dim)}}):
                        loaded, _ = make_pre_post_processors(cfg, pretrained_path=tmp, dataset_meta=metadata(hand),
                                                             ee=hand, preprocessor_overrides=overrides)
                        torch.testing.assert_close(loaded(sample)["observation.state"], expected)
                    loaded, _ = make_pre_post_processors(cfg, pretrained_path=tmp)
                    torch.testing.assert_close(loaded(sample)["observation.state"], expected)
                    with self.assertRaises(ValueError):
                        make_pre_post_processors(cfg, pretrained_path=tmp, ee="brainco" if hand == "dex3" else "dex3")
                    with self.assertRaises(ValueError):
                        make_pre_post_processors(cfg, pretrained_path=tmp,
                                                 dataset_meta=metadata("brainco" if hand == "dex3" else "dex3"))

    def test_default_and_saved_setting(self):
        self.assertTrue(ACTConfig(device="cpu").hand_joint_limit_normalization)
        for cls in (ACTConfig, DiffusionConfig, GrootConfig):
            with self.subTest(policy=cls.__name__), tempfile.TemporaryDirectory() as tmp:
                cfg = config(cls)
                cfg.save_pretrained(tmp)
                loaded = PreTrainedConfig.from_pretrained(tmp)
                self.assertTrue(loaded.hand_joint_limit_normalization)
                cfg.hand_joint_limit_normalization = False
                cfg.save_pretrained(tmp)
                self.assertFalse(PreTrainedConfig.from_pretrained(tmp).hand_joint_limit_normalization)

    def test_metadata_required_and_checkpoint_setting_mismatch(self):
        with self.assertRaises(ValueError):
            make_pre_post_processors(config(), dataset_stats=stats(28))
        pre, _ = make_pre_post_processors(config(), dataset_stats=stats(28), dataset_meta=metadata())
        with self.assertRaises(ValueError):
            configure_hand_normalization(pre, enabled=False, loaded=True)
        pre, _ = make_pre_post_processors(config(enabled=False), dataset_stats=stats(28))
        with self.assertRaises(ValueError):
            configure_hand_normalization(pre, enabled=True, dataset_meta=metadata(), loaded=True)

    def test_groot_padding_and_factory_reload(self):
        for hand in ("dex3", "brainco"):
            with self.subTest(hand=hand), tempfile.TemporaryDirectory() as tmp:
                meta = metadata(hand)
                dim = meta.features["observation.state"]["shape"][0]
                cfg = config(GrootConfig, hand)
                pre, post = make_pre_post_processors(cfg, dataset_stats=stats(dim), dataset_meta=meta)
                sample = {"observation.state": torch.full((2, dim), 0.2), "action": torch.full((2, dim), 0.3)}
                expected = pre(copy.deepcopy(sample))
                baseline = PolicyProcessorPipeline(steps=[GrootPackInputsStep(stats=stats(dim))])(
                    copy.deepcopy(sample)
                )
                torch.testing.assert_close(expected["state"][..., :14], baseline["state"][..., :14])
                torch.testing.assert_close(expected["action"], baseline["action"])
                self.assertTrue(torch.all(expected["state"][..., dim:] == 0))
                limits = resolve_hand_joint_limits(meta.features)
                lo, hi = torch.tensor(limits["lower"]), torch.tensor(limits["upper"])
                torch.testing.assert_close(expected["state"][..., 14:dim],
                                           (2 * (0.2 - lo) / (hi - lo) - 1).expand(2, 1, -1))
                pre.save_pretrained(tmp)
                post.save_pretrained(tmp)
                for data_stats in (None, stats(dim)):
                    restored, _ = make_pre_post_processors(cfg, pretrained_path=tmp, dataset_stats=data_stats,
                                                          dataset_meta=meta, ee=hand)
                    torch.testing.assert_close(restored(copy.deepcopy(sample))["state"], expected["state"])


if __name__ == "__main__":
    unittest.main()
