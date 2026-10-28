"""CPU tests for archetype ranking."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scope.archetype import blend_direction, mean_difference_direction, rank_features, slug  # noqa: E402
from scope.config import SAEConfig  # noqa: E402
from scope.sae import TopKSAE  # noqa: E402


def test_rank_prefers_features_that_win_consistently():
    neutral = torch.zeros(4, 5)
    archetype = torch.zeros(4, 5)
    archetype[:, 1] = 1.0  # always higher
    archetype[0, 2] = 8.0  # higher on one pair only, same mean as above x2
    archetype[:, 3] = -1.0  # lower: never ranked
    ranked = rank_features(archetype, neutral, top=3)
    assert [r["latent"] for r in ranked] == [1, 2]
    assert ranked[0]["consistency"] == 1.0 and ranked[1]["consistency"] == 0.25


def test_directions_are_unit_length():
    sae = TopKSAE(SAEConfig(d_in=8, n_latents=16, k=2))
    blend = blend_direction(sae, [{"latent": 1, "mean_diff": 2.0}, {"latent": 5, "mean_diff": 0.5}])
    assert torch.isclose(blend.norm(), torch.tensor(1.0))
    mdd = mean_difference_direction(torch.randn(3, 8) + 2, torch.randn(3, 8))
    assert torch.isclose(mdd.norm(), torch.tensor(1.0))


def test_slug():
    assert slug("Noir Detective!") == "noir-detective"


from scope import archetype as arch  # noqa: E402


def test_every_archetype_has_prompts_and_eval_prompts():
    groups = {"character", "companion", "anime", "research"}
    for name, spec in arch.ARCHETYPES.items():
        assert spec["group"] in groups, name
        assert spec["prompts"] in arch.PROMPT_SETS and spec["prompts"] in arch.EVAL_SETS, name
        assert spec["description"] and spec["contrast"], name
    for key, prompts in arch.PROMPT_SETS.items():
        if key == "everyday":
            continue
        assert len(prompts) >= 24 and len(set(prompts)) == len(prompts), key
        assert not set(prompts) & set(arch.EVAL_SETS[key]), f"{key}: eval prompts leak into pairs"


def test_split_pairs_is_seeded_and_disjoint():
    pairs = [{"prompt": str(i)} for i in range(24)]
    train, test = arch.split_pairs(pairs)
    assert (len(train), len(test)) == (16, 8)
    assert not {p["prompt"] for p in train} & {p["prompt"] for p in test}
    assert arch.split_pairs(pairs) == (train, test)
    assert arch.split_pairs(pairs[:2]) == (pairs[:2], [])


def test_detector_accuracy():
    assert arch.detector_accuracy(torch.tensor([2.0, 1.0, 0.0]), torch.tensor([1.0, 2.0, -1.0])) == 0.667
    assert arch.detector_accuracy(torch.empty(0), torch.empty(0)) is None


def test_spark_pairs_keeps_only_valid_pairs_for_asked_prompts(monkeypatch):
    reply = (
        '[{"prompt": "a", "archetype": "A!", "neutral": "a."},'
        ' {"prompt": "not asked", "archetype": "x", "neutral": "y"},'
        ' {"prompt": "b", "archetype": "", "neutral": "b."}]'
    )
    monkeypatch.setattr(arch, "chat", lambda *_a, **_k: reply)
    pairs = arch.spark_pairs("tsundere", arch.ARCHETYPES["tsundere"], ["a", "b", "c"])
    assert [p["prompt"] for p in pairs] == ["a"] and pairs[0]["source"] == "muse-spark"


def test_spark_pairs_survives_a_refusal(monkeypatch):
    monkeypatch.setattr(arch, "chat", lambda *_a, **_k: "I'd rather not write these.")
    assert arch.spark_pairs("manipulator", arch.ARCHETYPES["manipulator"], ["a", "b"]) == []


def test_scan_batch_keeps_known_latents(monkeypatch):
    rows = [{"latent": 5, "label": "mob boss talk"}, {"latent": 9, "label": "commas"}]
    monkeypatch.setattr(arch, "chat", lambda *_a, **_k: '[{"latent": 5, "persona": "mob boss"}, {"latent": 77, "persona": "x"}]')
    assert arch.scan_batch(rows) == [{"latent": 5, "label": "mob boss talk", "persona": "mob boss"}]
