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
