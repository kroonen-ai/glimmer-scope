"""CPU tests for labeling and steerability helpers."""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scope.label import describe  # noqa: E402
from scope.llm import json_from  # noqa: E402
from scope.model import steering  # noqa: E402
from scope.steerability import judge_prompt, plan_rows, score  # noqa: E402


def test_json_from_reads_fenced_and_bare_replies():
    assert json_from('Sure.\n```json\n[{"latent": 1}]\n```') == [{"latent": 1}]
    assert json_from('Here: {"strengths": [], "observed": "x"} done') == {"strengths": [], "observed": "x"}


def test_score_takes_the_best_concept_times_fluency():
    verdict = {"strengths": [
        {"strength": 0.25, "concept": 1, "fluency": 3},
        {"strength": 0.5, "concept": 3, "fluency": 2},
        {"strength": 1.0, "concept": 3, "fluency": 0},
    ]}
    assert score(verdict) == (6 / 9, 0.5)
    assert score({"strengths": []}) == (0.0, None)


def test_rows_are_feature_major():
    assert plan_rows([7, 9], 2) == [(7, 0), (7, 1), (9, 0), (9, 1)]


def test_judge_prompt_formats_without_brace_errors():
    text = judge_prompt({"latent": 1, "label": "ocean"}, ["the [[sea]] was calm"], ["a", "b"], {0.5: ["c", "d"]})
    assert '"ocean"' in text and "Steered at strength 0.5" in text and '{"strengths"' in text


def test_describe_marks_the_firing_token():
    row = {"latent": 3, "freq": 0.001, "examples": [{"act": 4.2, "left": "the deep", "token": " sea", "right": " floor"}]}
    assert "[[ sea]]" in describe(row)


class _Layer(nn.Module):
    def forward(self, x):
        return x


class _Fake:
    def __init__(self):
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_Layer()])


def test_per_row_steering_vectors_broadcast_over_positions():
    fake = _Fake()
    x = torch.zeros(2, 3, 4)
    vectors = torch.tensor([[1.0, 0, 0, 0], [0, 2.0, 0, 0]])
    with steering(fake, 0, vectors[:, None, :]):
        out = fake.model.language_model.layers[0](x)
    assert torch.equal(out[0, :, 0], torch.ones(3)) and torch.equal(out[1, :, 1], torch.full((3,), 2.0))
    assert torch.equal(fake.model.language_model.layers[0](x), x), "the hook is removed afterwards"
