"""CPU tests for the SAE and catalog helpers: run with `uv run --with torch --with numpy --with safetensors --with pytest pytest -q`."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scope.config import SAEConfig  # noqa: E402
from scope.interpret import merge_top  # noqa: E402
from scope.model import keep_mask  # noqa: E402
from scope.sae import TopKSAE  # noqa: E402


def small_cfg(**overrides):
    cfg = dict(d_in=32, n_latents=256, k=4, k_aux=16, dead_after_tokens=1000)
    cfg.update(overrides)
    return SAEConfig(**cfg)


def test_encode_keeps_exactly_k_latents_and_decoder_rows_are_unit():
    torch.manual_seed(0)
    sae = TopKSAE(small_cfg())
    vals, idx = sae.encode(torch.randn(10, 32))
    assert vals.shape == idx.shape == (10, 4)
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(256), atol=1e-5)


def test_learns_a_sparse_dictionary():
    torch.manual_seed(0)
    true_dict = torch.randn(64, 32)
    true_dict /= true_dict.norm(dim=1, keepdim=True)
    sae = TopKSAE(small_cfg(n_latents=128))
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    first = None
    for step in range(600):
        codes = torch.zeros(512, 64)
        picks = torch.randint(0, 64, (512, 3))
        codes.scatter_(1, picks, torch.rand(512, 3) * 3 + 0.5)
        x = codes @ true_dict
        out = sae(x)
        opt.zero_grad()
        out["loss"].backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()
        if first is None:
            first = out["fvu"].item()
    assert out["fvu"].item() < 0.25 * first


def test_dead_latents_get_aux_loss():
    torch.manual_seed(0)
    sae = TopKSAE(small_cfg(dead_after_tokens=1))
    sae(torch.randn(8, 32))
    out = sae(torch.randn(8, 32))
    assert out["dead"].item() > 0
    assert out["aux"].item() > 0


def test_steering_vector_converts_to_raw_units():
    sae = TopKSAE(small_cfg(input_scale=0.5))
    v = sae.steering_vector(3, 2.0)
    assert torch.allclose(v.norm(), torch.tensor(4.0), atol=1e-4)


def test_save_and_load_round_trip(tmp_path):
    sae = TopKSAE(small_cfg(input_scale=0.25))
    sae.save(tmp_path)
    again = TopKSAE.load(tmp_path)
    assert again.cfg.input_scale == 0.25
    assert torch.equal(again.W_dec, sae.W_dec)


def test_merge_top_keeps_the_highest_per_latent():
    n, top = 3, 2
    best_vals = np.full((n, top), -np.inf, dtype=np.float32)
    best_ptr = np.full((n, top), -1, dtype=np.int64)
    vals = np.array([1.0, 5.0, 3.0, 2.0, 9.0], dtype=np.float32)
    lat = np.array([0, 0, 0, 2, 2])
    ptr = np.array([10, 11, 12, 13, 14])
    best_vals, best_ptr = merge_top(best_vals, best_ptr, vals, lat, ptr, top)
    assert best_vals[0].tolist() == [5.0, 3.0] and best_ptr[0].tolist() == [11, 12]
    assert best_ptr[1].tolist() == [-1, -1]
    best_vals, best_ptr = merge_top(best_vals, best_ptr, np.array([4.0], dtype=np.float32), np.array([0]), np.array([20]), top)
    assert best_vals[0].tolist() == [5.0, 4.0] and best_ptr[0].tolist() == [11, 20]


def test_keep_mask_drops_first_position_bos_tokens_and_outliers():
    acts = torch.ones(1, 6, 4)
    acts[0, 4] *= 100  # attention-sink sized
    ids = torch.tensor([[7, 5, 7, 5, 5, 5]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    keep = keep_mask(acts, mask, ids, exclude_ids=(7,))
    assert keep.tolist() == [[False, True, False, True, False, False]]
