"""Feature catalog: how often each latent fires and the contexts where it fires hardest."""

import gzip
import json
from pathlib import Path

import numpy as np
import torch

from .data import token_batches
from .model import keep_mask, residual
from .sae import TopKSAE


def merge_top(best_vals, best_ptr, vals, latents, ptrs, top_n):
    """Merge one batch's (activation, latent, position) triples into per-latent top lists."""
    order = np.lexsort((-vals, latents))
    vals, latents, ptrs = vals[order], latents[order], ptrs[order]
    starts = np.searchsorted(latents, latents, side="left")
    rank = np.arange(len(latents)) - starts
    keep = rank < top_n
    vals, latents, ptrs, rank = vals[keep], latents[keep], ptrs[keep], rank[keep]
    cand_vals = np.full_like(best_vals, -np.inf)
    cand_ptr = np.full_like(best_ptr, -1)
    cand_vals[latents, rank] = vals
    cand_ptr[latents, rank] = ptrs
    all_vals = np.concatenate([best_vals, cand_vals], axis=1)
    all_ptr = np.concatenate([best_ptr, cand_ptr], axis=1)
    top = np.argsort(-all_vals, axis=1)[:, :top_n]
    return np.take_along_axis(all_vals, top, 1), np.take_along_axis(all_ptr, top, 1)


@torch.inference_mode()
def build_catalog(
    model,
    tokenizer,
    sae: TopKSAE,
    n_tokens: int,
    out_path: str,
    top_n: int = 16,
    seq_len: int = 512,
    batch: int = 32,
    chat_fraction: float = 0.3,
    seed: int = 1,
    split: str = "catalog",
) -> dict:
    n = sae.cfg.n_latents
    best_vals = np.full((n, top_n), -np.inf, dtype=np.float32)
    best_ptr = np.full((n, top_n), -1, dtype=np.int64)
    freq = np.zeros(n, dtype=np.int64)
    act_sum = np.zeros(n, dtype=np.float64)
    token_rows: list[np.ndarray] = []
    seen = 0
    kept_total = 0
    sink_ids = tuple(i for i in (tokenizer.bos_token_id,) if i is not None)

    for ids, mask in token_batches(tokenizer, seq_len, batch, chat_fraction, seed, split):
        ids_d, mask_d = ids.cuda(), mask.cuda()
        acts = residual(model, ids_d, mask_d, sae.cfg.layer)
        keep = keep_mask(acts, mask_d, ids_d, sink_ids).reshape(-1)
        x = acts.reshape(-1, acts.shape[-1]).float() * sae.cfg.input_scale
        vals_parts, idx_parts = [], []
        for chunk in range(0, x.shape[0], 4096):
            v, i = sae.encode(x[chunk : chunk + 4096])
            vals_parts.append(v)
            idx_parts.append(i)
        vals = torch.cat(vals_parts)
        idx = torch.cat(idx_parts)
        vals[~keep] = 0
        positions = seen + torch.arange(x.shape[0], device=x.device)
        flat_vals = vals.reshape(-1)
        live = flat_vals > 0
        v = flat_vals[live].cpu().numpy().astype(np.float32)
        lat = idx.reshape(-1)[live].cpu().numpy().astype(np.int64)
        ptr = positions.repeat_interleave(sae.cfg.k)[live].cpu().numpy()
        freq += np.bincount(lat, minlength=n)
        act_sum += np.bincount(lat, weights=v, minlength=n)
        best_vals, best_ptr = merge_top(best_vals, best_ptr, v, lat, ptr, top_n)
        token_rows.append(ids.numpy().reshape(-1))
        seen += x.shape[0]
        kept_total += int(keep.sum())
        if seen >= n_tokens:
            break

    tokens = np.concatenate(token_rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    alive = 0
    with gzip.open(out, "wt") as f:
        for latent in range(n):
            if freq[latent] == 0:
                continue
            alive += 1
            examples = []
            for val, p in zip(best_vals[latent], best_ptr[latent]):
                if p < 0:
                    continue
                row_start = (p // seq_len) * seq_len
                left = tokenizer.decode(tokens[max(row_start + 1, p - 24) : p])
                right = tokenizer.decode(tokens[p + 1 : min(row_start + seq_len, p + 9)])
                examples.append(
                    {"act": round(float(val), 3), "left": left, "token": tokenizer.decode([int(tokens[p])]), "right": right}
                )
            f.write(
                json.dumps(
                    {
                        "latent": latent,
                        "freq": freq[latent] / max(kept_total, 1),
                        "mean_act": act_sum[latent] / freq[latent],
                        "max_act": float(best_vals[latent][0]),
                        "examples": examples,
                    }
                )
                + "\n"
            )
    return {"tokens": seen, "kept": kept_total, "alive": alive, "latents": n}
