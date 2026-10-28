"""Streaming SAE training: the model fills a shuffle buffer, the SAE trains on half of it, repeat."""

import json
import math
import time
from pathlib import Path

import torch

from .config import SAEConfig, TrainConfig
from .data import token_batches
from .model import keep_mask, residual
from .sae import TopKSAE


class ActivationBuffer:
    def __init__(self, capacity: int, d: int, device: str):
        self.data = torch.empty(capacity, d, dtype=torch.bfloat16, device=device)
        self.n = 0

    @property
    def full(self) -> bool:
        return self.n == self.data.shape[0]

    def add(self, x: torch.Tensor) -> int:
        take = min(x.shape[0], self.data.shape[0] - self.n)
        self.data[self.n : self.n + take] = x[:take].to(self.data.dtype)
        self.n += take
        return take


def learning_rate(cfg: TrainConfig, sae_cfg: SAEConfig, step: int, total_steps: int) -> float:
    # Gao et al. scale the learning rate with 1/sqrt(latents); the base is tuned at 2^14.
    base = cfg.lr * math.sqrt((1 << 14) / sae_cfg.n_latents)
    warmup = 500
    if step < warmup:
        return base * (step + 1) / warmup
    decay_start = int(total_steps * 0.8)
    if step >= decay_start:
        return base * max(0.0, (total_steps - step) / max(1, total_steps - decay_start))
    return base


def train(model, tokenizer, sae_cfg: SAEConfig, cfg: TrainConfig, out_dir: str, commit=lambda: None) -> Path:
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_cfg.json").write_text(json.dumps(cfg.__dict__, indent=2))
    metrics = (out / "metrics.jsonl").open("a")
    device = "cuda"
    batches = token_batches(tokenizer, cfg.seq_len, cfg.model_batch, cfg.chat_fraction, cfg.seed)
    buffer = ActivationBuffer(cfg.buffer_tokens, sae_cfg.d_in, device)
    model_tokens = 0
    model_seconds = 0.0
    sink_ids = tuple(i for i in (tokenizer.bos_token_id,) if i is not None)

    def fill() -> None:
        nonlocal model_tokens, model_seconds
        while not buffer.full:
            ids, mask = next(batches)
            ids, mask = ids.to(device), mask.to(device)
            started = time.time()
            acts = residual(model, ids, mask, sae_cfg.layer)
            keep = keep_mask(acts, mask, ids, sink_ids)
            buffer.add(acts[keep])
            torch.cuda.synchronize()
            model_seconds += time.time() - started
            model_tokens += int(mask.sum())

    fill()
    sample = buffer.data[: min(buffer.n, 65536)].float()
    sae_cfg.input_scale = math.sqrt(sae_cfg.d_in / sample.pow(2).sum(-1).mean().item())
    sae = TopKSAE(sae_cfg).to(device)
    with torch.no_grad():
        sae.b_dec.copy_((sample * sae_cfg.input_scale).mean(0))
    del sample
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=6.25e-10)

    total_steps = cfg.total_tokens // cfg.sae_batch
    step = 0
    trained = 0
    next_checkpoint = cfg.checkpoint_every_tokens
    started = time.time()
    while trained < cfg.total_tokens:
        perm = torch.randperm(buffer.n, device=device)
        half = buffer.n // 2
        for start in range(0, half - cfg.sae_batch + 1, cfg.sae_batch):
            x = buffer.data[perm[start : start + cfg.sae_batch]].float() * sae_cfg.input_scale
            for group in opt.param_groups:
                group["lr"] = learning_rate(cfg, sae_cfg, step, total_steps)
            result = sae(x)
            opt.zero_grad(set_to_none=True)
            result["loss"].backward()
            sae.remove_parallel_grad()
            opt.step()
            sae.normalize_decoder()
            step += 1
            trained += cfg.sae_batch
            if step % cfg.log_every == 0:
                row = {
                    "step": step,
                    "tokens": trained,
                    "fvu": round(result["fvu"].item(), 5),
                    "aux": round(result["aux"].item(), 5),
                    "dead": round(result["dead"].item(), 5),
                    "lr": opt.param_groups[0]["lr"],
                    "model_tok_s": round(model_tokens / max(model_seconds, 1e-6)),
                    "elapsed_s": round(time.time() - started),
                }
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(json.dumps(row), flush=True)
            if trained >= next_checkpoint:
                sae.save(out / "checkpoint")
                commit()
                next_checkpoint += cfg.checkpoint_every_tokens
            if trained >= cfg.total_tokens:
                break
        rest = buffer.data[perm[half : buffer.n]].clone()
        buffer.data[: rest.shape[0]] = rest
        buffer.n = rest.shape[0]
        del rest
        if trained < cfg.total_tokens:
            fill()

    sae.save(out / "final")
    commit()
    return out / "final"
