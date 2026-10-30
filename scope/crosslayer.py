"""How features at neighboring layers relate, from what they fire on.

One forward pass captures the residual stream after each SAE's layer, each SAE encodes its own
layer, and for every pair of neighboring layers the products of their activations are summed per
(lower, upper) feature pair over the same tokens. The cosine of two features' activation
patterns, sum(a_i b_j) / sqrt(sum a_i^2 * sum b_j^2), is 1 when they fire on exactly the same
tokens in proportion. A feature's best partners one layer up are what it turns into; one layer
down, where it comes from.
"""

import torch

from .data import token_batches
from .model import keep_mask, residuals


class CoFire:
    """Sums of activation products between neighboring SAEs, stored as dense row blocks.

    A 65,536 x 65,536 float32 matrix is 17 GB; blocks keep each tensor under 2^31 elements.
    """

    def __init__(self, n_latents: int, n_layers: int, device: str = "cuda", block_rows: int = 16384):
        self.n, self.block_rows = n_latents, block_rows
        n_blocks = -(-n_latents // block_rows)
        self.cross = [
            [torch.zeros(min(block_rows, n_latents - b * block_rows) * n_latents, device=device) for b in range(n_blocks)]
            for _ in range(n_layers - 1)
        ]
        self.sq = [torch.zeros(n_latents, device=device, dtype=torch.float64) for _ in range(n_layers)]
        self.freq = [torch.zeros(n_latents, device=device, dtype=torch.long) for _ in range(n_layers)]
        self.tokens = 0

    def add(self, encodings: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """One (vals, idx) pair per layer, each [tokens, k], for the same tokens."""
        for layer, (vals, idx) in enumerate(encodings):
            flat_idx, flat_vals = idx.reshape(-1), vals.reshape(-1)
            self.sq[layer].index_add_(0, flat_idx, flat_vals.double() ** 2)
            self.freq[layer].index_add_(0, flat_idx, (flat_vals > 0).long())
        for pair, ((va, ia), (vb, ib)) in enumerate(zip(encodings, encodings[1:])):
            prod = (va[:, :, None] * vb[:, None, :]).reshape(-1)
            rows = ia[:, :, None].expand(-1, -1, ib.shape[1]).reshape(-1)
            cols = ib[:, None, :].expand(-1, ia.shape[1], -1).reshape(-1)
            live = prod > 0
            prod, rows, cols = prod[live], rows[live], cols[live]
            block = rows // self.block_rows
            for b, target in enumerate(self.cross[pair]):
                sel = block == b
                target.index_add_(0, (rows[sel] - b * self.block_rows) * self.n + cols[sel], prod[sel])
        self.tokens += encodings[0][0].shape[0]

    def matrix_rows(self, pair: int, start: int, stop: int) -> torch.Tensor:
        """Rows [start, stop) of one pair's product sums, start and stop inside one block."""
        b = start // self.block_rows
        offset = b * self.block_rows
        return self.cross[pair][b].view(-1, self.n)[start - offset : stop - offset]

    def row_chunks(self, chunk: int):
        start = 0
        while start < self.n:
            stop = min(start + chunk, self.n, (start // self.block_rows + 1) * self.block_rows)
            yield start, stop
            start = stop

    @torch.no_grad()
    def partners(self, top: int = 8, chunk: int = 4096) -> list[dict]:
        """For each neighboring pair: each lower feature's best partners up, each upper one's down."""
        out = []
        for pair in range(len(self.cross)):
            norm_lo = self.sq[pair].sqrt().float()
            norm_hi = self.sq[pair + 1].sqrt().float()
            up_cos, up_idx = [], []
            best_cos = torch.full((top, self.n), -1.0, device=norm_hi.device)
            best_idx = torch.zeros((top, self.n), dtype=torch.long, device=norm_hi.device)
            for start, stop in self.row_chunks(chunk):
                cos = self.matrix_rows(pair, start, stop) / (norm_lo[start:stop, None] * norm_hi[None, :]).clamp_min(1e-12)
                c, i = cos.topk(top, dim=1)
                up_cos.append(c)
                up_idx.append(i)
                c, i = cos.topk(min(top, stop - start), dim=0)
                merged_cos = torch.cat([best_cos, c])
                merged_idx = torch.cat([best_idx, i + start])
                keep = merged_cos.topk(top, dim=0).indices
                best_cos = merged_cos.gather(0, keep)
                best_idx = merged_idx.gather(0, keep)
            out.append({
                "up_idx": torch.cat(up_idx).int().cpu().numpy(),
                "up_cos": torch.cat(up_cos).cpu().numpy(),
                "down_idx": best_idx.t().int().contiguous().cpu().numpy(),
                "down_cos": best_cos.t().contiguous().cpu().numpy(),
            })
        return out


@torch.inference_mode()
def cofire(
    model,
    tokenizer,
    saes: list,
    n_tokens: int,
    top: int = 8,
    seq_len: int = 512,
    batch: int = 32,
    chat_fraction: float = 0.3,
    seed: int = 2,
    split: str = "catalog",
    chunk: int = 4096,
) -> dict:
    """Run text through the model once and match the SAEs' features layer to layer."""
    saes = sorted(saes, key=lambda s: s.cfg.layer)
    layers = [s.cfg.layer for s in saes]
    acc = CoFire(saes[0].cfg.n_latents, len(saes))
    sink_ids = tuple(i for i in (tokenizer.bos_token_id,) if i is not None)
    for ids, mask in token_batches(tokenizer, seq_len, batch, chat_fraction, seed, split):
        ids_d, mask_d = ids.cuda(), mask.cuda()
        acts = residuals(model, ids_d, mask_d, layers)
        # a token counts only if every layer keeps it, so all layers see the same tokens
        keep = mask_d.bool()
        for layer in layers:
            keep &= keep_mask(acts[layer], mask_d, ids_d, sink_ids)
        keep = keep.reshape(-1)
        xs = [acts[s.cfg.layer].reshape(-1, acts[s.cfg.layer].shape[-1])[keep].float() * s.cfg.input_scale for s in saes]
        for start in range(0, xs[0].shape[0], chunk):
            acc.add([s.encode(x[start : start + chunk]) for s, x in zip(saes, xs)])
        if acc.tokens >= n_tokens:
            break
    result = {"layers": layers, "tokens": acc.tokens}
    for layer, (sq, freq) in enumerate(zip(acc.sq, acc.freq)):
        result[f"freq_L{layers[layer]}"] = freq.cpu().numpy()
    for pair, found in enumerate(acc.partners(top)):
        lo, hi = layers[pair], layers[pair + 1]
        result[f"up_idx_L{lo}_L{hi}"] = found["up_idx"]
        result[f"up_cos_L{lo}_L{hi}"] = found["up_cos"]
        result[f"down_idx_L{hi}_L{lo}"] = found["down_idx"]
        result[f"down_cos_L{hi}_L{lo}"] = found["down_cos"]
    return result
