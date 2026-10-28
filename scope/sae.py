"""TopK sparse autoencoder (Gao et al., 2024) with an AuxK loss for dead latents.

Inputs are residual-stream vectors multiplied by ``cfg.input_scale``. Each
latent's decoder row is kept at unit norm, so a latent's activation is its
contribution's length in scaled units, and ``steering_vector`` converts a
target activation back to the raw residual stream.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .config import SAEConfig


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d, n = cfg.d_in, cfg.n_latents
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.b_enc = nn.Parameter(torch.zeros(n))
        w_dec = torch.randn(n, d)
        w_dec /= w_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(w_dec)
        self.W_enc = nn.Parameter(w_dec.t().clone())
        # Tokens since each latent last fired, for dead-latent tracking.
        self.register_buffer("since_fired", torch.zeros(n, dtype=torch.long))

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vals, idx = self.pre_acts(x).topk(self.cfg.k, dim=-1)
        return F.relu(vals), idx

    def decode(self, vals: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # embedding_bag sums the selected decoder rows without materializing
        # a [batch, k, d_in] tensor.
        return F.embedding_bag(idx, self.W_dec, per_sample_weights=vals, mode="sum") + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        pre = self.pre_acts(x)
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        vals = F.relu(vals)
        recon = F.embedding_bag(idx, self.W_dec, per_sample_weights=vals, mode="sum") + self.b_dec
        err = x - recon
        variance = (x - x.mean(0)).pow(2).sum(-1).mean()
        mse = err.pow(2).sum(-1).mean()
        loss = mse / variance

        dead = self.since_fired >= self.cfg.dead_after_tokens
        aux = torch.zeros((), device=x.device)
        if dead.any():
            k_aux = min(self.cfg.k_aux, int(dead.sum()))
            dead_pre = pre.masked_fill(~dead, float("-inf"))
            a_vals, a_idx = dead_pre.topk(k_aux, dim=-1)
            a_vals = F.relu(a_vals)
            aux_recon = F.embedding_bag(a_idx, self.W_dec, per_sample_weights=a_vals, mode="sum")
            aux = (err.detach() - aux_recon).pow(2).sum(-1).mean() / variance
            loss = loss + self.cfg.aux_coef * aux

        with torch.no_grad():
            fired = torch.zeros_like(self.since_fired, dtype=torch.bool)
            fired[idx[vals > 0].unique()] = True
            self.since_fired += x.shape[0]
            self.since_fired[fired] = 0

        return {"loss": loss, "fvu": (mse / variance).detach(), "aux": aux.detach(), "dead": dead.float().mean()}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_grad(self) -> None:
        """Drop the gradient component that would change decoder row norms."""
        if self.W_dec.grad is None:
            return
        parallel = (self.W_dec.grad * self.W_dec).sum(-1, keepdim=True)
        self.W_dec.grad -= parallel * self.W_dec

    def steering_vector(self, latent: int, activation: float) -> torch.Tensor:
        """Residual-stream vector that adds ``activation`` of one latent."""
        return self.W_dec[latent] * (activation / self.cfg.input_scale)

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items() if k != "since_fired"}
        save_file(tensors, str(path / "sae.safetensors"))
        (path / "cfg.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TopKSAE":
        from safetensors.torch import load_file

        path = Path(path)
        cfg = SAEConfig(**json.loads((path / "cfg.json").read_text()))
        sae = cls(cfg)
        sae.load_state_dict(load_file(str(path / "sae.safetensors")), strict=False)
        return sae.to(device)
