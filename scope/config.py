"""Shared constants for the Glimmer Scope pipeline."""

from dataclasses import asdict, dataclass

MODEL_ID = "meta-models/Muse-Glimmer-30B"
WEIGHTS_DIR = "/weights/Muse-Glimmer-30B"
OUTPUT_DIR = "/scope"

D_MODEL = 6656
N_LAYERS = 52


@dataclass
class SAEConfig:
    """A TopK sparse autoencoder over one layer's residual stream."""

    layer: int = 24  # 0-indexed decoder layer whose output the SAE reads
    n_latents: int = 65536
    k: int = 64
    d_in: int = D_MODEL
    # Scalar applied to raw activations so that E[||x||^2] == d_in; measured
    # from the first buffer and stored with the weights.
    input_scale: float = 1.0
    k_aux: int = 512
    aux_coef: float = 1 / 32
    dead_after_tokens: int = 10_000_000

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    total_tokens: int = 100_000_000
    seq_len: int = 1024
    model_batch: int = 16  # sequences per model forward
    sae_batch: int = 4096  # activation vectors per SAE step
    buffer_tokens: int = 1 << 20  # shuffle buffer, bf16 on the GPU
    lr: float = 2e-4
    chat_fraction: float = 0.3  # share of chat-formatted documents
    log_every: int = 50
    checkpoint_every_tokens: int = 10_000_000
    seed: int = 0
