"""Loading Glimmer, reading one layer's residual stream, and steering it.

Each ``MuseGlimmerTextDecoderLayer`` returns the residual stream as a plain
tensor, so a forward hook on ``model.model.language_model.layers[i]`` sees
exactly what the next layer reads. Reading stops the forward pass at that
layer, which skips the rest of the model.
"""

from contextlib import contextmanager

import torch

from .config import WEIGHTS_DIR


class StopForward(Exception):
    pass


def load(weights_dir: str = WEIGHTS_DIR, device: str = "cuda"):
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        weights_dir, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return model, tokenizer


def decoder_layers(model):
    return model.model.language_model.layers


@torch.inference_mode()
def residual(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer: int) -> torch.Tensor:
    """The residual stream after ``layer``, shape [batch, seq, d_model]."""
    captured = {}

    def hook(_module, _inputs, output):
        captured["acts"] = output
        raise StopForward

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        model.model.language_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except StopForward:
        pass
    finally:
        handle.remove()
    return captured["acts"]


@contextmanager
def steering(model, layer: int, vector: torch.Tensor):
    """Add ``vector`` to the residual stream after ``layer`` at every position."""

    def hook(_module, _inputs, output):
        return output + vector.to(output.dtype)

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def keep_mask(acts: torch.Tensor, attention_mask: torch.Tensor, outlier_factor: float = 10.0) -> torch.Tensor:
    """Positions worth training on: real tokens, not the first one, no attention-sink outliers."""
    keep = attention_mask.bool().clone()
    keep[:, 0] = False
    norms = acts.float().norm(dim=-1)
    median = norms[keep].median() if keep.any() else norms.median()
    keep &= norms < outlier_factor * median
    return keep
