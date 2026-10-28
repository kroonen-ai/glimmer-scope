"""Find the features behind an archetype, and steer with them.

Muse Spark writes matched pairs: the same everyday request answered once in
the archetype's voice and once plainly. Glimmer reads both answers with no
persona instruction, so a feature that fires more on the archetype side does
so because of the voice itself. Features are ranked by their mean paired
difference times how consistently the archetype side wins.

Three kinds of steering direction come out of it: each top feature alone, a
blend of the top features weighted by their difference, and the plain
difference of mean residuals over the same tokens (contrastive activation
addition), as the baseline the SAE has to beat.
"""

import torch

from .llm import chat, json_from
from .model import residual
from .steerability import answer_prefix

PAIRS_PROMPT = """Write {n} short exchanges that contrast two voices.

Archetype: {name}. {description}

For each exchange, write an everyday request a user might send (vary the topics: cooking, work, travel,
technology, money, health, hobbies, family, learning, shopping), then two replies of 60 to 110 words that
give the same practical content:
- "archetype": fully in the archetype's voice
- "neutral": a plain, friendly assistant voice
Never name the archetype or mention role-play in the replies.

Reply with only a JSON array of objects with "prompt", "archetype", "neutral"."""


def make_pairs(name: str, description: str, n: int = 24, per_call: int = 8) -> list[dict]:
    pairs: list[dict] = []
    while len(pairs) < n:
        reply = chat(PAIRS_PROMPT.format(n=per_call, name=name, description=description), max_tokens=12000)
        for item in json_from(reply):
            if all(isinstance(item.get(k), str) and item[k].strip() for k in ("prompt", "archetype", "neutral")):
                pairs.append({k: item[k].strip() for k in ("prompt", "archetype", "neutral")})
    return pairs[:n]


@torch.inference_mode()
def answer_activations(model, tokenizer, sae, prompt: str, answer: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean SAE activation per latent and mean raw residual over the answer's tokens."""
    prefix = answer_prefix(tokenizer, prompt)
    n_prefix = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    enc = tokenizer(prefix + answer, return_tensors="pt", add_special_tokens=False).to("cuda")
    acts = residual(model, enc.input_ids, enc.attention_mask, sae.cfg.layer)[0, n_prefix:].float()
    vals, idx = sae.encode(acts * sae.cfg.input_scale)
    dense = torch.zeros(sae.cfg.n_latents, device=acts.device)
    dense.index_add_(0, idx.reshape(-1), vals.reshape(-1))
    return dense / acts.shape[0], acts.mean(0)


def rank_features(archetype: torch.Tensor, neutral: torch.Tensor, top: int) -> list[dict]:
    """``archetype`` and ``neutral`` are [pairs, latents] mean activations, row-aligned."""
    diff = archetype - neutral
    mean_diff = diff.mean(0)
    consistency = (diff > 0).float().mean(0)
    score = mean_diff * consistency
    values, latents = score.topk(top)
    return [
        {
            "latent": int(latent),
            "score": round(float(value), 4),
            "mean_diff": round(float(mean_diff[latent]), 4),
            "consistency": round(float(consistency[latent]), 3),
            "archetype_mean": round(float(archetype[:, latent].mean()), 4),
            "neutral_mean": round(float(neutral[:, latent].mean()), 4),
        }
        for value, latent in zip(values, latents)
        if value > 0
    ]


def blend_direction(sae, ranked: list[dict]) -> torch.Tensor:
    weights = torch.tensor([r["mean_diff"] for r in ranked], device=sae.W_dec.device)
    rows = sae.W_dec[[r["latent"] for r in ranked]]
    direction = (weights[:, None] * rows).sum(0)
    return direction / direction.norm()


def mean_difference_direction(archetype_resid: torch.Tensor, neutral_resid: torch.Tensor) -> torch.Tensor:
    direction = (archetype_resid - neutral_resid).mean(0)
    return direction / direction.norm()


ARCHETYPES = {
    "pirate captain": "A swaggering sea captain who talks in nautical slang, says 'arr' and 'matey', and frames everything as a voyage.",
    "wise mentor": "A warm, patient teacher who guides with gentle questions, small stories, and encouragement instead of direct orders.",
    "noir detective": "A world-weary 1940s private eye narrating in clipped, moody first person, full of rain, shadows, and cynical similes.",
    "hype coach": "An overexcited motivational coach: huge energy, exclamation marks, sports metaphors, and relentless encouragement.",
    "stoic philosopher": "A calm Stoic who answers with measured reflections on what is in our control, virtue, and acceptance.",
}


def slug(name: str) -> str:
    return "-".join("".join(c if c.isalnum() else " " for c in name.lower()).split())
