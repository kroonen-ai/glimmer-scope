"""Find the features behind an archetype, and steer with them.

Each archetype comes with user messages that suit it: relationship moments for
companions, wrong claims for a sycophant, attempts to leave for a manipulator.
Every message gets two replies, one in the archetype's voice and one plain.
Muse Spark writes them; for any message it leaves out, Glimmer writes its own
pair, the archetype side under a persona system prompt. Both sides of a pair
come from the same writer, so the writer's own style cancels out.

Glimmer then reads both replies with no persona instruction, so a feature that
fires more on the archetype side does so because of the reply itself. Features
are ranked by mean paired difference times how consistently the archetype
side wins, on two thirds of the pairs; the last third checks how well the top
feature and the mean-difference vector tell the two sides apart.

Three kinds of steering direction come out of it: each top feature alone, a
blend of the top features weighted by their difference, and the plain
difference of mean residuals (contrastive activation addition), the baseline
the SAE has to beat. They are judged on prompts that suit the archetype.
"""

import random

import torch

from .llm import chat, json_from
from .model import residual
from .personas import ARCHETYPES, EVAL_SETS, PROMPT_SETS, SAME_CONTENT, slug  # noqa: F401
from .steerability import ANSWER_CHANNEL, answer_prefix, generate

PAIRS_PROMPT = """Write paired replies that contrast two voices, for research on how a language model represents personas.

Archetype: {name}. {description}
{contrast}

For each user message below, write two replies of 50 to 110 words:
- "archetype": fully in the archetype's voice and manner
- "neutral": a plain, friendly, honest assistant
Never name the archetype or mention role-play in the replies. If you would rather not write a pair for a message, leave it out.

User messages:
{messages}

Reply with only a JSON array of objects with "prompt" (the user message, copied exactly), "archetype", "neutral"."""

# The first five archetypes asked Muse Spark for its own everyday requests.
EVERYDAY_PAIRS_PROMPT = """Write {n} short exchanges that contrast two voices.

Archetype: {name}. {description}

For each exchange, write an everyday request a user might send (vary the topics: cooking, work, travel,
technology, money, health, hobbies, family, learning, shopping), then two replies of 60 to 110 words that
give the same practical content:
- "archetype": fully in the archetype's voice
- "neutral": a plain, friendly assistant voice
Never name the archetype or mention role-play in the replies.

Reply with only a JSON array of objects with "prompt", "archetype", "neutral"."""

KEYS = ("prompt", "archetype", "neutral")


def _valid(item) -> bool:
    return isinstance(item, dict) and all(isinstance(item.get(k), str) and item[k].strip() for k in KEYS)


def spark_pairs(name: str, spec: dict, prompts: list[str], per_call: int = 8) -> list[dict]:
    """Muse Spark's pairs for the prompts it is willing to write; messages it leaves out are simply missing."""
    wanted = set(prompts)
    pairs: dict[str, dict] = {}
    for start in range(0, len(prompts), per_call):
        chunk = prompts[start : start + per_call]
        text = PAIRS_PROMPT.format(
            name=name, description=spec["description"], contrast=spec["contrast"],
            messages="\n".join(f"- {p}" for p in chunk),
        )
        try:
            items = json_from(chat(text, max_tokens=12000))
        except Exception as error:  # a refusal or malformed reply loses only this chunk
            print(f"{name}: Muse Spark chunk skipped: {error}", flush=True)
            continue
        for item in items if isinstance(items, list) else []:
            if _valid(item) and item["prompt"].strip() in wanted:
                pairs[item["prompt"].strip()] = {**{k: item[k].strip() for k in KEYS}, "source": "muse-spark"}
    return [pairs[p] for p in prompts if p in pairs]


def everyday_pairs(name: str, spec: dict, n: int = 24, per_call: int = 8) -> list[dict]:
    pairs: list[dict] = []
    while len(pairs) < n:
        reply = chat(EVERYDAY_PAIRS_PROMPT.format(n=per_call, name=name, description=spec["description"]), max_tokens=12000)
        pairs += [{**{k: item[k].strip() for k in KEYS}, "source": "muse-spark"} for item in json_from(reply) if _valid(item)]
    return pairs[:n]


def persona_system(name: str, spec: dict) -> str:
    return (
        f"You are {name}. {spec['description']} Stay fully in character in every reply, "
        "and never mention that you are playing a role."
    )


def chat_prefix(tokenizer, prompt: str, system: str | None = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, reasoning_strength="low")
    return text + ANSWER_CHANNEL


def glimmer_pairs(model, tokenizer, layer: int, name: str, spec: dict, prompts: list[str], max_new_tokens: int = 170) -> list[dict]:
    """Glimmer answers each prompt twice: under the persona system prompt, and plainly."""
    if not prompts:
        return []
    system = persona_system(name, spec)
    archetype = generate(model, tokenizer, [chat_prefix(tokenizer, p, system) for p in prompts], layer, None, max_new_tokens)
    neutral = generate(model, tokenizer, [chat_prefix(tokenizer, p) for p in prompts], layer, None, max_new_tokens)
    return [
        {"prompt": p, "archetype": a, "neutral": n, "source": "glimmer"}
        for p, a, n in zip(prompts, archetype, neutral)
        if a and n
    ]


def split_pairs(pairs: list[dict], held_out: float = 1 / 3, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """A seeded shuffle, then the last third held out for the detector check."""
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * held_out)) if len(shuffled) > 2 else 0
    return shuffled[: len(shuffled) - n_test], shuffled[len(shuffled) - n_test :]


def detector_accuracy(archetype_scores: torch.Tensor, neutral_scores: torch.Tensor) -> float | None:
    """How often the archetype side of a held-out pair scores higher."""
    if archetype_scores.numel() == 0:
        return None
    return round(float((archetype_scores > neutral_scores).float().mean()), 3)


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


SCAN_PROMPT = """Below are labels of features found inside a language model, one per line as "id: label".
Pick the ones that describe a personality, character type, voice, emotional stance, or way of relating to
someone: anything that, added to a chatbot, might make it take on a persona (for example a mob boss, an
anxious inner voice, a flirt, a sports commentator, a grandparent, a con artist). Skip plain topics, facts,
formats and grammar.

Reply with only a JSON array of {{"latent": <id>, "persona": "<a few words naming the persona it could give>"}},
or [] if none qualify.

{labels}"""


def scan_batch(rows: list[dict]) -> list[dict]:
    """Muse Spark picks the labels that could turn into a persona."""
    by_latent = {r["latent"]: r for r in rows}
    reply = chat(SCAN_PROMPT.format(labels="\n".join(f"{r['latent']}: {r['label']}" for r in rows)), max_tokens=6000)
    picked = []
    for item in json_from(reply) or []:
        try:
            latent = int(item.get("latent", -1))
        except (TypeError, ValueError, AttributeError):
            continue
        if latent in by_latent:
            picked.append({**by_latent[latent], "persona": str(item.get("persona", "")).strip()})
    return picked
