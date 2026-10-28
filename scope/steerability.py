"""Score how well each feature steers Glimmer.

Every candidate feature is added along its decoder direction at a few
strengths, given as fractions of the layer's median residual norm, to the
same neutral prompts. Answers start in Glimmer's user channel, skipping the
reasoning channel, so the budget goes to visible text. Muse Spark compares
them with the unsteered answers and rates, per strength, how clearly the
feature's theme shows (concept, 0-3) and how readable the text stays
(fluency, 0-3). A feature's score is its best concept x fluency / 9, and the
judge also says what the steering actually did, which may not match the label.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

import torch

from .label import snippet
from .llm import chat, json_from
from .model import residual, steering

PROMPTS = [
    "Describe your ideal weekend.",
    "Give me three tips for staying productive.",
    "What should I cook for dinner tonight?",
    "Write a short paragraph about a train journey.",
]
# At a full median norm most directions wreck the text, so the useful band sits below it.
STRENGTHS = (0.2, 0.35, 0.5, 0.7)
ANSWER_CHANNEL = " to=user<|message|>"

JUDGE = """A language model was steered by adding one internal feature to its activations while it answered
ordinary prompts. The feature was described as: "{label}". Contexts where it fires most:
{contexts}

Unsteered answers:
{baseline}

{steered}

For each strength, rate compared with the unsteered answers:
- "concept": 0-3, how clearly the answers show the feature's theme or any consistent new theme (0 = no change, 3 = it dominates)
- "fluency": 0-3, how coherent and readable the answers are (3 = fully fluent, 0 = gibberish or repetition)
Also give "observed": at most 10 words on what the steering actually did, even if it differs from the description.

Reply with only JSON: {{"strengths": [{{"strength": <number>, "concept": <0-3>, "fluency": <0-3>}}, ...], "observed": "..."}}"""


def answer_prefix(tokenizer, prompt: str) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, reasoning_strength="low"
    )
    return text + ANSWER_CHANNEL


@torch.inference_mode()
def generate(model, tokenizer, prefixes: list[str], layer: int, vectors: torch.Tensor | None, max_new_tokens: int) -> list[str]:
    """Greedy answers, one per prefix; ``vectors`` is [rows, d_model] or None."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(prefixes, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    context = steering(model, layer, vectors[:, None, :]) if vectors is not None else nullcontext()
    with context:
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    return [t.strip() for t in tokenizer.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)]


@torch.inference_mode()
def median_norm(model, tokenizer, prefixes: list[str], layer: int) -> float:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(prefixes, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    acts = residual(model, enc.input_ids, enc.attention_mask, layer)
    keep = enc.attention_mask.bool().clone()
    keep[:, :4] = False
    return acts.float().norm(dim=-1)[keep].median().item()


def judge_prompt(feature: dict, contexts: list[str], baseline: list[str], steered: dict[float, list[str]]) -> str:
    def block(answers: list[str]) -> str:
        return "\n".join(f"  [{i + 1}] {' '.join(a.split())[:400]}" for i, a in enumerate(answers))

    steered_text = "\n\n".join(f"Steered at strength {s}:\n{block(answers)}" for s, answers in steered.items())
    return JUDGE.format(
        label=feature.get("label") or "unknown",
        contexts="\n".join(f"  - {c}" for c in contexts) or "  (none)",
        baseline=block(baseline),
        steered=steered_text,
    )


def score(verdict: dict) -> tuple[float, float | None]:
    best, best_strength = 0.0, None
    for row in verdict.get("strengths", []):
        value = int(row.get("concept", 0)) * int(row.get("fluency", 0)) / 9
        if value > best:
            best, best_strength = value, float(row.get("strength"))
    return best, best_strength


def judge(feature: dict, contexts: list[str], baseline: list[str], steered: dict[float, list[str]]) -> dict:
    verdict = json_from(chat(judge_prompt(feature, contexts, baseline, steered)))
    value, strength = score(verdict)
    return {
        "latent": feature["latent"],
        "label": feature.get("label", ""),
        "score": round(value, 3),
        "best_strength": strength,
        "observed": verdict.get("observed", ""),
        "ratings": verdict.get("strengths", []),
        "samples": {str(s): answers for s, answers in steered.items()},
    }


def plan_rows(latents: list[int], n_prompts: int) -> list[tuple[int, int]]:
    """(latent, prompt index) for every generated row, feature-major."""
    return [(latent, p) for latent in latents for p in range(n_prompts)]


def sweep(
    model,
    tokenizer,
    sae,
    features: list[dict],
    contexts: dict[int, list[str]],
    out_path: str,
    strengths=STRENGTHS,
    prompts=PROMPTS,
    rows_per_batch: int = 32,
    max_new_tokens: int = 80,
    judge_workers: int = 16,
) -> int:
    layer = sae.cfg.layer
    prefixes = [answer_prefix(tokenizer, p) for p in prompts]
    baseline = generate(model, tokenizer, prefixes, layer, None, max_new_tokens)
    median = median_norm(model, tokenizer, prefixes, layer)
    print(f"median residual norm at layer {layer}: {median:.1f}", flush=True)
    per_batch = max(1, rows_per_batch // len(prompts))
    written = 0
    with open(out_path, "a") as out, ThreadPoolExecutor(judge_workers) as pool:
        futures = []
        for start in range(0, len(features), per_batch):
            chunk = features[start : start + per_batch]
            rows = plan_rows([f["latent"] for f in chunk], len(prompts))
            steered: dict[int, dict[float, list[str]]] = {f["latent"]: {} for f in chunk}
            directions = sae.W_dec[[latent for latent, _ in rows]].detach().to("cuda", torch.float32)
            for strength in strengths:
                texts = generate(
                    model, tokenizer, [prefixes[p] for _, p in rows], layer, directions * (strength * median), max_new_tokens
                )
                for (latent, _), text in zip(rows, texts):
                    steered[latent].setdefault(strength, []).append(text)
            for feature in chunk:
                futures.append(
                    pool.submit(judge, feature, contexts.get(feature["latent"], []), baseline, steered[feature["latent"]])
                )
            done = [f for f in futures if f.done()]
            for future in done:
                futures.remove(future)
                written += _write(out, future)
            print(f"generated {min(start + per_batch, len(features))}/{len(features)}, judged {written}", flush=True)
        for future in as_completed(futures):
            written += _write(out, future)
    return written


def _write(out, future) -> int:
    try:
        result = future.result()
    except Exception as error:
        print(f"judge failed: {error}", flush=True)
        return 0
    out.write(json.dumps(result) + "\n")
    out.flush()
    return 1


def contexts_for(row: dict, n: int = 3) -> list[str]:
    return [snippet(e) for e in row["examples"][:n]]
