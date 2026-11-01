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
from .safety import screen
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


class Screened(Exception):
    """A judge prompt that touches sensitive subjects: never sent to Muse Spark (see scope.safety.screen)."""


class Filtered(Exception):
    """Meta's content filter rejected a judge request; each one counts against the API key."""


FILTER_LIMIT = 1  # a sweep stops itself, and every queue, at the first filter rejection
STRIKE_BUDGET = 6  # rejections the current key may take across all runs before judging refuses to start
STRIKES = "/scope/filter-strikes.log"
_filtered = 0


def strikes() -> int:
    import os

    return sum(1 for _ in open(STRIKES)) if os.path.exists(STRIKES) else 0


def judge(feature: dict, contexts: list[str], baseline: list[str], steered: dict[float, list[str]]) -> dict:
    if strikes() >= STRIKE_BUDGET:
        raise RuntimeError(f"Meta's filter has rejected {strikes()} requests on this key; judging is off until Robin decides")
    prompt = judge_prompt(feature, contexts, baseline, steered)
    if term := screen(prompt):
        # not sent to Meta now, but kept whole: label, contexts, baseline and every steered reply,
        # so it can be judged later without generating anything again
        raise Screened(json.dumps({"latent": feature.get("latent"), "name": feature.get("name"), "label": feature.get("label", ""),
                                   "reason": f"screened: the judge request mentions {term!r}, not sent to Muse Spark yet",
                                   "contexts": contexts, "baseline": baseline,
                                   "steered": {str(s): answers for s, answers in steered.items()}}))
    try:
        verdict = json_from(chat(prompt))
    except RuntimeError as error:
        if "content_policy" in str(error):
            raise Filtered(json.dumps({"latent": feature.get("latent"), "label": feature.get("label", ""),
                                       "reason": "Meta's content filter rejected the judge request", "prompt": prompt}))
        raise
    value, strength = score(verdict)
    return {
        "latent": feature.get("latent"),
        "name": feature.get("name"),
        "label": feature.get("label", ""),
        "score": round(value, 3),
        "best_strength": strength,
        "observed": verdict.get("observed", ""),
        "ratings": verdict.get("strengths", []),
        "samples": {str(s): answers for s, answers in steered.items()},
    }


def plan_rows(keys: list, n_prompts: int) -> list[tuple]:
    """(feature key, prompt index) for every generated row, feature-major."""
    return [(key, p) for key in keys for p in range(n_prompts)]


def feature_key(feature: dict):
    """Single features are keyed by latent; composite directions carry a name."""
    return feature.get("name") or feature["latent"]


def feature_direction(sae, feature: dict) -> torch.Tensor:
    direction = feature.get("direction")
    return direction if direction is not None else sae.W_dec[feature["latent"]]


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
            by_key = {feature_key(f): f for f in chunk}
            rows = plan_rows(list(by_key), len(prompts))
            steered: dict = {key: {} for key in by_key}
            directions = torch.stack(
                [feature_direction(sae, by_key[key]).detach().to("cuda", torch.float32) for key, _ in rows]
            )
            for strength in strengths:
                texts = generate(
                    model, tokenizer, [prefixes[p] for _, p in rows], layer, directions * (strength * median), max_new_tokens
                )
                for (key, _), text in zip(rows, texts):
                    steered[key].setdefault(strength, []).append(text)
            for key, feature in by_key.items():
                futures.append(pool.submit(judge, feature, contexts.get(key, []), baseline, steered[key]))
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
    except Screened as screened:
        with open(out.name.replace(".jsonl", "-screened.jsonl"), "a") as kept_out:
            kept_out.write(str(screened) + "\n")
        return 0
    except Filtered as filtered:
        # Kept on this machine to tighten the screen; never sent anywhere.
        global _filtered
        _filtered += 1
        with open(out.name.replace(".jsonl", "-filtered.jsonl"), "a") as kept_out:
            kept_out.write(str(filtered) + "\n")
        with open(STRIKES, "a") as strike_log:
            strike_log.write(json.loads(str(filtered))["label"] + "\n")
        print(f"judge failed: Meta's content filter rejected a request ({_filtered} this run)", flush=True)
        if _filtered >= FILTER_LIMIT:
            import os
            import subprocess

            print(f"STOPPING: {_filtered} filter rejections; stopping before the Meta key gets blocked", flush=True)
            out.flush()
            # whichever queue is running (queue2 handed over to queue3): stop it so nothing else is sent
            sessions = subprocess.run(["tmux", "ls", "-F", "#S"], capture_output=True, text=True).stdout.split()
            for session in sessions:
                if session.startswith("queue"):
                    subprocess.run(["tmux", "kill-session", "-t", f"={session}"], check=False)
            os._exit(3)
        return 0
    except Exception as error:
        print(f"judge failed: {error}", flush=True)
        return 0
    out.write(json.dumps(result, default=str) + "\n")
    out.flush()
    return 1


def contexts_for(row: dict, n: int = 3) -> list[str]:
    return [snippet(e) for e in row["examples"][:n]]
