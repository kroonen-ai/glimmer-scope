"""Name each feature from the contexts where it fires hardest."""

import gzip
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm import chat, json_from

PROMPT = """You are labeling features of a sparse autoencoder trained on the residual stream of a language model.
Each feature below comes with the text snippets where it activated most strongly. In each snippet the token
where it fired is marked [[like this]]; the number before a snippet is the activation.

For every feature, reply with:
- "label": what the feature detects, in at most 8 words, as specific as the evidence allows
- "kind": one of "concept" (a topic, entity, emotion, style, or idea), "syntax" (grammar, punctuation, structure),
  "token" (one literal token or spelling), "format" (markup, code, chat or template structure), "unclear"
- "steer": 0 to 3, how likely adding this feature during generation would push text toward a recognizable theme
  (3 = clearly, 0 = not at all; syntax, token and format features are usually 0 or 1)

Reply with only a JSON array, one object per feature, each with "latent", "label", "kind", "steer".

{features}"""


def snippet(example: dict) -> str:
    text = f"{example['left'][-160:]}[[{example['token']}]]{example['right'][:60]}"
    return " ".join(text.split())


def describe(row: dict, n_examples: int = 8) -> str:
    lines = [f"Feature {row['latent']} (fires on {row['freq'] * 100:.4f}% of tokens):"]
    for example in row["examples"][:n_examples]:
        lines.append(f"  {example['act']:.1f}: {snippet(example)}")
    return "\n".join(lines)


def candidates(catalog_path: str, min_freq: float = 1e-6, max_freq: float = 1e-2, min_examples: int = 4) -> Iterator[dict]:
    """Features that fire often enough to judge and rarely enough to mean something."""
    with gzip.open(catalog_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if min_freq <= row["freq"] <= max_freq and len(row["examples"]) >= min_examples:
                yield row


def label_batch(rows: list[dict]) -> list[dict]:
    reply = chat(PROMPT.format(features="\n\n".join(describe(r) for r in rows)))
    by_latent = {r["latent"]: r for r in rows}
    labeled = []
    for item in json_from(reply):
        latent = int(item.get("latent", -1))
        if latent not in by_latent:
            continue
        labeled.append(
            {
                "latent": latent,
                "label": str(item.get("label", "")).strip(),
                "kind": str(item.get("kind", "unclear")).strip(),
                "steer": int(item.get("steer", 0) or 0),
                "freq": by_latent[latent]["freq"],
                "max_act": by_latent[latent]["max_act"],
            }
        )
    return labeled


def label_catalog(
    catalog_path: str,
    out_path: str,
    limit: int,
    batch: int = 10,
    workers: int = 16,
    done: set | None = None,
    failures_path: str | None = None,
    filters: dict | None = None,
) -> int:
    """Label in batches; every feature a batch could not name goes to ``failures_path`` with the reason."""
    done = done or set()
    rows = [r for r in candidates(catalog_path, **(filters or {})) if r["latent"] not in done][:limit]
    batches = [rows[i : i + batch] for i in range(0, len(rows), batch)]
    written = 0
    failures = open(failures_path, "a") if failures_path else None

    def failed(latents: list[int], reason: str) -> None:
        if failures:
            failures.write(json.dumps({"latents": latents, "reason": reason[:400]}) + "\n")
            failures.flush()

    with open(out_path, "a") as out, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(label_batch, b): b for b in batches}
        for future in as_completed(futures):
            asked = [r["latent"] for r in futures[future]]
            try:
                labeled = future.result()
            except Exception as error:  # one bad batch must not lose the rest
                print(f"batch failed: {error}", flush=True)
                failed(asked, str(error))
                continue
            for item in labeled:
                out.write(json.dumps(item) + "\n")
            out.flush()
            named = {item["latent"] for item in labeled}
            if missing := [latent for latent in asked if latent not in named]:
                failed(missing, "Muse Spark's reply left these features out")
            written += len(labeled)
            if written and written % 500 < batch:
                print(f"labeled {written}", flush=True)
    if failures:
        failures.close()
    return written
