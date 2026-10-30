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


def candidates(catalog_path: str, min_freq: float = 1e-6, max_freq: float = 1e-2) -> Iterator[dict]:
    """Features that fire often enough to judge and rarely enough to mean something."""
    with gzip.open(catalog_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if min_freq <= row["freq"] <= max_freq and len(row["examples"]) >= 4:
                yield row


def batch_prompt(rows: list[dict]) -> str:
    return PROMPT.format(features="\n\n".join(describe(r) for r in rows))


def label_batch(rows: list[dict]) -> list[dict]:
    return parse_labels(chat(batch_prompt(rows)), rows)


def parse_labels(reply: str, rows: list[dict]) -> list[dict]:
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


def label_catalog(catalog_path: str, out_path: str, limit: int, batch: int = 10, workers: int = 16, done: set | None = None) -> int:
    done = done or set()
    rows = [r for r in candidates(catalog_path) if r["latent"] not in done][:limit]
    batches = [rows[i : i + batch] for i in range(0, len(rows), batch)]
    written = 0
    with open(out_path, "a") as out, ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(label_batch, b) for b in batches]
        for future in as_completed(futures):
            try:
                labeled = future.result()
            except Exception as error:  # one bad batch must not lose the rest
                print(f"batch failed: {error}", flush=True)
                continue
            for item in labeled:
                out.write(json.dumps(item) + "\n")
            out.flush()
            written += len(labeled)
            if written and written % 500 < batch:
                print(f"labeled {written}", flush=True)
    return written


def label_catalog_local(llm, catalog_path: str, out_path: str, limit: int, batch: int = 10, done: set | None = None) -> int:
    """label_catalog with a LocalLLM: the same prompts, answered on this GPU, tagged "by": "glimmer"."""
    done = done or set()
    rows = [r for r in candidates(catalog_path) if r["latent"] not in done][:limit]
    batches = [rows[i : i + batch] for i in range(0, len(rows), batch)]
    written, failed, step = 0, 0, llm.max_batch * 4
    with open(out_path, "a") as out:
        for start in range(0, len(batches), step):
            group = batches[start : start + step]
            replies = llm.complete([batch_prompt(b) for b in group], max_new_tokens=60 * batch)
            for rows_in, reply in zip(group, replies):
                try:
                    labeled = parse_labels(reply, rows_in)
                except Exception:  # one unparseable reply must not lose the rest
                    failed += 1
                    continue
                for item in labeled:
                    out.write(json.dumps({**item, "by": "glimmer"}) + "\n")
                written += len(labeled)
            out.flush()
            print(f"labeled {written} of {len(rows)} ({failed} replies unparseable)", flush=True)
    return written
