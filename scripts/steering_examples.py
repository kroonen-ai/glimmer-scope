"""Steering examples for the shareable site, generated through a local Studio.

For each chosen feature, the same everyday prompts are answered with its dial at the Studio default
strength, next to the unsteered answers: greedy, straight into the answer channel, as q4_check.py
does. Labels and what each feature does come from the run's labels and Muse Spark's steering
verdicts. Writes results/steering-examples.json for public_site.py.

    python scripts/steering_examples.py 24:64624 24:3000 36:12626 ...
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from public_site import RESULTS, RUNS  # noqa: E402
from q4_check import ANSWER_CHANNEL, call, vectors  # noqa: E402
from scope.personas import EVAL_SETS  # noqa: E402

TOKENS = 150


def tidy(text: str) -> str:
    """Drop channel tokens and end on the last whole sentence (or word) the token limit allows."""
    text = re.sub(r"<\|[a-z_]+\|>", " ", text).strip()
    cut = max(text.rfind(c) for c in ".!?")
    if cut > len(text) * 0.6:
        return text[:cut + 1]
    return text.rsplit(" ", 1)[0] + "…"


def complete(prefix: str, steering: dict) -> str:
    out = call("completion", {"prompt": prefix, "n_predict": TOKENS, "temperature": 0, "steering": steering})
    return tidy(out["content"])


def verdicts(layer: int) -> dict:
    run = RUNS / f"layer{layer}-64k-k64"
    best = {}
    for name in ("steerability.jsonl", "steerability-personas.jsonl"):
        if (run / name).exists():
            for r in map(json.loads, open(run / name)):
                if r["latent"] not in best or r["score"] > best[r["latent"]]["score"]:
                    best[r["latent"]] = r
    labels = {r["latent"]: r["label"] for r in map(json.loads, open(run / "labels.jsonl"))}
    return {latent: {**r, "label": labels.get(latent, r["label"])} for latent, r in best.items()}


def main():
    wanted = [tuple(map(int, arg.split(":"))) for arg in sys.argv[1:]]
    library = vectors()
    by_feature = {}
    for name, v in library.items():
        if v.get("group") in ("safety", "research"):  # private dials never go on the site
            continue
        try:
            by_feature[(v.get("layer", 24), int(name.rsplit("-", 1)[1]))] = (name, v)
        except (ValueError, IndexError):  # dials not named after a feature
            pass
    prompts = EVAL_SETS["everyday"]
    prefixes = [call("apply-template", {"messages": [{"role": "user", "content": p}],
                                        "chat_template_kwargs": {"reasoning_strength": "low"}})["prompt"] + ANSWER_CHANNEL
                for p in prompts]
    baseline = [complete(p, {}) for p in prefixes]
    found = {layer: verdicts(layer) for layer in {layer for layer, _ in wanted}}
    examples = []
    for layer, latent in wanted:
        if (layer, latent) not in by_feature:
            print(f"layer {layer} feature {latent}: no Studio dial, skipped")
            continue
        name, dial = by_feature[(layer, latent)]
        strength = dial.get("default", 0.5)
        verdict = found[layer][latent]
        replies = [complete(p, {name: strength}) for p in prefixes]
        examples.append({"layer": layer, "id": latent, "label": verdict["label"], "does": verdict.get("observed") or "",
                         "strength": strength, "replies": replies})
        print(f"layer {layer} feature {latent} at {strength}: {replies[0][:90]!r}", flush=True)
    out = RESULTS / "steering-examples.json"
    out.write_text(json.dumps({"prompts": prompts, "baseline": baseline, "examples": examples, "tokens": TOKENS}, indent=1))
    print(f"{len(examples)} examples -> {out}")


if __name__ == "__main__":
    main()
