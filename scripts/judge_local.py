"""Grade the steering replies Spark does not see, with a local model.

A sweep keeps every reply it could not send: *-screened.jsonl (sensitive, never sent), *-unjudged.jsonl
(judging paused, no label, Spark failing) and *-filtered.jsonl (Meta rejected it). This grades those
with the sweep's own judge prompt and scoring, and writes {run}/steerability-local.jsonl in the Mac
copy of the runs, each row with the model that judged it. Reruns skip what is already graded.

    python scripts/judge_local.py 30,21,18,27,15,33        # needs torch (scope.steerability)
    python scripts/judge_local.py 30,21 --personas          # the persona tests' kept replies
"""
import os
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scope.llm import json_from  # noqa: E402
from scope.steerability import judge_prompt, score  # noqa: E402

RUNS = Path(os.environ.get("GLIMMER_RUNS", "runs")).expanduser()
LOCAL = os.environ.get("LOCAL_LLM_URL", "http://localhost:8080/v1/chat/completions")
JUDGE = os.environ.get("LOCAL_LLM_MODEL", "local-model")
KEPT = ("steerability-screened.jsonl", "steerability-unjudged.jsonl", "steerability-filtered.jsonl")


def ask(prompt: str) -> str:
    body = json.dumps({"model": JUDGE, "messages": [{"role": "user", "content": prompt}], "max_tokens": 3000,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    for attempt in range(6):
        try:
            request = urllib.request.Request(LOCAL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.load(response)["choices"][0]["message"]["content"]
        except Exception:
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def jobs(layer: int, prefix: str = "steerability") -> list[dict]:
    base = RUNS / f"layer{layer}-64k-k64"
    done = set()
    for name in (f"{prefix}.jsonl", f"{prefix}-regraded.jsonl", f"{prefix}-local.jsonl"):
        if (base / name).exists():
            done |= {json.loads(line)["latent"] for line in open(base / name)}
    labels = {}
    if (base / "labels.jsonl").exists():
        labels = {r["latent"]: r["label"] for r in map(json.loads, open(base / "labels.jsonl"))}
    found = {}
    for name in (k.replace("steerability", prefix, 1) for k in KEPT):
        if (base / name).exists():
            for row in map(json.loads, open(base / name)):
                if "steered" in row and row["latent"] not in done:
                    row["label"] = row.get("label") or labels.get(row["latent"], "")
                    if row["label"]:
                        found[row["latent"]] = row
    return list(found.values())


def grade(row: dict) -> dict:
    feature = {"latent": row["latent"], "name": row.get("name"), "label": row["label"]}
    steered = {float(s): answers for s, answers in row["steered"].items()}
    verdict = json_from(ask(judge_prompt(feature, row["contexts"], row["baseline"], steered)))
    value, strength = score(verdict)
    return {"latent": row["latent"], "name": row.get("name"), "label": row["label"], "score": round(value, 3),
            "best_strength": strength, "observed": verdict.get("observed", ""), "ratings": verdict.get("strengths", []),
            "samples": row["steered"], "judge": JUDGE}


def main():
    layers = [int(x) for x in sys.argv[1].split(",")]
    prefix = "steerability-personas" if "--personas" in sys.argv else "steerability"
    rest = [a for a in sys.argv[2:] if a != "--personas"]
    limit = int(rest[0]) if rest else None
    work = [(layer, row) for layer in layers for row in jobs(layer, prefix)][:limit]
    print(f"{len(work)} kept replies to grade with {JUDGE}: " +
          ", ".join(f"L{layer} {sum(w[0] == layer for w in work)}" for layer in layers), flush=True)
    outs = {layer: open(RUNS / f"layer{layer}-64k-k64" / f"{prefix}-local.jsonl", "a") for layer in layers}
    graded = failed = 0
    with ThreadPoolExecutor(4) as pool:
        futures = {pool.submit(grade, row): layer for layer, row in work}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                failed += 1
                print(f"not graded: {str(error)[:160]}", flush=True)
                continue
            outs[futures[future]].write(json.dumps(result) + "\n")
            outs[futures[future]].flush()
            graded += 1
            if graded % 100 == 0:
                print(f"{time.strftime('%H:%M')} graded {graded}", flush=True)
    print(f"done: {graded} graded, {failed} not", flush=True)


if __name__ == "__main__":
    main()
