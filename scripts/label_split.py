"""Label the features still without a label, split by the judge's screen: clean ones by Muse Spark,
sensitive ones by a local model. Runs on a workstation, not the GPU box.

    python scripts/label_split.py spark 24,15,27,33     # META_API_KEY in the environment
    python scripts/label_split.py local 24,15,27,33     # a local model

Writes {run}/labels-spark.jsonl or labels-local.jsonl in the Mac copy of the runs, every label with
the model that wrote it, and skips features already labeled in labels.jsonl or either file, so a
rerun only does what is left. A Spark content-filter rejection stops the Spark side for good.
"""
import os
import gzip
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scope.label import PROMPT, describe  # noqa: E402
from scope.llm import MODEL as SPARK_MODEL, chat as spark_chat, json_from  # noqa: E402
from scope.safety import screen  # noqa: E402

RUNS = Path(os.environ.get("GLIMMER_RUNS", "runs")).expanduser()
LOCAL = os.environ.get("LOCAL_LLM_URL", "http://localhost:8080/v1/chat/completions")
LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "local-model")
BATCH = 10
STOP: list = []


def local_chat(prompt: str) -> str:
    body = json.dumps({"model": LOCAL_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 3000,
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


def todo(layer: int, target: str) -> tuple[Path, list[dict]]:
    base = RUNS / f"layer{layer}-64k-k64"
    done = set()
    for name in ("labels.jsonl", "labels-spark.jsonl", "labels-local.jsonl"):
        if (base / name).exists():
            done |= {json.loads(line)["latent"] for line in open(base / name)}
    rows = []
    with gzip.open(base / "catalog.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["latent"] not in done and bool(screen(describe(row))) == (target == "local"):
                rows.append(row)
    return base, rows


def label(rows: list[dict], ask, labeler: str) -> list[dict]:
    by_latent = {r["latent"]: r for r in rows}
    labeled = []
    for item in json_from(ask(PROMPT.format(features="\n\n".join(describe(r) for r in rows)))):
        try:
            latent = int(item.get("latent", -1))
        except (TypeError, ValueError):
            continue
        if latent in by_latent:
            labeled.append({"latent": latent, "label": str(item.get("label", "")).strip(),
                            "kind": str(item.get("kind", "unclear")).strip(), "steer": int(item.get("steer", 0) or 0),
                            "freq": by_latent[latent]["freq"], "max_act": by_latent[latent]["max_act"], "labeler": labeler})
    return labeled


def main():
    target, layers = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    ask, labeler, workers = (spark_chat, SPARK_MODEL, 24) if target == "spark" else (local_chat, LOCAL_MODEL, 4)
    jobs, outs, failures = [], {}, {}
    for layer in layers:
        base, rows = todo(layer, target)
        print(f"layer {layer}: {len(rows)} to label with {labeler}", flush=True)
        outs[layer] = open(base / f"labels-{target}.jsonl", "a")
        failures[layer] = open(base / f"label-failures-{target}.jsonl", "a")
        jobs += [(layer, rows[i:i + BATCH]) for i in range(0, len(rows), BATCH)]

    def run(job):
        layer, rows = job
        if STOP:
            raise RuntimeError("not sent: Spark stopped after a filter rejection")
        try:
            return label(rows, ask, labeler)
        except RuntimeError as error:
            if "content_policy" in str(error):
                STOP.append(rows[0]["latent"])
                with open(RUNS / "filter-strikes-mac.log", "a") as strikes:
                    strikes.write(f"label layer {layer} latents {[r['latent'] for r in rows]}\n")
            raise

    written = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            layer, rows = futures[future]
            try:
                labeled = future.result()
            except Exception as error:
                failures[layer].write(json.dumps({"latents": [r["latent"] for r in rows], "reason": str(error)[:400]}) + "\n")
                failures[layer].flush()
                continue
            for item in labeled:
                outs[layer].write(json.dumps(item) + "\n")
            outs[layer].flush()
            written += len(labeled)
            if written % 500 < BATCH:
                print(f"{time.strftime('%H:%M')} labeled {written}", flush=True)
    print(f"done: {written} labeled with {labeler}" + (" (Spark stopped after a filter rejection)" if STOP else ""), flush=True)


if __name__ == "__main__":
    main()
