"""Grade with Spark the steering replies a sweep kept unjudged (judging paused, Spark failing), on the box.

Rows of {run}/steerability-unjudged.jsonl that carry their replies and a label are sent through the
sweep's judge (so the screen still holds back anything sensitive; those stay for
scripts/judge_local.py). Scores go to {run}/steerability-regraded.jsonl, never into the file a
running sweep appends to. Reruns skip what is already scored.

    python scripts/judge_kept.py 30,21 [workers]
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scope.steerability import Filtered, Screened, judge, stop_judging  # noqa: E402

RUNS = Path("/scope")


def main():
    layers = [int(x) for x in sys.argv[1].split(",")]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    work = []
    for layer in layers:
        base = RUNS / f"layer{layer}-64k-k64"
        done = set()
        for name in ("steerability.jsonl", "steerability-regraded.jsonl"):
            if (base / name).exists():
                done |= {json.loads(line)["latent"] for line in open(base / name)}
        rows = {}
        if (base / "steerability-unjudged.jsonl").exists():
            for row in map(json.loads, open(base / "steerability-unjudged.jsonl")):
                if "steered" in row and row.get("label") and row["latent"] not in done:
                    rows[row["latent"]] = row
        work += [(layer, row) for row in rows.values()]
        print(f"layer {layer}: {len(rows)} kept replies to grade", flush=True)

    def one(job):
        layer, row = job
        feature = {"latent": row["latent"], "name": row.get("name"), "label": row["label"]}
        steered = {float(s): answers for s, answers in row["steered"].items()}
        return layer, judge(feature, row["contexts"], row["baseline"], steered)

    outs = {layer: open(RUNS / f"layer{layer}-64k-k64" / "steerability-regraded.jsonl", "a") for layer in layers}
    graded = held = 0
    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, job) for job in work]):
            try:
                layer, result = future.result()
            except Filtered as filtered:
                stop_judging(json.loads(str(filtered))["label"])
                print("FILTER REJECTION: kept, judging goes on", flush=True)
                held += 1
                continue
            except Screened:  # sensitive, or Spark failing: stays for later / a local model
                held += 1
                continue
            outs[layer].write(json.dumps(result, default=str) + "\n")
            outs[layer].flush()
            graded += 1
            if graded % 500 == 0:
                print(f"graded {graded}", flush=True)
    print(f"done: {graded} graded by Spark, {held} held back", flush=True)


if __name__ == "__main__":
    main()
