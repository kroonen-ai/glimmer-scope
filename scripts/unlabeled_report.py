"""Every feature of one SAE that has no label yet, with the reason, for checking by hand.

Reasons: never fired in the catalog (no text to label it from), the last labeling request that
covered it failed (the API's own error, e.g. a content filter or rate limit), Muse Spark's reply
left it out, or no labeling pass has reached it yet. Writes results/unlabeled-L<layer>.json and .md.

    python scripts/unlabeled_report.py runs/layer36-64k-k64
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

NEVER_FIRED = "never fired in the 5M-token catalog, so there is no text to label it from"
NOT_REACHED = "fired, but no labeling pass has reached it yet"


def main():
    run = Path(sys.argv[1]).expanduser()
    cfg = json.loads((run / "final" / "cfg.json").read_text())
    layer, n = cfg["layer"], cfg["n_latents"]
    labeled = set()
    for name in ("labels.jsonl", "labels-2.jsonl"):
        if (run / name).exists():
            labeled |= {json.loads(line)["latent"] for line in open(run / name)}
    fired = {}
    with gzip.open(run / "catalog.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            fired[row["latent"]] = row["freq"]
    last_failure = {}
    if (run / "label-failures.jsonl").exists():
        for line in open(run / "label-failures.jsonl"):
            row = json.loads(line)
            for latent in row["latents"]:
                last_failure[latent] = row["reason"]

    unlabeled = {}
    for latent in range(n):
        if latent in labeled:
            continue
        if latent not in fired:
            unlabeled[latent] = NEVER_FIRED
        else:
            unlabeled[latent] = last_failure.get(latent, NOT_REACHED)
    by_reason = defaultdict(list)
    for latent, reason in unlabeled.items():
        by_reason[reason].append(latent)

    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    rows = [{"latent": latent, "reason": reason, "freq": fired.get(latent)} for latent, reason in sorted(unlabeled.items())]
    (out / f"unlabeled-L{layer}.json").write_text(json.dumps(rows, indent=1))
    lines = [f"# Unlabeled features, layer {layer}", "",
             f"{len(labeled):,} of {n:,} features labeled, {len(unlabeled):,} not. Grouped by reason.", ""]
    for reason, latents in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        lines += [f"## {len(latents):,}: {reason}", "", ", ".join(map(str, latents)), ""]
    (out / f"unlabeled-L{layer}.md").write_text("\n".join(lines))
    print(f"layer {layer}: {len(labeled)} labeled, {len(unlabeled)} not: " +
          "; ".join(f"{len(v)} {k[:60]}" for k, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))))


if __name__ == "__main__":
    main()
