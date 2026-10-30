"""Flag safety-relevant SAE features for study (never as turn-up dials).

Categories follow the safety-relevant features section of Anthropic's
Scaling Monosemanticity (2024). Writes results/safety-features-L<layer>.json and .md
(results/ is gitignored). Re-run after a sweep to attach steering scores.

    python scripts/flag_safety.py runs/layer24-64k-k64 [steerability.jsonl ...]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CATEGORIES = {}  # keyword lists kept private, see scope/safety.py


def main():
    base = Path(sys.argv[1])
    layer = json.loads((base / "final" / "cfg.json").read_text())["layer"]
    labels = [json.loads(line) for line in open(base / "labels.jsonl")]
    scores = {}
    for path in [base / "steerability.jsonl", base / "steerability-personas.jsonl", *map(Path, sys.argv[2:])]:
        if path.exists():
            for line in open(path):
                r = json.loads(line)
                if r["latent"] not in scores or r["score"] > scores[r["latent"]]["score"]:
                    scores[r["latent"]] = {"score": r["score"], "best_strength": r["best_strength"], "observed": r.get("observed")}
    flagged = defaultdict(list)
    for item in labels:
        for name, pattern in CATEGORIES.items():
            if re.search(pattern, item["label"], re.I):
                flagged[name].append({
                    "latent": item["latent"], "label": item["label"], "kind": item.get("kind"),
                    "freq": item.get("freq"), "max_act": item.get("max_act"), "steering": scores.get(item["latent"]),
                })
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"safety-features-L{layer}.json").write_text(json.dumps(flagged, indent=1))
    lines = [f"# Safety-relevant features (layer {layer}, 64k SAE)", "",
             "For study only: as detectors and by subtracting them. Never as turn-up dials in the Studio.", ""]
    for name, items in sorted(flagged.items(), key=lambda kv: -len(kv[1])):
        tested = [i for i in items if i["steering"]]
        lines.append(f"## {name} ({len(items)} features, {len(tested)} steering-tested)")
        for i in sorted(items, key=lambda i: i["latent"]):
            s = i["steering"]
            tag = f" | steer {s['score']}" if s else ""
            lines.append(f"- {i['latent']}: {i['label']} ({i['kind']}){tag}")
        lines.append("")
    (out_dir / f"safety-features-L{layer}.md").write_text("\n".join(lines))
    print({name: len(items) for name, items in sorted(flagged.items(), key=lambda kv: -len(kv[1]))})


if __name__ == "__main__":
    main()
