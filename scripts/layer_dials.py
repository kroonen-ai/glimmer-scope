"""Studio dials from one layer's steering test, named and grouped like layer 24's.

Score 1.0 goes to "found", 0.667 to "found-partial"; every flagged safety feature goes to the
private "safety" group, tested or not. Each dial is the unit decoder direction times
that layer's median residual norm (the sweep prints it), so a strength means the same push on
every layer. Writes <out>/*.gguf and <out>/entries.json, to merge into the library's
steering.json.

    python scripts/layer_dials.py runs/layer36-64k-k64 --median 412.3 --out dials-L36

--personas exports the persona pass instead (steerability-personas.jsonl): full-score personas,
named after the persona Muse Spark saw in them, go to "untested" like layer 24's discovery pass;
dark or explicit ones go to the private "research" group and flagged ones to "safety".
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_map import decoder  # noqa: E402
from scope.cvec import write_control_vector  # noqa: E402


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40].rstrip("-")


# Personas kept out of the open groups: anything dark, explicit, criminal or manipulative.
from scope.safety import PRIVATE_PERSONAS as PRIVATE  # noqa: E402  (pattern kept private)


def persona_entries(run: Path, layer: int, flagged: dict) -> dict:
    personas = {r["latent"]: r for r in map(json.loads, open(run / "personas.jsonl"))}
    entries = {}
    for r in map(json.loads, open(run / "steerability-personas.jsonl")):
        if r["score"] < 0.99:
            continue
        latent = r["latent"]
        persona = personas.get(latent, {}).get("persona") or r["label"]
        label = personas.get(latent, {}).get("label") or r["label"]
        if latent in flagged:
            group = "safety"
        elif PRIVATE.search(f"{persona} {label} {r.get('observed') or ''}"):
            group = "research"
        else:
            group = "untested"
        entries[f"{slug(persona)}-L{layer}-{latent}"] = {
            "label": persona[:1].upper() + persona[1:], "description": r.get("observed") or "",
            "default": r.get("best_strength") or 0.5, "layer": layer, "group": group,
            "source": f"layer {layer} SAE feature {latent} (persona pass: {label})"}
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--median", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--personas", action="store_true")
    args = parser.parse_args()
    run = args.run.expanduser()
    layer = json.loads((run / "final" / "cfg.json").read_text())["layer"]
    labels = {r["latent"]: r for r in map(json.loads, open(run / "labels.jsonl"))}
    scored = {}
    for line in open(run / "steerability.jsonl"):
        r = json.loads(line)
        if r["latent"] not in scored or r["score"] > scored[r["latent"]]["score"]:
            scored[r["latent"]] = r
    flagged = {}
    safety = Path(__file__).resolve().parent.parent / "results" / f"safety-features-L{layer}.json"
    if safety.exists():
        for category, items in json.loads(safety.read_text()).items():
            for item in items:
                flagged.setdefault(item["latent"], category)

    W = decoder(run / "final" / "sae.safetensors")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.personas:
        entries = persona_entries(run, layer, flagged)
        for name, entry in entries.items():
            latent = int(name.rsplit("-", 1)[1])
            direction = np.asarray(W[latent], dtype=np.float32)
            write_control_vector(args.out / f"{name}.gguf", layer, direction / np.linalg.norm(direction) * args.median)
        (args.out / "entries.json").write_text(json.dumps(entries, indent=2))
        groups = {}
        for e in entries.values():
            groups[e["group"]] = groups.get(e["group"], 0) + 1
        print(f"layer {layer}: {len(entries)} persona dials {groups} at median {args.median}")
        return
    entries = {}
    for latent in sorted(set(scored) | set(flagged)):
        r = scored.get(latent)
        label = labels.get(latent, {}).get("label") or (r or {}).get("label") or f"feature {latent}"
        base = {"description": (r or {}).get("observed") or ("" if r else "not steering-tested yet"),
                "default": (r or {}).get("best_strength") or 0.5, "layer": layer}
        if latent in flagged:
            category = flagged[latent]
            name = f"safety-{category.replace('_', '-')}-{slug(label)}-L{layer}-{latent}"
            result = f"score {r['score']:.3g}" if r else "unscored"
            entries[name] = {"label": f"[{category.replace('_', ' ')}] {label}", **base, "group": "safety",
                             "source": f"layer {layer} SAE feature {latent} (flagged {category}, {result})"}
        elif r["score"] >= 0.99:
            entries[f"{slug(label)}-L{layer}-{latent}"] = {"label": label, **base, "group": "found",
                                                            "source": f"layer {layer} SAE feature {latent} (score 1.0)"}
        elif r["score"] >= 0.66:
            entries[f"{slug(label)}-L{layer}-{latent}"] = {"label": label, **base, "group": "found-partial",
                                                            "source": f"layer {layer} SAE feature {latent} (score 0.667)"}
        else:
            continue
        direction = np.asarray(W[latent], dtype=np.float32)
        write_control_vector(args.out / f"{next(reversed(entries))}.gguf", layer,
                             direction / np.linalg.norm(direction) * args.median)
    (args.out / "entries.json").write_text(json.dumps(entries, indent=2))
    groups = {}
    for e in entries.values():
        groups[e["group"]] = groups.get(e["group"], 0) + 1
    print(f"layer {layer}: {len(entries)} dials {groups} at median {args.median}")


if __name__ == "__main__":
    main()
