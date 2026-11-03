"""Grade locally the archetype candidates the screen kept from Spark, for the archetype itself.

The archetype sweep kept every candidate's steered replies; the ones whose judge request touched the
screen (sycophant and contrarian are tested on health and science claims) went to
archetypes/<name>/steerability-final-screened.jsonl instead of Spark. This grades them with
scope.archetype's TARGET_JUDGE on a local model (scripts/judge_local.py), adds
them to each archetype's target-final.json marked with the judge, and rebuilds the layer's
targets-final.json so every archetype picks its best direction from all its candidates.

    python scripts/archetype_local.py 15,18,21,27,30,33
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from judge_local import JUDGE, RUNS, ask  # noqa: E402
from scope.archetype import ARCHETYPES, TARGET_JUDGE, slug  # noqa: E402
from scope.llm import json_from  # noqa: E402


def grade(name: str, spec: dict, row: dict) -> dict:
    blocks = []
    for strength, answers in sorted(row["steered"].items(), key=lambda kv: float(kv[0])):
        lines = "\n".join(f"  [{i + 1}] {' '.join(a.split())[:350]}" for i, a in enumerate(answers))
        blocks.append(f"Strength {strength}:\n{lines}")
    verdict = json_from(ask(TARGET_JUDGE.format(name=name, description=spec["description"], steered="\n\n".join(blocks))))
    best, best_strength = 0.0, None
    for r in verdict.get("strengths", []):
        value = int(r.get("target", 0)) * int(r.get("fluency", 0)) / 9
        if value > best:
            best, best_strength = value, float(r.get("strength"))
    return {"candidate": row.get("name") or row.get("latent"), "target_score": round(best, 3), "best_strength": best_strength,
            "verdict": verdict.get("verdict", ""), "ratings": verdict.get("strengths", []), "judge": JUDGE}


def main():
    layers = [int(x) for x in sys.argv[1].split(",")]
    work = []
    for layer in layers:
        base = RUNS / f"layer{layer}-64k-k64" / "archetypes"
        for name, spec in ARCHETYPES.items():
            folder = base / slug(name)
            kept = folder / "steerability-final-screened.jsonl"
            if not kept.exists():
                continue
            existing = json.loads((folder / "target-final.json").read_text()) if (folder / "target-final.json").exists() else []
            done = {str(r.get("candidate")) for r in existing if "target_score" in r}
            rows = {}
            for row in map(json.loads, open(kept)):
                key = str(row.get("name") or row.get("latent"))
                if "steered" in row and key not in done:
                    rows[key] = row
            work += [(layer, name, spec, row) for row in rows.values()]
    print(f"{len(work)} kept archetype candidates to grade with {JUDGE}", flush=True)

    results = {}
    with ThreadPoolExecutor(4) as pool:
        futures = {pool.submit(grade, name, spec, row): (layer, name) for layer, name, spec, row in work}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                results.setdefault(futures[future], []).append(future.result())
            except Exception as error:
                print(f"not graded: {str(error)[:160]}", flush=True)
            if n % 100 == 0:
                print(f"graded {n}", flush=True)

    for layer in layers:
        base = RUNS / f"layer{layer}-64k-k64" / "archetypes"
        for (lay, name), rows in results.items():
            if lay != layer:
                continue
            path = base / slug(name) / "target-final.json"
            existing = json.loads(path.read_text()) if path.exists() else []
            path.write_text(json.dumps(existing + rows, indent=2))
        summary = {}
        for name in ARCHETYPES:
            path = base / slug(name) / "target-final.json"
            if not path.exists():
                continue
            rows = json.loads(path.read_text())
            scored = sorted((r for r in rows if "target_score" in r), key=lambda r: -r["target_score"])
            summary[name] = {"best": scored[0] if scored else None, "all": rows}
        (base / "targets-final.json").write_text(json.dumps(summary, indent=2))
        clean = sum(1 for v in summary.values() if (v["best"] or {}).get("target_score", 0) >= 0.99)
        print(f"layer {layer}: {len(summary)} archetypes, {clean} play cleanly", flush=True)


if __name__ == "__main__":
    main()
