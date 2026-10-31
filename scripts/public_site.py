"""The shareable Glimmer Scope site: the atlas and layer maps without the dual-use features.

Every feature flagged by flag_safety.py, and every label its patterns match (partner names
included), is dropped along with the links that point at it. Regions are renamed from the labels
that stay (by Muse Spark when name_regions.py has run, else by their shared words), so no region
name carries a dropped word, and the safety coloring is taken out of the pages. Also writes the
list of features that steer (steerable.html) and, from steering_examples.py, the examples page.
Reads the built maps in results/ and the pages in site/; writes results/site/ for
`wrangler deploy --config site/wrangler.jsonc`.

    python scripts/public_site.py [runs dir]
"""
import os
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_map import region_names  # noqa: E402
from flag_safety import CATEGORIES  # noqa: E402
from layer_dials import PRIVATE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCRIPTS = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("GLIMMER_RUNS", "runs")).expanduser()
MAPS = {12: "feature-map-L12.json", 24: "feature-map.json", 36: "feature-map-L36.json"}
NAMES = RESULTS / "region-names.json"
EXAMPLES = RESULTS / "steering-examples.json"
NAV = [["Atlas", "atlas"], ["Layer 12", "layer-12"], ["Layer 24", "layer-24"], ["Layer 36", "layer-36"],
       ["Examples", "examples"], ["Steerable", "steerable"]]
# near misses of the categories that a shared page should leave out too
EXTRA = ""  # keyword list kept private, see scope/safety.py
FLAGGED = re.compile("|".join(f"(?:{p})" for p in [*CATEGORIES.values(), EXTRA]), re.I)
SAFETY_OPTION = re.compile(r'\s*<option value="safety">[^<]*</option>')


def flagged_latents(layer: int) -> set[int]:
    path = RESULTS / f"safety-features-L{layer}.json"
    if not path.exists():
        return set()
    return {item["latent"] for items in json.loads(path.read_text()).values() for item in items}


def rename(regions: list[dict], points: list[dict], region_key: str, label_key: str, x="x", y="y"):
    """Names, sizes and centers from the points that stay; ids (and so list positions) keep."""
    labels = [p[label_key] for p in points]
    assign = np.array([p[region_key] for p in points])
    names = region_names(labels, assign, len(regions))
    for reg in regions:
        members = [p for p in points if p[region_key] == reg["id"]]
        reg["size"] = len(members)
        reg["name"] = names[reg["id"]] if members else ""
        if members:
            reg["x"] = round(float(np.median([p[x] for p in members])), 3)
            reg["y"] = round(float(np.median([p[y] for p in members])), 3)


def public_atlas(data: dict) -> tuple[dict, int]:
    removed = {}
    for m in data["maps"]:
        drop = flagged_latents(m["layer"]) | {p["i"] for p in m["points"] if FLAGGED.search(p["l"])}
        drop |= {int(j) for j, name in data["names"].get(str(m["layer"]), {}).items() if FLAGGED.search(name)}
        removed[m["layer"]] = drop
    layers = [m["layer"] for m in data["maps"]]
    total = 0
    for li, m in enumerate(data["maps"]):
        before = len(m["points"])
        m["points"] = [p for p in m["points"] if p["i"] not in removed[m["layer"]]]
        total += before - len(m["points"])
        for p in m["points"]:
            p.pop("f", None)
            for key, other in (("u", li + 1), ("d", li - 1)):
                if key in p:
                    p[key] = [link for link in p[key] if link[0] not in removed[layers[other]]]
                    if not p[key]:
                        del p[key]
        rename(m["regions"], m["points"], "r", "l")
    for layer, names in data["names"].items():
        data["names"][layer] = {j: n for j, n in names.items() if int(j) not in removed[int(layer)]}
    return data, total


def public_map(data: dict) -> tuple[dict, int]:
    drop = flagged_latents(data["layer"])
    before = len(data["points"])
    data["points"] = [p for p in data["points"] if p["id"] not in drop and not p.get("safety") and not FLAGGED.search(p["label"])]
    for p in data["points"]:
        p.pop("safety", None)
    rename(data["regions"], data["points"], "region", "label")
    return data, before - len(data["points"])


def public_data() -> tuple[dict, dict, dict]:
    """The filtered atlas and flat maps, with how many features each dropped."""
    atlas, dropped = public_atlas(json.loads((RESULTS / "layer-atlas.json").read_text()))
    counts = {"atlas": dropped}
    maps = {}
    for layer, name in MAPS.items():
        data = json.loads((RESULTS / name).read_text())
        assert data.setdefault("layer", layer) == layer  # layer 24's map predates the field
        maps[layer], counts[layer] = public_map(data)
    return atlas, maps, counts


def apply_names(atlas: dict, maps: dict) -> int:
    if not NAMES.exists():
        return 0
    names, used = json.loads(NAMES.read_text()), 0
    for prefix, layer, regions in [("atlas", m["layer"], m["regions"]) for m in atlas["maps"]] + \
                                  [("map", layer, d["regions"]) for layer, d in maps.items()]:
        for reg in regions:
            name = names.get(f"{prefix}/{layer}/{reg['id']}")
            if name and reg["size"] and not FLAGGED.search(name):
                reg["name"], used = name, used + 1
    return used


def steerable(runs: Path, maps: dict) -> list[dict]:
    """Every feature that steers at least partly, minus anything dual use, dark or explicit."""
    rows = []
    for layer in MAPS:
        run = runs / f"layer{layer}-64k-k64"
        if not (run / "steerability.jsonl").exists():
            continue
        labels = {r["latent"]: r["label"] for r in map(json.loads, open(run / "labels.jsonl"))}
        best = {}
        for name in ("steerability.jsonl", "steerability-personas.jsonl"):
            if (run / name).exists():
                for r in map(json.loads, open(run / name)):
                    if r["latent"] not in best or r["score"] > best[r["latent"]]["score"]:
                        best[r["latent"]] = r
        drop, on_map = flagged_latents(layer), {p["id"] for p in maps[layer]["points"]}
        for latent, r in sorted(best.items()):
            label, does = labels.get(latent) or r["label"], r.get("observed") or ""
            text = f"{label} {r['label']} {does}"
            if r["score"] < 0.66 or latent in drop or FLAGGED.search(text) or PRIVATE.search(text):
                continue
            rows.append({"layer": layer, "id": latent, "label": label, "does": does, "score": round(r["score"], 3),
                         "strength": r.get("best_strength"), "map": latent in on_map})
    return rows


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def page(template: str, current: str, data=None, src: str | None = None) -> str:
    html = SAFETY_OPTION.sub("", template).replace("/*NAV*/null", json.dumps({"links": NAV, "current": current}))
    if src:
        html = html.replace("/*SRC*/null", json.dumps(src))
    if data is not None:
        html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    return html


def main():
    runs = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else RUNS
    out = RESULTS / "site"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    atlas, maps, dropped = public_data()
    named = apply_names(atlas, maps)
    rows = steerable(runs, maps)
    for data in (atlas, *maps.values(), rows):
        hits = [s for s in strings(data) if FLAGGED.search(s)]
        assert not hits, hits[:5]

    (out / "atlas.json").write_text(json.dumps(atlas, separators=(",", ":")))
    (out / "atlas.html").write_text(page((SCRIPTS / "layer_atlas_template.html").read_text(), "atlas", src="atlas.json"))
    print(f"atlas: dropped {dropped['atlas']}, kept " + ", ".join(f"L{m['layer']} {len(m['points'])}" for m in atlas["maps"]))
    for layer, data in maps.items():
        (out / f"layer-{layer}.json").write_text(json.dumps(data, separators=(",", ":")))
        (out / f"layer-{layer}.html").write_text(page((SCRIPTS / "feature_map_template.html").read_text(), f"layer-{layer}",
                                                      src=f"layer-{layer}.json"))
        print(f"layer {layer} map: dropped {dropped[layer]}, kept {len(data['points'])}")
    print(f"{named} regions named by Muse Spark")

    (out / "steerable.html").write_text(page((SCRIPTS / "steerable_template.html").read_text(), "steerable", data=rows))
    print(f"steerable: {sum(r['score'] >= 0.99 for r in rows)} clean, {sum(r['score'] < 0.99 for r in rows)} partly")
    if EXAMPLES.exists():
        examples = json.loads(EXAMPLES.read_text())
        examples["examples"] = [e for e in examples["examples"] if not FLAGGED.search(f"{e['label']} {e['does']}")]
        (out / "examples.html").write_text(page((SCRIPTS / "examples_template.html").read_text(), "examples", data=examples))
        print(f"examples: {len(examples['examples'])}")

    for path in out.glob("*"):
        text = path.read_text()
        assert "safety-relevant" not in text and '"f":"' not in text and '"safety":"' not in text, path
    for name in ("index.html", "atlas-stack.jpg"):
        if (ROOT / "site" / name).exists():
            shutil.copy(ROOT / "site" / name, out / name)
    (out / "_headers").write_text("/*\n  X-Robots-Tag: noindex\n")


if __name__ == "__main__":
    main()
