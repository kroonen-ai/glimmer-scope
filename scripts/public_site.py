"""The shareable Glimmer Scope site: the atlas and layer maps without the dual-use features.

Every feature flagged by flag_safety.py, and every label its patterns match (partner names
included), is dropped along with the links that point at it. Regions are renamed from the labels
that stay, so no region name carries a dropped word, and the safety coloring is taken out of the
pages. Reads the built maps in results/ and site/index.html; writes results/site/, ready for
`wrangler pages deploy`.

    python scripts/public_site.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_map import region_names  # noqa: E402
from flag_safety import CATEGORIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
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


def page(template: str, data: dict) -> str:
    return SAFETY_OPTION.sub("", template).replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))


def main():
    out = RESULTS / "site"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    scripts = Path(__file__).resolve().parent
    atlas, dropped = public_atlas(json.loads((RESULTS / "layer-atlas.json").read_text()))
    (out / "atlas.html").write_text(page((scripts / "layer_atlas_template.html").read_text(), atlas))
    print(f"atlas: dropped {dropped}, kept " + ", ".join(f"L{m['layer']} {len(m['points'])}" for m in atlas["maps"]))
    stats = {"layers": {m["layer"]: {"concept": len(m["points"]), "labeled": m["labeled"]} for m in atlas["maps"]}}
    for name, layer in (("feature-map.json", 24), ("feature-map-L36.json", 36)):
        data = json.loads((RESULTS / name).read_text())
        assert data.setdefault("layer", layer) == layer  # layer 24's map predates the field
        data, dropped = public_map(data)
        (out / f"layer-{layer}.html").write_text(page((scripts / "feature_map_template.html").read_text(), data))
        print(f"layer {layer} map: dropped {dropped}, kept {len(data['points'])}")
    for path in out.glob("*.html"):
        text = path.read_text()
        assert "safety-relevant" not in text and '"f":"' not in text and '"safety":"' not in text, path
    shutil.copy(ROOT / "site" / "index.html", out / "index.html")
    (out / "_headers").write_text("/*\n  X-Robots-Tag: noindex\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
