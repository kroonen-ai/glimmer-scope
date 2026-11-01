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
import datetime
import gzip
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_map import region_names  # noqa: E402
from scope.safety import FLAGGED  # noqa: E402
from layer_dials import PRIVATE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCRIPTS = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("GLIMMER_RUNS", "runs")).expanduser()
# every layer whose map is built (feature_map.py names layer 24's map without a suffix)
MAPS = {layer: name for layer in range(1, 52)
        if (RESULTS / (name := "feature-map.json" if layer == 24 else f"feature-map-L{layer}.json")).exists()}
NAMES = RESULTS / "region-names.json"
EXAMPLES = RESULTS / "steering-examples.json"
# the layer maps sit in one compact group, so nine layers fit the header
NAV = [["Overview", "./"], ["Atlas", "atlas"], ["Layers", [[str(layer), f"layer-{layer}"] for layer in MAPS]],
       ["Examples", "examples"], ["Steerable", "steerable"]]
DEPTH = 52  # Glimmer's decoder layers
SITE = ROOT / "site"
# Studio's private map pages link back to Studio; llama-server wants the .html
STUDIO_NAV = [["Studio", "./"], ["Atlas", "atlas.html"], ["Layers", [[str(layer), f"layer-{layer}.html"] for layer in MAPS]]]
# near misses of the categories that a shared page should leave out too
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


def public_data(full: bool = False) -> tuple[dict, dict, dict]:
    """The atlas and flat maps with how many features each dropped: none when full (the site behind
    Cloudflare Access), the dual-use ones otherwise."""
    atlas = json.loads((RESULTS / "layer-atlas.json").read_text())
    atlas, dropped = (atlas, 0) if full else public_atlas(atlas)
    counts = {"atlas": dropped}
    maps = {}
    for layer, name in MAPS.items():
        data = json.loads((RESULTS / name).read_text())
        assert data.setdefault("layer", layer) == layer  # layer 24's map predates the field
        maps[layer], counts[layer] = (data, 0) if full else public_map(data)
        if full:  # names from the labels as they are now (labels keep arriving after a map is built)
            rename(data["regions"], data["points"], "region", "label")
    return atlas, maps, counts


def apply_names(atlas: dict, maps: dict, full: bool = False) -> int:
    if not NAMES.exists():
        return 0
    names, used = json.loads(NAMES.read_text()), 0
    for prefix, layer, regions in [("atlas", m["layer"], m["regions"]) for m in atlas["maps"]] + \
                                  [("map", layer, d["regions"]) for layer, d in maps.items()]:
        for reg in regions:
            name = names.get(f"{prefix}/{layer}/{reg['id']}")
            if name and reg["size"] and (full or not FLAGGED.search(name)):
                reg["name"], used = name, used + 1
    return used


def steerable(runs: Path, maps: dict, full: bool = False) -> list[dict]:
    """Every feature that steers at least partly; unless full, minus anything dual use, dark or explicit."""
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
            if r["score"] < 0.66 or (not full and (latent in drop or FLAGGED.search(text) or PRIVATE.search(text))):
                continue
            rows.append({"layer": layer, "id": latent, "label": label, "does": does, "score": round(r["score"], 3),
                         "strength": r.get("best_strength"), "map": latent in on_map})
    return rows


def layer_stats(runs: Path) -> dict:
    """Per layer with an SAE: features that fired, labeled, steering-tested, and how many steer."""
    stats = {}
    for layer in range(1, DEPTH):
        run = runs / f"layer{layer}-64k-k64"
        if not (run / "final" / "cfg.json").exists():
            continue
        fired = sum(1 for _ in gzip.open(run / "catalog.jsonl.gz", "rt")) if (run / "catalog.jsonl.gz").exists() else 0
        labeled = len({json.loads(line)["latent"] for line in open(run / "labels.jsonl")}) if (run / "labels.jsonl").exists() else 0
        best = {}
        for name in ("steerability.jsonl", "steerability-personas.jsonl"):
            if (run / name).exists():
                for r in map(json.loads, open(run / name)):
                    best[r["latent"]] = max(best.get(r["latent"], 0.0), r["score"])
        stats[layer] = {"fired": fired, "labeled": min(labeled, fired) if fired else labeled, "tested": len(best),
                        "clean": sum(v >= 0.99 for v in best.values()), "partly": sum(0.66 <= v < 0.99 for v in best.values())}
    return stats


def and_list(items) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def overview(template: str, stats: dict, atlas: dict) -> str:
    layers = sorted(stats)
    total = {k: sum(v[k] for v in stats.values()) for k in ("fired", "labeled", "tested", "clean")}
    span = f"{layers[0]} to {layers[-1]}"
    stat = lambda big, small: f'    <div class="stat"><b>{big}</b><span>{small}</span></div>'  # noqa: E731
    cards, rows = [], []
    for layer in layers:
        st, pct = stats[layer], round(layer / DEPTH * 100)
        tested = f"{st['tested']:,} steering-tested" if st["tested"] else "steering test in progress"
        if layer in MAPS:
            cards.append(f'      <a href="layer-{layer}"><span class="tag">Map</span><b>Layer {layer}</b>'
                         f'<span>{pct}% of the way in. {st["fired"]:,} features, {tested}.</span><em>Open layer {layer}</em></a>')
        pending = '<td class="pending">in progress</td>'
        rows.append(f"          <tr><td>{layer}</td><td>{pct}%</td><td>{st['fired']:,}</td>"
                    + (f"<td>{st['labeled']:,}</td>" if st["labeled"] >= st["fired"] else f'<td class="pending">{st["labeled"]:,}</td>')
                    + (f"<td>{st['tested']:,}</td><td class=\"clean\">{st['clean']:,}</td><td>{st['partly']:,}</td>" if st["tested"] else pending * 3)
                    + "</tr>")
    rates = [f"{st['clean'] / st['tested']:.0%} at layer {layer}" for layer, st in sorted(stats.items()) if st["tested"] >= 2000]
    atlas_layers = [m["layer"] for m in atlas["maps"]]
    fill = {
        "LEDE": f"Sparse autoencoders on {len(layers)} layers of Meta's open weight 30B model, every third from {span}: "
                "each feature that fired, what it responds to, and whether turning it up steers the model.",
        "STATS": "\n".join([stat(len(layers), f"layers, {span}"), stat(f"{total['fired']:,}", "features that fired"),
                            stat(f"{total['labeled']:,}", "labeled so far"),
                            stat(f"{total['tested']:,}", f"steering-tested, {total['clean']:,} steer cleanly")]),
        "STEP_TRAIN": f"A sparse autoencoder on each of {len(layers)} layers of Glimmer's residual stream, every third from "
                      f"{span}, splits what the model computes into 65,536 features.",
        "ATLAS_CARD": f"Layers {and_list(atlas_layers)} aligned. Slide between them or stack them, and click a feature to follow it up and down.",
        "LAYER_CARDS": "\n".join(cards),
        "TABLE_ROWS": "\n".join(rows),
        "DEPTH_NOTE": ("How well features steer depends on depth. Of the features tested so far, " + and_list(rates)
                       + " steer cleanly. The tests start with the features most likely to steer; layers marked in progress are being labeled and tested now.")
                      if rates else "",
        "UPDATED": datetime.date.today().strftime("%B %-d, %Y"),
    }
    for key, value in fill.items():
        template = template.replace(f"<!--{key}-->", value)
    assert "<!--" not in re.sub(r"<!--(THEME|HEADER|BARHEAD|SAFETY_NOTE)-->", "", template), "an overview placeholder is left"
    return template


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def brand(product: str = "Glimmer Scope") -> str:
    mark = (SITE / "brand" / "mark.svg.html").read_text().strip()
    return (f'<a class="brand" href="./" aria-label="{product}, by Kroonen AI">{mark}'
            '<span class="brand-word">kroonen<span>.ai</span></span><span class="brand-sep"></span>'
            f'<span class="brand-product">{product}</span></a>')


def nav(current: str, links=NAV, label="Glimmer Scope") -> str:
    mark = ' aria-current="page"'
    items = []
    for text, href in links:
        if isinstance(href, list):  # a group: the layer maps, by number
            inner = "".join(f'<a href="{h}" aria-label="Layer {t}"{mark if h == current else ""}>{t}</a>' for t, h in href)
            items.append(f'<span class="nav-group"><span class="nav-label">{text}</span>{inner}</span>')
        else:
            items.append(f'<a href="{href}"{mark if href == current else ""}>{text}</a>')
    return f'<nav class="site-nav" id="site-nav" aria-label="{label}">{"".join(items)}</nav>'


def page(template: str, current: str, data=None, src: str | None = None, private: bool = False, full: bool = False) -> str:
    """Fill a template: the theme and header (brand + nav) for the shared site, then its data.
    private=True is Studio's own copy: every feature, safety coloring kept, Studio's nav."""
    if private:
        head = f'<div class="bar-head">{brand("Glimmer Studio")}{nav(current, STUDIO_NAV, "Glimmer Studio")}</div>'
    else:
        template = template if full else SAFETY_OPTION.sub("", template)
        head = f'<div class="bar-head">{brand()}{nav(current)}</div>'
    header = f'<header class="site-header"><div class="inner">{brand()}{nav(current)}</div></header>'
    html = (template.replace("<!--THEME-->", '<link rel="stylesheet" href="theme.css">')
            .replace("<!--HEADER-->", header)
            .replace("<!--BARHEAD-->", head))
    if src:
        html = html.replace("/*SRC*/null", json.dumps(src))
    if data is not None:
        html = html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    return html


def main():
    # --full: nothing filtered, for glimmer.kroonen.ai behind Cloudflare Access
    full = "--full" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--full"]
    runs = Path(args[0]).expanduser() if args else RUNS
    out = RESULTS / "site"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    atlas, maps, dropped = public_data(full)
    named = apply_names(atlas, maps, full)
    rows = steerable(runs, maps, full)
    if not full:
        for data in (atlas, *maps.values(), rows):
            hits = [s for s in strings(data) if FLAGGED.search(s)]
            assert not hits, hits[:5]

    (out / "atlas.json").write_text(json.dumps(atlas, separators=(",", ":")))
    (out / "atlas.html").write_text(page((SCRIPTS / "layer_atlas_template.html").read_text(), "atlas", src="atlas.json", full=full))
    print(f"atlas: dropped {dropped['atlas']}, kept " + ", ".join(f"L{m['layer']} {len(m['points'])}" for m in atlas["maps"]))
    for layer, data in maps.items():
        (out / f"layer-{layer}.json").write_text(json.dumps(data, separators=(",", ":")))
        (out / f"layer-{layer}.html").write_text(page((SCRIPTS / "feature_map_template.html").read_text(), f"layer-{layer}",
                                                      src=f"layer-{layer}.json", full=full))
        print(f"layer {layer} map: dropped {dropped[layer]}, kept {len(data['points'])}")
    print(f"{named} regions named by Muse Spark")

    (out / "steerable.html").write_text(page((SCRIPTS / "steerable_template.html").read_text(), "steerable", data=rows))
    print(f"steerable: {sum(r['score'] >= 0.99 for r in rows)} clean, {sum(r['score'] < 0.99 for r in rows)} partly")
    if EXAMPLES.exists():
        examples = json.loads(EXAMPLES.read_text())
        if not full:
            examples["examples"] = [e for e in examples["examples"] if not FLAGGED.search(f"{e['label']} {e['does']}")]
        (out / "examples.html").write_text(page((SCRIPTS / "examples_template.html").read_text(), "examples", data=examples))
        print(f"examples: {len(examples['examples'])}")

    if not full:
        for path in out.glob("*.*"):
            text = path.read_text()
            assert "safety-relevant" not in text and '"f":"' not in text and '"safety":"' not in text, path
    note = ("This site is private: every feature is here, safety-relevant ones included, and the maps can color them."
            if full else "Features about weapons, cyberattacks, biological and chemical risk and other dual use topics are left out of these pages.")
    index = overview((SITE / "index.html").read_text(), layer_stats(runs), atlas)
    (out / "index.html").write_text(page(index.replace("<!--SAFETY_NOTE-->", note), "./"))
    for name in ("theme.css", "atlas-stack.jpg", "brand/Geist-Variable.woff2"):
        (out / name).parent.mkdir(exist_ok=True)
        shutil.copy(SITE / name, out / name)
    (out / "_headers").write_text("/*\n  X-Robots-Tag: noindex\n")


if __name__ == "__main__":
    main()
