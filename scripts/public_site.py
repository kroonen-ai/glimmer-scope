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
from scope.safety import FLAGGED, screen  # noqa: E402
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


# every file that holds steering grades; Spark's win, the local model's fill in what Spark never saw
GRADES = (("steerability.jsonl", None), ("steerability-regraded.jsonl", None), ("steerability-personas.jsonl", None),
          ("steerability-local.jsonl", "local"), ("steerability-personas-local.jsonl", "local"))
LOCAL_GRADER = "a local model"  # what the site says about grades not from Spark


def grades(run: Path) -> dict[int, tuple[float, str | None, dict]]:
    """latent -> (best score, None for Spark or "local", row)."""
    best = {}
    for name, judge in GRADES:
        if not (run / name).exists():
            continue
        for r in map(json.loads, open(run / name)):
            cur = best.get(r["latent"])
            if cur is None or (judge is None and (cur[1] is not None or r["score"] > cur[0])):
                best[r["latent"]] = (r["score"], judge, r)
    return best


def refresh_labels(data: dict, run: Path) -> None:
    """A map keeps its layout, but its labels come from the run as it is now (labels keep arriving)."""
    if not (run / "labels.jsonl").exists():
        return
    labels = {r["latent"]: r for r in map(json.loads, open(run / "labels.jsonl"))}
    graded = grades(run)
    for p in data["points"]:
        r = labels.get(p["id"])
        p["label"], p["kind"] = (r["label"], r.get("kind")) if r and r.get("label") else ("unlabeled", "unlabeled")
        g = graded.get(p["id"])
        p["score"] = g[0] if g else None
        p.pop("j", None)
        if g and g[1]:
            p["j"] = 1  # graded by a local model, not Spark


def public_data(full: bool = False, runs: Path | None = None) -> tuple[dict, dict, dict]:
    """The atlas and flat maps with how many features each dropped: none when full (the site behind
    Cloudflare Access), the dual-use ones otherwise."""
    atlas = json.loads((RESULTS / "layer-atlas.json").read_text())
    if atlas.get("files"):  # the nine-layer atlas (atlas_nine.py): one file per layer
        atlas["maps"] = [json.loads((RESULTS / f).read_text()) for f in atlas["files"]]
        if full:  # region names from the labels as they are now, like the layer maps
            for m in atlas["maps"]:
                rename(m["regions"], m["points"], "r", "l")
    atlas, dropped = (atlas, 0) if full else public_atlas(atlas)
    counts = {"atlas": dropped}
    maps = {}
    for layer, name in MAPS.items():
        data = json.loads((RESULTS / name).read_text())
        assert data.setdefault("layer", layer) == layer  # layer 24's map predates the field
        if runs:
            refresh_labels(data, runs / f"layer{layer}-64k-k64")
        maps[layer], counts[layer] = (data, 0) if full else public_map(data)
        if full:  # names from the labels as they are now (labels keep arriving after a map is built)
            rename(data["regions"], data["points"], "region", "label")
    return atlas, maps, counts


def apply_names(atlas: dict, maps: dict, full: bool = False) -> int:
    if not NAMES.exists():
        return 0
    names, used = json.loads(NAMES.read_text()), 0
    # the nine-layer atlas reuses the layer maps' regions, so the old atlas names no longer fit it
    atlas_maps = [] if atlas.get("files") else atlas["maps"]
    for prefix, layer, regions in [("atlas", m["layer"], m["regions"]) for m in atlas_maps] + \
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
        graded = grades(run)
        best = {lat: g[2] for lat, g in graded.items()}
        drop, on_map = flagged_latents(layer), {p["id"] for p in maps[layer]["points"]}
        for latent, r in sorted(best.items()):
            label, does = labels.get(latent) or r["label"], r.get("observed") or ""
            text = f"{label} {r['label']} {does}"
            if r["score"] < 0.66 or (not full and (latent in drop or FLAGGED.search(text) or PRIVATE.search(text))):
                continue
            rows.append({"layer": layer, "id": latent, "label": label, "does": does, "score": round(r["score"], 3),
                         "strength": r.get("best_strength"), "map": latent in on_map,
                         **({"judge": LOCAL_GRADER} if graded[latent][1] else {})})
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
        graded = grades(run)
        best = {lat: g[0] for lat, g in graded.items()}
        spark = [g[0] for g in graded.values() if not g[1]]
        stats[layer] = {"fired": fired, "labeled": min(labeled, fired) if fired else labeled, "tested": len(best),
                        "clean": sum(v >= 0.99 for v in best.values()), "partly": sum(0.66 <= v < 0.99 for v in best.values()),
                        "local": sum(1 for g in graded.values() if g[1]),
                        "spark_tested": len(spark), "spark_clean": sum(v >= 0.99 for v in spark)}
    return stats


def archetypes(runs: Path, layer: int) -> tuple[int, int, int]:
    """(clean by Spark's grades, clean counting the local model's too, archetypes) on a layer, from targets-final.json.
    the local model grades more generously, so Spark's count is the one to compare across layers."""
    path = runs / f"layer{layer}-64k-k64" / "archetypes" / "targets-final.json"
    if not path.exists():
        return 0, 0, 0
    data = json.loads(path.read_text())
    spark = sum(max([r["target_score"] for r in v["all"] if "target_score" in r and not r.get("judge")] or [0]) >= 0.99
                for v in data.values())
    every = sum(((v.get("best") or {}).get("target_score") or 0) >= 0.99 for v in data.values())
    return spark, every, len(data)


def lifespans(atlas: dict, threshold: float = 0.3) -> dict[int, tuple[float, float, int]]:
    """Per atlas layer: share of features whose thread (strongest partner while >= threshold) runs
    through every layer, share living on one layer only, and the feature count."""
    maps = atlas["maps"]
    n = len(maps)
    ids = [{p["i"]: p for p in m["points"]} for m in maps]

    def best(li, p, key, step):
        if not 0 <= li + step < n:
            return None
        for j, c in p.get(key, []):
            if j in ids[li + step]:
                return j, c
        return None

    up = [{} for _ in range(n)]
    down = [{} for _ in range(n)]
    for li in range(n - 1, -1, -1):
        for p in maps[li]["points"]:
            b = best(li, p, "u", 1)
            up[li][p["i"]] = 1 + up[li + 1][b[0]] if b and b[1] >= threshold else 0
    for li in range(n):
        for p in maps[li]["points"]:
            b = best(li, p, "d", -1)
            down[li][p["i"]] = 1 + down[li - 1][b[0]] if b and b[1] >= threshold else 0
    out = {}
    for li, m in enumerate(maps):
        spans = [up[li][p["i"]] + down[li][p["i"]] + 1 for p in m["points"]]
        out[m["layer"]] = (sum(x == n for x in spans) / len(spans), sum(x == 1 for x in spans) / len(spans), len(spans))
    return out


def and_list(items) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def overview(template: str, stats: dict, atlas: dict, runs: Path = RUNS) -> str:
    layers = sorted(stats)
    total = {k: sum(v[k] for v in stats.values()) for k in ("fired", "labeled", "tested", "clean")}
    span = f"{layers[0]} to {layers[-1]}"
    life = lifespans(atlas) if len(atlas["maps"]) > 1 and any("u" in p for p in atlas["maps"][0]["points"][:200]) else {}
    arch = {layer: archetypes(runs, layer) for layer in layers}
    all9 = sum(v[0] * v[2] for v in life.values()) / max(1, sum(v[2] for v in life.values())) if life else 0
    one = sum(v[1] * v[2] for v in life.values()) / max(1, sum(v[2] for v in life.values())) if life else 0
    stat = lambda big, small: f'    <div class="stat"><b>{big}</b><span>{small}</span></div>'  # noqa: E731
    cards, rows = [], []
    for layer in layers:
        st, pct = stats[layer], round(layer / DEPTH * 100)
        tested = f"{st['tested']:,} steering-tested" if st["tested"] else "steering test in progress"
        if layer in MAPS:
            cards.append(f'      <a href="layer-{layer}"><span class="tag">Map</span><b>Layer {layer}</b>'
                         f'<span>{pct}% of the way in. {st["fired"]:,} features, {tested}.</span><em>Open layer {layer}</em></a>')
        pending = '<td class="pending">in progress</td>'
        clean_a, every_a, graded_a = arch[layer]
        rows.append(f"          <tr><td>{layer}</td><td>{pct}%</td><td>{st['fired']:,}</td>"
                    + (f"<td>{st['labeled']:,}</td>" if st["labeled"] >= st["fired"] else f'<td class="pending">{st["labeled"]:,}</td>')
                    + (f"<td>{st['tested']:,}</td><td class=\"clean\">{st['clean']:,}</td><td>{st['partly']:,}</td>" if st["tested"] else pending * 3)
                    + (f"<td>{clean_a} of {graded_a}" + (f' <span class="extra">+{every_a - clean_a} by {LOCAL_GRADER}</span>' if every_a > clean_a else "")
                       + "</td>" if graded_a else pending)
                    + (f"<td>{life[layer][0]:.0%}</td>" if layer in life else "<td></td>")
                    + "</tr>")
    rates = [f"{st['clean'] / st['tested']:.0%} at layer {layer}" for layer, st in sorted(stats.items()) if st["tested"] >= 2000]
    # the findings use Spark's grades alone, so a more generous second grader can't make a layer look better
    rate = {layer: st["spark_clean"] / st["spark_tested"] for layer, st in stats.items() if st["spark_tested"] >= 2000}
    top_rate = max(rate.values()) if rate else 0
    mid = sorted(layer for layer, r in rate.items() if r >= 0.75 * top_rate)
    rest = [r for layer, r in rate.items() if layer not in mid]
    best_arch = max(layers, key=lambda layer: arch[layer][0])  # by Spark's grades
    finding = lambda big, title, text: f'      <div class="finding"><b>{big}</b><h3>{title}</h3><p>{text}</p></div>'  # noqa: E731
    findings = []
    if rate:
        findings.append(finding(f"{top_rate:.0%}", "Steering peaks mid-network",
                                f"About {min(rate[x] for x in mid):.0%} to {top_rate:.0%} of tested features steer cleanly at layers "
                                f"{and_list(mid)}, against {min(rest):.0%} to {max(rest):.0%} near the start and the end "
                                "(Muse Spark's grades)."
                                if rest else f"of tested features steer cleanly at layer {mid[0]}."))
    if life:
        findings.append(finding(f"{all9:.0%}", "A stable backbone",
                                f"of features belong to a thread that runs through all {len(layers)} layers, and {one:.0%} live on one layer "
                                f"only. Shallow layers keep their concepts ({life[layers[0]][0]:.0%} of layer {layers[0]} runs all the way), deep "
                                f"ones add their own ({life[layers[-1]][1]:.0%} of layer {layers[-1]} exists there only)."))
    if arch[best_arch][2]:
        findings.append(finding(f"{arch[best_arch][0]} of {arch[best_arch][2]}", "Characters at every depth",
                                f"Character archetypes, from pirate captain to gaslighter, show up as steering directions on every layer. "
                                f"The most play cleanly at layer {best_arch}."))
    atlas_layers = [m["layer"] for m in atlas["maps"]]
    fill = {
        "LEDE": f"Sparse autoencoders on {len(layers)} layers of Meta's open weight 30B model, every third from {span}: "
                "each feature that fired, what it responds to, whether turning it up steers the model, and how each concept "
                "travels from layer to layer.",
        "STATS": "\n".join([stat(len(layers), f"layers, {span}"),
                            stat(f"{total['fired']:,}", "features, every one labeled" if total["labeled"] >= total["fired"] else
                                 f"features, {total['labeled']:,} labeled so far"),
                            stat(f"{total['tested']:,}", f"steering tests, {total['clean']:,} steer cleanly"),
                            stat(f"{all9:.0%}" if life else len(layers), "of features run through all nine layers" if life else "layers mapped")]),
        "FINDINGS": "\n".join(findings),
        "STEP_TRAIN": f"A sparse autoencoder on each of {len(layers)} layers of Glimmer's residual stream, every third from "
                      f"{span}, splits what the model computes into 65,536 features.",
        "ATLAS_CARD": (f"All {len(atlas_layers)} layers lined up, every feature on each. Click one to follow its thread from layer "
                       f"{atlas_layers[0]} to {atlas_layers[-1]}, with its splits and merges.") if atlas.get("files") else
                      f"Layers {and_list(atlas_layers)} aligned. Slide between them or stack them, and click a feature to follow it up and down.",
        "LAYER_CARDS": "\n".join(cards),
        "TABLE_ROWS": "\n".join(rows),
        "DEPTH_NOTE": ("How well features steer depends on depth. Of the features tested so far, " + and_list(rates)
                       + " steer cleanly. The tests cover the features most likely to steer on each layer"
                       + ("; layers marked in progress are being labeled and tested now." if any(
                           not st["tested"] or st["labeled"] < st["fired"] for st in stats.values()) else "."))
                      if rates else "",
        "UPDATED": datetime.date.today().strftime("%B %-d, %Y"),
        "GRADERS": (f"Muse Spark 1.3 graded {total['tested'] - sum(v['local'] for v in stats.values()):,} of the steering tests. "
                    f"The other {sum(v['local'] for v in stats.values()):,}, on features whose text Meta's content filter would stop, "
                    f"were graded by {LOCAL_GRADER} and are marked as such on the maps and in the list.")
                   if sum(v["local"] for v in stats.values()) else "",
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


# the safe-browsing switch (site/safe.js), in the header of every shared page
SAFE_TOGGLE = ('<button type="button" class="safe-toggle" aria-pressed="true" title="Hides the labels, examples and '
               'region names of features that could be upsetting or misused; their points stay on the maps">'
               'Safe browsing <b>on</b></button>')
SAFE_SCRIPT = '<script src="safe.js"></script>'
# light or dark as chosen with the switch (safe.js), set before the page paints so it never flashes
THEME_EARLY = ('<script>try{var t=localStorage.getItem("glimmer-theme");if(t)document.documentElement.dataset.theme=t}'
               'catch(e){}</script>')
THEME_TOGGLE = '<button type="button" class="theme-toggle" aria-label="Switch between light and dark">◐</button>'
# safety categories whose examples could upset or be misused; sycophancy, deception, power seeking and
# AI-self features stay visible, they are part of what the maps are for
UPSETTING = {"cbrn", "weapons_violence", "cyber", "self_harm", "drugs", "sexual", "hate_bias"}


def categories(layer: int) -> dict[int, str]:
    path = RESULTS / f"safety-features-L{layer}.json"
    if not path.exists():
        return {}
    return {item["latent"]: cat for cat, items in json.loads(path.read_text()).items() for item in items}


def mark_hidden(atlas: dict, maps: dict, rows: list, examples: dict | None) -> int:
    """Mark (h) what safe browsing hides: features in an upsetting category or whose text matches the
    judge's screen, and region names that do. The data itself stays whole."""
    hidden = 0
    for layer, data in maps.items():
        cats = categories(layer)
        for p in data["points"]:
            if cats.get(p["id"]) in UPSETTING or p.get("safety") in UPSETTING or screen(p["label"]):
                p["h"] = 1
                hidden += 1
        for reg in data["regions"]:
            if reg.get("name") and screen(reg["name"]):
                reg["h"] = 1
    atlas["hn"] = {}
    for m in atlas["maps"]:
        cats = categories(m["layer"])
        for p in m["points"]:
            if cats.get(p["i"]) in UPSETTING or p.get("f") in UPSETTING or screen(p["l"]):
                p["h"] = 1
        for reg in m["regions"]:
            if reg.get("name") and screen(reg["name"]):
                reg["h"] = 1
    for layer, names in atlas["names"].items():
        atlas["hn"][layer] = [int(j) for j, name in names.items() if screen(name)]
    for r in rows:
        if categories(r["layer"]).get(r["id"]) in UPSETTING or screen(f"{r['label']} {r['does']}"):
            r["h"] = 1
    for e in (examples or {}).get("examples", []):
        if screen(" ".join([e["label"], e["does"], *e["replies"]])):
            e["h"] = 1
    return hidden


def page(template: str, current: str, data=None, src: str | None = None, private: bool = False, full: bool = False) -> str:
    """Fill a template: the theme and header (brand + nav) for the shared site, then its data.
    private=True is Studio's own copy: every feature, safety coloring kept, Studio's nav."""
    if private:
        head = f'<div class="bar-head">{brand("Glimmer Studio")}{nav(current, STUDIO_NAV, "Glimmer Studio")}</div>'
    else:
        template = template if full else SAFETY_OPTION.sub("", template)
        head = f'<div class="bar-head">{brand()}{nav(current)}<span class="toggles">{THEME_TOGGLE}{SAFE_TOGGLE}</span></div>{SAFE_SCRIPT}'
    header = (f'<header class="site-header"><div class="inner">{brand()}{nav(current)}<span class="toggles">{THEME_TOGGLE}{SAFE_TOGGLE}</span>'
              f'</div></header>{SAFE_SCRIPT}')
    html = (template.replace("<!--THEME-->", '<link rel="stylesheet" href="theme.css">' + THEME_EARLY)
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
    atlas, maps, dropped = public_data(full, runs)
    named = apply_names(atlas, maps, full)
    rows = steerable(runs, maps, full)
    examples = json.loads(EXAMPLES.read_text()) if EXAMPLES.exists() else None
    if full:
        print(f"safe browsing hides {mark_hidden(atlas, maps, rows, examples)} map features until it is turned off")
    if not full:
        for data in (atlas, *maps.values(), rows):
            hits = [s for s in strings(data) if FLAGGED.search(s)]
            assert not hits, hits[:5]

    if atlas.get("files"):
        (out / "atlas").mkdir(exist_ok=True)
        for f, m in zip(atlas["files"], atlas["maps"]):
            (out / f).write_text(json.dumps(m, separators=(",", ":")))
        (out / "atlas.json").write_text(json.dumps({**atlas, "maps": []}, separators=(",", ":")))
    else:
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
    if examples:
        if not full:
            examples["examples"] = [e for e in examples["examples"] if not FLAGGED.search(f"{e['label']} {e['does']}")]
        (out / "examples.html").write_text(page((SCRIPTS / "examples_template.html").read_text(), "examples", data=examples))
        print(f"examples: {len(examples['examples'])}")

    if not full:
        for path in out.glob("*.*"):
            text = path.read_text()
            assert "safety-relevant" not in text and '"f":"' not in text and '"safety":"' not in text, path
    note = ("Safe browsing is on by default. Some features fire on text that could be upsetting or misused: weapons, "
            "biological and chemical risk, cyberattacks, self-harm, hate, drugs and sexual content. Their points keep their "
            "place on every map, but their labels, examples and region names are hidden until you turn safe browsing off "
            "in the header. Nothing is removed from the data, and the SAEs themselves are not shared."
            if full else "Features about weapons, cyberattacks, biological and chemical risk and other dual use topics are left out of these pages.")
    index = overview((SITE / "index.html").read_text(), layer_stats(runs), atlas, runs)
    (out / "index.html").write_text(page(index.replace("<!--SAFETY_NOTE-->", note), "./"))
    for name in ("theme.css", "safe.js", "atlas-stack-light.jpg", "atlas-stack-dark.jpg", "brand/Geist-Variable.woff2"):
        (out / name).parent.mkdir(exist_ok=True)
        shutil.copy(SITE / name, out / name)
    (out / "_headers").write_text("/*\n  X-Robots-Tag: noindex\n")


if __name__ == "__main__":
    main()
