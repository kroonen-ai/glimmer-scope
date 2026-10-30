"""The Glimmer Atlas across layers: one map per SAE, aligned so they stack.

Each layer's concept features are placed by their unit decoder directions as in feature_map.py,
but with AlignedUMAP, which pulls features that match across neighboring layers (mutual best
partners from modal_app.py::crosslayer) toward the same spot. A subject then sits in roughly the
same place on every layer, and stacking the maps shows what persists, splits or appears with
depth. Each feature carries its best partners one layer down (where it comes from) and one layer
up (what it turns into). Writes results/layer-atlas.html and .json (results/ is gitignored).

    python scripts/layer_atlas.py runs layer12-64k-k64,layer24-64k-k64,layer36-64k-k64
"""
import json
import sys
from pathlib import Path

import numpy as np

from feature_map import decoder, region_names

MIN_LINK = 0.15  # partner cosine worth showing
MIN_ALIGN = 0.3  # mutual best partners at least this alike anchor the layouts together
PARTNERS = 4


def load_layer(base: Path, run: str, results: Path) -> dict:
    layer = json.loads((base / run / "final" / "cfg.json").read_text())["layer"]
    labels = [json.loads(line) for line in open(base / run / "labels.jsonl")]
    names = {r["latent"]: r["label"] for r in labels}
    concept = sorted((r for r in labels if r.get("kind") == "concept"), key=lambda r: r["latent"])
    scores = {}
    for name in ("steerability.jsonl", "steerability-personas.jsonl"):
        path = base / run / name
        if path.exists():
            for line in open(path):
                r = json.loads(line)
                if r["latent"] not in scores or r["score"] > scores[r["latent"]]:
                    scores[r["latent"]] = r["score"]
    safety = {}
    flagged = results / f"safety-features-L{layer}.json"
    if flagged.exists():
        for category, items in json.loads(flagged.read_text()).items():
            for item in items:
                safety.setdefault(item["latent"], category)
    W = decoder(base / run / "final" / "sae.safetensors")
    latents = np.array([r["latent"] for r in concept])
    X = np.asarray(W[latents], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return {"layer": layer, "concept": concept, "names": names, "scores": scores, "safety": safety, "X": X,
            "pos": {int(lat): i for i, lat in enumerate(latents)}}


def main():
    base = Path(sys.argv[1]).expanduser()
    runs = sys.argv[2].split(",")
    results = Path(__file__).resolve().parent.parent / "results"
    layers = sorted((load_layer(base, run, results) for run in runs), key=lambda d: d["layer"])
    tag = "-".join(f"L{d['layer']}" for d in layers)
    cross = np.load(base / "crosslayer" / f"{tag}.npz")
    for d in layers:
        print(f"layer {d['layer']}: {len(d['X'])} concept features", flush=True)

    # Mutual best partners between neighboring layers, both on the map, anchor the alignment.
    relations = []
    for lo, hi in zip(layers, layers[1:]):
        key = f"L{lo['layer']}_L{hi['layer']}"
        up_idx, up_cos = cross[f"up_idx_{key}"], cross[f"up_cos_{key}"]
        down_idx = cross[f"down_idx_L{hi['layer']}_L{lo['layer']}"]
        rel = {}
        for latent, i in lo["pos"].items():
            j = int(up_idx[latent, 0])
            if up_cos[latent, 0] >= MIN_ALIGN and int(down_idx[j, 0]) == latent and j in hi["pos"]:
                rel[i] = hi["pos"][j]
        relations.append(rel)
        best = up_cos[:, 0]
        print(f"{key}: {len(rel)} anchors; best-partner cosine median {np.median(best):.2f}", flush=True)

    import umap
    from sklearn.cluster import MiniBatchKMeans

    mapper = umap.AlignedUMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=0)
    mapper.fit([d["X"] for d in layers], relations=relations)

    maps, extra = [], {}
    for li, (d, xy) in enumerate(zip(layers, mapper.embeddings_)):
        k = max(20, round(len(xy) / 285))
        assign = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=4, batch_size=4096).fit_predict(xy)
        names = region_names([r["label"] for r in d["concept"]], assign, k)
        layer = d["layer"]
        extra_names = {}

        def links(idx_key, cos_key, other):
            out = {}
            if idx_key not in cross:
                return out
            idx, cos = cross[idx_key], cross[cos_key]
            for latent in d["pos"]:
                row = [[int(j), round(float(c), 2)] for j, c in zip(idx[latent, :PARTNERS], cos[latent, :PARTNERS]) if c >= MIN_LINK]
                if row:
                    out[latent] = row
                    for j, _ in row:
                        if j not in other["pos"]:
                            extra.setdefault(str(other["layer"]), {})[str(j)] = other["names"].get(j, "unlabeled")
            return out

        up = links(f"up_idx_L{layer}_L{layers[li + 1]['layer']}", f"up_cos_L{layer}_L{layers[li + 1]['layer']}", layers[li + 1]) if li + 1 < len(layers) else {}
        down = links(f"down_idx_L{layer}_L{layers[li - 1]['layer']}", f"down_cos_L{layer}_L{layers[li - 1]['layer']}", layers[li - 1]) if li > 0 else {}
        points = []
        for r, (x, y), c in zip(d["concept"], xy, assign):
            lat = r["latent"]
            p = {"i": lat, "x": round(float(x), 2), "y": round(float(y), 2), "l": r["label"], "r": int(c)}
            if lat in d["scores"]:
                p["s"] = d["scores"][lat]
            if lat in d["safety"]:
                p["f"] = d["safety"][lat]
            if lat in up:
                p["u"] = up[lat]
            if lat in down:
                p["d"] = down[lat]
            points.append(p)
        regions = []
        for c in range(k):
            members = xy[assign == c]
            regions.append({"id": c, "name": names[c], "size": int(len(members)),
                            "x": round(float(np.median(members[:, 0])), 2), "y": round(float(np.median(members[:, 1])), 2)})
        maps.append({"layer": layer, "points": points, "regions": regions,
                     "tested": len(d["scores"]), "labeled": len(d["names"])})

    data = {"maps": maps, "names": extra, "tokens": int(cross["tokens"])}
    (results / "layer-atlas.json").write_text(json.dumps(data, separators=(",", ":")))
    template = (Path(__file__).resolve().parent / "layer_atlas_template.html").read_text()
    (results / "layer-atlas.html").write_text(template.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {results / 'layer-atlas.html'}: " + ", ".join(f"layer {m['layer']} {len(m['points'])} points" for m in maps))


if __name__ == "__main__":
    main()
