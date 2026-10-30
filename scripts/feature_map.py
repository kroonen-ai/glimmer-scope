"""A map of the SAE's features, like the feature UMAPs in Scaling Monosemanticity.

Each labeled feature is placed by its decoder direction: unit-normalize, then UMAP to 2D with
cosine distance, so features that write similar things into the residual stream sit
together. K-means on a 10-D UMAP of the same neighbors splits the map into regions, each named from the words
its members' labels share most. Writes results/feature-map.html (interactive) and
results/feature-map.json (results/ is gitignored).

    python scripts/feature_map.py runs/layer24-64k-k64 [regions] [kinds]

kinds defaults to "concept" (a map of subjects); "all" maps format, syntax and token features too.
"""
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

STOP = set("""a an the and or of in on for to with by from as at into about over via
its their his her your our is are be being been that this these those which who whom
words word text phrases phrase terms term token tokens use used using related references
mentions mention content descriptions description general various specific common
pattern patterns random clear generic fragments fragment contexts context variants forms concept concepts
related etc style styles type types""".split())


def decoder(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    info = header["W_dec"]
    rows, cols = info["shape"]
    return np.memmap(path, dtype="<f4", mode="r", offset=8 + n + info["data_offsets"][0], shape=(rows, cols))


def region_names(labels: list[str], assign: np.ndarray, k: int) -> list[str]:
    tokenize = lambda s: [w for w in re.findall(r"[a-z][a-z'-]{2,}", s.lower()) if w not in STOP]  # noqa: E731
    doc_freq = Counter()
    per_cluster = [Counter() for _ in range(k)]
    for label, c in zip(labels, assign):
        words = set(tokenize(label))
        doc_freq.update(words)
        per_cluster[c].update(words)
    total = len(labels)
    names = []
    for c in range(k):
        size = max(1, int((assign == c).sum()))
        scored = sorted(per_cluster[c].items(),
                        key=lambda kv: -(kv[1] / size) * np.log(total / (1 + doc_freq[kv[0]])))
        names.append(" · ".join(w for w, _ in scored[:3]) or f"region {c}")
    return names


def main():
    base = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    kinds = sys.argv[3] if len(sys.argv) > 3 else "concept"
    labels = [json.loads(line) for line in open(base / "labels.jsonl")]
    if kinds != "all":
        labels = [r for r in labels if r.get("kind") in set(kinds.split(","))]
    labels.sort(key=lambda r: r["latent"])
    scores = {}
    for name in ("steerability.jsonl", "steerability-personas.jsonl"):
        path = base / name
        if path.exists():
            for line in open(path):
                r = json.loads(line)
                if r["latent"] not in scores or r["score"] > scores[r["latent"]]:
                    scores[r["latent"]] = r["score"]
    safety = {}
    layer = json.loads((base / "final" / "cfg.json").read_text())["layer"]
    flagged = Path(__file__).resolve().parent.parent / "results" / f"safety-features-L{layer}.json"
    if flagged.exists():
        for category, items in json.loads(flagged.read_text()).items():
            for item in items:
                safety.setdefault(item["latent"], category)

    W = decoder(base / "final" / "sae.safetensors")
    latents = np.array([r["latent"] for r in labels])
    X = np.asarray(W[latents], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    print(f"{len(X)} labeled features, decoder dim {X.shape[1]}", flush=True)

    from sklearn.cluster import MiniBatchKMeans
    import umap

    # UMAP straight on the unit decoder directions: 64 PCA dims keep only ~12% of their variance,
    # and the neighborhoods live in the rest. Regions come from a 10-D UMAP of the same graph, so
    # they line up with the 2-D layout.
    xy = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=0).fit_transform(X)
    z10 = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=10, metric="cosine", random_state=0).fit_transform(X)
    assign = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=4, batch_size=4096).fit_predict(z10)
    names = region_names([r["label"] for r in labels], assign, k)

    points = []
    for r, (x, y), c in zip(labels, xy, assign):
        points.append({"id": r["latent"], "x": round(float(x), 3), "y": round(float(y), 3), "label": r["label"],
                       "kind": r.get("kind"), "region": int(c), "score": scores.get(r["latent"]),
                       "safety": safety.get(r["latent"])})
    regions = []
    for c in range(k):
        members = xy[assign == c]
        regions.append({"id": c, "name": names[c], "size": int(len(members)),
                        "x": round(float(np.median(members[:, 0])), 3), "y": round(float(np.median(members[:, 1])), 3)})
    out = Path(__file__).resolve().parent.parent / "results"
    data = {"points": points, "regions": regions}
    (out / "feature-map.json").write_text(json.dumps(data))
    template = (Path(__file__).resolve().parent / "feature_map_template.html").read_text()
    (out / "feature-map.html").write_text(template.replace("/*DATA*/null", json.dumps(data)))
    print(f"wrote {out / 'feature-map.html'} with {len(points)} points in {k} regions", flush=True)


if __name__ == "__main__":
    main()
